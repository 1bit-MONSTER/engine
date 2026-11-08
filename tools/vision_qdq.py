#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""Streaming QDQ A8W8 for the vision graph: one weight tensor in memory at a time.

Used by `export_qwen3vl_vision.py quantize --stream`.

ORT's quantize_static needs ~64 GiB peak for this 1.6 GB graph (measured: the systemd scope recorded 64.1 G peak before
the OOM killer), which a shared box cannot spare. This does the same job by editing the graph directly:

  * weights: per-output-channel symmetric int8, emitted as QuantizeLinear -> DequantizeLinear pairs;
  * activations: per-tensor symmetric int8 before each MatMul, with the scale calibrated from a few images (the fp32
    graph is run once with every MatMul input exposed as an output, so no intermediate probing library is needed);
  * everything else (Conv, norms, softmax, split) stays float.

    qdq_stream.py <fp32 onnx> <out onnx> [--calib N]
"""
import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, "/tmp/qvl")


def matmul_weights(model):
    """(node, initializer, per_output_axis) for every layer that multiplies by a WEIGHT initializer.

    The tower's projections are `Gemm(A, W, bias)` with `transB=1` (torch exports Linear that way), so W is [N, K] and
    its per-output-channel scale runs along axis 0.  The `MatMul` nodes that remain are attention's A@K^T and A@V,
    where *both* operands are activations -- quantising those needs both inputs quantised and is deliberately left
    float here, since the projections carry the FLOPs.
    """
    inits = {i.name: i for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type not in ("Gemm", "MatMul") or len(n.input) < 2 or n.input[1] not in inits:
            continue
        w = numpy_helper.to_array(inits[n.input[1]])
        if w.ndim != 2:
            continue
        trans_b = 0
        if n.op_type == "Gemm":
            for at in n.attribute:
                if at.name == "transB":
                    trans_b = int(at.i)
        # reduce over K: axis 1 for W [N, K] (Gemm transB=1, the torch Linear export), axis 0 for W [K, N]
        out.append((n, inits[n.input[1]], 1 if (n.op_type == "Gemm" and trans_b == 1) else 0))
    return out, inits


def expose_outputs(model, names):
    have = {o.name for o in model.graph.output}
    for nm in names:
        if nm in have:
            continue
        model.graph.output.append(helper.make_tensor_value_info(nm, TensorProto.FLOAT, None))


def run_calibration(path, names, images, bucket, patch, temporal, real_image="", calib_pv="", clip_pct=99.9):
    import onnxruntime as ort
    # the same deterministic image generator the tool's check rows use, copied here so the streaming quantiser does not
    # depend on the harness script
    from hybrid_input import image_rows  # noqa: F401  (the real image path is exercised by the check rows)
    def make_image(rng, gh, gw, patch, temporal):
        H, W = gh * patch, gw * patch
        yy, xx = np.mgrid[0:H, 0:W]
        img = np.zeros((3, H, W), dtype=np.float32)
        img[0] = (yy / max(H - 1, 1)) * 0.8 + 0.1
        img[1] = (xx / max(W - 1, 1)) * 0.8 + 0.1
        img[2] = ((yy // 32 + xx // 32) % 2) * 0.7 + 0.1
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)
        blocks = img.reshape(3, gh, patch, gw, patch).transpose(1, 3, 0, 2, 4)
        if temporal != 1:
            blocks = np.repeat(blocks[:, :, None], temporal, axis=2)
        return blocks.reshape(gh * gw, 3 * temporal * patch * patch).astype(np.float32)

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = {nm: 0.0 for nm in names}   # the max over images of each image's p99.9 |x| (see --clip-percentile)
    feeds = []
    if calib_pv:                                # the real picture first: synthetic gradients do not bound a photo
        feeds.append(np.load(calib_pv).astype(np.float32))
    elif real_image:
        feeds.append(image_rows(real_image))
    rng = np.random.default_rng(3)
    for _ in range(images):
        feeds.append(make_image(rng, bucket, bucket, patch, temporal))
    for pv in feeds:
        grid = np.array([[1, bucket, bucket]], dtype=np.int64)
        outs = sess.run(names, {"pixel_values": pv, "image_grid_thw": grid})
        for nm, arr in zip(names, outs):
            # a single outlier must not set the whole tensor's scale: one activation here has absmax 1083 while the
            # mass of that tensor sits near zero, and max-based scales zero it out (measured: 103/104 tensors diverge,
            # cosine -0.03).  A high percentile keeps the dynamic range the mass actually uses.
            a_ = np.abs(arr.reshape(-1)).astype(np.float32)
            got[nm] = max(got[nm], float(np.percentile(a_, clip_pct)))
    return got


def quantise(src, dst, calib=6, clip_percentile=99.9, calib_pv="", bucket=28, patch=16, temporal=2):
    """Quantise `src` into `dst` (+ `dst`.data). Returns the number of quantised weights."""
    a = argparse.Namespace(src=src, dst=dst, calib=calib, bucket=bucket, patch=patch, temporal=temporal,
                           no_act_qdq=False, real_image="", calib_pv=calib_pv, clip_percentile=clip_percentile)
    return _run(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--calib", type=int, default=6)
    ap.add_argument("--no-act-qdq", action="store_true", help="weights only (W8A32): isolates which side loses the answer")
    ap.add_argument("--bucket", type=int, default=28)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--temporal", type=int, default=2)
    ap.add_argument("--real-image", default="", help="calibrate on this image too (needs PIL)")
    ap.add_argument("--calib-pv", default="", help="calibrate on these pixels (a .npy), so no PIL is needed here")
    ap.add_argument("--clip-percentile", type=float, default=99.9,
                    help="per-tensor |x| percentile used as the activation range (100 = plain max, which loses a tensor "
                         "to one outlier: measured 103/104 tensors diverge)")
    return _run(ap.parse_args())


def _run(a):
    model = onnx.load(a.src)                      # ~2 GB in memory; the weights are not copied per tensor
    pairs, inits = matmul_weights(model)
    print(f"{len(pairs)} MatMul weights to quantise")

    # --- calibrate the activations: expose each MatMul's A input, run a few images, take the absmax
    names = sorted({n.input[0] for n, _, _ in pairs})
    calib_model = onnx.load(a.src)
    expose_outputs(calib_model, names)
    cpath = a.dst + ".calib.onnx"
    onnx.save(calib_model, cpath)
    del calib_model
    act_absmax = run_calibration(cpath, names, a.calib, a.bucket, a.patch, a.temporal, a.real_image, a.calib_pv,
                                 a.clip_percentile)
    os.unlink(cpath)
    vals = sorted(act_absmax.values())
    tiny = [nm for nm, v in act_absmax.items() if v < 1e-3]
    print(f"calibrated {len(act_absmax)} activation tensors on {a.calib} images: "
          f"absmax min {vals[0]:.4g} median {vals[len(vals)//2]:.4g} max {vals[-1]:.4g}; "
          f"{len(tiny)} below 1e-3" + (f" (e.g. {', '.join(tiny[:4])})" if tiny else ""))

    # --- rewrite: per-channel int8 weights, per-tensor int8 activations, QDQ pairs
    new_inits, replaced = [], set()
    act_scale = {} if a.no_act_qdq else {nm: max(v / 127.0, 1e-8) for nm, v in act_absmax.items()}
    # QuantizeLinear's output type comes from its zero point, and its default is UINT8 -- which silently clamps every
    # negative activation to 0.  An int8 zero point makes the codes int8 and is what the DequantizeLinear below expects
    # (measured: without it the first Gemm's activations lose all negatives, 103/104 tensors diverge, cosine -0.03).
    for nm, sc in act_scale.items():
        new_inits.append(numpy_helper.from_array(np.array([sc], dtype=np.float32), name=nm + "_qdq_scale"))
        new_inits.append(numpy_helper.from_array(np.zeros([1], dtype=np.int8), name=nm + "_qdq_zero"))

    # quantise each weight into a new initializer pair, remembering what each Gemm's inputs become
    gemm_rewrite = {}
    for n, init, axis in pairs:
        w = numpy_helper.to_array(init)          # one tensor at a time; the 64 GiB ORT needs for this is why
        if w.ndim != 2 or 0 in w.shape:
            continue
        red = (1,) if axis == 1 else (0,)
        cols = w.shape[0] if axis == 1 else w.shape[1]
        amax = np.abs(w).max(axis=red, keepdims=True)
        scale = np.maximum(amax / 127.0, 1e-8).astype(np.float32)
        q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
        base = init.name + "_qdq"
        new_inits.append(numpy_helper.from_array(q, name=base + "_q"))
        new_inits.append(numpy_helper.from_array(scale.reshape(cols), name=base + "_s"))
        # the codes keep the weight's layout, so the per-channel axis is the OTHER one (scale runs along N)
        gemm_rewrite[id(n)] = (n, base, 1 - red[0])
        replaced.add(init.name)
        del w, q, amax, scale

    # rebuild the node list in topological order, putting each QDQ pair immediately before its consumer
    act_emitted = set()
    out_nodes = []
    for n in model.graph.node:
        rw = gemm_rewrite.get(id(n))
        if rw:
            _, base, dq_axis = rw
            a_name = n.input[0]
            if a_name in act_scale and a_name not in act_emitted:
                act_emitted.add(a_name)
                out_nodes.append(helper.make_node("QuantizeLinear",
                                                  [a_name, a_name + "_qdq_scale", a_name + "_qdq_zero"],
                                                  [a_name + "_qdq_q"], name=a_name + "_qdq_ql"))
                out_nodes.append(helper.make_node("DequantizeLinear", [a_name + "_qdq_q", a_name + "_qdq_scale"],
                                                  [a_name + "_qdq_dq"], name=a_name + "_qdq_dql"))
                for n2, _, _ in pairs:
                    if n2.input[0] == a_name:
                        n2.input[0] = a_name + "_qdq_dq"
            out_nodes.append(helper.make_node("DequantizeLinear", [base + "_q", base + "_s"], [base + "_dq"],
                                              name=base, axis=dq_axis))
            for i, inp in enumerate(n.input):
                if inp in replaced:
                    n.input[i] = base + "_dq"
        out_nodes.append(n)

    keep = [i for i in model.graph.initializer if i.name not in replaced]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(keep + new_inits)
    model.graph.ClearField("node")
    model.graph.node.extend(out_nodes)
    onnx.checker.check_model(model)
    # a previous run's weight blob is appended to, not replaced, by save_model with external data
    for stale in (a.dst, a.dst + ".data"):
        if os.path.exists(stale):
            os.unlink(stale)
    onnx.save_model(model, a.dst, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(a.dst) + ".data", size_threshold=1024)
    print(f"wrote {a.dst} ({os.path.getsize(a.dst) / 1e6:.1f} MB) + .data "
          f"({os.path.getsize(a.dst + '.data') / 1e6:.1f} MB); {len(replaced)} weights, {len(act_emitted)} activations")
    return len(replaced)


if __name__ == "__main__":
    sys.exit(main())
