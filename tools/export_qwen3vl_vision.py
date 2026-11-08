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
"""Export Qwen3-VL's vision tower + merger to ONNX, then check it against PyTorch.

The vision half of a Qwen3-VL prompt is one graph: patches -> tower (24 blocks) -> merger/projector, emitting one
2048-wide row per merged patch. `image_grid_thw` is what tells it the image's patch grid, so it is an input, which is
also what lets one exported graph take any image size the preprocessor produces (the grid is dynamic; the patch
count is not fixed at export time).

    export_qwen3vl_vision.py export <hf dir> <out dir> [--opset 20]
    export_qwen3vl_vision.py check  <hf dir> <out dir>            # gate (1): cosine >= 0.999 per patch, 5 images

`check` compares the ONNX graph's rows with the FP32 PyTorch tower's, patch by patch: the claim is that the export is
arithmetic-identical up to fp32 summation order, so a per-patch cosine near 1.0 and a max |delta| in the 1e-5 range are
the right rows; a mean-only comparison would hide one bad patch in a pool of good ones.

Deepstack: Qwen3-VL also emits feature sets from blocks [5, 11, 17] that the decoder adds at three of its own layers.
They are exported as separate outputs (`deepstack_0..2`) so the oracle has them, and `check` reports them the same
way; a consumer that only takes the pooled rows can ignore them, at the cost of the accuracy difference noted in
docs/npu.md.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

VISION_KEYS = ("visual", "vision")


def load_tower(hf_dir: str):
    """The vision tower only, built from the checkpoint's own vision weights.

    Loading `AutoModel.from_pretrained` here would materialise the 2B text half in fp32 as well (~9 GB); the tower is
    675 MB. `safe_open` reads one tensor at a time from the mmap, so the peak stays at the tower plus the largest
    tensor.
    """
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    cfg = AutoConfig.from_pretrained(hf_dir)
    vc = cfg.vision_config
    # Eager attention, not SDPA: torch's TorchScript exporter refuses SDPA with enable_gqa (the vision tower's
    # 16-head/... split), and an explicit matmul+softmax is also the shape a QDQ/NPU EP handles best. Setting it on
    # the config is what the model actually reads; the constructor kwarg alone was not enough.
    try:
        vc._attn_implementation = "eager"
    except Exception:
        pass
    tower = Qwen3VLVisionModel(vc)
    tower.config._attn_implementation = "eager"
    for m in tower.modules():
        if hasattr(m, "config"):
            try:
                m.config._attn_implementation = "eager"
            except Exception:
                pass
    tower.eval()

    want = {k: v for k, v in tower.state_dict().items()}
    state = {}
    with safe_open(os.path.join(hf_dir, "model.safetensors"), framework="pt", device="cpu") as f:
        for name in f.keys():
            if not name.startswith("model.visual."):
                continue
            key = name[len("model.visual."):]
            if key in want:
                state[key] = f.get_tensor(name).to(torch.float32)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    missing = [m for m in missing if m not in ("rotary_pos_emb.inv_freq",)]
    if missing:
        raise SystemExit(f"vision tower is missing {len(missing)} tensor(s), first: {missing[:3]}")
    print(f"loaded {len(state)} vision tensors (unexpected {len(unexpected)})")
    return tower, vc


class VisionGraph(torch.nn.Module):
    """pixel_values -> merged rows + the deepstack feature sets, which is what the decoder consumes.

    An `nn.Module` on purpose: torch 2.15's exporter (torch.export) refuses a plain class, and naming the submodules
    `tower` keeps the exported graph's node names readable.
    """

    def __init__(self, tower):
        super().__init__()
        self.tower = tower

    def forward(self, pixel_values, grid_thw):
        pooled, deep = self.tower(pixel_values, grid_thw=grid_thw)
        return (pooled, *deep) if deep else (pooled,)


def export(hf_dir: str, out_dir: str, opset: int):
    import torch

    tower, vc = load_tower(hf_dir)
    os.makedirs(out_dir, exist_ok=True)
    merge, patch = int(vc.spatial_merge_size), int(vc.patch_size)
    depth = int(vc.depth)
    # `image_grid_thw` is counted in PATCHES, not pixels: a 448x448 image at patch 16 is grid (1, 28, 28) =
    # 784 patches, which the merger folds to 196 rows. That bucket is the one the test image uses.
    side = 28
    n_patches = side * side
    pv = torch.zeros(n_patches, 3 * int(vc.temporal_patch_size) * patch * patch)
    grid = torch.tensor([[1, side, side]], dtype=torch.long)
    graph = VisionGraph(tower)
    names = ["pooled"] + [f"deepstack_{i}" for i in range(len(vc.deepstack_visual_indexes))]
    path = os.path.join(out_dir, "vision_fp32.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            graph, (pv, grid), path,
            input_names=["pixel_values", "image_grid_thw"], output_names=names,
            # a fixed bucket: the patch count is baked in (see dynamo=False below)

            opset_version=opset, do_constant_folding=True,
            # External data: the tower is 1.6 GB of fp32 weights, and a single inline protobuf of that size is where
            # ONNX tooling starts failing ("Wire format was corrupt" from the quantiser's pre-processing). Written as
            # `<graph>.onnx` + `<graph>.data`, which is also the shape AMD's own vision packs ship.
            external_data=True,
            # The legacy (TorchScript) exporter, not torch.export: the tower's positional interpolation loops over
            # the *values* of image_grid_thw (`torch.linspace(0, num_grid_per_side-1, h)` per image), which
            # torch.export cannot symbolic-shape.  This makes the exported graph a FIXED BUCKET: the grid is baked
            # in, and a different image size needs either another bucket or the tensor-rewritten interpolation
            # noted in docs/npu.md.  The owner's order allows fixed buckets, so this is the shipped shape.
            dynamo=False,
        )
    # Split the weights out by hand when the exporter kept them inline: the legacy exporter ignores
    # external_data=, and a 1.6 GB inline protobuf is where the quantiser's pre-processing fails to parse.
    try:
        import onnx
        big = os.path.getsize(path) > 512 * 1024 * 1024
        if big and not os.path.exists(path + ".data"):
            m = onnx.load(path)
            onnx.save_model(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                            location=os.path.basename(path) + ".data", size_threshold=1024)
            del m
            print(f"split weights to {os.path.basename(path)}.data "
                  f"({os.path.getsize(path) / 1e6:.1f} MB graph + {os.path.getsize(path + '.data') / 1e6:.1f} MB)")
    except ImportError:
        pass
    meta = {"source": os.path.abspath(hf_dir), "opset": opset, "hidden": int(vc.hidden_size),
            "out_hidden": int(vc.out_hidden_size), "patch": patch, "temporal_patch": int(vc.temporal_patch_size),
            "merge": merge, "depth": depth, "heads": int(vc.num_heads),
            "deepstack_indexes": list(vc.deepstack_visual_indexes),
            "num_position_embeddings": int(vc.num_position_embeddings), "bucket_patches": side}
    json.dump(meta, open(os.path.join(out_dir, "vision_fp32.json"), "w"), indent=1)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB), outputs {names}")
    return path


def make_image(rng, gh, gw, patch, temporal):
    """A deterministic image with structure (gradients, a checkerboard, noise), packed as the tower expects.
    `gh`/`gw` are in PATCHES (Qwen3-VL counts image_grid_thw in patches)."""
    H, W = gh * patch, gw * patch
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((3, H, W), dtype=np.float32)
    img[0] = (yy / max(H - 1, 1)) * 0.8 + 0.1
    img[1] = (xx / max(W - 1, 1)) * 0.8 + 0.1
    img[2] = ((yy // 32 + xx // 32) % 2) * 0.7 + 0.1
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    # [gh, gw, 3, patch, patch] -> [gh*gw, 3*temporal*patch*patch] (temporal=1 here: one frame per patch)
    blocks = img.reshape(3, gh, patch, gw, patch).transpose(1, 3, 0, 2, 4)
    if temporal != 1:
        blocks = np.repeat(blocks[:, :, None], temporal, axis=2)
    return blocks.reshape(gh * gw, 3 * temporal * patch * patch).astype(np.float32)


def check(hf_dir: str, out_dir: str, images: int, onnx_name: str = "vision_fp32.onnx"):
    import onnxruntime as ort
    import torch

    tower, vc = load_tower(hf_dir)
    graph = VisionGraph(tower)
    sess = ort.InferenceSession(os.path.join(out_dir, onnx_name), providers=["CPUExecutionProvider"])
    merge, patch, temporal = int(vc.spatial_merge_size), int(vc.patch_size), int(vc.temporal_patch_size)
    names = ["pooled"] + [f"deepstack_{i}" for i in range(len(vc.deepstack_visual_indexes))]
    meta = json.load(open(os.path.join(out_dir, "vision_fp32.json")))
    bucket = int(meta.get("bucket_patches", 28))
    rng = np.random.default_rng(1311)
    rows, worst = [], 1.0
    for i in range(images):
        # the exported graph is a FIXED bucket (dynamo=False, the tower's positional interpolation is
        # value-dependent), so every gate-1 image is that bucket; only the *content* differs per image.
        gh = gw = bucket
        pv = make_image(rng, gh, gw, patch, temporal)
        grid = torch.tensor([[1, gh, gw]], dtype=torch.long)   # patches, not pixels
        with torch.inference_mode():
            ref = graph(torch.from_numpy(pv), grid)
        got = sess.run(None, {"pixel_values": pv, "image_grid_thw": grid.numpy()})
        for name, r, g in zip(names, ref, got):
            r = r.numpy().reshape(-1)
            g = g.reshape(-1)
            if r.size != g.size:
                rows.append((i, name, float("nan"), float("nan"), r.size, g.size))
                worst = 0.0
                continue
            cos = float(np.dot(r, g) / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-12))
            dmax = float(np.abs(r - g).max())
            worst = min(worst, cos)
            rows.append((i, name, cos, dmax, r.size, g.size))
    for i, name, cos, dmax, n, gn in rows:
        print(f"image {i} {name:11} cosine {cos:.7f} max|d| {dmax:.3e} floats {n}/{gn}")
    ok = worst >= 0.999 and all(r[4] == r[5] for r in rows)
    print(f"gate 1 (cosine >= 0.999 per patch set, {images} images): {'PASS' if ok else 'FAIL'} (worst {worst:.7f})")
    return 0 if ok else 1


def hf_dir_of(fp32_dir: str) -> str:
    meta = json.load(open(os.path.join(fp32_dir, "vision_fp32.json")))
    return meta.get("source", "")


def bench(out_dir: str, images: int, providers: str):
    """Seconds per image through the graph, on the providers named (e.g. "CPUExecutionProvider").

    This is gate 3's other half: the CPU projector in llama.cpp measures 16.97 s/image on this box, and a row here is
    what our graph costs on the same machine. Not an accuracy row -- `check` is.
    """
    import onnxruntime as ort
    import time

    meta = json.load(open(os.path.join(out_dir, "vision_fp32.json")))
    bucket, patch, temporal = int(meta["bucket_patches"]), int(meta["patch"]), int(meta["temporal_patch"])
    provs = [p.strip() for p in providers.split(",") if p.strip()]
    sess = ort.InferenceSession(os.path.join(out_dir, "vision_fp32.onnx"), providers=provs)
    rng = np.random.default_rng(11)
    pv = make_image(rng, bucket, bucket, patch, temporal)
    grid = np.array([[1, bucket, bucket]], dtype=np.int64)
    t0 = time.time()
    sess.run(None, {"pixel_values": pv, "image_grid_thw": grid})          # warm-up (graph + arena)
    warm = time.time() - t0
    times = []
    for _ in range(images):
        pv = make_image(rng, bucket, bucket, patch, temporal)
        t0 = time.time()
        sess.run(None, {"pixel_values": pv, "image_grid_thw": grid})
        times.append(time.time() - t0)
    best, med = min(times), sorted(times)[len(times) // 2]
    print(f"providers={sess.get_providers()} images={images} warm-up {warm:.2f}s | "
          f"per image best {best:.2f}s median {med:.2f}s | {1.0 / best:.3f} images/s")
    print(f"cpu bar to beat (llama.cpp CPU projector, measured): 16.97 s/image")
    return 0


def quantize(fp32_dir: str, out_dir: str, calib: int):
    """QDQ int8, A8W8 per-channel -- the shape our ORT EP runs (Quark-style A8W8).

    Activations are quantised per tensor (QDQ), weights per output channel, both int8, and `image_grid_thw` is left
    alone (it is an integer input, not an activation). The calibration set is deterministic images at the exported
    bucket, so the row is reproducible; a real deployment would calibrate on its own data.
    """
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process  # kept for the record above

    meta = json.load(open(os.path.join(fp32_dir, "vision_fp32.json")))
    bucket, patch = int(meta["bucket_patches"]), int(meta["patch"])
    temporal, merge = int(meta["temporal_patch"]), int(meta["merge"])
    src = os.path.join(fp32_dir, "vision_fp32.onnx")

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rng = np.random.default_rng(7)
            self.items = []
            for _ in range(calib):
                pv = make_image(self.rng, bucket, bucket, patch, temporal)
                grid = np.array([[1, bucket, bucket]], dtype=np.int64)
                self.items.append({"pixel_values": pv, "image_grid_thw": grid})
            self.i = 0

        def get_next(self):
            if self.i >= len(self.items):
                return None
            it = self.items[self.i]
            self.i += 1
            return it

        def rewind(self):
            self.i = 0

    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "vision_qdq.onnx")
    # Pre-processing first: symbolic shape inference + the standard fusions. Without it ORT quantises a graph whose
    # tensors have no inferred ranges at several nodes and the calibration never reaches them — measured as a
    # systematic, not a precision, error (cosine 0.22 with it missing).
    # No quant_pre_process() here on purpose: with the weights in an external `.data` file it copies the model to a
    # temp dir and leaves the data behind ("Data of TensorProto … should be stored in /tmp/pre.quant.*/
    # vision_fp32.onnx.data, but it is not regular file"), and with the weights inline it fails earlier on
    # symbolic shape inference. The graph is a fixed bucket with concrete shapes, so the quantiser's own
    # pre-processing is not needed; this is recorded rather than worked around with a fragile temp-file dance.
    _ = quant_pre_process
    quantize_static(src, dst, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    calibrate_method=CalibrationMethod.Percentile,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    # Only the big projections. Quantising Conv (the patch embed), Add (residuals) and the
                    # norm/softmax paths as well was measured at cosine 0.22 against fp32 -- the graph is a ViT
                    # with per-token dynamic ranges and no QDQ pre-processing, so the sensitive ops stay float.
                    op_types_to_quantize=["MatMul"],
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True, "CalibPercentile": 99.9})
    out_meta = dict(meta)
    out_meta.update({"quant": "qdq-a8w8-perchannel-int8", "calib_images": calib})
    json.dump(out_meta, open(os.path.join(out_dir, "vision_qdq.json"), "w"), indent=1)
    print(f"wrote {dst} ({os.path.getsize(dst) / 1e6:.1f} MB) from {os.path.getsize(src) / 1e6:.1f} MB")
    # The tool gates itself: a QDQ graph that does not reproduce the fp32 rows is not a result, and shipping one
    # silently is the failure mode this whole exercise exists to avoid (two naive recipes measured 0.22 and 0.51
    # cosine before this check existed).
    print("checking the quantised graph against PyTorch before calling it done:")
    rc = check(hf_dir_of(fp32_dir), out_dir, 5, "vision_qdq.onnx")
    if rc != 0:
        print("QUANTISED GRAPH FAILS the per-patch cosine bar -- do not use it (see the rows above); the fp32 export "
              "is the graph gate 1 claims")
    return 0 if rc == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("hf_dir")
    p.add_argument("out_dir")
    p.add_argument("--opset", type=int, default=20)
    q = sub.add_parser("quantize")
    q.add_argument("fp32_dir")
    q.add_argument("out_dir")
    q.add_argument("--calib", type=int, default=32)
    b = sub.add_parser("bench")
    b.add_argument("out_dir")
    b.add_argument("--images", type=int, default=3)
    b.add_argument("--providers", default="CPUExecutionProvider")
    c = sub.add_parser("check")
    c.add_argument("hf_dir")
    c.add_argument("out_dir")
    c.add_argument("--images", type=int, default=5)
    c.add_argument("--onnx", default="vision_fp32.onnx", help="graph to check (e.g. vision_qdq.onnx)")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.hf_dir, a.out_dir, a.opset)
        return 0
    if a.cmd == "quantize":
        return quantize(a.fp32_dir, a.out_dir, a.calib)
    if a.cmd == "bench":
        return bench(a.out_dir, a.images, a.providers)
    return check(a.hf_dir, a.out_dir, a.images, a.onnx)


if __name__ == "__main__":
    sys.exit(main())
