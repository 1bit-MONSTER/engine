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
"""moe_expert_chunks: one routed expert's weights, in the form the NPU chunk-list kernels eat.

A MoE decode dispatch needs, per selected expert, one weight block per projection. The container
this repo's NPU route reads (docs/npu.md) stores an expert as contiguous Q4NX tiles
(`__metadata__.moe` has its byte offset), but those tiles carry an asymmetric q4_1 zero point,
while the pb16/pb16i families pack `w = code * d` — one multiplier per group and column, no bias.
So an expert has to be re-expressed as SYMMETRIC int4 with a per-group multiplier, and that
conversion is lossy on top of the container's own requantisation. This tool does it, writes the
arena's input file per expert, and measures what it costs — the number that decides whether MoE
decode on this route is worth the loss.

  moe_expert_chunks.py chunks <model dir> <out dir> [--layers N] [--experts N] [--pad-n N]
        per expert: `L<layer>_<gate|up|down>_e<expert>.tf16` = int4 codes as int8 [K][N] then
        fp16 d bits [K//128][N] (layout_w4.py's read_tf16 input), plus manifest.json with the
        geometry the kernel generator needs (K, N, n-tiles at NW=32 and NW=16). The manifest
        keeps the three sizes apart by format - `read_tf16_bytes` for the file this writes,
        `pb16_bytes` for what the device streams once the arena packs it, `src_bytes` for the
        container's own tiles - because they differ by 1.8x and a byte budget that mixes them is
        wrong by that factor.

  moe_expert_chunks.py verify <model dir> <out dir> [--layer N] [--experts N]
        read every written chunk back and compare with the container's own weights for that
        expert: relative RMS, cosine, max |dw|, per projection.

Both read only the container; nothing here touches a device.
"""
import argparse
import json
import os
import sys

import numpy as np

from moe_q4nx_ref import Q4nx
from q4nx_pack import TILE, untile

PROJ = {"gate": "mlp.experts.gate_proj.weight", "up": "mlp.experts.up_proj.weight",
        "down": "mlp.experts.down_proj.weight"}
GROUP = 128  # the arena's multiplier group: one d per 128 K-rows per column


def container_expert(q, name, e):
    """One expert's weights as the container holds them: float32 [K][N]."""
    info = q.moe_meta["tensors"][name]
    rt, kt = q.shape(name)[0], q.shape(name)[1]
    rte = rt // info["experts"]
    lo = info["offsets"][e]
    raw = np.asarray(q.mm[q.base + lo:q.base + lo + info["bytes"]]).reshape(rte, kt, TILE)
    return untile(raw.reshape(-1, TILE), info["rows"], info["cols"]).astype(np.float32).T


def pad_n(wt, n_pad):
    """Zero-pad a [K][N] block up to [K][n_pad]: the arena's n-tile geometry needs N a multiple of
    1,024 at NW=32, and an expert's 768 columns are not. It only ever WIDENS - down_proj's 2,048
    columns are already fine and must not be cut to some model-wide padding."""
    if n_pad is None or n_pad <= wt.shape[1]:
        return wt
    out = np.zeros((wt.shape[0], n_pad), np.float32)
    out[:, :wt.shape[1]] = wt
    return out


def quantise(wt):
    """[K][N] float32 -> (codes int8 symmetric int4, d float32 [K//GROUP][N])."""
    K, N = wt.shape
    g = wt.reshape(K // GROUP, GROUP, N)
    d = (np.abs(g).max(axis=1) / 7.0).astype(np.float32)
    safe = np.where(d > 0, d, 1.0)
    codes = np.clip(np.rint(g / safe[:, None, :]), -8, 7).reshape(K, N).astype(np.int8)
    return codes, d


def dequantise(codes, d):
    return codes.astype(np.float32) * np.repeat(d, GROUP, axis=0)


def info_n(q, name):
    return q.moe_meta["tensors"][name]["rows"]


def chunk_path(out, layer, proj, e):
    return os.path.join(out, f"L{layer}_{proj}_e{e}.tf16")


def write_chunk(path, codes, d):
    with open(path, "wb") as f:
        f.write(codes.reshape(-1).view(np.uint8).tobytes())
        f.write(d.astype(np.float16).view(np.uint16).tobytes())


def read_chunk(path, K, N):
    raw = np.fromfile(path, dtype=np.uint8)
    want = K * N + (K // GROUP) * N * 2
    if raw.size != want:
        raise SystemExit(f"{path}: {raw.size} B, want {want} B")
    codes = raw[:K * N].view(np.int8).reshape(K, N)
    d = raw[K * N:].view("<u2").view(np.float16).astype(np.float32).reshape(K // GROUP, N)
    return codes, d


def sizes(K, N):
    """The three sizes this tool has to keep apart, each named for the format it describes.

    read_tf16_bytes  what chunks() WRITES per expert: int8 int4 codes [K][N] (one byte each,
                     padded to N) followed by fp16 d [K//128][N]. This is layout_w4.py's
                     read_tf16 INPUT, a host-side conversion artifact, not a device stream.
    pb16_bytes       what the device streams after the arena packs those codes: nibble-packed
                     K*N/2 plus one fp16 multiplier per 32 K-rows per column. This is the number
                     a decode-byte budget is computed from.
    src_bytes        the same expert in the container: q4_1 tiles, K*N/32 chunks of 5120 B.
    """
    return {
        "read_tf16_bytes": K * N + (K // GROUP) * N * 2,
        "pb16_bytes": K * N // 2 + (K // 32) * N * 2,
        "src_bytes": K * N * 5 // 8,   # q4_1 tiles: 5120 B per 8192 weights
    }


def geometry(q, layer, pad_n=None):
    """What the kernel generator has to know for this model's experts, padding included.

    The n-tile constraint is not N % 32: a core owns CW = N/(tiles across) columns of the weight,
    so at NW=32 the per-core slice CW = N/32 must itself be whole 32-wide tiles (N % 1024 == 0),
    and at NW=16 that CW = N/16 must be whole 16-wide tiles (N % 256 == 0).
    """
    out = {}
    for proj, suffix in PROJ.items():
        info = q.moe_meta["tensors"][q.layer_name(layer, suffix)]
        K, N = info["cols"], info["rows"]
        n_out = max(N, pad_n or N)
        out[proj] = {
            "K": K, "N": N, "N_padded": n_out,
            **sizes(K, n_out),
            "src_bytes": info["bytes"],   # the container's own per-expert span, padding aside
            "cw_nw32": n_out // 32, "cw_nw16": n_out // 16,
            "ok_at_nw32": n_out % 1024 == 0, "ok_at_nw16": n_out % 256 == 0,
            "k_multiple_of_256": K % 256 == 0,
        }
    return out


def token_stream(q, layer, pad_n=None):
    """Bytes one decode step streams, per format: layers x top-k experts x 3 projections.

    Both are reported because they answer different questions and differ by 1.8x: pb16_bytes is
    the device's decoder stream (the number a byte budget uses), read_tf16_bytes is what a host
    reads to build it.
    """
    g = geometry(q, layer, pad_n)
    out = {}
    for key in ("pb16_bytes", "read_tf16_bytes", "src_bytes"):
        per_layer = sum(v[key] for v in g.values()) * q.top_k
        out[key.replace("_bytes", "_per_token")] = per_layer * q.layers
    return out


def chunks(args):
    q = Q4nx(args.model)
    os.makedirs(args.out, exist_ok=True)
    layers = range(q.layers if args.layers is None else args.layers)
    experts = range(q.experts if args.experts is None else args.experts)
    wrote = 0
    for L in layers:
        for proj, suffix in PROJ.items():
            name = q.layer_name(L, suffix)
            for e in experts:
                w = pad_n(container_expert(q, name, e), args.pad_n or info_n(q, name))
                write_chunk(chunk_path(args.out, L, proj, e), *quantise(w))
                wrote += 1
        print(f"layer {L}: 3 projections x {len(experts)} experts written", flush=True)
    manifest = {
        "container": os.path.abspath(args.model),
        "source": "model.q4nx __metadata__.moe.tensors[<name>].offsets (per-expert chunks)",
        "format": "int4 codes as int8 [K][N], then fp16 d [K//128][N] (layout_w4.py read_tf16)",
        "quantisation": "symmetric int4, one multiplier per 128 K-rows per column; the container's "
                        "q4_1 zero point is dropped, so verify's error is ON TOP of the container's",
        "geometry": geometry(q, layers[0], args.pad_n),
        "decode_stream": token_stream(q, layers[0], args.pad_n),
        "files": wrote,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["geometry"], indent=2))
    print(f"wrote {wrote} chunks + manifest.json to {args.out}")
    return 0


def verify(args):
    q = Q4nx(args.model)
    bad = 0
    for proj, suffix in PROJ.items():
        name = q.layer_name(args.layer, suffix)
        info = q.moe_meta["tensors"][name]
        K, N = info["cols"], info["rows"]
        worst = {"rel": 0.0, "cos": 1.0, "max": 0.0}
        for e in range(args.experts):
            ref = container_expert(q, name, e).astype(np.float64)
            back = dequantise(*read_chunk(chunk_path(args.out, args.layer, proj, e), K, N))
            err = np.abs(back.astype(np.float64) - ref)
            rel = float(np.sqrt((err ** 2).sum() / max((ref ** 2).sum(), 1e-30)))
            cos = float((back * ref).sum() / np.sqrt((back * back).sum() * (ref * ref).sum()))
            worst = {"rel": max(worst["rel"], rel), "cos": min(worst["cos"], cos),
                     "max": max(worst["max"], float(err.max()))}
        ok = worst["cos"] > 0.98
        bad += not ok
        print(f"{'ok  ' if ok else 'BAD'} {proj:5s} {K} x {N}: worst of {args.experts} experts "
              f"rel rms {worst['rel']:.4f}, cos {worst['cos']:.6f}, max |dw| {worst['max']:.4f}")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chunks")
    c.add_argument("model")
    c.add_argument("out")
    c.add_argument("--layers", type=int, default=None, help="only the first N layers")
    c.add_argument("--experts", type=int, default=None, help="only the first N experts")
    c.add_argument("--pad-n", dest="pad_n", type=int, default=None,
                   help="zero-pad each expert's N to this many columns (the arena needs 1,024 "
                        "multiples at NW=32; 768 -> 1024 costs 25 %% of the chunk)")
    v = sub.add_parser("verify")
    v.add_argument("model")
    v.add_argument("out")
    v.add_argument("--layer", type=int, default=0)
    v.add_argument("--experts", type=int, default=2)
    g = sub.add_parser("gate", help="expected int32 for one token, one layer, one projection")
    g.add_argument("model")
    g.add_argument("out")
    g.add_argument("--prompt", required=True)
    g.add_argument("--layer", type=int, default=0)
    g.add_argument("--proj", default="gate", choices=sorted(PROJ))
    g.add_argument("--experts", type=int, default=8)
    g.add_argument("--pad-n", dest="pad_n", type=int, default=None,
                   help="zero-pad N to this many columns (the arena's NW=32 geometry)")
    a = p.parse_args()
    return {"chunks": chunks, "verify": verify, "gate": gate}[a.cmd](a)



# --- the byte-exact gate a chunk-list kernel is judged by -------------------------------------

def gate(args):
    """Emit everything a device gate needs for ONE token, ONE layer, ONE projection.

    The kernel's contract is int8 A x int4 W -> int32 with per-row A scales, so the expected
    tensor is exact integer arithmetic, not a float comparison: this writes the activation row,
    its int8 form and scale, the router's choice, and each chosen expert's expected int32.
    """
    import moe_q4nx_ref as R

    q = Q4nx(args.model)
    fw = R.Forward(q)
    chat = R.Chat(args.model)
    ids = chat.encode(chat.render(args.prompt))
    fw.capture_layer = args.layer
    fw.capture_stop = True
    fw.logits(ids)
    h = fw.captured[-1] if fw.captured is not None else None
    if h is None:
        raise SystemExit(f"layer {args.layer} did not run")
    router = q.weight(q.layer_name(args.layer, "mlp.gate.weight"))
    p = R.softmax((router @ h).astype(np.float64))
    top = np.argsort(-p, kind="stable")[:args.experts]
    w = p[top] / p[top].sum() if q.norm_topk else p[top]

    # A: int8 per row with one scale, the convention the launch path takes (ascales)
    ascale = float(np.abs(h).max() / 127.0) or 1.0
    a8 = np.clip(np.rint(h / ascale), -127, 127).astype(np.int8)

    name = q.layer_name(args.layer, PROJ[args.proj])
    info = q.moe_meta["tensors"][name]
    K, N = info["cols"], info["rows"]
    os.makedirs(args.out, exist_ok=True)
    expected = {}
    for e in top:
        codes, d = quantise(pad_n(container_expert(q, name, int(e)), args.pad_n))
        exact = codes.astype(np.int32).T @ a8.astype(np.int32)      # [N]
        expected[int(e)] = {"d_max": float(d.max()),
                            "d_bits": d.astype(np.float16).view(np.uint16).tolist(),
                            "int32": exact.tolist()}
    out = {
        "layer": args.layer, "proj": args.proj, "K": K, "N": N,
        "N_padded": int(args.pad_n or N),
        "prompt": args.prompt, "prompt_tokens": len(ids),
        "router_top": [int(e) for e in top],
        "router_weights": [float(x) for x in w],
        "expert_weights": {int(e): float(x) for e, x in zip(top, w)},
        "ascale": ascale,
        "A_f32": h.tolist(),
        "A_int8": a8.tolist(),
        "expected_int32": expected,
        "how": "expected_int32 = codes^T . a8 in int32 (exact); scale by d and ascale on the host",
    }
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(out, f)
    print(f"layer {args.layer} {args.proj}: experts {[int(e) for e in top]}, "
          f"weights {[round(float(x), 4) for x in w]}, ascale {ascale:.6g}, K={K} N={N}")
    print(f"A_int8 range [{a8.min()}, {a8.max()}], expected int32 range "
          f"[{min(min(v['int32']) for v in expected.values())}, "
          f"{max(max(v['int32']) for v in expected.values())}]")
    print(f"wrote {args.out}/gate.json")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
