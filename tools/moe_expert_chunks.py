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

  moe_expert_chunks.py chunks <model dir> <out dir> [--layers N] [--experts N]
        per expert: `L<layer>_<gate|up|down>_e<expert>.tf16` = int4 codes as int8 [K][N] then
        fp16 d bits [K//128][N] (layout_w4.py's read_tf16 input), plus manifest.json with the
        geometry the kernel generator needs (K, N, n-tiles at NW=32 and NW=16).

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


def geometry(q, layer):
    """What the kernel generator has to know for this model's experts.

    The n-tile constraint is not N % 32: a core owns CW = N/(tiles across) columns of the weight,
    so at NW=32 the per-core slice CW = N/32 must itself be whole 32-wide tiles (N % 1024 == 0),
    and at NW=16 that CW = N/16 must be whole 16-wide tiles (N % 256 == 0).
    """
    out = {}
    for proj, suffix in PROJ.items():
        info = q.moe_meta["tensors"][q.layer_name(layer, suffix)]
        K, N = info["cols"], info["rows"]
        out[proj] = {
            "K": K, "N": N,
            "chunk_bytes": K * N // 2 + (K // 32) * N * 2,  # pb16(i) stream, 5 bits/weight
            "src_bytes_per_expert": info["bytes"],           # q4_1 tiles in the container
            "cw_nw32": N // 32, "cw_nw16": N // 16,
            "ok_at_nw32": N % 1024 == 0, "ok_at_nw16": N % 256 == 0,
            "k_multiple_of_256": K % 256 == 0,
        }
    return out


def token_stream(q, layer):
    """Bytes one decode step streams: 48 layers x top-k experts x 3 projections."""
    per_layer = sum(v["chunk_bytes"] for v in geometry(q, layer).values()) * q.top_k
    return {"per_layer": per_layer, "per_token": per_layer * q.layers}


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
                write_chunk(chunk_path(args.out, L, proj, e), *quantise(container_expert(q, name, e)))
                wrote += 1
        print(f"layer {L}: 3 projections x {len(experts)} experts written", flush=True)
    manifest = {
        "container": os.path.abspath(args.model),
        "source": "model.q4nx __metadata__.moe.tensors[<name>].offsets (per-expert chunks)",
        "format": "int4 codes as int8 [K][N], then fp16 d [K//128][N] (layout_w4.py read_tf16)",
        "quantisation": "symmetric int4, one multiplier per 128 K-rows per column; the container's "
                        "q4_1 zero point is dropped, so verify's error is ON TOP of the container's",
        "geometry": geometry(q, layers[0]),
        "decode_stream": token_stream(q, layers[0]),
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
    v = sub.add_parser("verify")
    v.add_argument("model")
    v.add_argument("out")
    v.add_argument("--layer", type=int, default=0)
    v.add_argument("--experts", type=int, default=2)
    a = p.parse_args()
    return chunks(a) if a.cmd == "chunks" else verify(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
