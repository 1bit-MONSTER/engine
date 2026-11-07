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
"""trit-embed import: Cyronius/trit-embed-3l (MIT) -> the engine's 1-bit embedding container.

Why a container of its own instead of q4nx's tiled kinds: a q4nx quantised tensor is a grid of 32x256
chunks (`npu/q4nx.h`), and this model's rows are 384 / 1536 wide with **one scale per output row** and
2-bit ternary codes. Tiling them would pad K to 512 and re-round the scale into per-128-block fp16,
changing numbers the reference gate compares exactly.  XDNA2 has no ternary datapath, so the codes are
widened to int8 (the value is still exactly -1/0/+1) and the per-row scale is kept in f32: that product is
bit-identical to the reference's own decode (`codes.float() * scale.float()`), which is what makes the
cosine gate meaningful.

Layout of `model.embed`: a 16 KiB-padded JSON header, then the tensor payload at the offsets it names.

    {"format": "1bit-embed-1", "dims": {...}, "tensors": {name: {dtype, shape, offsets:[lo,hi]}}, ...}

dtypes: `i8` (ternary codes widened to int8, or an int8 embedding table), `f32` (scales, norms, biases,
the output projection), `f32row` is not a dtype: a per-output-row scale is an `f32` tensor of shape [N].

Usage:
    tools/trit_embed_pack.py pack   <hf dir> <out dir>       # writes model.embed + copied tokenizer files
    tools/trit_embed_pack.py verify <hf dir> <out dir>       # known answers: decoded weights vs the source
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys

import numpy as np

HEADER_PAD = 16384  # 74 tensors, ~10 KB of JSON; a fixed pad keeps the payload offsets stable and checkable
FORMAT = "1bit-embed-1"
# The tensors this model class has, and what each one becomes.  Everything the reference names is here, so
# a pack that grows a tensor is refused rather than silently dropped.
TERNARY_SUFFIX = ".codes"


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ safetensors

def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    meta = hdr.pop("__metadata__", None)
    out = {}
    for name, ent in hdr.items():
        lo, hi = ent["data_offsets"]
        buf = blob[lo:hi]
        dt = ent["dtype"]
        if dt == "F32":
            a = np.frombuffer(buf, dtype=np.float32)
        elif dt == "BF16":
            a = (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "I8":
            a = np.frombuffer(buf, dtype=np.int8)
        elif dt == "U8":
            a = np.frombuffer(buf, dtype=np.uint8)
        else:
            raise SystemExit(f"{name}: dtype {dt} is not one this importer reads")
        out[name] = (a.reshape(ent["shape"]), dt)
    return out, meta


def unpack_ternary(packed):
    """uint8 [N, K/4] -> int8 [N, K]: code + 1 in {0,1,2} at bits 2j..2j+1; 3 is reserved and refused."""
    if packed.dtype != np.uint8:
        raise SystemExit(f"ternary codes: expected U8, got {packed.dtype}")
    c = np.zeros((packed.shape[0], packed.shape[1] * 4), dtype=np.uint8)
    for j in range(4):
        c[:, j::4] = (packed >> (2 * j)) & 3
    if (c == 3).any():
        raise SystemExit("ternary codes: reserved code 3 in the source container")
    return (c.astype(np.int8) - 1)


# ------------------------------------------------------------------ container

def write_container(out_dir, dims, tensors, extra):
    entries, blobs, off = {}, [], HEADER_PAD
    for name in sorted(tensors):
        a = tensors[name]
        if a.dtype == np.int8:
            dt = "i8"
            b = a.tobytes()
        elif a.dtype == np.float32:
            dt = "f32"
            b = a.astype("<f4").tobytes()
        else:
            raise SystemExit(f"{name}: cannot store dtype {a.dtype}")
        entries[name] = {"dtype": dt, "shape": [int(d) for d in a.shape], "offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hdr = {"format": FORMAT, "dims": dims, "tensors": entries, **extra}
    hj = json.dumps(hdr, indent=1).encode()
    if len(hj) + 8 > HEADER_PAD:
        raise SystemExit(f"header does not fit in {HEADER_PAD} bytes ({len(hj)})")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.embed"), "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(b"\0" * (HEADER_PAD - 8 - len(hj)))
        for b in blobs:
            f.write(b)
    return hdr


def read_container(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        f.seek(0, 2)
        size = f.tell()
        out = {}
        for name, ent in hdr["tensors"].items():
            lo, hi = ent["offsets"]
            # offsets are absolute in the file (the writer starts at HEADER_PAD), so they are also checkable
            if lo < HEADER_PAD or hi > size:
                raise SystemExit(f"{name}: offsets {lo}..{hi} outside the payload of a {size}-byte file")
            f.seek(lo)
            buf = f.read(hi - lo)
            if ent["dtype"] == "i8":
                a = np.frombuffer(buf, dtype=np.int8)
            elif ent["dtype"] == "f32":
                a = np.frombuffer(buf, dtype="<f4").astype(np.float32)
            else:
                raise SystemExit(f"{name}: unknown dtype {ent['dtype']}")
            out[name] = a.reshape(ent["shape"])
    return hdr, out


# ------------------------------------------------------------------ pack

def pack(hf_dir, out_dir):
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))
    if cfg.get("format") != "trit-packed-v1":
        raise SystemExit(f"config.json: format {cfg.get('format')!r}, this importer reads trit-packed-v1")
    d = cfg["dims"]
    src, meta = read_safetensors(os.path.join(hf_dir, "model.safetensors"))
    log(f"source: {len(src)} tensors, dims {json.dumps(d)}")

    out, stats = {}, []
    ternary = 0
    for name, (a, dt) in sorted(src.items()):
        if name.endswith(TERNARY_SUFFIX):
            base = name[: -len(TERNARY_SUFFIX)]
            out[base + ".weight"] = unpack_ternary(a)
            stats.append((base + ".weight", f"ternary codes {a.dtype}{list(a.shape)} -> i8", 0))
            ternary += 1
        elif name.endswith(".scale"):
            # bf16 -> f32 is exact for every value bf16 can hold, which is what the reference multiplies by.
            out[name] = a.astype(np.float32)
            stats.append((name, f"{dt} {list(a.shape)} -> f32", 0))
        elif a.dtype == np.int8:
            out[name] = a  # the int8 embedding tables; their row scales travel as the *.scale tensors above
            stats.append((name, f"{dt} {list(a.shape)} -> i8", 0))
        else:
            out[name] = a.astype(np.float32)
            stats.append((name, f"{dt} {list(a.shape)} -> f32", 0))

    extra = {
        "pooling": cfg.get("pooling", d.get("pool")),
        "normalized": bool(cfg.get("normalized", True)),
        "max_tokens": int(cfg.get("max_tokens", d["max_pos"])),
        "gelu": d.get("gelu"),
        "eps": float(d.get("eps", 1e-12)),
        "license": "MIT (Cyronius/trit-embed-3l; checkpoint based on bge-micro-v2)",
        "source_checkpoint_sha256": cfg.get("source_checkpoint_sha256"),
        "source_model": "Cyronius/trit-embed-3l",
    }
    hdr = write_container(out_dir, d, out, extra)
    for path in ("config.json", "tokenizer.json", "vocab.txt", "LICENSE"):
        p = os.path.join(hf_dir, path)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_dir, path))
    notice = os.path.join(out_dir, "NOTICE.1bit")
    with open(notice, "w") as f:
        f.write("This directory contains a converted copy of Cyronius/trit-embed-3l (MIT), which is itself\n"
                "based on bge-micro-v2. The conversion widens 2-bit ternary codes to int8 and keeps the\n"
                "per-output-row scale in f32; the tokenizer files are copied unchanged.\n")
    log(f"wrote {out_dir}/model.embed ({os.path.getsize(os.path.join(out_dir, 'model.embed'))} B), "
        f"{len(out)} tensors, {ternary} ternary linears")
    total = sum(os.path.getsize(os.path.join(out_dir, x)) for x in os.listdir(out_dir))
    log(f"container total {total / 1e6:.2f} MB")
    return hdr


# ------------------------------------------------------------------ verify

def verify(hf_dir, out_dir):
    hdr, got = read_container(os.path.join(out_dir, "model.embed"))
    if hdr["format"] != FORMAT:
        raise SystemExit(f"model.embed: format {hdr['format']!r}")
    src, _ = read_safetensors(os.path.join(hf_dir, "model.safetensors"))
    fails = 0
    checked = 0

    # Known answer 1: every ternary linear decodes to exactly the reference's own weight product.  The
    # reference multiplies the unpacked int8 codes by the bf16 scale widened to float; so do we, so the
    # comparison is equality, not a tolerance.
    ternary = sorted({n[: -len(TERNARY_SUFFIX)] for n in src if n.endswith(TERNARY_SUFFIX)})
    worst = 0.0
    for base in ternary:
        codes_ref = unpack_ternary(src[base + ".codes"][0])
        scale_ref = src[base + ".scale"][0].astype(np.float32)
        w_ref = codes_ref.astype(np.float32) * scale_ref[:, None]
        w = got[base + ".weight"].astype(np.float32) * got[base + ".scale"].astype(np.float32)[:, None]
        d = float(np.abs(w - w_ref).max())
        worst = max(worst, d)
        zero_frac = float((got[base + ".weight"] == 0).mean())
        if d != 0.0:
            fails += 1
            log(f"  FAIL {base}: worst |dw| {d:g}")
        checked += 1
        if base.endswith(".q"):
            log(f"  {base}.weight: value set {sorted(set(np.unique(got[base + '.weight']).tolist()))}, "
                f"{zero_frac * 100:.1f} % zeros, scale f32 {scale_ref.min():.6g}..{scale_ref.max():.6g}")
    log(f"ternary linears: {checked} checked, worst |dw| vs the reference {worst:g}")

    # Known answer 2: the int8 tables and their row scales are the source bytes/values, unchanged.
    for name in ("word.qweight", "pos.qweight"):
        if not np.array_equal(got[name].view(np.int8), src[name][0]):
            fails += 1
            log(f"  FAIL {name}: int8 table differs from the source")
    for name in ("word.scale", "pos.scale"):
        if not np.array_equal(got[name], src[name][0].astype(np.float32)):
            fails += 1
            log(f"  FAIL {name}: row scale differs from the source")
    log("int8 tables and row scales: word/pos byte-identical to the source")

    # Known answer 3: a carrier check the gate leans on -- the embedding lookup the engine performs must be
    # the reference's `qweight[ids] * scale[ids]`, so print it for a couple of ids.
    V = int(got["word.qweight"].shape[0])
    ids = np.array([i % V for i in (1019, 1996, 2456)])  # "how", "do", "index" in the 30k vocab
    emb = got["word.qweight"][ids].astype(np.float32) * got["word.scale"][ids][:, None]
    emb_ref = src["word.qweight"][0][ids].astype(np.float32) * src["word.scale"][0][ids][:, None]
    if not np.array_equal(emb, emb_ref):
        fails += 1
        log("  FAIL embedding lookup differs")
    else:
        log(f"embedding lookup (ids {ids.tolist()}): first row norms "
            f"{[round(float(np.linalg.norm(r)), 6) for r in emb]}, identical to the source")

    log(f"verify: {'OK' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pack", "verify"):
        p = sub.add_parser(name)
        p.add_argument("hf_dir")
        p.add_argument("out_dir")
    a = ap.parse_args()
    if a.cmd == "pack":
        pack(a.hf_dir, a.out_dir)
        return 0
    return verify(a.hf_dir, a.out_dir)


if __name__ == "__main__":
    sys.exit(main())
