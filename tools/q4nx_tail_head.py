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
"""Measure a 1.9 pack's tied head against the pack's own fp16 embedding.

engine#1693 was found with a scratch script: on `Qwen2.5-0.5B-Instruct_rai_1.9.0_npu` the head's self-check reported
`corr 0.885220`, `rel_err 0.987007` against the pack's own embedding although `tie_word_embeddings` is true.  Those two
figures are **not** a property of the tail: they are the #1661 self-check bug, which decoded the container's bf16 copy at
F16 (`embed_bytes` was repointed at `embed_bf16` and the readers kept `f16_to_f32`).  engine#1690 fixes only that decode
and changes no offset, and the importer's figure on this pack becomes `corr 0.999999`, `rel_err 0.0007`.  A number that
cannot be re-run is a finding nobody can check (and a decode that agrees with itself proves nothing), so this is that
measurement as a command: geometry from the numbers `npu/pack_import.cpp` uses, decode of the tail's tile grid, and three statistics
against the fp16 `model.embed_tokens.weight` that the pack already carries.

What this tool is NOT: a fix.  On the pack above it reports the geometry as *exact* (the region is `V/64` tiles of
`attn_tile_bytes(H)` and its size is the int4 code size times the per-tile envelope ratio, to 4 dp) while the decode
correlates far from 1, so the open question is the tail tile's envelope (which bytes hold scale/zero for which rows),
not the carve.  Printing all three numbers keeps both halves of that visible instead of one.

Usage:
  tools/q4nx_tail_head.py <pack-dir> [--tiles N] [--json]

Exit status: 0 measured, 2 the pack cannot be measured (no model.onnx.data / no embed_tokens / no .fconst), 3 measured
with a geometry that does not match the tail (so the numbers below it describe the wrong bytes).
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

K_TILE_ROWS = 64          # npu/pack_import.cpp: kTileRows
K_KBLOCK = 128            # npu/pack_import.cpp: kKBlock
K_NORM_OFF = 64           # npu/pack_import.cpp: kNormOff


def subv_bytes(rows):
    if rows <= 0 or rows % 64:
        return 0
    return 64 * (rows // 2) + 64 + rows


def group_bytes(k, rows):
    if k <= 0 or k % K_KBLOCK:
        return 0
    sv = subv_bytes(rows)
    return 0 if sv == 0 else (2 * (k // K_KBLOCK) + 2) * sv


def attn_tile_bytes(k):
    return group_bytes(k, K_TILE_ROWS)


def attn_payload_off(hidden):
    return K_NORM_OFF + hidden * 2


def head_offset(fconst_bytes, ntiles, tile):
    """Where the tied head's grid starts: the last `ntiles * tile` bytes of the `.fconst`.

    The importer derives the same number as `layers * stride + attn_payload_off(H)` -- past the per-layer payload, then
    the tail's own envelope -- and *verifies* it against the remaining file size (`want == have`, pack_import.cpp), so
    the two agree whenever the tail is the thing the file ends with. Using the subtraction here is deliberate: it needs
    neither the layer count nor the eps-anchor-derived `stride`, and a file whose end is something else shows up as a
    correlation that is not 1 rather than as a silently wrong offset.

    Reading from `attn_payload_off(H)` alone (without the `layers * stride` term) addresses the wrong bytes by the whole
    payload. That is a mistake made *while chasing* engine#1693 and it is worth naming, but it is not what #1693
    measured: at that offset this tool reads chance (tile corr -0.0098 / -0.0098 / +0.0197), never 0.885, and the
    importer never reads the head there at all -- `import_pack()` takes it at `tail_off = layers * stride`. What #1693
    recorded (0.885220 / 0.987007) is #1661's bf16-read-as-F16 decode, fixed by #1690. At the correct offset the decode
    correlates 1.0000 with the pack's own fp16 embedding, magnitude ratio 1.000, rel_err 0.0007 over four tiles on
    `Qwen2.5-0.5B-Instruct_rai_1.9.0_npu`.
    """
    return fconst_bytes - ntiles * tile


def bf16_to_f32(codes):
    """bf16_to_f32: the 16-bit pattern is the f32 word's high half."""
    return np.frombuffer((np.asarray(codes, np.uint16).astype(np.uint32) << 16).tobytes(), np.float32)


def config(pack):
    with open(os.path.join(pack, "config.json")) as f:
        return json.load(f)


def embedding(pack, vocab, hidden):
    """`model.embed_tokens.weight` as f16 [V, H], from model.onnx.data via its external_data record."""
    try:
        import onnx
    except ImportError:
        return None, "onnx is not installed"
    path = os.path.join(pack, "model.onnx")
    if not os.path.exists(path):
        return None, "no model.onnx"
    model = onnx.load(path, load_external_data=False)
    found = [t for t in model.graph.initializer if "embed_tokens" in t.name]
    if not found:
        return None, "no embed_tokens initializer in the graph"
    tensor = found[0]
    ext = {e.key: e.value for e in tensor.external_data}
    if "location" not in ext or "offset" not in ext or "length" not in ext:
        return None, "embed_tokens is not external data"
    off, length = int(ext["offset"]), int(ext["length"])
    data = os.path.join(pack, ext["location"])
    if not os.path.exists(data):
        return None, "no " + ext["location"]
    if length != vocab * hidden * 2:
        return None, "embed_tokens declares %d B, V*H*2 is %d" % (length, vocab * hidden * 2)
    view = np.memmap(data, dtype=np.uint8, mode="r", offset=off, shape=(length,))
    return np.frombuffer(view, dtype=np.float16, count=vocab * hidden).reshape(vocab, hidden).astype(np.float32), None


def fconst(pack):
    """The Token-stage .fconst: the one the tail belongs to (there may be a Prefill one too)."""
    hits = sorted(glob.glob(os.path.join(pack, "dd_metastate_*Token*.fconst")))
    return hits[0] if hits else None


def decode_tile(tile, k, rows):
    """The envelope `npu/pack_import.cpp:decode_tile` documents: bias [2 subvolumes], then a subvolume per (row half, k block)."""
    sv = subv_bytes(rows)
    nsubv = rows // 2
    bp = bytes(tile)
    zeros_off = 64 * nsubv
    scales_off = zeros_off + 64
    kbs = k // K_KBLOCK
    code = np.zeros((rows, k), np.int32)
    zp = np.zeros((rows, kbs), np.int32)
    scale = np.zeros((rows, kbs), np.float32)
    bias = np.zeros(rows, np.uint16)
    for h in range(2):
        r0 = 32 * h
        bias[r0:r0 + nsubv] = np.frombuffer(bp[h * sv:h * sv + 2 * nsubv], dtype="<u2")
        for kb in range(kbs):
            s = np.frombuffer(bp[(2 + 2 * kb + h) * sv:(3 + 2 * kb + h) * sv], dtype=np.uint8)
            for cp in range(nsubv // 2):
                n0 = r0 + 2 * cp
                base = (cp % 4) + (cp // 4) * 512
                b = s[base:base + K_KBLOCK * 4:4]
                code[n0, kb * K_KBLOCK:(kb + 1) * K_KBLOCK] = b & 0x0F
                code[n0 + 1, kb * K_KBLOCK:(kb + 1) * K_KBLOCK] = np.asarray(b, np.int32) >> 4
            for i in range(nsubv // 2):
                z = s[zeros_off + i]
                zp[r0 + 2 * i, kb] = z & 0x0F
                zp[r0 + 2 * i + 1, kb] = z >> 4
            scale[r0:r0 + nsubv, kb] = bf16_to_f32(
                np.frombuffer(s[scales_off:scales_off + 2 * nsubv].tobytes(), dtype="<u2"))
    w = (code - np.repeat(zp, K_KBLOCK, axis=1)) * np.repeat(scale, K_KBLOCK, axis=1)
    return w, bias


def measure(pack, tiles_wanted):
    cfg = config(pack)
    vocab = int(cfg["vocab_size"])
    hidden = int(cfg["hidden_size"])
    emb, why = embedding(pack, vocab, hidden)
    if emb is None:
        return None, 2, why
    path = fconst(pack)
    if not path:
        return None, 2, "no dd_metastate_*Token*.fconst"
    size = os.path.getsize(path)
    tile = attn_tile_bytes(hidden)
    env = attn_payload_off(hidden)
    ntiles = vocab // K_TILE_ROWS
    if tile == 0 or vocab % K_TILE_ROWS:
        return None, 2, "H %d is not a multiple of %d" % (hidden, K_KBLOCK)
    need = env + ntiles * tile
    # WHERE the head is: the LAST `ntiles * tile` bytes. `env` on its own is only the tail's envelope -- the importer
    # reaches the head as `layers*stride + env`, which equals this subtraction whenever its size check passes
    # (engine#1693: at `env` the decode correlates at chance, at the real offset 1.0000).
    base = head_offset(size, ntiles, tile)
    rows = []
    status = 0 if base >= 0 else 3
    with open(path, "rb") as f:
        for t in range(min(tiles_wanted, ntiles)):
            f.seek(base + t * tile)
            buf = f.read(tile)
            if len(buf) != tile:
                break
            w, bias = decode_tile(buf, hidden, K_TILE_ROWS)
            ref = emb[t * K_TILE_ROWS:(t + 1) * K_TILE_ROWS]
            flat, flatref = w.ravel().astype(np.float64), ref.ravel().astype(np.float64)
            corr = float(np.corrcoef(flat, flatref)[0, 1]) if flat.std() and flatref.std() else float("nan")
            rel = float(np.abs(flat - flatref).sum() / max(np.abs(flatref).sum(), 1e-12))
            rows.append({"tile": t, "corr": corr, "rel_err": rel,
                         "abs_ratio": float(np.abs(flat).sum() / max(np.abs(flatref).sum(), 1e-12))})
    out = {
        "pack": os.path.basename(os.path.abspath(pack)),
        "vocab": vocab, "hidden": hidden,
        "geometry": {"env": env, "head_off": base, "tile_bytes": tile, "tiles": ntiles, "tail_bytes": need,
                     "fconst_bytes": size,
                     "int4_ratio": (ntiles * tile) / (vocab * hidden / 2.0) if vocab * hidden else 0.0,
                     "tile_ratio": tile / (K_TILE_ROWS * hidden / 2.0) if hidden else 0.0,
                     "tail_fits": base >= 0},
        "tiles_measured": rows,
    }
    return out, status, None


def main(argv):
    ap = argparse.ArgumentParser(description="Measure a 1.9 pack's tied head against its own fp16 embedding (engine#1693).")
    ap.add_argument("pack")
    ap.add_argument("--tiles", type=int, default=4, help="tiles to decode from the tail's start (default 4)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv[1:])
    out, status, why = measure(a.pack, a.tiles)
    if out is None:
        print("q4nx_tail_head: cannot measure %s: %s" % (a.pack, why), file=sys.stderr)
        return status
    g = out["geometry"]
    if a.json:
        print(json.dumps(out, indent=2))
        return status
    print("%s: V %d H %d" % (out["pack"], out["vocab"], out["hidden"]))
    print("  head: the last %d tiles x %d B of %d B, so at %d (env-only base %d is the pre-head payload)"
          % (g["tiles"], g["tile_bytes"], g["fconst_bytes"], g["head_off"], g["env"]))
    print("  int4 ratio %.4f, per-tile ratio %.4f (equal means the surplus is the envelope, not extra tensors)"
          % (g["int4_ratio"], g["tile_ratio"]))
    for r in out["tiles_measured"]:
        print("  tile %4d  corr %+0.4f  rel_err %0.4f  |w|/|ref| %0.4f"
              % (r["tile"], r["corr"], r["rel_err"], r["abs_ratio"]))
    if status == 3:
        print("  the head grid is larger than the .fconst: the numbers above describe other bytes", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
