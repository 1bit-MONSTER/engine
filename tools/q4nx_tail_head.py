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

What this tool is NOT: a fix, and not a decoder for every head.  engine#1740 read "carves exactly, decodes to nothing"
on two packs, and the cause was neither a wrong carve nor a wrong block-128 envelope.  Measured, on the full packs:

  * `Qwen-2.5_1.5B_Instruct_rai_1.9.0_npu` (tied, H 1536): its head is **block 32** along K (`/lm_head/MatMul_Q4`,
    `block_size 32`; every layer op is 128).  This tool decodes the block-128 tile only, so it read that head as NaN /
    1e38 scales.  It now reads the head's `block_size` from the graph and refuses a non-128 head by name (exit 4).  The
    block-32 decoder is engine#1503's.
  * `Phi-3-mini-128k-instruct_rai_1.9.0_npu` (**untied**): the decode was right all along.  Against the upstream
    checkpoint's own `lm_head.weight` (rows 0-255) tiles 0-3 correlate +0.994-0.995.  It read "nothing" because an
    untied head was compared with the embedding, which is not its oracle.  An untied pack now needs `--ref`, or the
    tool says so and exits 5 instead of printing a correlation that means nothing.

And one statistic is gone: "int4 ratio == per-tile ratio" was printed as evidence that the region is this grid.  It is
an identity: `ntiles * tile / (V*H/2)` and `tile / (64*H/2)` are the same number whenever `ntiles = V/64`, which is how
`ntiles` is computed, so it agreed on a block-32 head too.  What the geometry line reports instead is whether the head
fits and how many zero bytes the file ends in.  The end-anchored offset assumes the head is the file's last bytes, and
the 1.5B pack's `.fconst` ends in 110,208,000 zero bytes.

Usage:
  tools/q4nx_tail_head.py <pack-dir> [--tiles N] [--json] [--ref FILE [--ref-dtype bf16|f16|f32]]

`--ref` is the head's own reference rows, raw little-endian `[rows, H]` from row 0, for an untied pack.  For example,
range-read the first 256 rows of `lm_head.weight` out of the upstream checkpoint's safetensors shard.  For a tied pack
the reference is the pack's own `model.embed_tokens.weight`, and `--ref` overrides it.

Exit status: 0 measured against an oracle; 2 the pack cannot be measured (no model.onnx.data / no embed_tokens / no
.fconst); 3 measured with a geometry that does not match the tail (so the numbers describe the wrong bytes); 4 refused:
the head's block size is not 128, so a block-128 decode would measure the wrong layout; 5 decoded, but no oracle (an
pack that DECLARES `tie_word_embeddings: false` and is given no `--ref`), so no correlation is reported.  A pack that
does not declare it is correlated against its embedding and labelled provisional, because "undeclared" is not "untied".
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


def head_decl(pack):
    """The head's MatMulNBits record as the graph declares it: `{name, K, N, block_size, bits}`, or None (engine#1740).

    The ops can sit in an `If` subgraph and carry a vendor name (`MatMulNBitsBf`, domain `com.ryzenai`), so this walks
    every subgraph and matches the op-type prefix.  A pack may declare different block sizes per op; the head's is what
    decides whether this tool's block-128 tile decoder applies to it.
    """
    try:
        import onnx
    except ImportError:
        return None
    path = os.path.join(pack, "model.onnx")
    if not os.path.exists(path):
        return None
    model = onnx.load(path, load_external_data=False)

    def walk(graph):
        for node in graph.node:
            yield node
            for attr in node.attribute:
                if attr.g.node:
                    yield from walk(attr.g)
                for sub in attr.graphs:
                    yield from walk(sub)

    for node in walk(model.graph):
        if node.op_type.startswith("MatMulNBits") and "lm_head" in node.name:
            a = {x.name: onnx.helper.get_attribute_value(x) for x in node.attribute}
            return {"name": node.name, "K": a.get("K"), "N": a.get("N"), "block_size": a.get("block_size"),
                    "bits": a.get("bits")}
    return None


def reference(path, dtype, hidden):
    """`--ref`: raw little-endian `[rows, H]` from row 0, as f32."""
    raw = np.fromfile(path, dtype={"bf16": "<u2", "f16": "<f2", "f32": "<f4"}[dtype])
    if raw.size % hidden:
        return None, "%s holds %d values, not a multiple of H %d" % (path, raw.size, hidden)
    vals = bf16_to_f32(raw) if dtype == "bf16" else raw.astype(np.float32)
    return vals.reshape(-1, hidden), None


def trailing_zeros(path):
    """How many zero bytes the file ends in (the end-anchored head offset assumes none)."""
    size = os.path.getsize(path)
    end = size
    with open(path, "rb") as f:
        while end > 0:
            start = max(0, end - (1 << 20))
            f.seek(start)
            nz = np.flatnonzero(np.frombuffer(f.read(end - start), np.uint8))
            if len(nz):
                return size - (start + int(nz[-1]) + 1)
            end = start
    return size


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


def measure(pack, tiles_wanted, ref_path=None, ref_dtype="bf16"):
    cfg = config(pack)
    vocab = int(cfg["vocab_size"])
    hidden = int(cfg["hidden_size"])
    # "declared untied" and "did not declare" are different states (engine#1747 review; npu/pack_import.cpp:2201 keeps
    # them apart the same way).  Absent key -- or no config.json at all, which is the genre-A case -- used to fall into
    # the untied branch, skip the correlation and exit 5 saying "an untied pack without --ref": an assertion the pack
    # never made, withholding a 1.0000 correlation its own embedding would have supplied.
    tie_declared = "tie_word_embeddings" in cfg
    tied = bool(cfg.get("tie_word_embeddings", False))
    decl = head_decl(pack)
    # engine#1740: a head declared at another block size has every subvolume offset elsewhere, so decoding it with the
    # block-128 tile reads scales out of code bytes (NaN / 1e38 on the 1.5B pack) -- a number about this tool, not the
    # pack.  Refused by name, the way the importer's own block gate refuses (npu/pack_import.cpp, engine#1503).
    if decl and decl.get("block_size") not in (None, K_KBLOCK):
        return None, 4, ("the head %s (K %s, N %s) is block_size %s along K; this tool decodes the block-%d tile only, "
                         "so a decode would measure the wrong layout (engine#1503 owns the block-32 decoder)"
                         % (decl["name"], decl["K"], decl["N"], decl["block_size"], K_KBLOCK))
    if ref_path:
        ref_all, why = reference(ref_path, ref_dtype, hidden)
        oracle = "--ref " + os.path.basename(ref_path)
    elif tied or not tie_declared:
        ref_all, why = embedding(pack, vocab, hidden)
        oracle = ("model.embed_tokens.weight (tied)" if tied else
                  "model.embed_tokens.weight (the pack does not declare tie_word_embeddings; the correlation is provisional)")
    else:
        ref_all, why, oracle = None, None, None     # DECLARED untied, no --ref: decode, but do not correlate against nothing
    if ref_path and ref_all is None:
        return None, 2, why
    if (tied or not tie_declared) and not ref_path and ref_all is None:
        return None, 2, why
    path = fconst(pack)
    if not path:
        return None, 2, "no dd_metastate_*Token*.fconst"
    size = os.path.getsize(path)
    tile = attn_tile_bytes(hidden)
    env = attn_payload_off(hidden)
    ntiles = vocab // K_TILE_ROWS
    # Two conditions, two sentences. They used to share one message, which blamed H when V was the problem:
    # `Hunyuan-0.5B-Instruct_rai_1.9.0_npu` has V 120818, not a multiple of the 64-row tile, and was refused with
    # "H 1024 is not a multiple of 128" -- H is; V is not, and V % 64 is a property of these packs, not a defect.
    if tile == 0:
        return None, 2, "H %d is not a multiple of %d, so one tile has no rows to divide" % (hidden, K_KBLOCK)
    if vocab % K_TILE_ROWS:
        return None, 2, ("V %d is not a multiple of the %d-row tile, so the head is not a whole number of tiles "
                         "(this is the pack's vocabulary, not its H)" % (vocab, K_TILE_ROWS))
    # WHERE the head is: the LAST `ntiles * tile` bytes. `env` on its own is only the tail's envelope -- the importer
    # reaches the head as `layers*stride + env`, which equals this subtraction whenever its size check passes
    # (engine#1693: at `env` the decode correlates at chance, at the real offset 1.0000).
    base = head_offset(size, ntiles, tile)
    zeros = trailing_zeros(path)
    rows = []
    status = 0 if base >= 0 else 3
    if base >= 0 and oracle is None:
        status = 5
    with open(path, "rb") as f:
        for t in range(min(tiles_wanted, ntiles)):
            if ref_all is not None and (t + 1) * K_TILE_ROWS > ref_all.shape[0]:
                break
            f.seek(base + t * tile)
            buf = f.read(tile)
            if len(buf) != tile:
                break
            w, bias = decode_tile(buf, hidden, K_TILE_ROWS)
            # `finite` at the decoder's own precision (f32): a scale read out of code bytes is a finite f64 near 1e38,
            # and the product only overflows once it is the f32 a consumer would store
            with np.errstate(all="ignore"):
                w32 = w.astype(np.float32)
            ok = np.isfinite(w32)
            with np.errstate(all="ignore"):
                w_std = float(w32[ok].astype(np.float64).std()) if ok.any() else float("nan")
            row = {"tile": t, "finite": float(ok.mean()), "w_std": w_std}
            if ref_all is not None:
                ref = ref_all[t * K_TILE_ROWS:(t + 1) * K_TILE_ROWS]
                flat, flatref = w.ravel().astype(np.float64), ref.ravel().astype(np.float64)
                with np.errstate(all="ignore"):
                    corr = float(np.corrcoef(flat, flatref)[0, 1]) if flat.std() and flatref.std() else float("nan")
                row.update({"corr": corr,
                            "rel_err": float(np.abs(flat - flatref).sum() / max(np.abs(flatref).sum(), 1e-12)),
                            "abs_ratio": float(np.abs(flat).sum() / max(np.abs(flatref).sum(), 1e-12))})
            rows.append(row)
    out = {
        "pack": os.path.basename(os.path.abspath(pack)),
        "vocab": vocab, "hidden": hidden, "tied": tied, "head": decl, "oracle": oracle,
        "geometry": {"env": env, "head_off": base, "tile_bytes": tile, "tiles": ntiles, "tail_bytes": env + ntiles * tile,
                     "fconst_bytes": size, "trailing_zero_bytes": zeros, "tail_fits": base >= 0},
        "tiles_measured": rows,
    }
    return out, status, None


def main(argv):
    ap = argparse.ArgumentParser(description="Measure a 1.9 pack's .fconst tail head against its oracle "
                                             "(engine#1693, #1740).")
    ap.add_argument("pack")
    ap.add_argument("--tiles", type=int, default=4, help="tiles to decode from the tail's start (default 4)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ref", help="the head's reference rows, raw little-endian [rows, H] from row 0 (untied packs)")
    ap.add_argument("--ref-dtype", choices=("bf16", "f16", "f32"), default="bf16")
    a = ap.parse_args(argv[1:])
    out, status, why = measure(a.pack, a.tiles, a.ref, a.ref_dtype)
    if out is None:
        print("q4nx_tail_head: %s %s: %s" % ("refused" if status == 4 else "cannot measure", a.pack, why),
              file=sys.stderr)
        return status
    g = out["geometry"]
    if a.json:
        print(json.dumps(out, indent=2))
        return status
    h = out["head"]
    print("%s: V %d H %d, %s, head %s" % (out["pack"], out["vocab"], out["hidden"],
                                          "tied" if out["tied"] else "UNTIED",
                                          "block %s" % h["block_size"] if h else "(not declared in the graph)"))
    print("  head: the last %d tiles x %d B of %d B, so at %d (env-only base %d is the pre-head payload)"
          % (g["tiles"], g["tile_bytes"], g["fconst_bytes"], g["head_off"], g["env"]))
    if g["trailing_zero_bytes"]:
        print("  the .fconst ends in %d zero bytes: the end-anchored offset assumes the head is its last bytes"
              % g["trailing_zero_bytes"])
    print("  oracle: %s" % (out["oracle"] or "none -- an untied head is not its embedding; pass --ref (exit 5)"))
    for r in out["tiles_measured"]:
        line = "  tile %4d  finite %.4f  w_std %.4f" % (r["tile"], r["finite"], r["w_std"])
        if "corr" in r:
            line += "  corr %+0.4f  rel_err %0.4f  |w|/|ref| %0.4f" % (r["corr"], r["rel_err"], r["abs_ratio"])
        print(line)
    if status == 3:
        print("  the head grid is larger than the .fconst: the numbers above describe other bytes", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
