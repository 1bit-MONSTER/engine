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
"""Check a Q4NX model directory's chunk layout without dequantizing it.

usage: tools/check_q4nx_dir.py <model dir | model.q4nx> [--json out.json]
       tools/check_q4nx_dir.py --self-test

Reads the container exactly as npu/model.cpp does (8-byte header length, JSON tensor table,
then the data) and, for every tensor, says which of the five chunk kinds its shape names
(npu/q4nx.h: q4_1 5120 B, q4_1f 6144 B, Q4_K 4736 B, Q8 8704 B, t2 2176 B) and whether the
declared bytes are the tiles that kind needs. Unknown chunk sizes and short tensors are errors
(exit 1): a file this tool cannot account for is a file the engine's decoders must not be
pointed at.

q4_1f is the form `1bit npu-pack import` writes for an OGA pack (engine#1185): the same 32x256
tile with fp32 scale/zero, so the pack's own parameters survive bit for bit. The device kernels
stream 5,120-byte q4_1, so the forward converts such a directory at load
(npu/q4nx_device.cpp, into the Q4NX cache); the directory itself is valid as written.

This is the check a published FastFlowLM `*-NPU2` directory goes through before the engine is
asked to serve it (docs/npu.md, "Q4NX chunks"). It is metadata only: nothing is dequantized,
no numpy, no gguf, and it opens the file to read the header and the file size, never the data.
It therefore cannot tell the two Q8 code layouts apart - the container does not record which
one it holds - so it reports the Q8 tensors and says so, rather than guessing.

`--against <source.gguf>` closes that gap the only way it can be closed: it decodes the first
tile of the Q8 tensors **both** ways and correlates each against the tensor in the GGUF the
container was packed from, so the answer comes from the weights instead of from the container's
own round trip (a packer/decoder pair that agrees on the wrong layout scores well against
itself). That mode is opt-in and is the only one that needs numpy and gguf.

A model directory's `config.json` also lets this tool compare each quantized tensor's **tile count**
with the geometry the model's own dimensions imply, which catches a container whose weight grids were
padded by a converter to a different grid than this engine reads (engine#1097: the FastFlowLM converter
pads `intermediate_size` to a 512-column grid, 8960 -> 9216, while the reader derives the chunk size from
the unpadded grid, so the container loads and every request fails with "dequant failed"). Those are
warnings by default and failures under `--strict`; a directory without `config.json` skips the check.
"""
import argparse
import contextlib
import io
import json
import os
import struct
import sys
import tempfile

# npu/q4nx.h: Chunk::Q4_1 = 5120, Q4_1F = 6144 (the imported-container form, engine#1185),
# Q4_K = 4736, Q8 = 8704. t2 (2176) has its own contract test and is not a tensor dtype here.
CHUNKS = {5120: "q4_1", 6144: "q4_1f", 4736: "Q4_K", 8704: "Q8"}
ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1}
# The quantized tensors the engine decodes: int8 codes in 32x256 tiles (npu/q4nx.h).
QUANT = ("I8",)


def read_header(path):
    """(tensor table, data base offset, file size) - npu/model.cpp:48-73, header only."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            raise ValueError(f"{path}: truncated ({size} bytes)")
        (n,) = struct.unpack("<Q", head)
        if n > size - 8:
            raise ValueError(f"{path}: header length {n} past the end of the file ({size} bytes)")
        raw = f.read(n)
    if len(raw) != n:
        raise ValueError(f"{path}: truncated header ({len(raw)} of {n} bytes)")
    table = json.loads(raw)
    if not isinstance(table, dict):
        raise ValueError(f"{path}: header is not a tensor table")
    return table, 8 + n, size


# The two 8704-byte Q8 code orders in circulation (npu/q4nx.h): byte index of code (row, column).
# A is what npu/q4nx.cpp's q4nx_dequant_tile_q8 decodes and what tools/gguf_to_q4nx.py --dtype8
# writes; B is the Qwen3.6-35B container. Only a check against the source weights can say which
# one a third-party container holds.
Q8_LAYOUTS = {
    "A": lambda r, k: k * 32 + r,
    "B": lambda r, k: (r // 16) * 4096 + (k // 32) * 512 + (k % 32) * 16 + (r % 16),
}
Q8_TILE = 8704        # 512 B of bf16 per-32-column scale (8 groups x 32 rows) + 8192 int8 codes


def q8_tile_floats(tile, layout):
    """The 8192 values of one Q8 tile as 32 rows x 256 columns (npu/q4nx.cpp's tile)."""
    if len(tile) != Q8_TILE:
        raise ValueError(f"a Q8 tile is {Q8_TILE} B, got {len(tile)}")
    index = Q8_LAYOUTS[layout]
    rows = []
    for r in range(32):
        row = []
        for k in range(256):
            (s,) = struct.unpack_from("<H", tile, ((k // 32) * 32 + r) * 2)
            (d,) = struct.unpack("<f", struct.pack("<I", s << 16))       # bf16 -> f32
            (c,) = struct.unpack_from("<b", tile, 512 + index(r, k))
            row.append(d * c)
        rows.append(row)
    return rows


def corr(a, b):
    """Pearson correlation of two equal-length sequences."""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def rel_rms(got, want):
    num = sum((x - y) ** 2 for x, y in zip(got, want)) / len(want)
    den = sum(y * y for y in want) / len(want)
    return (num / den) ** 0.5 if den else 0.0


def prod(shape):
    out = 1
    for v in shape:
        out *= v
    return out


# The tile grid the engine's readers assume (npu/forward/include/q4nx_dequant.h), and the two
# padding grids a packer may have used (npu/forward/src/q4nx_pack.c: q4nx_pad_k 256, q4nx_pad_n 512).
TILE_ROWS, TILE_COLS = 32, 256
N_GRID = 512


def dims_from_config(cfg):
    """The model dimensions a tensor's tile grid follows, or None if config.json lacks them."""
    try:
        h = int(cfg["hidden_size"])
        hd = int(cfg.get("head_dim", 0)) or h // int(cfg["num_attention_heads"])
        qd = int(cfg["num_attention_heads"]) * hd
        kvd = int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])) * hd
        im = int(cfg["intermediate_size"])
        v = int(cfg["vocab_size"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return {"H": h, "QD": qd, "KVD": kvd, "IM": im, "V": v}


def expected_rc(name, d):
    """(rows, cols) this weight must hold, from the model's dims; None for a tensor we cannot place."""
    if name == "lm_head.weight":
        return d["V"], d["H"]
    tail = name.split(".", 3)[3] if name.startswith("model.layers.") and name.count(".") > 3 else ""
    return {
        "self_attn.q_proj.weight": (d["QD"], d["H"]),
        "self_attn.k_proj.weight": (d["KVD"], d["H"]),
        "self_attn.v_proj.weight": (d["KVD"], d["H"]),
        "self_attn.o_proj.weight": (d["H"], d["QD"]),
        "mlp.gate_proj.weight": (d["IM"], d["H"]),
        "mlp.up_proj.weight": (d["IM"], d["H"]),
        "mlp.down_proj.weight": (d["H"], d["IM"]),
    }.get(tail)


def geometry_warnings(table, d):
    """Warnings for quantized tensors whose tile count disagrees with the model's dimensions.

    A padded grid is not a chunk-layout error - the chunks are valid - but it is not the grid this
    engine reads, so the container loads and fails on the first request (engine#1097)."""
    if not d:
        return []
    out = []
    for name, t in table.items():
        if name == "__metadata__" or not isinstance(t, dict):
            continue
        if t.get("dtype") not in QUANT:
            continue
        shape = t.get("shape") or []
        if len(shape) < 2 or any(not isinstance(v, int) or v <= 0 for v in shape):
            continue
        rc = expected_rc(name, d)
        if rc is None:
            continue
        rows, cols = rc
        if rows % TILE_ROWS or cols % TILE_COLS:
            out.append(f"{name}: {rows}x{cols} is not on the {TILE_ROWS}x{TILE_COLS} tile grid")
            continue
        tiles = prod(shape[:-1])
        want = (rows // TILE_ROWS) * (cols // TILE_COLS)
        if tiles == want:
            continue
        ntr, ntc = rows // TILE_ROWS, cols // TILE_COLS
        if tiles % ntr == 0:
            pc = (tiles // ntr) * TILE_COLS
            how = f"columns padded {cols} -> {pc} (the {N_GRID}-column grid)"
        elif tiles % ntc == 0:
            pr = (tiles // ntc) * TILE_ROWS
            how = f"rows padded {rows} -> {pr} (the {N_GRID}-column grid)"
        else:
            how = "a grid the model's dimensions do not describe"
        out.append(f"{name}: holds {tiles} tiles, the model's dimensions imply {want} - {how}; "
                   f"this engine derives the chunk size from the unpadded grid, so the tensor "
                   f"fails to decode (engine#1097)")
    return out


def check(path, verbose=True):
    """(errors, summary) for one .q4nx file."""
    table, base, size = read_header(path)
    errors = []
    kinds = {}      # chunk name -> tensor count
    chunk_bytes = 0
    unscanned = 0
    q8 = []
    for name, t in table.items():
        if name == "__metadata__" or not isinstance(t, dict) or "data_offsets" not in t:
            continue
        shape = t.get("shape") or []
        dtype = t.get("dtype", "")
        off = t.get("data_offsets") or []
        if not shape or any(not isinstance(v, int) or v <= 0 for v in shape):
            errors.append(f"{name}: unusable shape {shape}")
            continue
        if len(off) != 2 or off[1] < off[0] or off[1] > size - base:
            errors.append(f"{name}: bad data_offsets {off} (the file has {size - base} data bytes)")
            continue
        declared = off[1] - off[0]
        last = shape[-1]
        if dtype in QUANT and last in CHUNKS:
            tiles = prod(shape[:-1])
            want = tiles * last
            chunk_bytes += declared
            kinds[CHUNKS[last]] = kinds.get(CHUNKS[last], 0) + 1
            if last == 8704:
                q8.append(name)
        elif dtype in QUANT:
            errors.append(f"{name}: int8 tensor whose last dimension {last} names no chunk kind "
                          f"({', '.join(f'{k} {v}' for k, v in CHUNKS.items())})")
            continue
        elif dtype in ITEMSIZE:
            want = prod(shape) * ITEMSIZE[dtype]
        else:
            unscanned += 1
            if verbose:
                print(f"  {name}: dtype {dtype!r} not counted ({declared} B)")
            continue
        if declared != want:
            errors.append(f"{name}: {dtype} {shape} declares {declared} B, the layout needs {want} B")
    if verbose:
        print(f"{path}: {sum(kinds.values())} quantized tensor(s) in {len(table)} table entries, "
              f"{chunk_bytes} chunk bytes, {size} B file")
        for k in ("q4_1", "q4_1f", "Q4_K", "Q8"):
            if k in kinds:
                print(f"  {k:5s} {CHUNKS_REV[k]:5d} B  {kinds[k]:4d} tensor(s)")
        if "q4_1f" in kinds:
            # The alternative is refusing an imported directory, which is a valid container: the
            # engine converts it for the device at load (npu/q4nx_device.cpp, engine#1185).
            print(f"  q4_1f: {kinds['q4_1f']} tensor(s) with fp32 scale/zero (an imported OGA "
                  f"container); the engine repacks them to the device's 5,120-byte q4_1 at load")
        if q8:
            # Our own converter can say which of the two 8704-B Q8 layouts it wrote: it packs
            # code (row, column) at column*32 + row, the one npu/q4nx.cpp decodes, and records
            # the --dtype8 pattern that selected the tensors.
            own = table.get("__metadata__", {}).get("converter") == "tools/gguf_to_q4nx.py"
            how = ("packed by tools/gguf_to_q4nx.py --dtype8, so they use the layout "
                   "npu/q4nx.cpp decodes (code at column*32 + row)" if own else
                   "two code layouts share the 8704-byte chunk and the container does not say "
                   "which (npu/q4nx.h) - do not dequantize these with the q4_1-style decoder "
                   "unless the model is verified")
            print(f"  Q8: {len(q8)} tensor(s) ({', '.join(q8[:3])}{', ...' if len(q8) > 3 else ''}): "
                  f"{how}")
        if unscanned:
            print(f"  {unscanned} tensor(s) of an uncounted dtype (not quantized weights)")
    return errors, {"kinds": kinds, "tensors": sum(kinds.values()), "q8": len(q8), "bytes": chunk_bytes}


CHUNKS_REV = {v: k for k, v in CHUNKS.items()}


def q8_ground_truth(gguf_path, container_name):
    """The source tensor's top-left 32x256 values, or None if the GGUF has no such tensor.

    Only this path needs numpy and gguf; the container side stays pure python.
    """
    import gguf
    import numpy as np
    from gguf.quants import dequantize
    from gguf_to_q4nx import hf_name          # the packer's own GGUF -> Q4NX name mapping
    reader = gguf.GGUFReader(gguf_path)
    for t in reader.tensors:
        try:
            name = hf_name(t.name)
        except (SystemExit, KeyError, ValueError):   # a GGUF entry with no Q4NX name: not ours
            continue
        if name != container_name:
            continue
        # ggml stores [rows, columns]; the tile is rows-major from the source's own order.
        flat = np.asarray(dequantize(t.data, t.tensor_type), dtype="float32").reshape(-1)
        cols = int(t.shape[0])
        return [[float(flat[r * cols + c]) for c in range(256)] for r in range(32)], t.name
    return None, None


def check_q8_layout(path, gguf_path, samples=3):
    """(errors, rows): which Q8 code order the container's tiles actually hold, versus the GGUF."""
    table, base, _ = read_header(path)
    q8 = [n for n, t in table.items()
          if isinstance(t, dict) and t.get("dtype") == "I8" and (t.get("shape") or [0])[-1] == Q8_TILE]
    if not q8:
        print(f"{path}: no Q8 tensors, nothing to check against the source weights")
        return [], []
    errors, rows = [], []
    for name in q8[:samples]:
        off = table[name]["data_offsets"][0]
        with open(path, "rb") as f:
            f.seek(base + off)
            tile = f.read(Q8_TILE)
        want, gguf_name = q8_ground_truth(gguf_path, name)
        if want is None:
            print(f"  {name}: not in {os.path.basename(gguf_path)}, skipped")
            continue
        flat_want = [v for row in want for v in row]
        scores = {}
        for layout in ("A", "B"):
            got = [v for row in q8_tile_floats(tile, layout) for v in row]
            scores[layout] = (corr(got, flat_want), rel_rms(got, flat_want))
        a, b = scores["A"], scores["B"]
        rows.append({"tensor": name, "gguf": gguf_name, "A": a, "B": b})
        print(f"  {name} vs {gguf_name}:")
        for layout in ("A", "B"):
            print(f"    layout {layout}: corr {scores[layout][0]:+.4f}  rel-rms {scores[layout][1]:.4f}")
        # A is the order npu/q4nx.cpp decodes; anything else is served as noise, silently.
        if not (a[0] >= 0.99 and a[1] <= 0.10 and a[0] > b[0]):
            errors.append(f"{name}: not the layout npu/q4nx.cpp decodes "
                          f"(A corr {a[0]:+.4f} rel-rms {a[1]:.4f}, B corr {b[0]:+.4f})")
    if not rows:
        # Nothing was compared: passing here would be the same silent success this mode exists to
        # remove, e.g. the wrong GGUF for this container.
        errors.append(f"no Q8 tensor of {path} matched a tensor in {os.path.basename(gguf_path)}: "
                      f"nothing was verified")
    return errors, rows


def targets(arg):
    """The .q4nx files a directory names: model.q4nx plus any other (vision_weight.q4nx)."""
    if os.path.isdir(arg):
        found = sorted(os.path.join(arg, f) for f in os.listdir(arg) if f.endswith(".q4nx"))
        if not found:
            raise SystemExit(f"{arg}: no .q4nx file in this directory")
        return found
    return [arg]


def write_q4nx(path, tensors):
    """A minimal container: the header npu/model.cpp reads, then the tensors' bytes."""
    table, off, blob = {}, 0, b""
    for name, shape, dtype, filler in tensors:
        n = prod(shape) * ITEMSIZE[dtype] if dtype != "I8" else prod(shape[:-1]) * shape[-1]
        table[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        off += n
        blob += bytes([filler]) * n
    header = json.dumps(table).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + blob)


def self_test():
    """A good container passes; an unknown chunk size and a short tensor each fail."""
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good")
        os.mkdir(good)
        write_q4nx(os.path.join(good, "model.q4nx"), [
            ("proj.weight", [2, 4736], "I8", 0x11),       # 2 Q4_K tiles
            ("lm_head.weight", [3, 8704], "I8", 0x22),    # 3 Q8 chunks
            ("mlp.down_proj.weight", [1, 2, 5120], "I8", 0x33),  # 2 q4_1 tiles
            ("attn.q_proj.weight", [1, 6144], "I8", 0x66),       # an imported q4_1f tile
            ("norm.weight", [5120], "BF16", 0x44),
            ("ssm_a", [48], "F32", 0x55),
        ])
        errors, summary = check(os.path.join(good, "model.q4nx"), verbose=False)
        if errors:
            bad.append(f"good container reported {errors}")
        if summary["tensors"] != 4 or summary["q8"] != 1:
            bad.append(f"good container counted {summary}")
        if sorted(summary["kinds"]) != ["Q4_K", "Q8", "q4_1", "q4_1f"]:
            bad.append(f"good container kinds {summary['kinds']}")
        if summary["kinds"].get("q4_1f") != 1:
            bad.append(f"the imported fp32 chunk was not counted: {summary['kinds']}")

        # an int8 tensor whose last dimension names no chunk kind: the engine cannot decode it
        unknown = os.path.join(tmp, "unknown")
        os.mkdir(unknown)
        write_q4nx(os.path.join(unknown, "model.q4nx"), [("proj.weight", [4, 8192], "I8", 0x66)])
        errors, _ = check(os.path.join(unknown, "model.q4nx"), verbose=False)
        if not any("names no chunk kind" in e for e in errors):
            bad.append(f"unknown chunk size not refused: {errors}")

        # a tensor shorter than its layout needs
        short = os.path.join(tmp, "short")
        os.mkdir(short)
        write_q4nx(os.path.join(short, "model.q4nx"), [("proj.weight", [2, 4736], "I8", 0x77)])
        p = os.path.join(short, "model.q4nx")
        with open(p, "r+b") as f:
            f.seek(-100, os.SEEK_END)
            f.truncate()
        errors, _ = check(p, verbose=False)
        if not any("bad data_offsets" in e for e in errors):
            bad.append(f"short tensor not refused: {errors}")

        # a container with no Q8 tiles: --against compared nothing, so the run must not
        # print the success line for a code-order check that did not happen.
        noq8 = os.path.join(tmp, "noq8")
        os.mkdir(noq8)
        write_q4nx(os.path.join(noq8, "model.q4nx"), [("proj.weight", [2, 4736], "I8", 0xAB)])
        buf = io.StringIO()
        argv = sys.argv
        sys.argv = ["check_q4nx_dir.py", noq8, "--against", os.path.join(tmp, "unused.gguf")]
        try:
            with contextlib.redirect_stdout(buf):
                rc_noq8 = main()
        finally:
            sys.argv = argv
        said = buf.getvalue()
        if rc_noq8 != 0:
            bad.append(f"no-Q8 container with --against returned {rc_noq8}: {said}")
        if "hold the code order" in said:
            bad.append(f"no-Q8 container claimed a code order was checked: {said}")
        if "no Q8 tile was decoded" not in said:
            bad.append(f"no-Q8 container did not say it compared nothing: {said}")

        # the directory case: every .q4nx in it is checked
        two = os.path.join(tmp, "two")
        os.mkdir(two)
        write_q4nx(os.path.join(two, "model.q4nx"), [("norm.weight", [5120], "BF16", 0x88)])
        write_q4nx(os.path.join(two, "vision_weight.q4nx"), [("patch.weight", [1, 5120], "I8", 0x99)])
        if len(targets(two)) != 2:
            bad.append(f"directory targets {targets(two)}")

        # the two Q8 code orders: build a tile whose codes were written in A's order and prove
        # A decodes it back and B does not, so --against's verdict cannot come from the decoder
        # agreeing with itself (no gguf here: this pins the byte order, the GGUF supplies truth).
        def bf16(x):
            """Round a float through bf16 (the tile's scale type) and back."""
            (u,) = struct.unpack("<I", struct.pack("<f", x))
            return struct.unpack("<f", struct.pack("<I", (u >> 16) << 16))[0]

        def bf16_bits(x):
            return struct.unpack("<I", struct.pack("<f", x))[0] >> 16

        scales = [bf16(1.0 + g + r * 1e-3) for g in range(8) for r in range(32)]
        code = {(r, k): ((r * 7 + k * 13) % 251) - 125 for r in range(32) for k in range(256)}
        payload = bytearray(Q8_TILE)
        for i, s in enumerate(scales):
            struct.pack_into("<H", payload, i * 2, bf16_bits(s))
        for (r, k), c in code.items():
            struct.pack_into("<b", payload, 512 + Q8_LAYOUTS["A"](r, k), c)
        want = [scales[(k // 32) * 32 + r] * code[(r, k)] for r in range(32) for k in range(256)]
        flat_a = [v for row in q8_tile_floats(bytes(payload), "A") for v in row]
        flat_b = [v for row in q8_tile_floats(bytes(payload), "B") for v in row]
        if corr(flat_a, want) < 0.9999 or rel_rms(flat_a, want) > 1e-3:
            bad.append(f"A-order tile does not decode back to itself: corr {corr(flat_a, want):.6f}")
        if corr(flat_b, want) > 0.5:
            bad.append(f"B scores {corr(flat_b, want):.4f} on an A-order tile, so the check is blind")

    # Geometry: a tensor on the model's own grid is quiet, one a converter padded to the 512 grid
    # warns (engine#1097 - the container loads, then every request fails with "dequant failed").
    dims = dims_from_config({"hidden_size": 1536, "intermediate_size": 8960, "num_attention_heads": 12,
                             "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 151936})
    if not dims:
        bad.append("dims_from_config returned nothing for a complete config")
    else:
        ok_table = {"model.layers.0.mlp.down_proj.weight": {"dtype": "I8", "shape": [48, 35, 5120]}}
        padded = {"model.layers.0.mlp.down_proj.weight": {"dtype": "I8", "shape": [1728, 5120]},
                  "model.layers.0.mlp.gate_proj.weight": {"dtype": "I8", "shape": [1728, 5120]}}
        if geometry_warnings(ok_table, dims):
            bad.append(f"a 48x35 grid is the model's own grid but warned: {geometry_warnings(ok_table, dims)}")
        gw = geometry_warnings(padded, dims)
        if len(gw) != 2:
            bad.append(f"a 512-padded MLP grid produced {len(gw)} warning(s), expected 2")
        elif "columns padded 8960 -> 9216" not in gw[0] or "rows padded 8960 -> 9216" not in gw[1]:
            bad.append(f"the padding explanation is not what the grids say: {gw}")
        if geometry_warnings({"norm.weight": {"dtype": "F32", "shape": [1536]}}, dims):
            bad.append("a non-quantized tensor produced a geometry warning")
    if bad:
        print("self-test FAILED:")
        for b in bad:
            print("  " + b)
        return 1
    print("self-test passed: 4 chunk kinds counted (q4_1, q4_1f, Q4_K, Q8), unknown chunk size "
          "refused, short tensor refused, directory scans every .q4nx, a Q8 tile written in A's order "
          "decodes as A and not as B, a no-Q8 container claims no code order, and a 512-padded grid "
          "is named as padded while the model's own grid is quiet")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="a Q4NX model directory (or one model.q4nx)")
    ap.add_argument("--json", help="write the per-file summary here")
    ap.add_argument("--self-test", action="store_true", help="synthetic containers: pass/fail cases")
    ap.add_argument("--against", metavar="SOURCE.gguf",
                    help="decode every Q8 tile both ways and correlate against the GGUF the "
                         "container was packed from (needs numpy and gguf)")
    ap.add_argument("--q8-samples", type=int, default=3, metavar="N",
                    help="how many Q8 tensors --against decodes (default 3)")
    ap.add_argument("--strict", action="store_true",
                    help="treat a geometry warning (a padded weight grid this engine cannot read) "
                         "as a failure")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.path:
        ap.error("a model directory or .q4nx file is required (or --self-test)")
    out, rc = {}, 0
    compared = 0        # Q8 tensors actually decoded and correlated, across all files
    warnings = 0        # geometry warnings (padded weight grids) across all files
    for path in targets(args.path):
        errors, summary = check(path)
        out[path] = summary
        cfg_path = os.path.join(os.path.dirname(path) or ".", "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except (OSError, ValueError) as e:
                cfg = None
                print(f"WARN {cfg_path}: unreadable ({e}), geometry not checked")
            geoms = geometry_warnings(read_header(path)[0], dims_from_config(cfg) if cfg else None)
            summary["geometry_warnings"] = len(geoms)
            warnings += len(geoms)
            for g in geoms:
                print(f"WARN {g}")
                if args.strict:
                    errors.append(g)
        if args.against:
            print(f"{path}: Q8 code layout against {os.path.basename(args.against)}")
            q8_errors, rows = check_q8_layout(path, args.against, args.q8_samples)
            summary["q8_layout"] = rows
            compared += len(rows)
            errors += q8_errors
        for e in errors:
            print(f"FAIL {e}")
        rc = rc or (1 if errors else 0)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if rc:
        print("chunk layout NOT verified")
    elif args.against and compared:
        print("chunk layout verified, and the Q8 tiles hold the code order npu/q4nx.cpp decodes")
    elif args.against:
        # No Q8 tensor was decoded, so nothing was compared: say that instead of
        # printing the success line for a check that did not run.
        print("chunk layout verified; no Q8 tile was decoded, so no code order was checked "
              "against the source weights")
    else:
        print("chunk layout verified: every quantized tensor's shape names a known chunk kind")
    if warnings and not rc:
        print(f"but {warnings} quantization tensor(s) sit on a padded grid this engine reads at a "
              f"different stride: the container will load and fail on the first request "
              f"(engine#1097) - `--strict` fails on this")
    return rc


if __name__ == "__main__":
    sys.exit(main())
