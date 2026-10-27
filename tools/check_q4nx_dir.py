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
then the data) and, for every tensor, says which of the three chunk kinds its shape names
(npu/q4nx.h: q4_1 5120 B, Q4_K 4736 B, Q8 8704 B) and whether the declared bytes are the
tiles that kind needs. Unknown chunk sizes and short tensors are errors (exit 1): a file this
tool cannot account for is a file the engine's decoders must not be pointed at.

This is the check a published FastFlowLM `*-NPU2` directory goes through before the engine is
asked to serve it (docs/npu.md, "Q4NX chunks"). It is metadata only: nothing is dequantized,
no numpy, no gguf, and it opens the file to read the header and the file size, never the data.
It therefore cannot tell the two Q8 code layouts apart - the container does not record which
one it holds - so it reports the Q8 tensors and says so, rather than guessing.
"""
import argparse
import json
import os
import struct
import sys
import tempfile

# npu/q4nx.h: Chunk::Q4_1 = 5120, Q4_K = 4736, Q8 = 8704
CHUNKS = {5120: "q4_1", 4736: "Q4_K", 8704: "Q8"}
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


def prod(shape):
    out = 1
    for v in shape:
        out *= v
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
        for k in ("q4_1", "Q4_K", "Q8"):
            if k in kinds:
                print(f"  {k:5s} {CHUNKS_REV[k]:5d} B  {kinds[k]:4d} tensor(s)")
        if q8:
            print(f"  Q8: {len(q8)} tensor(s) ({', '.join(q8[:3])}{', ...' if len(q8) > 3 else ''}): "
                  "two code layouts share the 8704-byte chunk and the container does not say which "
                  "(npu/q4nx.h) - do not dequantize these with the q4_1-style decoder unless the "
                  "model is verified")
        if unscanned:
            print(f"  {unscanned} tensor(s) of an uncounted dtype (not quantized weights)")
    return errors, {"kinds": kinds, "tensors": sum(kinds.values()), "q8": len(q8), "bytes": chunk_bytes}


CHUNKS_REV = {v: k for k, v in CHUNKS.items()}


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
            ("norm.weight", [5120], "BF16", 0x44),
            ("ssm_a", [48], "F32", 0x55),
        ])
        errors, summary = check(os.path.join(good, "model.q4nx"), verbose=False)
        if errors:
            bad.append(f"good container reported {errors}")
        if summary["tensors"] != 3 or summary["q8"] != 1:
            bad.append(f"good container counted {summary}")
        if sorted(summary["kinds"]) != ["Q4_K", "Q8", "q4_1"]:
            bad.append(f"good container kinds {summary['kinds']}")

        # an int8 tensor whose last dimension names no chunk kind: the engine cannot decode it
        unknown = os.path.join(tmp, "unknown")
        os.mkdir(unknown)
        write_q4nx(os.path.join(unknown, "model.q4nx"), [("proj.weight", [4, 6144], "I8", 0x66)])
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

        # the directory case: every .q4nx in it is checked
        two = os.path.join(tmp, "two")
        os.mkdir(two)
        write_q4nx(os.path.join(two, "model.q4nx"), [("norm.weight", [5120], "BF16", 0x88)])
        write_q4nx(os.path.join(two, "vision_weight.q4nx"), [("patch.weight", [1, 5120], "I8", 0x99)])
        if len(targets(two)) != 2:
            bad.append(f"directory targets {targets(two)}")
    if bad:
        print("self-test FAILED:")
        for b in bad:
            print("  " + b)
        return 1
    print("self-test passed: 3 chunk kinds counted, unknown chunk size refused, short tensor refused, "
          "directory scans every .q4nx")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="a Q4NX model directory (or one model.q4nx)")
    ap.add_argument("--json", help="write the per-file summary here")
    ap.add_argument("--self-test", action="store_true", help="synthetic containers: pass/fail cases")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.path:
        ap.error("a model directory or .q4nx file is required (or --self-test)")
    out, rc = {}, 0
    for path in targets(args.path):
        errors, summary = check(path)
        out[path] = summary
        for e in errors:
            print(f"FAIL {e}")
        rc = rc or (1 if errors else 0)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
    if rc:
        print("chunk layout NOT verified")
    else:
        print("chunk layout verified: every quantized tensor's shape names a known chunk kind")
    return rc


if __name__ == "__main__":
    sys.exit(main())
