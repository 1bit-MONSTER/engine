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
"""Regenerate tests/golden/oga_pack_tiles.{bin,json}: q4_1 tiles of a fixed synthetic
(codes, scale, zero) tensor, written the way the container has always written them.

The tiling below is copied from tools/gguf_to_q4nx.py (`bf16_bits`, `from_bf16`,
`tiles`), which is the implementation that produced every Q4NX model the NPU routes
run today. tests/npu_pack_import_test.cpp builds the same synthetic tensor in C++ and
compares its tiles byte for byte, so the pack importer's writer is checked against the
established container layout and not only against its own decoder.

usage: tools/oga_pack_tiles_golden.py [--check]
"""
import json
import struct
import sys

import numpy as np

ROWS, COLS, TILE = 32, 256, 5120
N, K = 64, 384          # 1.5 column tiles: exercises the zero padding as well
KP = ((K + COLS - 1) // COLS) * COLS


def bf16_bits(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def tiles(codes, scale, zero):
    """q4_1 parts -> [tiles, 5120] bytes, tiles row-major over the 32 x 256 grid."""
    N, K = codes.shape
    RT, KT = N // ROWS, K // COLS

    def plane(v):
        return bf16_bits(v).reshape(RT, ROWS, KT, 8).transpose(0, 2, 3, 1).reshape(RT * KT, 256)

    c = codes.astype(np.uint8).reshape(RT, 2, 8, 2, KT, 8, 32)   # rt, h, r2, parity, kt, g, j
    packed = c[:, :, :, 0] | (c[:, :, :, 1] << 4)
    packed = packed.transpose(0, 3, 1, 4, 5, 2).reshape(RT * KT, 4096)  # rt, kt, h, g, j, r2
    return np.concatenate([plane(scale).view(np.uint8), plane(zero).view(np.uint8), packed], axis=1)


def synthetic():
    """The same tensor the C++ test builds: codes (n*7 + k*13) % 16, and parameters that
    differ per (row, 32-column group), with zero padding past K."""
    n = np.arange(N)[:, None]
    k = np.arange(KP)[None, :]
    codes16 = ((n * 7 + k * 13) % 16).astype(np.uint8)
    codes16[:, K:] = 0                                     # padding columns
    g = np.arange(KP // 32)[None, :]
    r = np.arange(N)[:, None]
    scale = np.where(g * 32 < K, 0.001 * (g % 7 + 1) * (1 + 0.0 * r), 0.0).astype(np.float32)
    zero = np.where(g * 32 < K, -0.05 * (g % 3) - 0.01 * (r % 5), 0.0).astype(np.float32)
    return codes16, scale, zero


def main():
    codes, scale, zero = synthetic()
    t = tiles(codes, scale, zero)
    out = "tests/golden/oga_pack_tiles"
    meta = {
        "what": "q4_1 tiles of a synthetic (codes, scale, zero) tensor, as tools/gguf_to_q4nx.py writes them",
        "rows_32": ROWS,
        "cols_256": COLS,
        "tile_bytes": TILE,
        "N": N,
        "K": K,
        "padded_K": KP,
        "tiles": int(t.shape[0]),
        "codes": "(n*7 + k*13) % 16 for k < K, 0 in the padding",
        "scale": "0.001 * (g % 7 + 1) for the 32-column group g, 0 in the padding",
        "zero": "-0.05 * (g % 3) - 0.01 * (n % 5), 0 in the padding",
        "generator": "tools/oga_pack_tiles_golden.py",
    }
    if "--check" in sys.argv:
        have = open(out + ".bin", "rb").read()
        print("in sync" if have == t.tobytes() else "OUT OF SYNC: re-run without --check")
        return 0 if have == t.tobytes() else 1
    open(out + ".bin", "wb").write(t.tobytes())
    info = dict(meta)
    info["sha256"] = __import__("hashlib").sha256(t.tobytes()).hexdigest()
    open(out + ".json", "w").write(json.dumps(info, indent=2) + "\n")
    print(f"wrote {out}.bin ({t.nbytes} B) and {out}.json  sha256 {info['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
