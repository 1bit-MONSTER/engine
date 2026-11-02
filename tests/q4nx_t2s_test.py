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
"""q4nx_t2s_test.py -- the contract of the 2-bit ternary stream (tools/q4nx_t2s.py).

No model, no device, no toolchain: a synthetic tensor shaped like a Bonsai-2 projection (ternary trits with
one fp16 scale per 128-K block per column) goes through the engine's own pb16i packer, and the stream derived
from it must

  1. be 2,176 B per k=256 x n=32 tile, i.e. 0.47222x the pb16i bytes, 2.125 bits/weight;
  2. expand back to **the pb16i image's own int4 code bytes**, block for block, on every tile;
  3. decode back to the image's codes and multipliers, with the 128-K compaction losing no value;
  4. be reproduced byte for byte by `stream()` (GGUF-side input) and `from_pb16i_image()` (image-side input);
  5. refuse, with a named error, an image whose four 32-K multipliers inside a 128-K block are not equal;
  6. refuse a stream carrying the reserved 2-bit pattern.

Run: tests/q4nx_t2s_test.py   (ctest `q4nx_t2s`)
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import q4nx_pb16i                                    # noqa: E402
import q4nx_t2s as t2s                               # noqa: E402

K, N = 512, 1024                                     # whole phase-B tiles, 4 k-chunks x 32 columns
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not ok:
        fails.append(name)


def fixture(seed=7):
    """A ternary weight with per-128-K fp16 scales, drawn the way a PTQ1_0 tensor is shaped."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(-1, 2, size=(K, N), dtype=np.int8)
    codes[rng.random((K, N)) < 0.34] = 0               # a third of a ternary column is zero
    # A PTQ1_0 column's scales are one magnitude per 128-K block with a narrow exponent spread, and the
    # fp16 mantissa is 11 bits: d = m * 2^e with m in [1024, 2047] is exactly what the format stores.
    m = rng.integers(1024, 2048, size=(K // 128, N)).astype(np.int64)
    e = (-14 + rng.integers(0, 3, size=(K // 128, N))).astype(np.int64)   # <= 2 exponents apart per column
    d = m.astype(np.float64) * np.ldexp(1.0, e)
    dbits = d.astype(np.float16).view(np.uint16)
    if not np.array_equal(d, dbits.view(np.float16).astype(np.float64)):
        raise SystemExit("fixture: a synthetic scale is not fp16-exact")
    return codes, dbits


def main():
    codes, dbits = fixture()
    img, e_ref, mode = q4nx_pb16i.pack(codes, dbits, K, N)
    tiles = img.shape[0]
    st = t2s.from_pb16i_image(img, K, N)

    # 1. geometry and size
    check("tile count", tiles == (K // q4nx_pb16i.KC) * (N // q4nx_pb16i.NW), f"{tiles}")
    check("stream size", st.size == t2s.w_bytes(K, N), f"{st.size} B, {t2s.w_bytes(K, N)} expected")
    t_bytes, pb_bytes, ratio = t2s.size_report(K, N)
    check("ratio is 0.47222x pb16i", abs(ratio - 0.4722222) < 1e-6, f"{ratio:.6f}")
    check("2.125 bits/weight", abs(t2s.bits_per_weight() - 2.125) < 1e-12, f"{t2s.bits_per_weight():.4f}")

    # 2. the expansion IS the pb16i image's int4 code bytes (pb16i orders the 128-B blocks k-major,
    #    t2s n-major: this is the one documented permutation, and nothing else may differ)
    have = t2s.expand(st, K, N)                                          # [tiles][4096], pb16i order
    check("expand matches the pb16i code area byte for byte",
          np.array_equal(have, img[:, :q4nx_pb16i.CODE_BYTES]),
          f"{have.size} B, {int(np.count_nonzero(have != img[:, :q4nx_pb16i.CODE_BYTES]))} differ")

    # 3. the decode is the image's, and the 128-K compaction kept the multiplier
    dcodes, dms = t2s.decode(st, K, N)
    icodes, ims = t2s._split(img)
    check("decode returns the image's codes", np.array_equal(dcodes, icodes))
    check("decode returns the image's multipliers", np.array_equal(dms, ims.reshape(-1, 2, 2, 4, 16)[:, :, :, 0, :]))

    # 4. both entry points agree byte for byte
    st2, e_ref2, mode2 = t2s.stream(codes, dbits, K, N)
    check("stream() == from_pb16i_image()", np.array_equal(st, st2))
    check("the side table is the pb16i one", np.array_equal(e_ref, e_ref2) and mode == mode2)

    # 5. an image whose 32-K multipliers differ inside a 128-K block is refused, not rounded
    bad = img.copy()
    flat = np.ascontiguousarray(bad[:, q4nx_pb16i.CODE_BYTES:]).view("<i2").reshape(tiles, 256).copy()
    flat[0, 16] = np.int16(flat[0, 16] + 1)              # second 32-K group of the first 128-K block, col 0
    bad[:, q4nx_pb16i.CODE_BYTES:] = flat.view(np.uint8).reshape(tiles, 512)
    try:
        t2s.from_pb16i_image(bad, K, N)
        check("a non-shared 128-K multiplier is refused", False, "no exception")
    except t2s.NotRepresentable as ex:
        check("a non-shared 128-K multiplier is refused", "128-K" in str(ex), str(ex)[:60] + "...")

    # 6. a reserved code (3) is refused by the expander and the decoder
    cor = bytearray(st.tobytes())
    cor[0] |= 0x03                                        # code 0 of the first wire byte := 3
    for name, fn in (("expand", t2s.expand), ("decode", t2s.decode)):
        try:
            fn(bytes(cor), K, N)
            check(f"reserved code refused by {name}", False, "no exception")
        except ValueError as ex:
            check(f"reserved code refused by {name}", "reserved" in str(ex))

    print(f"\n{len(fails)} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
