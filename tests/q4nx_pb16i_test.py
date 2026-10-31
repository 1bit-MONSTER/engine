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
"""tools/q4nx_pb16i.py: the pb16i image contract, checked without a model or a device.

usage: tests/q4nx_pb16i_test.py        (needs numpy)

1. Every geometry the 27B uses produces exactly K*N/2 + (K/32)*N*2 bytes = tiles*4,608,
   and a geometry that is not whole tiles is refused.
2. The packer puts element (i,j) where npu/forward/include/q4nx_pb16i.h says it is
   (restated here from the header's own rule: tile = (c, ki, nt, r), code at
   ((kb*2+nb)*16+rr)*16+cc / 2, ms at 4096 + 2*((nb*8+g)*16+cc)), and the mapping is a
   bijection: each code byte is claimed twice, each ms entry 32 times, per tile.
3. A synthetic ternary tensor round-trips: codes and ms read back out of the image equal
   what went in, for several (K,N).
4. The side table is e_ref int16[N] little-endian followed by one mode byte, and the
   int32/int64 split follows the converter's own worst-case bound.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import q4nx_pb16i as Q  # noqa: E402

FAIL = []


def check(ok, what, K=0, N=0):
    if not ok:
        FAIL.append(f"{what} (K={K} N={N})")


def header_index(i, j, K, N):
    """Restated from q4nx_pb16i.h::index(), independently of the tool's own copy."""
    cw = N // (Q.COLS * Q.ROWS)
    c = j // (N // Q.COLS)
    jr = j % (N // Q.COLS)
    r = jr // cw
    jn = jr % cw
    nt, nb, cc = jn // Q.NW, (jn % Q.NW) // 16, jn % 16
    kb, rr = (i % Q.KC) // 16, (i % Q.KC) % 16
    tile = (((c * (K // Q.KC) + i // Q.KC) * (cw // Q.NW) + nt) * Q.ROWS + r)
    e = ((kb * 2 + nb) * 16 + rr) * 16 + cc
    return tile * Q.TILE + e // 2, bool(e & 1), tile * Q.TILE + Q.CODE_BYTES + \
        2 * ((nb * 8 + ((i % Q.KC) // 32)) * 16 + cc)


GEOMS = [(5120, 17408), (17408, 5120), (5120, 10240), (6144, 5120), (5120, 6144),
         (5120, 1024), (256, 1024)]


def test_geometry():
    for K, N in GEOMS:
        check(Q.fits(K, N), "fits() refused a whole-tile geometry", K, N)
        check(Q.w_bytes(K, N) == K * N // 2 + (K // 32) * N * 2, "size != codes + ms", K, N)
        check(Q.w_bytes(K, N) % Q.TILE == 0, "size is not whole tiles", K, N)
        check(Q.w_bytes(K, N) // Q.TILE == 8 * (K // 256) * ((N // 32) // 32) * 4,
              "tile count", K, N)
        # the tool's index and the header's reading of the same element must agree
        for i in range(0, K, max(1, K // 7)):
            for j in range(0, N, max(1, N // 7)):
                a, b = Q.index(i, j, K, N), header_index(i, j, K, N)
                check(a == b, "tool index != header index", K, N)
    for K, N in [(255, 1024), (256, 1000), (0, 1024), (256, 0)]:
        check(not Q.fits(K, N), "a non-whole-tile geometry must be refused", K, N)


def test_bijection():
    for K, N in [(5120, 1024), (256, 1024), (5120, 6144)]:
        code_hits = np.zeros(Q.CODE_BYTES, dtype=np.int32)
        ms_hits = np.zeros((Q.TILE - Q.CODE_BYTES) // 2, dtype=np.int32)   # 256 int16 entries
        cw, nt_n, kt = N // (Q.COLS * Q.ROWS), N // (Q.COLS * Q.ROWS * Q.NW), K // Q.KC
        for c in (0, Q.COLS - 1):
            for ki in (0, kt - 1):
                for nt in (0, nt_n - 1):
                    for r in (0, Q.ROWS - 1):
                        code_hits[:] = 0
                        ms_hits[:] = 0
                        col0 = c * (N // Q.COLS) + r * cw + nt * Q.NW
                        base = (((c * kt + ki) * nt_n + nt) * Q.ROWS + r) * Q.TILE
                        for kk in range(Q.KC):
                            for nn in range(Q.NW):
                                cb, hi, mb = Q.index(ki * Q.KC + kk, col0 + nn, K, N)
                                code_hits[cb - base] += 1
                                ms_hits[(mb - base - Q.CODE_BYTES) // 2] += 1
                        check((code_hits == 2).all(), "a code byte is not claimed twice", K, N)
                        check((ms_hits == 32).all(), "an ms entry is not claimed 32 times", K, N)


def test_roundtrip():
    rng = np.random.default_rng(7)
    for K, N in [(5120, 1024), (256, 1024), (5120, 6144), (6144, 5120)]:
        codes = rng.integers(-1, 2, size=(K, N), dtype=np.int8)
        e = rng.integers(-18, -15, size=(K // 128, N))
        m = rng.integers(1100, 1600, size=(K // 128, N))
        bits = np.ldexp(m.astype(np.float64), e).astype(np.float16).view(np.uint16)
        img, e_ref, mode = Q.pack(codes, bits, K, N)
        check(img.shape == (Q.w_bytes(K, N) // Q.TILE, Q.TILE), "image shape", K, N)
        bad = Q.verify_image(img, codes, bits, K, N, e_ref)
        check(bad == 0, f"{bad} image slots disagree with the source scales", K, N)
        # w reconstructed from the image == code * m * 2^e
        got = np.empty((K, N), dtype=np.float64)
        for i in range(0, K, max(1, K // 11)):
            for j in range(0, N, max(1, N // 11)):
                cb, hi, mb = Q.index(i, j, K, N)
                byte = int(img.reshape(-1)[cb])
                trit = (byte >> 4) if hi else (byte & 0x0F)
                trit = trit - 16 if trit >= 8 else trit
                ms = int(np.frombuffer(img.reshape(-1)[mb:mb + 2].tobytes(), dtype='<i2')[0])
                got[i, j] = np.ldexp(float(trit * ms), int(e_ref[j]))
        # float64 is exact here: |trit*ms| <= 32767 and the scale is a power of two
        per_group = np.repeat(m.astype(np.float64) * np.ldexp(1.0, e.astype(np.int64)), 128,
                              axis=0)                      # [K][N] the real weight scale
        want = codes.astype(np.float64) * per_group
        for i in range(0, K, max(1, K // 11)):
            for j in range(0, N, max(1, N // 11)):
                check(got[i, j] == want[i, j], "w reconstructed from the image != code*scale", K, N)
        del want
        # the side table
        ref = Q.side_table(e_ref, mode)
        check(len(ref) == 2 * N + 1, "side table size", K, N)
        check(ref[-1] == mode and mode in (0, 1), "mode byte", K, N)
        back = np.frombuffer(ref[:-1], dtype='<i2')
        check(np.array_equal(back, e_ref), "e_ref round-trip", K, N)


def test_ptq1_0():
    """The reader and packer against a real ternary GGUF, end to end.

    There is no synthetic encoder here on purpose: PTQ1_0's 128 trits per block are not a
    plain base-3 number (ggml stores them in staged 32/16/8 chunks, five trits per byte),
    so a hand-built block would test my own inverse, not ggml's layout.  Instead: pack a
    real tensor and compare the image BYTE FOR BYTE against a set whose device output was
    already gated bit-exact (the frozen pb16i sidecar set).  Skipped when the model is not
    on this machine, so CI stays green."""
    gguf = os.environ.get("Q4NX_PB16I_GGUF",
                          "/home/bcloud/models/bonsai2-27b/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
    ref = os.environ.get("Q4NX_PB16I_REF", "/home/bcloud/models/npu/bonsai2-pb16i-sidecars")
    if not os.path.exists(gguf) or not os.path.isdir(ref):
        print("  (reader test skipped: no ternary GGUF or reference image set on this machine)")
        return
    want = {}
    for name, dims, tt, raw in Q.gguf_tensors(gguf):
        if name in ("blk.0.attn_gate.weight", "blk.3.attn_k.weight"):
            want[name] = (dims, raw)
    for name, (dims, raw) in want.items():
        K, N = dims
        codes, bits = Q.read_ptq1_0(raw, K, N)
        check(codes.shape == (K, N), "reader shape", K, N)
        check(int(np.abs(codes).max()) <= 1, "reader returned a non-ternary code", K, N)
        img, e_ref, mode = Q.pack(codes, bits, K, N)
        stem = name[:-len(".weight")].replace(".", "_") + "_weight"
        for suffix, got in ((".pb16i", img.reshape(-1).tobytes()),
                            (".ref", Q.side_table(e_ref, mode))):
            path = os.path.join(ref, stem + suffix)
            if not os.path.exists(path):
                print(f"  (no reference {stem}{suffix}; skipped)")
                continue
            with open(path, "rb") as f:
                check(got == f.read(), f"{stem}{suffix} != the gated reference image", K, N)


def main():
    test_geometry()
    test_bijection()
    test_roundtrip()
    test_ptq1_0()
    if FAIL:
        for f in FAIL[:20]:
            print("FAIL", f)
        print(f"q4nx_pb16i_test: {len(FAIL)} failure(s)")
        return 1
    print("q4nx_pb16i_test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
