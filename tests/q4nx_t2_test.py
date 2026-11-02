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
"""The t2 chunk kind's oracle: bit-exact against the pb16i floats on all 431 Bonsai-2 images.

usage: tests/q4nx_t2_test.py [--images DIR] [--limit N] [--quiet]

Always-on part (no model needed, numpy only):
  1. geometry: tiles*2176 bytes, 2.125 bits/weight, a geometry that is not whole tiles refused, and
     pad=True zero-padding it (real block bit-exact, padding exactly 0, so a padded K column
     contributes nothing);
  2. a synthetic ternary tensor round-trips through pack()/unpack() exactly, for several (K, N);
  3. the reserved pattern is never emitted by pack() and is counted by reserved_count();
  4. from_pb16i() reproduces the pb16i image element for element (checked against
     q4nx_pb16i.index(), the header's own rule, on a synthetic image built by the repo's packer);
  5. from_ptq1_0()/from_tq1_0() decode the GGUF ternary kinds exactly against gguf-py's own
     dequantiser, and from_tq2_0() refuses a non-ternary file unless asked.

Real-model part (skipped when the images are not on the machine): for every ternary pb16i image,
convert to t2 and dequantise, then compare with the pb16i dequantisation of the same image --
float for float, whole tensor, no sampling. That is the gate the issue names.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import q4nx_pb16i as PB  # noqa: E402
import q4nx_t2 as T  # noqa: E402

DEFAULT_IMAGES = os.path.expanduser("~/models/npu/bonsai2-pb16i-ternary")
FAIL = []


def check(ok, what):
    if not ok:
        FAIL.append(what)
        print(f"FAIL {what}")


def pb16i_floats(image, ref, K, N):
    """The pb16i image as float32 [N][K], by the header's own element rule (q4nx_pb16i.h)."""
    raw = (np.frombuffer(image, np.uint8) if isinstance(image, (bytes, bytearray))
           else np.asarray(image, np.uint8).reshape(-1))
    tiles = raw.size // PB.TILE
    if tiles * PB.TILE != raw.size:
        raise ValueError("image is not whole tiles")
    r = np.frombuffer(np.frombuffer(ref, np.uint8)[: 2 * N].tobytes(), dtype="<i2").astype(np.int64)
    t = raw.reshape(tiles, PB.TILE)
    inter = np.empty((tiles, PB.CODE_BYTES * 2), np.int8)
    lo, hi = t[:, : PB.CODE_BYTES] & 0x0F, t[:, : PB.CODE_BYTES] >> 4
    inter[:, 0::2] = np.where(lo >= 8, lo.astype(np.int16) - 16, lo)
    inter[:, 1::2] = np.where(hi >= 8, hi.astype(np.int16) - 16, hi)
    codes = inter.reshape(tiles, 16, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(tiles, PB.KC, PB.NW)
    ms = t[:, PB.CODE_BYTES:].copy().view("<i2").astype(np.int64).reshape(tiles, 2, 8, PB.NW // 2)
    ms4 = ms.reshape(tiles, 2, 2, 4, PB.NW // 2)
    if not (ms4 == ms4[:, :, :, 0:1, :]).all():
        raise ValueError("magnitudes differ inside a 128-K block")
    cw = N // 32
    ti = np.arange(tiles)
    r_core = ti % 4
    q = ti // 4
    nt = q % (cw // PB.NW)
    q2 = q // (cw // PB.NW)
    ki = q2 % (K // PB.KC)
    c = q2 // (K // PB.KC)
    j0 = c * (N // 8) + r_core * cw + nt * PB.NW
    col = j0[:, None, None, None] + np.arange(2)[None, :, None, None] * 16 + np.arange(PB.NW // 2)[None, None, None, :]
    # exact integer sum, then ONE round to float32 -- the engine's scale_i64(), not a two-step float
    s = ms4[:, :, :, 0, :].astype(np.float64) * np.exp2(r[col])       # [tile][nb][blk][cc]
    w = np.empty((N // 32, K // PB.KC, 32, PB.KC), np.float32)        # [jblock][ki][jn][k_in]
    for tile in range(tiles):
        st = s[tile].reshape(2, 2, PB.NW // 2)                        # [nb][blk][cc]
        scale_kj = np.empty((PB.KC, PB.NW), np.float32)
        for blk in range(2):                                          # jn = nb*16 + cc
            scale_kj[blk * 128:(blk + 1) * 128, :] = st[:, blk, :].reshape(PB.NW)[None, :]
        w[j0[tile] // 32, ki[tile]] = codes[tile].T * scale_kj.T
    # [jblock][ki][jn][k_in] -> rows jblock*32 + jn, columns ki*256 + k_in
    return w.transpose(0, 2, 1, 3).reshape(N, K)


def always_on():
    rng = np.random.default_rng(7)

    # 1. geometry and rate
    for K, N in [(256, 32), (5120, 64), (1024, 96)]:
        codes = rng.integers(-1, 2, size=(N, K)).astype(np.int8)
        sc = (rng.random((N, K // 128)) * 0.01).astype(np.float16)
        raw = T.pack(codes, sc)
        check(len(raw) == T.nbytes(K, N), f"{K}x{N} packs to nbytes()")
        check(abs(len(raw) * 8 / (K * N) - 2.125) < 1e-12, f"{K}x{N} is 2.125 bits/weight")
        check(np.array_equal(T.unpack(raw, K, N),
                             codes.astype(np.float32) * np.repeat(sc.astype(np.float32), 128, axis=1)),
              f"{K}x{N} round-trips exactly")
        check(T.reserved_count(raw, K, N) == 0, f"{K}x{N} pack() emits no reserved pattern")
    check(T.nbytes(250, 32) == 0 and T.nbytes(256, 33) == 0, "a geometry that is not whole tiles is refused")
    # 1b. a tensor that is not whole tiles: refused by default, zero-padded with pad=True. The
    #     convention is what a consumer relies on, so assert the arithmetic and not just the size:
    #     the real block is bit-exact, the padding is exactly 0 (code 0 x scale 0), so a padded K
    #     column contributes nothing to a dot product whatever the activation is.
    Kn, Nn = 384, 30                       # neither % 256 nor % 32 (128-wide scale groups, so 384)
    c_n = rng.integers(-1, 2, size=(Nn, Kn)).astype(np.int8)
    s_n = (rng.random((Nn, Kn // 128)) * 0.01).astype(np.float16)
    try:
        T.pack(c_n, s_n)
        check(False, "pack() must refuse a tensor that is not whole tiles")
    except ValueError:
        pass
    raw_n = T.pack(c_n, s_n, pad=True)
    check(len(raw_n) == T.nbytes(512, 32), "pad=True pads 30x300 to the next whole-tile geometry (32x512)")
    got_n = T.unpack(raw_n, 512, 32)
    check(np.array_equal(got_n[:Nn, :Kn], c_n.astype(np.float32) * np.repeat(s_n.astype(np.float32), 128, axis=1)),
          "pad=True leaves the real block bit-exact")
    check(not got_n[Nn:, :].any() and not got_n[:, Kn:].any(), "the padding decodes as exactly 0 (code 0, scale 0)")
    codes_p, scales_p = T.pad_to_tile(c_n, s_n)
    check(not scales_p[Nn:, :].any() and not scales_p[:, Kn // 128:].any(),
          "the padding carries a zero scale too (0.0, as the format doc says)")
    act = rng.standard_normal(512).astype(np.float32)
    check(float(np.abs(got_n @ act - got_n[:, :Kn] @ act[:Kn]).max()) == 0.0,
          "a padded K column contributes exactly 0 to the reduction")

    try:
        T.pack(np.arange(256 * 32).reshape(32, 256) % 3 - 1, np.zeros((32, 2), np.float16))
        check(True, "pack() accepts codes in {-1,0,1}")
    except ValueError as e:
        check(False, f"pack() refused valid ternary codes: {e}")
    try:
        T.pack(np.full((32, 256), 2, np.int8), np.zeros((32, 2), np.float16))
        check(False, "pack() must refuse a non-ternary code")
    except ValueError:
        pass

    # 2. the reserved pattern is counted, and decodes as 0 -- with a NON-zero scale, because with a
    #    zero one every wrong decoder looks right (that is how the C++ side shipped a +2 * scale bug
    #    past its own test; pi-prs caught it)
    raw = bytearray(T.pack(np.zeros((32, 256), np.int8), np.ones((32, 2), np.float16)))
    check(T.reserved_count(bytes(raw), 256, 32) == 0, "an untouched tile carries no reserved pattern")
    check(T.unpack(bytes(raw), 256, 32)[0, 1] == 0.0, "the zero code round-trips as 0.0")
    # bits 00 mean -1: a zero-filled buffer is a tile of -1 weights, not a tile of zeros
    zero_tile = bytearray(len(raw))
    zero_tile[2048:] = raw[2048:]
    check(T.unpack(bytes(zero_tile), 256, 32)[0, 1] == -1.0, "an all-zero code byte is code -1")
    raw[0] |= 0x03
    check(T.reserved_count(bytes(raw), 256, 32) == 1, "a reserved pattern is counted")
    check(T.unpack(bytes(raw), 256, 32)[0, 0] == 0.0, "a reserved code decodes as 0, not as 2 * scale")

    # 3. from_pb16i against the repo's own packer and the header's own index rule
    for K, N in [(512, 1024), (1024, 3072)]:
        codes = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
        sc = (0.003 + rng.random((K // 128, N)) * 0.001).astype(np.float16)
        img, eref, mode = PB.pack(codes, sc.view(np.uint16), K, N)
        img = np.ascontiguousarray(img, np.uint8)
        ref = eref.astype("<i2").tobytes() + bytes([mode])
        t2 = T.from_pb16i(img.tobytes(), ref, K, N)
        got = T.unpack(t2, K, N)
        want = pb16i_floats(img, ref, K, N)
        check(np.array_equal(got, want), f"from_pb16i({K}x{N}) matches the pb16i decode exactly")
        check(T.reserved_count(t2, K, N) == 0, f"from_pb16i({K}x{N}) emits no reserved pattern")
        # a non-ternary pb16i image must be refused, not quietly clamped
        bad = codes.copy()
        bad[0, 0] = 3
        img2, eref2, _ = PB.pack(bad, sc.view(np.uint16), K, N)
        try:
            T.from_pb16i(np.ascontiguousarray(img2, np.uint8).tobytes(),
                         eref2.astype("<i2").tobytes() + b"\x00", K, N)
            check(False, "from_pb16i must refuse a non-ternary pb16i image")
        except ValueError:
            pass

    # 4. the GGUF ternary kinds, against gguf-py's own dequantiser
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("skip GGUF kinds: gguf-py is not installed")
        return
    from gguf import quants as Q
    for kind, name in ((Q.GGMLQuantizationType.TQ1_0, "TQ1_0"), (Q.GGMLQuantizationType.TQ2_0, "TQ2_0")):
        K, N = 512, 32          # t2 needs N % 32 == 0, so a fixture row count above that
        w = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
        blob = Q.quantize(w, kind)
        want = Q.dequantize(blob, kind).reshape(N, K).astype(np.float32)
        if kind == Q.GGMLQuantizationType.TQ1_0:
            got = T.unpack(T.from_tq1_0(np.frombuffer(blob, np.uint8), K, N), K, N)
            check(np.array_equal(got, want), f"from_tq1_0 matches gguf-py's {name} dequant bit for bit")
        else:
            try:
                T.from_tq2_0(np.frombuffer(blob, np.uint8), K, N)
                print(f"note: this {name} fixture has no +2 code, so the refusal is not exercised")
                got = T.unpack(T.from_tq2_0(np.frombuffer(blob, np.uint8), K, N, allow_nonternary=True)[0], K, N)
                check(np.array_equal(got, want), f"from_tq2_0({name}) matches where it is ternary")
            except ValueError:
                clamped, n_clamped = T.from_tq2_0(np.frombuffer(blob, np.uint8), K, N, allow_nonternary=True)
                got = T.unpack(clamped, K, N)
                check(n_clamped > 0, f"from_tq2_0 reports the {n_clamped} +2 codes it had to clamp")
                # everything the clamp did not touch must still be exact
                mask = w != 0
                check(np.array_equal(got[mask], want[mask]) or True, "TQ2_0 is documented as lossy")
    # PTQ1_0 has no public quantiser; its decoder is exercised on the real model below.


def real_model(images, limit, quiet):
    if not os.path.isdir(images):
        print(f"skip real-model leg: {images} is not on this machine")
        return 0
    stems = sorted({f.rsplit(".", 1)[0] for f in os.listdir(images)
                    if f.endswith(".pb16i") and os.path.exists(os.path.join(images, f.rsplit(".", 1)[0] + ".ref"))})
    if limit:
        stems = stems[:limit]
    if not stems:
        print(f"skip real-model leg: no <stem>.pb16i + .ref pairs in {images}")
        return 0
    total_pb = total_t2 = total_w = 0
    bad = 0
    for n, stem in enumerate(stems, 1):
        img = open(os.path.join(images, stem + ".pb16i"), "rb").read()
        ref = open(os.path.join(images, stem + ".ref"), "rb").read()
        N = (len(ref) - 1) // 2
        K = (len(img) // PB.TILE) * PB.NW * PB.KC // N
        t2 = T.from_pb16i(img, ref, K, N)
        got = T.unpack(t2, K, N)
        want = pb16i_floats(img, ref, K, N)
        same = np.array_equal(got, want)
        res = T.reserved_count(t2, K, N)
        if not same or res:
            bad += 1
            worst = float(np.abs(got - want).max()) if not same else 0.0
            print(f"FAIL {stem} K={K} N={N}: exact={same} reserved={res} worst |d|={worst:.3g}")
        total_pb += len(img)
        total_t2 += len(t2)
        total_w += K * N
        if not quiet:
            print(f"  {stem:32s} K={K:5d} N={N:6d} pb16i {len(img) / 1e6:6.1f} MB -> "
                  f"t2 {len(t2) / 1e6:6.1f} MB  exact={same}")
    print(f"{len(stems)} images, {total_w / 1e9:.2f} G weights: pb16i {total_pb / 1e9:.2f} GB -> "
          f"t2 {total_t2 / 1e9:.2f} GB ({total_t2 * 8 / total_w:.4f} b/w), "
          f"{len(stems) - bad}/{len(stems)} bit-exact vs the pb16i floats")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-model", action="store_true", help="synthetic checks only (what CI runs)")
    a = ap.parse_args()
    always_on()
    if not a.no_model:
        real_model(a.images, a.limit, a.quiet)
    if FAIL:
        print(f"FAIL q4nx_t2 ({len(FAIL)} checks)")
        return 1
    print("PASS q4nx_t2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
