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
"""The acceptance gate for a t2 GEMV target: the pinned accumulation order, bit for bit (engine#804).

usage: tests/npu_t2_target_gate.py [--tensor FILE.t2 --dump FILE.bin --n N --k K] [--quiet]

A t2 target -- the Hexagon one first (#804), anything else that decodes the chunk kind later -- has
to produce the same y a CPU oracle produces, and "the same" means bit-identical, not close.  The
order the 128-K group sums enter the accumulator is part of the container's contract (docs/npu.md,
"The t2 chunk kind"): per 128-K group an exact int32 sum, then y += f32(group) * f32(scale), left to
right.  A target that accumulates in float64, that multiplies by the row's scales once at the end,
or that adds the groups in another order is wrong even at 1e-7 -- because the layer above it was
measured against the pinned order, not against an approximation of it.

So this gate is strict in both directions: it accepts an implementation that follows the order, and
it refuses four plausible near-misses.  A gate that cannot fail proves nothing, which is why the
always-on half is mostly the refusal cases.

Always-on (numpy only, no model, no device):
  1. the oracle accepts its own output, for several (K, N) including a single tile;
  2. it refuses a one-bit change in y, a truncated dump, the same tensor accumulated in float64
     (the "close enough" version) and one accumulated with the groups in reverse (identical
     group sums, different order) -- the last two only at the production width, because with a
     handful of groups those two implementations produce identical bits (checked, and stated);
  3. the reserved code (3) contributes 0 and never 2*scale, so a target that expands it as a third
     ternary value fails on a tensor that carries one.

Real-tensor half (skipped unless --tensor is given): reads a t2 file plus a target's dump of y and
prints the bit-exact count, the SNR and the first mismatch.  This is the same check the target's own
stage gate runs; here it is written once, from the container's side, so the target is compared with
the contract instead of with its own reference.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import q4nx_t2 as T  # noqa: E402

GROUP = T.GROUP  # 128 columns (K) per scale


def oracle(codes, sb, x):
    """y[n] = sum over groups, left to right, of f32(exact int32 group sum) * f32(scale).

    codes: int8 [N][K] in {-1,0,1}, sb: fp16 scale bits uint16 [N][K//GROUP], x: int8/int32 [K].
    """
    N = codes.shape[0]
    scale = sb.view(np.float16).astype(np.float32)
    y = np.zeros(N, np.float32)
    for g in range(codes.shape[1] // GROUP):
        c = codes[:, g * GROUP:(g + 1) * GROUP].astype(np.int32)
        s = c @ x[g * GROUP:(g + 1) * GROUP].astype(np.int32)  # exact: |s| <= 128*127
        prod = np.multiply(s.astype(np.float32), scale[:, g], dtype=np.float32)
        y = np.add(y, prod, dtype=np.float32)
    return y


def snr_db(ref, got):
    """Signal-to-noise, or inf when the two are bit-identical."""
    ref = ref.astype(np.float32)
    d = ref - got.astype(np.float32)
    noise = float(np.sum(d.astype(np.float64) ** 2))
    if noise == 0.0:
        return float("inf")
    return 10.0 * np.log10(float(np.sum(ref.astype(np.float64) ** 2)) / noise)


def compare(ref, got):
    """(bit-exact count, first mismatch index or -1)."""
    a = ref.view(np.uint32)
    b = np.asarray(got, np.float32).view(np.uint32)
    if a.shape != b.shape:
        return 0, 0
    bad = np.nonzero(a != b)[0]
    return int(a.size - bad.size), (int(bad[0]) if bad.size else -1)


class Gate:
    def __init__(self, quiet=False):
        self.quiet = quiet
        self.passed = 0
        self.failed = []

    def check(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            if not self.quiet:
                print(f"  ok   {name} {detail}".rstrip())
        else:
            self.failed.append(name)
            print(f"  FAIL {name} {detail}".rstrip())

    def accepts(self, name, ref, got):
        n, first = compare(ref, got)
        self.check(name, first < 0, f"{n}/{ref.size} bit-exact" + (f" first@{first}" if first >= 0 else ""))

    def refuses(self, name, ref, got, why):
        n, first = compare(ref, got)
        same_len = np.asarray(got, np.float32).shape == ref.shape
        detail = f"{first}/{ref.size} differ" if not same_len else f"snr {snr_db(ref, got):.1f} dB"
        self.check(name, first >= 0, f"refused as intended ({detail}, {why})")


def synthetic(k, n, seed):
    """A tensor through the container's own packer, its codes read back with the container's own rule."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
    # 0.5 + m/2048 is exact in fp16, so the packer never rounds a test scale
    sb = (np.uint16(0x3800) | rng.integers(0, 2048, size=(n, k // GROUP)).astype(np.uint16))
    raw = T.pack(codes, sb.view(np.float16))
    c, s = T.unpack_bits(raw, k, n)
    x = rng.integers(-127, 128, size=k).astype(np.int32)
    return c, s, x


def near_misses(codes, sb, x):
    """The four wrong-but-close implementations a target could plausibly ship."""
    N, K = codes.shape
    scale = sb.view(np.float16).astype(np.float32)
    gs = [codes[:, g * GROUP:(g + 1) * GROUP].astype(np.int32)
          @ x[g * GROUP:(g + 1) * GROUP].astype(np.int32) for g in range(K // GROUP)]
    # float64 accumulation, cast once at the end
    y64 = np.zeros(N, np.float64)
    for g in range(K // GROUP):
        y64 += gs[g].astype(np.float64) * scale[:, g].astype(np.float64)
    # the pinned order, but added right to left
    y_rev = np.zeros(N, np.float32)
    for g in reversed(range(K // GROUP)):
        prod = np.multiply(gs[g].astype(np.float32), scale[:, g], dtype=np.float32)
        y_rev = np.add(y_rev, prod, dtype=np.float32)
    # the whole row scaled once: sum of exact ints * the row's mean scale
    y_one = (np.sum(np.stack(gs), axis=0).astype(np.float32)
             * np.float32(np.mean(scale, axis=1)))
    return {"float64 accumulation": y64.astype(np.float32),
            "groups added in reverse": y_rev,
            "scale applied once per row": y_one}


def reserved_case(seed):
    """A tensor whose first tile carries the reserved code (3): it must contribute 0."""
    k, n = 512, 32
    rt, kt = n // T.ROWS, k // T.COLS
    rng = np.random.default_rng(seed)
    codes = rng.integers(-1, 2, size=(n, k)).astype(np.int8)
    sb = np.uint16(0x3800) | rng.integers(0, 2048, size=(n, k // GROUP)).astype(np.uint16)
    raw = bytearray(T.pack(codes, sb.view(np.float16)))
    raw[0] |= 0xC0  # byte 0 holds rows 0..3 of K=0..3 at bits 6-7 = reserved
    codes2, sb2 = T.unpack_bits(bytes(raw), k, n)
    x = rng.integers(-127, 128, size=k).astype(np.int32)
    return k, n, codes2, sb2, x, bytes(raw)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", help="a .t2 file to check a target's dump against")
    ap.add_argument("--dump", help="the target's y, float32 little-endian, N values, no header")
    ap.add_argument("--n", type=int, help="rows, for --tensor")
    ap.add_argument("--k", type=int, help="columns, for --tensor")
    ap.add_argument("--x", help="the x vector the target consumed, int8; generated when omitted")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    g = Gate(a.quiet)
    print("t2 target gate: pinned accumulation order")

    # k=5120 is the production width (40 groups): that is where float64 accumulation and a reversed
    # group order stop being bit-identical to the pinned order, so the two near-misses are only
    # actually refused there.  The narrow shapes stay for the tile-edge cases.
    for k, n in ((256, 32), (512, 64), (5120, 64)):
        codes, sb, x = synthetic(k, n, seed=k + n)
        ref = oracle(codes, sb, x)
        g.accepts(f"accepts its own y ({n}x{k})", ref, ref)
        one_ulp = ref.copy()
        one_ulp[3] = np.nextafter(one_ulp[3], np.float32(np.inf))
        g.refuses(f"refuses a one-bit change ({n}x{k})", ref, one_ulp, "y[3] moved by 1 ULP")
        g.refuses(f"refuses a short dump ({n}x{k})", ref, ref[:-1], "one value missing")
        nm = near_misses(codes, sb, x)
        if k >= 5120:
            for why, wrong in nm.items():
                g.refuses(f"refuses {why} ({n}x{k})", ref, wrong, why)
        else:
            # below a few groups these two near-misses are not separable: a handful of f32 adds of
            # exact products cannot round, so float64 and a reversed order give identical bits.  That
            # is a property of the arithmetic, not a hole in the gate -- a target only ever pointed at
            # short K cannot be told apart by this check, which is why the production width is tested.
            for why in ("float64 accumulation", "groups added in reverse"):
                g.accepts(f"indistinguishable below 4 groups: {why} ({n}x{k})", ref, nm[why])

    rk, rn, codes, sb, x, raw = reserved_case(11)
    g.check("reserved code counted once", T.reserved_count(raw, rk, rn) > 0, "the tensor carries a reserved code")
    ref = oracle(codes, sb, x)
    g.check("reserved code decodes as 0", bool(np.all(np.isin(codes, (-1, 0, 1)))), "no +-2 anywhere")
    # the decode a target ships if it forgets the reserved rule: code 3 read as +2
    bits = (np.frombuffer(raw, np.uint8).reshape(-1, T.TILE)[:, :T.CODE_BYTES]
            .reshape(-1, T.ROWS, 64))
    bad = np.empty(bits.shape + (4,), np.int8)
    for s_ in range(4):
        b = (bits >> (2 * s_)) & 3
        bad[..., s_] = np.where(b == 3, np.int8(2), b.astype(np.int8) - 1)
    rt, kt = rn // T.ROWS, rk // T.COLS
    bad = bad.reshape(rt, kt, T.ROWS, T.COLS).transpose(0, 2, 1, 3).reshape(rn, rk)
    g.refuses("refuses reserved read as +2", ref, oracle(bad, sb, x), "a target that expands 3 as +2")

    if a.tensor:
        if not (a.dump and a.n and a.k):
            raise SystemExit("--tensor needs --dump, --n and --k")
        raw = open(a.tensor, "rb").read()
        codes, sb = T.unpack_bits(raw, a.k, a.n)
        x = (np.fromfile(a.x, np.int8).astype(np.int32) if a.x
             else np.random.default_rng(0).integers(-127, 128, size=a.k).astype(np.int32))
        got = np.fromfile(a.dump, np.float32)
        ref = oracle(codes, sb, x)
        print(f"{a.tensor}: {a.n}x{a.k}, {T.reserved_count(raw, a.k, a.n)} reserved codes, "
              f"{got.size} values dumped")
        n, first = compare(ref, got)
        print(f"  {n}/{ref.size} bit-exact, snr {snr_db(ref, got):.2f} dB"
              + ("" if first < 0 else f", first mismatch y[{first}] {ref[first]!r} vs {got[first]!r}"))
        g.check("real tensor bit-exact", first < 0, "")

    print(f"{g.passed} passed, {len(g.failed)} failed")
    return 0 if not g.failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
