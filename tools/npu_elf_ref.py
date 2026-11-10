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
"""The host reference half of the per-op kernel check (TNPU-ALL W3 gate: "bit-exact against the host reference").

A per-op family is four int8 GEMMs (QKV/O/GU/D) at two M tiers. The gate wants the device result compared against a host
reference on 3 seeds plus an edge case; this produces that reference and the inputs, so the device half only has to run the
ELF and diff two files.

What is deliberate here:

  * the inputs are written to disk and the SEED IS PRINTED, so a failing run can be reproduced from the line rather than
    from someone's memory of the command;
  * the reference is computed TWICE — once by dequantising B with the column scale and multiplying in fp64, once by
    accumulating the int32 dot product and applying one scale at the end — because the folded form is what a kernel
    actually does, and the two agreeing is the proof that the fold is legal at this shape;
  * the edge case is not "a random seed that happens to be awkward": all-zero columns (scale floor), all-saturating
    columns (codes at +/-127), and a column whose scale would divide by zero, because those are where absmax/round paths
    differ from a re-derived one.

Usage:
  npu_elf_ref.py --K 3072 --N 5120 --M 16 --seed 1 --out DIR        # one case
  npu_elf_ref.py --self-check                                       # the three edge cases, exit 0 only if all agree

Outputs, all in DIR: a.i8 (M x K, row-major), b.i8 (K x N, row-major, packed by the container's b8 layout elsewhere),
scales.f32 (N per-column scales), c.i32 (M x N int32 accumulator), c.f32 (M x N dequantised result).
"""
import argparse
import os
import struct
import sys

try:
    import numpy as np
except ImportError:                                            # the box has numpy; say so plainly if it ever does not
    sys.exit("npu_elf_ref: numpy is required")


def prng_bits(n, seed):
    """n values drawn from xorshift64*, in [0,1) -- the seed is printed so a case is reproducible anywhere."""
    mask = (1 << 64) - 1
    x = seed if seed else 1
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        x ^= x >> 12
        x ^= (x << 25) & mask
        x ^= x >> 27
        out[i] = ((x * 2685821657736338717) & mask) / float(1 << 64)
    return out


def quantise(weights, axis):
    """int8 codes + per-(check) scale, the way a container holds them: absmax/127 with a 1.0 floor on an all-zero slice.

    axis=0 -> a per-COLUMN scale (B, the weight matrix: `amax` is taken along K, one value per output column)
    axis=1 -> a per-ROW scale (A, the activations: one value per token)

    Those two lines were swapped until #1805's audit: `axis=1` reduces ALONG axis 1 and broadcasts as `(-1, 1)`, i.e. one scale per ROW, and
    `run_case()` agrees with the code (`a_w, axis=1` = per token, `b_w, axis=0` = per column). The self-check now asserts the convention so the
    docstring and the code cannot drift again.

    `codes = rint(weights / scale)` rounds half to even (numpy's default). The packer's mode is not stated here; if it differs on exact .5
    boundaries the codes differ there and nowhere else, which is the edge the saturating case is built to reach.
    """
    amax = np.max(np.abs(weights), axis=axis).astype(np.float64)
    amax[amax < 1e-12] = 1.0                       # the container's floor: an all-zero slice quantises to zeros at scale 1
    scale = amax / 127.0
    shape = (1, -1) if axis == 0 else (-1, 1)
    codes = np.rint(weights / np.broadcast_to(scale.reshape(shape), weights.shape))
    return np.clip(codes, -127, 127).astype(np.int8), scale


def edge_weights(rows, cols, kind, rng):
    """Edge cases applied to the WEIGHT matrix, because that is where the column scales live (review round on #1805):
    a zero column is what floors a scale, and saturating weights are what puts codes at +/-127."""
    w = (rng.reshape(rows, cols) - 0.5) * 2.0
    third = max(1, cols // 3)
    if kind == "zero":
        w[:, :third] = 0.0                          # columns whose absmax is 0 -> scale floors at 1.0, codes all zero
    elif kind == "full":
        w[:, :third] = 1.0                          # one column fully saturating at +127
        w[:, third:2 * third] = -1.0
    elif kind == "floor":
        w[:] = 0.0                                  # every column floors
    else:
        raise SystemExit(f"unknown edge kind {kind}")
    return w


def reference(a_codes, a_scale, b_codes, b_scale):
    """(int64 exact, int32 accumulation, folded float, dequantise-then-multiply).

    Two independent things are being checked here, and the review round on #1805 is why both exist:
      * the ACCUMULATOR: int64 arithmetic is exact, and int32 must agree with it element for element -- that is a check
        that can fail, unlike (a@b)*s versus a@(b*s), which are the same float expression;
      * the FOLD: applying the row and column scales after the integer dot must equal dequantising both operands first.
    """
    a64 = a_codes.astype(np.int64) @ b_codes.astype(np.int64)
    a32 = a_codes.astype(np.int32) @ b_codes.astype(np.int32)
    folded = a64.astype(np.float64) * a_scale[:, None] * b_scale[None, :]
    deq = (a_codes.astype(np.float64) * a_scale[:, None]) @ (b_codes.astype(np.float64) * b_scale[None, :])
    return a64, a32, folded, deq


def write_case(out, a, b, a_scale, b_scale, acc, folded):
    os.makedirs(out, exist_ok=True)
    a.tofile(os.path.join(out, "a.i8"))
    b.tofile(os.path.join(out, "b.i8"))
    a_scale.astype(np.float32).tofile(os.path.join(out, "a_scales.f32"))
    b_scale.astype(np.float32).tofile(os.path.join(out, "scales.f32"))
    acc.astype(np.int32).tofile(os.path.join(out, "c.i32"))
    folded.astype(np.float32).tofile(os.path.join(out, "c.f32"))


def run_case(K, N, M, seed, kind, out):
    rng_a = prng_bits(M * K, seed)
    rng_b = prng_bits(K * N, seed + 1)
    a_w = (rng_a.reshape(M, K) - 0.5) * 2.0
    b_w = edge_weights(K, N, kind, rng_b) if kind else (rng_b.reshape(K, N) - 0.5) * 2.0
    a_codes, a_scale = quantise(a_w, axis=1)        # per row (per token)
    b_codes, b_scale = quantise(b_w, axis=0)        # per column
    a64, a32, folded, deq = reference(a_codes, a_scale, b_codes, b_scale)
    rel = float(np.max(np.abs(folded - deq)) / max(1e-12, float(np.max(np.abs(deq)))))
    props = {
        "zero_b_cols": int(np.sum(np.all(b_codes == 0, axis=0))),
        # a column FLOORS only when its weights are all zero -- absmax exactly 1.0 also gives scale 1/127, so counting
        # scales would confuse "floored" with "saturating fixture" (caught by running this after the first rewrite)
        "floored_cols": int(np.sum(np.all(b_w == 0.0, axis=0))),
        "max_abs_code": int(np.max(np.abs(b_codes.astype(np.int32)))),
        "int32_matches_int64": bool(np.array_equal(a32, a64.astype(np.int32))),
        "int64_fits_int32": bool(np.max(np.abs(a64)) <= 2**31 - 1),
    }
    if out:
        write_case(out, a_codes, b_codes, a_scale, b_scale, a32, folded)
    return a64, a32, folded, deq, rel, props


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=1536)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--edge", choices=("zero", "full", "floor"), help="edge case instead of a seeded random matrix")
    ap.add_argument("--out", help="directory to write a.i8/b.i8/scales.f32/c.i32/c.f32 into")
    ap.add_argument("--self-check", action="store_true", help="run the three edge cases and the seeded case, exit 0 iff the reference agrees with itself")
    a = ap.parse_args()

    if a.self_check:
        bad = 0
        for kind in (None, "zero", "full", "floor"):
            a64, a32, folded, deq, rel, p = run_case(a.K, a.N, a.M, a.seed, kind, None)
            ok = rel <= 1e-6 and p["int32_matches_int64"] and p["int64_fits_int32"]
            # each edge case must REACH the property it names, or the arm tests nothing (review round on #1805)
            if kind == "zero":
                ok = ok and p["zero_b_cols"] > 0 and p["floored_cols"] > 0
            if kind == "floor":
                ok = ok and p["floored_cols"] == a.N
            if kind == "full":
                ok = ok and p["max_abs_code"] == 127
            print(f"  {'ok  ' if ok else 'FAIL'} K={a.K} N={a.N} M={a.M} edge={kind or 'seeded'} "
                  f"rel={rel:.1e} acc32==acc64={p['int32_matches_int64']} zero_cols={p['zero_b_cols']} "
                  f"floored={p['floored_cols']} maxcode={p['max_abs_code']}")
            bad += 0 if ok else 1
        # engine#1805 audit: the docstring's axes were swapped for as long as this tool existed, so assert the convention too -- a per-column
        # scale comes from reducing along axis 0, and a 3x4 must give 4 column scales and 3 row scales. Re-swapping either line fails here.
        _w = np.ones((3, 4))
        _col, _col_s = quantise(_w, axis=0)
        _row, _row_s = quantise(_w, axis=1)
        _conv = _col_s.shape == (4,) and _row_s.shape == (3,)
        print(f"  {'ok  ' if _conv else 'FAIL'} axis convention: axis=0 -> {_col_s.shape[0]} column scales, "
              f"axis=1 -> {_row_s.shape[0]} row scales (3x4 matrix)")
        bad += 0 if _conv else 1
        print("npu_elf_ref self-check: " + ("ok" if not bad else f"{bad} FAILED"))
        return 0 if not bad else 1

    a64, a32, folded, deq, rel, p = run_case(a.K, a.N, a.M, a.seed, a.edge, a.out)
    print(f"K={a.K} N={a.N} M={a.M} seed={a.seed} edge={a.edge or '-'} rel={rel:.3e} "
          f"acc={int(a64.min())}..{int(a64.max())} zero_cols={p['zero_b_cols']} floored={p['floored_cols']} "
          f"maxcode={p['max_abs_code']} acc32==acc64={p['int32_matches_int64']}" + (f" out={a.out}" if a.out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
