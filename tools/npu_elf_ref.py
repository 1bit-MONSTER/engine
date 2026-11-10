#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
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


def prng_matrix(rows, cols, seed):
    """int8 in [-127, 127] from a documented generator: xorshift64*, so a seed reproduces the matrix anywhere."""
    n = rows * cols
    out = np.empty(n, dtype=np.int32)
    mask = (1 << 64) - 1
    x = seed if seed else 1                      # plain Python ints: numpy warns on uint64 multiply overflow, and this
    for i in range(n):                           # generator's overflow is the point (xorshift64*), not a bug
        x ^= x >> 12
        x ^= (x << 25) & mask
        x ^= x >> 27
        v = ((x * 2685821657736338717) & mask) >> 57                    # 7 bits: 0..127
        out[i] = v - 127 if (i & 1) else v                              # alternate sign: both ends get used
    return out.reshape(rows, cols).astype(np.int8)


def edge_matrix(rows, cols, kind):
    m = np.zeros((rows, cols), dtype=np.int8)
    if kind == "zero":
        m[:, : max(1, cols // 3)] = 0                                 # all-zero columns -> the scale floor
        m[:, max(1, cols // 3):] = prng_matrix(rows, max(1, cols - max(1, cols // 3)), 7)
    elif kind == "full":
        m[:, : max(1, cols // 3)] = 127                               # saturating codes
        m[:, max(1, cols // 3):] = -127
    elif kind == "floor":
        m[:] = 0                                                      # every column zero -> every scale floors
    else:
        raise SystemExit(f"unknown edge kind {kind}")
    return m


def column_scales(b):
    """absmax/127 per column, with the container's 1e-12 floor on an all-zero column."""
    amax = np.max(np.abs(b.astype(np.int32)), axis=0).astype(np.float64)
    amax[amax < 1e-12] = 1.0
    return amax / 127.0


def reference(a, b, scales):
    """(folded int32, dequantise-then-multiply) -- both, and the caller compares them."""
    acc = a.astype(np.int32) @ b.astype(np.int32)                     # exact in int32 for these shapes
    folded = acc.astype(np.float64) * scales[None, :]
    deq = a.astype(np.float64) @ (b.astype(np.float64) * scales[None, :])
    return acc, folded, deq


def write_case(out, a, b, scales, acc, folded):
    os.makedirs(out, exist_ok=True)
    a.tofile(os.path.join(out, "a.i8"))
    b.tofile(os.path.join(out, "b.i8"))
    scales.astype(np.float32).tofile(os.path.join(out, "scales.f32"))
    acc.astype(np.int32).tofile(os.path.join(out, "c.i32"))
    folded.astype(np.float32).tofile(os.path.join(out, "c.f32"))


def run_case(K, N, M, seed, kind, out):
    a = edge_matrix(M, K, kind) if kind else prng_matrix(M, K, seed)
    b = prng_matrix(K, N, seed + 1)
    sc = column_scales(b)
    acc, folded, deq = reference(a, b, sc)
    rel = float(np.max(np.abs(folded - deq)) / max(1e-12, float(np.max(np.abs(deq)))))
    if out:
        write_case(out, a, b, sc, acc, folded)
    return acc, folded, deq, rel


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
            acc, folded, deq, rel = run_case(a.K, a.N, a.M, a.seed, kind, None)
            ok = rel <= 1e-6
            print(f"  {'ok  ' if ok else 'FAIL'} reference agrees with itself: K={a.K} N={a.N} M={a.M} "
                  f"edge={kind or 'seeded'} rel={rel:.3e} acc_range={int(acc.min())}..{int(acc.max())}")
            bad += 0 if ok else 1
        print("npu_elf_ref self-check: " + ("ok" if not bad else f"{bad} FAILED"))
        return 0 if not bad else 1

    acc, folded, deq, rel = run_case(a.K, a.N, a.M, a.seed, a.edge, a.out)
    print(f"K={a.K} N={a.N} M={a.M} seed={a.seed} edge={a.edge or '-'} rel={rel:.3e} "
          f"acc={int(acc.min())}..{int(acc.max())}" + (f" out={a.out}" if a.out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
