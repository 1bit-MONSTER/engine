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
"""Frozen harness generator for the NPU kernel arena.

    harness/gen.py TASK [--seeds 3] [--force]

Writes, into tasks/<TASK>/harness/:

  geometry.txt            K, N, M (M = the activation rows of the decode twin)
  seed<s>_A.bin           A[M][K]   int8, row-major  (the device A BO layout)
  seed<s>_B.bin           B[K][N]   int8, row-major  (the packB_blocked output
                                                     layout: what the engine
                                                     streams to the device)
  seed<s>_expected.bin    C[M][N]   int32, row-major, = A @ B EXACTLY
  seed<s>_scales.bin      float32[N]  per-column B scales (provenance only: the
                          kernel never reads a scale, the host multiplies it back
                          in; the file is here so a submission cannot quietly
                          change the quantisation it is scored on)
  SHA256SUMS              every file above

The matrices are synthetic but drawn from the distributions the real 27B
container produces after the engine's own quantiser:

  * B is TERNARY with the container's real structure: one magnitude per 32-row
    group, then 0 / +mag / -mag per element (a third of each column is zero).
    That is what a ternary column's int8 codes look like after the engine's
    per-column amax pack, and it keeps the lossless 2-bit packing lever open.
  * A rows are Gaussian clipped to int8 (post-amax activations).

int8 x int8 -> int32 is exact, so the expected tensor is a plain integer
matmul, not a float model: a submission that differs in one element is wrong
(layout, stride, tile order, accumulation width), there is no tolerance to
argue about.  3 seeds so a submission cannot fit one input.

Regenerating is allowed only with --force and must be recorded in the task's
README: a frozen harness is the arena's ground truth, and re-freezing it
invalidates every score.
"""
import argparse
import hashlib
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
GEO = {"GU": (5120, 17408), "D": (17408, 5120), "QKV": (5120, 10240),
       "O": (6144, 5120), "Z": (5120, 6144)}
M_ROWS = 16          # the decode twin's baked activation tile


def make_b(k, n, rng):
    """B in the shape the Bonsai-2 container really has: TERNARY.

    Bonsai-2-27B PTQ1_0 stores ternary weights with one scale per 32-weight group,
    and the engine's packB then quantises each column of a 64-column block with a
    single amax.  For a ternary column that leaves every row with exactly ONE
    magnitude: code[k][j] is 0, +mag_k or -mag_k, with mag_k constant down the row
    (it moves only at the 32-row group boundary).  That structure is what makes a
    2-bit device-side packing lossless, so the harness has to carry it: a
    dense/uniform column here would quietly make the biggest available lever look
    impossible.
    """
    b = np.zeros((k, n), dtype=np.int8)
    mag = np.empty(k, dtype=np.int16)                   # one magnitude per row
    for g in range(0, k, 32):                           # ... per 32-row group
        mag[g:g + 32] = rng.integers(40, 128, size=1)   # the group's |scale|
    for j in range(n):
        w = rng.integers(-1, 2, size=k)                 # -1, 0, +1, ~1/3 each
        b[:, j] = (w * mag).astype(np.int8)
    return b


def make_a(m, k, rng):
    a = np.clip(np.round(rng.normal(0.0, 40.0, size=(m, k))), -127, 127)
    return a.astype(np.int8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--k", type=int, default=0, help="override K (toy tasks)")
    ap.add_argument("--n", type=int, default=0, help="override N (toy tasks)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.task in GEO:
        k, n = GEO[args.task]
    elif args.k and args.n:
        k, n = args.k, args.n
    else:
        print(f"unknown task {args.task}: pass --k and --n for a toy geometry")
        return 2
    out = ROOT / "tasks" / args.task / "harness"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "SHA256SUMS").exists() and not args.force:
        print(f"{out}/SHA256SUMS exists; frozen (use --force to re-freeze)")
        return 0

    (out / "geometry.txt").write_text(f"K {k}\nN {n}\nM {M_ROWS}\n")
    for s in range(1, args.seeds + 1):
        rng = np.random.default_rng(1000 * s + k + n)   # deterministic per (task, seed)
        a = make_a(M_ROWS, k, rng)
        b = make_b(k, n, rng)
        # int32 accumulation is exact here: |sum| <= K * 127 * 127 < 2^31 at K=17408.
        c = a.astype(np.int32) @ b.astype(np.int32)
        assert c.dtype == np.int32 and np.abs(c).max() < 2**31
        a.tofile(out / f"seed{s}_A.bin")
        b.tofile(out / f"seed{s}_B.bin")
        c.tofile(out / f"seed{s}_expected.bin")
        # per-column scale the engine would have used (amax/127 of the f32 column)
        rng2 = np.random.default_rng(7000 + s)
        scales = (rng2.uniform(0.004, 0.02, size=n)).astype(np.float32)
        scales.tofile(out / f"seed{s}_scales.bin")
        print(f"seed{s}: A {a.shape} B {b.shape} C {c.shape} "
              f"|C|max={int(np.abs(c).max())} nz={int((b != 0).sum())/b.size:.1%}")

    h = hashlib.sha256()
    lines = []
    for p in sorted(out.iterdir()):
        if p.name == "SHA256SUMS" or p.is_dir():
            continue
        d = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{d}  {p.name}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"frozen {len(lines)} files under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
