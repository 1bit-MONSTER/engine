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
"""Measurement driver for the Q4_0 x Q8_1 few-column matmul task (KernelForge's driver contract).

Correctness (default): SNR of every case against a float64 oracle computed from the same quantized
blocks, over ordinary inputs AND adversarial ones (huge and tiny scales, all-edge nibbles 0 / 15,
saturated int8 activations, a zero column), so a kernel that is only right on friendly data fails.
--bench-mode: median kernel time per case (HIP events), `case_ms` lines.
--profile-run: warm the kernel, then run it a few times for the profiler.

Cases are Qwen3.8-27B's FFN matrices (5120 -> 17408 and 17408 -> 5120) with 8 columns (a
speculative verification batch), 4 columns, and 1 column (plain decode, which must not regress).
The pip ROCm torch has no gfx1151 kernels of its own, so everything but the kernel under test runs
on the CPU.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch

from gemv_q4_0_kernel import QK, gemv_q4_0

CASES = [  # (ncols_x = K, nrows = N, ncols = batch)
    (5120, 17408, 8),
    (17408, 5120, 8),
    (5120, 17408, 4),
    (5120, 17408, 1),
]
CHECK_CASES = [(512, 96, c) for c in range(1, 9)] + [(5120, 1024, 8), (17408, 256, 3)]
_SEED = 5


def quantize_q4_0(w, rng=None, mode="normal"):
    """w: float [N, K] -> (bytes [N*K/32*18], dequant ints q in 0..15 [N, K/32, 32], scales d [N, K/32])."""
    N, K = w.shape
    b = w.reshape(N, K // QK, QK)
    amax_idx = np.abs(b).argmax(axis=-1)
    maxv = np.take_along_axis(b, amax_idx[..., None], axis=-1)[..., 0]
    d = (maxv / -8.0).astype(np.float16)
    df = d.astype(np.float64)
    inv = np.where(df != 0, 1.0 / np.where(df != 0, df, 1), 0.0)
    q = np.clip(np.floor(b * inv[..., None] + 8.5), 0, 15).astype(np.uint8)
    if mode == "edges":
        q = np.where(rng.random(q.shape) < 0.5, 0, 15).astype(np.uint8)
    packed = (q[..., :16] | (q[..., 16:] << 4)).astype(np.uint8)
    raw = np.zeros((N, K // QK, 18), dtype=np.uint8)
    raw[..., 0:2] = d.view(np.uint8).reshape(N, K // QK, 2)
    raw[..., 2:] = packed
    return raw.reshape(-1), q, d


def quantize_q8_1(a):
    """a: float [C, K] -> (bytes [C*K/32*36], ints q [C, K/32, 32], d [C, K/32], s = d*sum(q) as half)."""
    C, K = a.shape
    b = a.reshape(C, K // QK, QK)
    amax = np.abs(b).max(axis=-1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0)
    q = np.round(b * inv[..., None]).clip(-127, 127).astype(np.int8)
    d16 = d.astype(np.float16)
    s16 = (d * q.astype(np.float64).sum(axis=-1)).astype(np.float16)
    raw = np.zeros((C, K // QK, 36), dtype=np.uint8)
    raw[..., 0:2] = d16.view(np.uint8).reshape(C, K // QK, 2)
    raw[..., 2:4] = s16.view(np.uint8).reshape(C, K // QK, 2)
    raw[..., 4:] = q.view(np.uint8)
    return raw.reshape(-1), q, d16, s16


def reference(q4, d4, q8, d8, s8):
    """float64 out [C, N] = sum_b d4 * (d8 * sum_j q4_j q8_j - 8 * s8), as the kernel defines it."""
    dots = np.einsum("nbj,cbj->cnb", q4.astype(np.float64), q8.astype(np.float64))
    return np.einsum("nb,cnb->cn", d4.astype(np.float64),
                     d8.astype(np.float64)[:, None, :] * dots - 8.0 * s8.astype(np.float64)[:, None, :])


def make_case(K, N, C, mode, seed=_SEED):
    rng = np.random.default_rng(seed + K + 7 * N + 13 * C)
    w = rng.standard_normal((N, K))
    a = rng.standard_normal((C, K))
    if mode == "scales":        # scales across many orders of magnitude, per block
        w *= 10.0 ** rng.uniform(-4, 3, size=(N, K // QK, 1)).repeat(QK, axis=2).reshape(N, K)
        a *= 10.0 ** rng.uniform(-3, 2, size=(C, K // QK, 1)).repeat(QK, axis=2).reshape(C, K)
    if mode == "saturated":     # activations at the int8 limits, one column all zero
        a = np.sign(a) * 50.0
        a[C - 1] = 0.0
    xr, q4, d4 = quantize_q4_0(w, rng, "edges" if mode == "edges" else "normal")
    yr, q8, d8, s8 = quantize_q8_1(a)
    x = torch.from_numpy(xr).cuda()
    y = torch.from_numpy(yr).cuda()
    return x, y, (q4, d4, q8, d8, s8)


def snr_db(ref, test):
    noise = float(((test - ref) ** 2).mean())
    sig = float((ref ** 2).mean())
    if noise <= 0.0:
        return 200.0   # exact: above any real SNR, so the minimum over cases stays informative
    return 10.0 * math.log10(sig / noise) if sig > 0 else 0.0


def _run_correctness():
    worst, ok = float("inf"), True
    for mode in ("normal", "scales", "edges", "saturated"):
        for K, N, C in CHECK_CASES:
            x, y, parts = make_case(K, N, C, mode)
            out = gemv_q4_0(x, y, K, N, C)
            torch.cuda.synchronize()
            out = out.cpu().double().numpy()
            ref = reference(*parts)
            finite = bool(np.isfinite(out).all())
            s = snr_db(ref, out) if finite else float("nan")
            close = finite and np.allclose(out, ref, rtol=1e-3, atol=1e-3 * max(1e-30, float(np.abs(ref).max())))
            if not finite or not close or s < 80:
                print(f"# {mode} K{K} N{N} C{C}: SNR {s:.2f} dB, finite {finite}, allclose {close}")
            worst = min(worst, s) if finite else float("nan")
            ok = ok and close
    print(f"SNR: {worst:.2f} dB")
    print(f"allclose: {ok}")
    return 0


def _run_bench(warmup, iters):
    for K, N, C in CASES:
        x, y, _ = make_case(K, N, C, "normal")
        for _ in range(warmup):
            gemv_q4_0(x, y, K, N, C)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            gemv_q4_0(x, y, K, N, C)
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        times.sort()
        for t in times:
            print(f"wall_ms: {t:.6f}")
        print(f"case_ms: K{K}_N{N}_C{C} {times[len(times) // 2]:.6f}")
    return 0


def _run_profile():
    for K, N, C in CASES:
        x, y, _ = make_case(K, N, C, "normal")
        for _ in range(6):
            gemv_q4_0(x, y, K, N, C)
    torch.cuda.synchronize()
    return 0


def main():
    ap = argparse.ArgumentParser(description="Q4_0 x Q8_1 few-column matmul task driver")
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--profile-run", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args, _ = ap.parse_known_args()
    if not torch.cuda.is_available():
        print("error: no GPU available (torch.cuda.is_available() is False)")
        return 1
    if args.profile_run:
        return _run_profile()
    if args.bench_mode:
        return _run_bench(args.warmup, args.iters)
    return _run_correctness()


if __name__ == "__main__":
    sys.exit(main())
