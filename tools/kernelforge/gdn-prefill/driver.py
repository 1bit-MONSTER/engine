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
"""Measurement driver for the gated delta-net prefill task (KernelForge's driver contract).

Correctness (default): SNR of every output against a float64 torch oracle, per case.
--bench-mode: median kernel time per case over --iters timed runs (HIP events), `case_ms` lines.
--profile-run: warm the kernel, then run it a few times for the profiler.

Cases are Qwen3.8-27B's delta-net layer (48 value heads, 16 shared q/k heads, head size 128)
over one 512-token micro-batch, without (K=1) and with (K=8) speculative-rollback snapshots.
"""
from __future__ import annotations

import argparse
import math
import sys

import torch

from gdn_prefill_kernel import S_V, gdn_prefill

CASES = [  # (tokens, value heads, q/k heads, K)
    (512, 48, 16, 1),
    (512, 48, 16, 8),
]
CHECK_CASES = CASES + [(33, 8, 4, 1), (100, 8, 4, 17), (16, 4, 4, 2)]
_SEED = 7


def _case_id(T, H, Hk, K):
    return f"T{T}_H{H}_Hk{Hk}_K{K}"


def _inputs(T, H, Hk, device):
    gen = torch.Generator(device="cpu").manual_seed(_SEED)
    def r(*shape):
        return torch.randn(*shape, generator=gen)
    q = torch.nn.functional.normalize(r(T, Hk, S_V), dim=-1)
    k = torch.nn.functional.normalize(r(T, Hk, S_V), dim=-1)
    v = r(T, H, S_V)
    # log decay over ggml's own test range (test-backend-ops GATED_DELTA_NET): hard gates make any
    # product of gates underflow, which a gentler range let a campaign miss (tools/kernelforge/README.md)
    g = -1e-4 - torch.rand(T, H, generator=gen) * (20.0 - 1e-4)
    beta = torch.sigmoid(r(T, H))
    state = r(H, S_V, S_V) * 0.1
    return [t.to(device=device, dtype=torch.float32).contiguous() for t in (q, k, v, g, beta, state)]


def _reference(q, k, v, g, beta, state, K):
    """float64 delta rule, token by token, on the CPU (the pip ROCm torch has no gfx1151 kernels of
    its own: only the kernel under test runs on the GPU). Returns out and the state slots."""
    T, H = v.shape[0], v.shape[1]
    Hk = q.shape[1]
    idx = torch.arange(H) % Hk
    q, k, v, g, beta, S = (t.detach().cpu().double() for t in (q, k, v, g, beta, state))
    out = torch.empty_like(v)
    slots = torch.empty((max(K, 1), H, S_V, S_V), dtype=torch.float64)
    scale = 1.0 / math.sqrt(S_V)
    for t in range(T):
        kt, qt = k[t, idx], q[t, idx]                        # [H, 128]
        gv = torch.exp(g[t])[:, None]                        # [H, 1]
        kv = torch.einsum("hci,hi->hc", S, kt)
        delta = (v[t] - gv * kv) * beta[t][:, None]
        S = gv[:, :, None] * S + delta[:, :, None] * kt[:, None, :]
        out[t] = scale * torch.einsum("hci,hi->hc", S, qt)
        slot = T - 1 - t
        if slot < max(K, 1):
            slots[slot] = S
    return out, slots


def _snr_db(ref, test):
    ref, test = ref.double(), test.double()
    noise = ((test - ref) ** 2).mean().item()
    sig = (ref ** 2).mean().item()
    if noise <= 0.0:
        return 100.0
    return 10.0 * math.log10(sig / noise) if sig > 0 else 0.0


def _run_correctness(device):
    worst = float("inf")
    ok = True
    for T, H, Hk, K in CHECK_CASES:
        q, k, v, g, beta, state = _inputs(T, H, Hk, device)
        out, states = gdn_prefill(q, k, v, g, beta, state, K)
        torch.cuda.synchronize()
        out, states = out.cpu(), states.cpu()
        ref_out, ref_slots = _reference(q, k, v, g, beta, state, K)
        n = min(max(K, 1), T)
        snr = min(_snr_db(ref_out, out), _snr_db(ref_slots[:n], states[:n]))
        close = torch.allclose(out.double(), ref_out, atol=1e-3, rtol=1e-3) and \
            torch.allclose(states[:n].double(), ref_slots[:n], atol=1e-3, rtol=1e-3)
        print(f"# {_case_id(T, H, Hk, K)}: SNR {snr:.2f} dB, allclose {close}")
        worst = min(worst, snr)
        ok = ok and close
    print(f"SNR: {worst:.2f} dB")
    print(f"allclose: {ok}")
    return 0


def _run_bench(warmup, iters, device):
    for T, H, Hk, K in CASES:
        args = _inputs(T, H, Hk, device)
        for _ in range(warmup):
            gdn_prefill(*args, K)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            gdn_prefill(*args, K)
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        times.sort()
        for t in times:
            print(f"wall_ms: {t:.6f}")
        print(f"case_ms: {_case_id(T, H, Hk, K)} {times[len(times) // 2]:.6f}")
    return 0


def _run_profile(device):
    for T, H, Hk, K in CASES:
        args = _inputs(T, H, Hk, device)
        for _ in range(6):
            gdn_prefill(*args, K)
    torch.cuda.synchronize()
    return 0


def main():
    ap = argparse.ArgumentParser(description="gated delta-net prefill task driver")
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--profile-run", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    args, _ = ap.parse_known_args()
    if not torch.cuda.is_available():
        print("error: no GPU available (torch.cuda.is_available() is False)")
        return 1
    if args.profile_run:
        return _run_profile("cuda")
    if args.bench_mode:
        return _run_bench(args.warmup, args.iters, "cuda")
    return _run_correctness("cuda")


if __name__ == "__main__":
    sys.exit(main())
