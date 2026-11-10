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
"""Pins the int8 host-reference contract the per-op family gate needs (engine#1808).

A kernel family can only be gated *bit-exactly* against a host arm if both agree on three things that are easy to
differ on silently: the container's grouping (a host using 32 instead of GROUP gets a different scale), the int4
clip (one's-complement `[-8, 7]` versus a symmetric `[-8, 8]` moves any saturating weight), and the accumulator
width (int32 is exact only while `8 * 127 * K` stays inside it). Each of those has an arm here, so a change in the
oracle that would make a host reference disagree for *its own* reasons fails in CI instead of on a device gate.
"""
import importlib.util
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_oracle():
    spec = importlib.util.spec_from_file_location("moe_expert_chunks", ROOT / "tools" / "moe_expert_chunks.py")
    module = importlib.util.module_from_spec(spec)
    saved_argv, saved_path = sys.argv, list(sys.path)
    sys.argv = ["moe_expert_chunks"]  # import only: never run a subcommand
    sys.path.insert(0, str(ROOT / "tools"))  # the oracle imports its siblings by bare name
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv, sys.path[:] = saved_argv, saved_path
    return module


def main():
    m = load_oracle()
    failures = 0

    def check(ok, label):
        nonlocal failures
        if ok:
            print(f"ok   {label}")
        else:
            print(f"FAIL {label}")
            failures += 1

    # 1. The contraction is exact against a slow reference: no fast-path (BLAS) rounding may creep in, because the
    #    gate's `max|d| == 0` rests on integer arithmetic being order-independent.
    rng = np.random.default_rng(11)
    codes, _ = m.quantise(rng.standard_normal((256, 96)).astype(np.float32))
    a8, _ = m.quantise_i8_row(rng.standard_normal(256).astype(np.float32))
    fast = m.int32_accum(codes, a8)
    slow = np.array([sum(int(codes[k, n]) * int(a8[k]) for k in range(codes.shape[0]))
                     for n in range(codes.shape[1])], dtype=np.int32)
    check(np.array_equal(fast, slow), "int32_accum equals a slow per-element reference (exactness, no BLAS rounding)")
    check(fast.dtype == np.int32, "int32_accum returns int32")

    # 2. The container contract: grouping and int4 clip. A host using either of these differently disagrees with the
    #    device for its own reasons, which is the failure this whole issue exists to remove.
    check(m.GROUP == 128, f"GROUP is 128 (the weight scale's divisor a host must use), got {m.GROUP}")
    codes, d = m.quantise(rng.standard_normal((512, 64)).astype(np.float32))
    check(codes.dtype == np.int8, "codes are int8 in an int4 container")
    check(int(codes.min()) >= -8 and int(codes.max()) <= 7,
          f"int4 codes stay in [-8, 7], got [{int(codes.min())}, {int(codes.max())}]")
    check(d.shape == (512 // m.GROUP, 64), f"one scale per (group, column): {d.shape}")

    rng2 = np.random.default_rng(3)
    wt = (rng2.standard_normal((512, 64)) * 40).astype(np.float32)  # large spread: saturates the clip
    codes3, d3 = m.quantise(wt)
    # A property that CAN fail, unlike "codes are in range" (which is unreachable: the ratio |g|/scale is bounded by 7
    # by construction, so +8 never appears however wide the clip is). The round-trip error must stay inside half a
    # step of the group's own scale: change GROUP, the /7 divisor, or the rounding mode and this breaks.
    err = np.abs(m.dequantise(codes3, d3) - wt)
    half_step = np.repeat(d3, m.GROUP, axis=0) / 2.0
    check(bool((err <= half_step + 1e-6).all()),
          f"dequantisation stays inside half a step of the group scale (max err {float(err.max()):.4f}, "
          f"max half-step {float(half_step.max()):.4f})")
    check(int(codes3.min()) >= -8 and int(codes3.max()) <= 7,
          "a saturating weight still lands in [-8, 7] (the bound is the container's, not the data's)")

    # 3. The activation rule: scale = max|a|/127, and an all-zero row must not divide by zero.
    a = rng2.standard_normal(300).astype(np.float32)
    a8, scale = m.quantise_i8_row(a)
    # Computed in ONE precision on purpose: float32 division of a float32 absmax (what the oracle does) and float64
    # division of the same value differ in the last bits, and for a `max|d| == 0` gate that difference is decisive --
    # a host arm that divides in float64 disagrees with the device for its own reasons. This arm pins the precision,
    # not just the constant.
    check(abs(scale - float(np.float32(np.abs(a).max() / 127.0))) == 0.0,
          "activation scale is max|a| / 127 computed in float32 (a float64 divide differs in the last bits)")
    check(abs(int(np.abs(a8.astype(np.int32)).max()) - 127) <= 1, "the largest activation maps to +-127")
    zero8, zero_scale = m.quantise_i8_row(np.zeros(16, dtype=np.float32))
    check(zero_scale == 1.0 and not zero8.any(), "an all-zero activation row uses scale 1.0 and quantises to zeros")

    # 4. The accumulator-width premise of `max|d| == 0`, as arithmetic over the real per-op roles.
    for k in (2048, 8192, 11008):
        bound = 8 * 127 * k
        check(bound < 2**31, f"K={k}: worst-case |sum| = {bound:,} stays inside int32 (headroom x{2**31 // bound})")

    print(("ok   " if not failures else "FAIL ") + f"{4} arms, {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
