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
"""tools/split_ceiling.py -- what an iGPU+NPU M-split can buy, from two measured rates.

engine#754 ("split prefill in the hybrid route") is blocked on a number, and the number has been written
down three times with three different answers because the iGPU's own prefill rate is not a constant of
the box (docs/hetero-prefill.md): 40.7 TFLOP/s before the 2026-11-01 reset, 12.1-12.4 on the builds that
carried engine#782's regression, ~44 once the pinned kernel set stopped using the slow tiles.  A ceiling
quoted without the rates it came from is therefore either stale or unfalsifiable.

The ceiling obeys one formula, and this tool is that formula:

    The iGPU takes  g = gpu_ms / M  per row.  One NPU tile is  t_ms  and covers T rows, so the NPU takes
    n = t_ms / T per row.  With both engines running at once on disjoint row ranges, the wall clock is
    the slower side, minimised when both finish together:

        tiles*  = g*M / (t_ms + T*g)          (clamped to [0, ceil(M/T)])
        wall*   = max(g*(M - tiles**T), tiles**t_ms)
        ceiling = gpu_ms / wall*              (per GEMM; always <= g/n, the throughput ratio)

    End to end the split covers the GEMMs only, so with s = the split-able share of prefill device time:

        e2e     = 1 / ((1 - s) + s/ceiling)

Both are exact given the inputs; nothing here is a model of the hardware, and the tool refuses to invent
one: every argument is a measurement and the provenance is echoed with the answer.

    split_ceiling.py --gpu-ms 18.4 --npu-tile-ms 8.61 --rows 16384
    split_ceiling.py --gpu-tile-ms 1.441 --npu-tile-ms 8.61 --rows 16384 --gemm-share 0.515
    split_ceiling.py --gpu-ms 65.71 --npu-tile-ms 8.61 --rows 16384 --gemm-share 0.523 --json
    split_ceiling.py --self-test

Which iGPU rate is fed in decides the answer, and both are defensible only if named: the microbenchmark
(`mosaic_bench`, 0.313 ms per 256-row group at M=16384) is 3.4-4.6x faster than the engine's own prefill
capture (1.06-1.44 ms per group, read with the bundle's timestamp_frequency_hz). A split runs beside the
engine, so the ceiling should be quoted from the engine's rate; `SELF_TEST` carries both, and the third
measurement's rows are the ones the #1270 review recomputed by hand.

--self-test replays the four points docs/hetero-prefill.md records (pre-reset, engine#782 regression,
post-#782, and the end-to-end factor at the measured attention share) and fails if the arithmetic stops
reproducing them, so the page and the tool cannot drift apart silently.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

TILE_ROWS = 256  # the NPU prefill kernel's row tile, the same T the recorded measurements use


def ceiling(gpu_ms: float, npu_tile_ms: float, rows: int, tile: int = TILE_ROWS) -> dict:
    """The optimum M-split for one GEMM, from the iGPU's time for all `rows` and one NPU tile's time."""
    if gpu_ms <= 0 or npu_tile_ms <= 0 or rows <= 0 or tile <= 0:
        raise ValueError("gpu-ms, npu-tile-ms, rows and tile must all be > 0")
    g = gpu_ms / rows            # ms per row, iGPU
    n = npu_tile_ms / tile       # ms per row, NPU
    max_tiles = math.ceil(rows / tile)
    # Both sides finish together at k = g*M / (t + T*g) tiles, but the tile count is an integer and the
    # optimum is not always the nearest one (with k=6.8, 6 tiles leave the iGPU critical and 7 leave the
    # NPU critical; the measurement decides which is smaller). Under the linear-row model below the best
    # integer tile count is a scan, not a rounding.
    best = None
    for k in range(0, max_tiles + 1):
        wall_k = max(g * max(0, rows - k * tile), k * npu_tile_ms)
        if wall_k > 0 and (best is None or wall_k < best[1]):
            best = (k, wall_k)
    tiles, wall = best if best else (0, gpu_ms)
    return {
        "rows": rows,
        "tile": tile,
        "gpu_ms": gpu_ms,
        "npu_tile_ms": npu_tile_ms,
        "gpu_ms_per_row": g,
        "npu_ms_per_row": n,
        "ratio": n / g,                       # NPU row cost / iGPU row cost (>1 means the NPU is slower)
        "optimum_npu_tiles": tiles,
        "optimum_npu_share": (tiles * tile) / rows,
        "wall_ms": wall,
        "ceiling": gpu_ms / wall,
    }


def end_to_end(per_gemm_ceiling: float, gemm_share: float) -> float:
    """Amdahl for the split: only `gemm_share` of prefill device time is a GEMM the split can touch."""
    if not 0.0 <= gemm_share <= 1.0:
        raise ValueError("gemm-share must be in [0, 1]")
    if per_gemm_ceiling < 1.0:
        raise ValueError("a ceiling below 1 would mean the split is slower; not a ceiling")
    return 1.0 / ((1.0 - gemm_share) + gemm_share / per_gemm_ceiling)


SELF_TEST = [
    # (label, gpu_ms, npu_tile_ms, rows, expected ceiling, tolerance)   -- from docs/hetero-prefill.md
    ("pre-reset 2026-11-01, 4B up_gate M=16384 (20.03 ms, 8.61 ms/tile)", 20.03, 8.61, 16384, 1.036, 0.006),
    # docs/hetero-prefill.md measured 1.098-1.128x on that boot; the linear-row model here lands at 1.104x,
    # because it cannot know that the iGPU's time is not exactly proportional to its row count
    ("engine#782 regression, 4B up_gate M=16384 (65.71 ms, 8.61 ms/tile)", 65.71, 8.61, 16384, 1.104, 0.02),
    ("post-#782 (44 TFLOP/s), 4B up_gate M=16384 (18.4 ms, 8.61 ms/tile)", 18.4, 8.61, 16384, 1.032, 0.006),
    # the 2026-11-06 re-measurement on the release-rc libs (9-11 TFLOP/s): the split pays most here, and
    # that is exactly why the number must never be quoted without the lib it came from
    # measured 1.134x at 4 tiles with mosaic_bench --mode split; the model says 1.104x, so this entry asserts
    # the *gap* rather than the number: the model must stay below the measurement and within 4 points of it
    ("2026-11-06 release-rc libs, 4B down M=16384 (77.97 ms, 10.22 ms/tile)", 77.97, 10.22, 16384, 1.104, 0.04),
    # The corrected third measurement (review of #1270 at d0122cec), read with the bundle's 99.81 MHz
    # `timestamp_frequency_hz` instead of assuming nanoseconds. These three rows are what the review recomputed by
    # hand -- and they are a measurement of a *pre-#782 library*: the capture's `libggml-hrx` was four days older
    # than the pin and still contained the spilling `m128_w8_f2_sb` tiles (24 references against 0 in the
    # post-fix build). They are pinned here because they are the arithmetic the discussion quotes, not because
    # they describe the shipped engine; the row that does is the microbenchmark on the post-fix pin above.
    ("in-engine capture, 1.441 ms/256 rows, today's tile (92.22 ms, 8.61 ms/tile)", 92.224, 8.61, 16384, 1.164, 0.004),
    ("in-engine at #754's 4x bar (2.15 ms/tile)", 92.224, 2.15, 16384, 1.650, 0.01),
    ("in-engine, slower per-dispatch reading 2.883 ms/256 rows", 184.512, 8.61, 16384, 1.333, 0.004),
    # The post-fix runtime's own capture (2026-11-07 18:34, ~/.cache/hw-gate/build-13a7fda57a, gpu-kernels fd69590,
    # 0 references to the spilling tiles). This is the row that describes the engine the pin builds: matmuls 2.3x
    # faster than the pre-fix library, attention grown to 57.1 % of busy, so the split's end-to-end worth shrinks.
    ("post-fix runtime, in-engine 0.855 ms/256 rows", 54.72, 8.61, 16384, 1.085, 0.004),
]


def self_test() -> int:
    bad = 0
    for label, gpu_ms, tile_ms, rows, want, tol in SELF_TEST:
        got = ceiling(gpu_ms, tile_ms, rows)
        ok = abs(got["ceiling"] - want) <= tol
        print(f"{'ok  ' if ok else 'FAIL'} {label}: ceiling {got['ceiling']:.3f} (want {want:.3f} +-{tol}) "
              f"optimum {got['optimum_npu_tiles']} of {math.ceil(rows / TILE_ROWS)} tiles")
        bad += 0 if ok else 1
    # The end-to-end factor at the measured attention share of Qwen3-4B's pp16384 device time (47.7 % attention,
    # so 52.3 % is GEMM the split could touch), on the fast-iGPU point.
    e2e = end_to_end(ceiling(18.4, 8.61, 16384)["ceiling"], 0.523)
    ok = 1.005 <= e2e <= 1.025
    print(f"{'ok  ' if ok else 'FAIL'} end-to-end at a 52.3 % split-able share: {e2e:.3f}x (want 1.005..1.025)")
    bad += 0 if ok else 1
    # Refusals, not just the happy path: bad inputs must not produce a number.
    for kwargs, why in (
        ({"gpu_ms": 0, "npu_tile_ms": 8.61, "rows": 16384}, "gpu_ms=0"),
        ({"gpu_ms": 18.4, "npu_tile_ms": -1, "rows": 16384}, "negative tile"),
        ({"gpu_ms": 18.4, "npu_tile_ms": 8.61, "rows": 0}, "rows=0"),
    ):
        try:
            ceiling(**kwargs)
            print(f"FAIL refusal missing for {why}")
            bad += 1
        except ValueError:
            print(f"ok   refused {why}")
    for args, why in (((1.03, 1.5), "gemm-share > 1"), ((0.9, 0.5), "ceiling < 1")):
        try:
            end_to_end(*args)
            print(f"FAIL refusal missing for {why}")
            bad += 1
        except ValueError:
            print(f"ok   refused {why}")
    print("self-test:", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="What an iGPU+NPU M-split can buy, from two measured rates.")
    ap.add_argument("--gpu-ms", type=float, help="the iGPU's measured time for all --rows, in ms")
    ap.add_argument("--gpu-tile-ms", type=float,
                    help="the iGPU's measured time per --tile rows (the unit the prefill captures report); "
                         "an alternative to --gpu-ms")
    ap.add_argument("--npu-tile-ms", type=float, help="the NPU kernel's measured time for one --tile-row launch, in ms")
    ap.add_argument("--rows", type=int, default=TILE_ROWS, help=f"rows in the GEMM (default {TILE_ROWS})")
    ap.add_argument("--tile", type=int, default=TILE_ROWS, help=f"rows per NPU launch (default {TILE_ROWS})")
    ap.add_argument("--gemm-share", type=float, default=None,
                    help="split-able share of prefill device time (adds the end-to-end factor)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--self-test", action="store_true", help="replay the measured points in docs/hetero-prefill.md")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.gpu_ms is not None and a.gpu_tile_ms is not None:
        ap.error("give --gpu-ms or --gpu-tile-ms, not both")
    if a.gpu_tile_ms is not None:
        if a.gpu_tile_ms <= 0 or a.rows <= 0:
            print("split_ceiling: --gpu-tile-ms and --rows must be > 0", file=sys.stderr)
            return 2
        a.gpu_ms = a.gpu_tile_ms * a.rows / a.tile
    if a.gpu_ms is None or a.npu_tile_ms is None:
        ap.error("--gpu-ms (or --gpu-tile-ms) and --npu-tile-ms are required (or --self-test)")
    try:
        out = ceiling(a.gpu_ms, a.npu_tile_ms, a.rows, a.tile)
        if a.gemm_share is not None:
            out["gemm_share"] = a.gemm_share
            out["end_to_end"] = end_to_end(out["ceiling"], a.gemm_share)
    except ValueError as e:
        print(f"split_ceiling: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    print(f"iGPU {out['gpu_ms']} ms for {out['rows']} rows = {out['gpu_ms_per_row'] * 1000:.4f} us/row")
    print(f"NPU  {out['npu_tile_ms']} ms per {out['tile']}-row tile = {out['npu_ms_per_row'] * 1000:.4f} us/row "
          f"({out['ratio']:.2f}x the iGPU's row cost)")
    print(f"optimum split: {out['optimum_npu_tiles']} NPU tile(s) = "
          f"{out['optimum_npu_share'] * 100:.1f} % of the rows -> {out['wall_ms']:.2f} ms")
    print(f"ceiling: {out['ceiling']:.3f}x per GEMM")
    if "end_to_end" in out:
        print(f"end to end at a {out['gemm_share'] * 100:.1f} % split-able share: {out['end_to_end']:.3f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
