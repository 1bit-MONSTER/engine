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
"""tools/split_ceiling.py: the M-split ceiling, against the points docs/hetero-prefill.md records.

Why this test and not only the tool's --self-test: the number this tool prints is a *decision* (engine#754
is blocked on it), and the same number has been written down three times with three different answers because
the iGPU side moved under it.  The test pins all three, so a change to the arithmetic that quietly makes the
split look worth building fails here, and a page that drifts from the tool fails in review because the two
tables are the same four rows.

No model, no device, no numpy: stdlib only, a few microseconds.
"""
import importlib.util
import pathlib
import subprocess
import sys

TOOL = pathlib.Path(__file__).resolve().parent.parent / "tools" / "split_ceiling.py"
spec = importlib.util.spec_from_file_location("split_ceiling", TOOL)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)

fails = 0


def check(label, cond, detail=""):
    global fails
    print(f"{'ok  ' if cond else 'FAIL'} {label}{(' -- ' + detail) if detail and not cond else ''}")
    fails += 0 if cond else 1


# 1. The three iGPU states the page records, at Qwen3-4B up_gate M=16384 with our 8.61 ms T=256 tile.
for label, gpu_ms, want, tol in (
    ("pre-reset boot, 40.7 TFLOP/s", 20.03, 1.036, 0.006),
    ("engine#782 regression, 12.4 TFLOP/s", 65.71, 1.104, 0.02),
    ("post-#782, ~44 TFLOP/s", 18.4, 1.032, 0.006),
):
    got = sc.ceiling(gpu_ms, 8.61, 16384)
    check(f"ceiling, {label}: {got['ceiling']:.3f}x", abs(got["ceiling"] - want) <= tol, f"want {want}+-{tol}")

# 1b. The corrected third measurement (review of #1270 at d0122cec): reading the capture's device ticks with the
#     bundle's 99.81 MHz frequency rather than as nanoseconds gives the in-engine iGPU rate, and the three rows
#     that review recomputed by hand must come out of the same arithmetic. These are the numbers a split would
#     actually run beside -- the microbenchmark is 3-4x faster than the engine -- so a regression here is the
#     difference between "the split cannot pay" and "it pays".
for label, gpu_ms, tile_ms, want, want_tiles in (
    ("in-engine 1.441 ms/256 rows, today's tile", 92.224, 8.61, 1.164, 9),
    ("in-engine at the 4x bar (2.15 ms/tile)", 92.224, 2.15, 1.650, 26),
    ("in-engine, slower per-dispatch reading 2.883 ms/256 rows", 184.512, 8.61, 1.333, 16),
):
    got = sc.ceiling(gpu_ms, tile_ms, 16384)
    check(f"{label}: {got['ceiling']:.3f}x at {got['optimum_npu_tiles']} tiles",
          abs(got["ceiling"] - want) <= 0.004 and got["optimum_npu_tiles"] == want_tiles,
          f"want {want}+-0.004 at {want_tiles} tiles")

# 2. The optimum is a scan over integer tile counts, not a rounded continuous one: at 12.4 TFLOP/s the
#    continuous optimum is 6.8 tiles, and 7 (the nearest) is *worse* than 6 because the NPU becomes the
#    critical path.  A rounding implementation would report 60.27 ms there instead of 59.55.
reg = sc.ceiling(65.71, 8.61, 16384)
check("integer-tile optimum is 6, not the rounded 7", reg["optimum_npu_tiles"] == 6, str(reg["optimum_npu_tiles"]))

# 3. The end-to-end factor: attention is 47.7 % of Qwen3-4B's pp16384 device time (the recorded split), so a
#    split can only touch 52.3 % of it.  With the fast iGPU that turns 1.032x per GEMM into 1.017x e2e -- the
#    number that decides engine#754, and the one the page quotes.
e2e = sc.end_to_end(sc.ceiling(18.4, 8.61, 16384)["ceiling"], 0.523)
check(f"end-to-end at 52.3 % split-able share, microbenchmark ceiling: {e2e:.3f}x", 1.005 <= e2e <= 1.025)
# The corrected third measurement puts attention at 48.5 % (51.5 % split-able) and the in-engine GEMM rate at
# ~1.44 ms per 256-row group, which is the combination #754 has to be decided on: ~1.08x end to end.
e2e_engine = sc.end_to_end(sc.ceiling(92.224, 8.61, 16384)["ceiling"], 0.515)
check(f"end-to-end at 51.5 % split-able share, in-engine ceiling: {e2e_engine:.3f}x",
      1.06 <= e2e_engine <= 1.09, f"got {e2e_engine:.3f}")

# 4. Amdahl's shape, not a slogan: an all-GEMM prefill gets the per-GEMM ceiling, a no-GEMM prefill gets 1.0,
#    and the factor is monotone in the share between them.
check("all-GEMM share gives the per-GEMM ceiling", abs(sc.end_to_end(1.03, 1.0) - 1.03) < 1e-9)
check("no-GEMM share gives 1.0", abs(sc.end_to_end(1.03, 0.0) - 1.0) < 1e-9)
check("e2e is monotone in the share", sc.end_to_end(1.03, 0.25) < sc.end_to_end(1.03, 0.75))

# 5. Refusals before numbers: a zero/negative input or an out-of-range share must raise, because a tool that
#    returns a ceiling for a nonsense rate is how a stale number gets quoted with a fresh date.
for kwargs in ({"gpu_ms": 0, "npu_tile_ms": 8.61, "rows": 16384},
               {"gpu_ms": 18.4, "npu_tile_ms": 0, "rows": 16384},
               {"gpu_ms": 18.4, "npu_tile_ms": 8.61, "rows": 0}):
    try:
        sc.ceiling(**kwargs)
        check(f"refused {kwargs}", False, "no ValueError")
    except ValueError:
        check(f"refused {kwargs}", True)
for a in ((1.03, -0.1), (1.03, 1.1), (0.9, 0.5)):
    try:
        sc.end_to_end(*a)
        check(f"refused end_to_end{a}", False, "no ValueError")
    except ValueError:
        check(f"refused end_to_end{a}", True)

# 6. The CLI is the contract the page tells a reader to run: --self-test must pass and --json must carry the
#    two numbers the decision needs.
r = subprocess.run([sys.executable, str(TOOL), "--self-test"], capture_output=True, text=True)
check("tools/split_ceiling.py --self-test exits 0", r.returncode == 0, r.stdout + r.stderr)
r = subprocess.run([sys.executable, str(TOOL), "--gpu-ms", "18.4", "--npu-tile-ms", "8.61",
                    "--rows", "16384", "--gemm-share", "0.523", "--json"], capture_output=True, text=True)
import json
try:
    out = json.loads(r.stdout)
    ok = abs(out["ceiling"] - 1.032) <= 0.006 and abs(out["end_to_end"] - 1.017) <= 0.01
    check("--json carries ceiling and end_to_end", ok, r.stdout)
except Exception as e:  # noqa: BLE001 - any parse failure is the finding
    check("--json carries ceiling and end_to_end", False, f"{e}: {r.stdout} {r.stderr}")

# 6b. --gpu-tile-ms is the spelling the captures report in, and it must be the same number as --gpu-ms
r1 = subprocess.run([sys.executable, str(TOOL), "--gpu-tile-ms", "1.441", "--npu-tile-ms", "8.61",
                     "--rows", "16384", "--json"], capture_output=True, text=True)
r2 = subprocess.run([sys.executable, str(TOOL), "--gpu-ms", str(1.441 * 64), "--npu-tile-ms", "8.61",
                     "--rows", "16384", "--json"], capture_output=True, text=True)
try:
    same = json.loads(r1.stdout)["ceiling"] == json.loads(r2.stdout)["ceiling"]
    check("--gpu-tile-ms agrees with --gpu-ms", same, f"{r1.stdout} vs {r2.stdout}")
except Exception as e:  # noqa: BLE001
    check("--gpu-tile-ms agrees with --gpu-ms", False, f"{e}: {r1.stdout} {r1.stderr}")
r3 = subprocess.run([sys.executable, str(TOOL), "--gpu-ms", "92.2", "--gpu-tile-ms", "1.441",
                     "--npu-tile-ms", "8.61"], capture_output=True, text=True)
check("refuses both spellings at once", r3.returncode == 2, f"rc={r3.returncode}")

print("split_ceiling:", "PASS" if fails == 0 else f"FAIL ({fails})")
sys.exit(0 if fails == 0 else 1)
