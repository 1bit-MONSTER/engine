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
"""scripts/repack_gguf.py: what the repack may claim about a container's tile order (engine#1110).

The load-time guard (#1130) refuses a `container.tile_order` this decoder does not implement, but
`tools/npu_plan.py` writes `engine-1` unconditionally, so the guard is green on a container whose
TILES are another order -- which repacks cleanly and decodes as noise at HTTP 200 (measured with
FLM_Q4NX_Converter `d1d5232`).  The repack now measures the order and refuses the directory; this
test pins the mapping from the checker's verdict to (stamp value, fatal) with the four shapes that
occur, so the policy cannot drift from the tool an operator runs by hand.

usage: tests/repack_tile_order_test.py        (stdlib only: no numpy, no gguf-py, no model, no device)
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "repack_gguf", os.path.join(HERE, os.pardir, "scripts", "repack_gguf.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

FAIL = []


def check(ok, what, why=""):
    if ok:
        print("ok  ", what)
        return
    FAIL.append(what)
    print("FAIL", what, ("  -- " + why) if why else "")


# (rc, line) -> (stamp value, fatal), with the lines tools/q4nx_diff.py verdict() really returns.
CASES = [
    # a correct requantized container: the honest Q4_K_M -> q4_1 error, the other order far above it
    (0, "matches layout ours (worst max|err| 5.600e-02, next layout flm worst max|err| 2.6e+00)",
     "engine-1", False, "the measured order is the one npu_plan.py declares"),
    # a round trip, exact on every tile
    (0, "matches layout ours (168 tensor(s) exactly)", "engine-1", False, "an exact round trip is ours"),
    # the d1d5232 converter: no order matches, because the tiles are not these weights
    (1, "matches NO layout: closest is flm with worst max|err| 2.474e+00 (> 1.000e-01, or not 3x "
        "better than ours) - the container's tiles are not these weights", "", True,
     "a container that is not these weights must not become a cached directory"),
    # a container written in the other order: it MATCHES, but not the order npu/q4nx.cpp decodes
    (0, "matches layout flm (worst max|err| 5.6e-02, next layout ours worst max|err| 2.6e+00)",
     "", True, "a matching FLM container is still one this engine cannot read"),
    # an architecture the checker cannot dequantize: nothing compared, so nothing may be claimed
    (2, "nothing was compared", "unverified", False, "no evidence is not evidence of the right order"),
]

for rc, line, want_order, want_fatal, why in CASES:
    got_order, got_fatal = R.tile_order_policy(rc, line)
    check(got_order == want_order and got_fatal == want_fatal,
          f"{line[:44]!r} -> ({want_order!r}, fatal={want_fatal})", why)

# `unverified` must never be spelled `engine-1`: the whole point is that the stamp stops asserting.
check(R.tile_order_policy(2, "nothing was compared")[0] != "engine-1",
      "an unverified directory does not claim engine-1")

# `--tile-check`'s exit codes are the policy's, not the checker's rc: verdict() returns 0 for ANY
# matching layout, including the one this engine cannot decode (engine#1110 review, blocker 1).
EXIT_CASES = [
    ("matches layout ours (worst max|err| 5.600e-02, next layout flm worst max|err| 2.6e+00)", 0, 0,
     "the engine-1 order exits 0"),
    ("matches layout flm (worst max|err| 5.600e-02, next layout ours worst max|err| 2.6e+00)", 0, 1,
     "a matching layout this decoder does not implement exits 1, not 0"),
    # the exact line pi-prs measured: a 0-tensor pair used to come back as this 'match'
    ("matches layout flm (worst max|err| 0.000e+00, next layout ours worst max|err| 0.000e+00)", 0, 1,
     "the 0-vs-0 tie is not a pass"),
    ("matches NO layout: closest is flm ...", 1, 1, "no match exits 1"),
    ("nothing was compared", 2, 2, "nothing compared exits 2"),
]
for line, rc, want_exit, why in EXIT_CASES:
    got = R.tile_check_exit(rc, line)
    check(got == want_exit, f"--tile-check exit {want_exit} for {line[:40]!r}", why)

print(f"repack_tile_order_test: {len(FAIL)} failure(s)")
sys.exit(1 if FAIL else 0)
