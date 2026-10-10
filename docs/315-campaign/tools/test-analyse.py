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
"""Regression test for the pre-registered analyser (315-analyse.py).

The acceptance decision for engine#315 is whatever this script prints, so its behaviour on synthetic
consoles is pinned here. Every case below is one the campaign can actually produce, and three of them
guard bugs that were fixed on the record: a rate whose denominator was the *clean* arm count instead of
all in-regime arms, a hardcoded zero numerator, and Face B being filed as harness truncation rather than
as a fault.

Run: python3 docs/315-campaign/tools/test-analyse.py     (exit 0 = all cases as expected)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSE = os.path.join(HERE, "..", "315-analyse.py")


def line(pair, cell="l", verdict="OK", nan=0, req=66, parse=0, gtt=98, utf8=0, alloc=0):
    return (f"  pair {pair} arm {cell}: {verdict} nan={nan} req={req} ret3=0 parse={parse} "
            f"gtt_start={gtt}G utf8={utf8} alloc={alloc}\n")


def run(arms):
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.writelines(arms)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, ANALYSE, path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    out = r.stdout
    m = re.search(r"cell fix \(l\): scheduled=(\d+) in-regime=(\d+) clean=(\d+) fault=(\d+) "
                  r"invalid=(\d+) out-of-regime=(\d+)", out)
    rate = re.search(r"fault rate (\d+)/(\d+) =", out)
    ub = re.search(r"upper bound on the true rate = ([\d.]+)%", out)
    assert m, out
    return {
        "exit": r.returncode,
        "clean": int(m.group(3)), "fault": int(m.group(4)), "invalid": int(m.group(5)),
        "oor": int(m.group(6)),
        "num": int(rate.group(1)) if rate else None, "den": int(rate.group(2)) if rate else None,
        "ub": float(ub.group(1)) if ub else None,
        "accepted": "PASS" in out and "NOT MET" not in out,
    }


CLEAN60 = [line(p) for p in range(1, 61)]
CASES = [
    # name, arms, expected subset
    ("60 clean in-regime arms pass, with the power statement at 5.0%",
     CLEAN60, dict(exit=0, clean=60, fault=0, ub=5.0, accepted=True)),
    # The denominator must be all in-regime arms, not the clean count (fixed bug).
    ("one Face-A fault fails and is reported over all in-regime arms",
     CLEAN60 + [line(61, "l", "FAULT", nan=1, req=10)],
     dict(exit=1, clean=60, fault=1, num=1, den=61, accepted=False)),
    # Face B is a fault face, not harness-side truncation (fixed bug).
    ("one Face-B fault (utf8) fails even though every request completed",
     CLEAN60 + [line(61, "l", utf8=1)],
     dict(exit=1, clean=60, fault=1, num=1, den=61, accepted=False)),
    ("a JSON parse failure fails",
     CLEAN60 + [line(61, "l", parse=1)],
     dict(exit=1, clean=60, fault=1, accepted=False)),
    # Gate edges: below 95 and above 110 are both out of regime, never clean.
    ("arms at 94 GiB are out of regime, not clean",
     [line(p, gtt=94) for p in range(1, 61)], dict(exit=1, clean=0, oor=60, accepted=False)),
    ("arms above 110 GiB are out of regime, not clean",
     [line(p, gtt=120) for p in range(1, 61)], dict(exit=1, clean=0, oor=60, accepted=False)),
    # Allocation failure is invalid, excluded from the regime, and never counted as clean.
    ("a startup allocation failure is invalid, not clean",
     [line(p, alloc=1) for p in range(1, 61)], dict(exit=1, clean=0, invalid=60, accepted=False)),
    ("an INVALID arm is excluded rather than faulted, so 60 clean arms still pass",
     CLEAN60 + [line(61, "l", "INVALID", req=0)], dict(exit=0, clean=60, fault=0, invalid=1, accepted=True)),
    # The request-count floor is part of the fault rule.
    ("an arm that stops at 65 of 66 requests is a fault",
     CLEAN60 + [line(61, "l", req=65)], dict(exit=1, clean=60, fault=1, accepted=False)),
    ("61 clean arms pass and tighten the bound to 4.9%",
     CLEAN60 + [line(61)], dict(exit=0, clean=61, fault=0, ub=4.9, accepted=True)),
]


def main():
    bad = 0
    for name, arms, want in CASES:
        got = run(arms)
        wrong = {k: (want[k], got.get(k)) for k in want if got.get(k) != want[k]}
        if wrong:
            bad += 1
            print(f"FAIL {name}")
            for k, (w, g) in wrong.items():
                print(f"       {k}: expected {w!r}, got {g!r}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(CASES) - bad}/{len(CASES)} cases as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
