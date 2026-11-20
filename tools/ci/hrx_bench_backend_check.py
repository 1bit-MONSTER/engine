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
"""hrx_bench_backend_check.py -- did this bench log come from a backend that initialised?

A llama-bench run whose HRX runtime failed to initialise still prints a full results
table, labelled `backend HRX`, from whatever path it fell back to. engine#2858 is that
row: a shipped bench tree on a box it was not built on reported IQ1_S `pp512 11.22` on
2026-11-18 and `73.61` on the same tree a day later, and in BOTH logs the runtime said

    hrx_gpu_initialize(0) failed: ... symbol 'hsa_amd_queue_create' not found in library
    ggml_hrx: the HRX runtime did not initialise, so this backend registers no devices

So neither number was an HRX measurement, and comparing either to a working local build
compares two different backends. The failure is silent in the only artefact anybody
keeps -- the table -- which is why this is a check and not a convention.

    hrx_bench_backend_check.py --log build/bench.log      # a kept bench log
    hrx_bench_backend_check.py --self-test

Exit 0 = a table from an initialised backend, 1 = the log says the backend did not
initialise (or a run supplied both), 2 = cannot tell (no table and no verdict either
way, or the log could not be read). A check that cannot look is not a pass.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# The runtime says this itself when it fails; naming the sentence once keeps the check and
# the binary from drifting (the text is ggml-hrx.cpp's, matched case-insensitively).
NO_DEVICES = re.compile(r"the HRX runtime did not initialise|registers no devices", re.I)
# A results row: `| model | size | params | backend | … |` -- the table llama-bench prints.
ROW = re.compile(r"^\|\s*[^|]+\|\s*[^|]+\|\s*[^|]+\|\s*HRX\s*\|", re.I)


def verdict(log_text: str):
    """(rc, reason) for one captured bench log."""
    lines = [ln for ln in (log_text or "").splitlines()]
    if not lines:
        return 2, "the log is empty: no table and no runtime verdict"
    bad = [ln.strip() for ln in lines if NO_DEVICES.search(ln)]
    rows = [ln for ln in lines if ROW.search(ln)]
    if bad:
        # Even when the table is there: a table whose backend did not initialise is the
        # whole point of this check, so the bad line outranks a clean-looking results block.
        return 1, f"the runtime says it did not initialise ({len(rows)} result row(s) beside it): {bad[0][:120]}"
    if not rows:
        return 2, "no result row and no runtime verdict: this does not say a backend initialised"
    return 0, f"{len(rows)} result row(s), no initialisation failure in the log"


def self_test() -> int:
    fails = 0

    def ok(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        fails += 0 if cond else 1

    table = "| qwen35 27B IQ1_S | 5.76 GiB | 26.90 B | HRX | -1 | 8 | pp512 | 73.61 ± 0.14 |\n"
    init_fail = ("hrx_gpu_initialize(0) failed: NOT_FOUND; symbol 'hsa_amd_queue_create' not found in library\n"
                 "ggml_hrx: the HRX runtime did not initialise, so this backend registers no devices\n")

    rc, why = verdict(table)
    ok(rc == 0, f"a table with no initialisation failure is a measurement: {why[:60]}")
    rc, why = verdict(init_fail + table)
    ok(rc == 1, "the engine#2858 shape fails: a table printed by a runtime that did not initialise")
    ok("did not initialise" in why, "and the reason quotes the line rather than counting rows")
    rc, _why = verdict("ggml_hrx: the HRX runtime did not initialise, so this backend registers no devices\n")
    ok(rc == 1, "the failure alone fails too (a log kept before the table was printed)")
    rc, _why = verdict("")
    ok(rc == 2, "an empty log is cannot-tell, not a pass")
    rc, _why = verdict("usage: llama-bench [options]\n")
    ok(rc == 2, "a log with neither a table nor a verdict is cannot-tell")
    # case-insensitivity: the binary's wording is the invariant, not its capitalisation
    rc, _why = verdict("GGML_HRX: The HRX Runtime Did Not Initialise\n" + table)
    ok(rc == 1, "the match is case-insensitive, so a reworded capitalisation cannot pass")
    rc, _why = verdict("| model | size | params | CPU | 4 | 4 | pp512 | 12.0 |\n")
    ok(rc == 2, "a CPU table is not an HRX measurement, and this tool does not pass it as one")

    print("hrx_bench_backend_check --self-test:", "ok" if fails == 0 else f"{fails} failure(s)")
    return 0 if fails == 0 else 1


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", help="a captured llama-bench log")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.log:
        ap.print_help()
        return 2
    p = pathlib.Path(a.log)
    try:
        text = p.read_text(errors="replace")
    except OSError as e:
        print(f"hrx_bench_backend_check: cannot tell: cannot read {p}: {e}", file=sys.stderr)
        return 2
    rc, why = verdict(text)
    print(f"hrx_bench_backend_check: {p}: {why}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
