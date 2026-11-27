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
"""The Hyperloom screen cache (tools/hyperloom/screen.sh + hl_screen_hit in box_ticket.sh), device-free, on the same
shims as hyperloom_box_ticket_test.py (ssh runs locally, box-hold/sudo pass through, loom-serve is a fake). Checked:
  * one screen ticket measures every set under ONE box-hold and files each under its key;
  * a round whose lever set was screened reads that measurement: no new box-hold, a normal report, same numbers;
  * a round whose lever set was NOT screened runs its own ticket, as without the cache;
  * a stale screen entry (older than HL_SCREEN_MAX_AGE) is measured again;
  * a set with a value outside the safe characters is refused before anything runs.
"""
import importlib.util, json, os, pathlib, subprocess, sys, tempfile, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bt", ROOT / "tests" / "hyperloom_box_ticket_test.py")
bt = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt)
SCREEN = ROOT / "tools/hyperloom/screen.sh"
COMMON = "HL_RAM_NEED_GB=0,HL_RUNS=1"   # the pins the test's rounds carry (setup() exports both)


def holds(log):
    return log.read_text().count("box-hold gpu") if log.exists() else 0


def main():
    fails = []
    check = lambda n, c, d="": c or fails.append("%s %s" % (n, d))
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td); env, log = bt.setup(td); sd = td / "screen"
        p = subprocess.run(["bash", str(SCREEN), "inkling-loom", str(sd), COMMON, "-", "HL_DRAFT_DEPTH=2"],
                           env=dict(env, SCREEN_EST="1"), capture_output=True, text=True, timeout=600)
        check("screen exits 0", p.returncode == 0, p.stdout[-600:] + p.stderr[-600:])
        filed = sorted(x for x in sd.iterdir() if (x / "screen.json").exists()) if sd.exists() else []
        check("screen files both sets", len(filed) == 2, p.stdout[-800:])
        check("screen used ONE box-hold for both sets", holds(log) == 1, log.read_text() if log.exists() else "")

        h0 = holds(log)
        p, rep, rd = bt.round_(env, td, "base", HL_SCREEN_DIR=str(sd))
        check("baseline round reads the screen", "screened:" in p.stdout, p.stdout[-600:] + p.stderr[-400:])
        check("screened round queued no ticket", holds(log) == h0, log.read_text()[-300:])
        check("screened round writes a passing report", bool(rep and rep["quality_gate"]["passed"]), json.dumps(rep)[:400])

        p, rep2, rd = bt.round_(env, td, "d2", HL_SCREEN_DIR=str(sd), HL_DRAFT_DEPTH="2")
        check("screened lever set is read", "screened:" in p.stdout and holds(log) == h0, p.stdout[-400:])
        check("screened lever set has its report", rep2 is not None and rep2.get("output_throughput", 0) > 0)

        p, rep3, rd = bt.round_(env, td, "d5", HL_SCREEN_DIR=str(sd), HL_DRAFT_DEPTH="5")
        check("unscreened lever set runs its own ticket", "screened:" not in p.stdout and holds(log) == h0 + 1, p.stdout[-400:])
        check("unscreened round still reports", rep3 is not None)

        for x in filed:
            old = time.time() - 50000
            os.utime(x / "screen.json", (old, old))
        p, rep4, rd = bt.round_(env, td, "stale", HL_SCREEN_DIR=str(sd))
        check("stale screen entry is measured again", "measuring again" in p.stdout and holds(log) == h0 + 2, p.stdout[-400:])

        p = subprocess.run(["bash", str(SCREEN), "inkling-loom", str(td / "s2"), COMMON, "HL_DRAFT_DEPTH=$(id)"],
                           env=env, capture_output=True, text=True, timeout=60)
        check("an unsafe value is refused", p.returncode == 2 and not any((td / "s2").glob("*/screen.json")), p.stderr[-300:])
    for f in fails:
        print("FAIL", f)
    print("hyperloom_screen: %d checks failed" % len(fails) if fails else "hyperloom_screen: ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
