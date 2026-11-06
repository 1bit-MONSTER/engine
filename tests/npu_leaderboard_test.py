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
"""docs/npu-leaderboard.md is generated, and nothing checked that it is what the records produce.

The page says so at the top ("edit the records, not this file") and tools/npu_leaderboard.py is the only writer,
yet a record could be added, corrected or deleted and the page would keep the old table forever: every number on
it would still read as measured, and the file a reader trusts would be a stale projection of files they never
open. That is the class of defect this repository keeps finding -- a generated artifact with no regeneration check
-- so the check lives here instead of in a reviewer's memory.

Three things are asserted:
1. regenerating into a temp file reproduces docs/npu-leaderboard.md byte for byte (the page is current);
2. the comparison is not vacuous: a single changed character in the regenerated copy is caught, with the line
   named (this is the self-check -- without it a comparison that always passed would look exactly the same);
3. the record directory still validates (tools/npu_leaderboard.py --check), and the page carries one row per
   record, so a record that is silently dropped from the table cannot pass.

usage: tests/npu_leaderboard_test.py        (pure python, no model, no device)
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "npu_leaderboard.py")
PAGE = os.path.join(ROOT, "docs", "npu-leaderboard.md")
RECORDS = os.path.join(ROOT, "bench", "npu-leaderboard")

fails = 0


def check(cond, what):
    global fails
    print(f"{'PASS' if cond else 'FAIL'} {what}")
    if not cond:
        fails += 1


def first_difference(got, want):
    """(line number, ours, theirs) of the first differing line, or None when equal."""
    gl, wl = got.splitlines(), want.splitlines()
    for i, (a, b) in enumerate(zip(gl, wl)):
        if a != b:
            return i + 1, a, b
    if len(gl) != len(wl):
        return min(len(gl), len(wl)) + 1, f"({len(gl)} lines)", f"({len(wl)} lines)"
    return None


def regenerate(out):
    return subprocess.run([sys.executable, TOOL, "--out", out], capture_output=True, text=True)


committed = open(PAGE, encoding="utf-8").read()
with tempfile.TemporaryDirectory() as tmp:
    fresh = os.path.join(tmp, "npu-leaderboard.md")

    # 1. the page is current.
    r = regenerate(fresh)
    check(r.returncode == 0, f"the generator runs (rc={r.returncode})")
    if r.returncode != 0:
        print(r.stdout + r.stderr)
    got = open(fresh, encoding="utf-8").read() if os.path.exists(fresh) else ""
    diff = first_difference(got, committed)
    if diff:
        print(f"  first difference at line {diff[0]}:\n    regenerated: {diff[1][:160]}\n    committed:   {diff[2][:160]}")
    check(diff is None, "docs/npu-leaderboard.md is what the records in bench/npu-leaderboard/ produce "
                        "(run tools/npu_leaderboard.py and commit the result)")

    # 2. the comparison above is a real check: one changed value must be caught, and the line named.
    if got:
        m = re.search(r"\d+\.\d+ tok/s", got)
        marker = "999.99 tok/s"
        broken = got[:m.start()] + marker + got[m.end():] if m else got.replace("## strixhalo", "## strixhalo-x", 1)
        check(broken != got, "the self-check can construct a one-value change to test the comparison with")
        d = first_difference(broken, committed)
        check(d is not None, "a single changed value in the regenerated page is caught")
        if d:
            check(marker in d[1] or "strixhalo-x" in d[1], f"the line it names is the one changed (line {d[0]})")

    # 3. the records validate, and the page has one table row per record (plus the header row).
    rc = subprocess.run([sys.executable, TOOL, "--check"], capture_output=True, text=True)
    check(rc.returncode == 0, f"the record directory validates (rc={rc.returncode}: {rc.stdout.strip()})")
    records = sorted(f for f in os.listdir(RECORDS) if f.endswith(".json"))
    body = [ln for ln in committed.splitlines()
            if ln.startswith("| ") and not ln.startswith("| model |") and not ln.startswith("| --- |")]
    check(len(body) == len(records),
          f"one table row per record ({len(body)} row(s) for {len(records)} record(s))")
    for name in records:  # a record that never reaches the table is a silent hole
        check(f"`{name[:-5]}`" in committed, f"record {name[:-5]} appears in the page")

# 4. --out without a path is refused rather than silently writing the committed page.
bare = subprocess.run([sys.executable, TOOL, "--out"], capture_output=True, text=True)
check(bare.returncode != 0, f"--out with no path is refused (rc={bare.returncode})")

print(f"\n{'all checks passed' if not fails else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
