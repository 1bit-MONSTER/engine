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
#
# tests/exit_code_table_test.py [ROOT] — docs/ci-runners.md's "Exit codes on the GPU and NPU path" table against the
# scripts it names (review of engine#2539: the table is a contract and prose rots). Every "<script> exits it" or
# "<script> sets it" in a row's scripts cell must be backed by `exit <rc>` or `rc=<rc>` in that script, and every row
# must name an emitter (a script that exits, sets or passes the code, or "any payload" / "all" / "the payload").
# A code that is computed rather than literal is written that way in the row ("computed by ...") and is not grepped.
# The 2 row's quoted stderr phrases (review of engine#2544: a lane greps its log for them) must each be printed by one of
# the scripts, verbatim.
import re
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
SCRIPTS = {"gpu-hold": "scripts/gpu-hold", "npu-hold": "scripts/npu-hold", "hw-gate.sh": "scripts/hw-gate.sh",
           "window-run.sh": "scripts/window-run.sh", "coder-llm-ticketed": "scripts/coder-llm-ticketed"}

doc = (root / "docs/ci-runners.md").read_text()
m = re.search(r"^## Exit codes on the GPU and NPU path.*?(?=^## )", doc, re.S | re.M)
if not m:
    print("FAIL: no 'Exit codes on the GPU and NPU path' section in docs/ci-runners.md")
    sys.exit(1)
rows = [l for l in m.group(0).splitlines() if re.match(r"^\| \**\d+\** \|", l)]
fail, checked = 0, 0
if len(rows) < 10:
    print(f"FAIL: the table has {len(rows)} rows, expected at least 10 (did the row format change?)")
    fail = 1
for row in rows:
    cells = [c.strip() for c in row.strip().strip("|").split("|")]
    rc, where = cells[0].strip("*"), cells[2]
    named = False
    for seg in where.split(";"):
        if re.search(r"pass(es)? it through|relabels|any payload|^all$|the payload|computed by", seg):
            named = True
        if "computed by" in seg:   # a code computed at run time (window-run's 69) has no literal to grep
            continue
        verb = re.search(r"\b(exits?|sets) (it\b|\()", seg)
        if not verb:
            continue
        named = True
        for name in [n for n in SCRIPTS if re.search(r"(?<![\w-])" + re.escape(n) + r"(?![\w-])", seg[:verb.start()])]:
            text = (root / SCRIPTS[name]).read_text()
            checked += 1
            if not re.search(rf"\bexit {rc}\b|\brc={rc}\b", text):
                print(f"FAIL: rc {rc}: the table says {name} {verb.group(1)} it, but {SCRIPTS[name]} has no 'exit {rc}' or 'rc={rc}'")
                fail = 1
    if not named:
        print(f"FAIL: rc {rc}: the scripts cell names no emitter ({where[:80]})")
        fail = 1
two = [r for r in rows if re.match(r"^\| \**2\** \|", r)]
phrases = re.findall(r"`([^`]+)`", two[0].split("|")[2]) if two else []
if len(phrases) < 6:
    print(f"FAIL: the 2 row quotes {len(phrases)} stderr phrases, expected at least 6 (one per cause)")
    fail = 1
alltext = "".join((root / f).read_text() for f in SCRIPTS.values())
for ph in phrases:
    if ph not in alltext:
        print(f"FAIL: rc 2: the stderr phrase '{ph}' is not printed by any of the scripts")
        fail = 1
if checked < 8:
    print(f"FAIL: only {checked} (code, script) claims were checked; the parser no longer reads the table")
    fail = 1
if not fail:
    print(f"exit_code_table: ok ({len(rows)} rows, {checked} (code, script) claims backed by the scripts; {len(phrases)} rc 2 stderr phrases found verbatim)")
sys.exit(fail)
