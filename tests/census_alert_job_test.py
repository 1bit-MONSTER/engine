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

"""census_alert_job_test.py -- what the census alert job does with a QUIET run.

engine#2846: the workflow computed an `alert` output and nothing read it, so the coverage body was assembled for
every scheduled run whose watch job succeeded. A quiet run therefore filed a "coverage breach" issue whose excerpt
was the per-row stream and advisory `->` lines — no finding anywhere in it. The issue that reported this was itself
the evidence: 13:42Z on 2026-11-17, zero `!!` rows, filed.

This runs the job's own `run:` block (extracted from the workflow, not a copy) with a stub `gh` on PATH, and asserts
what it files in each of the three states:

    watch succeeded, alert=false   -> nothing at all          (the regression)
    watch succeeded, alert=true    -> one coverage issue
    watch failed                   -> one failure issue

Exit: 0 all three hold, 1 one does not, 77 the block could not be found.
"""

import base64
import os
import pathlib
import re
import subprocess
import sys
import tempfile

WORKFLOW = pathlib.Path(".github/workflows/census-watch.yml")

GH_STUB = """#!/bin/bash
# Log every call, then answer the three shapes `post()` and the label line use.
echo "$@" >> "$GH_CALLS"
case "$*" in
  "label create"*) exit 0 ;;
  "issue list"*)   [ -n "${GH_OPEN:-}" ] && echo "${GH_OPEN}" || true; exit 0 ;;
  "issue view"*)   echo "${GH_SEEN:-0}"; exit 0 ;;
  "issue create"*) echo "created"; exit 0 ;;
  "issue comment"*) echo "commented"; exit 0 ;;
esac
exit 0
"""


def block():
    text = WORKFLOW.read_text()
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.strip().startswith("- name: File the breach")), None)
    if start is None:
        return None
    run = next((i for i in range(start, len(lines)) if lines[i].strip().startswith("run:")), None)
    if run is None:
        return None
    indent = len(lines[run]) - len(lines[run].lstrip())
    out = []
    for ln in lines[run + 1:]:
        if ln.strip():
            if len(ln) - len(ln.lstrip()) <= indent:
                break
            out.append(ln[indent + 2:] if len(ln) > indent + 2 else "")
        else:
            out.append("")
    return "\n".join(out)


def run_case(env_extra):
    with tempfile.TemporaryDirectory() as d:
        stub = pathlib.Path(d) / "gh"
        stub.write_text(GH_STUB)
        stub.chmod(0o755)
        calls = pathlib.Path(d) / "calls"
        calls.write_text("")
        env = dict(os.environ, PATH=f"{d}:{os.environ['PATH']}", GH_CALLS=str(calls),
                   GH_TOKEN="x", GH_REPO="o/r", RUN_URL="https://example/run",
                   LINES=base64.b64encode(b"  !! UNCOVERED X (new class): 1 model(s), e.g. a/b\n").decode(),
                   FAILURE="2|" + base64.b64encode(b"boom").decode(), SIMULATED="")
        env.update(env_extra)
        script = pathlib.Path(d) / "step.sh"
        script.write_text(block())
        p = subprocess.run(["bash", "-e", str(script)], capture_output=True, text=True, env=env)
        return calls.read_text(), p


def main():
    if not WORKFLOW.is_file():
        print(f"census_alert_job: skip: no {WORKFLOW}", file=sys.stderr)
        return 77
    if block() is None:
        print("census_alert_job: cannot find the alert job's `run:` block: the step moved, so this arm is looking at "
              "nothing", file=sys.stderr)
        return 77
    fails = 0

    def check(ok, what, detail=""):
        nonlocal fails
        print(f"{'PASS' if ok else 'FAIL'}  {what}" + (f"  [{detail}]" if detail and not ok else ""))
        fails += 0 if ok else 1

    calls, p = run_case({"WATCH_RESULT": "success", "ALERT": "false"})
    check(p.returncode == 0, "a quiet run exits 0", p.stderr[-200:])
    check("issue create" not in calls and "issue comment" not in calls,
          "a QUIET run files nothing (engine#2846)", calls[-200:])

    calls, p = run_case({"WATCH_RESULT": "success", "ALERT": "true"})
    check(p.returncode == 0, "an alerting run exits 0", p.stderr[-200:])
    check(calls.count("issue create") == 1, "an alerting run files exactly one coverage issue", calls[-200:])

    calls, p = run_case({"WATCH_RESULT": "failure", "ALERT": "false"})
    check(p.returncode == 0, "a failed watch exits 0 from this step", p.stderr[-200:])
    check(calls.count("issue create") == 1, "a failed watch still files (the alert output is false there)", calls[-200:])

    print(f"census_alert_job: {'ok' if not fails else f'{fails} failure(s)'}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
