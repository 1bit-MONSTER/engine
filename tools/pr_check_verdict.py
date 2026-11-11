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
"""pr_check_verdict.py -- say what a PR's "red" actually is, per check, from its conclusion.

Why this exists (engine#2077): `pr_agent` read red on 32 of 40 pull_request runs, and the conclusion
was CANCELLED -- every push cancels the queued review, so a check that never ran was being read as a
check that failed. The issue's own closing advice: "the first thing to read from now on is the check's
conclusion, not its colour: CANCELLED and FAILURE need different answers." This is that reading, made
mechanical, so a lane does not have to remember which red is which.

A conclusion is classified four ways, and the fourth is the point:

  FAILED     -- ran and failed. A verdict about the diff: fix it.
  CANCELLED  -- did not run. Says nothing about the diff; re-run (or wait for the newer run).
  SKIPPED    -- deliberately not run (a gate that does not apply). Says nothing about the diff.
  neutral    -- SUCCESS, or a run still in progress (its conclusion is empty).

`mergeable` is yes only when every check is neutral. `unknown` is reported distinctly from `no`, because
"a check was cancelled" and "a check failed" must not both print as "no" -- that conflation is what made
32 cancelled reviews look like 32 rejections. The exit code keeps the three apart too: 0 yes, 1 no,
3 unknown, so a caller cannot silently fold "did not run" into "failed".

usage: pr_check_verdict.py <rollup.json>            classify a saved `statusCheckRollup` document
       pr_check_verdict.py --self-test               fixtures only, no network
"""
import json
import sys

FAILED = "FAILED"
CANCELLED = "CANCELLED"
SKIPPED = "SKIPPED"
NEUTRAL = "neutral"

CLASS_OF = {
    "FAILURE": FAILED,
    "TIMED_OUT": FAILED,
    "STARTUP_FAILURE": FAILED,
    "CANCELLED": CANCELLED,
    "SKIPPED": SKIPPED,
    "NEUTRAL": NEUTRAL,
    "SUCCESS": NEUTRAL,
    "": NEUTRAL,          # still running: no conclusion yet, so nothing to say about the diff
}


def classify(conclusion):
    """A conclusion string -> one of the four classes. Unknown conclusions are FAILED, never neutral:
    a conclusion this tool does not know must not be read as a pass."""
    return CLASS_OF.get((conclusion or "").upper(), FAILED if conclusion else NEUTRAL)


def read_rollup(doc):
    """[(name, status, conclusion, class)] from a statusCheckRollup document (CheckRun and StatusContext)."""
    rows = []
    for c in doc or []:
        name = c.get("name") or c.get("context") or "?"
        if c.get("__typename") == "StatusContext":
            rows.append((name, c.get("state", ""), c.get("state", ""), classify(c.get("state"))))
        else:
            rows.append((name, c.get("status", ""), c.get("conclusion") or "", classify(c.get("conclusion"))))
    return rows


def verdict(rows):
    """(mergeable, counts) where mergeable is 'yes', 'no', or 'unknown'."""
    counts = {FAILED: 0, CANCELLED: 0, SKIPPED: 0, NEUTRAL: 0}
    for _, _, _, cls in rows:
        counts[cls] += 1
    if counts[FAILED]:
        return "no", counts
    if counts[CANCELLED]:
        return "unknown", counts
    return "yes", counts


def report(rows):
    out = []
    for name, status, conclusion, cls in rows:
        if cls == NEUTRAL and status.upper() == "IN_PROGRESS":
            out.append(f"  {cls:<9} {name} (still running)")
        elif cls != NEUTRAL:
            out.append(f"  {cls:<9} {name} (conclusion {conclusion or 'none'})")
    return "\n".join(out)


def self_test():
    cases = [
        # (rollup, expected verdict, why)
        ([{"name": "build", "status": "COMPLETED", "conclusion": "SUCCESS"}], "yes", "one green check"),
        ([{"name": "build", "status": "COMPLETED", "conclusion": "FAILURE"}], "no", "a failure is a no"),
        ([{"name": "pr_agent", "status": "COMPLETED", "conclusion": "CANCELLED"}], "unknown",
         "a cancelled check never ran -- not a verdict about the diff"),
        ([{"name": "a", "status": "COMPLETED", "conclusion": "CANCELLED"},
          {"name": "b", "status": "COMPLETED", "conclusion": "FAILURE"}], "no", "failure outranks cancelled"),
        ([{"name": "hw_gate", "status": "COMPLETED", "conclusion": "SKIPPED"}], "yes", "a deliberate skip is not a red"),
        ([{"name": "x", "status": "IN_PROGRESS", "conclusion": None}], "yes", "running with no conclusion yet"),
        ([{"name": "x", "status": "COMPLETED", "conclusion": "TIMED_OUT"}], "no", "a timeout is a real failure"),
        ([{"name": "x", "status": "COMPLETED", "conclusion": "SOMETHING_NEW"}], "no",
         "an unknown conclusion must not read as a pass"),
        ([], "yes", "no checks at all"),
    ]
    bad = 0
    for doc, want, why in cases:
        got, _ = verdict(read_rollup(doc))
        if got != want:
            print(f"FAIL self-test: {why}: got {got}, wanted {want}")
            bad += 1
        else:
            print(f"ok   {why}")
    # the classification the issue asks for, stated as its own arm
    if classify("CANCELLED") == CANCELLED and classify("FAILURE") == FAILED:
        print("ok   CANCELLED and FAILURE are different classes (engine#2077)")
    else:
        print("FAIL CANCELLED and FAILURE must be different classes")
        bad += 1
    print(("ok   " if not bad else "FAIL ") + f"{len(cases) + 1} arms, {bad} failure(s)")
    return 1 if bad else 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    try:
        with open(argv[1], encoding="utf-8") as f:
            doc = json.load(f)
    except OSError as e:
        print(f"pr_check_verdict: cannot read {argv[1]}: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"pr_check_verdict: {argv[1]} is not JSON: {e}", file=sys.stderr)
        return 2
    rows = read_rollup(doc)
    v, counts = verdict(rows)
    body = report(rows)
    if body:
        print(body)
    print(f"mergeable: {v}  ({counts[FAILED]} failed, {counts[CANCELLED]} cancelled, "
          f"{counts[SKIPPED]} skipped, {counts[NEUTRAL]} ok)")
    # Three exit codes, not two: a caller that treats "unknown" as "no" is the defect this tool exists to
    # remove (engine#2077 -- 32 cancelled reviews read as 32 rejections), so the shell must be able to tell
    # "a check failed" from "a check did not run".
    return {"yes": 0, "no": 1, "unknown": 3}[v]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
