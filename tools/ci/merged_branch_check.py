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
"""merged_branch_check.py — which recently merged PRs still have commits on their branch?

engine#2652: a PR merges at its REVIEWED head, so anything pushed afterwards is silently lost
(four cases on 2026-11-15/16). The PR object cannot tell you: `head.sha` is FROZEN at merge time,
which is why a merged_at-vs-commit-date test is useless -- `#2623` and `#2625` both report a head
OLDER than their merge while both lost commits.

But the freeze is the reference if the branch survived. For a merged PR:

    frozen head (gh pr view --json headRefOid)   vs   live branch tip (git ls-remote)

A tip that differs from the frozen head means commits exist on the branch that the merge did not
carry. That is a CANDIDATE, not a verdict -- the author may have recovered them in a follow-up PR
(the fleet's own practice: #2623 -> #2640, #2625 -> #2641) -- so each candidate is reported for the
content grep engine#2652 prescribes:

    git show origin/main:<path> | grep '<marker from the last commit>'

THREE outcomes, and the third is why this is worth running:

    same          nothing was pushed after the merge
    ahead         commits after the merge: grep main for a marker from each, or find the follow-up PR
    unverifiable  the branch is gone, so only the content grep can answer; listed, never passed

Usage:
    tools/ci/merged_branch_check.py --check --since-hours 24 [--repo OWNER/NAME]
    tools/ci/merged_branch_check.py --self-test

Exit: 0 no candidates; 1 at least one `ahead`; 2 usage, or the merge list could not be read.
`unverifiable` does not fail the run -- this tool cannot check those, and pretending otherwise is
the failure mode it exists to name.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys


def git(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], capture_output=True, text=True)
    return p.returncode, (p.stdout if p.returncode == 0 else p.stderr)


def gh_json(*args: str) -> object | None:
    p = subprocess.run(["gh", *args], capture_output=True, text=True)
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return None


def heads_map(remote: str = "origin") -> dict[str, str]:
    """{branch: tip-sha} for every head on the remote, in ONE ls-remote call."""
    rc, out = git("ls-remote", "--heads", remote)
    if rc != 0:
        return {}
    m: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            m[parts[1][len("refs/heads/"):]] = parts[0]
    return m


def classify(frozen: str, tip: str | None) -> str:
    """The three outcomes, pure so the self-test drives every case."""
    if tip is None:
        return "unverifiable"
    return "same" if tip == frozen else "ahead"


def check(repo: str, since_hours: float, heads: dict[str, str]) -> int:
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=since_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = gh_json("pr", "list", "-R", repo, "--state", "merged", "--limit", "100",
                   "--search", f"merged:>={since}", "--json", "number,headRefOid,headRefName,mergedAt,title")
    if rows is None:
        print(f"merged_branch_check: cannot read the merged PRs of {repo} (gh pr list failed)", file=sys.stderr)
        return 2
    ahead = unver = 0
    for pr in rows:
        state = classify(pr["headRefOid"], heads.get(pr["headRefName"]))
        if state == "same":
            continue
        if state == "ahead":
            ahead += 1
            print(f"  AHEAD        #{pr['number']} {pr['headRefName']}: branch tip {heads[pr['headRefName']][:9]} != "
                  f"merged head {pr['headRefOid'][:9]} -- grep main for a marker from each commit after the merge")
            print(f"               {pr['title'][:96]}")
        else:
            unver += 1
            print(f"  UNVERIFIABLE #{pr['number']} {pr['headRefName']}: branch deleted, so only the content grep can answer")
    print(f"merged_branch_check: {len(rows)} merged PR(s) in the last {since_hours:g}h; "
          f"{ahead} ahead, {unver} unverifiable, {len(rows) - ahead - unver} same")
    return 1 if ahead else 0


def self_test() -> int:
    bad = 0

    def chk(cond: bool, msg: str) -> None:
        nonlocal bad
        if not cond:
            bad += 1
        print(("ok   " if cond else "FAIL ") + msg)

    frozen = "0193a29e5a" + "0" * 30
    chk(classify(frozen, frozen) == "same", "a tip equal to the frozen head is 'same'")
    chk(classify(frozen, "f7700ba0a" + "0" * 30) == "ahead",
        "a tip that differs from the frozen head is 'ahead' (#2623: 0193a29e5 -> f7700ba0a)")
    chk(classify(frozen, None) == "unverifiable",
        "a deleted branch is 'unverifiable', never passed (#2625's branch was gone)")

    # heads_map parses one ls-remote listing, and ignores anything that is not refs/heads/
    sample = ("0193a29e5a\trefs/heads/merged\n"
              "f7700ba0a\trefs/heads/ahead\n"
              "deadbeef1\ttags/v1\n")
    import io
    import unittest.mock as mock
    with mock.patch("subprocess.run", return_value=__import__("types").SimpleNamespace(
            returncode=0, stdout=sample, stderr="")):
        m = heads_map()
    chk(m == {"merged": "0193a29e5a", "ahead": "f7700ba0a"}, f"heads_map parses refs/heads only ({m})")

    print("merged_branch_check --self-test:", "ok" if bad == 0 else f"{bad} failure(s)")
    return 0 if bad == 0 else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--repo", default="1bit-MONSTER/engine")
    ap.add_argument("--since-hours", type=float, default=24.0)
    ap.add_argument("--heads-from", help="a file of `<sha>\\trefs/heads/<name>` lines, instead of asking the remote")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.check:
        print("merged_branch_check: one of --check or --self-test is required", file=sys.stderr)
        return 2
    if a.heads_from:
        heads = {}
        with open(a.heads_from, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                    heads[parts[1][len("refs/heads/"):]] = parts[0]
    else:
        heads = heads_map()
    return check(a.repo, a.since_hours, heads)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
