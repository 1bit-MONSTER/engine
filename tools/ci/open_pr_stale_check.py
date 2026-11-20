#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""open_pr_stale_check.py -- run the stale-branch judgement over every open PR (engine#2992).

`branch_stale_check.py` already decides whether merging a branch would lose a path it never changed, and it does it
from the merge-base rather than the two-dot endpoint. What it cannot do is find the branches: it judges whichever ref
you hand it, so today it only runs when a human runs it (`scripts/pre-pr-checks.sh`, local and opt-in).

Measured 2026-11-20 across all six repos: seven of seven engine PRs reviewed carried a stale base, and one displayed
as +596/-0 while a merge would have deleted 1475 lines. A check that needs a human to remember is not a guard.

This enumerates the open PRs and asks the existing tool about each one. It adds no diff logic of its own: the
merge-base reasoning, its docstring and its arms stay in one place.

The signal is DELETIONS, not distance. A branch `behind 1` with 55 deletions is the ordinary "someone merged after
you pushed" case, so the deletion count is printed next to `behind` precisely so nobody reads distance as the result.

Exit: 0 clean, 1 at least one PR would lose work, 2 cannot tell (no `gh`, no network, a ref that will not read).
"""
import argparse
import json
import os
import subprocess
import sys

TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "branch_stale_check.py")


def run(cmd, cwd=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def open_prs(repo_dir):
    """[(number, base, head)] or None when gh cannot answer."""
    result = run(["gh", "pr", "list", "--state", "open", "--limit", "100",
                  "--json", "number,baseRefName,headRefName"], cwd=repo_dir)
    if result.returncode != 0:
        return None
    try:
        prs = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return [(int(p["number"]), p["baseRefName"], p["headRefName"]) for p in prs]


def deletions(repo_dir, base, head):
    """Lines a two-dot diff would show as deleted, i.e. what the PR's endpoint view hides. None when unreadable."""
    result = run(["git", "diff", "--numstat", f"origin/{base}", f"origin/{head}"], cwd=repo_dir)
    if result.returncode != 0:
        return None
    total = 0
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[1].isdigit():
            total += int(fields[1])
    return total


def behind(repo_dir, base, head):
    result = run(["git", "rev-list", "--count", f"origin/{head}..origin/{base}"], cwd=repo_dir)
    return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip().isdigit() else None


def verdict(rows):
    """(exit status, failures) from [(number, side, text)] where side is 'fail' or 'cannot_tell'."""
    failures = [row for row in rows if row[1] == "fail"]
    cannot = [row for row in rows if row[1] == "cannot_tell"]
    if failures:
        return 1, failures
    if cannot:
        return 2, cannot
    return 0, []


def check(repo_dir, limit=25):
    prs = open_prs(repo_dir)
    if prs is None:
        print("open_pr_stale_check: `gh pr list` failed - cannot tell, and that is not a pass", file=sys.stderr)
        return 2
    run(["git", "fetch", "--quiet", "origin", "+refs/heads/*:refs/remotes/origin/*"], cwd=repo_dir)
    rows = []
    for number, base, head in prs:
        delta = behind(repo_dir, base, head)
        gone = deletions(repo_dir, base, head)
        judged = run([sys.executable, TOOL, "--base", f"origin/{base}", "--head", f"origin/{head}",
                      "--quiet", "--limit", str(limit)], cwd=repo_dir)
        if judged.returncode == 1:
            print(f"PR #{number}: behind {delta} of origin/{base}, endpoint deletions {gone}")
            print(judged.stdout.rstrip())
            rows.append((number, "fail", judged.stdout))
        elif judged.returncode != 0:
            print(f"open_pr_stale_check: PR #{number} ({head}): cannot tell\n{judged.stderr.rstrip()}",
                  file=sys.stderr)
            rows.append((number, "cannot_tell", judged.stderr))
        elif gone:
            # The endpoint diff shows deletions but the merge keeps them: say so, because that is the case a reader
            # of GitHub's "+596/-0" cannot see, and it is the number this issue says to watch.
            print(f"PR #{number}: behind {delta} of origin/{base}, endpoint deletions {gone} - nothing lost")
    status, reported = verdict(rows)
    if status == 0:
        print(f"open_pr_stale_check: {len(prs)} open PR(s), none would lose a path its branch never changed")
    return status


def self_test():
    arms = [
        ("clean", [(1, "clean", "")], 0),
        ("one losing PR fails the run", [(1, "clean", ""), (2, "fail", "x")], 1),
        ("cannot tell is not a pass", [(1, "cannot_tell", "")], 2),
        ("cannot tell does not mask a failure", [(1, "cannot_tell", ""), (2, "fail", "x")], 1),
    ]
    bad = 0
    for name, rows, expect in arms:
        got, _ = verdict(rows)
        ok = got == expect
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {name}: rc {got} (expect {expect})")
    return 1 if bad else 0


def main():
    parser = argparse.ArgumentParser(description="Judge every open PR for a stale base (engine#2992).")
    parser.add_argument("--repo", default=".", help="checkout to read PRs from (default: .)")
    parser.add_argument("--limit", type=int, default=25, help="how many files branch_stale_check names per PR")
    parser.add_argument("--self-test", action="store_true", help="run the aggregation arms and exit")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    return check(os.path.abspath(args.repo), args.limit)


if __name__ == "__main__":
    sys.exit(main())
