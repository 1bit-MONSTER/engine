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
"""merge_content_check.py: is the work a PR *added* actually on main?

engine#2652 is why this exists. A merge lands the head sha recorded when the merge was queued, not the branch
tip, so a commit pushed after that point is never merged and nothing says so: the PR shows `merged`, the branch
shows the commit, and `main` does not have it. Four instances in one night, one of which (`WEAK_AFTER_S`,
`--distances`) needed a whole restore PR (#2640) and was only noticed because somebody counted symbols by hand.

**The PR badge is not evidence.** `merged: true` says a merge happened, not that the diff you reviewed is the
diff that landed. The evidence is the added lines being present in `origin/main:<path>`.

This takes the added lines from a PR's diff and asks main for each one. Lines a lost commit introduced are
absent; a wording change produces a visible false positive rather than a silent gap, which is the right way to
be wrong for a check you run before relying on a merge.

    tools/ci/merge_content_check.py 2623                 # every added line of that PR, against origin/main
    tools/ci/merge_content_check.py --branch my-branch   # commits of mine that are NOT on main (the exact check)
    tools/ci/merge_content_check.py --sha 1f87538        # is that commit an ancestor of origin/main?
    tools/ci/merge_content_check.py --self-test          # no network, no git

**The content arm cannot see the failure mode at all -- measured, and it is the reason `--branch` exists.**
`#2623` lost `WEAK_AFTER_S` and `--distances` to a post-merge push, and its *recorded* diff contains **0**
occurrences of `WEAK_AFTER_S`: the diff is frozen at the head that merged, so the lost commits are not in it.
A `--pr` content check is therefore structurally blind to engine#2652 -- what it does find is lines a later PR
deleted, which is a different question. Run across the 59 merges of 11-15/16 it flagged 22 PRs, none of which
was a lost commit.

**So the question is asked about commits, not diffs**: `git rev-list origin/main..<my-branch>` is the set of
work of mine that has not landed, and that set is exactly what to look at before relying on a merge.

**Neither content/sha arm alone is exact, and each is wrong in a different direction**:

* `--sha` asks "is this commit an ancestor of main?" -- exact for a **merge commit**, but a **squash** merge
  never keeps the branch sha, so a squashed head always reads as absent (measured: #2647's `45f0c29e6` is
  not in main though its content is, because it merged as `c94a488`).
* the content arm asks the rule's literal question and survives squash, but it compares the PR's *current*
  diff against *current* main, so a line a later PR refactored reads as missing (measured on #2623, where
  #2645 rewrote the same `GAP` handling the lost commits had touched).

`--pr` therefore reports the merge strategy first: **squash -> believe the content arm; merge commit ->
believe the sha arm.**
"""
from __future__ import annotations

import argparse
import subprocess
import sys

# A line shorter than this, or one that is only punctuation, is too common to be evidence of anything: a lost
# `}` and a landed `}` read the same. Keeping the cut low means comments and one-liners are still checked.
MIN_EVIDENCE = 8
PUNCTUATION = set('{}()[];,.:+-*/=<>!&|"')


def is_evidence(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < MIN_EVIDENCE:
        return False
    return any(ch not in PUNCTUATION and not ch.isspace() for ch in stripped)


def missing(added: dict[str, list[str]], read) -> list[tuple[str, str]]:
    """Added lines (path -> lines) that `read(path)` does not contain, in stable order.

    `read` returns the file's text, or None when the path is not in main at all -- a file that exists only on
    the merged branch is the strongest form of "this never landed".
    """
    out: list[tuple[str, str]] = []
    for path in sorted(added):
        text = read(path)
        for line in added[path]:
            if not is_evidence(line):
                continue
            if text is None or line.strip() not in text:
                out.append((path, line.strip()))
    return out


def parse_patch(patch: str) -> list[str]:
    """The `+` lines of one file's patch, minus the `+++` header."""
    out = []
    for raw in patch.splitlines():
        if raw.startswith("+++") or not raw.startswith("+"):
            continue
        out.append(raw[1:])
    return out


def was_on_main_then_removed(line: str, path: str, ref: str = "origin/main") -> bool:
    """True when `line` existed in main's history and a later commit removed it.

    Without this, every line a subsequent PR legitimately deleted reads as "lost", and the check cries wolf:
    run across the 59 merges of 11-15/16 it flagged 22 PRs, which cannot be right. The discriminator is
    history, and it is the one the hand census in #2640 used -- a symbol main once had is not a symbol main
    never got.
    """
    res = subprocess.run(
        ["git", "log", ref, "--oneline", "--max-count=1", f"-S{line.strip()}", "--", path],
        capture_output=True, text=True)
    return res.returncode == 0 and bool(res.stdout.strip())


def gh_json(args: list[str]):
    import json

    res = subprocess.run(["gh", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"gh {' '.join(args)} failed: {res.stderr.strip()}")
    return json.loads(res.stdout)


def main_at(path: str, ref: str = "origin/main") -> str | None:
    res = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True)
    return res.stdout if res.returncode == 0 else None


def check_branch(branch: str, ref: str) -> int:
    """Commits in `branch` that are NOT in `ref` -- the set at risk from engine#2652.

    Exact for the question "did my work land", which no diff-based check can answer: a commit pushed after the
    merge was queued is absent from the PR's frozen diff, so the only visible trace is that the commit is not an
    ancestor of main.
    """
    res = subprocess.run(["git", "rev-list", f"{ref}..{branch}", "--oneline"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print(f"cannot rev-list {ref}..{branch}")
        return 2
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    if not lines:
        print(f"every commit in {branch} is in {ref}")
        return 0
    print(f"{len(lines)} commit(s) in {branch} are NOT in {ref} -- if that branch's PR already merged, these are"
          f" the ones a queued-head merge dropped (engine#2652):")
    for ln in lines:
        print(f"  {ln}")
    print(f"Branch from {ref} and open a new PR with them.")
    print(f"  A SQUASH merge puts no branch commit in {ref}, so run this BEFORE relying on the merge. Measured: over"
          f" this repo's 198 remote branches, 95 have a merged PR and still carry commits not in {ref} -- all"
          f" squash, so that count is the squash rate, not 95 losses. The sampled merges (#859, #808, #2291,"
          f" #1930) each have one parent.")
    return 1


def check_sha(sha: str, ref: str) -> int:
    """Is `sha` an ancestor of `ref`? Exact answer to "did the commit I reviewed land" (engine#2652)."""
    res = subprocess.run(["git", "merge-base", "--is-ancestor", sha, ref], capture_output=True)
    if res.returncode == 0:
        subject = subprocess.run(
            ["git", "show", "-s", "--format=%s", sha], capture_output=True, text=True).stdout.strip()
        print(f"{sha[:7]} is in {ref}: {subject}")
        return 0
    if res.returncode == 1:
        print(f"{sha[:7]} is NOT in {ref}. A merge lands the head recorded when the merge was queued (engine#2652):"
              f" if this was pushed after that point it is lost. Branch from {ref} and open a new PR.")
        return 1
    print(f"cannot resolve {sha} or {ref} (git merge-base exit {res.returncode})")
    return 2


def run_pr(num: int, repo: str, ref: str) -> int:
    pr = gh_json(["api", f"repos/{repo}/pulls/{num}"])
    files = gh_json(["api", f"repos/{repo}/pulls/{num}/files?per_page=100"])
    added = {f["filename"]: parse_patch(f.get("patch", "")) for f in files if f.get("patch")}

    print(
        f"#{num} {pr['state']} merged={pr['merged']} merge_commit={(pr.get('merge_commit_sha') or '')[:7]}"
        f" -- {len(added)} file(s) with a patch, checked against {ref}"
    )
    merge_sha = pr.get("merge_commit_sha") or ""
    strategy = "unknown"
    if merge_sha:
        parents = subprocess.run(["git", "rev-list", "--parents", "-n1", merge_sha],
                                 capture_output=True, text=True).stdout.split()
        strategy = "merge commit" if len(parents) >= 3 else "squash"
    head = pr["head"]["sha"]
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", head, ref], capture_output=True).returncode == 0
    print(f"  strategy={strategy}; reviewed head {head[:7]} ancestor-of-{ref}={anc}"
          f" ({'expected' if strategy != 'merge commit' or anc else 'LOST HEAD'})")
    if strategy == "squash":
        print("  squash merge: the branch sha never lands, so the sha arm cannot answer -- content arm is the authority")

    absent = missing(added, lambda p: main_at(p, ref))
    lost = [(path, line) for path, line in absent if not was_on_main_then_removed(line, path, ref)]
    removed = [(path, line) for path, line in absent if was_on_main_then_removed(line, path, ref)]
    if removed:
        print(f"  {len(removed)} added line(s) are absent but were on {ref} and later removed by a commit"
              f" -- a later PR deleted them, not a lost merge")
    bad = lost
    if not bad:
        print(f"all {sum(1 for v in added.values() for ln in v if is_evidence(ln))} evidence line(s) are"
              f" accounted for on {ref} (present, or removed by a later commit)")
        return 0

    by_path: dict[str, list[str]] = {}
    for path, line in bad:
        by_path.setdefault(path, []).append(line)
    for path, lines in by_path.items():
        print(f"  MISSING from {ref}: {path} ({len(lines)} line(s))")
        for line in lines[:3]:
            print(f"    {line[:100]}")
    print(
        f"{len(bad)} added line(s) are not on {ref}. If this PR is `merged`, a commit pushed after the merge was"
        f" queued is lost (engine#2652): branch from {ref} and open a new PR with it."
    )
    return 1


def self_test() -> int:
    """Arms: a landed line, a lost line, a lost file, and the noise the check must ignore."""
    cases = [
        (
            "a landed line is not reported",
            {"a.py": ["def landed_helper(x):"]},
            lambda p: "prefix\ndef landed_helper(x):\n    return x\n",
            [],
        ),
        (
            "a lost line is reported",
            {"a.py": ["def lost_helper(x):"]},
            lambda p: "def other_helper(x):\n    return x\n",
            [("a.py", "def lost_helper(x):")],
        ),
        (
            "a file that exists only on the branch is reported",
            {"only_on_branch.py": ["def something_real(x):"]},
            lambda p: None,
            [("only_on_branch.py", "def something_real(x):")],
        ),
        (
            "punctuation and short lines are not evidence",
            {"a.py": ["}", "    }", "x = 1", "        )"]},
            lambda p: "",
            [],
        ),
        (
            "a comment long enough to identify a commit is evidence",
            {"a.py": ["# WEAK_AFTER_S is the measured 5 s cut"]},
            lambda p: "",
            [("a.py", "# WEAK_AFTER_S is the measured 5 s cut")],
        ),
    ]
    failures = 0
    for name, added, reader, want in cases:
        got = missing(added, reader)
        if got != want:
            print(f"FAIL {name}\n  want {want}\n  got  {got}")
            failures += 1
        else:
            print(f"ok   {name}")
    # parse_patch must not treat the +++ header as an added line, or every file reports its own path as missing.
    got = parse_patch("@@ -1 +1 @@\n--- a/x\n+++ b/x\n+def real_line(a):\n context\n")
    if got != ["def real_line(a):"]:
        print(f"FAIL parse_patch drops the +++ header and keeps + lines\n  got {got}")
        failures += 1
    else:
        print("ok   parse_patch drops the +++ header and keeps + lines")
    print(f"merge_content_check: {failures} failure(s)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pr", nargs="?", type=int, help="PR number to check")
    ap.add_argument("--repo", default="1bit-MONSTER/engine")
    ap.add_argument("--ref", default="origin/main", help="ref to check against (default origin/main)")
    ap.add_argument("--branch", help="list commits in this branch that are NOT in --ref (the engine#2652 set)")
    ap.add_argument("--sha", help="check this commit is an ancestor of --ref (exact for a merge commit)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.branch:
        return check_branch(args.branch, args.ref)
    if args.sha:
        return check_sha(args.sha, args.ref)
    if args.pr is None:
        ap.error("pass a PR number, or --self-test")
    return run_pr(args.pr, args.repo, args.ref)


if __name__ == "__main__":
    sys.exit(main())
