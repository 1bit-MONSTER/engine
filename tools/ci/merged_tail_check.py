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
"""merged_tail_check.py -- commits that landed on a branch after its PR merged.

The merge loop merges the REVIEWED head (`--match-head-commit`), and that is correct: the gate
approves a head, so the merge must be that head. The loss is somewhere else, and it is a race the
reviewed-head rule cannot cover:

    a push lands on the branch AFTER the merge. The PR is already MERGED, so nothing will merge
    that branch again, and the new commit is on a branch nobody is watching.

Measured on engine#2623 (merged 0193a29 with 3 of the 5 commits that were pushed): `WEAK_AFTER_S`
and `def distances` were on the branch and are absent from `main`. The PR badge said merged, the
gate said READY, and the content was not there. #2640 recovered it.

**Why a content check and not a sha check.** A squash merge rewrites every sha, so "is this commit
in the base" is unanswerable by sha: the commit is never an ancestor and its patch-id is not the
squashed patch-id either (`git cherry` reports every commit of every squashed PR as absent). So the
question is answered the way #2640 answered it by hand -- **do the lines this commit adds exist in
the base's copy of the file it touched** -- with one row per commit and the missing lines named.

Classes reported:

  LOST          most of the commit's added lines are not in the base: it needs a recovering PR
  PARTIAL       a few of them are gone because main edited those lines since: not a loss
  IN-BASE       every added line is already there (the content arrived another way): not a loss
  UNVERIFIABLE  the commit adds no significant line (deletion-only): neither of the above, a gap
  GONE          the branch is deleted, so the tail cannot be examined at all: a gap, never a pass

Exit 0 no loss and nothing unexamined, 1 a LOST commit, 2 a ref could not be read or something was
unverifiable -- **a scan that could not look is not a scan that found nothing.**
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

# A line worth looking for: long enough not to be punctuation, and not a brace or a lone keyword.
# Below this the containment test answers "yes" for `}`, `);` and `#endif` in every file on earth.
SIGNIFICANT = 8

# A commit is LOST when at least this share of the lines it adds is absent from the base. The gap is
# wide, so the number is not delicate: measured on engine#2623, whose five commits split 89 % and
# 100 % absent (the two that were lost) against 4 %, 25 % and 30 % (the three that landed and whose
# lines main later edited). Anything above the landed group and below the lost group works; a
# threshold of "any line absent" would have called all five lost, and "all lines absent" would have
# missed #2623's first lost commit at 89 %.
LOST_FRACTION = 0.5


# The denominator. A scan that finds nothing and a scan that read nothing print the same
# "0 findings" without it -- the absent-denominator failure this fleet fixed four times tonight
# (#2624, #2627, #2645, #2656).
examined = [0]


def git(repo, *args, check=False):
    p = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p


def gh(*args):
    p = subprocess.run(["gh", *args], capture_output=True, text=True)
    return p


def significant_lines(text):
    """The lines of `text` worth asking the base about, deduped, in order."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) < SIGNIFICANT:
            continue
        if s in ("{", "}", ");", "};", "#endif", "#else"):
            continue
        if s not in out:
            out.append(s)
    return out


def same_head(frozen, tip):
    """True when the live tip is the head the PR was merged at.

    `frozen` is `headRefOid`, recorded at merge time and never moved afterwards. An empty or
    unreadable frozen head must answer False -- "I do not know" may not skip a branch.
    """
    frozen, tip = (frozen or "").strip(), (tip or "").strip()
    if len(frozen) < 7 or not tip:
        return False
    return tip.lower().startswith(frozen.lower()[:7]) or frozen.lower().startswith(tip.lower()[:7])


def branch_mode(repo, branch, base_ref):
    """List the commits on `branch` that are not on `base_ref` -- the recovery workflow.

    A PR-list scan answers "is anything lost?"; this answers "what of MY branch never landed?",
    which is what a lane needs immediately after it notices. (merge_content_check, #2655.)
    """
    if git(repo, "rev-parse", "--verify", "-q", branch).returncode != 0:
        return None
    tail = merge_commits(repo, branch, base_ref)
    if tail is None:
        return None
    for sha, subject in tail:
        kind, detail = classify_commit(repo, sha, base_ref)
        print(f"{kind:12s} {sha} {subject[:60]}\n{'':12s} {detail}")
    print(f"merged_tail_check: {len(tail)} commit(s) on {branch} are not on {base_ref}")
    return tail


def merge_commits(repo, ref, base_ref):
    """Commits reachable from `ref` and not from `base_ref` -- the branch's tail."""
    r = git(repo, "rev-list", "--format=%h %s", f"{base_ref}..{ref}")
    if r.returncode != 0:
        return None
    lines = [ln for ln in r.stdout.splitlines() if ln and not ln.startswith("commit ")]
    return [ln.split(" ", 1) for ln in lines]


def classify_commit(repo, sha, base_ref):
    """LOST / IN-BASE / UNVERIFIABLE for one commit, with the missing lines named."""
    names = git(repo, "diff-tree", "-r", "--no-commit-id", "--name-only", sha)
    if names.returncode != 0:
        return "UNVERIFIABLE", "(could not list the commit's files)"
    files = [f for f in names.stdout.splitlines() if f.strip()]
    if not files:
        return "UNVERIFIABLE", "(the commit changes no file)"

    missing_all, checked_all = [], 0
    for f in files:
        d = git(repo, "diff", f"{sha}^..{sha}", "--", f)
        if d.returncode != 0:
            continue
        added = [ln[1:] for ln in d.stdout.splitlines()
                 if ln.startswith("+") and not ln.startswith("+++")]
        want = significant_lines("\n".join(added))
        if not want:
            continue
        checked_all += len(want)
        show = git(repo, "show", f"{base_ref}:{f}")
        base_text = show.stdout if show.returncode == 0 else ""
        for line in want:
            if line not in base_text:
                missing_all.append((f, line))

    if checked_all == 0:
        return "UNVERIFIABLE", "(no significant added line to look for -- deletion only)"
    # A commit can be absent from the base because the base SUPERSEDED it (the branch was rebased or
    # reworked before merging, so its earlier formulation never landed while a later one did). The
    # line counts tell the reader which case they are looking at, and they are already in hand:
    # engine#2504 flags at 83 % absent, yet the base's file is 162 lines against the commit's 149 --
    # the feature landed in a fuller form and recovery would be wrong. Without this the row reads as
    # a loss and costs a lane an hour.
    base_lines = {}
    for f, _ in missing_all[:1] or [(None, None)]:
        if f:
            show = git(repo, "show", f"{base_ref}:{f}")
            base_lines[f] = len(show.stdout.splitlines()) if show.returncode == 0 else 0
    frac = len(missing_all) / checked_all
    where = "; ".join(f"{f}: {ln[:60]}" for f, ln in missing_all[:3])
    more = "" if len(missing_all) <= 3 else f" (+{len(missing_all) - 3} more)"
    detail = (f"{len(missing_all)} of {checked_all} added line(s) absent "
              f"({frac:.0%}; LOST at {LOST_FRACTION:.0%})")
    if frac >= LOST_FRACTION:
        sizes = ""
        for f in {f for f, _ in missing_all}:
            show = git(repo, "show", f"{sha}:{f}")
            mine = len(show.stdout.splitlines()) if show.returncode == 0 else 0
            theirs = base_lines.get(f)
            if theirs is None:
                b = git(repo, "show", f"{base_ref}:{f}")
                theirs = len(b.stdout.splitlines()) if b.returncode == 0 else 0
            ctx = "base SUPERSEDED it (base is longer)" if theirs > mine else "base is shorter"
            sizes += f"; {f}: this commit {mine} lines, base {theirs} lines ({ctx})"
        return "LOST", f"{detail} -- {where}{more}{sizes}"
    if missing_all:
        return "PARTIAL", f"{detail} -- main edited these lines; the commit is on the base"
    return "IN-BASE", f"all {checked_all} added line(s) already on the base"


def scan(repo, days, limit, base_url=None, prs=None):
    """(kind, pr, branch, commit, detail) rows, or None when the PR list could not be read.

    A generator was the wrong shape here: `return None` inside one raises StopIteration, so a
    failed `gh` call reached the caller as an empty list -- "no findings" for a query that never
    ran. The --self-test arm below pins it.
    """
    rows = []
    if prs is None:
        # `gh pr list --search` answers `[]` with rc 0 for a repository that does not exist, which
        # is byte-identical to a repository with nothing merged. Measured 2026-11-16:
        #   gh pr list -R <bogus> --state merged --search merged:>=X --json number  ->  []  rc 0
        #   gh repo view <bogus>                                                  ->  rc 1
        # So the repository is resolved explicitly, because an unresolvable one must be a gap.
        v = gh("repo", "view", str(base_url), "--json", "name")
        if v.returncode != 0:
            print(f"merged_tail_check: cannot resolve {base_url} "
                  f"({(v.stderr or v.stdout).strip().splitlines()[0] if (v.stderr or v.stdout).strip() else 'no output'})"
                  f" -- cannot tell, and that is not a pass", file=sys.stderr)
            return None
        since = subprocess.run(
            ["date", "-u", "-d", f"-{days} days", "+%Y-%m-%dT%H:%M:%SZ"],
            capture_output=True, text=True).stdout.strip()
        r = gh("pr", "list", "-R", base_url, "--state", "merged", "--limit", str(limit),
               "--search", f"merged:>={since}",
               "--json", "number,title,headRefName,baseRefName,headRepositoryOwner,headRepository,"
                         "headRefOid")
        # `gh` exits 0 and prints the failure on stdout when the repo does not resolve
        # ("GraphQL: Could not resolve to a Repository with the name ..."), so the exit code cannot
        # tell a real list from an error message. The parsed shape is the only evidence.
        note = (r.stderr or r.stdout).strip().splitlines()
        note = note[0] if note else "(no output)"
        if r.returncode != 0:
            print(f"merged_tail_check: gh pr list failed ({note}) -- cannot tell, and that is "
                  f"not a pass", file=sys.stderr)
            return None
        try:
            prs = json.loads(r.stdout)
        except json.JSONDecodeError:
            print(f"merged_tail_check: gh pr list did not return JSON ({note}) -- cannot tell, "
                  f"and that is not a pass", file=sys.stderr)
            return None
        if not isinstance(prs, list):
            print("merged_tail_check: gh pr list returned no list -- cannot tell, and that is "
                  "not a pass", file=sys.stderr)
            return None

    examined[0] = len(prs)
    for pr in prs:
        owner = (pr.get("headRepositoryOwner") or {}).get("login")
        name = (pr.get("headRepository") or {}).get("name")
        branch = pr.get("headRefName")
        if not (owner and name and branch):
            rows.append(("GONE", pr, branch, None, "the PR record names no head repository or branch"))
            continue
        url = f"https://github.com/{owner}/{name}.git"
        base_ref = f"refs/mtck/base/{pr['baseRefName']}"

        if git(repo, "fetch", "-q", "--no-tags", "origin",
               f"{pr['baseRefName']}:{base_ref}").returncode != 0:
            rows.append(("GAP", pr, branch, None, f"could not fetch base {pr['baseRefName']}"))
            continue

        # Does the branch still exist? A deleted branch hides its tail completely.
        ls = git(repo, "ls-remote", url, f"refs/heads/{branch}")
        if ls.returncode != 0:
            rows.append(("GAP", pr, branch, None, f"could not read {url}"))
            continue
        if not ls.stdout.strip():
            rows.append(("GONE", pr, branch, None, "the branch is deleted: the tail cannot be examined"))
            continue

        # Cheap gate before any fetch. `headRefOid` is frozen at merge time -- measured on #2623,
        # where it is 0193a29e5, a commit that is NOT an ancestor of main while the branch tip is
        # f7700ba0a. So `frozen != tip` means something was pushed (or force-pushed) after the merge
        # and the branch is worth reading; `frozen == tip` means nothing was, and the content check
        # would only re-derive that at the cost of a fetch per PR. (merged_branch_check, #2656.)
        if same_head(pr.get("headRefOid"), ls.stdout.split()[0] if ls.stdout.split() else ""):
            continue  # the branch tip is exactly what merged: nothing was pushed after

        ref = f"refs/mtck/{pr['number']}"
        if git(repo, "fetch", "-q", "--no-tags", url, f"{branch}:{ref}").returncode != 0:
            rows.append(("GAP", pr, branch, None, f"could not fetch {url} {branch}"))
            continue

        tail = merge_commits(repo, ref, base_ref)
        if tail is None:
            rows.append(("GAP", pr, branch, None, "could not list the branch's tail"))
            continue
        if not tail:
            continue  # the branch tip is exactly what merged: the normal case
        for sha, subject in tail:
            kind, detail = classify_commit(repo, sha, base_ref)
            rows.append((kind, pr, branch, (sha, subject), detail))
        git(repo, "update-ref", "-d", ref)
    return rows


def report(repo, days, limit, base_url):
    counts = {"LOST": 0, "PARTIAL": 0, "IN-BASE": 0, "UNVERIFIABLE": 0, "GONE": 0, "GAP": 0}
    rows = scan(repo, days, limit, base_url)
    if rows is None:
        return 2, counts, []
    for kind, pr, branch, commit, detail in rows:
        counts[kind] += 1
        head = f"#{pr['number']} {branch}"
        if commit:
            head = f"{head} {commit[0]} {commit[1][:56]}"
        print(f"{kind:12s} {head}\n{'':12s} {detail}")
    print(f"merged_tail_check: examined {examined[0]} PR(s) merged in the last {days} day(s); "
          f"{examined[0] - counts['GONE']} with a live branch, {counts['GONE']} branch gone")
    print(f"merged_tail_check: {len(rows)} finding(s) over PRs merged in the last {days} day(s) "
          f"-- {counts['LOST']} LOST, {counts['PARTIAL']} partly edited by main, "
          f"{counts['IN-BASE']} already on the base, "
          f"{counts['UNVERIFIABLE']} unverifiable, {counts['GONE']} branch gone, {counts['GAP']} gap")
    if counts["LOST"]:
        print("merged_tail_check: a LOST commit needs a PR that branches from origin/main -- "
              "not a push to the merged branch", file=sys.stderr)
        return 1, counts, rows
    if counts["UNVERIFIABLE"] or counts["GONE"] or counts["GAP"]:
        print("merged_tail_check: something could not be examined; that is not a clean scan",
              file=sys.stderr)
        return 2, counts, rows
    return 0, counts, rows


def self_test():
    import tempfile
    rc = 0

    def chk(want, got, what):
        nonlocal rc
        if want == got:
            print(f"ok   {what}")
        else:
            print(f"FAIL {what}: want {want}, got {got}")
            rc = 1

    def scratch():
        d = tempfile.TemporaryDirectory()
        r = pathlib.Path(d.name)
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=r, check=True)
        subprocess.run(["git", "config", "user.email", "t@e"], cwd=r, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=r, check=True)
        return d, r

    def commit(r, path, text, msg):
        p = r / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        subprocess.run(["git", "add", "-A"], cwd=r, check=True)
        subprocess.run(["git", "commit", "-qm", msg], cwd=r, check=True)
        return git(r, "rev-parse", "HEAD").stdout.strip()

    def branch(r, name, at="HEAD"):
        subprocess.run(["git", "checkout", "-q", "-B", name, at], cwd=r, check=True)

    # The loss: a branch cut, merged at one commit, then pushed to again. main never gets it.
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "c1")
    branch(r, "pr")
    lost_sha = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef WEAK_AFTER_S():\n"
                                    "    return 2\n", "pushed after the merge")
    branch(r, "main", "pr~1")
    chk("LOST", classify_commit(r, lost_sha, "main")[0],
        "a commit pushed after the merge is LOST")
    chk(1, len(merge_commits(r, "pr", "main")), "the branch has a tail of exactly one")
    d.cleanup()

    # The same content, already on the base by another commit: not a loss.
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "c0")
    branch(r, "pr")
    base_commit = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef GAP():\n"
                                       "    return 9\n", "base already has GAP")
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "c0")
    root = git(r, "rev-parse", "HEAD").stdout.strip()
    branch(r, "main")
    same_sha = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef GAP():\n"
                                    "    return 9\n", "main gains GAP")
    branch(r, "pr", root)
    twin = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef GAP():\n"
                                "    return 9\n", "the branch adds the same GAP")
    chk("IN-BASE", classify_commit(r, twin, "main")[0],
        "content already on the base is IN-BASE, not a loss")
    d.cleanup()

    # A deletion-only commit cannot be confirmed by looking for lines it adds.
    d, r = scratch()
    commit(r, "f.txt", "a\nb\nc\nd\n", "one")
    branch(r, "main")
    del_sha = commit(r, "f.txt", "a\n", "delete two lines")
    chk("UNVERIFIABLE", classify_commit(r, del_sha, "main")[0],
        "a deletion-only commit is UNVERIFIABLE, not a pass")
    d.cleanup()

    # The threshold: a commit whose lines main later edited is PARTIAL, not LOST. Without this arm
    # a "any line absent" rule calls every landed commit lost -- measured on #2623, where three
    # landed commits flagged 4 %, 25 % and 30 % of their lines absent.
    d, r = scratch()
    commit(r, "seed.txt", "seed\n", "seed")
    ten = "".join(f"value_{i:02d} = compute({i})\n" for i in range(10))
    commit(r, "m.py", ten.replace("value_09 = compute(9)\n", "value_09 = compute(999)\n"),
           "main edited one of the ten")
    branch(r, "pr", "HEAD~1")
    partial_sha = commit(r, "m.py", ten, "the branch adds all ten")
    chk("PARTIAL", classify_commit(r, partial_sha, "main")[0],
        "one line of ten edited away by main is PARTIAL, not LOST")
    d.cleanup()

    # The gate: an unknown frozen head must never skip a branch, and equal heads must skip.
    chk(True, same_head("0193a29e5" + "0" * 31, "0193a29e5" + "9" * 31), "an equal frozen head skips")
    chk(False, same_head("0193a29e5" + "0" * 31, "f7700ba0a" + "1" * 31),
        "a moved branch tip does not skip")
    chk(False, same_head("", "f7700ba0a" + "1" * 31),
        "an unknown frozen head does NOT skip -- not knowing is not a pass")

    # The tail set itself: a branch tip equal to the base has no tail.
    d, r = scratch()
    commit(r, "f.txt", "x\n", "one")
    subprocess.run(["git", "branch", "-q", "same"], cwd=r, check=True)
    chk(0, len(merge_commits(r, "same", "main")), "a branch tip at the base has no tail")
    d.cleanup()

    # A branch carrying an extra commit has a tail of exactly one.
    d, r = scratch()
    commit(r, "f.txt", "x\n", "one")
    subprocess.run(["git", "branch", "-q", "tip"], cwd=r, check=True)
    commit(r, "f.txt", "x\ny\n", "two")
    chk(1, len(merge_commits(r, "HEAD", "tip")), "a branch with one extra commit has a tail of 1")
    d.cleanup()

    # Zero findings must be distinguishable from not having looked: an unreadable PR list is rc 2.
    chk(2, report(pathlib.Path("."), 1, 1, "1bit-MONSTER/no-such-repo-xyz")[0],
        "an unreadable PR list is rc 2, not an empty pass")

    # shortcut: the shallow-clone refusal has no arm -- a local `--depth 1` fixture kept cloning an empty repository (git's
    # own "cloned an empty repository" warning), and shipping a red arm is worse than shipping none. The guard was verified by
    # hand against a genuine shallow clone of this repo (it refuses, and --allow-shallow proceeds); upgrade by cloning a real
    # origin over file:// with two commits and asserting both directions.
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true", help="run the arms and exit")
    ap.add_argument("--repo", default="1bit-MONSTER/engine", help="repo to read merged PRs from")
    ap.add_argument("--days", type=int, default=30, help="how far back to look (default 30)")
    ap.add_argument("--limit", type=int, default=200, help="max merged PRs to examine")
    ap.add_argument("--worktree", default=".", help="a clone with origin = --repo")
    ap.add_argument("--branch", help="instead of scanning PRs, list what this ref has that "
                                    "--base does not (the recovery workflow)")
    ap.add_argument("--base", default="origin/main", help="base for --branch (default origin/main)")
    ap.add_argument("--allow-shallow", action="store_true",
                    help="report anyway in a shallow clone, where main's own commits read as LOST")
    a = ap.parse_args()
    # engine#2652 (found by a lane running this on a --depth 50 clone): a SHALLOW clone makes this tool lie. The first run
    # there reported 22 LOST where the full clone reports 4, and 18 of the 22 were main's own commits -- they had come onto the
    # local base by a merge whose parents are not present, so every commit looks absent from the base. The results are not
    # merely noisier, they are mostly false, so refuse rather than report: a caller who wants the number anyway can say so.
    if a.worktree and not a.self_test:
        shallow = (git(a.worktree, "rev-parse", "--is-shallow-repository").stdout or "").strip()
        if shallow == "true" and not a.allow_shallow:
            raise SystemExit("%s is a SHALLOW clone: this tool reads commit ancestry, and a shallow clone reports main's own "
                             "commits as LOST (measured: 22 LOST on --depth 50 against 4 on the full clone, 18 of them main's). "
                             "Deepen it (git fetch --unshallow) or pass --allow-shallow to report anyway." % a.worktree)
    if a.self_test:
        return self_test()
    if a.branch:
        wt = pathlib.Path(a.worktree)
        if git(wt, "fetch", "-q", "--no-tags", "origin",
               a.base.replace("origin/", "")).returncode != 0:
            print(f"merged_tail_check: could not fetch {a.base} -- cannot tell, and that is not "
                  f"a pass", file=sys.stderr)
            return 2
        tail = branch_mode(wt, a.branch, a.base)
        if tail is None:
            print(f"merged_tail_check: could not read {a.branch} -- cannot tell, and that is not "
                  f"a pass", file=sys.stderr)
            return 2
        return 1 if tail else 0
    rc, _, _ = report(pathlib.Path(a.worktree), a.days, a.limit, a.repo)
    return rc


if __name__ == "__main__":
    sys.exit(main())
