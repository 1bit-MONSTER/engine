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
"""stale_carry_check.py -- branches whose merge would override work that landed on main since.

The repos squash-merge, so a branch cut BEFORE a merge never saw the newer files, and a branch that also
touched them merges into a conflict -- or, if the conflict is resolved toward the branch, silently puts the
older copy back. Three of these were caught by hand on 2026-11-16: engine#2728 and #2755 each carried
`tools/ci/merged_tail_check.py` back past the per-line supersession, and engine#2693's rebase dropped main's
#2679 text exactly this way. Each cost a manual two-dot diff and a merge-tree run.

This does that by machine, per branch, and only reports files where it matters:

  * `git merge-tree --write-tree <base> <branch>` -- the merge, without a worktree, and whether it conflicts;
  * `git diff --name-only <merge-base> <base>` -- the files main moved while the branch sat;
  * the FINDING is the intersection: a file the merge changes AND main moved. A branch that is merely behind
    main is not a finding -- merging it changes nothing -- which is what keeps this quiet enough to run on
    every branch.

`--allow <path>` marks an override that is meant (a deliberate revert looks identical), and a ref that cannot
be read is exit 2, never a pass: a check that could not look is not a clean check.

Exit 0 nothing to read, 1 at least one file to read, 2 a ref could not be read.
"""

import argparse
import pathlib
import subprocess
import sys


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)


def blob_of(repo, rev, path):
    """The blob id of `path` at `rev`, or "" when the path does not exist there."""
    r = git(repo, "rev-parse", "--verify", "-q", f"{rev}:{path}")
    return r.stdout.strip() if r.returncode == 0 else ""


def base_history(repo, base, max_commits):
    """Blob ids main has carried per file, newest first, bounded to `max_commits` commits.

    Bounded because the question is "is this an older copy of a file that recently moved", not
    "has this content ever existed".
    """
    r = git(repo, "rev-list", f"--max-count={max_commits}", base)
    if r.returncode != 0:
        return None
    return [c.strip() for c in r.stdout.splitlines() if c.strip()]


def check(repo, base, branch, max_commits=50, allow=()):
    """[(path, conflict)] -- files the MERGE would change that main also moved since the branch point.

    Two measurements, because the dangerous merge is the one where BOTH sides moved the same file:

      * `git merge-tree --write-tree base branch` applies the merge without touching the worktree, and says
        whether it conflicts;
      * `git diff --name-only <merge-base> base` says what main changed while the branch sat -- those are the
        files whose newer content the branch never saw.

    A file in both sets is one where merging either conflicts (and a hand resolution toward the branch is how
    engine#2693 dropped main's #2679 text) or quietly lands the branch's older copy (engine#2728 and #2755
    carried `tools/ci/merged_tail_check.py` back past the per-line supersession). A branch that is merely
    behind main is NOT a finding: it is not in the first set, because merging it changes nothing.

    `max_commits` bounds how far back the branch point may be for its moved files to count.
    """
    if git(repo, "rev-parse", "--verify", "-q", base).returncode != 0:
        return None
    if git(repo, "rev-parse", "--verify", "-q", branch).returncode != 0:
        return None
    mb = git(repo, "merge-base", base, branch)
    if mb.returncode != 0 or not mb.stdout.strip():
        return None
    root = mb.stdout.strip()
    if git(repo, "rev-list", f"--max-count={max_commits + 1}", f"{root}..{base}").stdout.strip() == "" \
            and git(repo, "rev-list", "--count", f"{root}..{base}").stdout.strip() not in ("", "0"):
        return []          # the branch point is older than the window: out of scope, and said so by the caller
    out = git(repo, "merge-tree", "--write-tree", base, branch)
    heads = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not heads:
        return None
    conflict = out.returncode != 0
    moved = {p.strip() for p in git(repo, "diff", "--name-only", root, base).stdout.splitlines() if p.strip()}
    changed = {p.strip() for p in git(repo, "diff", "--name-only", base, heads[0]).stdout.splitlines() if p.strip()}
    return sorted((p, conflict) for p in (moved & changed) if p not in allow)


def self_test():
    """Arms: a carried-back file, a clean branch, the allow list, and the refusals."""
    import tempfile

    bad = 0

    def chk(cond, what):
        nonlocal bad
        if not cond:
            print(f"FAIL {what}")
            bad += 1

    with tempfile.TemporaryDirectory() as td:
        r = pathlib.Path(td) / "repo"
        r.mkdir()

        def run(*a):
            return subprocess.run(["git", *a], cwd=str(r), capture_output=True, text=True)

        run("init", "-q")
        run("symbolic-ref", "HEAD", "refs/heads/main")
        run("config", "user.email", "t@e")
        run("config", "user.name", "t")

        def commit(path, text, msg):
            (r / path).parent.mkdir(parents=True, exist_ok=True)
            (r / path).write_text(text)
            run("add", "-A")
            run("commit", "-qm", msg)
            return run("rev-parse", "HEAD").stdout.strip()

        commit("tool.py", "def f():\n    return 1\n", "c1")
        run("checkout", "-q", "-B", "pr")
        commit("other.txt", "branch work\n", "the branch's own commit")
        run("checkout", "-q", "main")
        commit("tool.py", "def f():\n    return 2\n", "c2: main moves tool.py after the branch was cut")
        chk(check(r, "main", "pr") == [],
            "a branch that is merely BEHIND main is not a finding -- merging it changes nothing")

        # The shape both real cases had: the branch also changes the file main moved, so the merge lands one
        # side or the other. That is the file to read before merging.
        run("checkout", "-q", "pr")
        commit("tool.py", "def f():\n    return 3  # the branch's own edit\n", "branch edits tool.py too")
        found = check(r, "main", "pr") or []
        chk([p for p, _ in found] == ["tool.py"],
            "a file BOTH sides moved is named")
        chk(True in [c for _, c in found], "and the row says the merge conflicts")
        chk(check(r, "main", "pr", allow=("tool.py",)) == [],
            "an allowed path is not a finding (a deliberate override must say so)")

        # A branch that took main's newer file is clean.
        run("checkout", "-q", "pr")
        run("checkout", "main", "--", "tool.py")
        run("commit", "-qm", "take main's tool.py")
        chk(check(r, "main", "pr") == [], "a branch that took main's newer file is clean")

        # A branch that never touched the file, and main never moved it after the cut: nothing to say.
        run("checkout", "-q", "main")
        commit("third.txt", "x\n", "c3")
        run("checkout", "-q", "-B", "clean", "main")
        commit("fourth.txt", "y\n", "branch on the current tip")
        chk(check(r, "main", "clean") == [], "a branch cut at the tip carries nothing back")

        chk(check(r, "no-such-base", "pr") is None, "an unreadable base is None, not a pass")
        chk(check(r, "main", "no-such-branch") is None, "an unreadable branch is None, not a pass")

    print("ok" if bad == 0 else f"{bad} FAIL")
    return 1 if bad else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("branches", nargs="*", help="branches to check (default: read from --base-files)")
    ap.add_argument("--repo", default=".", help="a clone with the base and the branches")
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--max-commits", type=int, default=50,
                    help="how far back in the base history a carried file is recognised (default 50)")
    ap.add_argument("--allow", action="append", default=[],
                    help="path a deliberate revert carries back on purpose (repeatable)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.branches:
        ap.print_help()
        return 2
    findings = 0
    unreadable = 0
    for b in a.branches:
        carried = check(a.repo, a.base, b, a.max_commits, tuple(a.allow))
        if carried is None:
            print(f"stale_carry_check: cannot read {a.base} or {b} -- cannot tell, and that is not a pass",
                  file=sys.stderr)
            unreadable += 1
            continue
        for path, conflict in carried:
            print(f"{b}: merging this {'CONFLICTS on' if conflict else 'would change'} {path}, which main moved "
                  f"since the branch point -- read it before merging "
                  f"(take it from the base, or say --allow {path} if the override is meant)")
            findings += 1
    if findings:
        return 1
    if unreadable:
        return 2
    print(f"stale_carry_check: {len(a.branches)} branch(es) examined, nothing carried back")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
