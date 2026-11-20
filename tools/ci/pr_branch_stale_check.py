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
"""pr_branch_stale_check: an open PR whose branch is behind main would REVERT what landed since it was cut.

GitHub shows a PR against its merge-base, so a stale branch reads as a clean additive diff on the page while the merge applies
`git diff origin/main origin/<branch>` — which deletes everything the base gained in between. On 2026-11-20 seven of seven engine
PRs reviewed carried a stale base; one displayed `+596/-0` and would have deleted 1475 lines, including a copyright exemption and
two tools' worth of work. This is the pre-merge twin of `merged_tail_check`: same silent loss, one step earlier where nothing looks.

The signal is DELETIONS, not distance. `behind 1` with no deletions is the ordinary "someone merged after you pushed" case and is
not a finding; a branch that is behind AND whose merge deletes lines is the one that reverts a day's work.
"""
import argparse, json, pathlib, subprocess, sys

def run(*a, cwd=None):
    r = subprocess.run(a, capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr

def open_prs(repo):
    """[(number, branch)] from `gh`, or raise SystemExit(2) — an unreadable list is not an empty pass."""
    rc, out, err = run("gh", "pr", "list", "-R", repo, "--state", "open", "--limit", "50",
                       "--json", "number,headRefName")
    if rc != 0:
        raise SystemExit(f"pr_branch_stale_check: cannot list {repo} PRs (rc {rc}): {err.strip()} -- cannot tell, not a pass")
    return [(p["number"], p["headRefName"]) for p in json.loads(out or "[]")]

def stale_finding(repo_dir, branch, base="origin/main"):
    """(behind, reverted_lines, files) where the merge deletes lines MAIN gained, or None when nothing is at risk.

    `add == 0` (whole-file removal) was the first filter and it missed the common case: a branch that also edits the file main
    grew shows `+N/-M` with both non-zero, so the revert of #2789 -- folded_name reverted inside a file it was editing -- counted
    as zero. The question that separates a revert from a reformat is whether MAIN added lines to that file after the merge-base.
    """
    rc, out, _ = run("git", "rev-list", "--count", f"{branch}..{base}", cwd=repo_dir)
    if rc != 0:
        return None
    behind = int(out.strip() or 0)
    if behind == 0:
        return None
    rc, mb, _ = run("git", "merge-base", base, branch, cwd=repo_dir)
    if rc != 0 or not mb.strip():
        return None
    merge_base = mb.strip()
    rc, out, _ = run("git", "diff", "--numstat", base, branch, cwd=repo_dir)
    if rc != 0:
        return None
    reverted, files = 0, []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        _add, sub, path = parts
        if not sub.isdigit() or int(sub) == 0:
            continue
        rc2, g, _ = run("git", "diff", "--numstat", merge_base, base, "--", path, cwd=repo_dir)
        if rc2 != 0 or not g.strip():
            continue
        gained = sum(int(l.split("\t")[0]) for l in g.splitlines() if l.split("\t")[0].isdigit())
        if gained > 0:
            reverted += int(sub)
            files.append(path)
    if not files:
        return None
    return behind, reverted, files


def self_test():
    ok = True
    def chk(cond, what):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok = ok and cond
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        g = lambda *a: subprocess.run(("git", *a), cwd=d, capture_output=True, text=True)
        g("init", "-q"); g("symbolic-ref", "HEAD", "refs/heads/main")
        g("config", "user.email", "t@e"); g("config", "user.name", "t")
        pathlib.Path(d, "f.txt").write_text("one\ntwo\n")
        g("add", "-A"); g("commit", "-qm", "base")
        g("branch", "stale")                                     # branch cut here
        pathlib.Path(d, "f.txt").write_text("one\ntwo\nthree\n")
        g("add", "-A"); g("commit", "-qm", "lands after the cut")  # main moves on
        g("update-ref", "refs/remotes/origin/main", "HEAD")
        rc_up = stale_finding(d, "main")
        chk(rc_up is None, "a branch level with main is not a finding at all (no distance, nothing to revert)")
        behind, deleted, files = stale_finding(d, "stale") or (0, 0, [])
        chk(behind == 1, "a branch cut before one commit is behind 1")
        # Any behind branch shows deletions here: main's newer lines are absent from it, so the merge removes them. That
        # IS the defect — an earlier version of this arm asserted "deletes nothing" and was wrong about the mechanic.
        chk(deleted == 1, "…and the merge deletes exactly the line main gained, which is the revert")
        g("checkout", "-q", "stale")
        pathlib.Path(d, "f.txt").write_text("one\n")             # the stale branch would drop a line
        g("commit", "-qam", "stale edit")
        g("checkout", "-q", "main")
        behind, deleted, files = stale_finding(d, "stale") or (0, 0, [])
        chk(behind == 1 and deleted > 0, "a behind branch whose merge deletes lines IS a finding")
        chk("f.txt" in files, "…and names the file it would delete")

    # engine#2992 review: a HUNK-level revert -- the shape seen most often. main gains a line in a file the branch is also
    # editing, so `numstat` is +N/-M with both non-zero; the old `add == 0` filter called that zero.
    with tempfile.TemporaryDirectory() as d:
        g = lambda *a: subprocess.run(("git", *a), cwd=d, capture_output=True, text=True)
        g("init", "-q"); g("symbolic-ref", "HEAD", "refs/heads/main")
        g("config", "user.email", "t@e"); g("config", "user.name", "t")
        pathlib.Path(d, "f.txt").write_text("alpha\nbeta\ngamma\n")
        g("add", "-A"); g("commit", "-qm", "base")
        g("branch", "hunk")
        pathlib.Path(d, "f.txt").write_text("alpha\nbeta\ngamma\nMAIN_GAINED\n")
        g("add", "-A"); g("commit", "-qm", "main gains a line in f.txt")
        g("update-ref", "refs/remotes/origin/main", "HEAD")
        g("checkout", "-q", "hunk")
        g("update-ref", "refs/remotes/origin/hunk", "HEAD")
        pathlib.Path(d, "f.txt").write_text("ALPHA\nBETA\ngamma\n")   # branch rewrites the region
        g("commit", "-qam", "branch edits the same file")
        f = stale_finding(d, "hunk")
        chk(f is not None, "a HUNK-level revert (numstat +2/-2, both non-zero) IS a finding -- the case `add == 0` missed")
        chk(f is not None and "f.txt" in f[2], "…and it names the file main grew and the merge would delete from")

    # …while a branch that is behind but content-identical stays quiet. My first fixture for this ("a reformat behind main")
    # asserted quiet and FAILED, correctly: a behind branch lacking main's OTHER commits deletes those too, so the reformat was
    # reverting main's change to a second file. The harmless shape is a branch that carries the same content main gained — behind
    # by ancestry, identical by tree.
    with tempfile.TemporaryDirectory() as d:
        g = lambda *a: subprocess.run(("git", *a), cwd=d, capture_output=True, text=True)
        g("init", "-q"); g("symbolic-ref", "HEAD", "refs/heads/main")
        g("config", "user.email", "t@e"); g("config", "user.name", "t")
        pathlib.Path(d, "shared.txt").write_text("alpha\nbeta\n")
        g("add", "-A"); g("commit", "-qm", "base")
        g("branch", "mine")
        pathlib.Path(d, "shared.txt").write_text("alpha\nbeta\ngamma\n")
        g("commit", "-qam", "main gains gamma")
        g("update-ref", "refs/remotes/origin/main", "HEAD")
        g("checkout", "-q", "mine")
        pathlib.Path(d, "shared.txt").write_text("alpha\nbeta\ngamma\n")   # same content, independent commit
        g("commit", "-qam", "the branch already carries gamma")
        chk(stale_finding(d, "mine") is None,
            "a branch behind by ancestry but identical in content is NOT a finding -- nothing of main's is deleted")
    return 0 if ok else 1

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="1bit-MONSTER/engine")
    ap.add_argument("--root", default=".")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    root = pathlib.Path(a.root).resolve()
    bad = 0
    for n, branch in open_prs(a.repo):
        run("git", "fetch", "-q", "origin", branch, cwd=root)
        f = stale_finding(root, f"origin/{branch}")
        if f is None:
            print(f"#{n} {branch}: up to date with main")
            continue
        behind, deleted, files = f
        if deleted == 0:
            print(f"#{n} {branch}: behind {behind}, deletes nothing -- fine")
            continue
        bad += 1
        print(f"#{n} {branch}: BEHIND {behind} and its merge DELETES {deleted} line(s) in {len(files)} file(s): "
              f"{', '.join(files[:3])}{' …' if len(files) > 3 else ''}", file=sys.stderr)
    if bad:
        print(f"pr_branch_stale_check: {bad} open PR(s) would revert main -- rebase or re-cut them before merging", file=sys.stderr)
        return 1
    print("pr_branch_stale_check: no open PR would revert main")
    return 0

if __name__ == "__main__":
    sys.exit(main())
