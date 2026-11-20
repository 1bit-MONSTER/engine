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

Exit: 0 every open PR merges without removing lines main has; 1 at least one would; 2 CANNOT TELL -- an unreadable PR list, a full
page (truncation), or a branch whose ref could not be fetched. A 0 means every branch was read.

GitHub shows a PR against its merge-base, so a stale branch reads as a clean additive diff on the page while the merge applies
`git diff origin/main origin/<branch>` — which deletes everything the base gained in between. On 2026-11-20 seven of seven engine
PRs reviewed carried a stale base; one displayed `+596/-0` and would have deleted 1475 lines, including a copyright exemption and
two tools' worth of work. This is the pre-merge twin of `merged_tail_check`: same silent loss, one step earlier where nothing looks.

The signal is DELETIONS, not distance. `behind 1` with no deletions is the ordinary "someone merged after you pushed" case and is
not a finding; a branch that is behind AND whose merge deletes lines is the one that reverts a day's work.
"""
import argparse, json, pathlib, subprocess, sys

EXIT_CLEAN, EXIT_FINDING, EXIT_CANNOT_TELL = 0, 1, 2

def cannot_tell(why):
    """Exit 2, never 1: an unreadable queue is not a finding and not a pass (engine#2992 review)."""
    print(f"pr_branch_stale_check: {why} -- cannot tell, not a pass", file=sys.stderr)
    raise SystemExit(EXIT_CANNOT_TELL)


def verdict_exit(unreadable, findings):
    """The exit code for a finished run, as a pure function so the table can be asserted without a network or a fixture."""
    if unreadable:
        return EXIT_CANNOT_TELL
    return EXIT_FINDING if findings else EXIT_CLEAN


def run(*a, cwd=None):
    r = subprocess.run(a, capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr

PAGE = 50

def open_prs(repo):
    """[(number, branch)] from `gh`, or raise SystemExit(2) — an unreadable or TRUNCATED list is not a pass."""
    rc, out, err = run("gh", "pr", "list", "-R", repo, "--state", "open", "--limit", str(PAGE),
                       "--json", "number,headRefName")
    if rc != 0:
        cannot_tell(f"cannot list {repo} PRs (rc {rc}): {err.strip()}")
    prs = [(p["number"], p["headRefName"]) for p in json.loads(out or "[]")]
    if len(prs) >= PAGE:
        # engine#2992 review: the queue has run at 20-30, so a full page is close enough that an unchecked PR is plausible.
        cannot_tell(f"{len(prs)} open PRs is the whole page -- the list may be truncated (raise PAGE)")
    return prs

def stale_finding(repo_dir, branch, base="origin/main"):
    """(behind, reverted_lines, files) measuring THE MERGE, or None when the merge removes nothing.

    Two earlier versions measured `git diff base branch`, which is *what base has that the branch lacks* — not what a merge
    applies. On a branch that only edits `a.txt` while base grew `b.txt`, that diff reports base's own `b.txt` lines as deletions
    although a three-way merge keeps them; and on a branch that reverts a line inside a file base never touched, it reports the
    wrong file. engine#2992 review ran both against `git merge` and `git merge --squash`: the diff was wrong in both directions.

    So measure the merge itself: `merge-tree --write-tree` materialises the merged tree without touching the worktree, and the
    deletions between base and that tree are exactly what merging removes. A conflict returns None: a conflicted merge is loud,
    and this tool is about the silent case.
    """
    rc, out, _ = run("git", "rev-list", "--count", f"{branch}..{base}", cwd=repo_dir)
    if rc != 0:
        return None
    behind = int(out.strip() or 0)
    if behind == 0:
        return None
    rc, out, _ = run("git", "merge-tree", "--write-tree", base, branch, cwd=repo_dir)
    if rc != 0 or not out.strip():
        return None                     # a conflicted merge is visible to the author; not this tool's business
    merged_tree = out.splitlines()[0].strip()
    rc, out2, _ = run("git", "diff", "--numstat", base, merged_tree, cwd=repo_dir)
    if rc != 0:
        return None
    reverted, files = 0, []
    for line in out2.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        add, sub = int(parts[0]), int(parts[1])
        # A PURE deletion is the revert; a replacement (add > 0) is the branch editing a line it already had. On the review's
        # fixture A that is the difference between "the branch edited a.txt" and "the branch removed main's work".
        if sub > 0 and add == 0:
            reverted += sub
            files.append(parts[2])
    if reverted == 0:
        return None
    return behind, reverted, files


def self_test():
    """Arms derived from what `git merge` actually does, not from what a two-dot diff says (engine#2992 review)."""
    ok = True
    def chk(cond, what):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok = ok and cond

    def repo(files):
        import tempfile
        d = tempfile.TemporaryDirectory()
        g = lambda *a: subprocess.run(("git", *a), cwd=d.name, capture_output=True, text=True)
        g("init", "-q"); g("symbolic-ref", "HEAD", "refs/heads/main")
        g("config", "user.email", "t@e"); g("config", "user.name", "t")
        for name, text in files.items():
            pathlib.Path(d.name, name).write_text(text)
        g("add", "-A"); g("commit", "-qm", "base")
        return d, g

    def commit(g, d, files, msg):
        for name, text in files.items():
            pathlib.Path(d, name).write_text(text)
        g("add", "-A"); g("commit", "-qm", msg)
        g("update-ref", "refs/remotes/origin/main", "HEAD")

    # A branch level with main: no distance, nothing to revert.
    d, g = repo({"a.txt": "a1\n"})
    chk(stale_finding(d.name, "main") is None, "a branch level with main is not a finding")

    # Fixture A from the review: main gains lines in a file the branch never touches. A three-way merge KEEPS main's version, so
    # the two-dot diff's "deletions" were main's own lines. Must be quiet.
    d, g = repo({"a.txt": "a1\n", "b.txt": "b1\nb2\n"})
    g("branch", "feat")
    commit(g, d.name, {"b.txt": "b1\nb2\nb3\nb4\nb5\n"}, "main grows b.txt")
    g("checkout", "-q", "feat")
    pathlib.Path(d.name, "a.txt").write_text("a1a\n")
    g("commit", "-qam", "branch edits only a.txt")
    chk(stale_finding(d.name, "feat") is None,
        "review fixture A: main grew a file the branch never touched -- the merge keeps it, so NOT a finding")

    # Fixture B from the review: the branch reverts a line inside a file MAIN never touched. The merge deletes it, and the two-dot
    # diff named the wrong file (d.txt, which the merge keeps).
    d, g = repo({"c.txt": "L1\nL2\nL3\n", "d.txt": "d0\n"})
    g("branch", "revert")
    commit(g, d.name, {"d.txt": "d0\nd1\n"}, "main grows d.txt only")
    g("checkout", "-q", "revert")
    pathlib.Path(d.name, "c.txt").write_text("L1\nL3\n")          # drops L2
    g("commit", "-qam", "branch reverts L2")
    f = stale_finding(d.name, "revert")
    chk(f is not None, "review fixture B: a reverted line inside an untouched file IS a finding")
    chk(f is not None and f[2] == ["c.txt"], "…and names c.txt (the file the merge deletes from), not d.txt")
    chk(f is not None and f[1] == 1, "…and the count is the one line the merge removes")

    # A whole-file removal by the branch is still the loudest case.
    d, g = repo({"keep.txt": "k\n", "gone.txt": "g1\ng2\n"})
    g("branch", "drop")
    commit(g, d.name, {"added.txt": "x\n"}, "main adds a file")
    g("checkout", "-q", "drop")
    pathlib.Path(d.name, "gone.txt").unlink()
    g("add", "-A"); g("commit", "-qm", "branch deletes gone.txt")
    f = stale_finding(d.name, "drop")
    chk(f is not None and "gone.txt" in f[2], "a branch that deletes a whole file IS a finding and names it")

    # A branch behind by ancestry but identical in content removes nothing.
    d, g = repo({"shared.txt": "alpha\nbeta\n"})
    g("branch", "mine")
    commit(g, d.name, {"shared.txt": "alpha\nbeta\ngamma\n"}, "main gains gamma")
    g("checkout", "-q", "mine")
    pathlib.Path(d.name, "shared.txt").write_text("alpha\nbeta\ngamma\n")
    g("commit", "-qam", "the branch already carries gamma")
    chk(stale_finding(d.name, "mine") is None,
        "behind by ancestry but identical in content is NOT a finding")

    # engine#2992 review: no arm exercised the EXIT TABLE, which is how an unreadable queue came to report itself as a finding
    # (rc 1) and a fetch failure as a clean run (rc 0). These need no git fixture and no network.
    chk(verdict_exit(unreadable=False, findings=0) == 0, "exit table: clean run is 0")
    chk(verdict_exit(unreadable=False, findings=2) == 1, "exit table: a finding is 1")
    chk(verdict_exit(unreadable=True, findings=0) == 2, "exit table: an unreadable queue is 2, not a clean 0")
    chk(verdict_exit(unreadable=True, findings=3) == 2, "exit table: cannot-tell outranks findings -- both need looking, one needs a rerun")
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
    findings, unreadable = 0, []
    for n, branch in open_prs(a.repo):
        ref = f"origin/{branch}"
        rc, _out, err = run("git", "fetch", "-q", "origin", branch, cwd=root)
        if rc != 0:
            # engine#2992 review: this used to fall through to `stale_finding` returning None, which printed "removes nothing
            # main has" and exited 0 -- a 0 that meant "could not look". A branch that cannot be fetched is a cannot-tell.
            unreadable.append(f"#{n} {branch}: cannot fetch the branch ({err.strip().splitlines()[0] if err.strip() else 'fetch failed'})")
            print(f"#{n} {branch}: CANNOT READ -- the branch could not be fetched, so nothing was measured", file=sys.stderr)
            continue
        f = stale_finding(root, ref)
        if f is None:
            print(f"#{n} {branch}: merging this branch removes nothing main has")
            continue
        behind, deleted, files = f
        findings += 1
        print(f"#{n} {branch}: BEHIND {behind}; merging it DELETES {deleted} line(s) main has, in {len(files)} file(s): "
              f"{', '.join(files[:3])}{' …' if len(files) > 3 else ''}", file=sys.stderr)
    if findings:
        print(f"pr_branch_stale_check: {findings} open PR(s) would delete lines main has "
              f"(measured with `git merge-tree --write-tree`, not a two-dot diff) -- rebase or re-cut before merging",
              file=sys.stderr)
    elif not unreadable:
        print("pr_branch_stale_check: no open PR would revert main (every branch was read)")
    rc = verdict_exit(unreadable, findings)
    if rc == EXIT_CANNOT_TELL:
        print(f"pr_branch_stale_check: {len(unreadable)} of the open PR(s) could not be read, so this is not a clean run",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
