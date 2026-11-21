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


def advisory_exit(verdict):
    """The exit `--advisory` turns a verdict into: a finding becomes 0, cannot-tell becomes 77 (the runner's skip)."""
    if verdict == EXIT_FINDING:
        return 0
    if verdict == EXIT_CANNOT_TELL:
        return 77
    return verdict


def single_branch_exit(fetch_ok, behind, finding):
    """The exit code for ONE named branch (engine#3076), as a pure function so its table is assertable without a network.

    The hard gate needs a distinction the open-PR path never had to draw: `stale_finding` returns None both when the branch is
    LEVEL with main (clean) and when the merge could not be measured (cannot tell). Advisory may treat both as "nothing to say";
    a gate that blocks a merge may not, so `behind` decides between them here.
    """
    if not fetch_ok:
        return EXIT_CANNOT_TELL
    if finding is not None:
        return EXIT_FINDING
    return EXIT_CLEAN if behind == 0 else EXIT_CANNOT_TELL


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
    """(behind, reverted, carried, files) measuring THE MERGE, or None when the merge removes none of main's lines.

    Three versions of this measurement were wrong before it settled, each caught by a review fixture rather than by reading:

      1. `git diff base branch` reports what BASE has that the branch lacks, which is not what a merge applies;
      2. filtering on `add == 0` (whole-file removal) missed a branch that also edits the file -- `+N/-M` looks like a rewrite;
      3. keeping that filter inside the merge measure missed the same shape again: a branch that removes main's line AND adds
         something in the same file reads `1  1`, so "add == 0" drops it while the merge does delete the line.

    So nothing is filtered. Every deletion between base and the merged tree is a line the merge removes from main, and the two
    classes are reported separately because they mean different things to the author:

      * `reverted` -- lines MAIN added after the merge-base (the merge undoes newer work);
      * `carried`  -- lines main still carries from before (the branch removes work main has).

    Both are findings. A reformat reads as one carried line, which the engine#2992 review accepted explicitly: a file named with a
    count of 1 against silent loss is the right trade.
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
    rc, out, _ = run("git", "merge-tree", "--write-tree", base, branch, cwd=repo_dir)
    if rc != 0 or not out.strip():
        return None                     # a conflicted merge is loud; not this tool's business
    merged_tree = out.splitlines()[0].strip()

    def texts(args):
        rc, o, _ = run(*args, cwd=repo_dir)
        if rc != 0:
            return []
        return [l[1:] for l in o.splitlines() if l[:1] in ("+", "-") and not l.startswith(("+++", "---")) and l[1:].strip()]

    gained_since_cut = set(texts(("git", "diff", "-U0", mb.strip(), base)))          # what main added after the branch was cut
    rc, o, _ = run("git", "diff", "-U0", base, merged_tree, cwd=repo_dir)
    if rc != 0:
        return None
    deleted = [l[1:] for l in o.splitlines() if l.startswith("-") and not l.startswith("---") and l[1:].strip()]
    rc, names, _ = run("git", "diff", "--name-only", base, merged_tree, cwd=repo_dir)
    files = [f for f in names.splitlines() if f.strip()]
    reverted = sum(1 for l in deleted if l in gained_since_cut)
    carried = len(deleted) - reverted
    if reverted + carried == 0:
        return None
    return behind, reverted, carried, files


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
    fa = stale_finding(d.name, "feat")
    chk(fa is not None and fa[1] == 0 and fa[2] == 1,
        "review fixture A: the branch's own replaced line is CARRIED (1), not a revert of main's newer work (0)")
    chk(fa is not None and "a.txt" in fa[3] and "b.txt" not in fa[3],
        "…and b.txt -- which main grew and the merge keeps -- is NOT named")

    # Fixture B from the review: the branch reverts a line inside a file MAIN never touched. The merge deletes it, and the two-dot
    # diff named the wrong file (d.txt, which the merge keeps).
    d, g = repo({"c.txt": "L1\nL2\nL3\n", "d.txt": "d0\n"})
    g("branch", "revert")
    commit(g, d.name, {"d.txt": "d0\nd1\n"}, "main grows d.txt only")
    g("checkout", "-q", "revert")
    pathlib.Path(d.name, "c.txt").write_text("L1\nL3\n")          # drops L2
    g("commit", "-qam", "branch reverts L2")
    f = stale_finding(d.name, "revert")
    chk(f is not None, "review fixture B: a line removed from a file main never touched IS a finding")
    chk(f is not None and f[3] == ["c.txt"], "…and names c.txt (the file the merge deletes from), not d.txt")
    chk(f is not None and f[1] + f[2] == 1, "…and the count is the one line the merge removes")

    # The shape the filter hid: the branch removes one of main's lines AND adds something in the same file (`+1/-1`).
    d, g = repo({"mix.txt": "K1\nK2\n"})
    g("branch", "mix")
    commit(g, d.name, {"other.txt": "o\n"}, "main moves elsewhere")
    g("checkout", "-q", "mix")
    pathlib.Path(d.name, "mix.txt").write_text("K1\nNOTE\n")     # drops K2, adds NOTE
    g("commit", "-qam", "branch removes K2 and adds NOTE")
    f = stale_finding(d.name, "mix")
    chk(f is not None, "a branch that removes main's line AND adds one (+1/-1) IS a finding -- the case `add == 0` hid")
    chk(f is not None and "mix.txt" in f[3], "…and it names mix.txt")

    # A whole-file removal by the branch is still the loudest case.
    d, g = repo({"keep.txt": "k\n", "gone.txt": "g1\ng2\n"})
    g("branch", "drop")
    commit(g, d.name, {"added.txt": "x\n"}, "main adds a file")
    g("checkout", "-q", "drop")
    pathlib.Path(d.name, "gone.txt").unlink()
    g("add", "-A"); g("commit", "-qm", "branch deletes gone.txt")
    f = stale_finding(d.name, "drop")
    chk(f is not None and "gone.txt" in f[3], "a branch that deletes a whole file IS a finding and names it")

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
    # engine#3057: `--advisory` is for a caller that scans OTHER lanes' PRs, so a finding must not fail it -- while
    # cannot-tell must stay visible as 77 (the runner's skip): a caller that could not look must not look green.
    chk(advisory_exit(EXIT_FINDING) == 0, "advisory: a finding is reported and exits 0, so it cannot block another lane's gate")
    chk(advisory_exit(EXIT_CANNOT_TELL) == 77, "advisory: cannot-tell is 77 (a skip), never 0")
    chk(advisory_exit(EXIT_CLEAN) == 0, "advisory: a clean run stays 0")
    # engine#3076: the single-branch table, which is the whole difference between an advisory row and a gate
    chk(single_branch_exit(False, 0, None) == EXIT_CANNOT_TELL,
        "--branch: a branch that cannot be fetched is cannot-tell (2), not clean")
    chk(single_branch_exit(True, 3, (3, 1, 1, ["f"])) == EXIT_FINDING,
        "--branch: the merge deletes main's lines -> 1")
    chk(single_branch_exit(True, 0, None) == EXIT_CLEAN,
        "--branch: level with main -> 0")
    chk(single_branch_exit(True, 3, None) == EXIT_CANNOT_TELL,
        "--branch: behind but the merge could not be measured -> 2, never 0")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="1bit-MONSTER/engine")
    ap.add_argument("--root", default=".")
    ap.add_argument("--advisory", action="store_true",
                    help="report every stale branch and exit 0 even with findings -- for a caller that scans OTHER "
                         "lanes' PRs (scripts/pre-pr-checks.sh), where a hard failure would let one lane's branch "
                         "block another lane's gate. Cannot-tell still answers 77 (skip), never 0 (engine#3057)")
    ap.add_argument("--branch", default=None, metavar="NAME",
                    help="evaluate ONE named branch as a hard gate: 0 clean, 1 its merge deletes lines main has, 2 cannot tell. "
                         "Takes the ref directly, so it needs no `gh` call and no token -- which is what lets a pull_request job "
                         "run it (engine#3076). The open-PR scan stays advisory; this mode is never advisory")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    root = pathlib.Path(a.root).resolve()
    if a.branch:
        ref = f"origin/{a.branch}"
        rc, _out, err = run("git", "fetch", "-q", "origin", a.branch, cwd=root)
        fetch_ok, behind, finding = rc == 0, 0, None
        if fetch_ok:
            rc, out, _ = run("git", "rev-list", "--count", f"{ref}..origin/main", cwd=root)
            if rc != 0:
                behind = -1                  # distance unknown: cannot_tell, never a clean 0
            else:
                behind = int(out.strip() or 0)
                if behind:
                    finding = stale_finding(root, ref)
        # The exit code FIRST, then the message: a run that could not look must not print a verdict at all. The first draft
        # printed "removes nothing main has" for an unfetchable branch before exiting 2 -- a green line in a red run, which is
        # the exact failure this tool exists to prevent (engine#3076 review).
        code = single_branch_exit(fetch_ok, behind, finding)
        if code == EXIT_CANNOT_TELL:
            if not fetch_ok:
                cannot_tell(f"{a.branch}: cannot fetch the branch "
                            f"({err.strip().splitlines()[0] if err.strip() else 'fetch failed'})")
            if behind < 0:
                cannot_tell(f"{a.branch}: cannot compare with origin/main (git rev-list failed)")
            cannot_tell(f"{a.branch}: BEHIND {behind}, but the merge could not be measured "
                        f"(no merge base, or the tree could not be written)")
        if finding is None:
            print(f"{a.branch}: level with main; merging this branch removes nothing main has")
            return code
        _b, reverted, carried, files = finding
        parts = []
        if reverted:
            parts.append(f"REVERTS {reverted} line(s) main added after the branch was cut")
        if carried:
            parts.append(f"removes {carried} line(s) main still carries")
        print(f"{a.branch}: BEHIND {finding[0]}; merging it " + " and ".join(parts) +
              f" in {len(files)} file(s): {', '.join(files[:3])}{' …' if len(files) > 3 else ''}", file=sys.stderr)
        return code
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
        behind, reverted, carried, files = f
        findings += 1
        parts = []
        if reverted:
            parts.append(f"REVERTS {reverted} line(s) main added after the branch was cut")
        if carried:
            parts.append(f"removes {carried} line(s) main still carries")
        print(f"#{n} {branch}: BEHIND {behind}; merging it " + " and ".join(parts) +
              f" in {len(files)} file(s): {', '.join(files[:3])}{' …' if len(files) > 3 else ''}", file=sys.stderr)
    if findings:
        print(f"pr_branch_stale_check: {findings} open PR(s) would delete lines main has "
              f"(measured with `git merge-tree --write-tree`, not a two-dot diff) -- rebase or re-cut before merging",
              file=sys.stderr)
    elif not unreadable:
        print("pr_branch_stale_check: no open PR would revert main (every branch was read)")
    rc = verdict_exit(unreadable, findings)
    if a.advisory:
        return advisory_exit(rc)
    if a.advisory and rc == EXIT_FINDINGS:
        # advisory: the finding is printed above and this run does not fail -- see the flag's help
        return 0
    if a.advisory and rc == EXIT_CANNOT_TELL:
        # a caller that cannot fail must still not look green: the runner counts 77 as a visible skip (engine#2932)
        return 77
    if rc == EXIT_CANNOT_TELL:
        print(f"pr_branch_stale_check: {len(unreadable)} of the open PR(s) could not be read, so this is not a clean run",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
