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


class CannotTell(Exception):
    """The measurement could not be made -- which is NOT the same answer as "level with the base" (engine#3082).

    `stale_finding` returns `None` for a clean answer, so every "could not look" case used to return `None` too, and both
    call sites read that as clean. In `--pr` mode -- the per-PR gate #3076 added -- that printed *"merging it removes
    nothing <base> has"* and exited **0 for a CONFLICTED pull request**, which is the one answer a merge gate must never
    give. The tool's own rule for its other mode says it: *"cannot-tell still answers 77 (skip), never 0"*. So the
    unmeasurable cases raise this, and the callers map it to their own cannot-tell channel (2 / skip).
    """


ADVISORY = False


def cannot_tell(why):
    """Exit 2, never 1: an unreadable queue is not a finding and not a pass (engine#2992 review).

    Under `--advisory` it exits 77 instead, which is what the caller counts as a visible SKIP. The flag's own help and
    `scripts/pre-pr-checks.sh` (which maps 77 to `skip` and everything else to a failure) both promise that, but every
    `cannot_tell` raised early used to bypass the end-of-run mapping and fail a lane's pre-PR gate for a condition it
    could not see -- a truncated list, an unfetchable ref, or (engine#3076) a shallow clone. One place decides it now.
    """
    print(f"pr_branch_stale_check: {why} -- cannot tell, not a pass", file=sys.stderr)
    raise SystemExit(77 if ADVISORY else EXIT_CANNOT_TELL)


def pr_cannot_tell(pr, branch, why):
    """The per-PR mode's channel for "could not measure": exit 2, never 0 (engine#3082).

    Named rather than inlined so an arm can assert it: the defect was exactly this mapping, and an arm that needs `gh`
    and a real PR to run would not have been written.
    """
    cannot_tell(f"#{pr} {branch}: {why}")


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


def shallow_problem(is_shallow, rc):
    """The shallow decision as a pure function (engine#3076), so its table is assertable without building a shallow clone."""
    if rc != 0:
        return "cannot tell whether this is a shallow clone (git rev-parse failed)"
    if str(is_shallow).strip() == "true":
        return ("ran in a SHALLOW clone, where the distance to the base is not knowable, so a stale branch would read as "
                "level -- fetch full history (`git fetch --unshallow`) and run again")
    return None


def shallow_guard(root, mode):
    """Refuse to answer in a SHALLOW clone (engine#3076): history is truncated, so `rev-list --count A..B` returns 0 for a
    branch that is in fact behind, and the distance is what decides every verdict in this file. Measured on a `--depth 1`
    clone of this repo: `--pr 3098` printed "merging it removes nothing main has" and exited 0, where a full clone reported
    that the merge conflicts and could not be measured. A 0 must mean every branch was read, and in a shallow clone none was.
    """
    rc, out, _ = run("git", "rev-parse", "--is-shallow-repository", cwd=root)
    problem = shallow_problem(out, rc)
    if problem:
        cannot_tell(f"{mode} {problem}")


def run(*a, cwd=None):
    r = subprocess.run(a, capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout, r.stderr

PAGE = 50

def open_prs(repo):
    """[(number, branch)] from `gh`, or raise SystemExit(2) — an unreadable or TRUNCATED list is not a pass."""
    rc, out, err = run("gh", "pr", "list", "-R", repo, "--state", "open", "--limit", str(PAGE),
                       "--json", "number,headRefName,baseRefName")
    if rc != 0:
        cannot_tell(f"cannot list {repo} PRs (rc {rc}): {err.strip()}")
    # engine#3071: the base is the PR's OWN target, not `main`. Where every PR targets a pin branch (the llama.cpp
    # fork targets `1bit/hrx-vulkan-patched`), measuring against origin/main printed "removes nothing main has" while
    # the merge into the real target deleted lines -- the false clean this tool exists to prevent. Measured on fork
    # #457: against origin/master nothing, against its own base 64 lines of common/sampling.cpp alone.
    prs = [(p["number"], p["headRefName"], p.get("baseRefName") or "main") for p in json.loads(out or "[]")]
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
        raise CannotTell(f"cannot count the commits {branch} is behind {base} (git rc {rc})")
    behind = int(out.strip() or 0)
    if behind == 0:
        return None
    rc, mb, _ = run("git", "merge-base", base, branch, cwd=repo_dir)
    if rc != 0 or not mb.strip():
        raise CannotTell(f"no merge base between {base} and {branch}")
    rc, out, _ = run("git", "merge-tree", "--write-tree", base, branch, cwd=repo_dir)
    if rc != 0 or not out.strip():
        # A conflicted merge cannot be measured. It used to `return None` -- "not this tool's business" was right when the
        # tool only reported on other lanes' PRs, and is exactly wrong in the per-PR gate: the gate answered "removes
        # nothing, rc 0" for a PR whose merge does not apply (engine#3082).
        raise CannotTell(f"the merge of {branch} into {base} conflicts, so what it would delete cannot be measured")
    merged_tree = out.splitlines()[0].strip()

    def texts(args):
        rc, o, _ = run(*args, cwd=repo_dir)
        if rc != 0:
            return []
        return [l[1:] for l in o.splitlines() if l[:1] in ("+", "-") and not l.startswith(("+++", "---")) and l[1:].strip()]

    gained_since_cut = set(texts(("git", "diff", "-U0", mb.strip(), base)))          # what main added after the branch was cut
    rc, o, _ = run("git", "diff", "-U0", base, merged_tree, cwd=repo_dir)
    if rc != 0:
        raise CannotTell(f"cannot diff {base} against the merged tree (git rc {rc})")
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
        # engine#3082: `origin/main` has to exist from the start, or the FIRST arm ("a branch level with main is not a
        # finding") measures nothing: rev-list fails, and under the old code that failure returned None -- which is the
        # clean answer -- so the arm passed while testing nothing. This change surfaced it; the base commit now sets the
        # remote ref the same way every later `commit()` does.
        g("update-ref", "refs/remotes/origin/main", "HEAD")
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

    # engine#3082: a CONFLICTED merge cannot be measured, and the per-PR gate used to answer "removes nothing, rc 0" for
    # it -- because `stale_finding` returned None both for "level with the base" and for "could not look", and every
    # caller read None as clean. The two branches below edit the same line of the same file, one commit apart, so the
    # merge genuinely conflicts (the same fixture the issue reports).
    d, g = repo({"clash.txt": "one\ntwo\n"})
    g("branch", "clash")
    commit(g, d.name, {"clash.txt": "one\nMAIN\n"}, "main edits line 2")
    g("checkout", "-q", "clash")
    pathlib.Path(d.name, "clash.txt").write_text("one\nBRANCH\n")
    g("commit", "-qam", "the branch edits the same line")
    try:
        f = stale_finding(d.name, "clash")
        chk(False, "a conflicted merge is CANNOT-TELL, not a clean None (engine#3082)")
    except CannotTell as e:
        chk("conflict" in str(e), "a conflicted merge is CANNOT-TELL, and says why: " + str(e)[:60])
    # ... and the per-PR mode's channel for it is 2 (skip), never 0 (clean) -- the mapping the defect was in.
    try:
        pr_cannot_tell(7, "feat", "the merge conflicts")
        chk(False, "the per-PR cannot-tell channel exits")
    except SystemExit as e:
        chk(e.code == EXIT_CANNOT_TELL, "the per-PR cannot-tell channel exits 2 (a skip), not 0 (a clean merge)")

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
    # engine#3076: a rebased branch is a NON-FAST-FORWARD update of the remote-tracking ref, and a plain refspec refuses it, so
    # the tool answered cannot-tell on exactly the branches it exists to catch. Proved against a real (local) origin rather than
    # asserted: build one, force-push a branch, leave the tracking ref behind, and require the mirroring spec to work while the
    # plain one does not -- so a revert of the `+` is caught instead of rediscovered.
    import tempfile
    d2 = tempfile.TemporaryDirectory()
    def gg(*a, cwd=None):
        return subprocess.run(("git", *a), cwd=cwd, capture_output=True, text=True)
    origin, work = str(pathlib.Path(d2.name, "origin.git")), str(pathlib.Path(d2.name, "w"))
    gg("init", "-q", "--bare", origin)
    gg("clone", "-q", origin, work)
    for k, v in (("user.email", "t@e"), ("user.name", "t")):
        gg("config", k, v, cwd=work)
    pathlib.Path(work, "a.txt").write_text("a\n")
    gg("add", "-A", cwd=work); gg("commit", "-qm", "base", cwd=work)
    gg("branch", "-M", "main", cwd=work); gg("push", "-q", "origin", "main", cwd=work)
    gg("checkout", "-q", "-b", "rebased", cwd=work)
    pathlib.Path(work, "a.txt").write_text("a\nb\n")
    gg("commit", "-qam", "first attempt", cwd=work); gg("push", "-q", "origin", "rebased", cwd=work)
    first = gg("rev-parse", "HEAD", cwd=work).stdout.strip()
    gg("checkout", "-q", "main", cwd=work); gg("checkout", "-q", "-B", "rebased", cwd=work)
    pathlib.Path(work, "a.txt").write_text("a\nb\nc\n")
    gg("commit", "-qam", "rebased", cwd=work); gg("push", "-q", "-f", "origin", "rebased", cwd=work)
    gg("update-ref", "refs/remotes/origin/rebased", first, cwd=work)   # what a worktree that missed the force-push holds
    # engine#3076: the shallow decision, as a table. The git plumbing is one `rev-parse` line; what needs pinning is which
    # answer it produces, because a shallow clone makes a branch that IS behind read as level (`rev-list --count A..B` cannot
    # see past the graft) -- measured on a `--depth 1` clone of this repo, where `--pr` exited 0 for a branch whose merge
    # conflicts. So: shallow -> a reason, not shallow -> None, and an unreadable probe -> a reason (never "probably fine").
    chk(shallow_problem("true", 0) is not None, "a shallow clone yields a cannot-tell reason, not a pass")
    chk(shallow_problem("false", 0) is None, "a full clone yields no reason (the happy path stays quiet)")
    chk(shallow_problem("", 1) is not None, "a failed shallow probe is cannot-tell, never an assumed full clone")

    plain = gg("fetch", "-q", "origin", "refs/heads/rebased:refs/remotes/origin/rebased", cwd=work)
    forced = gg("fetch", "-q", "origin", "+refs/heads/rebased:refs/remotes/origin/rebased", cwd=work)
    chk(plain.returncode != 0, "the plain refspec cannot mirror a REBASED branch (non-fast-forward) -- the bug")
    chk(forced.returncode == 0, "the mirroring refspec keeps the fix: a rebased branch is fetched, so it can be measured")

    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="1bit-MONSTER/engine")
    ap.add_argument("--root", default=".")
    ap.add_argument("--advisory", action="store_true",
                    help="report every stale branch and exit 0 even with findings -- for a caller that scans OTHER "
                         "lanes' PRs (scripts/pre-pr-checks.sh), where a hard failure would let one lane's branch "
                         "block another lane's gate. Cannot-tell still answers 77 (skip), never 0 (engine#3057)")
    ap.add_argument("--pr", type=int, default=0,
                    help="check ONE pull request: its head against ITS OWN base, without reading the open-PR set. This is "
                         "the mode a per-PR gate needs (engine#3076) -- the open-PR-set mode is advisory by construction, "
                         "because one lane's stale branch must not fail another lane's handover")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    global ADVISORY
    ADVISORY = a.advisory
    root = pathlib.Path(a.root).resolve()
    shallow_guard(root, "--pr" if a.pr else "the open-PR scan")
    findings, unreadable = 0, []

    if a.pr:
        # engine#3076: one PR, its own base. `gh pr view` for the two refs, then the same `stale_finding` the set mode
        # uses -- the verdict logic exists once, so this mode cannot drift from the other.
        rc, out, err = run("gh", "pr", "view", str(a.pr), "-R", a.repo, "--json", "headRefName,baseRefName")
        if rc != 0:
            cannot_tell(f"cannot read PR #{a.pr} (rc {rc}): {err.strip()[:200]}")
        info = json.loads(out)
        branch, base = info["headRefName"], info.get("baseRefName") or "main"
        # engine#3076: the destination is OUR remote-tracking ref, so it is MIRRORED, never fast-forwarded. Without the `+`
        # git refuses the update as non-fast-forward whenever the branch was rebased or force-pushed -- normal life in this
        # repo -- and the tool answers cannot-tell on exactly the branches it exists to catch. Measured on #3074
        # (`pkg-sampler-liveguard`, 3360ccd) and controlled before/after on #3080 (see the regression arm).
        for name in (branch, base):
            ref = f"refs/heads/{name}:refs/remotes/origin/{name}"
            rc, _o, err = run("git", "fetch", "-q", "origin", "+" + ref, cwd=root)
            if rc != 0:
                cannot_tell(f"cannot fetch {name} (rc {rc}): {err.strip()[:120]}")
        try:
            found = stale_finding(root, f"origin/{branch}", f"origin/{base}")
        except CannotTell as e:
            pr_cannot_tell(a.pr, branch, str(e))
        if found is None:
            print(f"#{a.pr} {branch}: merging it removes nothing {base} has")
            return 0
        behind, reverted, carried, files = found
        print(f"#{a.pr} {branch}: BEHIND {behind} {base}; merging it reverts {reverted} line(s) {base} added and "
              f"removes {carried} line(s) it still carries, in {len(files)} file(s): {', '.join(files[:3])}",
              file=sys.stderr)
        return 1

    for n, branch, base in open_prs(a.repo):
        ref = f"origin/{branch}"
        rc, _out, err = run("git", "fetch", "-q", "origin",
                            f"+refs/heads/{branch}:refs/remotes/origin/{branch}", cwd=root)
        if rc != 0:
            # engine#2992 review: this used to fall through to `stale_finding` returning None, which printed "removes nothing
            # main has" and exited 0 -- a 0 that meant "could not look". A branch that cannot be fetched is a cannot-tell.
            unreadable.append(f"#{n} {branch}: cannot fetch the branch ({err.strip().splitlines()[0] if err.strip() else 'fetch failed'})")
            print(f"#{n} {branch}: CANNOT READ -- the branch could not be fetched, so nothing was measured", file=sys.stderr)
            continue
        # the base is the PR's target: `origin/<baseRefName>`, fetched too, so a fork PR measured against its pin
        # branch cannot report clean while deleting that branch's newer lines (engine#3071)
        base_ref = f"origin/{base}"
        rc, _o, err = run("git", "fetch", "-q", "origin", f"+refs/heads/{base}:refs/remotes/{base_ref}", cwd=root)
        if rc != 0:
            unreadable.append(f"#{n} {branch}: cannot fetch its base {base} ({err.strip()[:80]})")
            print(f"#{n} {branch}: CANNOT READ -- its base {base} could not be fetched", file=sys.stderr)
            continue
        try:
            f = stale_finding(root, ref, base_ref)
        except CannotTell as e:
            unreadable.append(f"#{n} {branch}: {e}")
            print(f"#{n} {branch}: CANNOT READ -- {e}", file=sys.stderr)
            continue
        if f is None:
            print(f"#{n} {branch}: merging this branch removes nothing {base} has")
            continue
        behind, reverted, carried, files = f
        findings += 1
        parts = []
        if reverted:
            parts.append(f"REVERTS {reverted} line(s) {base} added after the branch was cut")
        if carried:
            parts.append(f"removes {carried} line(s) {base} still carries")
        print(f"#{n} {branch}: BEHIND {behind}; merging it " + " and ".join(parts) +
              f" in {len(files)} file(s): {', '.join(files[:3])}{' …' if len(files) > 3 else ''}", file=sys.stderr)
    if findings:
        print(f"pr_branch_stale_check: {findings} open PR(s) would delete lines their base has "
              f"(measured with `git merge-tree --write-tree`, not a two-dot diff) -- rebase or re-cut before merging",
              file=sys.stderr)
    elif not unreadable:
        print("pr_branch_stale_check: no open PR would revert its base (every branch was read)")
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
