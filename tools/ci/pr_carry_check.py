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

"""pr_carry_check.py -- find a PR that carries another PR's files, or one whose work is already on base.

A stacked branch merges before the PRs under it, so reviewing the stack reviews two decisions as
one. The same file arrives a third way: a PR whose own work is already on `base`, which reads as a
fourth duplicate of something that landed an hour ago. Both are invisible to a diffstat, because a
carried file looks exactly like a file the PR wrote.

    pr_carry_check.py <repo> <ref> <ref> ... [--base REF]

Refs are things you already fetched (`git fetch origin pull/N/head:refs/pr/N`); this tool never
touches the network and never writes the index, so it is safe to run in a worktree someone is
using. Exit 0 when every changed path is the PR's own, 1 when it is not, 2 on a usage error.

**The other half, so you do not have to hit both failures to find both tools.** This asks *"does the PR
ADD files that are not its own?"*. The complement is *"does it REMOVE lines that base still has?"* --
`tools/ci/registration_loss_check.py`, which is what to run when a branch reads MERGEABLE while quietly
deleting another PR's registrations. A head can pass one and fail the other: a re-base that takes another
lane's files fails this tool, and one that drops main's rows fails that one. Run both on a branch you did
not write.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

BASE = "origin/main"


def git(repo, *args):
    """stdout of a git command, and "" when git refuses -- callers check what they need."""
    out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def die(message):
    print(f"pr_carry_check: {message}", file=sys.stderr)
    sys.exit(2)


def resolve(repo, ref):
    sha = git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if not sha:
        die(f"{ref!r} is not a commit in {repo} -- fetch it first, this tool does not")
    return sha


def on_base(repo, base, path):
    """The blob `path` has on base, or None when base does not have the path at all."""
    return git(repo, "rev-parse", "--verify", f"{base}:{path}") or None


def changed(repo, base, ref):
    """{path: blob-on-ref} for every path this ref changes against the merge base with base."""
    mb = git(repo, "merge-base", base, ref)
    if not mb:
        die(f"no merge base between {base!r} and {ref!r}")
    out = {}
    for line in git(repo, "diff", "--name-only", mb, ref).splitlines():
        if line:
            out[line] = git(repo, "rev-parse", "--verify", f"{ref}:{line}")
    return out


def classify(repo, base, refs):
    """(carried, landed, own) rows: carried by 2+ refs and absent from base; landed = the blob on base."""
    per = {ref: changed(repo, base, ref) for ref in refs}
    carriers = {}
    for ref, paths in per.items():
        for path, blob in paths.items():
            if blob and on_base(repo, base, path) is None:
                carriers.setdefault((path, blob), []).append(ref)
    carried = sorted((path, tuple(sorted(refs_))) for (path, _), refs_ in carriers.items() if len(refs_) > 1)
    landed, own = [], []
    for ref, paths in per.items():
        for path, blob in paths.items():
            base_blob = on_base(repo, base, path)
            if blob and base_blob and blob == base_blob:
                landed.append((ref, path))
            elif (path, blob) not in carriers or len(carriers[(path, blob)]) < 2:
                own.append((ref, path))
    return carried, sorted(landed), own


def added_lines(repo, base, ref):
    """The lines this ref ADDS, as a set -- the unit two PRs share when one carries another's WORK."""
    mb = git(repo, "merge-base", base, ref)
    if not mb:
        die(f"no merge base between {base!r} and {ref!r}")
    out = set()
    for line in git(repo, "diff", "-U0", mb, ref).splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            out.add(line[1:].strip())   # indentation-only differences must not hide a copied block
    out.discard("")
    return out


WORK_FLOOR = 10   # a shared *block*; below this two PRs share boilerplate, not work


def shared_work(repo, base, refs):
    """[(smaller, bigger, n)] for each pair where one ref's added lines are a SUBSET of the other's.

    llama.cpp #333/#334 are the case: the same 36 added lines, but into a branch each, so the file-level
    comparison in classify() sees two independent PRs. A subset is the shape that cannot be coincidence --
    an identical block plus extra lines is still one PR taking another's work and extending it.
    """
    per = {ref: added_lines(repo, base, ref) for ref in refs}
    rows = []
    for i, a in enumerate(sorted(per)):
        for b in sorted(per)[i + 1:]:
            la, lb = per[a], per[b]
            if len(la) < WORK_FLOOR or len(lb) < WORK_FLOOR:
                continue
            if la <= lb or lb <= la:
                small, big = (a, b) if la <= lb else (b, a)
                rows.append((small, big, min(len(la), len(lb))))
    return rows


def run(repo, refs, base):
    for ref in refs:
        resolve(repo, ref)
    resolve(repo, base)
    carried, landed, own = classify(repo, base, refs)
    work = shared_work(repo, base, refs)
    for path, who in carried:
        print(f"CARRIED  {path}  -- identical in {', '.join(who)} and absent from {base}")
    for ref, path in landed:
        print(f"LANDED   {path}  -- {ref} changes it to what {base} already has")
    for small, big, n in work:
        print(f"SHARED   {n} added line(s) of {small} are also in {big}  -- one PR's work, two branches")
    print()
    print(f"pr_carry_check: {len(refs)} ref(s) against {base}: carried={len(carried)} landed={len(landed)} "
          f"own={len(own)} shared_work={len(work)}")
    return 1 if (carried or landed or work) else 0


def self_test():
    failures = 0

    def check_that(ok, what):
        nonlocal failures
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            failures += 1

    repo = tempfile.mkdtemp(prefix="pr-carry-check-")
    try:
        def run_git(*args):
            subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)

        run_git("init", "-q", "-b", "main")
        run_git("config", "user.email", "t@e")
        run_git("config", "user.name", "t")
        with open(os.path.join(repo, "shared.txt"), "w", encoding="utf-8") as f:
            f.write("base\n")
        with open(os.path.join(repo, "own.txt"), "w", encoding="utf-8") as f:
            f.write("base\n")
        run_git("add", "-A")
        run_git("commit", "-qm", "base")
        run_git("update-ref", "refs/heads/base", "HEAD")

        # A adds carried.txt and edits own.txt; B adds the same carried.txt bytes -- the #2286 shape.
        for branch in ("a", "b"):
            run_git("checkout", "-q", "-B", branch, "refs/heads/base")
            with open(os.path.join(repo, "carried.txt"), "w", encoding="utf-8") as f:
                f.write("one PR's work\n")
            if branch == "a":
                with open(os.path.join(repo, "own.txt"), "w", encoding="utf-8") as f:
                    f.write("a's own change\n")
            run_git("add", "-A")
            run_git("commit", "-qm", branch)

        # C branched before `later` gained carried.txt and adds the same bytes -- the "already landed"
        # shape (#2045 vs #2086): the PR re-adds a file main already has, because main got it by another merge.
        run_git("checkout", "-q", "-B", "c", "refs/heads/base")
        with open(os.path.join(repo, "carried.txt"), "w", encoding="utf-8") as f:
            f.write("one PR's work\n")
        run_git("add", "-A")
        run_git("commit", "-qm", "c")
        run_git("checkout", "-q", "-B", "later", "refs/heads/base")
        with open(os.path.join(repo, "carried.txt"), "w", encoding="utf-8") as f:
            f.write("one PR's work\n")
        run_git("add", "-A")
        run_git("commit", "-qm", "later gained the file by another merge")

        carried, landed, own = classify(repo, "refs/heads/base", ["a", "b"])
        check_that(len(carried) == 1 and carried[0][0] == "carried.txt",
                   "a file added identically by two refs is CARRIED")
        check_that(len(carried[0][1]) == 2, "and it names both refs")
        check_that(not landed, "a changed file is not reported as LANDED")
        check_that(("a", "own.txt") in own, "a ref's own edit stays its own")
        check_that(("b", "own.txt") not in own, "the ref that did not edit own.txt has no row for it")

        run_git("checkout", "-q", "-B", "e", "refs/heads/base")
        with open(os.path.join(repo, "only.txt"), "w", encoding="utf-8") as f:
            f.write("e's own new file\n")
        run_git("add", "-A")
        run_git("commit", "-qm", "e")
        carried2, _, own2 = classify(repo, "refs/heads/base", ["a", "e"])
        check_that(not carried2, "a file added by one ref alone is not CARRIED")
        check_that(("e", "only.txt") in own2, "and it stays the ref's own")
        carried3, landed3, _ = classify(repo, "refs/heads/later", ["c"])
        check_that(("c", "carried.txt") in landed3, "a ref that re-adds what base already has is LANDED")
        check_that(not carried3, "and it is not also CARRIED (one ref, not two)")

        # the #333/#334 shape: the SAME added block into a different file, so the file-level test sees
        # two independent PRs and only the added lines give it away
        for branch, fname in (("w1", "one.cpp"), ("w2", "two.cpp")):
            run_git("checkout", "-q", "-B", branch, "refs/heads/base")
            body = "".join(f"    row_{i}(x);\n" for i in range(12))
            with open(os.path.join(repo, fname), "w", encoding="utf-8") as f:
                f.write("void t() {\n" + body + "}\n")
            run_git("add", "-A")
            run_git("commit", "-qm", branch)
        carried5, landed5, _ = classify(repo, "refs/heads/base", ["w1", "w2"])
        check_that(not carried5 and not landed5, "the same work into two files is neither carried nor landed")
        work5 = shared_work(repo, "refs/heads/base", ["w1", "w2"])
        check_that(len(work5) == 1 and work5[0][2] == 14, "but it IS shared work: one row, 14 added lines (the block and its braces)")

        # below the floor two PRs share boilerplate, not work
        for branch, fname in (("b1", "b1.cpp"), ("b2", "b2.cpp")):
            run_git("checkout", "-q", "-B", branch, "refs/heads/base")
            with open(os.path.join(repo, fname), "w", encoding="utf-8") as f:
                f.write("#include <cstdio>\n#include <vector>\nint main() { return 0; }\n")
            run_git("add", "-A")
            run_git("commit", "-qm", branch)
        check_that(not shared_work(repo, "refs/heads/base", ["b1", "b2"]),
                   "three shared boilerplate lines are below the work floor")

        # a subset counts: one PR takes the block and extends it
        check_that(len(shared_work(repo, "refs/heads/base", ["w1", "b1"])) == 0,
                   "disjoint changes share no work")

        # A path that one side deletes must not read as carried: there is no blob to compare.
        run_git("checkout", "-q", "-B", "d", "refs/heads/base")
        run_git("rm", "-q", "shared.txt")
        run_git("commit", "-qm", "d")
        carried4, landed4, _ = classify(repo, "refs/heads/base", ["a", "d"])
        check_that(not carried4 and not landed4, "a deletion on one side is neither carried nor landed")
    finally:
        shutil.rmtree(repo, ignore_errors=True)

    print(f"pr_carry_check --self-test: {'FAIL' if failures else 'ok'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="find a PR that carries another PR's files, or one whose work is already on base")
    parser.add_argument("repo", nargs="?")
    parser.add_argument("refs", nargs="*")
    parser.add_argument("--base", default=BASE, help=f"what a PR is against (default: {BASE})")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.repo or not args.refs:
        parser.error("a repo and at least one ref are required")
    return run(args.repo, args.refs, args.base)


if __name__ == "__main__":
    sys.exit(main())
