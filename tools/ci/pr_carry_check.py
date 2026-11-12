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


def run(repo, refs, base):
    for ref in refs:
        resolve(repo, ref)
    resolve(repo, base)
    carried, landed, own = classify(repo, base, refs)
    for path, who in carried:
        print(f"CARRIED  {path}  -- identical in {', '.join(who)} and absent from {base}")
    for ref, path in landed:
        print(f"LANDED   {path}  -- {ref} changes it to what {base} already has")
    print()
    print(f"pr_carry_check: {len(refs)} ref(s) against {base}: carried={len(carried)} landed={len(landed)} own={len(own)}")
    return 1 if (carried or landed) else 0


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
