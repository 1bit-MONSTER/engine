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
"""branch_stale_check.py -- a branch that is behind carries older copies of files it never touched.

The fleet squash-merges, so the branch's tree becomes main's tree for every path in the diff. On a branch cut
before another PR landed, the endpoint diff lists that PR's files as *modified* -- the branch's copy is the older
one -- and the squash reverts them. Five PRs on 2026-11-13/14 had exactly that shape: #358 would have removed
llama.cpp#357's alignment refusal and its case, #361 would have removed #355, #2021 would have removed #2387's
ctest row, #2387 would have removed #2366's arms, #2397 would have removed #2021's single-offset guard.

`git diff base...branch` (three dots) answers "what did the branch change", which is why it does not show this: the
files it misses are the ones the branch never changed. This check asks the other question directly.

  for every path the base has that the branch does NOT change:
      is the branch's copy byte-identical to the base's?

A difference there cannot be the branch's intent -- it did not touch the file -- so it is a stale copy, and merging
it would write the old content over the base's. Files the branch does change are excluded, so a legitimate
divergence is never reported.

Exit 0 quiet, 1 stale files found, 2 a ref could not be read (which is not a pass).
"""

import argparse
import subprocess
import sys


def git(*args, cwd=None):
    return subprocess.run(("git",) + args, capture_output=True, text=True, cwd=cwd)


def ref_exists(rev):
    return git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}").returncode == 0


def blob_sha(rev, path):
    """The blob id at rev:path, or None when the path does not exist there."""
    result = git("rev-parse", "--verify", "--quiet", f"{rev}:{path}")
    return result.stdout.strip() if result.returncode == 0 else None


def paths_at(rev):
    listed = git("ls-tree", "-r", "--name-only", "-z", rev)
    if listed.returncode != 0:
        return None
    return [p for p in listed.stdout.split("\0") if p]


def touched_by_branch(base, head, merge_base):
    """Paths the branch itself changed, relative to where it diverged."""
    result = git("diff", "--name-only", "-z", f"{merge_base}...{head}")
    if result.returncode != 0:
        return None
    return {p for p in result.stdout.split("\0") if p}


def main():
    parser = argparse.ArgumentParser(description="Report files where the branch carries an older copy.")
    parser.add_argument("--base", default="origin/main", help="base ref (default: origin/main)")
    parser.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    parser.add_argument("--limit", type=int, default=25, help="how many files to name (default: 25)")
    args = parser.parse_args()

    for label, rev in (("base", args.base), ("head", args.head)):
        if not ref_exists(rev):
            print(f"branch_stale_check: {label} ref '{rev}' cannot be read - cannot tell, and that is not a pass",
                  file=sys.stderr)
            return 2

    merge_base = git("merge-base", args.base, args.head)
    if merge_base.returncode != 0:
        print(f"branch_stale_check: no merge base for {args.base} and {args.head} ({merge_base.stderr.strip()})",
              file=sys.stderr)
        return 2
    merge_base = merge_base.stdout.strip()

    base_paths = paths_at(args.base)
    head_paths = paths_at(args.head)
    if base_paths is None or head_paths is None:
        print("branch_stale_check: could not list one side - cannot tell, and that is not a pass", file=sys.stderr)
        return 2

    touched = touched_by_branch(args.base, args.head, merge_base)
    if touched is None:
        print("branch_stale_check: could not diff the branch against its merge base", file=sys.stderr)
        return 2

    stale = []
    added_by_base = []
    for path in base_paths:
        base_blob = blob_sha(args.base, path)
        head_blob = blob_sha(args.head, path)
        if path in touched:
            # The branch's own change. A file it deleted on purpose lands here too, which is why a divergence on
            # a touched path is never reported.
            continue
        if head_blob is None:
            # Present on the base, absent at the head, and the branch never touched it: the file arrived on the
            # base after the fork, and a merge supplies it. Harmless, so it is counted and not failed -- an arm
            # that can only fire on the harmless case is an arm that would teach readers to ignore the check.
            added_by_base.append(path)
        elif base_blob != head_blob:
            stale.append(path)

    # The check reads commits (`git show <ref>:<path>`). Uncommitted work is invisible to it, so say so rather
    # than letting a dirty tree read as a clean one -- measured while wiring it: an uncommitted stale copy
    # produced "no file differs" and nothing else.
    if args.head == "HEAD":
        dirty = git("status", "--porcelain")
        changed = [line for line in dirty.stdout.splitlines() if line.strip() and not line.startswith("??")]
        if changed and not args.quiet:
            print(f"branch_stale_check: HEAD is what was read; {len(changed)} uncommitted path(s) are NOT included "
                  f"- commit them and re-run to check the branch you are handing over")

    if not args.quiet:
        print(f"branch_stale_check: base {args.base} {len(base_paths)} path(s), branch touches {len(touched)} of them; "
              f"{len(stale)} older cop(ies) (the base also added {len(added_by_base)} path(s) the branch predates, "
              f"which a merge supplies)")

    for path in stale[: args.limit]:
        print(f"FAIL branch_stale_check: {path} - the branch carries an older copy and never changed it")
    if len(stale) > args.limit:
        print(f"FAIL branch_stale_check: … and {len(stale) - args.limit} more older cop(ies)")
    if stale:
        print("branch_stale_check: rebase onto the base - a squash merge of this branch would write these older "
              "copies over it", file=sys.stderr)
        return 1
    if not args.quiet:
        print("branch_stale_check: no file the branch left alone differs from the base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
