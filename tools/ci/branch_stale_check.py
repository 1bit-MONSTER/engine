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
"""branch_stale_check.py -- a branch that is behind, and what its merge would actually write.

A branch cut before another PR landed carries the older copy of every file that PR changed. Its endpoint diff
(`git diff base..branch`, two dots) shows those files as reverted, and that reading is wrong: a squash merge applies
`merge-base..branch`, a path the branch never changed has no hunk there, and the base's copy survives (engine#2410,
measured in a scratch repo and on #2021 x cmake/tests-pre-pr-checks.cmake; the fleet merges with
`gh pr merge --squash`).

So this check reads the merge itself (`git merge-tree --write-tree base head`) and reports:

  FAIL  the merge conflicts, or it would leave a path the branch never changed different from the base's copy
        (a real loss: the base's content would not survive);
  note  the branch is behind on N path(s) it never changed: the merge keeps the base's copies, but the branch's
        own tests ran against the older ones, so rebase before trusting them.

Paths changed on BOTH sides are where a conflict resolution can silently take the branch's side; this check does
not judge a resolution (compare it against both parents by hand, as on #2021/#2368).

Exit 0 no loss, 1 a conflict or a lost base change, 2 a ref could not be read (which is not a pass).
"""

import argparse
import os
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
    parser = argparse.ArgumentParser(description="Report what merging the branch would lose from the base.")
    parser.add_argument("--base", default="origin/main", help="base ref (default: origin/main)")
    parser.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    # engine#2443: the other two tools in tools/ci carry --self-test and these two kept their arms in
    # tests/*_test.py, so `--self-test` here answered "unrecognized arguments" -- which reads as "there is
    # no self-test" rather than "its arms live elsewhere". The arms stay where they are; this runs them.
    parser.add_argument("--self-test", action="store_true",
                        help="run the arms in tests/branch_stale_check_test.py and exit with their status")
    parser.add_argument("--quiet", action="store_true", help="only print failures")
    parser.add_argument("--limit", type=int, default=25, help="how many files to name (default: 25)")
    args = parser.parse_args()
    if args.self_test:
        return subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), "tests/branch_stale_check_test.py")]).returncode


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

    behind = []
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
            behind.append(path)

    # engine#2410: what lands is the merge, not the branch's endpoint, so read the merge.
    merged = git("merge-tree", "--write-tree", "--no-messages", args.base, args.head)
    if merged.returncode not in (0, 1) or not merged.stdout.strip():
        print(f"branch_stale_check: git merge-tree failed ({merged.stderr.strip()}) - cannot tell, and that is not a pass",
              file=sys.stderr)
        return 2
    merged_lines = merged.stdout.splitlines()
    merge_tree = merged_lines[0].strip()
    conflicted = sorted({line.split("\t", 1)[1] for line in merged_lines[1:] if "\t" in line})
    lost = [path for path in behind + added_by_base if blob_sha(merge_tree, path) != blob_sha(args.base, path)]

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
        print(f"branch_stale_check: base {args.base} {len(base_paths)} path(s), branch touches {len(touched)}; "
              f"behind on {len(behind)} it never changed (the base also added {len(added_by_base)} the branch "
              f"predates); the merge keeps the base's copy of {len(behind) + len(added_by_base) - len(lost)} of them")

    for path in conflicted[: args.limit]:
        print(f"FAIL branch_stale_check: {path} - the merge with the base conflicts here")
    for path in lost[: args.limit]:
        print(f"FAIL branch_stale_check: {path} - the branch never changed it, yet the merge would not keep the "
              f"base's copy")
    if conflicted or lost:
        print("branch_stale_check: rebase onto the base and resolve, then re-run", file=sys.stderr)
        return 1
    if behind and not args.quiet:
        for path in behind[: args.limit]:
            print(f"note branch_stale_check: {path} - behind the base (the merge keeps the base's copy)")
        if len(behind) > args.limit:
            print(f"note branch_stale_check: ... and {len(behind) - args.limit} more")
        print("branch_stale_check: nothing the base has is lost by the merge; the branch's own tests ran against the "
              "older copies above, so rebase before trusting them")
        return 0
    if not args.quiet:
        print("branch_stale_check: the merge keeps every base change the branch left alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
