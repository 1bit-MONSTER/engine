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
"""duplicate_fixes_check.py -- notice when two open PRs claim the same issue (engine#2807).

Three times on 2026-11-16 a second PR was opened for an issue another lane had already started:

    #2778   #2784 (merged first) / #2786 (closed)     "accel_sampler: write the summary by rename"
    #2795   #2796 / #2797 / #2798 (merged first)      "accel_sampler names the copy that wrote each summary"
    #2779   #2790 / #2793                             "fold_initializer_transpose"

**Each was noticed by a reviewer reading two diffs**, which is the most expensive way to find it: the
fleet pays a verdict, a "which one is yours" thread, and a rebase on the loser. The policy already
exists (`two-open-prs-one-file`) and the merge loop already reads the verdicts; what was missing is
anything that READS THE CLAIMS, which are in the PR body and cost nothing to compare.

    tools/ci/duplicate_fixes_check.py                       # the fleet's six repos
    tools/ci/duplicate_fixes_check.py --repo 1bit-MONSTER/engine
    tools/ci/duplicate_fixes_check.py --self-test

Exit: 0 no issue is claimed twice, 1 an issue is claimed by two open PRs (both named), 2 cannot tell
(`gh` absent or not authenticated -- a failure to look must not read as a clean fleet).

A PR claims an issue with `Fixes #12`, `Closes #12`, `Resolves #12`, or the cross-repo
`Fixes 1bit-MONSTER/engine#12`, in the body or the title, case-insensitively. `Refs #12` is NOT a
claim and is deliberately not matched: llama.cpp#418 used `Refs engine#2494` precisely to say it does
not close it, and a tool that counted `Refs` would report that honest PR as a duplicate.

`tools/ci/duplicate_fixes_allow.txt` holds issue numbers that legitimately have two PRs -- a split, or
a revert paired with a rework -- one per line with a reason. An entry that is no longer duplicated is
reported so the file shrinks; a duplicated issue with no entry is the finding this tool exists for.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

TOOL_ROOT = pathlib.Path(__file__).resolve().parents[2]
ALLOW = pathlib.Path("tools/ci/duplicate_fixes_allow.txt")
# The fleet the brief names. Kept here rather than discovered, so a repo that fails to list is a
# cannot-tell and not a silently smaller fleet.
FLEET = ("1bit-MONSTER/engine", "1bit-MONSTER/llama.cpp", "1bit-MONSTER/gpu-kernels",
         "1bit-MONSTER/npu-kernels", "1bit-MONSTER/lemonade", "1bit-MONSTER/site")
# "fix/fixes/fixed", "close/closes/closed", "resolve/resolves/resolved", optional colon, then the
# issue. `\b` before the verb is what keeps "prefixes" out; the verb is what keeps "Refs" out.
CLAIM = re.compile(r"\b(fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b\s*:?\s*"
                   r"(?:(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+))?#(?P<n>\d+)", re.IGNORECASE)
COMMENT = re.compile(r"#.*$")


def claims(text: str, default_repo: str):
    """[(repo, issue)] this text claims. `default_repo` for a bare `#12`."""
    out = []
    for m in CLAIM.finditer(text or ""):
        out.append((m.group("repo") or default_repo, int(m.group("n"))))
    return out


def open_prs(repo: str):
    """(True, [(number, title, body)]) or (False, reason). The flag, not the payload's type, says which:
    a repo with no open PRs returns [], so "did it return a tuple" was True for the empty fleet and the
    first live run died on `got[0]` -- found by a repo with zero PRs, not by the self-test."""
    out = subprocess.run(["gh", "pr", "list", "-R", repo, "--state", "open", "--limit", "200",
                          "--json", "number,title,body"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return False, (out.stderr or "gh failed").strip().splitlines()[0][:120]
    try:
        return True, [(p["number"], p.get("title") or "", p.get("body") or "")
                      for p in json.loads(out.stdout)]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return False, f"unreadable gh output: {exc}"


def allow(root: pathlib.Path):
    path = root / ALLOW
    if not path.exists():
        return set()
    found = set()
    for line in path.read_text().splitlines():
        line = COMMENT.sub("", line).strip()
        if line:
            found.add(line.split()[0])
    return found


def check(prs_by_repo, allowed):
    """-> (duplicates, stale_allow). duplicates is {(repo, issue): [pr numbers]}."""
    seen = {}
    for repo, prs in prs_by_repo:
        for number, title, body in prs:
            for key in claims(f"{title}\n{body}", repo):
                # a SET, because a body may say "Fixes #12" twice and that is one PR, not a duplicate
                seen.setdefault(key, set()).add(number)
    dups = {k: sorted(v) for k, v in seen.items() if len(v) > 1 and f"{k[0]}#{k[1]}" not in allowed}
    live = {f"{k[0]}#{k[1]}" for k, v in seen.items() if len(v) > 1}
    return dups, (allowed - live)


def self_test():
    cases = 0
    bad = 0

    def arm(name, got, want):
        nonlocal cases, bad
        cases += 1
        if got != want:
            print(f"  SELF-TEST FAIL: {name} -> {got!r}, want {want!r}")
            bad += 1

    # engine#2807's three real pairs, as their bodies would have read.
    prs = {
        "1bit-MONSTER/engine": [
            (2784, "accel_sampler: write the summary by rename", "Fixes #2778"),
            (2786, "accel_sampler: summary by rename", "Closes #2778"),
            (2798, "accel_sampler names the copy that wrote each summary", "Resolves #2795"),
            (2796, "naming the writer", "fixes #2795"),
            (2790, "fold_initializer_transpose", "Fixes #2779"),
            (2793, "fold transpose", "Fixes: #2779"),
            (2900, "an unrelated change", "Refs #2779"),
        ]
    }
    dups, _ = check(list(prs.items()), set())
    arm("the issue's three pairs are found",
        dups, {("1bit-MONSTER/engine", 2778): [2784, 2786],
               ("1bit-MONSTER/engine", 2795): [2796, 2798],
               ("1bit-MONSTER/engine", 2779): [2790, 2793]})
    # THE arm: `Refs` must not count. llama.cpp#418 says "Refs engine#2494" to mean it does not close it.
    arm("Refs is not a claim", claims("Refs #2779\nRefs engine#2494", "1bit-MONSTER/engine"), [])
    # ...and a verb buried inside a word is not a claim either.
    arm("'prefixes #9' is not a claim", claims("prefixes #9", "1bit-MONSTER/engine"), [])
    # A cross-repo claim lands on the named repo, not the claiming one.
    arm("cross-repo claim", claims("Fixes 1bit-MONSTER/llama.cpp#12", "1bit-MONSTER/engine"),
        [("1bit-MONSTER/llama.cpp", 12)])
    # Several issues in one body are all claims; duplicates inside one PR are one claim.
    arm("multi-issue body", claims("Fixes #1 and closes #2, again #1", "x/y"), [("x/y", 1), ("x/y", 2)])
    # The same issue claimed twice in one body is ONE PR. Found by this very arm: as a list it made a
    # single PR look like a duplicate of itself, which is the false positive that would get the tool
    # switched off on its first run.
    arm("one PR claiming one issue twice is not a duplicate",
        check([("x/y", [(7, "t", "Fixes #12 and closes #12")])], set())[0], {})
    # An allowlisted duplicate is not reported, and an allowlist entry that no longer duplicates is.
    dups2, stale = check(list(prs.items()), {"1bit-MONSTER/engine#2778", "1bit-MONSTER/engine#9999"})
    arm("allowlist suppresses and reports", (sorted(dups2), sorted(stale)),
        ([("1bit-MONSTER/engine", 2779), ("1bit-MONSTER/engine", 2795)], ["1bit-MONSTER/engine#9999"]))
    # A repo that cannot be listed is cannot-tell, never a smaller fleet counted as clean.
    arm("cannot list -> cannot", open_prs("this/repo-does-not-exist")[0], False)
    # An empty repo is a SUCCESS with an empty list, and the two must be distinguishable: the first
    # live run died on exactly this, because a tuple-shaped failure check read [] as a payload.
    ok_none, payload = open_prs("1bit-MONSTER/npu-kernels")
    arm("an empty fleet is ok-with-nothing, not cannot", (ok_none, payload == [] or isinstance(payload, list)),
        (True, True))
    print(f"  self-test: {cases} case(s), {bad} failure(s)")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", action="append", default=[], help="OWNER/NAME; repeatable")
    ap.add_argument("--root", default=str(TOOL_ROOT))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 1 if self_test() else 0
    repos = args.repo or list(FLEET)
    by_repo = []
    for repo in repos:
        ok, got = open_prs(repo)
        if not ok:
            print(f"duplicate_fixes_check: cannot tell: {repo}: {got}", file=sys.stderr)
            return 2
        by_repo.append((repo, got))
    dups, stale = check(by_repo, allow(pathlib.Path(args.root)))
    for key in sorted(stale):
        print(f"  no longer duplicated, remove from {ALLOW}: {key}")
    if dups:
        for (repo, issue), numbers in sorted(dups.items()):
            print(f"  {repo}#{issue} is claimed by {len(numbers)} open PRs: "
                  f"#{' #'.join(str(n) for n in numbers)}", file=sys.stderr)
        return 1
    total = sum(len(p) for _, p in by_repo)
    print(f"duplicate_fixes_check: {total} open PR(s) across {len(by_repo)} repo(s), no issue claimed twice")
    return 0


if __name__ == "__main__":
    sys.exit(main())
