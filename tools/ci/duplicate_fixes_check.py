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
"""duplicate_fixes_check.py -- which open PRs claim the same issue.

Three times on 2026-11-16/17 a second PR was opened for an issue another lane had already
started, and each time a human reviewer noticed rather than an instrument:

    #2778  #2784 / #2786        #2795  #2796 / #2797 / #2798        #2779  #2790 / #2793

The cost is not the duplicate diff, it is the review: two verdicts, a "which one is yours"
comment thread, and a merge order that nobody chose. The fleet already has the policy
(`two-open-prs-one-file`); what it lacked is the one-line answer to "does anything else claim
this issue right now", which is what this prints.

It reads the claim from the PR's **title and body** (`Fixes #N`, `Closes #N`, `Fixes and closes
#N`), because that is where a claim is made, and it reports every issue claimed by two or more
open PRs with the PRs' age and draft state, so the reader can tell "one is a draft awaiting a
slot" from "two lanes are both working".

A LIMIT, measured on the fleet: a PR that *works* an issue without a closing keyword is invisible here.
#2804 worked #2803 with a mention (`engine#2803, measured 2026-11-16 …`) while #2805 claimed it in the
title, and both were open; this tool reported nothing. The narrowing that causes it is deliberate (see
CLAIM), and the issue-side join (`gh api repos/O/R/issues/N/timeline`) returns both of those PRs *and*
mere mentions (#2795's timeline lists #2809 and #2807), so it is an annotation at best, not a trigger.

Usage:
    duplicate_fixes_check.py                    # the fleet's six repos
    duplicate_fixes_check.py --repos engine     # a subset
    duplicate_fixes_check.py --json
    duplicate_fixes_check.py --self-test        # hermetic: fixtures, no network
Exit: 0 no issue claimed twice, 1 at least one issue claimed by two or more open PRs, 2 cannot tell.
"""
import argparse
import json
import re
import subprocess
import sys

DEFAULT_REPOS = ('engine', 'llama.cpp', 'gpu-kernels', 'npu-kernels', 'lemonade', 'site')
OWNER = '1bit-MONSTER'
PAGE = 100

# The closing keyword must be IMMEDIATELY followed by the reference (`Fixes #12`, `fixes: #12`, `Closes #12`), which is
# also GitHub's own rule for closing. Two widenings were tried and both produced false claims on real bodies:
#   * a bare `#12` -- every PR that discusses an issue became a claimant;
#   * `keyword ... #12` with up to 24 characters between -- #2793's body says "Refs #2779 (the layer its fix does not
#     cover), #2728", and the noun "fix" plus a distant `#2728` read as a claim on 2728. Found by running the parser over
#     the real texts of tonight's seven PRs rather than over my own fixtures, which is why that check is in the PR body.
CLAIM = re.compile(r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s*#(\d+)', re.I)


# A claim written INSIDE code is documentation, not a claim. The PRs most likely to contain `Fixes #12` are the ones
# explaining how to write one -- #2809's own body pastes the syntax it looks for -- so without this the detector flags
# the PR that documents the convention, which is the one reading a lane cannot skip.
FENCE = re.compile(r"```.*?```", re.S)     # a fenced block
SPAN = re.compile(r"`[^`\n]*`")            # an inline code span


def claims(text):
    """The issue numbers a PR title+body claims to close, ignoring any claim written inside code."""
    text = SPAN.sub(" ", FENCE.sub(" ", text or ""))
    return sorted({int(n) for n in CLAIM.findall(text)})


def duplicates(prs):
    """[(issue, [pr, ...])] for every issue claimed by two or more of `prs`, oldest PR first.

    `prs` is [{'number', 'title', 'body', 'draft', 'age_hours'}]. A pair where one is a draft is still reported: a draft
    that claims an issue is a lane working on it, which is the thing the second lane cannot see.
    """
    by_issue = {}
    for pr in prs:
        for issue in claims(f"{pr.get('title') or ''}\n{pr.get('body') or ''}"):
            by_issue.setdefault(issue, []).append(pr)
    out = []
    for issue in sorted(by_issue):
        group = sorted(by_issue[issue], key=lambda p: p.get('number') or 0)
        if len(group) > 1:
            out.append((issue, group))
    return out


def describe(issue, group):
    lines = [f"#{issue} is claimed by {len(group)} open PR(s):"]
    for pr in group:
        draft = ' [draft]' if pr.get('draft') else ''
        age = pr.get('age_hours')
        age_s = f", {age:.1f} h old" if isinstance(age, (int, float)) else ''
        lines.append(f"  #{pr.get('number')} {pr.get('title') or ''}".rstrip() + f"{draft}{age_s}")
    lines.append("  one of these should close, or say in both threads which lands first")
    return "\n".join(lines)


def gh(path, paginate=False):
    """One page, or every page, of a `gh api` list endpoint (the same shape as pr_review_queue.py)."""
    if not paginate:
        return subprocess.run(['gh', 'api', path], capture_output=True, text=True).stdout
    out, page = [], 1
    while True:
        sep = '&' if '?' in path else '?'
        chunk = subprocess.run(['gh', 'api', f"{path}{sep}per_page={PAGE}&page={page}"],
                               capture_output=True, text=True).stdout
        try:
            got = json.loads(chunk or '[]')
        except json.JSONDecodeError:
            break
        if not isinstance(got, list):
            # A failed read must never read as "nothing claimed twice": that is the false negative this tool exists for.
            msg = got.get('message', got) if isinstance(got, dict) else got
            raise RuntimeError(f"gh api {path} did not answer with a list: {msg}")
        if not got:
            break
        out.extend(got)
        if len(got) < PAGE:
            break
        page += 1
    return json.dumps(out)


def open_prs(repo):
    import datetime
    raw = gh(f"repos/{OWNER}/{repo}/pulls?state=open&per_page={PAGE}", paginate=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    prs = []
    for p in json.loads(raw or '[]'):
        created = p.get('created_at')
        age = None
        if created:
            try:
                age = (now - datetime.datetime.fromisoformat(created.replace('Z', '+00:00'))).total_seconds() / 3600
            except ValueError:
                age = None
        prs.append({'number': p.get('number'), 'title': p.get('title') or '', 'body': p.get('body') or '',
                    'draft': bool(p.get('draft')), 'age_hours': age})
    return prs


def self_test():
    fails = 0

    def ok(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        fails += 0 if cond else 1

    # one claim, in the shapes lanes actually write
    ok(claims("pins: llama.cpp 1 -> 2 (Fixes #2757)") == [2757], "Fixes #N in a title is a claim")
    ok(claims("body text\n\nFixes #2778.") == [2778], "Fixes #N in a body is a claim")
    ok(claims("Closes #12 and fixes #13") == [12, 13], "two claims in one body are both read")
    ok(claims("Refs #2778 only") == [], "Refs is not a claim -- the whole point of not matching a bare #N")
    ok(claims("see #2778 for context") == [], "a bare mention is not a claim")
    ok(claims("Refs #2779 (the layer its fix does not cover), #2728 (where the defect came from)") == [],
       "a NOUN 'fix' followed by a distant #N is not a claim -- #2793's real body, which the 24-character gap did fire on")
    ok(claims("") == [] and claims(None) == [], "empty and None bodies claim nothing")
    ok(claims("use `Fixes #12` in your PR body") == [],
       "a claim inside an inline code span is documentation, not a claim")
    ok(claims("```\nFixes #12\n```") == [], "a claim inside a fenced block is documentation too")
    ok(claims("```\nFixes #12\n```\n\nCloses #34") == [34],
       "...and a real claim beside a fenced one still counts, which is the case a naive strip gets wrong")

    # the shape that happened three times tonight
    prs = [
        {'number': 2784, 'title': 'accel_sampler: write the summary by rename (Fixes #2778)', 'body': '', 'draft': False, 'age_hours': 1.0},
        {'number': 2786, 'title': 'diag: the sampler summary is renamed into place (engine#2778)', 'body': 'Fixes #2778', 'draft': False, 'age_hours': 0.5},
        {'number': 2790, 'title': 'unrelated', 'body': 'Fixes #2779', 'draft': False, 'age_hours': 2.0},
    ]
    dup = duplicates(prs)
    ok([i for i, _ in dup] == [2778], "one duplicate is reported, and only the claimed-twice issue")
    ok([p['number'] for p in dup[0][1]] == [2784, 2786], "the pair is named oldest first")
    ok('one of these should close' in describe(*dup[0]), "the report says what to do about it")

    # a draft claimant still counts: that is a lane working, which the second lane cannot see
    prs[1]['draft'] = True
    ok([p['number'] for p in duplicates(prs)[0][1]] == [2784, 2786], "a draft claimant is reported too")
    ok('[draft]' in describe(*duplicates(prs)[0]), "... and marked as a draft")

    # no claim, no finding -- the failure mode of a detector that always fires
    ok(duplicates([{'number': 1, 'title': 'a', 'body': 'Refs #5', 'draft': False},
                   {'number': 2, 'title': 'b', 'body': 'Refs #5', 'draft': False}]) == [],
       "two PRs that merely REFER to an issue are not a duplicate")

    # review of #2809: a repo that could not be read must not print the clean sentence, and must not be masked by a
    # duplicate found elsewhere. Both are one-liners that were wrong in the first version.
    ok(clean_line([], 0) is not None, "with nothing found and every repo read, the clean sentence prints")
    ok(clean_line([], 1) is None, "... but NOT when a repo could not be read (that repo is where a claimant would hide)")
    ok(clean_line([{'repo': 'engine'}], 0) is None, "... and not when something was found")
    ok(exit_code([], 0) == 0 and exit_code([{'repo': 'engine'}], 0) == 1, "0 none / 1 duplicate")
    ok(exit_code([{'repo': 'engine'}], 1) == 2, "cannot-tell outranks a duplicate rather than being swallowed by it")
    ok(exit_code([], 2) == 2, "and it is not a pass on its own either")

    print("self-test ok" if not fails else f"self-test FAILED ({fails})")
    return 0 if not fails else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repos', nargs='+', default=list(DEFAULT_REPOS))
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    found, failed = [], 0
    for repo in a.repos:
        try:
            prs = open_prs(repo)
        except (RuntimeError, json.JSONDecodeError) as e:
            print(f"duplicate_fixes: cannot tell for {repo}: {e}", file=sys.stderr)
            failed += 1
            continue
        for issue, group in duplicates(prs):
            found.append({'repo': repo, 'issue': issue,
                          'prs': [{'number': p['number'], 'title': p['title'], 'draft': p['draft'],
                                   'age_hours': p['age_hours']} for p in group]})
    if a.json:
        print(json.dumps(found, indent=1))
    else:
        for f in found:
            print(f"{f['repo']}: " + describe(f['issue'], f['prs']))
        clean = clean_line(found, failed)
        if clean:
            print(clean)
    return exit_code(found, failed)


def clean_line(found, failed):
    """The "nothing claimed twice" sentence, or None.

    It must not print when a repo could not be read: that repo is exactly where the third claimant would be, and the
    sentence would then be the false negative this tool's docstring forbids in its own words (review of #2809).
    """
    return None if found or failed else "duplicate_fixes: no open issue is claimed by more than one open PR"


def exit_code(found, failed):
    """2 beats 1 beats 0. A duplicate in one repo must not mask "cannot tell" in another (review of #2809)."""
    if failed:
        return 2
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
