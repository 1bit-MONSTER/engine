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
"""pr_review_queue.py -- which open PRs still need a verdict at their CURRENT head.

A review is a claim about a head. So the question is not "has this PR been reviewed" but "does any review name the head
this PR has now" -- and a PR that has moved since its verdict is back in the queue, which is the whole point of pinning
the 7-hex head in the verdict line.

Two things this gets right that a hand-rolled `gh api` loop gets wrong, both of them measured tonight:

  * **`gh api` defaults to `per_page=30`.** A PR with 30 reviews silently loses the newest ones, so a verdict that
    exists reads as missing -- and a reviewer repeats work, or asserts on the record that a verdict was never given.
    Both list endpoints here page until exhausted.
  * **Merged and draft PRs are not in the queue.** A draft is not asking for a verdict, and a merged PR cannot take one.

Usage:
    pr_review_queue.py                       # the fleet's six repos
    pr_review_queue.py --repos engine site   # a subset
    pr_review_queue.py --min-age-hours 1     # only those past the 1 h gate
    pr_review_queue.py --json
    pr_review_queue.py --self-test
"""

import argparse
import json
import subprocess
import sys

DEFAULT_REPOS = ('engine', 'llama.cpp', 'gpu-kernels', 'npu-kernels', 'lemonade', 'site')
OWNER = '1bit-MONSTER'
PAGE = 100


def gh(path, paginate=False):
    """One page, or every page, of a `gh api` list endpoint."""
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
        if not got:
            break
        out.extend(got)
        if len(got) < PAGE:
            break
        page += 1
    return json.dumps(out)


def reviewed_at(reviews, head):
    """True if any review body carries a VERDICT naming this head prefix.

    A verdict line carries the 7-hex head (`VERDICT: READY at `abc1234``); a body that names a longer sha covers the
    shorter prefix too, which is why this is a substring test on the head rather than an equality on the whole sha.
    """
    for r in reviews:
        body = r.get('body') or ''
        if 'VERDICT' in body and head in body:
            return True
    return False


def queue(repos, min_age_hours=0.0, now_epoch=None):
    """[{repo, number, head, age_hours, title}] for open, non-draft PRs with no verdict at their head."""
    import datetime
    now_epoch = now_epoch if now_epoch is not None else datetime.datetime.now(datetime.timezone.utc).timestamp()
    out = []
    for repo in repos:
        for pr in json.loads(gh(f'repos/{OWNER}/{repo}/pulls?state=open', paginate=True) or '[]'):
            if pr.get('draft'):
                continue
            head = (pr.get('head') or {}).get('sha', '')
            if not head:
                continue
            revs = json.loads(gh(f"repos/{OWNER}/{repo}/pulls/{pr['number']}/reviews", paginate=True) or '[]')
            if reviewed_at(revs, head[:7]):
                continue
            created = datetime.datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00')).timestamp()
            age = (now_epoch - created) / 3600.0
            if age < min_age_hours:
                continue
            out.append({'repo': repo, 'number': pr['number'], 'head': head[:7],
                        'age_hours': round(age, 2), 'title': (pr.get('title') or '')[:48]})
    return sorted(out, key=lambda r: -r['age_hours'])


def self_test():
    rc = 0

    def chk(want, got, what):
        nonlocal rc
        if want == got:
            print(f"ok   {what}")
        else:
            print(f"FAIL {what}: want {want}, got {got}")
            rc = 1

    chk(True, reviewed_at([{'body': 'VERDICT: READY at `abc1234`'}], 'abc1234'), 'a verdict naming the head counts')
    chk(True, reviewed_at([{'body': 'VERDICT: READY at `abc1234def`'}], 'abc1234'),
        'a verdict naming a longer sha of the same head counts too')
    chk(False, reviewed_at([{'body': 'VERDICT: READY at `9999999`'}], 'abc1234'),
        'a verdict at a DIFFERENT head does not count: that is the case the queue exists for')
    chk(False, reviewed_at([{'body': 'looks good to me'}], 'abc1234'),
        'a review without a verdict line does not count')
    # The guard is the VERDICT word, not the head: removing it left --self-test at rc 0, because every arm above
    # passed a body that contained BOTH. A comment that quotes the head while reviewing is the shape that separates
    # them, and it is the common one -- reviewers quote the sha they looked at.
    chk(False, reviewed_at([{'body': 'I read abc1234 carefully and it looks fine to me'}], 'abc1234'),
        'a comment that NAMES the head but is not a verdict does not count')
    chk(False, reviewed_at([{'body': None}], 'abc1234'), 'a null body does not raise')
    chk(True, reviewed_at([{'body': 'VERDICT: CHANGES at `abc1234`'}], 'abc1234'),
        'CHANGES is a verdict: the PR was reviewed, the answer was no')
    print(f"pr_review_queue --self-test: {'ok' if rc == 0 else 'FAIL'}")
    return rc


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repos', nargs='+', default=list(DEFAULT_REPOS))
    ap.add_argument('--min-age-hours', type=float, default=0.0)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    q = queue(a.repos, a.min_age_hours)
    if a.json:
        print(json.dumps(q, indent=1))
        return 0
    if not q:
        print('pr_review_queue: every open PR has a verdict at its current head')
        return 0
    for r in q:
        print(f"  {r['repo']:12} #{r['number']:<5} {r['age_hours']:6.2f}h  head={r['head']}  {r['title']}")
    print(f"pr_review_queue: {len(q)} open PR(s) with no verdict at their current head")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
