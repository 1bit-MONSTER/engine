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
import re
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
        # A list endpoint answers with a list. An error object (`{"message": ...}`, a rate limit, a renamed repo) is a
        # dict, and `out.extend(dict)` silently extends with the DICT'S KEYS - strings - which then explode in the
        # caller as `'str' object has no attribute 'get'` (this crashed the whole queue on 2026-11-14 07:13Z). Reported,
        # not skipped: a failed read must not come out as "no verdicts exist", which is exactly the false-negative this
        # tool exists to avoid.
        if not isinstance(got, list):
            msg = got.get('message', got) if isinstance(got, dict) else got
            raise RuntimeError(f"gh api {path} did not answer with a list: {msg}")
        if not got:
            break
        out.extend(got)
        if len(got) < PAGE:
            break
        page += 1
    return json.dumps(out)


# A verdict does not begin inside an inline code span. Lanes quote the verdict *format* (`` `VERDICT: READY at
# `abc1234` ``) and quote **other people's** verdicts when reconciling them, and both were being read as verdicts by the
# author. Measured 2026-11-14: this lane's own reconciliation comment on npu-kernels#70 quoted `VERDICT: NEEDS-CHANGE` at
# the head, so `--state` reported a READY PR as held by a CHANGES that no reviewer had written.
VERDICT_LINE = re.compile(r'^(?![^A-Za-z0-9]*`[^`]*VERDICT)[^A-Za-z0-9]*VERDICT\b', re.MULTILINE)


VERDICT_KIND = re.compile(r'^(?![^A-Za-z0-9]*`[^`]*VERDICT)[^A-Za-z0-9]*VERDICT\b[^\n]*', re.MULTILINE)


def verdict_kind(body):
    """'READY' / 'CHANGES' / None from a verdict line.

    Three forms are in use and a queue that reads only one of them is wrong about the other two: `VERDICT: READY at
    `abc1234``, `VERDICT: CHANGES at `abc1234``, and the two older shapes `VERDICT: APPROVE at `abc1234`` /
    `VERDICT at `abc1234`: READY`. The kind is the first token after the word VERDICT that is one of the known ones.
    """
    found = VERDICT_KIND.search(body or '')
    if not found:
        return None
    words = re.findall(r'[A-Z][A-Z_-]+', found.group(0))
    for word in words[1:]:
        if word in ('READY', 'APPROVE', 'APPROVED'):
            return 'READY'
        # `NEEDS-CHANGE`, `CHANGES-REQUESTED`, `NOT-READY`: a lane writes the verdict in prose as often as in the
        # two canonical words, and treating those as "no verdict" makes the newest RECOGNISED verdict win -- which
        # is how engine#2397 read as READY here while its newest line said `VERDICT: NEEDS-CHANGE at this head`.
        if word in ('CHANGES', 'CHANGES_REQUESTED', 'CHANGES-REQUESTED', 'CHANGE', 'NEEDS-CHANGE', 'NEEDS-CHANGES',
                    'REQUEST-CHANGES', 'NOT-READY'):
            return 'CHANGES'
    return 'OTHER'   # a verdict line whose kind this does not know: say so, never fall through to an older READY


def newest_verdict(reviews, comments, head):
    """(kind, timestamp, head_named) for the newest verdict object that names this head, or None.

    Newest across BOTH object types, because the merge gate holds on the newest one and the two are posted minutes
    apart by different lanes: on engine#2369 the same head carried four verdicts inside 35 seconds -- CHANGES, a
    comment CHANGES, a READY, another comment CHANGES -- and a tool that read reviews only would have called it
    reviewed-and-clean.
    """
    named = None      # newest verdict that names THIS head
    any_verdict = None  # newest verdict-shaped object, head named or not
    for item in list(reviews) + list(comments):
        body = item.get('body') or ''
        kind = verdict_kind(body)
        if kind is None:
            continue
        when = item.get('submitted_at') or item.get('created_at') or ''
        if any_verdict is None or when > any_verdict[1]:
            any_verdict = (kind, when)
        if head in body and (named is None or when > named[1]):
            named = (kind, when)
    if any_verdict is not None and (named is None or any_verdict[1] > named[1]):
        # A verdict newer than the last one that names the head: `VERDICT: NEEDS-CHANGE at this head` is a real
        # line a lane wrote on engine#2397 and it carries no sha, so the attributed answer there was an older
        # READY. Reporting the old READY as current is the failure the queue exists to prevent, so say what
        # actually happened instead of choosing between two wrong answers.
        return ('UNATTRIBUTED', any_verdict[1])
    return named


def reviewed_at(reviews, head):
    """True if any review OR COMMENT body carries a VERDICT line naming this head prefix.

    Two things this has to get right, both learned from a review of this tool:

    * a verdict posted as an **issue comment** counts. Most lanes put `VERDICT: ...` in a comment, not in a review
      object -- measured on the open queue, four PRs (llama.cpp#280, site#17/#16/#15) had a comment verdict and no
      review at all, so a reviews-only queue asked for reviews that already existed;
    * the token alone is not a verdict: `no VERDICT at abc1234 yet` contains both the word and the sha and means the
      opposite. The verdict has to START a line (markdown decoration before it is fine), which is the convention the
      lanes actually write.

    A verdict line carries the 7-hex head (`VERDICT: READY at `abc1234``); a body that names a longer sha covers the
    shorter prefix too, which is why this is a substring test on the head rather than an equality on the whole sha.
    """
    for r in reviews:
        body = r.get('body') or ''
        if head in body and VERDICT_LINE.search(body):
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
            # Both channels: a review body and an issue comment. Same list, because `reviewed_at` asks one question.
            revs = json.loads(gh(f"repos/{OWNER}/{repo}/pulls/{pr['number']}/reviews", paginate=True) or '[]')
            revs += json.loads(gh(f"repos/{OWNER}/{repo}/issues/{pr['number']}/comments", paginate=True) or '[]')
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
    # The 07:13Z crash, reproduced from the shape that caused it: an error object where a list belongs. A guard that
    # cannot fail is not a guard, so this drives `gh` itself (a stubbed binary) rather than a hand-built list.
    import tempfile as _tf, os as _os
    _d = _tf.mkdtemp()
    _stub = _os.path.join(_d, 'gh')
    with open(_stub, 'w') as f:
        f.write('#!/bin/sh\nprintf \'{"message":"API rate limit exceeded"}\'\n')
    _os.chmod(_stub, 0o700)
    _old = _os.environ.get('PATH', '')
    _os.environ['PATH'] = _d + ':' + _old
    try:
        gh('repos/x/y/pulls?state=open', paginate=True)
        crashed = 'no exception'
    except RuntimeError as e:
        crashed = 'rate limit' in str(e)
    finally:
        _os.environ['PATH'] = _old
    chk(True, crashed, 'a non-list page is reported, not extended into the queue as strings')
    # The false POSITIVE, measured on my own comment (npu-kernels#70, 07:5xZ): reconciling two verdicts quotes the one being
    # disagreed with, and the quote named the head. A quotation is not a verdict - in either direction.
    chk(False, reviewed_at([{'body': '`VERDICT: NEEDS-CHANGE` at `abc1234` measured:'}], 'abc1234'),
        'a backticked verdict being QUOTED is not a verdict at this head')
    chk(False, reviewed_at([{'body': 'Write it as `VERDICT: READY at <head>`'}], 'abc1234'),
        'quoting the verdict FORMAT is not a verdict either')
    chk(False, reviewed_at([{'body': 'looks good to me'}], 'abc1234'),
        'a review without a verdict line does not count')
    # The guard is the VERDICT word, not the head: removing it left --self-test at rc 0, because every arm above
    # passed a body that contained BOTH. A comment that quotes the head while reviewing is the shape that separates
    # them, and it is the common one -- reviewers quote the sha they looked at.
    chk(False, reviewed_at([{'body': 'I read abc1234 carefully and it looks fine to me'}], 'abc1234'),
        'a comment that NAMES the head but is not a verdict does not count')
    # The review of this tool found the arm above too weak: it carries no VERDICT token at all, so it passes even
    # when the matcher is `'VERDICT' in body`. This one has both tokens and still means the opposite.
    chk(False, reviewed_at([{'body': 'no VERDICT at abc1234 yet'}], 'abc1234'),
        'a sentence that DISCLAIMS a verdict is not a verdict')
    chk(False, reviewed_at([{'body': None}], 'abc1234'), 'a null body does not raise')
    chk(True, reviewed_at([{'body': '**VERDICT: READY** `abc1234`'}], 'abc1234'),
        'a verdict with markdown decoration before it still counts')
    chk(True, reviewed_at([{'body': 'VERDICT: CHANGES at `abc1234`'}], 'abc1234'),
        'CHANGES is a verdict: the PR was reviewed, the answer was no')
    print(f"pr_review_queue --self-test: {'ok' if rc == 0 else 'FAIL'}")
    return rc


def state(repos):
    """Per open PR: the newest verdict object, and whether it is a CHANGES at the current head.

    `queue()` answers "is anybody home"; this answers "what does the record say right now", which is the question a
    lane asks when a CHANGES is holding the gate and it cannot tell whether the author answered it.
    """
    rows = []
    for repo in repos:
        prs = json.loads(gh(f'repos/{OWNER}/{repo}/pulls?state=open&per_page={PAGE}', paginate=True) or '[]')
        for pr in prs:
            if pr.get('draft'):
                continue
            head = ((pr.get('head') or {}).get('sha') or '')[:7]   # verdicts name the 7-hex prefix
            revs = json.loads(gh(f"repos/{OWNER}/{repo}/pulls/{pr['number']}/reviews", paginate=True) or '[]')
            coms = json.loads(gh(f"repos/{OWNER}/{repo}/issues/{pr['number']}/comments", paginate=True) or '[]')
            newest = newest_verdict(revs, coms, head)
            rows.append({'repo': repo, 'number': pr['number'], 'head': head[:7],
                         'kind': newest[0] if newest else None,
                         'at': newest[1] if newest else None,
                         'title': (pr.get('title') or '')[:60]})
    return rows


def self_test_state():
    """The two shapes that made `--state` lie until they were run against the live queue."""
    problems = []

    def chk(want, got, what):
        if want != got:
            problems.append(f"{what}: want {want!r}, got {got!r}")

    revs = [{'body': 'VERDICT at `abc1234`: READY', 'submitted_at': '2026-11-14T03:37:05Z'}]
    coms = [{'body': 'VERDICT: NEEDS-CHANGE at this head -- the rule', 'created_at': '2026-11-14T03:57:04Z'}]
    chk(('UNATTRIBUTED', '2026-11-14T03:57:04Z'), newest_verdict(revs, coms, 'abc1234'),
        'a newer verdict that does not name the head is UNATTRIBUTED, not the older READY')
    chk('CHANGES', verdict_kind('VERDICT: NEEDS-CHANGE at this head'), 'NEEDS-CHANGE is a change request')
    chk('OTHER', verdict_kind('VERDICT: something nobody has seen before'),
        'an unknown kind is OTHER, never a silent none')

    # the attributed case must still prefer the newest NAMED verdict
    revs2 = [{'body': 'VERDICT: CHANGES at `abc1234`', 'submitted_at': '2026-11-14T03:00:00Z'},
             {'body': 'VERDICT: READY at `abc1234`', 'submitted_at': '2026-11-14T03:10:00Z'}]
    chk(('READY', '2026-11-14T03:10:00Z'), newest_verdict(revs2, [], 'abc1234'),
        'the newest verdict that names the head wins')

    if problems:
        for problem in problems:
            print('FAIL ' + problem)
        return 1
    print('pr_review_queue --state self-test: ok')
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repos', nargs='+', default=list(DEFAULT_REPOS))
    ap.add_argument('--min-age-hours', type=float, default=0.0)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--state', action='store_true',
                    help='per open PR: the newest verdict object across reviews and comments')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test() if not a.state else self_test_state()
    if a.state:
        rows = state(a.repos)
        if a.json:
            print(json.dumps(rows, indent=1))
            return 0
        for row in rows:
            kind = row['kind'] or 'none'
            print(f"  {row['repo']:12} #{row['number']:<5} head={row['head']}  newest={kind:8} "
                  f"{row['at'] or '-':20}  {row['title']}")
        held = [row for row in rows if row['kind'] == 'CHANGES']
        print(f"pr_review_queue --state: {len(rows)} open PR(s), {len(held)} whose newest verdict is CHANGES")
        return 0
    try:
        q = queue(a.repos, a.min_age_hours)
    except RuntimeError as e:
        print(f"pr_review_queue: {e}", file=sys.stderr)
        return 2
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
