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
"""Post a PR verdict that the fleet's tooling can read, and refuse the four ways it goes wrong.

Every lane shares one GitHub account, and a verdict is the fleet's unit of review. Four failures cost time on
2026-11-15/16, each measured, and each is a refusal here rather than a review afterwards:

  1. **A pending review blocks everyone.** GitHub allows one PENDING review per user per PR. Another lane's draft
     rejects every post with "User can only have one pending review per pull request", and a lane that does not
     know the rule reads it as a transient error. `--check` names the pending id to clear.
  2. **A verdict must name the head it is about.** `pr_review_queue` matches the 7-hex head in the first line, so a
     body starting "VERDICT: READY" is invisible and the PR stays in the queue: measured twice.
  3. **A stale body reviews the previous head.** Branches moved under me twice in one night; the body's sha must be
     the current head, or the verdict is about code that is gone.
  4. **An issue comment is not a review.** `gh pr view --json comments` lists issue comments only, so a posted
     review looks like a failure and gets posted twice: measured once, and the duplicate cannot be deleted.

Usage:
    tools/ci/pr_verdict.py --check REPO PR                 # pending review? head? verdicted at this head?
    tools/ci/pr_verdict.py --post REPO PR BODY_FILE VERDICT [--dry-run]

`--post` refuses (exit 2) when the body's first line does not carry the current head, or when a pending review
exists; it posts and then re-reads to prove the review landed with this head in it.
"""

import argparse
import json
import re
import subprocess
import sys

HEAD_RE = re.compile(r"\b([0-9a-f]{7,40})\b")
VERDICTS = ("READY", "CHANGES")


def first_line_sha(body):
    """The head a verdict is about: the 7-hex in the FIRST line, which is what pr_review_queue matches."""
    first = body.splitlines()[0] if body.strip() else ""
    m = HEAD_RE.search(first)
    return m.group(1) if m else None


def verdict_in_first_line(body):
    first = body.splitlines()[0] if body.strip() else ""
    for v in VERDICTS:
        if v in first:
            return v
    return None


def classify(body, head):
    """(ok, reason) for a body against a head -- pure, so the self-test can drive every case."""
    if not body.strip():
        return False, "empty body"
    sha = first_line_sha(body)
    if sha is None:
        return False, "the first line names no head: pr_review_queue matches the 7-hex there and will not see this"
    if not head.startswith(sha) and not sha.startswith(head[:7]):
        return False, f"the first line names {sha}, the head is {head[:7]}: this would review code that is gone"
    if verdict_in_first_line(body) is None:
        return False, f"the first line names no verdict ({', '.join(VERDICTS)}); a reader cannot tell what it says"
    return True, "ok"


def pending_ids(reviews):
    """Ids of PENDING reviews from a /pulls/N/reviews payload: the ones that block every other lane's post."""
    return [r["id"] for r in reviews if r.get("state") == "PENDING"]


def verdicted_at(reviews_and_comments, head):
    """Whether any review or issue comment carries this head with a verdict word in it."""
    for x in reviews_and_comments:
        b = x.get("body") or ""
        if head[:7] in b and any(v in b for v in VERDICTS):
            return True
    return False


def gh(*args):
    p = subprocess.run(["gh", *args], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def api_json(path):
    rc, out, err = gh("api", path)
    if rc != 0:
        print(f"pr_verdict: cannot read {path}: {err.strip()[:120]}", file=sys.stderr)
        return None
    return json.loads(out)


def do_check(repo, pr):
    head = (api_json(f"repos/{repo}/pulls/{pr}") or {}).get("head", {}).get("sha", "")
    if not head:
        return 2
    reviews = api_json(f"repos/{repo}/pulls/{pr}/reviews?per_page=100") or []
    comments = api_json(f"repos/{repo}/issues/{pr}/comments?per_page=100") or []
    print(f"head {head[:7]}")
    pend = pending_ids(reviews)
    if pend:
        print(f"PENDING review(s) {pend}: every lane's post to this PR will be rejected until they are cleared")
        print(f"  clear with: gh api -X DELETE repos/{repo}/pulls/{pr}/reviews/<id>")
    print(f"verdicted at this head: {'yes' if verdicted_at(reviews + comments, head) else 'NO'}")
    return 0


def do_post(repo, pr, body_file, verdict, dry_run):
    head = (api_json(f"repos/{repo}/pulls/{pr}") or {}).get("head", {}).get("sha", "")
    if not head:
        return 2
    try:
        with open(body_file, encoding="utf-8") as f:
            body = f.read()
    except OSError as e:
        print(f"pr_verdict: {e}", file=sys.stderr)
        return 2
    ok, why = classify(body, head)
    if not ok:
        print(f"pr_verdict: refusing to post: {why}", file=sys.stderr)
        return 2
    pend = pending_ids(api_json(f"repos/{repo}/pulls/{pr}/reviews?per_page=100") or [])
    if pend:
        print(f"pr_verdict: refusing to post: pending review(s) {pend} block it "
              f"(clear: gh api -X DELETE repos/{repo}/pulls/{pr}/reviews/<id>)", file=sys.stderr)
        return 2
    if dry_run:
        print(f"would post {verdict} about {head[:7]} to {repo}#{pr}")
        return 0
    rc, out, err = gh("pr", "review", str(pr), "--repo", repo, "--comment", "--body-file", body_file)
    if rc != 0:
        print(f"pr_verdict: post failed: {(err or out).strip()[:200]}", file=sys.stderr)
        return 2
    if not verdicted_at(api_json(f"repos/{repo}/pulls/{pr}/reviews?per_page=100") or [], head):
        print(f"pr_verdict: posted, but no review at {head[:7]} is readable back: check the PR by hand", file=sys.stderr)
        return 2
    print(f"posted {verdict} about {head[:7]} (read back)")
    return 0


def self_test():
    H = "abcdef1234567890"
    assert first_line_sha("abcdef1 READY — ok\nmore") == "abcdef1"
    assert first_line_sha("VERDICT: READY\nabcdef1 in the body") is None          # the trap: sha not in line 1
    assert verdict_in_first_line("abcdef1 CHANGES — no") == "CHANGES"
    assert verdict_in_first_line("abcdef1 maybe") is None

    ok, why = classify("abcdef1 READY — one holder verified", H)
    assert ok, why
    ok, why = classify("VERDICT: READY\nthe head is abcdef1", H)
    assert not ok and "first line names no head" in why, why
    ok, why = classify("1234567 READY", H)                                       # stale head
    assert not ok and "code that is gone" in why, why
    ok, why = classify("abcdef1 approved", H)                                     # no verdict word in line 1
    assert not ok and "no verdict" in why, why
    ok, why = classify("", H)
    assert not ok and "empty" in why, why

    assert pending_ids([{"id": 1, "state": "PENDING"}, {"id": 2, "state": "COMMENTED"}]) == [1]
    assert pending_ids([{"id": 3, "state": "COMMENTED"}]) == []
    assert verdicted_at([{"body": "abcdef1 READY — x"}], H)
    assert not verdicted_at([{"body": "READY — no head here"}], H)
    assert not verdicted_at([{"body": "abcdef1 looks fine"}], H)
    print("pr_verdict self-test: ok (head in line 1, stale head, missing verdict, empty, pending, read-back)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", nargs=2, metavar=("REPO", "PR"))
    ap.add_argument("--post", nargs=4, metavar=("REPO", "PR", "BODY_FILE", "VERDICT"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.check:
        return do_check(*a.check)
    if a.post:
        repo, pr, body, verdict = a.post
        if verdict not in VERDICTS:
            print(f"pr_verdict: VERDICT must be one of {VERDICTS}", file=sys.stderr)
            return 2
        return do_post(repo, pr, body, verdict, a.dry_run)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
