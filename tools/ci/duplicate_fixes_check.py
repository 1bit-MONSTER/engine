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

How the two kinds of claim differ (engine#2842):

  * `Fixes`/`Closes`/`Resolves #N` is a CLAIM: two of them on one issue is a duplicate, and that is reported with no
    other evidence needed.
  * `Refs #N` is a REFERENCE: almost every PR reviewed on 2026-11-16/17 used it, because a PR that fixes part of an
    issue must not close it -- which made the shipped tool report a clean repo while #2839/#2840 both worked #2836.
    A reference is weaker evidence (it is also how a PR says "this is context"), so a reference pair is reported only
    when the PRs also change a COMMON FILE: two lanes editing the same code toward one issue.

A LIMIT, measured on the fleet and unchanged by the above: a PR that works an issue with a *bare mention*
(`engine#2803, measured …`) is still invisible. #2804 worked #2803 that way while #2805 claimed it in the title. The
narrowing is deliberate (see CLAIM), and the issue-side join (`gh api repos/O/R/issues/N/timeline`) returns both of
those PRs *and* mere mentions (#2795's timeline lists #2809 and #2807), so it is an annotation at best, not a trigger.

Usage:
    duplicate_fixes_check.py                    # the fleet's six repos
    duplicate_fixes_check.py --repos engine     # a subset
    duplicate_fixes_check.py --json
    duplicate_fixes_check.py --self-test        # hermetic: fixtures, no network
Exit: 0 no issue claimed twice, 1 at least one issue claimed by two or more open PRs, 2 cannot tell.
"""
import argparse
import json
import os
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
# engine#3060: markdown emphasis around the NUMBER (`Closes **#3041**`, `Fixes _#7_`) is how a large share of this
# fleet writes a claim, and the old pattern stopped at the `**` -- so such a PR counted as a bare mention and went
# missing from "claimed by" lists (measured on #3047: its body reads `Closes **#3041**`). Backticks are deliberately
# NOT matched: a claim inside a code span is documentation, which `claims()` already strips.
CLAIM = re.compile(r'\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s*[*_~]*#(\d+)', re.I)

# engine#2842: `Refs #N` is a claim too, but a WEAKER one. Almost every PR reviewed on 2026-11-16/17 said `Refs` rather than
# `Fixes` -- because a PR that fixes part of an issue must not close it -- so reading closing keywords only made the common
# case invisible and reported a clean repo while #2839/#2840 both worked #2836. Treating `Refs` as equal to `Fixes` is the
# other failure: `Refs` is also used for context, and a pair that merely mentions the same issue is not a duplicate. So a
# ref pair counts only when the two PRs also TOUCH OVERLAPPING FILES -- the same "overlap, not issue-claim" rule the review
# of #2809 asked for on the false-positive side, which makes one rule serve both directions.
REF = re.compile(r'\brefs?\b\s*:?\s*[*_~]*#(\d+)', re.I)


# A claim written INSIDE code is documentation, not a claim. The PRs most likely to contain `Fixes #12` are the ones
# explaining how to write one -- #2809's own body pastes the syntax it looks for -- so without this the detector flags
# the PR that documents the convention, which is the one reading a lane cannot skip.
FENCE = re.compile(r"```.*?```", re.S)     # a fenced block
SPAN = re.compile(r"`[^`\n]*`")            # an inline code span


def claims(text):
    """The issue numbers a PR title+body claims to close, ignoring any claim written inside code."""
    text = SPAN.sub(" ", FENCE.sub(" ", text or ""))
    return sorted({int(n) for n in CLAIM.findall(text)})


def referenced(text):
    """The issue numbers a PR title+body REFERS to (`Refs #N`), ignoring anything written inside code.

    Weaker than `claims()`: a reference says "this work is about #N", not "this closes #N", so it is only a duplicate
    signal when the referring PRs also overlap in files (see `duplicates`).
    """
    text = SPAN.sub(" ", FENCE.sub(" ", text or ""))
    return sorted({int(n) for n in REF.findall(text)})


def duplicates(prs, files_of=None):
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

    # engine#2842: refs are a duplicate signal only with file overlap. `files_of` maps a PR number to its changed-file
    # set; when it is not supplied a ref group is not reported at all, which keeps `duplicates(prs)` -- the shape every
    # existing arm uses -- meaning exactly what it meant before.
    if files_of:
        # everything that touches each issue -- closers and referrers together, because the pair a reviewer needs to see
        # can be one of each (`Fixes #N` beside `Refs #N`), which is exactly the #2839/#2840 shape one keyword over.
        touched = {}
        for pr in prs:
            text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
            for issue in set(claims(text)) | set(referenced(text)):
                touched.setdefault(issue, []).append(pr)
        reported = {issue for issue, _ in out}
        for issue in sorted(touched):
            if issue in reported:
                continue          # two closing claims already reported it, and that is the stronger statement
            group = sorted(touched[issue], key=lambda p: p.get('number') or 0)
            if len(group) < 2:
                continue
            files = {p.get('number'): set(files_of.get(p.get('number')) or ()) for p in group}
            # a PR joins the finding only if it shares a changed file with another in the group: `Refs` alone is context,
            # and two PRs that mention one issue while touching different files are different work.
            linked = [p for p in group
                      if any(files[p.get('number')] & files[q.get('number')] for q in group if q is not p)]
            if len(linked) > 1:
                out.append((issue, linked))
    return out


ALLOW_FILE = "duplicate_fixes_allow.txt"


def read_allow(path):
    """{issue: reason} from an allow file, one `<issue> <reason>` per line; missing file is an empty allow.

    Why it exists (engine#2797/#2798): two open PRs that BOTH say `Fixes #N` are this tool's strongest signal and
    also how a legitimate HALF is written -- one PR the behaviour, the second the arm that proves it can fail.
    Nothing in the PR text separates those shapes, so the separation is declared here, with a reason, and stays
    visible: an allowed issue is still printed, with its reason, and simply stops counting as a finding. The next
    failure mode is a stale entry, so every allowed issue that no open pair claims any more is reported too.
    """
    out = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                # `#2946 behaviour plus arm` is how anyone writes an issue, so a LEADING `#` is the issue, and a
                # comment only starts at ` #` -- splitting on the first `#` would drop every real entry, which is
                # exactly what it did until the live run showed an allow file with one line and zero effects.
                line = re.split(r"\s+#", line, maxsplit=1)[0].strip()
                if line.startswith("#"):
                    line = line[1:]
                if not line:
                    continue
                bits = line.split(None, 1)
                if not bits[0].isdigit():
                    continue
                out[int(bits[0])] = (bits[1] if len(bits) > 1 else "").strip() or "(no reason given)"
    except FileNotFoundError:
        return {}
    return out


def apply_allow(found, allow):
    """(kept, allowed, stale): findings that still count, findings a declared pair exempts, entries nobody claims."""
    allowed = [f for f in found if f['issue'] in allow]
    kept = [f for f in found if f['issue'] not in allow]
    stale = sorted(set(allow) - {f['issue'] for f in allowed})
    return kept, allowed, stale


def finding(repo, issue, group):
    """One finding, squashed from a duplicate group -- and `body` MUST survive the squash.

    engine#3048: this used to be inline in `main()` and kept four keys, dropping `body`. `describe()` re-derives
    whether the PRs CLAIM the issue (Fixes/Closes) or merely REFER to it from title+body, so every finding printed its
    weakest wording -- three PRs that all said `Closes #N` were reported as "referenced ... that touch the same files".
    It is a function rather than four inline lines because the bug was two functions apart: an arm over `describe()`
    cannot see what `main()` puts into it, and my first two arms passed against the unfixed code for exactly that reason.
    """
    return {'repo': repo, 'issue': issue,
            'prs': [{'number': p['number'], 'title': p['title'], 'body': p.get('body') or '',
                     'draft': p['draft'], 'age_hours': p['age_hours']} for p in group]}


def describe(issue, group):
    """The report line. `Fixes`/`Closes` is a claim on the issue; `Refs` plus overlapping files is two lanes working the
    same code, which is a different conversation, so the sentence says which one it is (engine#2842)."""
    closing = [p for p in group
               if issue in claims(f"{p.get('title') or ''}\n{p.get('body') or ''}")]
    if closing:
        lines = [f"#{issue} is claimed by {len(group)} open PR(s):"]
    else:
        lines = [f"#{issue} is referenced by {len(group)} open PR(s) that touch the same files:"]
    for pr in group:
        draft = ' [draft]' if pr.get('draft') else ''
        age = pr.get('age_hours')
        age_s = f", {age:.1f} h old" if isinstance(age, (int, float)) else ''
        lines.append(f"  #{pr.get('number')} {pr.get('title') or ''}".rstrip() + f"{draft}{age_s}")
    # engine#3205: the old sentence offered two remedies, and the second one cannot work -- `duplicates()` groups
    # `gh api …/pulls`, so this row is computed from OPEN pull requests only and no amount of agreeing in the
    # threads changes what it reads. Measured tonight: three lanes opened #3201/#3203/#3204 for one fix, all
    # three threads agreed within 12 minutes that #3201 lands first, and the row stayed red repo-wide for the
    # 25 minutes #3203 remained open. Say the action that clears it, or a lane follows the sentence and waits.
    lines.append("  exactly one must CLOSE: this row is computed from OPEN pull requests, so agreeing in the "
                 "threads cannot clear it")
    return "\n".join(lines)


def run_gh(args):
    """`gh api …` -> stdout, or RuntimeError when gh could not run or did not answer.

    engine#2962 review: a bare `subprocess.run` here made the two shapes that actually happen read as answers.
    `gh` missing raised `FileNotFoundError` out of `main()` (a traceback, rc 1 -- read as a duplicate finding), and
    `gh` failing returned EMPTY STDOUT, which `json.loads(raw or '[]')` turned into "no open PRs" -- a silent clean
    from the row that polices duplicates. Both are failures to look, so both leave as RuntimeError, which the
    per-repo handler already turns into `failed` and `--gate` into 77.
    """
    try:
        p = subprocess.run(['gh'] + list(args), capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"gh could not run: {e}")
    if p.returncode != 0:
        raise RuntimeError(f"gh api exited {p.returncode}: {(p.stderr or '').strip()[:200]}")
    return p.stdout


def gh(path, paginate=False):
    """One page, or every page, of a `gh api` list endpoint (the same shape as pr_review_queue.py)."""
    if not paginate:
        return run_gh(['api', path])
    out, page = [], 1
    while True:
        sep = '&' if '?' in path else '?'
        chunk = run_gh(['api', f"{path}{sep}per_page={PAGE}&page={page}"])
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


def normalise_repo(name):
    """'engine' or '1bit-MONSTER/engine' -> 'engine'; anything else -> None.

    Why (found by running it): open_prs builds `repos/{OWNER}/{repo}/pulls`, so passing the full `owner/name` -- the
    natural way to name a repository on a command line -- asked GitHub for `1bit-MONSTER/1bit-MONSTER/engine`, got a
    404, and reported "cannot tell". In --gate mode cannot-tell is a SKIP, so a typo or the wrong spelling disabled
    this gate permanently and said nothing. A name it does not know is a usage error and must be named as one.
    """
    name = (name or "").strip().strip("/")
    if name.lower().startswith(OWNER.lower() + "/"):
        name = name[len(OWNER) + 1:]
    return name if name in DEFAULT_REPOS else None


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


def files_of(repo, numbers):
    """{pr number: set(changed paths)} for the PRs the overlap test needs, and only those.

    engine#2842: fetched lazily on purpose -- the common case (two closing claims) needs no file data at all, so the
    tool's cost must not grow by one request per open PR just to answer a question most runs do not ask. A PR whose
    file list cannot be read maps to an empty set, which makes it non-overlapping: an unreadable list must not
    manufacture a duplicate, and `duplicates` still reports the closing-claim pairs without it.
    """
    out = {}
    for n in numbers:
        try:
            raw = gh(f"repos/{OWNER}/{repo}/pulls/{n}/files?per_page={PAGE}", paginate=True)
            out[n] = {f.get('filename') for f in json.loads(raw or '[]') if f.get('filename')}
        except (RuntimeError, json.JSONDecodeError):
            out[n] = set()
    return out


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
    # engine#3205: this arm used to assert the first clause only, so a report whose advice could not clear the
    # row still passed it (mutation: '… or reconsider the weather' left the self-test green). Assert the
    # contract instead: the action that clears the row, and the fact that a thread comment is not one.
    ok('must CLOSE' in describe(*dup[0]), "the report names the action that clears the row")
    ok('cannot clear it' in describe(*dup[0]),
       "…and says a thread comment is not it, because the row reads open PRs only")
    ok('say in both threads' not in describe(*dup[0]),
       "…and no longer offers the remedy that cannot work")

    # a draft claimant still counts: that is a lane working, which the second lane cannot see
    prs[1]['draft'] = True
    ok([p['number'] for p in duplicates(prs)[0][1]] == [2784, 2786], "a draft claimant is reported too")
    ok('[draft]' in describe(*duplicates(prs)[0]), "... and marked as a draft")

    # engine#2797/#2798: two PRs that BOTH say `Fixes #N` are the strongest signal and also how a legitimate HALF
    # is written, and nothing in the text separates them -- so the exemption is declared, with a reason, and the
    # entry that outlives its pair is reported rather than silently kept.
    # apply_allow takes the shape main() builds: one dict per finding, not duplicates()'s (issue, group) tuples
    pairs = [{'repo': 'engine', 'issue': issue, 'prs': group} for issue, group in duplicates(prs)]
    kept, allowed, stale = apply_allow(pairs, {2778: 'the behaviour plus the arm that proves it'})
    ok(kept == [] and [f['issue'] for f in allowed] == [2778],
       "a declared pair stops counting as a finding, and is reported as allowed instead")
    ok(apply_allow(pairs, {2778: 'x', 9999: 'gone'})[2] == [9999],
       "an allow entry no open pair claims any more is STALE, so it is reported rather than kept forever")
    ok(apply_allow(pairs, {})[0] == pairs, "an empty allow changes nothing")
    # engine#2962: '1bit-MONSTER/engine' used to ask GitHub for 1bit-MONSTER/1bit-MONSTER/engine, get a 404 and be
    # reported as "cannot tell" -- which --gate turns into a SKIP, so the gate could be disabled by a spelling.
    ok(normalise_repo('engine') == 'engine', "a bare repo name is accepted")
    ok(normalise_repo('1bit-MONSTER/engine') == 'engine', "... and the owner/name spelling normalises instead of 404ing")
    ok(normalise_repo('1bit-MONSTER/engin') is None, "a name it does not know is refused, not turned into a skip")
    # engine#2982/#2979: every pass compared OPEN PRs with each other, so when one of a pair merged the pair
    # stopped existing and the gate said clean. These arms are the three states of that third pass.
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    op = {'number': 9002, 'title': 'fix the thing', 'body': 'Fixes #9001', 'merged_at': None}
    mp = {'number': 9001, 'title': 'fix the thing', 'body': 'Fixes #9001', 'merged_at': now}
    fmap = {9002: ['a.cpp'], 9001: ['a.cpp']}
    tw, nt = mended_twins([op], [mp], files_of=lambda n: fmap[n], days=2)
    ok(len(tw) == 1 and tw[0][0] == 9001, "an open PR re-doing a just-merged fix is a TWIN when they share a file")
    tw2, nt2 = mended_twins([op], [mp], files_of=lambda n: ['a.cpp'] if n == 9002 else ['b.cpp'], days=2)
    ok(len(tw2) == 0 and len(nt2) == 1, "... a follow-up touching other files is a NOTE, not a finding")
    old_mp = dict(mp, merged_at="2020-01-01T00:00:00Z")
    tw3, nt3 = mended_twins([op], [old_mp], files_of=lambda n: fmap[n], days=2)
    ok(len(tw3) == 0 and len(nt3) == 0, "... and a merge outside the window is not a twin (or a note)")
    # engine#2991/#2993: 'engine#2987: ...' is neither a claim nor a Refs, and the gate said clean on a real pair
    a = {'number': 9101, 'title': 'crash after the skip line', 'body': '**engine#9100: a row that crashed**'}
    b = {'number': 9102, 'title': 'skip line then segfault', 'body': 'Fixes #9100.'}
    bm = bare_mentions([a, b], files_of=lambda n: ['tools/ci/ctest_known_fail.py'])
    ok(len(bm) == 1 and bm[0][0] == 9100, "a bare #N plus a shared file is reported when neither names a verb")
    bm2 = bare_mentions([a, b], files_of=lambda n: ['tools/ci/ctest_known_fail.py'] if n == 9101 else ['x.cpp'])
    ok(len(bm2) == 0, "... and is silent when the two PRs touch different files")
    a2 = {'number': 9101, 'title': 'engine#9100 twice', 'body': 'body: engine#9100 again'}
    bm3 = bare_mentions([a2], files_of=lambda n: ['f.cpp'])
    ok(len(bm3) == 0, "... and one PR naming the issue twice does not pair with ITSELF")
    # the file is written by a human, who writes `#2797`, not `2797`
    import pathlib
    import tempfile
    d = tempfile.TemporaryDirectory()
    ap = pathlib.Path(d.name) / "allow.txt"
    ap.write_text("#2797 the behaviour plus the arm\n\n2950 second entry  # trailing note\n")
    got = read_allow(str(ap))
    ok(got.get(2797, "").startswith("the behaviour"), "a leading `#` is the issue number, not a comment")
    ok(got.get(2950, "") == "second entry", "... and a ` #` starts a trailing comment")
    d.cleanup()

    # no claim, no finding -- the failure mode of a detector that always fires
    ok(duplicates([{'number': 1, 'title': 'a', 'body': 'Refs #5', 'draft': False},
                   {'number': 2, 'title': 'b', 'body': 'Refs #5', 'draft': False}]) == [],
       "two PRs that merely REFER to an issue are not a duplicate")

    # engine#2842: the population that made this tool report a false clean -- #2839 and #2840 are both open on #2836 and
    # both say `Refs`, not `Fixes`. The two PRs below are their real bodies and their real files.
    ref_prs = [
        {'number': 2839, 'title': 'fix/2836-stale-expectations-row',
         'body': 'Refs #2836 — the reachable half of it: the row is re-armed when the table is regenerated',
         'draft': False, 'age_hours': 0.6},
        {'number': 2840, 'title': 'cl-game/2836-stale-expectations',
         'body': 'Refs #2836 (the row half; the pin bump it waits on is separate)',
         'draft': False, 'age_hours': 0.5},
    ]
    same_files = {2839: {'CMakeLists.txt', 'tools/ci/check_stale_expectations.py'},
                  2840: {'CMakeLists.txt', 'tools/ci/check_stale_expectations.py'}}
    ok(referenced(ref_prs[0]['body']) == [2836], "a `Refs #N` body is read as a reference")
    ok(claims(ref_prs[0]['body']) == [], "... and NOT as a closing claim, which is the distinction the fix rests on")
    ok(duplicates(ref_prs) == [], "two `Refs` PRs are not reported when no file list is given (the old meaning is kept)")
    dup_ref = duplicates(ref_prs, same_files)
    ok([i for i, _ in dup_ref] == [2836], "two `Refs` PRs on one issue that touch the same files ARE one finding (#2842)")
    ok([p['number'] for p in dup_ref[0][1]] == [2839, 2840], "... naming both PRs, oldest first")
    ok('touch the same files' in describe(*dup_ref[0]),
       "... and the sentence says they touch the same files rather than claiming the issue")
    # engine#3048: the squash in `main()` dropped `body`, and `describe()` re-derives the claim from title+body -- so
    # every finding printed its weakest wording even when every PR in it said `Closes #N`. These arms go THROUGH
    # finding(), because an arm that hand-builds the dict cannot see the squash (mine could not, twice).
    closed = [{'number': 9101, 'title': 'a', 'body': 'Closes #9100.', 'draft': False, 'age_hours': 1.0},
              {'number': 9102, 'title': 'b', 'body': 'Fixes #9100.', 'draft': False, 'age_hours': 2.0}]
    f = finding('engine', 9100, closed)
    ok(f['prs'][0].get('body') == 'Closes #9100.',
       "the squash that builds a finding KEEPS the body -- an arm over describe() alone cannot see this (engine#3048)")
    ok('is claimed by' in describe(f['issue'], f['prs']),
       "... so a pair that both say Closes is described as a claim, not as a reference")
    bodyless = [{'number': 9101, 'title': 'a', 'body': '', 'draft': False, 'age_hours': 1.0},
                {'number': 9102, 'title': 'b', 'body': 'Refs #9100.', 'draft': False, 'age_hours': 2.0}]
    bf = finding('engine', 9100, bodyless)
    ok('touch the same files' in describe(bf['issue'], bf['prs']),
       "... and only a body that does not claim it falls back to the weaker sentence")
    disjoint = {2839: {'tools/ci/check_stale_expectations.py'}, 2840: {'docs/hw-gate.md'}}
    ok(duplicates(ref_prs, disjoint) == [],
       "two `Refs` PRs whose files do not overlap are NOT a duplicate -- the false positive the weaker keyword would add")
    # one PR closes it and the other only references it: without a file list there is nothing to compare, so it is not
    # reported -- the reference might be context. WITH overlap it is reported, which is the case a reviewer wants to see.
    mixed = [dict(ref_prs[0], body='Fixes #2836'), ref_prs[1]]
    ok(duplicates(mixed) == [], "one closing claim plus one `Refs` is NOT reported without a file list")
    ok([i for i, _ in duplicates(mixed, same_files)] == [2836], "... but IS reported once the files overlap")
    ok('is claimed by' in describe(*duplicates(mixed, same_files)[0]),
       "... and reads as a claim, because one of the two closes it")
    # engine#3060: the sentence is decided from the BODY, and the finding carries the body (engine#3048/#3052) -- so the
    # note must ask the same question. A pair where BOTH claim is a finding; the note must not restate it with the
    # opposite reason, which is what it did for seven pairs on the 23:5xZ queue.
    both_claim = [dict(ref_prs[0], body='Fixes #2836'), dict(ref_prs[1], body='Closes #2836')]
    ok(note_line(2836, both_claim[0], both_claim[1], ['f.cpp']) is None,
       "two PRs that both claim the issue print no note -- the finding above is the report")
    one_claim = note_line(2836, ref_prs[0], dict(ref_prs[1], body='Fixes #2836'), ['f.cpp'])
    ok(one_claim is not None and 'only names it' in one_claim,
       "... but one claimer plus one mention still prints a note, and it says which is which")
    ok('neither says Fixes/Closes' in note_line(2836, ref_prs[0], ref_prs[1], ['f.cpp']),
       "... and the old sentence is kept only where it is true: two PRs that both merely name the issue")
    # engine#3060: a claim whose issue number is marked up is still a claim
    # engine#3060 / review of #3064 (pi-i679): the markup set is *, _ and ~ (strikethrough) -- contiguous markers only,
    # directly before the number. A BACKTICKED number stays a non-claim: it is inside code, and this tool's stated rule
    # is that a claim written inside code is documentation. Both directions are pinned so a later widening to "any
    # punctuation" has to argue with a failing test.
    ok(claims("Closes **#3041** -- the row has been red") == [3041], "a bold claim is a claim (engine#3060)")
    ok(claims("Fixes __#12__") == [12] and claims("Closes *#7*") == [7] and claims("Fixes ~~#9~~") == [9],
       "... and so are underscore, italic and strikethrough markup")
    ok(claims("Closes `#12` -- quoted in a log line") == [],
       "... while a backticked NUMBER is inside code and is not a claim")
    ok(referenced("Refs **#7**") == [7], "... and so does a marked-up reference")
    # a reference inside code is documentation, like a claim inside code
    ok(referenced("use `Refs #12` next to the PR body") == [], "a `Refs` inside an inline code span is not a reference")

    # review of #2809: a repo that could not be read must not print the clean sentence, and must not be masked by a
    # duplicate found elsewhere. Both are one-liners that were wrong in the first version.
    ok(clean_line([], 0) is not None, "with nothing found and every repo read, the clean sentence prints")
    ok(clean_line([], 1) is None, "... but NOT when a repo could not be read (that repo is where a claimant would hide)")
    ok(clean_line([{'repo': 'engine'}], 0) is None, "... and not when something was found")
    ok(exit_code([], 0) == 0 and exit_code([{'repo': 'engine'}], 0) == 1, "0 none / 1 duplicate")
    ok(exit_code([{'repo': 'engine'}], 1) == 2, "cannot-tell outranks a duplicate rather than being swallowed by it")
    ok(exit_code([], 2) == 2, "and it is not a pass on its own either")

    # engine#2962 review: both ways of failing to look must be cannot-tell, never a clean and never a traceback.
    # Driven through a stub on PATH, because that is the state a runner is actually in when gh is absent or 403s.
    import os, tempfile
    with tempfile.TemporaryDirectory() as td:
        stub = pathlib.Path(td) / "gh"
        stub.write_text("#!/bin/sh\nexit 1\n")
        stub.chmod(0o755)
        saved = os.environ.get("PATH", "")
        os.environ["PATH"] = td + os.pathsep + saved
        try:
            try:
                gh("repos/1bit-MONSTER/engine/pulls?state=open", paginate=False)
                refused_failing = False
            except RuntimeError:
                refused_failing = True
        finally:
            os.environ["PATH"] = saved
        ok(refused_failing, "a gh that exits non-zero is cannot-tell, not empty output ('[]' = no open PRs)")

    saved = os.environ.get("PATH", "")
    os.environ["PATH"] = "/nonexistent"
    try:
        try:
            gh("repos/1bit-MONSTER/engine/pulls?state=open", paginate=False)
            refused_missing = False
        except RuntimeError:
            refused_missing = True
    finally:
        os.environ["PATH"] = saved
    ok(refused_missing, "a missing gh is cannot-tell too, not a FileNotFoundError out of main()")

    print("self-test ok" if not fails else f"self-test FAILED ({fails})")
    return 0 if not fails else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--repos', nargs='+', default=list(DEFAULT_REPOS))
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    # engine#2807 review: `pre-pr-checks.sh` runs this at handover, and it is that script's only NETWORK row.
    # A network blip must not red a gate whose other sixteen rows are offline, and a legitimate HALF (a fix plus
    # its arm) is indistinguishable from a duplicate until an exemption exists -- so the caller can ask for the
    # finding to be printed without the exit status deciding anything. The page-vs-runner comparator in
    # pre-pr-checks.sh only understands rows whose command starts with `python3`, which is why this is a flag
    # here rather than a `sh -c ... || true` wrapper there.
    ap.add_argument('--allow', help=f'`<issue> <reason>` per line, exempting a declared pair (default: {ALLOW_FILE} beside this tool)')
    # engine#2807/#2842: the gate mode pre-pr-checks.sh uses. Its exit codes are the script's own vocabulary --
    # 0 clean, 1 a finding, 77 "cannot run here" (the ctest skip code) -- because the script classifies a row by
    # these three and by nothing else, and the page-vs-runner comparator forbids a `sh -c` wrapper that would
    # map them outside the tool.
    ap.add_argument('--mended-days', type=int, default=2,
                    help='an open PR re-doing work merged in the last N days is a finding (default 2)')
    ap.add_argument('--gate', action='store_true',
                    help='pre-pr-checks mode: 0 clean, 1 a duplicate, 77 cannot tell (network/gh unavailable)')
    ap.add_argument('--advisory', action='store_true',
                    help='print findings but always exit 0 (for the pre-PR gate; see --advisory in docs/pre-pr-checks.md)')
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    unknown = [r for r in a.repos if normalise_repo(r) is None]
    if unknown:
        print(f"duplicate_fixes: --repos {unknown[0]!r} is not a repository this tool reads; "
              f"use one of {', '.join(DEFAULT_REPOS)} (with or without the '{OWNER}/' prefix)",
              file=sys.stderr)
        return 2
    found, failed = [], 0
    mended, notes = [], []
    for repo in (normalise_repo(r) for r in a.repos):
        try:
            prs = open_prs(repo)
        except (RuntimeError, json.JSONDecodeError) as e:
            print(f"duplicate_fixes: cannot tell for {repo}: {e}", file=sys.stderr)
            failed += 1
            continue
        # first pass without files: closing claims only, as before
        try:
            recently = json.loads(gh(f"repos/{OWNER}/{repo}/pulls?state=closed&per_page={PAGE}"
                                     f"&sort=updated&direction=desc", paginate=False) or '[]')
            # files_of's real contract is (repo, numbers) -> {number: set(files)}; mended_twins wants a 1-arg
            # lookup. Calling it directly would be a TypeError on the first twin found -- a latent crash in
            # exactly the path that matters, invisible until a twin exists.
            def _files(n, _repo=repo):
                return files_of(_repo, [n]).get(n, ())
            twins, nts = mended_twins(prs, [x for x in recently if x.get('merged_at')], _files, a.mended_days)
            mended += [(repo,) + t for t in twins]
            notes += [(repo,) + n for n in nts]
        except (RuntimeError, json.JSONDecodeError):
            pass          # this pass is extra information; the open-vs-open passes above are the gate
        first = duplicates(prs)
        # second pass: which issues are touched by two or more PRs and were not explained by a closing pair
        touched = {}
        for pr in prs:
            text = f"{pr.get('title') or ''}\n{pr.get('body') or ''}"
            for issue in set(claims(text)) | set(referenced(text)):
                touched.setdefault(issue, []).append(pr)
        reported = {issue for issue, _ in first}
        need = sorted({p['number'] for issue, group in touched.items()
                       if issue not in reported and len(group) > 1 for p in group})
        groups = duplicates(prs, files_of(repo, need) if need else None)
        # a bare `#N` mention plus a shared file: a NOTE, never a finding (see bare_mentions)
        try:
            bm = bare_mentions(prs, lambda n: files_of(repo, [n]).get(n, ()))
            notes += [(repo,) + t for t in bm]
        except (RuntimeError, json.JSONDecodeError):
            pass
        for issue, group in groups:
            found.append(finding(repo, issue, group))
    allow = read_allow(a.allow or os.path.join(os.path.dirname(os.path.abspath(__file__)), ALLOW_FILE))
    for repo, issue, op, mp, shared in mended:
        found.append({'repo': repo, 'issue': issue, 'mended_by': mp['number'],
                      'prs': [{'number': op['number'], 'title': op.get('title') or '',
                               'draft': False, 'age_hours': 0}]})
    found, allowed, stale = apply_allow(found, allow)
    if a.json:
        print(json.dumps(found, indent=1))
    else:
        for f in allowed:
            print(f"{f['repo']}: #{f['issue']} is claimed twice and ALLOWED ({allow[f['issue']]})\n  "
                  + describe(f['issue'], f['prs']).replace("\n", "\n  "), file=sys.stderr)
        for issue in stale:
            print(f"duplicate_fixes: allow entry #{issue} is STALE -- no open pair claims it any more "
                  f"({allow[issue]}); delete the line", file=sys.stderr)
        for repo, issue, op, mp, shared in mended:
            print(f"{repo}: #{issue} was already closed by #{mp['number']} "
                  f"({(mp.get('merged_at') or '')[:16]}), and open #{op['number']} claims it again")
            print(f"  they touch {', '.join(shared)}" if shared else "")
            print(f"  if this is a follow-up, say `Refs #{issue}` and what differs; if it re-does the work, "
                  f"close it")
        for note in notes:
            repo, issue = note[0], note[1]
            if len(note) == 4:      # (repo, issue, open_pr, merged_pr) -- a follow-up with the wrong verb
                op, mp = note[2], note[3]
                print(f"duplicate_fixes: {repo} #{op['number']} says `Fixes #{issue}`, which #{mp['number']} "
                      f"already merged -- `Refs #{issue}` is the verb if this is a follow-up (not a finding)",
                      file=sys.stderr)
            else:                   # (repo, issue, a, b, shared) -- two open PRs, one file, no verb
                # NOT `a`/`b`: `a` is the argparse Namespace in this function, and rebinding it here made
                # `if a.gate:` an AttributeError on the first run that printed a note -- a crash that only
                # fires when the tool has something to say.
                pa, pb, shared = note[2], note[3], note[4]
                line = note_line(issue, pa, pb, shared)
                if line is not None:      # None = already a finding; do not restate it with a false reason
                    print(f"duplicate_fixes: {line}", file=sys.stderr)
        for f in found:
            print(f"{f['repo']}: " + describe(f['issue'], f['prs']))
        clean = clean_line(found, failed)
        if clean:
            print(clean)
    if a.gate:
        # cannot-tell is a SKIP here, not a failure: this is the script's only network row and a blip must not red
        # a gate whose other sixteen rows are offline. A duplicate is a finding and exits 1.
        if failed:
            print("duplicate_fixes: cannot tell for at least one repo -- skip, not a pass", file=sys.stderr)
            return 77
        return exit_code(found, failed)
    if a.advisory:
        if found or failed:
            print("duplicate_fixes: advisory run -- nothing above changes this run's exit status",
                  file=sys.stderr)
        return 0
    return exit_code(found, failed)


def claims_issue(p, issue):
    """Does this PR CLOSE the issue? `describe()` and the note line must agree on the answer (engine#3060)."""
    return issue in claims(f"{p.get('title') or ''}\n{p.get('body') or ''}")


def note_line(issue, pa, pb, shared):
    """The sentence for a bare-mention pair, or None when the strong pass already reports the pair.

    engine#3060: the note said "neither says Fixes/Closes" for every pair -- including two PRs that both say
    `Closes #N`, because the population is a bare `#N` scan and nothing asked. That pair is a FINDING, so the run
    printed the finding and then a quieter restatement of it whose reason was false (seven such lines on the
    2026-11-21 23:5xZ queue, all of them #3058/#3059-shaped: both bodies said `Fixes #3057.`). Now: two claimers are
    suppressed, one claimer names which is which, and the original sentence is kept only for the case that makes it
    true -- the #2991 shape, where one PR only names the issue and the note is the only detector.
    """
    ca, cb = claims_issue(pa, issue), claims_issue(pb, issue)
    if ca and cb:
        return None
    if ca or cb:
        claimer, other = (pa, pb) if ca else (pb, pa)
        return (f"#{claimer['number']} says it closes #{issue} and #{other['number']} only names it, "
                f"and they touch {', '.join(shared)} -- a note, not a finding")
    return (f"#{pa['number']} and #{pb['number']} both name #{issue} and touch {', '.join(shared)} "
            f"-- neither says Fixes/Closes, so this is a note, not a finding")


def bare_mentions(prs, files_of=None):
    """[(issue, a, b, shared)] -- two open PRs on one file that name one issue WITHOUT a closing verb.

    Why (found live, 2026-11-20, ninety minutes after the previous hole): #2991 opened with
    "**engine#2987: a row that crashed after printing the device-skip line is a failure**" and #2993 with
    "Fixes #2987." -- same file, same issue, same fix. `claims()` did not match the first (no verb) and
    `references()` did not either (no "Refs"), so the gate reported clean on a real duplicate.

    A bare `#N` is how a lane names an issue it is DESCRIBING rather than closing, so it is evidence of a
    pair and not a claim about one. It returns notes and never findings: two PRs on one file mention a
    long-running issue (#2478) in passing all the time, and this tool's worst failure is refusing a
    legitimate half. The shared-file requirement is what keeps the note worth printing.
    """
    MENTION = re.compile(r'#(\d+)')
    by_issue = {}
    for p in prs:
        text = f"{p.get('title') or ''}\n{p.get('body') or ''}"
        for m in MENTION.finditer(text):
            # keyed by NUMBER, not appended: a body that names the issue in the title and again in the prose
            # would otherwise put one PR in the group twice and pair it WITH ITSELF. I recorded exactly this
            # bug this morning in a throwaway measurement script (17 pairs became 2 after deduping) and then
            # wrote it again here -- "a duplicate detector that counts one PR as two claimants has the defect
            # it was built to find, one level down".
            by_issue.setdefault(int(m.group(1)), {})[p['number']] = p
    out = []
    for issue, groupd in sorted(by_issue.items()):
        group = list(groupd.values())
        if len(group) < 2 or files_of is None:
            continue
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                shared = set(files_of(a['number'])) & set(files_of(b['number']))
                if shared:
                    out.append((issue, a, b, sorted(shared)))
    return out


def mended_twins(open_prs, merged_prs, files_of=None, days=2):
    """[(issue, open_pr, merged_pr, files)] -- an open PR re-doing work a recently merged PR already claims.

    Why (found live, 2026-11-20): every pass in this tool compared OPEN pull requests with each other. When one
    of a pair merges, the pair stops existing and the gate reports clean -- so #2982 (`Fixes #2978`, opened 15:09)
    duplicated #2979, which merged the identical one-line fix at 15:10, and nothing noticed. A gate that only
    looks at the open half of the population cannot see the case where the answer is already on main.

    A time bound rather than a count: issues like engine#2478 accumulate merged PRs for weeks, and comparing an
    open claim against ALL history would report every long-running issue as a duplicate.

    It is a *twin* only when the two touch a shared file -- otherwise it is a follow-up that named the wrong
    verb, which `note` reports and does not fail: `Fixes #N` for an issue already fixed should be `Refs #N`, but
    a text nit is not worth a red gate (this tool must not refuse what it should accept).
    """
    cut = None
    twins, notes = [], []
    for p in merged_prs:
        at = p.get("merged_at") or ""
        if not at:
            continue
        if cut is None:
            import datetime
            cut = (datetime.datetime.now(datetime.timezone.utc)
                   - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        if at < cut:
            continue
        for op in open_prs:
            shared = set(claims(f"{op.get('title') or ''}\n{op.get('body') or ''}")) & \
                     set(claims(f"{p.get('title') or ''}\n{p.get('body') or ''}"))
            for issue in shared:
                if files_of is None:
                    twins.append((issue, op, p, [])); continue
                a, b = set(files_of(op["number"])), set(files_of(p["number"]))
                if a and b and a & b:
                    twins.append((issue, op, p, sorted(a & b)))
                else:
                    notes.append((issue, op, p))
    return twins, notes


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
