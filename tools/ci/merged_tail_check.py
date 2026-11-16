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
"""merged_tail_check.py -- commits that landed on a branch after its PR merged.

The merge loop merges the REVIEWED head (`--match-head-commit`), and that is correct: the gate
approves a head, so the merge must be that head. The loss is somewhere else, and it is a race the
reviewed-head rule cannot cover:

    a push lands on the branch AFTER the merge. The PR is already MERGED, so nothing will merge
    that branch again, and the new commit is on a branch nobody is watching.

Measured on engine#2623 (merged 0193a29 with 3 of the 5 commits that were pushed): `WEAK_AFTER_S`
and `def distances` were on the branch and are absent from `main`. The PR badge said merged, the
gate said READY, and the content was not there. #2640 recovered it.

**Why a content check and not a sha check.** A squash merge rewrites every sha, so "is this commit
in the base" is unanswerable by sha: the commit is never an ancestor and its patch-id is not the
squashed patch-id either (`git cherry` reports every commit of every squashed PR as absent). So the
question is answered the way #2640 answered it by hand -- **do the lines this commit adds exist in
the base's copy of the file it touched** -- with one row per commit and the missing lines named.

Classes reported:

  LOST          most of the commit's added lines are not in the base AND no base file is longer:
                it needs a recovering PR
  SUPERSEDED    most added lines absent, but a later commit REPLACED one of those lines: the base
                changed THAT line after this PR merged (`git log -S <line> <merge>..<base> -- <file>`),
                or -- the fallback that stays when no merge commit is in hand -- every affected base
                file is LONGER. Either way a later commit rewrote that region, which is not evidence
                of a loss. Scoped to the merge commit rather than to line counts, because a one-line
                value file (a pin bump) is never "longer" and read as LOST for four different PRs
                (engine#2586, #2572, #2583, #2554) -- and to the LINE rather than the file, because a
                commit that touched some other line of the same file would otherwise turn a genuine
                loss into SUPERSEDED, and report() exits 0 on SUPERSEDED.
  PARTIAL       a few of them are gone because main edited those lines since: not a loss
  IN-BASE       every added line is already there (the content arrived another way): not a loss
  UNVERIFIABLE  the commit adds no significant line (deletion-only): neither of the above, a gap
  GONE          neither the branch nor `refs/pull/<n>/head` exists: a gap, never a pass. The PR
                ref is the MERGE-TIME head, so what it shows is the head the PR merged at -- it is a
                fallback for a deleted branch (the merge loop deletes them), not a way to see a push
                that landed afterwards, which is why the branch route is tried first. A row examined
                through that ref says so in its branch column, so a reader can tell which route it was.

Exit 0 no loss and nothing unexamined, 1 a LOST commit, 2 a ref could not be read or something was
unverifiable -- **a scan that could not look is not a scan that found nothing.**
"""

import argparse
import json
import os
import pathlib
import re
import shutil
import tempfile
import subprocess
import sys

# A line worth looking for: long enough not to be punctuation, and not a brace or a lone keyword.
# Below this the containment test answers "yes" for `}`, `);` and `#endif` in every file on earth.
SIGNIFICANT = 8

# A commit is LOST when at least this share of the lines it adds is absent from the base. The gap is
# wide, so the number is not delicate: measured on engine#2623, whose five commits split 89 % and
# 100 % absent (the two that were lost) against 4 %, 25 % and 30 % (the three that landed and whose
# lines main later edited). Anything above the landed group and below the lost group works; a
# threshold of "any line absent" would have called all five lost, and "all lines absent" would have
# missed #2623's first lost commit at 89 %.
LOST_FRACTION = 0.5


# The denominator. A scan that finds nothing and a scan that read nothing print the same
# "0 findings" without it -- the absent-denominator failure this fleet fixed four times tonight
# (#2624, #2627, #2645, #2656).
examined = [0]
# same_head() returns before any fetch: those branches are assumed unchanged, not compared.
unchanged = [0]


def git(repo, *args, check=False):
    p = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p


# `gh` may simply not be installed -- the CI container does not have it, and the first version of this
# tool crashed there instead of reporting that it could not look, which is the same fault the tool
# exists to catch one level up. Missing, unauthorised and network-down all answer like this: a
# non-zero result with a reason, never an exception and never an empty success.
GH_BIN = "gh"


def gh(*args):
    try:
        return subprocess.run([GH_BIN, *args], capture_output=True, text=True)
    except (FileNotFoundError, PermissionError, OSError) as e:
        return subprocess.CompletedProcess([GH_BIN, *args], 127, "",
                                           f"{GH_BIN} could not be run: {e}")


def significant_lines(text):
    """The lines of `text` worth asking the base about, deduped, in order."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) < SIGNIFICANT:
            continue
        if s in ("{", "}", ");", "};", "#endif", "#else"):
            continue
        if s not in out:
            out.append(s)
    return out


# The identifier a missing line is ABOUT, for the fallback below: the longest identifier-ish run in it,
# long enough to be distinctive and not a keyword that appears in every file.
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{5,}")
TOKEN_STOP = {"return", "define", "include", "static", "struct", "string", "sizeof", "unsigned",
              "continue", "function", "default", "template", "typename", "operator"}


def distinctive_token(line):
    """The longest identifier in `line`, or "" when it carries nothing distinctive."""
    best = ""
    for tok in TOKEN_RE.findall(line):
        if tok in TOKEN_STOP or len(tok) < 6:
            continue
        if len(tok) > len(best):
            best = tok
    return best


def moved_since(repo, line, since, base_ref, path, base_text):
    """How a later commit on the base superseded `line`: "line", "token", or "" for not at all.

    Two questions, weakest evidence last, and BOTH are needed (engine#2693 review follow-up):

      * `git log -S <line>` -- the base carried this exact line and a later commit changed it. This is
        the pin-bump shape (#2586/#2572: `sha=<old>` replaced by `sha=<new>`), and it is silent when the
        line never landed at all, because the occurrence count of a string that was never there cannot
        change.
      * `git log -S <token>` **only when the base carries that token**: the content landed in other words
        and a later commit touched it. This is #2504 exactly -- its missing lines are all silent on the
        exact test, yet a later commit (a20f6ee89, #2508) rewrites `httplib::Server`, which main carries
        three times. Requiring the token to be present is what stops a dropped feature from being called
        superseded because some unrelated commit elsewhere in the file happened to mention a shared word.
    """
    if git(repo, "log", "--oneline", "-S", line, f"{since}..{base_ref}", "--", path).stdout.strip():
        return "line"
    tok = distinctive_token(line)
    # shortcut: the token is the line's longest identifier, which on a real line can be a generic word
    # (measured: #2504 resolves through `executable`). The remaining weakness is therefore a SUPERSEDED for a
    # loss, not the reverse -- the detail string names the token so a reader can rule it out, and the exact-line
    # test above always gets first refusal. Upgrade by requiring the token to be absent from the rest of the base
    # file before the merge, so a word the whole file already used cannot decide it.
    if tok and tok in base_text and \
            git(repo, "log", "--oneline", "-S", tok, f"{since}..{base_ref}", "--", path).stdout.strip():
        return "token"
    return ""


def same_head(frozen, tip):
    """True when the live tip is the head the PR was merged at.

    `frozen` is `headRefOid`, recorded at merge time and never moved afterwards. An empty or
    unreadable frozen head must answer False -- "I do not know" may not skip a branch.
    """
    frozen, tip = (frozen or "").strip(), (tip or "").strip()
    if len(frozen) < 7 or not tip:
        return False
    return tip.lower().startswith(frozen.lower()[:7]) or frozen.lower().startswith(tip.lower()[:7])


def branch_mode(repo, branch, base_ref, merged_head=None):
    """List the commits on `branch` that are not on `base_ref` -- the recovery workflow.

    A PR-list scan answers "is anything lost?"; this answers "what of MY branch never landed?",
    which is what a lane needs immediately after it notices. (merge_content_check, #2655.)
    """
    if git(repo, "rev-parse", "--verify", "-q", branch).returncode != 0:
        return None
    tail = merge_commits(repo, branch, base_ref, merged_head)
    if tail is None:
        return None
    for sha, subject in tail:
        kind, detail = classify_commit(repo, sha, base_ref)
        print(f"{kind:12s} {sha} {subject[:60]}\n{'':12s} {detail}")
    print(f"merged_tail_check: {len(tail)} commit(s) on {branch} are not on {base_ref}")
    return tail


def merge_commits(repo, ref, base_ref, merged_head=None):
    """Commits reachable from `ref` and not from `base_ref` -- the branch's tail.

    `merged_head` (folded in from engine#2683) subtracts the PR's own history rather than replacing the
    base with it: `base..ref ^merged_head`. Subtracting is the tighter of the two, because a branch
    rebased onto a newer base after its PR merged carries the base's commits, which `merged_head..ref`
    would re-examine. For a non-squash merge the frozen head IS an ancestor of the base, so the
    exclusion removes nothing and every existing LOST keeps its meaning -- a detector fix may only
    remove false positives.
    """
    specs = [f"{base_ref}..{ref}"] + ([f"^{merged_head}"] if merged_head else [])
    r = git(repo, "rev-list", "--format=%h %s", *specs)
    if r.returncode != 0:
        return None
    lines = [ln for ln in r.stdout.splitlines() if ln and not ln.startswith("commit ")]
    return [ln.split(" ", 1) for ln in lines]


def classify_commit(repo, sha, base_ref, after=None):
    """LOST / IN-BASE / UNVERIFIABLE for one commit, with the missing lines named.

    # engine#2679 asked for a content test here rather than a line subset, and I measured the two
    # candidates against the real commit rather than choosing one:
    #   git diff old new | git apply --cached --reverse            -> rc 1  (too strict: rewording)
    #   … --reverse -C1                                            -> rc 1  (likewise)
    #   … --reverse --3way                                         -> rc 0  on the true case AND on
    #                                                                        a genuine loss
    # So there is no apply test that separates "the base kept this change in other words" from "the
    # base does not have it": one is too strict, the other too permissive. What the row does instead
    # is show the evidence -- the fraction and the missing lines -- and say that a reworded
    # equivalent is the thing to rule out by reading it.
    """
    names = git(repo, "diff-tree", "-r", "--no-commit-id", "--name-only", sha)
    if names.returncode != 0:
        return "UNVERIFIABLE", "(could not list the commit's files)"
    files = [f for f in names.stdout.splitlines() if f.strip()]
    if not files:
        return "UNVERIFIABLE", "(the commit changes no file)"

    missing_all, checked_all = [], 0
    for f in files:
        d = git(repo, "diff", f"{sha}^..{sha}", "--", f)
        if d.returncode != 0:
            continue
        added = [ln[1:] for ln in d.stdout.splitlines()
                 if ln.startswith("+") and not ln.startswith("+++")]
        want = significant_lines("\n".join(added))
        if not want:
            continue
        checked_all += len(want)
        # engine#2652 sweep: a SUBMODULE GITLINK is not a blob, so `git show <rev>:<path>` prints nothing for it
        # and every added line then reads as absent -- that is why #2583 (third_party/llama.cpp: pin to
        # 148e7f329) and #2554 (third_party/laya: source af8ada06) were the only rows a two-day sweep still
        # reported after the pin-bump class was fixed. A gitlink's content IS its sha, so compare that:
        # equal -> the pointer landed (IN-BASE for this file), different -> fall through to the line test, which
        # will decide LOST/SUPERSEDED for the surrounding text.
        tree = git(repo, "ls-tree", base_ref, "--", f)
        fields = tree.stdout.split()
        if len(fields) >= 3 and fields[0] == "160000":
            mine = git(repo, "ls-tree", sha, "--", f).stdout.split()
            if len(mine) >= 3 and mine[2] == fields[2]:
                continue                      # same submodule commit on both sides: nothing is missing
        show = git(repo, "show", f"{base_ref}:{f}")
        base_text = show.stdout if show.returncode == 0 else ""
        for line in want:
            if line not in base_text:
                missing_all.append((f, line))

    if checked_all == 0:
        # NOT "deletion only": a commit whose every added line was filtered out (too short to be
        # distinctive) lands here too, and the operator sent to look for a deletion would find none.
        added_n = sum(1 for f in files for ln in git(repo, "diff", f"{sha}^..{sha}", "--", f).stdout.splitlines()
                      if ln.startswith("+") and not ln.startswith("+++")) if files else 0
        return "UNVERIFIABLE", (f"(no added line survived the significance filter: {added_n} added, "
                                f"0 usable -- not necessarily deletion)")
    # A commit can be absent from the base because the base SUPERSEDED it (the branch was rebased or
    # reworked before merging, so its earlier formulation never landed while a later one did). The
    # line counts tell the reader which case they are looking at, and they are already in hand:
    # engine#2504 flags at 83 % absent, yet the base's file is 162 lines against the commit's 149 --
    # the feature landed in a fuller form and recovery would be wrong. Without this the row reads as
    # a loss and costs a lane an hour.
    base_lines = {}
    for f, _ in missing_all[:1] or [(None, None)]:
        if f:
            show = git(repo, "show", f"{base_ref}:{f}")
            # An unreadable base path is UNKNOWN, not empty. Reading it as 0 makes the size
            # comparison say "base is shorter" for a file the base may well have rewritten, and
            # that biases every verdict toward LOST -- mea 2026-11-16 while writing the SUPERSEDED
            # arm: a 12-line base file read as "base 0 lines (base is shorter)".
            base_lines[f] = len(show.stdout.splitlines()) if show.returncode == 0 else None
    frac = len(missing_all) / checked_all
    where = "; ".join(f"{f}: {ln[:60]}" for f, ln in missing_all[:3])
    more = "" if len(missing_all) <= 3 else f" (+{len(missing_all) - 3} more)"
    detail = (f"{len(missing_all)} of {checked_all} added line(s) absent "
              f"({frac:.0%}; LOST at {LOST_FRACTION:.0%})")
    if frac >= LOST_FRACTION:
        sizes = ""
        superseded = True   # every missing path has a LONGER base file -> a later commit rewrote it
        for f in {f for f, _ in missing_all}:
            show = git(repo, "show", f"{sha}:{f}")
            mine = len(show.stdout.splitlines()) if show.returncode == 0 else 0
            theirs = base_lines.get(f)
            if theirs is None:
                b = git(repo, "show", f"{base_ref}:{f}")
                # still None when the base cannot be read: "unknown size", not "zero lines"
                theirs = len(b.stdout.splitlines()) if b.returncode == 0 else None
            if theirs is None:
                superseded = False
                sizes += f"; {f}: this commit {mine} lines, base size UNREADABLE"
                continue
            longer = theirs > mine
            # engine#2652 sweep: "the base file is LONGER" is a proxy for "a later edit replaced this line", and
            # it fails for every one-line value file -- a pin bump rewrites the value and leaves the file the
            # same length (measured: #2586/#2572 config/gpu-kernels.config sha=, #2583/#2554 Subproject commit;
            # all reported LOST while main had been bumped after the merge). The exact test needs the PR's
            # MERGE commit as the lower bound: a move before it is this PR's own landing (the squash-tail case,
            # which must stay LOST), a move after it is a later supersession.
            #
            # engine#2693 review: a merge-commit scope on its own is still FILE-granular, so any later PR that
            # edited one unrelated line of the same file turned a genuine LOST into SUPERSEDED -- and
            # report() exits 0 on SUPERSEDED, so the tool's most important class went quiet (measured: the
            # same fixture read LOST on the pre-change version and SUPERSEDED on this one, with the commit
            # still absent). Ask the exact question instead: did a later commit REPLACE one of THESE lines?
            # `git log -S` finds the commits that change a string's occurrence count, which is what "a pin
            # bump rewrote the value" is; a commit that touched only another line is not that.
            since = after or sha
            # engine#2693 review follow-up (cl-infra): this test used to run only when the base file was NOT
            # longer ("a longer base file already answers it"), and that is the one case it does not answer.
            # A longer file only says the base GREW: a feature dropped by the merge plus twenty unrelated
            # lines added afterwards still read SUPERSEDED, and report() exits 0 on SUPERSEDED, so the class
            # went quiet exactly as before (measured: a 3-line function dropped by the merge, 20 unrelated
            # lines added later, 3 of 3 added lines absent -> SUPERSEDED, detail "base is longer"). Ask the
            # question on every file when there is a merge commit to scope it with; moved_since() also asks
            # the weaker token question, so #2504 stays SUPERSEDED for a reason rather than by file length.
            moved, how = "", ""
            if after is not None:
                for path, line in missing_all:
                    if path != f:
                        continue
                    moved = moved_since(repo, line, since, base_ref, f, base_text)
                    if moved:
                        how = line if moved == "line" else distinctive_token(line)
                        break
            superseded = superseded and (bool(moved) if after is not None else longer)
            if moved == "line":
                ctx = "the line was replaced on the base after this PR merged"
            elif moved == "token":
                ctx = f"the base carries `{how}`, which a later commit changed, so the content landed in other words"
            elif longer:
                ctx = "base is longer, but no later commit replaced these lines"
            else:
                ctx = "base is shorter"
            sizes += f"; {f}: this commit {mine} lines, base {theirs} lines ({ctx})"
        # A later commit that rewrote every affected file leaves the added lines absent for the
        # wrong reason. Measured 2026-11-16 on #2504: 15 of 18 lines "absent", base file longer,
        # and `httplib::Server` -- the commit's own subject -- present three times on main. The
        # evidence for SUPERSEDED was already in the detail string while the verdict said LOST.
        if superseded:
            return "SUPERSEDED", (f"{detail} -- {where}{more}{sizes} -- the base rewrote every "
                                  f"affected file, so this is not evidence of a loss")
        return "LOST", f"{detail} -- {where}{more}{sizes}"
    if missing_all:
        return "PARTIAL", f"{detail} -- main edited these lines; the commit is on the base"
    return "IN-BASE", f"all {checked_all} added line(s) already on the base"


def scan(repo, days, limit, base_url=None, prs=None):
    """(kind, pr, branch, commit, detail) rows, or None when the PR list could not be read.

    A generator was the wrong shape here: `return None` inside one raises StopIteration, so a
    failed `gh` call reached the caller as an empty list -- "no findings" for a query that never
    ran. The --self-test arm below pins it.
    """
    rows = []
    if prs is None:
        # `gh pr list --search` answers `[]` with rc 0 for a repository that does not exist, which
        # is byte-identical to a repository with nothing merged. Measured 2026-11-16:
        #   gh pr list -R <bogus> --state merged --search merged:>=X --json number  ->  []  rc 0
        #   gh repo view <bogus>                                                  ->  rc 1
        # So the repository is resolved explicitly, because an unresolvable one must be a gap.
        v = gh("repo", "view", str(base_url), "--json", "name")
        if v.returncode != 0:
            print(f"merged_tail_check: cannot resolve {base_url} "
                  f"({(v.stderr or v.stdout).strip().splitlines()[0] if (v.stderr or v.stdout).strip() else 'no output'})"
                  f" -- cannot tell, and that is not a pass", file=sys.stderr)
            return None
        since = subprocess.run(
            ["date", "-u", "-d", f"-{days} days", "+%Y-%m-%dT%H:%M:%SZ"],
            capture_output=True, text=True).stdout.strip()
        r = gh("pr", "list", "-R", base_url, "--state", "merged", "--limit", str(limit),
               "--search", f"merged:>={since}",
               "--json", "number,title,headRefName,baseRefName,headRepositoryOwner,headRepository,"
                         "headRefOid,mergeCommit")
        # `gh` exits 0 and prints the failure on stdout when the repo does not resolve
        # ("GraphQL: Could not resolve to a Repository with the name ..."), so the exit code cannot
        # tell a real list from an error message. The parsed shape is the only evidence.
        note = (r.stderr or r.stdout).strip().splitlines()
        note = note[0] if note else "(no output)"
        if r.returncode != 0:
            print(f"merged_tail_check: gh pr list failed ({note}) -- cannot tell, and that is "
                  f"not a pass", file=sys.stderr)
            return None
        try:
            prs = json.loads(r.stdout)
        except json.JSONDecodeError:
            print(f"merged_tail_check: gh pr list did not return JSON ({note}) -- cannot tell, "
                  f"and that is not a pass", file=sys.stderr)
            return None
        if not isinstance(prs, list):
            print("merged_tail_check: gh pr list returned no list -- cannot tell, and that is "
                  "not a pass", file=sys.stderr)
            return None

    examined[0] = len(prs)
    for pr in prs:
        owner = (pr.get("headRepositoryOwner") or {}).get("login")
        name = (pr.get("headRepository") or {}).get("name")
        branch = pr.get("headRefName")
        if not (owner and name and branch):
            rows.append(("GONE", pr, branch, None, "the PR record names no head repository or branch"))
            continue
        url = f"https://github.com/{owner}/{name}.git"
        base_ref = f"refs/mtck/base/{pr['baseRefName']}"

        if git(repo, "fetch", "-q", "--no-tags", "origin",
               f"{pr['baseRefName']}:{base_ref}").returncode != 0:
            rows.append(("GAP", pr, branch, None, f"could not fetch base {pr['baseRefName']}"))
            continue

        # Does the branch still exist? A deleted branch hides its tail completely.
        ls = git(repo, "ls-remote", url, f"refs/heads/{branch}")
        if ls.returncode != 0:
            rows.append(("GAP", pr, branch, None, f"could not read {url}"))
            continue
        if not ls.stdout.strip():
            # engine#2652 sweep: 196 of 200 PRs merged in two days reported GONE because the merge loop deletes
            # its branches, so this tool was blind on 98 % of the population. GitHub keeps the PR head reachable
            # at refs/pull/<n>/head after the branch is gone -- that ref is how #2679 was settled -- so use it
            # before giving up, and say which route was used so a reader knows what was examined.
            pull = f"refs/pull/{pr['number']}/head"
            lsp = git(repo, "ls-remote", url, pull)
            if lsp.returncode == 0 and lsp.stdout.strip():
                branch = pull
            else:
                rows.append(("GONE", pr, branch, None,
                             "the branch is deleted and refs/pull/<n>/head is absent: the tail cannot be examined"))
                continue

        # Cheap gate before any fetch. `headRefOid` is frozen at merge time -- measured on #2623,
        # where it is 0193a29e5, a commit that is NOT an ancestor of main while the branch tip is
        # f7700ba0a. So `frozen != tip` means something was pushed (or force-pushed) after the merge
        # and the branch is worth reading; `frozen == tip` means nothing was, and the content check
        # would only re-derive that at the cost of a fetch per PR. (merged_branch_check, #2656.)
        if same_head(pr.get("headRefOid"), ls.stdout.split()[0] if ls.stdout.split() else ""):
            unchanged[0] += 1   # assumed unchanged, not compared -- the summary says so
            continue  # the branch tip is exactly what merged: nothing was pushed after

        ref = f"refs/mtck/{pr['number']}"
        if git(repo, "fetch", "-q", "--no-tags", url, f"{branch}:{ref}").returncode != 0:
            rows.append(("GAP", pr, branch, None, f"could not fetch {url} {branch}"))
            continue

        # The tail is what the branch gained AFTER the head the PR merged at -- not what `main` is
        # not an ancestor of. A squash merge rewrites every commit, so `main..branch` is the WHOLE
        # branch history and every pre-merge commit gets re-litigated against a base that took its
        # content in other words (engine#2679: #2659's own commits reported LOST at 71 % and 100 %).
        # Against the frozen head the same question has one answer: what did the push add?
        frozen = pr.get("headRefOid") or ""
        if frozen and git(repo, "cat-file", "-e", f"{frozen}^{{commit}}").returncode == 0:
            tail = merge_commits(repo, ref, base_ref, frozen)
            origin = f"after the merged head {frozen[:7]}"
        else:
            tail = merge_commits(repo, ref, base_ref)
            origin = "against the base (the frozen head is unavailable)"
        if tail is None:
            rows.append(("GAP", pr, branch, None, "could not list the branch's tail"))
            continue
        if not tail:
            continue  # the branch tip is exactly what merged: the normal case
        for sha, subject in tail:
            kind, detail = classify_commit(repo, sha, base_ref,
                                           after=(pr.get("mergeCommit") or {}).get("oid"))
            rows.append((kind, pr, branch, (sha, subject), detail))
        git(repo, "update-ref", "-d", ref)
    return rows


def report(repo, days, limit, base_url):
    counts = {"SUPERSEDED": 0, "LOST": 0, "PARTIAL": 0, "IN-BASE": 0, "UNVERIFIABLE": 0, "GONE": 0, "GAP": 0}
    rows = scan(repo, days, limit, base_url)
    if rows is None:
        return 2, counts, []
    for kind, pr, branch, commit, detail in rows:
        counts[kind] += 1
        head = f"#{pr['number']} {branch}"
        if commit:
            head = f"{head} {commit[0]} {commit[1][:56]}"
        print(f"{kind:12s} {head}\n{'':12s} {detail}")
    print(f"merged_tail_check: examined {examined[0]} PR(s) merged in the last {days} day(s); "
          f"{examined[0] - counts['GONE']} with a live branch, {counts['GONE']} branch gone, "
          f"{unchanged[0]} tip identical to the merged head (assumed, not compared)")
    print(f"merged_tail_check: {len(rows)} finding(s) over PRs merged in the last {days} day(s) "
          f"-- {counts['LOST']} LOST, {counts['PARTIAL']} partly edited by main, "
          f"{counts['IN-BASE']} already on the base, {counts['SUPERSEDED']} rewritten by a later commit, "
          f"{counts['UNVERIFIABLE']} unverifiable, {counts['GONE']} branch gone, {counts['GAP']} gap")
    if counts["LOST"]:
        print("merged_tail_check: a LOST commit needs a PR that branches from origin/main -- "
              "not a push to the merged branch", file=sys.stderr)
        return 1, counts, rows
    if counts["UNVERIFIABLE"] or counts["GONE"] or counts["GAP"]:
        print("merged_tail_check: something could not be examined; that is not a clean scan",
              file=sys.stderr)
        return 2, counts, rows
    return 0, counts, rows


def self_test():
    import tempfile
    rc = 0

    def chk(want, got, what):
        nonlocal rc
        if want == got:
            print(f"ok   {what}")
        else:
            print(f"FAIL {what}: want {want}, got {got}")
            rc = 1

    def scratch():
        d = tempfile.TemporaryDirectory()
        r = pathlib.Path(d.name)
        # `git init -b` needs git >= 2.28; the self-test must not depend on the runner's git.
        subprocess.run(["git", "init", "-q"], cwd=r, check=True)
        subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=r, check=True)
        subprocess.run(["git", "config", "user.email", "t@e"], cwd=r, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=r, check=True)
        return d, r

    def commit(r, path, text, msg):
        p = r / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        subprocess.run(["git", "add", "-A"], cwd=r, check=True)
        subprocess.run(["git", "commit", "-qm", msg], cwd=r, check=True)
        return git(r, "rev-parse", "HEAD").stdout.strip()

    def branch(r, name, at="HEAD"):
        subprocess.run(["git", "checkout", "-q", "-B", name, at], cwd=r, check=True)

    # The loss: a branch cut, merged at one commit, then pushed to again. main never gets it.
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "c1")
    branch(r, "pr")
    lost_sha = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef WEAK_AFTER_S():\n"
                                    "    return 2\n", "pushed after the merge")
    branch(r, "main", "pr~1")
    chk("LOST", classify_commit(r, lost_sha, "main")[0],
        "a commit pushed after the merge is LOST")

    chk(1, len(merge_commits(r, "pr", "main")), "the branch has a tail of exactly one")
    d.cleanup()

    # The `after=` path had no arm at any head: a pin bump whose one line the base replaced is
    # SUPERSEDED when the base touched that file AFTER the merge, and LOST without it. Same tree,
    # both answers, so the arm is load-bearing rather than decorative.
    d, r = scratch()
    commit(r, "seed.txt", "seed\n", "seed")
    branch(r, "pr")
    pin = commit(r, "pins.config", "sha=53c0ffee00\n", "the pin this branch landed")
    branch(r, "main", "pr~1")
    merged_at = commit(r, "other.txt", "unrelated\n", "the merge point: main goes no further here")
    chk("LOST", classify_commit(r, pin, "main")[0],
        "a one-line pin bump the base never touched is LOST")
    chk("LOST", classify_commit(r, pin, "main", after=merged_at)[0],
        "and it stays LOST with after= while the base never carried that line: a commit that only "
        "wrote a different value into the same file is not a supersession, and SUPERSEDED exits 0")
    commit(r, "pins.config", "sha=53c0ffee00\n", "the same pin lands on the base by another route")
    commit(r, "pins.config", "sha=60b8eef000\n", "a later bump replaces that line")
    chk("SUPERSEDED", classify_commit(r, pin, "main", after=merged_at)[0],
        "the same commit is SUPERSEDED once a later commit replaced the line itself")
    chk("LOST", classify_commit(r, pin, "main")[0],
        "and without after= it is LOST again: the exit-0 path is what after= selects")
    d.cleanup()

    # The same content, already on the base by another commit: not a loss.
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "c0")
    branch(r, "pr")
    base_commit = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef GAP():\n"
                                       "    return 9\n", "base already has GAP")
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "c0")
    root = git(r, "rev-parse", "HEAD").stdout.strip()
    branch(r, "main")
    same_sha = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef GAP():\n"
                                    "    return 9\n", "main gains GAP")
    branch(r, "pr", root)
    twin = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef GAP():\n"
                                "    return 9\n", "the branch adds the same GAP")
    chk("IN-BASE", classify_commit(r, twin, "main")[0],
        "content already on the base is IN-BASE, not a loss")
    d.cleanup()

    # A deletion-only commit cannot be confirmed by looking for lines it adds.
    d, r = scratch()
    commit(r, "f.txt", "a\nb\nc\nd\n", "one")
    branch(r, "main")
    del_sha = commit(r, "f.txt", "a\n", "delete two lines")
    chk("UNVERIFIABLE", classify_commit(r, del_sha, "main")[0],
        "a deletion-only commit is UNVERIFIABLE, not a pass")
    d.cleanup()

    # The threshold: a commit whose lines main later edited is PARTIAL, not LOST. Without this arm
    # a "any line absent" rule calls every landed commit lost -- measured on #2623, where three
    # landed commits flagged 4 %, 25 % and 30 % of their lines absent.
    d, r = scratch()
    commit(r, "seed.txt", "seed\n", "seed")
    ten = "".join(f"value_{i:02d} = compute({i})\n" for i in range(10))
    commit(r, "m.py", ten.replace("value_09 = compute(9)\n", "value_09 = compute(999)\n"),
           "main edited one of the ten")
    branch(r, "pr", "HEAD~1")
    partial_sha = commit(r, "m.py", ten, "the branch adds all ten")
    chk("PARTIAL", classify_commit(r, partial_sha, "main")[0],
        "one line of ten edited away by main is PARTIAL, not LOST")
    d.cleanup()

    # The fifth state. A later commit that REWRITES the file leaves the branch's lines absent and
    # the base file longer. Measured 2026-11-16: #2504's 15-of-18 "absent" lines, base longer,
    # and the commit's own subject (`httplib::Server`) on main three times. The evidence was in the
    # detail string while the verdict said LOST.
    d, r = scratch()
    commit(r, "seed.txt", "seed\n", "seed")
    branch(r, "pr")
    sup_sha = commit(r, "s.py", "".join(f"branch_line_{i} = {i}\n" for i in range(6)),
                     "the branch adds six lines")
    # The rewrite must land on the BASE, not on the branch: branch() moves HEAD, so without this
    # checkout main never receives s.py and the fixture tests "unreadable base", not "rewritten".
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    commit(r, "s.py", "".join(f"rewritten_{i} = f({i})\n" for i in range(12)),
           "a later commit rewrites the region and is longer")
    chk("SUPERSEDED", classify_commit(r, sup_sha, "main")[0],
        "a base that rewrote every affected file is SUPERSEDED, not LOST")
    d.cleanup()

    # The gate: an unknown frozen head must never skip a branch, and equal heads must skip.
    chk(True, same_head("0193a29e5" + "0" * 31, "0193a29e5" + "9" * 31), "an equal frozen head skips")
    chk(False, same_head("0193a29e5" + "0" * 31, "f7700ba0a" + "1" * 31),
        "a moved branch tip does not skip")
    chk(False, same_head("", "f7700ba0a" + "1" * 31),
        "an unknown frozen head does NOT skip -- not knowing is not a pass")

    # engine#2679: after a SQUASH merge the tail must start at the frozen head, not at "whatever
    # main is not an ancestor of". This fixture is the measured shape: two commits authored, the
    # base holding their change in different words, then one push after the merge.
    d, r = scratch()
    commit(r, "seed.txt", "seed\n", "seed")
    branch(r, "pr")
    c1 = commit(r, "tool.py", "def a():\n    return 1\n", "authored 1")
    c2 = commit(r, "tool.py", "def a():\n    return 1\n\n\ndef b():\n    return 2\n", "authored 2")
    branch(r, "main", "pr~2")      # the base, fork point
    commit(r, "tool.py", "def a():\n    return 1\n\n\ndef b():\n    return 2  # squashed\n",
           "the squash merge, in other words")
    branch(r, "pr", c2)            # back on the branch at c2 = the frozen head
    c3 = commit(r, "tool.py", "def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\n"
                              "def c():\n    return 3\n", "pushed after the merge")
    tail = merge_commits(r, "pr", "main", c2)      # the shipped call: base-relative, minus the frozen head
    chk(1, len(tail), "excluding the frozen head, the tail is the one commit pushed after the merge")
    chk(True, tail[0][0] and c3.startswith(tail[0][0]),
        "and it is that commit, not the authored ones")
    chk(False, any(c1.startswith(sha) or c2.startswith(sha) for sha, _ in tail),
        "neither authored commit is re-litigated after the squash")
    chk(3, len(merge_commits(r, "pr", "main")),
        "without the exclusion the tail is the WHOLE branch history -- the #2679 false positive")

    # The exclusion SUBTRACTS the PR's history rather than replacing the base with it, which is what
    # makes it right when the branch was rebased onto a newer base after the merge: those base commits
    # are in `base..ref` and would otherwise be re-examined as if they were post-merge work.
    branch(r, "main", "pr~1")
    commit(r, "other.txt", "a commit main gained while the PR sat merged\n", "main moves on")
    branch(r, "pr", c3)
    commit(r, "tool.py", "def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\n"
                         "def c():\n    return 3\n\n\ndef d():\n    return 4\n",
           "rebased onto the newer base, then pushed")
    chk(1, len(merge_commits(r, "pr", "main", c3)),
        "a rebased branch's tail is still only its own new commit, not the base it was rebased onto")
    d.cleanup()

    # The tail set itself: a branch tip equal to the base has no tail.
    d, r = scratch()
    commit(r, "f.txt", "x\n", "one")
    subprocess.run(["git", "branch", "-q", "same"], cwd=r, check=True)
    chk(0, len(merge_commits(r, "same", "main")), "a branch tip at the base has no tail")
    d.cleanup()

    # A branch carrying an extra commit has a tail of exactly one.
    d, r = scratch()
    commit(r, "f.txt", "x\n", "one")
    subprocess.run(["git", "branch", "-q", "tip"], cwd=r, check=True)
    commit(r, "f.txt", "x\ny\n", "two")
    chk(1, len(merge_commits(r, "HEAD", "tip")), "a branch with one extra commit has a tail of 1")
    d.cleanup()

    # The shape the tool exists for and a normal merge-commit repo never produces: a SQUASH merge rewrites every
    # sha, so the *whole* branch tail reads as absent by sha (tail = 2 here, not 1), and a push that lands after
    # the merge is on a branch nothing will merge again. The added line is deliberately realistic -- a one-char
    # line is filtered by significant_lines() and lands in the UNVERIFIABLE branch instead, which would have made
    # this arm pass for the wrong reason.
    d, r = scratch()
    big = "".join(f"value_{i:02d} = compute({i})\n" for i in range(10))
    commit(r, "m.py", big.replace("value_09 = compute(9)\n", ""), "base")
    branch(r, "main")
    subprocess.run(["git", "checkout", "-q", "-b", "pr"], cwd=r, check=True)
    squash_sha = commit(r, "m.py", big, "the branch adds all ten")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    subprocess.run(["git", "merge", "-q", "--squash", "pr"], cwd=r, check=True)
    commit(r, "m.py", big, "squashed: the branch adds all ten")
    subprocess.run(["git", "checkout", "-q", "pr"], cwd=r, check=True)
    tail_sha = commit(r, "m.py", big + "late_value = compute(99)\n", "pushed after the squash merge")
    # 2, not 1, and that is the point: a squash merge rewrites the sha of the commit it landed, so the branch
    # commit itself reads as absent from main as well. This is exactly why the sha half cannot decide this case
    # and the content arm must -- which is the arm below.
    chk(2, len(merge_commits(r, "pr", "main")),
        "after a squash merge the whole branch tail reads as absent by sha (2: the squashed commit and the push)")
    chk("LOST", classify_commit(r, tail_sha, "main")[0],
        "a commit pushed after a squash merge is LOST by content")
    d.cleanup()

    # engine#2693 review: the `after` form is what the live scan always uses, and it had no arm at all --
    # every call above is the two-argument form, so the exit-0 path below was only ever exercised against
    # the real repo. A LOST commit whose file was ALSO edited elsewhere on the base after the merge: the
    # file-level test called that a supersession and report() exits 0 on it. Measured 2026-11-16 on the
    # same fixture: LOST on the version before the merge-commit scope, SUPERSEDED on the one with it.
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "base")
    branch(r, "pr")
    commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef merged():\n    return 2\n",
           "the head the merge landed")
    branch(r, "main")                       # a squash merge: same content, new sha
    merge = git(r, "rev-parse", "HEAD").stdout.strip()
    branch(r, "pr")
    late_sha = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef merged():\n    return 2\n"
                                  "\n\ndef WEAK_AFTER_S():\n    return 3\n", "pushed after the merge")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    commit(r, "tool.py", "def kept():\n    return 7\n\n\ndef merged():\n    return 2\n",
           "an unrelated PR edits kept() on main after the merge")
    chk("LOST", classify_commit(r, late_sha, "main", after=merge)[0],
        "a LOST commit whose file was edited elsewhere after the merge stays LOST with after=")
    d.cleanup()

    # engine#2693 review follow-up (cl-infra): the per-line test ran only when the base file was NOT longer,
    # so "the base is longer" was still a proxy verdict -- and a file growing is the common case. A feature
    # dropped by the merge plus an unrelated commit that ADDS lines to the same file read SUPERSEDED, and
    # report() exits 0 on that, which is the silence this tool exists to prevent. Measured before the fix:
    # 3 of 3 added lines absent, detail "base is longer" -> SUPERSEDED.
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "base")
    branch(r, "pr")
    commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef unrelated():\n    return 0\n",
           "the head the merge landed")
    branch(r, "main")                       # a squash merge: same content, new sha
    merge = git(r, "rev-parse", "HEAD").stdout.strip()
    branch(r, "pr")
    dropped_sha = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef unrelated():\n    return 0\n"
                                    "\n\ndef recover_route_lock(rows):\n    total = sum(rows)\n"
                                    "    return total / max(len(rows), 1)\n", "pushed after the merge")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef unrelated():\n    return 0\n"
                         + "".join(f"UNRELATED_{i} = {i}\n" for i in range(20)),
           "an unrelated commit GROWS the same file after the merge")
    chk("LOST", classify_commit(r, dropped_sha, "main", after=merge)[0],
        "a LOST commit stays LOST when an unrelated commit only GROWS the same base file")
    d.cleanup()

    # ...and the case that must stay SUPERSEDED, so the arm above cannot be satisfied by calling everything
    # LOST: the base carries the content in OTHER WORDS and a later commit changed it (#2504's shape, measured
    # there as `-S <line>` silent while `-S httplib::Server` finds a20f6ee89 and main carries it three times).
    # The token test requires the token to be ON the base, which is what keeps a dropped feature from being
    # called superseded by an unrelated commit that happens to mention a shared word.
    d, r = scratch()
    commit(r, "tool.py", "def kept():\n    return 1\n", "base")
    branch(r, "pr")
    commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef unrelated():\n    return 0\n",
           "the head the merge landed")
    branch(r, "main")                       # the squash merge, in other words: the feature is NOT here
    merge = git(r, "rev-parse", "HEAD").stdout.strip()
    branch(r, "pr")
    rewritten_sha = commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef unrelated():\n    return 0\n"
                                      "\n\ndef recover_route_lock(rows):\n    total = sum(rows)\n"
                                      "    return total / max(len(rows), 1)\n", "pushed after the merge")
    subprocess.run(["git", "checkout", "-q", "main"], cwd=r, check=True)
    commit(r, "tool.py", "def kept():\n    return 1\n\n\ndef unrelated():\n    return 0\n\n"
                         "# recover_route_lock lives in the C++ guard test now (#2508)\n",
           "a later commit carries the same content by name, in other words")
    chk("SUPERSEDED", classify_commit(r, rewritten_sha, "main", after=merge)[0],
        "a base that carries the missing line's own token, changed later, is SUPERSEDED (#2504's shape)")
    d.cleanup()

    # The gitlink route (#2583/#2554): a submodule pointer prints nothing from `git show <rev>:<path>`, so
    # every added `Subproject commit ...` line read as absent. The same pointer on both sides is IN-BASE,
    # and without this arm that half is the one part of the change no test reaches.
    d, r = scratch()
    commit(r, "seed.txt", "seed\n", "seed")
    subprocess.run(["git", "update-index", "--add", "--cacheinfo",
                    "160000," + "1" * 40 + ",third_party/pinned"], cwd=r, check=True)
    subprocess.run(["git", "commit", "-qm", "the branch pins the submodule"], cwd=r, check=True)
    pin = git(r, "rev-parse", "HEAD").stdout.strip()
    chk("IN-BASE", classify_commit(r, pin, "main")[0],
        "a submodule pointer already at the same sha is IN-BASE, not LOST")
    d.cleanup()

    # Zero findings must be distinguishable from not having looked: an unreadable PR list is rc 2.
    chk(2, report(pathlib.Path("."), 1, 1, "1bit-MONSTER/no-such-repo-xyz")[0],
        "an unreadable PR list is rc 2, not an empty pass")

    # shortcut: the shallow-clone refusal has no arm -- a local `--depth 1` fixture kept cloning an empty repository (git's
    # own "cloned an empty repository" warning), and shipping a red arm is worse than shipping none. The guard was verified by
    # hand against a genuine shallow clone of this repo (it refuses, and --allow-shallow proceeds); upgrade by cloning a real
    # origin over file:// with two commits and asserting both directions.
    # And the arm CI added: this row FAILED in the build container because `gh` is not installed
    # there, so the call raised FileNotFoundError instead of answering "I could not look". Measured
    # on #2659: 11/11 arms locally, 75 - merged_tail_check_self (Failed) in CI.
    global GH_BIN
    saved, GH_BIN = GH_BIN, "gh-that-is-not-installed"
    try:
        chk(2, report(pathlib.Path("."), 1, 1, "1bit-MONSTER/engine")[0],
            "a missing gh is rc 2, not a crash and not an empty pass")
    finally:
        GH_BIN = saved

    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--self-test", action="store_true", help="run the arms and exit")
    ap.add_argument("--repo", default="1bit-MONSTER/engine", help="repo to read merged PRs from")
    ap.add_argument("--days", type=int, default=30, help="how far back to look (default 30)")
    ap.add_argument("--limit", type=int, default=200, help="max merged PRs to examine")
    ap.add_argument("--worktree", default=".", help="a clone with origin = --repo")
    ap.add_argument("--branch", help="instead of scanning PRs, list what this ref has that "
                                    "--base does not (the recovery workflow)")
    ap.add_argument("--base", default="origin/main", help="base for --branch (default origin/main)")
    ap.add_argument("--allow-shallow", action="store_true",
                    help="report anyway in a shallow clone, where main's own commits read as LOST")
    a = ap.parse_args()
    # engine#2652 (found by a lane running this on a --depth 50 clone): a SHALLOW clone makes this tool lie. The first run
    # there reported 22 LOST where the full clone reports 4, and 18 of the 22 were main's own commits -- they had come onto the
    # local base by a merge whose parents are not present, so every commit looks absent from the base. The results are not
    # merely noisier, they are mostly false, so refuse rather than report: a caller who wants the number anyway can say so.
    if a.worktree and not a.self_test:
        shallow = (git(a.worktree, "rev-parse", "--is-shallow-repository").stdout or "").strip()
        if shallow == "true" and not a.allow_shallow:
            raise SystemExit("%s is a SHALLOW clone: this tool reads commit ancestry, and a shallow clone reports main's own "
                             "commits as LOST (measured: 22 LOST on --depth 50 against 4 on the full clone, 18 of them main's). "
                             "Deepen it (git fetch --unshallow) or pass --allow-shallow to report anyway." % a.worktree)
    if a.self_test:
        return self_test()
    if a.branch:
        wt = pathlib.Path(a.worktree)
        if git(wt, "fetch", "-q", "--no-tags", "origin",
               a.base.replace("origin/", "")).returncode != 0:
            print(f"merged_tail_check: could not fetch {a.base} -- cannot tell, and that is not "
                  f"a pass", file=sys.stderr)
            return 2
        tail = branch_mode(wt, a.branch, a.base)
        if tail is None:
            print(f"merged_tail_check: could not read {a.branch} -- cannot tell, and that is not "
                  f"a pass", file=sys.stderr)
            return 2
        return 1 if tail else 0
    rc, _, _ = report(pathlib.Path(a.worktree), a.days, a.limit, a.repo)
    return rc


if __name__ == "__main__":
    sys.exit(main())
