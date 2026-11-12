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

"""conflict_triage.py -- classify a merge's conflicts by KIND, so a big merge is worked by category.

engine#2138 is a 149-file / 538-hunk merge of AMD's rebased HRX line into our fork, and its own table names the reason
a file-by-file walk is the wrong instrument:

    ggml/src/ggml-hrx/kernel-corpus | 41 paths, mostly add/delete or rename with no markers | 30

A conflict with no markers is invisible to every text-based tool -- `grep '^<<<<<<<'` reports 103 files on the same
merge while the real count is 149 (engine#1945's triage). The categories therefore have to come from the INDEX, not
from the file body, and they decide the work:

    content   markers to resolve, one judgement per hunk        -> a person reads both sides
    add/add   both sides added the path                         -> often `--ours` wholesale, after checking
    mod/del   one side deleted, the other changed it            -> a decision about intent, not text
    rename    a rename on one side against an edit on the other -> follow the lineage before resolving

`git merge-tree --write-tree` (git >= 2.38) reports all of it without a checkout, which is what makes this safe to run
on a tree nobody wants to disturb.

usage: conflict_triage.py <repo> <theirs> [--base BASE] [--json]
       conflict_triage.py --markers REF [--json]   classify the markers already committed in a tree
       conflict_triage.py --self-test

The two halves answer different questions and both are needed on a long merge. `<theirs>` says what a merge *would*
conflict on; `--markers REF` says what a tree that has already been merged still *carries*, which is the state a
work-in-progress merge is in for hours and the state engine#327's draft reports (23 files, 92 hunks).
exit:  0 no conflicts, 1 conflicts found, 2 usage or git error
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

KINDS = ("content", "add/add", "mod/del", "rename")


def die(message):
    """Say it on stderr and exit 2 -- the documented code for a usage or git error.

    `raise SystemExit("text")` exits **1**, which is this tool's "conflicts found" code, so a bad ref would have
    been indistinguishable from a merge that needs work.
    """
    print(message, file=sys.stderr)
    raise SystemExit(2)


def git(repo, *args, check=False):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def must_resolve(repo, ref):
    """`ref^{commit}`, or exit 2 -- a ref git cannot resolve must never look like a clean merge.

    `git merge-tree` returns 1 both for "conflicts found" and for "<ref> - not something we can merge", so the
    return code cannot tell them apart. Measured on a throwaway repo: a typo'd ref printed "no conflicts" and
    exited 0, which is the unreadable-is-not-zero defect inside this tool.
    """
    out = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if out.returncode != 0 or not out.stdout.strip():
        die(f"conflict_triage: cannot resolve {ref!r} in {repo} (not a commit here)")
    return out.stdout.strip()


def merge_tree(repo, base, theirs):
    """The tree git would produce, its conflicted paths, and the messages that classify them."""
    must_resolve(repo, "HEAD")
    must_resolve(repo, theirs)
    if base:
        must_resolve(repo, base)
    args = ["merge-tree", "--write-tree", "--name-only"]
    if base:
        args.append(f"--merge-base={base}")
    # `git merge-tree --write-tree [base] ours theirs`: ours is the checked-out HEAD.
    args += ([base] if base else []) + ["HEAD", theirs]
    out = git(repo, *args)
    if out.returncode == 0:
        return {"tree": out.stdout.splitlines()[0].strip() if out.stdout.strip() else "", "paths": [], "info": ""}
    if out.returncode != 1:
        die(f"conflict_triage: git merge-tree failed: {out.stderr.strip()[:400]}")
    # rc 1 is git's code for "conflicts" AND for a ref it could not merge; the refs are verified above, so a message
    # here is something else refusing and must not be reported as a conflict-free merge.
    if "not something we can merge" in (out.stdout + out.stderr):
        die(f"conflict_triage: git could not merge the refs: {out.stderr.strip()[:300]}")
    lines = out.stdout.splitlines()
    tree = lines[0].strip() if lines else ""
    paths, info_start = [], None
    for i, line in enumerate(lines[1:], 1):
        if line.startswith("Auto-merging") or line.startswith("CONFLICT") or line.startswith("changed in"):
            info_start = i
            break
        if line.strip():
            paths.append(line.strip())
    info = "\n".join(lines[info_start:]) if info_start is not None else ""
    return {"tree": tree, "paths": paths, "info": info}


def message_path(info_line):
    """The path a CONFLICT line names, taken from git's own wording rather than matched against a list.

    Two shapes, and suffix-matching the path list got both wrong on a real merge (5 of 202 paths fell to
    `unmapped` that git had classified):
      CONFLICT (content): Merge conflict in <path>
      CONFLICT (modify/delete): <path> deleted in HEAD and modified in ...
    A rename message names a pair and is left to the caller's suffix match.
    """
    marker = "Merge conflict in "
    at = info_line.find(marker)
    if at >= 0:
        return info_line[at + len(marker):].strip()
    m = re.match(r"CONFLICT \([a-z/]+\): (?P<path>.+?) (?:deleted|modified|renamed) ", info_line)
    return m.group("path").strip() if m else None


def classify(info_line):
    """The kind of conflict from git's own message for it, or None when the message does not say."""
    # The specific kinds FIRST: git spells an add/add conflict as `CONFLICT (add/add): Merge conflict in <path>`,
    # so testing the generic phrase first reports every add/add as content -- which the self-test caught.
    if "add/add" in info_line:
        return "add/add"
    if "modify/delete" in info_line or "delete/modify" in info_line:
        return "mod/del"
    if "rename/rename" in info_line or "rename/delete" in info_line or "rename/add" in info_line:
        return "rename"
    if "Merge conflict in" in info_line:
        return "content"
    return None


def rows(paths, info, markers):
    """[(path, kind, hunks)] for every conflicted path.

    `markers` maps path -> hunk count for the paths that carry conflict markers in the working tree; a path git
    reports without a message and without markers is the case this tool exists for, and it is reported as `unmapped`
    rather than guessed.
    """
    by_path = {}
    for line in info.splitlines():
        kind = classify(line)
        if kind is None:
            continue
        got = message_path(line)
        if got is None:
            # git's rename messages name two paths ("x => y"), so fall back to a unique suffix match.
            for p in paths:
                if line.endswith(p) and p not in by_path:
                    by_path[p] = kind
                    break
            continue
        if got in paths and got not in by_path:
            by_path[got] = kind
    out = []
    for p in paths:
        kind = by_path.get(p)
        if kind is None:
            kind = "content" if markers.get(p) else "unmapped"
        out.append((p, kind, markers.get(p, 0)))
    return out


def count_markers(root):
    """path -> number of `<<<<<<<` lines, for a working tree that already has the merge in it."""
    counts = {}
    for path in pathlib.Path(root).rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n = body.count("\n<<<<<<< ") + (1 if body.startswith("<<<<<<< ") else 0)
        if n:
            counts[path.relative_to(root).as_posix()] = n
    return counts


def summarise(triaged):
    total = len(triaged)
    per = {k: 0 for k in KINDS + ("unmapped",)}
    for _, kind, _ in triaged:
        per[kind] = per.get(kind, 0) + 1
    return total, per


MARK = "<<<<<<<"


def marker_rows(repo, ref):
    """[(path, hunks)] for every path in `ref` whose blob carries conflict markers."""
    out = git(repo, "grep", "-c", f"^{MARK}", ref, "--")
    # `git grep -c` exits 1 with no output when nothing matches, which is the clean case rather than an error.
    if out.returncode == 1 and not out.stdout.strip():
        return []
    if out.returncode != 0:
        die(f"conflict_triage: git grep failed on {ref!r}: {out.stderr.strip()[:300]}")
    rows = []
    prefix = f"{ref}:"
    for line in out.stdout.splitlines():
        rest = line[len(prefix):] if line.startswith(prefix) else line
        # the count is after the LAST colon: a path may not contain one, but splitting at the first colon took
        # "<ref>" as the path and "<path>:<count>" as the number, which silently produced an empty list
        path, _, count = rest.rpartition(":")
        try:
            rows.append((path, int(count)))
        except ValueError:
            continue
    return sorted(rows)


def area_of(path):
    """The grouping a resolver works in: the directory two levels deep, which is where these trees divide."""
    parts = path.split("/")
    if parts[0] == "ggml" and len(parts) > 2:
        return "/".join(parts[:3])
    return "/".join(parts[:2]) if len(parts) > 1 else parts[0]


def markers(repo, ref, as_json=False):
    rows = marker_rows(repo, ref)
    if not rows:
        print(f"conflict_triage: no conflict markers in {ref}")
        return 0
    per_area = {}
    for path, hunks in rows:
        a = area_of(path)
        f, h = per_area.get(a, (0, 0))
        per_area[a] = (f + 1, h + hunks)
    total = sum(h for _, h in rows)
    if as_json:
        print(json.dumps({"ref": ref, "total_files": len(rows), "total_hunks": total,
                          "areas": {k: {"files": v[0], "hunks": v[1]} for k, v in sorted(per_area.items())},
                          "rows": [{"path": p, "hunks": h} for p, h in rows]}, indent=2))
        return 1
    for path, hunks in rows:
        print(f"{hunks:4d}  {path}")
    print()
    for a, (f, h) in sorted(per_area.items(), key=lambda kv: -kv[1][1]):
        print(f"conflict_triage: {a:32} {f:3d} file(s) {h:4d} hunks")
    print(f"conflict_triage: {len(rows)} file(s), {total} hunk(s) still carry markers in {ref}")
    return 1


def run(repo, theirs, base=None, as_json=False):
    result = merge_tree(repo, base, theirs)
    if not result["paths"]:
        print("conflict_triage: no conflicts")
        return 0
    triaged = rows(result["paths"], result["info"], count_markers(repo))
    total, per = summarise(triaged)
    if as_json:
        print(json.dumps({"tree": result["tree"], "total": total, "counts": per,
                          "rows": [{"path": p, "kind": k, "hunks": n} for p, k, n in sorted(triaged)]}, indent=2))
        return 1
    for path, kind, hunks in sorted(triaged):
        print(f"{kind:9} {hunks:4d}  {path}")
    print()
    print("conflict_triage: %d conflicted path(s): %s -- %s" % (
        total, " ".join(f"{k}={per[k]}" for k in KINDS + ("unmapped",) if per.get(k)),
        "content needs a reading, everything else needs a decision"))
    return 1


def _die_rc():
    """The exit code `die()` produces, measured in a child process."""
    out = subprocess.run([sys.executable, __file__, "--self-test-die"], capture_output=True, text=True)
    return out.returncode


def self_test():
    failures = 0

    def check_that(ok, what):
        nonlocal failures
        print(("ok   " if ok else "FAIL ") + what)
        if not ok:
            failures += 1

    check_that(classify("CONFLICT (content): Merge conflict in CMakeLists.txt") == "content",
               "a content conflict is 'content'")
    check_that(classify("CONFLICT (add/add): Merge conflict in npu/x.cpp") == "add/add",
               "an add/add conflict is 'add/add'")
    check_that(classify("CONFLICT (modify/delete): a.h deleted in HEAD and modified in theirs") == "mod/del",
               "a modify/delete conflict is 'mod/del'")
    check_that(classify("CONFLICT (rename/rename): a.cpp renamed in both") == "rename",
               "a rename conflict is 'rename'")
    check_that(classify("Auto-merging something") is None, "an auto-merge line is not a conflict")
    # The case the tool exists for: a path git reports with no message and no markers.
    triaged = rows(["ggml/src/ggml-hrx/corpus/a.loom", "CMakeLists.txt"],
                   "CONFLICT (content): Merge conflict in CMakeLists.txt", {})
    kinds = {p: k for p, k, _ in triaged}
    check_that(kinds["CMakeLists.txt"] == "content", "a message classifies its own path")
    check_that(kinds["ggml/src/ggml-hrx/corpus/a.loom"] == "unmapped",
               "a path with no message and no markers is 'unmapped', not guessed")
    triaged = rows(["a/x.loom"], "", {"a/x.loom": 3})
    check_that(triaged[0][1] == "content" and triaged[0][2] == 3,
               "no message + markers is content with its hunk count")
    total, per = summarise([("a", "content", 2), ("b", "add/add", 0), ("c", "unmapped", 0)])
    check_that(total == 3 and per["content"] == 1 and per["add/add"] == 1 and per["unmapped"] == 1,
               "the summary counts each kind")
    # A path containing a space must classify from its message (the suffix match).
    triaged = rows(["dir with space/a b.txt"], "CONFLICT (content): Merge conflict in dir with space/a b.txt", {})
    check_that(triaged[0][1] == "content", "a path with spaces classifies from its message")
    # Found by running it on a real merge (engine#2138): 5 of 202 paths fell to `unmapped` because the classifier
    # suffix-matched the path list instead of reading git's own message. Both message shapes must carry their path.
    check_that(message_path("CONFLICT (content): Merge conflict in ggml/src/CMakeLists.txt")
               == "ggml/src/CMakeLists.txt", "a content message carries its path")
    check_that(message_path("CONFLICT (modify/delete): docs/a b.md deleted in HEAD and modified in theirs")
               == "docs/a b.md", "a modify/delete message carries its path, spaces included")
    triaged = rows(["ggml/src/CMakeLists.txt"],
                   "CONFLICT (content): Merge conflict in ggml/src/CMakeLists.txt", {})
    check_that(triaged[0][1] == "content", "a path whose message names it is never 'unmapped'")
    # The exit codes are the contract: 2 for "I could not answer", 1 for "conflicts", 0 for "none". A bad ref used to
    # take the 1 path (SystemExit's default) or, before the refs were verified, the 0 path -- both worth an arm.
    check_that(_die_rc() == 2, "die() exits 2, not SystemExit's default 1 (which is 'conflicts found')")
    # `--markers` parsing, both bugs found by running it on engine#327: `git grep -c` prefixes every line with the
    # ref, and the count is after the LAST colon -- splitting at the first colon took "<ref>" as the path and
    # "<path>:<count>" as the number, so every row was dropped and the mode reported a clean tree.
    with tempfile.TemporaryDirectory() as repo:
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
        with open(os.path.join(repo, "a.cpp"), "w") as f:
            f.write("int a;\n<<<<<<< ours\nx\n=======\ny\n>>>>>>> theirs\n<<<<<<< ours\nz\n=======\nw\n>>>>>>> theirs\n")
        with open(os.path.join(repo, "clean.cpp"), "w") as f:
            f.write("int b;\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "m"], cwd=repo, check=True)
        got = marker_rows(repo, "HEAD")
        check_that(got == [("a.cpp", 2)], f"markers: a two-hunk file is reported with its count (got {got})")
        check_that(area_of("ggml/src/ggml-hrx/runtime/x.cpp") == "ggml/src/ggml-hrx",
                   "markers: the area is the directory a resolver works in")
        check_that(area_of("src/models/qwen4exp.cpp") == "src/models", "markers: and for a src subdirectory")
    print(f"conflict_triage self-test: {'ok' if not failures else f'{failures} failure(s)'}")
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("repo", nargs="?")
    ap.add_argument("theirs", nargs="?")
    ap.add_argument("--base", default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--markers", metavar="REF", default=None,
                    help="classify the conflict markers already committed in REF")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if "--self-test-die" in sys.argv:
        die("conflict_triage: self-test of die()")
    if args.self_test:
        return self_test()
    if args.markers:
        return markers(args.repo or ".", args.markers, args.json)
    if not args.repo or not args.theirs:
        ap.print_help()
        return 2
    return run(args.repo, args.theirs, args.base, args.json)


if __name__ == "__main__":
    sys.exit(main())
