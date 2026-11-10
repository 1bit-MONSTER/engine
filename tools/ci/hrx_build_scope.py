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
"""Decide whether an HRX build at PR time needs to compile, or only to configure (engine#1499).

`hrx-build.yml` triggers on `CMakeLists.txt`, which is right: an `ONEBIT_HRX=ON` build reads it. But a PR whose only change
there is a ctest *registration* -- `add_test(...)`, `set_tests_properties(...)` -- creates no target and sets no flag, so it
cannot change what compiles; it can still break the **configure** (those lines are evaluated at configure time, including
inside `if(ONEBIT_HRX)` blocks that `ci.yml` never enters). Such a PR should pay a configure, not an eight-minute compile on
the bench runner.

This decides that, and it is deliberately **conservative**: anything it is not certain about is `full`. The build it is
skipping is the one that catches a broken HRX pin, so a wrong `configure-only` costs more than a slow PR.

Rules -- `configure-only` (alias `ctest-only`) requires ALL of:

1. every changed path *that triggers the workflow* is a **registration file** -- `CMakeLists.txt` or a
   `cmake/tests-*.cmake` family file included from it (engine#1696 moved the `gpu_hold_*` registrations into one, because
   every test-adding PR edited the same contiguous block there and any two of them conflicted by construction); any other
   trigger path is `full`. The
   triggers are read from `hrx-build.yml`'s `paths:` at run time, so the two cannot drift. A path outside them (the
   new test's own script, docs) is not an input of the HRX build: on its own it does not run this job at all, so it
   must not force a compile here either. Without it, no real registration PR would ever skip: replayed over main, the
   four registration-only commits and #1491 each carried their test file and were all `full` (engine#1500). If the
   workflow's list cannot be read, every changed path counts, as before;
2. every added and removed line in those files matches `add_test(`, `set_tests_properties(` or `set_property(TEST ... `,
   or is blank/comment -- a balance check then requires the removed and added lines to agree in kind, so a registration
   cannot be swapped for a `target_link_libraries`;
3. the registrations do not sit inside a `foreach`/`function`/`macro` block, because a registration emitted from a loop or a
   function runs at configure time in a way a line-matching rule cannot reason about.

usage:
  tools/ci/hrx_build_scope.py --base <sha> [--head HEAD]      # decide from the git history (prints full|configure-only)
  tools/ci/hrx_build_scope.py --diff FILE --name-only FILE     # decide from saved `git diff` output (used by --self-test)
  tools/ci/hrx_build_scope.py --self-test
"""
import argparse
import os
import re
import subprocess
import sys

WORKFLOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".github", "workflows", "hrx-build.yml")


def workflow_triggers(path=WORKFLOW):
    """hrx-build.yml's pull_request `paths:` list, or None when it cannot be read (then every path counts)."""
    try:
        text = open(path).read()
    except OSError:
        return None
    # A YAML comment inside the list is legal and must be skipped: the first version stopped at the first line that was
    # not a `- entry`, so one comment truncated the trigger set silently and every entry after it became "not a trigger".
    # Found while adding the `cmake/tests-*.cmake` entry for engine#1696 (the entry itself landed later, with the glob
    # match in is_trigger() below; the skip is still the right behaviour for anyone who adds a comment).
    m = re.search(r"^\s*paths:\s*\n((?:\s*(?:#[^\n]*|-\s*\S+)\s*\n)+)", text, re.M)
    if not m:
        return None
    listed = re.findall(r"^\s*-\s*(\S+)\s*$", m.group(1), re.M)
    return listed or None


def _glob_re(entry):
    """A paths-filter entry as GitHub reads it: `*` stays inside one path segment, `**` crosses segments."""
    return re.compile("^" + re.escape(entry).replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace("\0", ".*") + "$")


def is_trigger(path, triggers):
    # A glob entry (`cmake/tests-*.cmake`) must match as a glob. Read literally it matched nothing, so a PR touching
    # only a family file came out "no trigger path" -> full, and the compile ran for a registration-only change.
    return triggers is None or any(
        _glob_re(t).match(path) if "*" in t else (path == t or path.startswith(t.rstrip("/") + "/")) for t in triggers)

REGISTRATION = re.compile(r"^\s*(add_test\s*\(|set_tests_properties\s*\(|set_property\s*\(\s*TEST\b)")
BENIGN = re.compile(r"^\s*(#|$)")
BLOCK_OPEN = re.compile(r"^\s*(foreach|function|macro)\s*\(", re.IGNORECASE)


def classify_line(line):
    """What a changed CMake line is, for the balance check."""
    if BENIGN.match(line):
        return "benign"
    if REGISTRATION.match(line):
        return "registration"
    return "other"


# engine#1696: a family of registrations may live in cmake/tests-<family>.cmake and be included by CMakeLists.txt. Such a
# file is a registration file in exactly the sense the rule below means: it creates no target and sets no flag, so a PR
# whose only change is a registration there must still classify `configure-only`.
REGISTRATION_FILE = re.compile(r"^(?:CMakeLists\.txt|cmake/tests-[A-Za-z0-9._-]+\.cmake)$")


def is_registration_file(path):
    return bool(REGISTRATION_FILE.match(path))


def decide(changed_paths, cmake_diff, triggers=None):
    """(verdict, reason) for one PR.

    `changed_paths` is every path the PR touches; `cmake_diff` is the unified diff text for the CMakeLists.txt files;
    `triggers` is the workflow's paths filter (None: every path counts).
    """
    if not [p for p in changed_paths if p]:
        return "full", "no changed paths were given"
    paths = sorted({p for p in changed_paths if p and is_trigger(p, triggers)})
    if not paths:
        return "full", "no changed path is one of the workflow's triggers (a manual run): not provable"
    other = [p for p in paths if not is_registration_file(p)]
    if other:
        return "full", f"{len(other)} changed path(s) outside the registration files ({', '.join(other[:3])})"
    added = [l[1:] for l in cmake_diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in cmake_diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    if not added and not removed:
        # The diff spans every registration file since #1744, so naming CMakeLists.txt here would misreport a mode
        # change on cmake/tests-*.cmake -- and this tool's whole output is the reason string.
        return "full", ("the registration file(s) appear in the diff with no line changes (rename or mode change): "
                        "not provable")
    kinds_add = [classify_line(l) for l in added]
    kinds_del = [classify_line(l) for l in removed]
    for kind, lines in (("added", added), ("removed", removed)):
        for line, kind_of_line in zip(lines, [classify_line(l) for l in lines]):
            if kind_of_line == "other":
                return "full", f"{kind} line is not a ctest registration: {line.strip()[:80]!r}"
    # No equality requirement: a PR that only *adds* registrations is exactly the case this exists for, and a
    # registration swapped for anything else was already caught above by the per-line kind check (a
    # `target_link_libraries` + line is "other", not "registration").
    # A registration emitted from inside a loop or a function is not a line this rule can reason about.
    for line in added:
        if BLOCK_OPEN.match(line):
            return "full", f"a block opened in the diff: {line.strip()[:80]!r}"
    if not [k for k in kinds_add if k == "registration"]:
        return "full", "no ctest registration line is being ADDED (a pure removal cannot be told from a rewrite)"
    return "configure-only", (f"{len([k for k in kinds_add if k == 'registration'])} ctest registration line(s), "
                              f"registration files only")


# The diff decide() classifies must cover EVERY registration file, not only CMakeLists.txt. With only CMakeLists.txt
# here, a family file's own lines were never read: an add_test in CMakeLists.txt next to a target_link_libraries in
# cmake/tests-gpu-hold.cmake came out configure-only, skipping the compile that exists to catch exactly that (found
# probing main at bc21ba3 after #1699). The pathspec mirrors REGISTRATION_FILE.
REGISTRATION_PATHSPEC = ["CMakeLists.txt", ":(glob)cmake/tests-*.cmake"]


def from_git(base, head, cwd=None):
    names = subprocess.run(["git", "diff", "--name-only", base, head], capture_output=True, text=True, check=True, cwd=cwd)
    diff = subprocess.run(["git", "diff", base, head, "--", *REGISTRATION_PATHSPEC], capture_output=True, text=True,
                          check=True, cwd=cwd)
    return decide(names.stdout.splitlines(), diff.stdout, workflow_triggers())


def git_arm():
    """from_git() on a real repo: the layer the diff-text cases above cannot reach. None when git is unavailable."""
    import shutil
    import tempfile
    if not shutil.which("git"):
        return None
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        def git(*a):
            subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=d, check=True,
                           capture_output=True)
        git("init", "-q")
        os.makedirs(os.path.join(d, "cmake"))
        for f, text in (("CMakeLists.txt", "add_test(NAME a COMMAND a)\n"),
                        ("cmake/tests-gpu-hold.cmake", "add_test(NAME b COMMAND b)\n")):
            open(os.path.join(d, f), "w").write(text)
        git("add", "-A")
        git("commit", "-q", "-m", "base")
        base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True, check=True).stdout.strip()
        arms = [
            ("git: an add_test in CMakeLists.txt next to a target line in the family file",
             {"CMakeLists.txt": "add_test(NAME c COMMAND c)\n",
              "cmake/tests-gpu-hold.cmake": "target_link_libraries(onebit PRIVATE x)\n"}, "full"),
            ("git: registrations added to both files",
             {"CMakeLists.txt": "add_test(NAME d COMMAND d)\n",
              "cmake/tests-gpu-hold.cmake": "add_test(NAME e COMMAND e)\n"}, "configure-only"),
        ]
        for name, appends, want in arms:
            git("checkout", "-q", "-B", "arm", base)
            for f, text in appends.items():
                open(os.path.join(d, f), "a").write(text)
            git("commit", "-q", "-am", name)
            got, why = from_git("HEAD^1", "HEAD", cwd=d)
            ok = got == want
            print(("ok   " if ok else "FAIL ") + f"{name} -> {got}" + ("" if ok else f" (wanted {want}: {why})"))
            bad += 0 if ok else 1
    return bad


def self_test():
    """The cases that make the rule trustworthy: each one an artefact someone would actually push."""
    cases = [
        # name, paths, diff, expected verdict
        # engine#1696: a registration added to an included family file is still a registration -- this is the shape every
        # later gpu_hold_* test PR has, and it must not pay the ONEBIT_HRX compile
        ("a registration added to an included cmake/tests-*.cmake family file",
         ["CMakeLists.txt", "cmake/tests-gpu-hold.cmake"],   # the script it runs is not a build input, see the #1491 arm
         "+++ b/cmake/tests-gpu-hold.cmake\n+add_test(NAME gpu_hold_new COMMAND bash x.sh)\n",
         "configure-only"),
        # the first move itself edits CMakeLists.txt with an include(), which is not a registration: full, and that is
        # correct -- the rule must not learn to wave through arbitrary lines in that file
        ("the first move, which adds include() to CMakeLists.txt",
         ["CMakeLists.txt", "cmake/tests-gpu-hold.cmake"],
         "+++ b/CMakeLists.txt\n+include(cmake/tests-gpu-hold.cmake)\n", "full"),
        # ... and a change there that is not a registration still pays the compile
        ("a non-registration line in cmake/tests-*.cmake",
         ["cmake/tests-gpu-hold.cmake"],
         "+++ b/cmake/tests-gpu-hold.cmake\n+target_link_libraries(x y)\n", "full"),
        ("a lone add_test with its if/endif context",
         ["CMakeLists.txt"],
         "+++ b/CMakeLists.txt\n@@ -1,3 +1,6 @@\n add_executable(t tests/t.cpp)\n+# a new check\n+add_test(NAME t COMMAND t)\n+\n", "configure-only"),
        ("swap a registration for a target link (must NOT skip the build)",
         ["CMakeLists.txt"],
         "+++ b/CMakeLists.txt\n@@ -1,2 +1,2 @@\n-add_test(NAME a COMMAND a)\n+target_link_libraries(a PRIVATE b)\n", "full"),
        ("a registration plus a flag change",
         ["CMakeLists.txt"],
         "+++ b/CMakeLists.txt\n@@ -1,2 +1,3 @@\n-add_test(NAME a COMMAND a)\n+add_test(NAME a COMMAND a)\n+option(ONEBIT_NEW \"x\" OFF)\n", "full"),
        ("a registration plus another file",
         ["CMakeLists.txt", "app/serve.cpp"],
         "+++ b/CMakeLists.txt\n@@ -1,2 +1,3 @@\n add_test(NAME a COMMAND a)\n+add_test(NAME b COMMAND b)\n", "full"),
        ("an unchanged-content entry (rename or mode only)",
         ["CMakeLists.txt"], "+++ b/CMakeLists.txt\n", "full"),
        ("only a comment and a blank line",
         ["CMakeLists.txt"],
         "+++ b/CMakeLists.txt\n@@ -1,1 +1,2 @@\n+# note\n\n", "full"),
        # set_tests_properties creates no target either, so a pure addition of one is configure-only too
        ("set_tests_properties alone",
         ["CMakeLists.txt"],
         "+++ b/CMakeLists.txt\n@@ -1,1 +1,2 @@\n+set_tests_properties(x PROPERTIES RESOURCE_LOCK route_ports)\n", "configure-only"),
        ("removing a registration without adding one (a rewrite we cannot see)",
         ["CMakeLists.txt"],
         "+++ b/CMakeLists.txt\n@@ -1,2 +1,1 @@\n-add_test(NAME a COMMAND a)\n", "full"),
        ("a registration inside a foreach (a template cannot be line-matched)",
         ["CMakeLists.txt"],
         "+++ b/CMakeLists.txt\n@@ -1,1 +1,2 @@\n+add_test(NAME c COMMAND c)\n+foreach(x IN LISTS y)\n", "full"),
        ("no paths at all (a shallow checkout that failed to diff)",
         [], "+++ b/CMakeLists.txt\n+add_test(NAME a COMMAND a)\n", "full"),
    ]
    trig = ["third_party/llama.cpp", "CMakeLists.txt", "cmake/tests-*.cmake", "cmake/hrx.cmake", "app/serve.cpp"]
    one_reg = "+++ b/CMakeLists.txt\n@@ -1,1 +1,2 @@\n+add_test(NAME t COMMAND t)\n"
    tcases = [
        ("the #1491 shape: a registration plus its test file and script (not build inputs)",
         ["CMakeLists.txt", "tests/t_test.sh", "os/v3/puppy/patch-sfs.sh"], one_reg, "configure-only"),
        ("a registration plus a trigger path (app/serve.cpp)", ["CMakeLists.txt", "app/serve.cpp"], one_reg, "full"),
        ("a registration plus a gitlink bump", ["CMakeLists.txt", "third_party/llama.cpp"], one_reg, "full"),
        ("a file under a trigger directory", ["CMakeLists.txt", "cmake/hrx.cmake/x"], one_reg, "full"),
        ("no trigger path at all (a manual run)", ["docs/x.md"], "", "full"),
        # The glob entry: a family file alone is a trigger and a registration file, so a registration there is
        # configure-only. Read as a literal string the entry matched nothing and this came out full.
        ("a registration in a family file only, with the glob trigger", ["cmake/tests-gpu-hold.cmake"],
         "+++ b/cmake/tests-gpu-hold.cmake\n+add_test(NAME gpu_hold_new COMMAND bash x.sh)\n", "configure-only"),
        # `*` stays in one segment, as in GitHub's filter: a file one directory down is not that entry's match.
        ("a path one directory under the glob is not its match", ["cmake/tests-x/y.cmake"], one_reg, "full"),
    ]
    bad = 0
    for name, paths, diff, want in tcases:
        got, why = decide(paths, diff, trig)
        ok = got == want
        print(("ok   " if ok else "FAIL ") + f"{name} -> {got}" + ("" if ok else f" (wanted {want}: {why})"))
        bad += 0 if ok else 1
    real = workflow_triggers()
    ok = bool(real) and "CMakeLists.txt" in real and "cmake/hrx.cmake" in real and "cmake/tests-*.cmake" in real
    print(("ok   " if ok else "FAIL ") + f"hrx-build.yml's paths filter reads as {len(real or [])} trigger(s)")
    bad += 0 if ok else 1
    for name, paths, diff, want in cases:
        got, why = decide(paths, diff)
        ok = got == want
        print(("ok   " if ok else "FAIL ") + f"{name} -> {got}" + ("" if ok else f" (wanted {want}: {why})"))
        bad += 0 if ok else 1
    g = git_arm()
    if g is None:
        print("skip git: no git on PATH, from_git() not exercised")
    else:
        bad += g
    print(f"hrx_build_scope --self-test: {bad} failure(s)")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--diff")
    ap.add_argument("--name-only")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.diff and a.name_only:
        verdict, why = decide(open(a.name_only).read().splitlines(), open(a.diff).read())
    elif a.base:
        verdict, why = from_git(a.base, a.head)
    else:
        ap.error("--base (with --head), or --diff with --name-only, or --self-test")
    print(verdict)
    print(f"reason: {why}", file=sys.stderr)
    return 0 if verdict == "configure-only" else 10


if __name__ == "__main__":
    sys.exit(main())
