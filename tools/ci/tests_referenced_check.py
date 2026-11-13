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
"""Every tests/*.sh is mentioned by something that can run it (engine#1820).

`ci_targets_check.py` asks this question about *targets* -- a test whose executable CI never builds. It cannot
see the other half: a standalone shell test that nothing references at all. Three of those turned up on 2026-11-10
in one night (#1806 registered one that had no runner; #1777 added one nothing referenced; #1809 added one whose
target ci.yml never builds, which is the case ci_targets_check does cover). A test nobody runs cannot fail, so it
reads as green for ever -- the same defect as #1595, one file kind over.

  need = every tests/*.sh
         AND every tests/*_test.py that no RUNNER names and that no other test imports (engine#2018)
         AND every tools/**/*.py that declares a --self-test (engine#1901): a tool that ships one is claiming it can be
         verified without its real inputs, so an unregistered one is an arm nobody runs. For those the corpus is
         RUNNERS only (CMakeLists.txt, cmake/, .github/workflows/) -- a mention in another tool's source is a citation,
         not a runner. That distinction is the count: 2 with the wide corpus, 5 with this one, and 5 is the honest
         number because `tools/mtp_head_ref.py` is only *cited* by two other tools.
  have = every file that MENTIONS its basename, in a place that can run things:
         CMakeLists.txt, cmake/**, scripts/**, tools/**, tests/**, .github/workflows/**
  drift = need - have - allow-list

A mention inside the file itself does not count: a test's own header or usage line naming itself is not a runner,
and counting it would make the whole check vacuous (measured on this tree: it is the one mistake the hand-written
loop made first).

Documentation is deliberately NOT a reference: a sentence in a `.md` that names a test is not something that runs it.

The allow-list is `tools/ci/unreferenced_tests.txt`, one basename per line with an optional `# reason` -- the same
shape as `add_test_gaps.txt`. An entry that no longer needs to be there is reported so the list can shrink, which
keeps it honest rather than accumulating.

usage: tools/ci/tests_referenced_check.py [--check] [--self-test] [--root DIR]
Exit: 0 every test is reachable, 1 an unreferenced one (named), 2 cannot tell (no tests/ directory).
"""

import argparse
import pathlib
import re
import sys

ALLOW = pathlib.Path("tools/ci/unreferenced_tests.txt")
# engine#2018: the existing debt, as a SET baseline rather than 35 allow-list entries with invented reasons -- the same
# shape as tools/ci/add_test_gaps.txt (#1595): the check fails when a NEW unregistered test appears, and reports an entry
# that is no longer needed so the list shrinks.  A per-file allow-list of "not a row yet" would be 35 lines nobody can
# tell apart from "forgotten".
PYBASE = pathlib.Path("tools/ci/unreferenced_py_tests.txt")
SCAN = ("", "cmake", "scripts", "tools", "tests", ".github/workflows")
SUFFIXES = {".txt", ".cmake", ".yml", ".yaml", ".sh", ".py"}


def allow_list(root):
    """{basename: reason} from the allow-list; comments and blanks skipped."""
    path = root / ALLOW
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition("#")
        # Normalised to a basename: `tests/foo.sh` and `foo.sh` mean the same entry. Measured before this: a
        # path-form entry matched no test, so it suppressed nothing AND was reported as "no such test exists any
        # more", two contradicting messages about one line and the format never named.
        out[pathlib.PurePosixPath(name.strip().rstrip("/")).name] = reason.strip()
    return out


def py_baseline(root):
    """{basename} from the python-test baseline; comments and blanks skipped."""
    path = root / PYBASE
    out = set()
    if not path.is_file():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.add(pathlib.Path(line).name)
    return out


def scan_files(root):
    """Every file that can reference a test, as (path, text) -- docs and the allow-list excluded."""
    for sub in SCAN:
        base = root / sub if sub else root
        if not base.is_dir():
            continue
        pattern = "*" if sub else "CMakeLists.txt"
        for path in sorted(base.glob(pattern)):
            if path.is_dir() or path.suffix not in SUFFIXES:
                continue
            if path == root / ALLOW:
                continue  # the allow-list names the file too; it is not a runner
            try:
                yield path, path.read_text(errors="replace")
            except OSError:
                continue


# The FULL path after the variable, not the tail that happens to contain `tests/`: `tools/tests/x.sh` is a
# different file from `tests/x.sh`, and clipping the prefix made this report two files that do exist.
MENTION = re.compile(r"\$\{CMAKE_SOURCE_DIR\}/([A-Za-z0-9_./+-]+\.sh)|(?<![\w/.-])(tests/[A-Za-z0-9_.+-]+\.sh)")


def missing_mentions(root):
    """[(runner, mentioned path)] for a `tests/*.sh` a RUNNER names that is not in the tree.

    The other direction of this file's question, and the one that actually bit: engine#2183 added a row whose
    script the rebase had dropped. The row existed, `referenced()` was satisfied by the mention in CMakeLists.txt,
    `ci_targets_check` follows `$<TARGET_FILE:...>` targets and not plain paths, so **every gate stayed green and
    the row reported "Not Run"** for ever. A row that names a file that is not there is a row that cannot fail.
    """
    out = []
    for sub in RUNNERS:
        base = root / sub
        paths = [base] if base.is_file() else sorted(base.rglob("*")) if base.is_dir() else []
        for path in paths:
            if path.is_dir() or path.suffix not in SUFFIXES:
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            for m in MENTION.finditer(text):
                mentioned = m.group(1) or m.group(2)
                if not (root / mentioned).exists():
                    out.append((str(path.relative_to(root)), mentioned))
    return sorted(set(out))


def referenced(root, need):
    """{name: [files]}, ignoring a file mentioning its own basename."""
    found = {name: [] for name in need}
    for path, text in scan_files(root):
        for name in need:
            if name not in text:
                continue
            if path.name == name:
                continue  # a test naming itself is not a runner
            found[name].append(str(path.relative_to(root)))
    return found


SELFTEST = re.compile(r"--self[-_]?test\b")
RUNNERS = ("CMakeLists.txt", "cmake", ".github/workflows")
# A tool's `--self-test` can also be run from a python test, so THAT gate reads one more directory. The python-test gate must
# not: with "tests" in the shared tuple, `tests/lonely_test.py` mentioning its own name satisfied "some runner names it" and the
# test registered itself (measured on a fixture -- the guard against a test naming itself lives in the other function's path
# shape, not in this haystack). engine#1901's tools half and engine#2129's scripts half are unchanged.
TOOL_RUNNERS = RUNNERS + ("tests",)


def unregistered_selftest_tools(root):
    """tools/**/*.py and scripts/*.sh that declare a self-test and that no RUNNER names (engine#1901, engine#2129).

    `scripts/*.sh` was outside this corpus until a self-test there was found with no row at all: `scripts/pkg-proof.sh
    --self-test` passes on its own and no ctest row or workflow step ran it, so the arm was dead code that every gate
    called green. A self-test is a claim about a script, and the claim needs a runner whichever directory the script is in.
    """
    tools = root / "tools"
    scripts = root / "scripts"
    cand = []
    if tools.is_dir():
        cand += [p for p in tools.rglob("*.py")]
    if scripts.is_dir():
        cand += [p for p in scripts.glob("*.sh")]
    if not cand:
        return []
    runner_text = []
    for sub in TOOL_RUNNERS:
        p = root / sub
        files = [p] if p.is_file() else (sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [])
        for f in files:
            try:
                runner_text.append(f.read_text(errors="replace"))
            except OSError:
                pass
    drift = []
    for tool in sorted(cand):
        try:
            src = tool.read_text(errors="replace")
        except OSError:
            continue
        if not SELFTEST.search(src):
            continue
        if not any(tool.name in text for text in runner_text):
            drift.append(str(tool.relative_to(root)))
    return drift


def unregistered_py_tests(root):
    """tests/*_test.py that no RUNNER names, excluding files another test imports (engine#2018).

    The corpus is RUNNERS, not "any mention": a test importing another test is a citation, the same distinction the
    tools rule makes.  The `*_test.py` name plus "not imported by a sibling" is what separates a test from a helper
    without guessing -- a helper a test imports is called, and a test nothing calls is dead.
    """
    tests = root / "tests"
    if not tests.is_dir():
        return []
    cand = sorted(tests.glob("*_test.py"))
    if not cand:
        return []
    imported = set()
    for p in list(tests.glob("*.py")) + sorted((root / "tools").glob("*.py")) if (root / "tools").is_dir() else list(tests.glob("*.py")):
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for q in cand:
            if p == q:
                continue
            stem = q.stem
            if re.search(r"^\s*(?:import\s+%s\b|from\s+%s\s+import)" % (re.escape(stem), re.escape(stem)),
                         text, re.M):
                imported.add(q.name)
    runner_text = []
    for sub in RUNNERS:
        base = root / sub
        # RUNNERS holds one FILE (CMakeLists.txt) and two directories; testing is_dir() on all three made every
        # cmake registration invisible to this rule, which the self-test arm "a cleared baseline entry is reported"
        # caught -- registering a python test in CMakeLists.txt did not clear its baseline line.
        paths = [base] if base.is_file() else (
            [p for p in sorted(base.rglob("*")) if p.is_file() and p.suffix in {".txt", ".cmake", ".yml", ".yaml", ".py", ".sh"}]
            if base.is_dir() else [])
        for path in paths:
            try:
                runner_text.append(path.read_text(errors="replace"))
            except OSError:
                continue
    out = []
    for q in cand:
        if q.name in imported:
            continue
        if not any(q.name in t for t in runner_text):
            out.append(str(q.relative_to(root)))
    return out


def check(root):
    tests = root / "tests"
    if not tests.is_dir():
        print("tests_referenced: cannot tell: no tests/ directory under %s" % root, file=sys.stderr)
        return 2
    need = sorted(p.name for p in tests.glob("*.sh"))
    allow = allow_list(root)
    found = referenced(root, need)
    drift = [n for n in need if not found[n] and n not in allow]
    gone = [n for n in allow if n not in need]
    idle = [n for n in need if n in allow and found[n]]
    for name in drift:
        print("tests_referenced: %s is referenced by nothing that can run it -- register it (add_test in "
              "CMakeLists.txt or cmake/*.cmake), or list it in %s with the reason it must not be a row" % (name, ALLOW))
    for name in gone:
        print("tests_referenced: %s is in %s but no such test exists any more -- drop the line" % (name, ALLOW))
    for name in idle:
        print("tests_referenced: %s is in %s but it IS referenced now -- drop the line, it suppresses nothing"
              % (name, ALLOW))
    # engine#2018: a SET baseline, not an allow-list -- fail on a NEW one, report one that cleared.
    py_now = {pathlib.Path(n).name: n for n in unregistered_py_tests(root)}
    py_base = py_baseline(root)
    py_drift = sorted(rel for name, rel in py_now.items() if name not in py_base)
    py_cleared = sorted(n for n in py_base if n not in py_now)
    for rel in py_drift:
        print("tests_referenced: %s is a test that no runner names -- add a ctest row (add_python_test in "
              "CMakeLists.txt or cmake/*.cmake), or add it to %s if it cannot be one yet" % (rel, PYBASE))
    for name in py_cleared:
        print("tests_referenced: %s is in %s and no longer needs to be -- drop the line" % (name, PYBASE))
    for runner, mentioned in missing_mentions(root):
        print("tests_referenced: %s names %s, which is not in the tree -- the row exists and can only report "
              "Not Run" % (runner, mentioned))
    tools_drift = unregistered_selftest_tools(root)
    for rel in tools_drift:
        print("tests_referenced: %s declares a self-test and no runner names it -- add a ctest row (add_test in "
              "CMakeLists.txt or cmake/*.cmake) or a workflow step, or the arm is dead code" % rel)
    if drift or gone or idle or py_drift or tools_drift or missing_mentions(root):
        return 1
    print("tests_referenced: all %d test script(s) are referenced%s; %d tool self-test(s) and %d python test(s) are registered" % (
        len(need), "" if not allow else " (%d deliberately not a row: %s)" % (len(allow), ", ".join(sorted(allow))),
        len(_selftest_tools(root)), len(list((root / "tests").glob("*_test.py"))) - len(unregistered_py_tests(root))))
    return 0


def _selftest_tools(root):
    """Every file the corpus above covers that declares a self-test -- counted with the same two directories."""
    out = []
    for sub, pat in (("tools", "**/*.py"), ("scripts", "*.sh")):
        d = root / sub
        if d.is_dir():
            out += [p for p in d.glob(pat) if SELFTEST.search(p.read_text(errors="replace"))]
    return out


def self_test():
    """Synthetic roots: a check that only ever sees a good tree proves nothing."""
    import tempfile

    cases = []

    def root_with(files):
        d = pathlib.Path(tempfile.mkdtemp())
        for rel, body in files.items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)
        return d

    cases_root = {}   # hoisted: the scripts/ arms below need it, and it used to be created after case 1
    # 0a/0b. a self-test in scripts/ -- the corpus that was missing until one was found with no row (engine#2129)
    # (a clean tests/ corpus is part of the fixture: without one the tool answers 2 "cannot tell" before it reaches this)
    base = {"tests/ok.sh": "#!/bin/sh\n", "CMakeLists.txt": "add_test(NAME ok COMMAND bash tests/ok.sh)\n"}
    r = root_with(dict(base, **{"scripts/helper.sh": "#!/bin/sh\ncase \"${1:-}\" in --self-test) exit 0;; esac\n"}))
    cases.append(("a scripts/*.sh self-test with no runner is drift", check(r), 1, "helper.sh"))
    cases_root["a scripts/*.sh self-test with no runner is drift"] = r

    r = root_with(dict(base, **{"scripts/helper.sh": "#!/bin/sh\ncase \"${1:-}\" in --self-test) exit 0;; esac\n",
                                 "cmake/tests-helper.cmake": "add_test(NAME helper_self COMMAND bash scripts/helper.sh --self-test)\n"}))
    cases.append(("...and a registration in cmake/*.cmake clears it", check(r), 0, None))

    # 1. a test nobody mentions -> drift, and it is NAMED
    r = root_with({"tests/lonely.sh": "#!/bin/sh\n", "CMakeLists.txt": "# nothing here\n"})
    cases.append(("an unreferenced test is drift", check(r), 1, "lonely.sh")); cases_root["an unreferenced test is drift"] = r

    # 1b. the inverse: a row that names a script which is not in the tree. engine#2183 shipped one -- the row
    # existed, the mention satisfied the check above, `ci_targets_check` follows $<TARGET_FILE:> and not plain
    # paths, and the row reported "Not Run" with every gate green.
    r = root_with({"tests/ok.sh": "#!/bin/sh\n",
                   "CMakeLists.txt": "add_test(NAME ok COMMAND bash ${CMAKE_SOURCE_DIR}/tests/ok.sh)\n"
                                     "add_test(NAME gone COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gone.sh)\n"})
    cases.append(("a row naming a script that is not there is drift", check(r), 1, "tests/gone.sh"))
    cases_root["a row naming a script that is not there is drift"] = r

    # 1c. and the prefix matters: `tools/tests/x.sh` is a different file from `tests/x.sh`. Clipping the prefix
    #   here reported two existing files as missing on the real tree, which is the false positive this arm pins.
    r = root_with({"tests/ok.sh": "#!/bin/sh\n", "tools/tests/kept.sh": "#!/bin/sh\n",
                   "CMakeLists.txt": "add_test(NAME ok COMMAND bash ${CMAKE_SOURCE_DIR}/tools/tests/kept.sh)\n"
                                     "add_test(NAME ok2 COMMAND bash tests/ok.sh)\n"})
    cases.append(("tools/tests/x.sh is not tests/x.sh", check(r), 0, None))

    # 2. referenced from CMakeLists.txt -> clean
    r = root_with({"tests/ok.sh": "#!/bin/sh\n", "CMakeLists.txt": "add_test(NAME ok COMMAND bash tests/ok.sh)\n"})
    cases.append(("a registration in CMakeLists.txt counts", check(r), 0, None))

    # 3. referenced from an included family file -> clean
    r = root_with({"tests/fam.sh": "#!/bin/sh\n", "CMakeLists.txt": "",
                   "cmake/tests-family.cmake": "add_test(NAME fam COMMAND bash tests/fam.sh)\n"})
    cases.append(("a registration in cmake/*.cmake counts", check(r), 0, None))

    # 4. referenced from a workflow -> clean
    r = root_with({"tests/ci.sh": "#!/bin/sh\n", "CMakeLists.txt": "",
                   ".github/workflows/ci.yml": "run: bash tests/ci.sh\n"})
    cases.append(("a mention in a workflow counts", check(r), 0, None))

    # 4b. a PYTHON test naming only itself is drift even though `tests/` is a run site for TOOL self-tests -- the two gates
    # read different corpora on purpose (measured: sharing one tuple let tests/lonely_test.py register itself)
    r = root_with({"tests/lonely_test.py": "import sys\n# usage: tests/lonely_test.py\nprint('--self-test')\n",
                   "CMakeLists.txt": ""})
    cases.append(("a python test that names only itself is still drift", check(r), 1, "lonely_test.py"))
    cases_root["a python test that names only itself is still drift"] = r

    # 5. a test naming ONLY itself -> drift (the vacuous case)
    r = root_with({"tests/self.sh": "#!/bin/sh\n# usage: tests/self.sh\n", "CMakeLists.txt": ""})
    cases.append(("a test naming itself is not a runner", check(r), 1, "self.sh")); cases_root["a test naming itself is not a runner"] = r

    # 6. allow-listed with a reason -> clean, and the reason is carried
    r = root_with({"tests/keeper.sh": "#!/bin/sh\n", "CMakeLists.txt": "",
                   str(ALLOW): "keeper.sh   # an operator measurement: always exits 0, so a row would be green noise\n"})
    cases.append(("an allow-listed test is not drift", check(r), 0, None))

    # 7. an allow entry for a test that is gone -> a failure, because a list that only gets WARNED about grows
    r = root_with({"tests/one.sh": "#!/bin/sh\n", "CMakeLists.txt": "tests/one.sh\n",
                   str(ALLOW): "gone.sh   # removed long ago\n"})
    cases.append(("an allow entry for a deleted test fails", check(r), 1, "gone.sh")); cases_root["an allow entry for a deleted test fails"] = r

    # 8. an allow entry for a test that IS referenced -> a failure: it suppresses nothing, which is how the list rots
    r = root_with({"tests/one.sh": "#!/bin/sh\n", "CMakeLists.txt": "add_test(NAME one COMMAND bash tests/one.sh)\n",
                   str(ALLOW): "one.sh   # but it is registered\n"})
    cases.append(("an allow entry that suppresses nothing fails", check(r), 1, "one.sh")); cases_root["an allow entry that suppresses nothing fails"] = r

    # 9b. engine#1901: a tool that declares a self-test and is named by no RUNNER is reported, and a mention in
    #     another tool's source does not count as one -- that distinction is the difference between 2 and 5 on main.
    base_ok = {"tests/ok.sh": "#!/bin/sh\n", "CMakeLists.txt": "add_test(NAME ok COMMAND tests/ok.sh)\n"}
    r = root_with(dict(base_ok, **{"tools/quiet.py": "import sys\n# --self-test\n"}))
    cases.append(("a tool self-test no runner names is drift", check(r), 1, "quiet.py"))
    cases_root["a tool self-test no runner names is drift"] = r
    r2 = root_with(dict(base_ok, **{"tools/cited.py": "# --self-test\n",
                                    "tools/other.py": "see tools/cited.py for the arm\n"}))
    cases.append(("a citation in another tool is not a runner", check(r2), 1, "cited.py"))
    cases_root["a citation in another tool is not a runner"] = r2
    r3 = root_with({"tests/ok.sh": "#!/bin/sh\n",
                    "CMakeLists.txt": "add_test(NAME ok COMMAND tests/ok.sh)\n"
                                      "add_test(NAME t COMMAND tools/registered.py --self-test)\n",
                    "tools/registered.py": "# --self-test\n"})
    cases.append(("a registered tool self-test is clean", check(r3), 0, None))

    # 9. the entry may be written either way: path form and basename mean the same line
    r = root_with({"tests/keeper2.sh": "#!/bin/sh\n", "CMakeLists.txt": "",
                   str(ALLOW): "tests/keeper2.sh   # path form is accepted\n"})
    cases.append(("a path-form allow entry is accepted", check(r), 0, None))

    # 9a. engine#2018: a NEW unregistered python test fails, and a test another test IMPORTS does not count
    r = root_with({"tests/one.sh": "#!/bin/sh\n", "CMakeLists.txt": "add_test(NAME one COMMAND bash tests/one.sh)\n",
                   "tests/fresh_test.py": "def f(): pass\n"})
    cases.append(("an unregistered tests/*_test.py is drift", check(r), 1, "fresh_test.py"))
    cases_root["an unregistered tests/*_test.py is drift"] = r

    # 9b. the helper discriminator: imported by a sibling test -> not a test that needs a runner
    r = root_with({"tests/one.sh": "#!/bin/sh\n", "CMakeLists.txt": "add_test(NAME one COMMAND bash tests/one.sh)\n",
                   "tests/helper_test.py": "def h(): pass\n",
                   "tests/caller_test.py": "from helper_test import h\nh()\n",
                   "tools/ci/unreferenced_py_tests.txt": "tests/caller_test.py\n"})
    cases.append(("a test another test imports is not drift", check(r), 0, None))

    # 8. no tests/ at all -> cannot tell, not "clean"
    r = root_with({"CMakeLists.txt": ""})
    cases.append(("no tests/ directory is 'cannot tell'", check(r), 2, None))

    failures = 0
    import contextlib
    import io
    for what, got, want, mentions in cases:
        if got != want:
            print("FAIL %s: rc %s, wanted %s" % (what, got, want))
            failures += 1
            continue
        if mentions:
            # re-run capturing the output: the verdict must NAME the offending test, or a reader cannot act on it
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                if what.startswith("a stale"):
                    check(cases_root[what])
                else:
                    check(cases_root[what])
            if mentions not in buf.getvalue():
                print("FAIL %s: rc was right but the output does not name %s" % (what, mentions))
                failures += 1
                continue
        print("ok   %s" % what)
    print("tests_referenced self-test: %d failure(s)" % failures)
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".", help="tree to inspect (default: the current directory)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    return check(pathlib.Path(a.root).resolve())


if __name__ == "__main__":
    sys.exit(main())
