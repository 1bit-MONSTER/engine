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


def unregistered_selftest_tools(root):
    """tools/**/*.py that declare a self-test and that no RUNNER names (engine#1901)."""
    tools = root / "tools"
    if not tools.is_dir():
        return []
    runner_text = []
    for sub in RUNNERS:
        p = root / sub
        files = [p] if p.is_file() else (sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [])
        for f in files:
            try:
                runner_text.append(f.read_text(errors="replace"))
            except OSError:
                pass
    drift = []
    for tool in sorted(tools.rglob("*.py")):
        try:
            src = tool.read_text(errors="replace")
        except OSError:
            continue
        if not SELFTEST.search(src):
            continue
        if not any(tool.name in text for text in runner_text):
            drift.append(str(tool.relative_to(root)))
    return drift


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
    tools_drift = unregistered_selftest_tools(root)
    for rel in tools_drift:
        print("tests_referenced: %s declares a self-test and no runner names it -- add a ctest row (add_test in "
              "CMakeLists.txt or cmake/*.cmake) or a workflow step, or the arm is dead code" % rel)
    if drift or gone or idle or tools_drift:
        return 1
    print("tests_referenced: all %d test script(s) are referenced%s; %d tool self-test(s) are registered" % (
        len(need), "" if not allow else " (%d deliberately not a row: %s)" % (len(allow), ", ".join(sorted(allow))),
        len(_selftest_tools(root))))
    return 0


def _selftest_tools(root):
    tools = root / "tools"
    return [p for p in tools.rglob("*.py") if SELFTEST.search(p.read_text(errors="replace"))] if tools.is_dir() else []


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

    # 1. a test nobody mentions -> drift, and it is NAMED
    r = root_with({"tests/lonely.sh": "#!/bin/sh\n", "CMakeLists.txt": "# nothing here\n"})
    cases_root = {}
    cases.append(("an unreferenced test is drift", check(r), 1, "lonely.sh")); cases_root["an unreferenced test is drift"] = r

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
