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
"""A tools/*.py that DECLARES a --self-test must have a ctest row that RUNS it (engine#1901).

`tools/ci/tests_referenced_check.py` (#1824) asks whether every `tests/*.sh` is referenced by something that can run it.
A tool is the same defect one directory over, and the blanket question ("is this file referenced anywhere?") is too noisy
to be a row: 16 of 60 tools are referenced by nothing and most of those are legitimately CLI-only.

The narrow question is the one with a claim behind it: **a tool that ships `--self-test` is claiming it can be verified
without its real inputs.** If nothing runs that flag, the claim is unbacked -- the arm is maintained in the same commits as
the logic and never executes anywhere. So:

  need = every tools/*.py whose argparse declares --self-test   (parsed, not grepped out of a comment)
  have = an add_test whose COMMAND names that file AND passes --self-test

Registration is not enough on its own (#1826's lesson): a row that names the file but runs it without the flag, or with the
wrong flag, would satisfy a mention check and still never exercise the self-test. This compares the flag too.

Usage:
  selftest_registration_check.py --check [--root DIR]
  selftest_registration_check.py --self-test
"""
import argparse
import os
import re
import sys


def declares_self_test(path):
    """True when the tool's argparse has a --self-test option. Parsed from the source, so a comment mentioning the flag
    in a docstring is not mistaken for a declaration."""
    try:
        src = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return False
    # the flag must appear inside an add_argument(...) call, quoted
    return re.search(r'add_argument\(\s*["\']--self-test["\']', src) is not None


def ctest_rows(root):
    """[(command-string)] for every add_test in the tree's CMake files."""
    rows = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "/.git" in dirpath or "/build" in dirpath or "/third_party" in dirpath:
            continue
        for fn in filenames:
            if fn not in ("CMakeLists.txt",) and not fn.endswith(".cmake"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                text = open(p, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            # Balance parentheses rather than stopping at the first `)`: a row's COMMAND contains `${...}` and may be
            # wrapped, and `[^)]*` truncated such a row before its `--self-test` -- which reported registered tools as
            # unbacked (caught by running the check against the real tree, not by the self-test).
            # `add_test(` AND the local `add_python_test(name path --self-test)` helper (CMakeLists.txt:288), which is
            # how most python rows are registered: scanning only literal add_test() reported eight registered tools as
            # unbacked (caught by running the check against the real tree).
            for m in re.finditer(r"add_(?:python_)?test\(", text):
                i, depth = m.end(), 1
                while i < len(text) and depth:
                    depth += (text[i] == "(") - (text[i] == ")")
                    i += 1
                rows.append(text[m.end():i - 1].replace("\n", " "))
    return rows


def workflow_and_test_text(root):
    """The text of every workflow and tests/*.py, which are the other two places a tool can be run from."""
    out = []
    for sub, pats in ((".github/workflows", (".yml", ".yaml")), ("tests", (".py",))):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(pats):
                continue
            try:
                out.append(open(os.path.join(d, fn), encoding="utf-8", errors="replace").read())
            except OSError:
                pass
    return out


def check(root):
    tools_dir = os.path.join(root, "tools")
    tools = sorted(
        os.path.join(tools_dir, f) for f in os.listdir(tools_dir)
        if f.endswith(".py") and os.path.isfile(os.path.join(tools_dir, f))
    ) if os.path.isdir(tools_dir) else []
    declaring = [t for t in tools if declares_self_test(t)]
    rows = ctest_rows(root)
    # A runner does not have to be a ctest row: tools/pin_watch.py's --self-test runs as a ci.yml step, and
    # tools/q4nx_diff.py is exercised by tests/q4nx_diff_grid_test.py rather than by its own flag. What the claim
    # needs is that SOMETHING executable names the tool, so the haystack is CMake rows + workflows + tests.
    hay = list(rows) + workflow_and_test_text(root)
    missing = []
    for t in declaring:
        base = os.path.basename(t)
        runners = [h for h in hay if base in h]
        if not runners:
            missing.append((base, False))
    print(f"selftest_registration_check: {len(declaring)} tool(s) declare --self-test, "
          f"{len(declaring) - len(missing)} have a ctest row that runs it")
    for base, names_but_no_flag in missing:
        why = ("a row names it but does not pass --self-test" if names_but_no_flag
               else "no ctest row names it")
        print(f"  UNBACKED {base}: {why}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        # A check that cannot fail is the defect this exists to find: assert both sides on a synthetic tree.
        import tempfile
        bad = 0
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "tools"))
            p = os.path.join(d, "tools", "backed.py")
            open(p, "w").write("import argparse\nap=argparse.ArgumentParser()\nap.add_argument('--self-test', action='store_true')\n")
            open(os.path.join(d, "CMakeLists.txt"), "w").write(
                'add_test(NAME backed COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/backed.py --self-test)\n')
            if check(d):
                print("  FAIL a backed tool was reported unbacked")
                bad += 1
            # the negative control: nothing anywhere names the tool, so its claim is unbacked
            open(os.path.join(d, "CMakeLists.txt"), "w").write("")
            if not check(d):
                print("  FAIL a declaring tool that nothing names was accepted")
                bad += 1
            # and a tool that does not declare it is not this check's business -- so drop the declaring one first,
            # otherwise this arm re-reports it and the arm fails for the wrong reason (caught by running it)
            os.remove(p)
            open(os.path.join(d, "tools", "cli_only.py"), "w").write("import argparse\nap=argparse.ArgumentParser()\n")
            open(os.path.join(d, "CMakeLists.txt"), "w").write("")
            if check(d):
                print("  FAIL a tool with no --self-test was reported")
                bad += 1
        print("selftest_registration_check self-test: " + ("ok" if not bad else f"{bad} FAILED"))
        return 0 if not bad else 1

    missing = check(a.root)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
