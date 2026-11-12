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
# Copyright 2026 The 1bit-MONSTER Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""engine#1864: a Python test must be registered AFTER find_package(Python3).

`add_test(... COMMAND ${Python3_EXECUTABLE} script.py ...)` before the find_package expands the interpreter to the empty
string, so `CTestTestfile.cmake` holds the script as the program. ctest then runs it through its exec bit -- and a
mode-only rewrite (tools/ci/hrx_build_scope.py, 100755 -> 100644) turns the row into BAD_COMMAND with nothing visibly
wrong in the source. Nine registrations were in that state.

The rule is a file-order rule, so it is checked here on the file, with no configure and no build: every
`${Python3_EXECUTABLE}` use in CMakeLists.txt must come after the first `find_package(Python3`. `add_python_test(...)`
satisfies it by construction, which is the other half of the fix.
"""

import argparse
import pathlib
import re
import sys

USE = re.compile(r"\$\{Python3_EXECUTABLE\}")
FIND = re.compile(r"^\s*find_package\(\s*Python3\b", re.MULTILINE)


def _strip_comments(text: str) -> str:
    """CMake comments out, line numbers preserved: the fix's own comment names the variable it forbids."""
    return "\n".join(re.sub(r"#.*$", "", line) for line in text.split("\n"))


def offenders(text: str):
    """Line numbers of Python3_EXECUTABLE uses that precede the first find_package(Python3)."""
    text = _strip_comments(text)
    finds = list(FIND.finditer(text))
    if not finds:
        # No find_package at all: every use is unbound. Report them all rather than pass silently.
        first = None
    else:
        first = finds[0].start()
    out = []
    for m in USE.finditer(text):
        if first is None or m.start() < first:
            out.append(text.count("\n", 0, m.start()) + 1)
    return out


# engine#1979: a registration file is INCLUDED from a CMakeLists, so whether its `${Python3_EXECUTABLE}` is bound depends on
# where the `include()` sits, not on anything inside the file -- recursing into the `.cmake` file would check the wrong
# thing. The rule for those is the include's position: an `include(<f>.cmake)` that pulls in `${Python3_EXECUTABLE}` or
# `add_python_test(` must come after the first `find_package(Python3`. The argument is the first word, so the
# CMake-legal trailing keywords (`OPTIONAL`, `NO_POLICY_SCOPE`, `RESULT_VARIABLE <v>`) do not hide an include.
INCLUDE = re.compile(r"^\s*include\(\s*([^)\s]+)[^)]*\)", re.MULTILINE)


def include_offenders(root: pathlib.Path, cmake: pathlib.Path, text: str):
    """Line numbers of includes that bind Python before find_package(Python3) does.

    Covered: an include whose argument is a path to a file that exists. An include of a variable or a module
    (`include(${_file})`, `include(<name>)`) is skipped, because there is no file to read here -- so a registration
    file reached through a `file(GLOB ...)` loop is outside this rule; its position is the loop's.
    """
    # One text for every offset: stripping comments shortens it, so the find_package's offset taken from the raw
    # text and an include's offset taken from the stripped text are not comparable, and a comment above a correct
    # find_package made the include after it an "offender" at the wrong line (review of engine#2012).
    text = _strip_comments(text)
    finds = list(FIND.finditer(text))
    first = finds[0].start() if finds else None
    out = []
    for m in INCLUDE.finditer(text):
        target = m.group(1)
        if target.startswith("$") or target.startswith("<"):
            continue  # a variable or a module: its contents are not a file we can read here
        # `include()` is resolved against the including file's directory, and (for a top-level CMakeLists in this repo)
        # against the source root as well: try both rather than assume which one the tree uses.
        path = (cmake.parent / target).resolve()
        if not path.is_file():
            path = (root / target).resolve()
        if not path.is_file():
            continue
        body = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        if "${Python3_EXECUTABLE}" not in body and "add_python_test(" not in body:
            continue
        # m.start(1) is the include's own argument: the pattern's leading `\s*` also swallows the blank lines above it
        # (which is what a stripped comment leaves), so m.start() can sit lines earlier than the include.
        if first is None or m.start(1) < first:
            out.append(text.count("\n", 0, m.start(1)) + 1)
    return out


def check(root: pathlib.Path) -> int:
    bad = 0
    for cmake in sorted(root.rglob("CMakeLists.txt")):
        if any(part in {"build", ".git", "third_party"} for part in cmake.parts):
            continue
        text = cmake.read_text(encoding="utf-8", errors="replace")
        lines = offenders(text)
        for ln in lines:
            print(f"FAIL {cmake.relative_to(root)}:{ln}: ${{Python3_EXECUTABLE}} used before find_package(Python3)")
            bad += 1
        for ln in include_offenders(root, cmake, text):
            print(f"FAIL {cmake.relative_to(root)}:{ln}: includes a registration file before find_package(Python3), "
                  "so its ${Python3_EXECUTABLE} is empty there")
            bad += 1
    if bad:
        print(f"FAIL {bad} registration(s) run a script through its exec bit instead of through the interpreter")
        return 1
    print("ok: every ${Python3_EXECUTABLE} use, and every include that binds it, follows find_package(Python3)")
    return 0


def self_test() -> int:
    after = "find_package(Python3 COMPONENTS Interpreter)\nadd_test(NAME a COMMAND ${Python3_EXECUTABLE} s.py)\n"
    before = "add_test(NAME a COMMAND ${Python3_EXECUTABLE} s.py)\nfind_package(Python3 COMPONENTS Interpreter)\n"
    none = "add_test(NAME a COMMAND ${Python3_EXECUTABLE} s.py)\n"
    commented = "# ${Python3_EXECUTABLE} in a comment is prose, not a use\nadd_test(NAME a COMMAND ${Python3_EXECUTABLE} s.py)\n"
    helper = "find_package(Python3 COMPONENTS Interpreter)\nfunction(add_python_test n)\n  add_test(NAME ${n} COMMAND ${Python3_EXECUTABLE} ${ARGN})\nendfunction()\nadd_python_test(a s.py)\n"
    # 4b. engine#1979: an include that pulls in a registration file is bound only if it comes after the find_package
    import tempfile
    inc_tmp = tempfile.TemporaryDirectory()      # removed when the self-test returns
    inc_root = pathlib.Path(inc_tmp.name)
    (inc_root / "cmake").mkdir()
    (inc_root / "cmake" / "reg.cmake").write_text("add_python_test(x tools/x.py --self-test)\n")
    (inc_root / "later").mkdir()
    (inc_root / "later" / "CMakeLists.txt").write_text(
        "find_package(Python3 COMPONENTS Interpreter)\ninclude(cmake/reg.cmake)\n")
    (inc_root / "earlier").mkdir()
    (inc_root / "earlier" / "CMakeLists.txt").write_text(
        "include(cmake/reg.cmake)\nfind_package(Python3 COMPONENTS Interpreter)\n")
    cases = [
        (after, [], "a use after the find_package is fine"),
        (before, [1], "a use before it is an offender"),
        (none, [1], "with no find_package every use is unbound"),
        (helper, [], "the helper's own use comes after the find_package"),
        (commented, [2], "a comment mentioning the variable is not a use"),
    ]
    bad = 0
    # the include cases run on real files, through check(), because the rule reads the included file
    inc_ok = check(inc_root)
    print(("ok   " if inc_ok == 1 else "FAIL ") + "an include of a registration file before find_package is caught, "
          "and the same include after it is not (got rc %d, want 1 because the 'earlier' tree is a counter-example)" % inc_ok)
    if inc_ok != 1:
        bad += 1
    # ... and each tree by itself, with the line: a rule that flagged both trees, or the wrong one, also gives rc 1.
    find = "find_package(Python3 COMPONENTS Interpreter)\n"
    inc = "include(cmake/reg.cmake)\n"
    note = "# a comment long enough to move every offset after it: the interpreter is found below, before the include\n"
    inc_cases = [
        (inc + find, [1], "an include before the find_package is the offender, at its own line"),
        (find + inc, [], "an include after the find_package is not"),
        (note + find + inc, [], "a comment above a correct find_package does not make the include an offender"),
        (note + note + inc + find, [3], "a comment above a wrong order still reports the include's own line"),
        (find + "# " + inc, [], "a commented-out include is not an include"),
        (inc, [1], "with no find_package the include is unbound"),
        (find + "include(${_registration_file})\n", [], "an include of a variable is outside the rule"),
        ("include(cmake/reg.cmake OPTIONAL)\n" + find, [1], "an OPTIONAL include before the find_package is still an offender"),
        ("include(cmake/reg.cmake RESULT_VARIABLE _r NO_POLICY_SCOPE)\n" + find, [1],
         "an include with RESULT_VARIABLE and NO_POLICY_SCOPE before the find_package is still an offender"),
        (find + "include(cmake/reg.cmake OPTIONAL)\n", [], "an OPTIONAL include after the find_package is not"),
    ]
    for text, want, why in inc_cases:
        got = include_offenders(inc_root, inc_root / "CMakeLists.txt", text)
        if got != want:
            print(f"FAIL self-test: {why}: want {want}, got {got}")
            bad += 1
    print(("ok   " if not bad else "FAIL ") + "%d include trees checked one by one" % len(inc_cases))
    for text, want, why in cases:
        got = offenders(text)
        if got != want:
            print(f"FAIL self-test: {why}: want {want}, got {got}")
            bad += 1
    print("self-test ok" if not bad else f"self-test FAILED ({bad})")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.check:
        return check(pathlib.Path(a.root).resolve())
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
