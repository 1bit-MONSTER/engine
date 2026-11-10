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
    if bad:
        print(f"FAIL {bad} registration(s) run a script through its exec bit instead of through the interpreter")
        return 1
    print("ok: every ${Python3_EXECUTABLE} use follows find_package(Python3)")
    return 0


def self_test() -> int:
    after = "find_package(Python3 COMPONENTS Interpreter)\nadd_test(NAME a COMMAND ${Python3_EXECUTABLE} s.py)\n"
    before = "add_test(NAME a COMMAND ${Python3_EXECUTABLE} s.py)\nfind_package(Python3 COMPONENTS Interpreter)\n"
    none = "add_test(NAME a COMMAND ${Python3_EXECUTABLE} s.py)\n"
    commented = "# ${Python3_EXECUTABLE} in a comment is prose, not a use\nadd_test(NAME a COMMAND ${Python3_EXECUTABLE} s.py)\n"
    helper = "find_package(Python3 COMPONENTS Interpreter)\nfunction(add_python_test n)\n  add_test(NAME ${n} COMMAND ${Python3_EXECUTABLE} ${ARGN})\nendfunction()\nadd_python_test(a s.py)\n"
    cases = [
        (after, [], "a use after the find_package is fine"),
        (before, [1], "a use before it is an offender"),
        (none, [1], "with no find_package every use is unbound"),
        (helper, [], "the helper's own use comes after the find_package"),
        (commented, [2], "a comment mentioning the variable is not a use"),
    ]
    bad = 0
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
