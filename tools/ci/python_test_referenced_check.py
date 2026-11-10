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
"""A `tests/*_test.py` with no ctest row is a file nothing runs (engine#2018).

The sweep covers three file kinds and this is the third:

- `tests/*.sh`   -> tools/ci/tests_referenced_check.py (#1824)
- `tools/*.py` with `--self-test` -> tools/ci/selftest_registration_check.py (#1911)
- `tests/*_test.py` -> here: a row is `add_test(... COMMAND ${Python3_EXECUTABLE} <path> ...)`, or the file is listed
  with a reason in tools/ci/unreferenced_python_tests.txt.

Both directions are checked, because an excuse that has outlived its reason is the same defect one step later: a name
in the allow file that IS registered now passes as "deliberately unrun", and the next reader believes it.

usage: tools/ci/python_test_referenced_check.py [--check | --self-test] [--root DIR]
"""

import argparse
import glob
import os
import re
import sys

ALLOW = "tools/ci/unreferenced_python_tests.txt"


def cmake_text(root):
    """Every CMake file a row can live in: the root list, the cmake/ registration files, and one level of subproject."""
    out = []
    for pattern in ("CMakeLists.txt", "cmake/*.cmake", "*/CMakeLists.txt"):
        for p in sorted(glob.glob(os.path.join(root, pattern))):
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    out.append((p, f.read()))
            except OSError:
                continue
    return out


def python_tests(root):
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "tests/*_test.py")))


def excuses(root):
    """name -> reason, from the allow file. A line without a reason is refused: an unexplained excuse is a silencer."""
    path = os.path.join(root, ALLOW)
    out = {}
    if not os.path.exists(path):
        return out
    bad = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, sep, reason = line.partition(":")
            if not sep or not reason.strip():
                bad.append((n, line))
                continue
            out[name.strip()] = reason.strip()
    if bad:
        raise SystemExit("python_test_referenced_check: %s line(s) with no reason: %s"
                         % (len(bad), ", ".join(str(b[0]) for b in bad)))
    return out


def check(root, quiet=False):
    tests = python_tests(root)
    text = "\n".join(t for _, t in cmake_text(root))
    allow = excuses(root)
    unrun, stale = [], []
    for t in tests:
        named = t in text
        if not named and t not in allow:
            unrun.append(t)
        elif named and t in allow:
            stale.append(t)
    for t in unrun:
        print("  UNRUN %s: no add_test row names it, and %s does not excuse it" % (t, ALLOW), file=sys.stderr)
    for t in stale:
        print("  STALE %s: excused in %s but a row DOES name it -- drop the excuse" % (t, ALLOW), file=sys.stderr)
    if not quiet:
        print("python_test_referenced_check: %d test(s), %d run by a row, %d excused, %d unrun, %d stale"
              % (len(tests), len(tests) - len(unrun) - len(allow) + len(stale), len(allow) - len(stale), len(unrun), len(stale)))
    return 0 if not unrun and not stale else 1


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def self_test():
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as d:
        write(os.path.join(d, "CMakeLists.txt"),
              "add_test(NAME a COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/a_test.py)\n")
        write(os.path.join(d, "tests/a_test.py"), "")
        write(os.path.join(d, "tests/b_test.py"), "")
        write(os.path.join(d, "tests/c_test.py"), "")
        # b is unrun with no excuse -> must fail; c is excused -> must pass; nothing stale yet
        r = check(d, quiet=True)
        if r == 0:
            print("  FAIL an unrun test with no excuse passed the check", file=sys.stderr)
            ok = False
        else:
            print("  ok   an unrun test with no excuse fails the check")
        write(os.path.join(d, ALLOW), "# name: reason\nb_test.py: run by hand, needs a 27B pack on disk\n"
              "c_test.py: needs the pinned llama.cpp checkout\n")
        if check(d, quiet=True) != 0:
            print("  FAIL an excused test did not clear the check", file=sys.stderr)
            ok = False
        else:
            print("  ok   an excused test clears the check")
        # the other direction: an excuse for a file that IS registered is stale
        write(os.path.join(d, ALLOW), "a_test.py: nobody needs this\nb_test.py: needs a pack\n"
              "c_test.py: needs the pinned llama.cpp checkout\n")
        if check(d, quiet=True) == 0:
            print("  FAIL an excuse for a registered test did not fail as stale", file=sys.stderr)
            ok = False
        else:
            print("  ok   an excuse that outlived its reason fails as stale")
        # an excuse with no reason is refused rather than silently silencing a test
        write(os.path.join(d, ALLOW), "b_test.py:\n")
        try:
            excuses(d)
            print("  FAIL a reasonless excuse was accepted", file=sys.stderr)
            ok = False
        except SystemExit:
            print("  ok   a reasonless excuse is refused")
    print("python_test_referenced_check self-test: %s" % ("ok" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.check:
        return check(a.root)
    print(__doc__.strip().splitlines()[-1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
