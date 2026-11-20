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
"""engine#2836: what the fork's `test-hrx-*` population is at the pin, and how much of it can run here.

The issue carried "43 rows" and that number is wrong at the pinned sha: measured twice -- by a six-line text scan and
by a structural parse that tracks `if()` nesting -- the fork registers **38** `add_test(NAME test-hrx-...)` rows, of
which **17** carry `SKIP_RETURN_CODE 77` beside them (they skip cleanly with no device) and **21** do not. The 21 are
the rows a device-free build would actually execute, and no engine job executes any of them.

This reads a checkout's `tests/CMakeLists.txt` (`--root <dir>` or a pinned `gh api` dump, `--file <path>`), because the
number in a CI row's message has to be the number the row will see.

usage: fork_hrx_rows.py (--root DIR | --file CMakeLists.txt) [--expect N] [--self-test]
exit:  0 all rows agree with --expect (or no --expect), 1 disagreement, 2 cannot tell
"""
import argparse
import os
import re
import sys

ADD = re.compile(r'add_test\(NAME (test-hrx-[a-z0-9-]+)')
PROP = r'set_tests_properties\({name}[^)]*?SKIP_RETURN_CODE\s+77'
BLOCK = 8            # a row's properties follow it within a few lines in this file
DEPTH = re.compile(r'^(if|foreach|while|function|macro)\s*\(', re.I)
POP = re.compile(r'^(endif|endforeach|endwhile|endfunction|endmacro)\b', re.I)


def rows(text):
    """[(name, skips_cleanly)] for every `test-hrx-*` row in the file, in file order."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        m = ADD.search(line)
        if not m:
            continue
        name = m.group(1)
        block = "\n".join(lines[i:i + BLOCK])
        out.append((name, bool(re.search(PROP.format(name=re.escape(name)), block, re.S))))
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="a fork checkout (reads tests/CMakeLists.txt under it)")
    ap.add_argument("--file", help="the CMakeLists.txt itself")
    ap.add_argument("--expect", type=int, help="the row count a caller (a CI row, an issue) claims")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv[1:])
    if a.self_test:
        return self_test()
    path = a.file or (os.path.join(a.root, "tests", "CMakeLists.txt") if a.root else None)
    if not path:
        print("usage: fork_hrx_rows.py (--root DIR | --file CMakeLists.txt)", file=sys.stderr)
        return 2
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            found = rows(f.read())
    except OSError as e:
        print(f"cannot tell: {path}: {e}", file=sys.stderr)
        return 2
    if not found:
        # An empty result is not "no rows": the file parsed, so either the fork moved them or the pattern drifted.
        print(f"cannot tell: no test-hrx-* row matched in {path}; the registration shape may have changed",
              file=sys.stderr)
        return 2
    skip = [n for n, s in found if s]
    run = [n for n, s in found if not s]
    print(f"{path}: {len(found)} test-hrx-* row(s): {len(skip)} with SKIP_RETURN_CODE 77, {len(run)} without")
    print("run on a device-free build: " + ", ".join(run))
    if a.expect is not None and a.expect != len(found):
        print(f"expected {a.expect} row(s), measured {len(found)}: a count in an issue or a CI row is stale",
              file=sys.stderr)
        return 1
    return 0


def self_test():
    fixture = """
if (target ggml-hrx)
    add_test(NAME test-hrx-plain COMMAND test-hrx-plain)
    add_test(NAME test-hrx-skips COMMAND test-hrx-skips)
    set_tests_properties(test-hrx-skips PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME test-other COMMAND test-other)
endif()
"""
    found = rows(fixture)
    checks = [
        ("only test-hrx rows are counted", [n for n, _ in found], ["test-hrx-plain", "test-hrx-skips"]),
        ("a 77 beside the row marks it skippable", [s for _, s in found], [False, True]),
        ("a 77 further away does not leak to the next row", rows(
            "add_test(NAME test-hrx-a COMMAND a)\nset_tests_properties(test-hrx-a PROPERTIES SKIP_RETURN_CODE 77)\n"
            "add_test(NAME test-hrx-b COMMAND b)\n"), [("test-hrx-a", True), ("test-hrx-b", False)]),
        ("an empty file yields no rows, not a claim of zero", rows(""), []),
    ]
    bad = 0
    for name, got, want in checks:
        if got == want:
            print(f"ok   {name}")
        else:
            print(f"FAIL {name}: want {want!r} got {got!r}")
            bad += 1
    print("self-test ok" if not bad else f"{bad} failed")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
