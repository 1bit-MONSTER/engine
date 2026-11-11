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
"""Every submodule must be checked out at the commit the tree pins (engine#2129).

A build compiles whatever is in the working tree. Every check in this directory reads the tree it is handed, so
none of them can see that a submodule is sitting on a different commit than the one the superproject pins - a
build that compiles an un-pinned dependency passes every gate we have.

Measured on 2026-11-11, which is why this exists: the shared `~/1bit-engine/third_party/llama.cpp` was checked
out 69 commits off the pin (`c16424f` against `7a1f2b81`), and, on the files the engine actually compiles
(`common/jinja`, via `CMakeLists.txt:147`), **behind** the pin by `f08bd76f6 "jinja: accept \\xNN escapes in the
lexer"`. Jinja decides prompt rendering, so an engine built there renders a `\\xNN`-bearing template differently -
not a build failure, but a difference that reads as a model bug.

`git submodule status` marks exactly that in one character, and this is the check for it:

    ' '  the checkout IS the pinned commit                      -> pass
    '+'  the checkout is a DIFFERENT commit (the #2129 shape)    -> fail
    '-'  not initialised (nothing to compare; the build may be
         missing the source or using a system copy)             -> fail with --require-initialised
    'U'  merge conflicts in the submodule                       -> fail

Usage:
  tools/ci/submodule_pin_check.py --check [--root .] [--require-initialised]
  tools/ci/submodule_pin_check.py --self-test
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

# `git submodule status --recursive` prints one line per submodule:
#   <flag><sha> <path> (<describe>)
# The flag is a single character from " +-U". The sha is always 40 hex.
STATUS = re.compile(r"^([ +\-U])([0-9a-f]{40}) (\S+)(?: \((.*)\))?$")


def parse_status(text: str) -> list[tuple[str, str, str, str]]:
    """[(flag, sha, path, describe)] - the parser is the testable part, so it takes text, not a repo."""
    rows = []
    for line in text.splitlines():
        m = STATUS.match(line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(3), m.group(4) or ""))
        elif line.strip():
            # An unparseable line is not a submodule we can judge; keep it so the caller can report it rather
            # than silently skipping a line that may be the one that matters.
            rows.append(("?", "", line.strip(), ""))
    return rows


def verdict(rows: list[tuple[str, str, str, str]], require_initialised: bool) -> tuple[int, list[str]]:
    """(rc, messages). rc 0 pass, 1 a submodule is not at its pin, 2 nothing readable."""
    if not rows:
        return 2, ["submodule_pin: no submodules reported - cannot tell, and that is not a pass"]
    out: list[str] = []
    bad = unknown = uninit = 0
    for flag, sha, path, describe in rows:
        if flag == " ":
            continue
        if flag == "?":
            unknown += 1
            out.append(f"submodule_pin: cannot parse this status line, so not judged: {path}")
        elif flag == "+":
            bad += 1
            out.append(
                f"submodule_pin: {path} is checked out at {sha[:12]}, which is NOT the commit this tree pins, "
                f"so the build compiles an un-pinned dependency. Repair, in this order: if the PIN is right, "
                f"`git submodule update --init -- {path}`; if the checkout is right, the gitlink is stale and the "
                f"pin must move (git add {path}) - the two must agree, and nothing else reads this"
            )
        elif flag == "-":
            if require_initialised:
                bad += 1
                out.append(
                    f"submodule_pin: {path} is not initialised (pinned at {sha[:12]}) - a build here compiles "
                    f"whatever it finds instead. Repair: git submodule update --init -- {path}"
                )
            else:
                uninit += 1
                out.append(f"submodule_pin: {path} is not initialised (pinned at {sha[:12]}) - not compared")
        elif flag == "U":
            bad += 1
            out.append(f"submodule_pin: {path} has merge conflicts in its own work tree")
    if bad:
        return 1, out
    if unknown and not (len(rows) - unknown):
        return 2, out
    out.append(
        f"submodule_pin: {len(rows) - bad - unknown - uninit} submodule(s) at their pinned commit"
        + (f", {uninit} not initialised" if uninit else "")
        + (f", {unknown} unparseable" if unknown else "")
    )
    return 0, out


def run_git(root: pathlib.Path) -> tuple[str, int]:
    try:
        p = subprocess.run(
            ["git", "-C", str(root), "submodule", "status", "--recursive"],
            capture_output=True, text=True, check=False,
        )
    except OSError as e:
        return f"cannot run git: {e}", 2
    if p.returncode != 0:
        return (p.stderr or "").strip() or f"git submodule status exited {p.returncode}", 2
    return p.stdout, 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument(
        "--require-initialised",
        action="store_true",
        help="treat a submodule that is not checked out as a failure (a release build should use this)",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    text, rc = run_git(pathlib.Path(args.root).resolve())
    if rc != 0:
        print(f"submodule_pin: cannot tell: {text}")
        return 2
    rc, messages = verdict(parse_status(text), args.require_initialised)
    for m in messages:
        print(m)
    return rc


def self_test() -> int:
    bad = 0

    def check(ok: bool, what: str) -> None:
        nonlocal bad
        print(("  ok   " if ok else "  FAIL ") + what)
        if not ok:
            bad += 1

    A = "a" * 40
    B = "b" * 40
    at_pin = f" {A} third_party/llama.cpp (v1-3-g{ A[:7] })\n"
    ahead = f"+{B} third_party/llama.cpp (v1-9-g{ B[:7] })\n"
    uninit = f"-{A} third_party/llama.cpp\n"
    conflict = f"U{A} third_party/llama.cpp\n"
    unparseable = "this is not a status line\n"

    check(len(parse_status(at_pin)) == 1, "a ' ' line parses")
    check(parse_status(at_pin)[0][0] == " ", "and its flag is ' '")
    check(parse_status(at_pin)[0][2] == "third_party/llama.cpp", "and its path is the submodule path")
    check(parse_status(ahead)[0][0] == "+", "a '+' line parses with the + flag (the engine#2129 shape)")
    check(parse_status(uninit)[0][0] == "-", "a '-' line parses with the - flag")
    check(parse_status(at_pin + ahead)[1][1] == B, "two lines parse in order")

    check(verdict(parse_status(at_pin), False) == (0, verdict(parse_status(at_pin), False)[1]) and
          verdict(parse_status(at_pin), False)[0] == 0, "at the pin -> pass")
    check(verdict(parse_status(ahead), False)[0] == 1, "a different commit -> fail")
    check(verdict(parse_status(conflict), False)[0] == 1, "merge conflicts -> fail")
    check(verdict(parse_status(uninit), False)[0] == 0, "not initialised -> pass by default (not compared)")
    check(verdict(parse_status(uninit), True)[0] == 1, "... and FAIL with --require-initialised")
    check(verdict([], False)[0] == 2, "no submodules read is 'cannot tell', not a pass")
    check(verdict(parse_status(unparseable), False)[0] == 2, "only unparseable lines is also 'cannot tell'")
    check(
        any("NOT the commit this tree pins" in m for m in verdict(parse_status(ahead), False)[1]),
        "the failure names what is wrong",
    )
    check(
        any("not initialised" in m for m in verdict(parse_status(uninit), True)[1]),
        "... and so does the uninitialised one",
    )
    # one bad among several still fails: the check is not 'all lines' or 'the first line'
    check(verdict(parse_status(at_pin + ahead + at_pin), False)[0] == 1,
          "one '+' among three lines still fails")

    print("submodule_pin self-test:", "ok" if bad == 0 else f"{bad} FAILED")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
