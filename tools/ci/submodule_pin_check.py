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

`git submodule status` cannot report one more state a build cares about: a submodule that IS at the pinned commit and
has uncommitted changes in its own work tree. It prints ' ' for that, and a build compiles the uncommitted files. On
2026-11-11 the same box that motivated this check had 301 such files in `third_party/llama.cpp`. Those are reported
always, and --require-clean turns them into a failure.

Usage:
  tools/ci/submodule_pin_check.py --check [--root .] [--require-initialised] [--require-clean]
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


def verdict(rows: list[tuple[str, str, str, str]], require_initialised: bool,
            dirty: dict[str, int] | None = None, require_clean: bool = False) -> tuple[int, list[str]]:
    """(rc, messages). rc 0 pass, 1 a submodule is not at its pin, 2 nothing readable.

    `dirty` maps a path to the number of paths its work tree has changed (git status --porcelain). A submodule at its
    pin with a dirty work tree compiles code the pin does not name; it is always reported and fails only under
    --require-clean, because a deliberate local patch is a real thing a box does and the fix for it (move the pin) is
    not always the caller's to make."""
    if not rows:
        return 2, ["submodule_pin: no submodules reported - cannot tell, and that is not a pass"]
    out: list[str] = []
    bad = unknown = uninit = dirty_at_pin = 0
    for flag, sha, path, describe in rows:
        if flag == " ":
            n, ndel = (dirty or {}).get(path, (0, 0))
            if n:
                dirty_at_pin += 1
                # `checkout -- .` restores the work tree FROM THE INDEX, so it cannot bring back a deletion staged in the
                # index (`D `) -- `checkout HEAD -- .` restores both columns, and for a pin checkout HEAD is the pin. Measured
                # on a throwaway repo: after `checkout -- .` the `D ` row is still there; after `checkout HEAD -- .` the
                # porcelain is empty.
                if ndel == n:
                    out.append(
                        f"submodule_pin: {path} is at the pinned {sha[:12]} but {n} path(s) the pin names are MISSING "
                        f"from its work tree, so the build compiles without them. "
                        f"Repair: git -C {path} checkout HEAD -- ."
                    )
                else:
                    # Mixed: say how many of the changes are deletions, because that is the state a repair half-does.
                    out.append(
                        f"submodule_pin: {path} is at the pinned {sha[:12]} but its work tree has {n} changed "
                        f"path(s){', ' + str(ndel) + ' of them deletions the build compiles without' if ndel else ''}"
                        f", so the build compiles code the pin does not name. Repair: git -C {path} stash, or "
                        f"move the pin if the change is meant to stay (git add {path}); "
                        f"`git -C {path} status` lists the rest"
                    )
                if require_clean:
                    bad += 1
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
    if dirty_at_pin and not require_clean:
        out.append(f"submodule_pin: WARNING, not a failure: {dirty_at_pin} submodule(s) at their pin with a dirty work tree (--require-clean makes this an error)")
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


def run_dirty(root: pathlib.Path, rows: list[tuple[str, str, str, str]]) -> dict[str, tuple[int, int]]:
    """{path: (changed, deleted)} for the submodules that ARE at their pin.

    Only those: a '+' or 'U' row already fails and its dirty count would be noise. `git status --porcelain` inside the
    submodule is the only thing that reports this state - `git submodule status` prints ' ' for it.

    Deletions are counted separately because the repair differs and the count alone cannot tell them apart: on 2026-11-12
    all three dirty work trees on the box were 301/32/20 **deletions** (files the pin names, absent from disk, 64 of them
    `.jinja` templates the renderer compiles), and the message then said `stash` - advice for a work tree somebody edited.
    """
    out: dict[str, tuple[int, int]] = {}
    for flag, _sha, path, _describe in rows:
        if flag != " ":
            continue
        try:
            p = subprocess.run(["git", "-C", str(root / path), "status", "--porcelain"],
                               capture_output=True, text=True, check=False)
        except OSError:
            continue
        if p.returncode == 0:
            rows_changed = [l for l in p.stdout.splitlines() if l.strip()]
            if rows_changed:
                # porcelain v1 is `XY path`: X is the index, Y the work tree, so a deletion shows as `D ` or ` D`
                deleted = len([l for l in rows_changed if "D" in l[:2]])
                out[path] = (len(rows_changed), deleted)
    return out


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
    ap.add_argument(
        "--require-clean",
        action="store_true",
        help="treat a submodule that is at its pin with a dirty work tree as a failure (CI uses this)",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    text, rc = run_git(pathlib.Path(args.root).resolve())
    if rc != 0:
        print(f"submodule_pin: cannot tell: {text}")
        return 2
    rows = parse_status(text)
    rc, messages = verdict(rows, args.require_initialised, run_dirty(pathlib.Path(args.root).resolve(), rows),
                           args.require_clean)
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
    # engine#2129, the half `git submodule status` cannot report: at the pin, dirty work tree
    # deletions-only is its own state with its own repair (2026-11-12: 301/32/20 deletions on the box, message said `stash`)
    _del_msgs = verdict(parse_status(at_pin), False, {"third_party/llama.cpp": (301, 301)})[1]
    check(any("MISSING" in m for m in _del_msgs),
          "a work tree whose paths are all DELETED says so")
    check(any("checkout HEAD -- ." in m for m in _del_msgs),
          "...and names `checkout HEAD -- .`: bare `checkout -- .` restores from the index and cannot undo a STAGED deletion")
    check(any("200" in m or "of them deletions" in m for m in
              verdict(parse_status(at_pin), False, {"third_party/llama.cpp": (201, 200)})[1]),
          "a mixed work tree names how many of its changes are deletions")
    check(any("stash" in m for m in verdict(parse_status(at_pin), False, {"third_party/llama.cpp": (301, 200)})[1]),
          "a mixed work tree keeps the stash/pin advice")
    check(verdict(parse_status(at_pin), False, {"third_party/llama.cpp": (301, 301)})[0] == 0,
          "...and deletions alone still do not fail without --require-clean")

    check(verdict(parse_status(at_pin), False, {"third_party/llama.cpp": (301, 0)})[0] == 0,
          "at the pin with 301 dirty paths -> pass by default (a deliberate local patch is a real thing)")
    check(any("301 changed" in m for m in verdict(parse_status(at_pin), False, {"third_party/llama.cpp": (301, 0)})[1]),
          "... but it IS reported, with its count")
    check(verdict(parse_status(at_pin), False, {"third_party/llama.cpp": (301, 0)}, True)[0] == 1,
          "... and FAILS with --require-clean")
    check(verdict(parse_status(at_pin), False, {}, True)[0] == 0,
          "a clean submodule at its pin still passes with --require-clean")
    check(not any("changed" in m for m in verdict(parse_status(ahead), False, {"third_party/llama.cpp": (5, 0)})[1]),
          "a dirty '+' row is not double-counted (the commit is the failure there)")

    # one bad among several still fails: the check is not 'all lines' or 'the first line'
    check(verdict(parse_status(at_pin + ahead + at_pin), False)[0] == 1,
          "one '+' among three lines still fails")

    # run_dirty() itself -- where the column logic lives -- against a real repo with one work-tree deletion (` D`) and one
    # staged deletion (`D `). The arms above feed verdict() a dict, so without this the new logic is only covered by hand.
    import tempfile as _tf
    rd_root = pathlib.Path(_tf.mkdtemp())
    sub = rd_root / "third_party" / "x"
    sub.mkdir(parents=True)
    def _git(*args):
        return subprocess.run(["git", "-C", str(sub), *args], capture_output=True, text=True, check=False)
    _git("init", "-q")
    _git("config", "user.email", "t@t"); _git("config", "user.name", "t")
    (sub / "a.txt").write_text("a\n"); (sub / "b.txt").write_text("b\n")
    _git("add", "."); _git("commit", "-qm", "base")
    _git("rm", "-q", "--cached", "b.txt"); (sub / "a.txt").unlink()
    got = run_dirty(rd_root, [(" ", "0" * 40, "third_party/x", "")])
    # 3 changed, 2 deleted: `git rm --cached b.txt` leaves the file untracked (`?? b.txt`), and an untracked path is also
    # code the pin does not name -- so the count is right and the intent of the arm is the DELETION count, both columns.
    check(got.get("third_party/x") == (3, 2),
          f"run_dirty() counts both porcelain columns as deletions (got {got.get('third_party/x')})")

    print("submodule_pin self-test:", "ok" if bad == 0 else f"{bad} FAILED")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
