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
"""A branch that ADDS a ctest row name the base tip already registers fails at Configure, and only for the
branch that lost the race.

`CMakeLists.txt` is where every lane appends its registration row, so a merged row's **name** becomes a
namespace nobody reserves. Two lanes fixing the same hole from two directions each add the same row; whichever
one is rebased first sees `add_test given test NAME "X" which already exists in this directory` and stops in
Configure -- on a change that is correct.

Two live instances on 2026-11-21, both worth reading before trusting a green build:

  * `registry_build_self` merged to `main` at 17:06:10 (`b299efbc4`, engine#3217). engine#3220's branch was cut
    before that and adds the identical line, so **its** Configure is the one that fails while engine#3219, the
    same fix on an older base, is green;
  * the row existed twice at all because engine#3216 (a checker counting a tool's *name* as wiring) is why
    neither lane could see the other's registration.

What already catches this, and why none of it does:

  * `pr_branch_stale_check` measures **deletions** -- a branch that adds a row deletes nothing;
  * `duplicate_fixes_check` groups claims by **issue**, and the pair above is two different issues;
  * CMake catches it, at Configure, after a push, on one side -- i.e. in CI, costing the single build queue and
    a diagnosis cycle (#1979 covered the stale-hunk cousin of this, which a rebase shows you as a conflict; this
    one shows up as a CMake error).

So: intersect the row names a branch **adds** with the names present in `<base>:CMakeLists.txt` -- the base
**tip**, because that is where the duplicate appears. Rebased, the diff stops adding the row and this goes quiet,
which is the fix.

Narrow on purpose: CMakeLists.txt only, and only the two registration spellings this tree uses
(`add_test(NAME X` and `add_python_test(X`). A name computed from a variable is not compared (the arms pin that),
because a check that guesses at expansion is a check that reports names it did not read.

Two shapes that are NOT duplicates, both measured on `main`'s own history before this row landed: a row REWRITTEN
in place (engine#3018 removed and re-added the name in one diff), and the same name registered inside a **different
`if()` branch** (engine#2989), where the two are mutually exclusive at configure time. The first is subtracted by
reading the diff's `-` side; the second is cannot-tell, because a name-only comparison cannot see which branch
configure takes.

Exit: 0 no collision, 1 the branch adds a name the base already registers (and the diff does not remove it),
2 cannot tell (no git, an unresolvable base, a shallow clone where `base...HEAD` is not a real comparison, or a
collision where either side sits inside an `if()` block).

  tools/ci/ctest_row_name_check.py --check [--base origin/main] [--root .]
  tools/ci/ctest_row_name_check.py --self-test
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROW_FILE = "CMakeLists.txt"
# Everything that registers into the ONE ctest namespace: the top-level list plus the family files its glob at the end
# include()s (engine#1979; CONTRIBUTING.md, "A new test's registration goes in its own file").
ROW_PATHS = (ROW_FILE, "cmake/tests-*.cmake")
# `git diff` matches that glob in a pathspec and `git ls-tree` does not (measured), so the tree side lists `cmake/`
# and filters on the prefix/suffix.
ROW_GLOB_PREFIX, ROW_GLOB_SUFFIX = "cmake/tests-", ".cmake"
# The two spellings this tree registers a test with. `add_python_test` is the repo's own wrapper (it calls
# `add_test(NAME ...)`), so both create the same namespace and both must be read.
ROW_RES = (
    re.compile(r"add_test\(\s*NAME\s+([A-Za-z0-9_.+-]+)"),
    re.compile(r"add_python_test\(\s*([A-Za-z0-9_.+-]+)"),
)


def git(root: pathlib.Path, *args: str):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)


def row_names(text: str) -> dict[str, tuple[str, int]]:
    """name -> (the line that registers it, how many `if(`-blocks deep it sits).

    The depth is what tells a real duplicate from two mutually exclusive registrations: engine#3018 rewrote a row in
    place (the same diff removes and re-adds the name) and engine#2989 registered `step_users` a second time inside a
    *different* `if()` branch. Both are green at Configure; a name-only diff called both duplicates.
    """
    out: dict[str, tuple[str, int]] = {}
    depth = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        low = stripped.lower()
        for rx in ROW_RES:
            m = rx.search(line)
            if m:
                # Shallowest sighting wins: a name registered twice in ONE file (once outside a guard, once inside)
                # is a duplicate that does not depend on the guard, so the pair is not "guarded".
                prev = out.get(m.group(1))
                if prev is None or depth < prev[1]:
                    out[m.group(1)] = (stripped.strip(), depth)
        # `if(`…`endif()` nesting, counted after the row so a registration inside a block is seen at that depth
        if re.match(r"(?i)if\s*\(", stripped):
            depth += 1
        elif re.match(r"(?i)endif\s*\(", stripped):
            depth = max(0, depth - 1)
    return out


def counted_row_names(lines: list[str]) -> dict[str, int]:
    """name -> how many times these lines register it.

    `row_names()` is a dict on purpose (a name is either taken or not), so it collapses exactly the case this is for:
    a branch that registers the SAME name twice collides with itself, and no base entry has to exist for Configure
    to fail (`add_test given test NAME "x" which already exists in this directory`).
    """
    out: dict[str, int] = {}
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        for rx in ROW_RES:
            m = rx.search(line)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def changed_row_lines(root: pathlib.Path, base: str):
    """(added, removed) registration lines of this branch's diff, or (None, None) when git cannot answer.

    BOTH sides, since engine#3018: `-add_test(NAME X COMMAND t "")` / `+add_test(NAME X COMMAND t)` is one row
    rewritten in place -- a name the same diff also removes is not a duplicate, after a rebase or before it.
    """
    p = git(root, "diff", "--unified=0", f"{base}...HEAD", "--", *ROW_PATHS)
    if p.returncode != 0:
        return None, None
    added = [l[1:] for l in p.stdout.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in p.stdout.splitlines() if l.startswith("-") and not l.startswith("---")]
    return added, removed


def tree_rows(root: pathlib.Path, rev: str) -> dict[str, tuple[str, int]] | None:
    """Every name registered anywhere in the namespace at `rev`, with the if-depth it really has in that tree.

    Read from the FILES at `rev`, never from the diff: a row added inside an `if()` block that already existed has no
    `if(` line in its diff, so a depth taken from the added lines reads 0 and calls a mutually exclusive pair a
    collision (measured: the #2989 shape with the guard already on the base).
    """
    p = git(root, "ls-tree", "-r", "--name-only", rev, "--", ROW_FILE, "cmake/")
    if p.returncode != 0 or ROW_FILE not in p.stdout.split():
        return None
    out: dict[str, tuple[str, int]] = {}
    for path in [q for q in p.stdout.split() if q == ROW_FILE or _family_file(q)]:
        q = git(root, "show", f"{rev}:{path}")
        if q.returncode != 0:
            return None
        for name, seen in row_names(q.stdout).items():
            # Lowest depth wins, and a name registered twice in one tree takes the shallower of the two: a single
            # UNCONDITIONAL sighting means the collision does not depend on a guard, so the pair is not "guarded".
            if name not in out or seen[1] < out[name][1]:
                out[name] = seen
    return out


def _family_file(path: str) -> bool:
    return path.startswith(ROW_GLOB_PREFIX) and path.endswith(ROW_GLOB_SUFFIX) and "/" not in path[len(ROW_GLOB_PREFIX):]


def check(root: pathlib.Path, base: str) -> tuple[int, list[str]]:
    """(rc, messages)."""
    if git(root, "rev-parse", "--git-dir").returncode != 0:
        return 2, [f"{root}: not a git checkout, so the base cannot be compared"]
    shallow = git(root, "rev-parse", "--is-shallow-repository").stdout.strip()
    if shallow == "true":
        # A depth-N clone cannot compute `base...HEAD`: the merge base is outside the truncated history, so a
        # diff that looks clean proves nothing. cannot-tell, not a pass (engine#3106's rule).
        return 2, [f"{root}: a shallow clone, so `{base}...HEAD` is not a comparison the base register was read from"]
    if git(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").returncode != 0:
        return 2, [f"--base {base} does not resolve: fetch it, then re-run (cannot tell, not a pass)"]
    added, removed = changed_row_lines(root, base)
    known, head = tree_rows(root, base), tree_rows(root, "HEAD")
    if added is None or known is None or head is None:
        return 2, [f"cannot read the row namespace at {base} or diff against it (cannot tell, not a pass)"]
    # Which names the diff adds/removes, from the diff; the SHAPE of a registration (its if-depth) from the tree it
    # lands in, because that is what Configure sees.
    taken = {n: head.get(n, (line, 0)) for n, line in
             ((n, v[0]) for n, v in row_names("\n".join(added)).items())}
    freed = set(row_names("\n".join(removed)))
    # A name this branch registers TWICE collides with itself: no base entry, no guard, and Configure stops on it.
    # Counted before the dict above collapses the repeat; nothing here needs the base or the tree, so it is a hard
    # collision rather than a cannot-tell (measured: rc 0 before, `cmake -S . -B` rc 1 "already exists").
    counts = counted_row_names(added)
    # An ADDITION is never cancelled by a removal: a removal only cancels the base's copy, which is what `freed` and
    # `clashes` below are for. Sharing `freed` here forgave -1/+2 of one name in one diff (a row moved into a family
    # file and written twice), which leaves two copies in the tree and stops Configure.
    doubled = sorted(n for n, count in counts.items() if count > 1)
    if doubled:
        return 1, ([f"{len(doubled)} ctest row name(s) this branch registers MORE THAN ONCE:"]
                   + [f"  {n}: registered {counts[n]}x by this branch" for n in doubled]
                   + ["  two registrations of one name fail Configure whatever the base holds: drop one"])
    if not taken:
        return 0, []
    clashes = sorted(n for n in taken if n in known and n not in freed)
    if not clashes:
        # Either a fresh name, or a row REWRITTEN in place (the same diff removed it) -- engine#3018's shape.
        return 0, []
    # Mutual exclusion is only possible when BOTH sides are guarded: an unconditional registration at either end is
    # a duplicate whenever the other side's guard is on, which is a Configure failure, not a cannot-tell (measured
    # both directions on the real tree: `or` answered 2 while `cmake -S . -B` died on the duplicate).
    guarded = [n for n in clashes if taken[n][1] and known[n][1]]
    if guarded:
        # engine#2989 registered a name a second time inside a DIFFERENT `if()` branch; both are green at Configure
        # because the branches are mutually exclusive, and a name-only diff cannot see that. cannot-tell, with the
        # one action that answers it, rather than a red row on a merge that was correct.
        return 2, (
            [f"{len(guarded)} ctest row name(s) this branch adds are also registered at {base}, but at least one side "
             f"sits inside an `if()` block, which can make the registrations mutually exclusive:"]
            + [f"  {n}: branch -> {taken[n][0][:90]} (if-depth {taken[n][1]}); {base} -> {known[n][0][:90]} "
               f"(if-depth {known[n][1]})" for n in guarded]
            + ["  configure the tree to find out (cannot tell, not a pass), or rename one of the two rows"])
    msgs = [f"{len(clashes)} ctest row name(s) this branch ADDS are already registered at {base}:"]
    for n in clashes:
        msgs.append(f"  {n}")
        msgs.append(f"    this branch: {taken[n][0]}")
        msgs.append(f"    {base}: {known[n][0]}")
    msgs.append("  rebase onto the base: the duplicate row disappears from the diff and the name is registered once "
                "(CMake stops the Configure of whichever branch rebases second, and only that one)")
    return 1, msgs


def self_test() -> int:
    import tempfile

    rc = 0

    def chk(want, got, what):
        nonlocal rc
        if want == got:
            print(f"ok   {what}")
        else:
            print(f"FAIL {what}: want {want!r}, got {got!r}")
            rc = 1

    def repo(tmp: pathlib.Path, name: str) -> pathlib.Path:
        d = tmp / name
        d.mkdir()
        for k, v in (("user.email", "t@e"), ("user.name", "t"), ("commit.gpgsign", "false")):
            subprocess.run(["git", "-C", str(d), "init", "-q"], check=True)
            subprocess.run(["git", "-C", str(d), "config", k, v], check=True)
        (d / ROW_FILE).write_text("# base\nadd_test(NAME taken_row COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(d), "commit", "-qm", "base"], check=True)
        subprocess.run(["git", "-C", str(d), "branch", "-M", "main"], check=True)
        # A BRANCH, not commits on main: `main..HEAD` is empty when the work is on main itself, which made three
        # of these arms pass vacuously the first time they were written (caught by arm 1 failing, not by review).
        subprocess.run(["git", "-C", str(d), "checkout", "-q", "-b", "feature"], check=True)
        return d

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = pathlib.Path(tmp)

        # 1. the #3220 shape: the branch adds the very line the base tip already registers
        d = repo(tmpd, "same_name")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "add_test(NAME taken_row COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "adds a duplicate"], check=True)
        got, msgs = check(d, "main")
        chk(1, got, "a branch that re-adds a name the base registers is reported (engine#3220's Configure failure)")
        chk(True, any("taken_row" in m for m in msgs), "...and the message names the row")

        # 2. the cross-spelling case: base registers with add_test(NAME X), the branch adds add_python_test(X)
        d = repo(tmpd, "cross_spelling")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "add_python_test(taken_row ${CMAKE_SOURCE_DIR}/x.py)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "same name, other wrapper"], check=True)
        chk(1, check(d, "main")[0], "the two registration spellings share one namespace (add_python_test vs add_test(NAME")

        # 3. a fresh name is not a collision
        d = repo(tmpd, "fresh_name")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "add_python_test(brand_new_self ${CMAKE_SOURCE_DIR}/y.py --self-test)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "a new row"], check=True)
        chk(0, check(d, "main")[0], "a row with a name the base does not have is quiet")

        # 3d. the branch registers ONE NEW NAME TWICE: no base entry needed, no guard involved, and Configure fails
        # on it (measured: `cmake -S . -B` -> `add_test given test NAME "new_dup" which already exists in this
        # directory` while the added-lines comparison answered "no row name the branch adds is already registered").
        d = repo(tmpd, "added_twice")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text()
                                  + "add_python_test(new_dup ${CMAKE_SOURCE_DIR}/x.py)\n"
                                  + "add_python_test(new_dup ${CMAKE_SOURCE_DIR}/x.py)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "one new name, registered twice"], check=True)
        got, msgs = check(d, "main")
        chk(1, got, "a name the branch adds TWICE is a collision with no base involved")
        chk(True, any("new_dup" in m for m in msgs), "...and the message names the doubled row")

        # 3e. a row REMOVED here and written TWICE in a new family file: `-1/+2` of one name leaves two copies in the
        # tree, and a set of freed NAMES forgives it (measured: the tree holds 2 copies while the added-lines
        # comparison, the `n not in freed` suppression and `freed` itself all answer clean).
        d = repo(tmpd, "moved_and_doubled")
        (d / "cmake").mkdir()
        (d / ROW_FILE).write_text("# base\n")
        (d / "cmake/tests-family.cmake").write_text(
            "add_test(NAME taken_row COMMAND true)\nadd_test(NAME taken_row COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(d), "commit", "-qm", "the row moves to a family file, written twice"], check=True)
        got, msgs = check(d, "main")
        chk(1, got, "a row REMOVED here and registered twice in a new family file is a collision (a freed NAME forgives it)")
        chk(True, any("taken_row" in m for m in msgs), "...and the message names the moved row")

        # 3f. a row PARKED as a comment beside its live copy is one registration, not two -- `main` carries four such
        # comments (CMakeLists.txt:383, 649, 2354, 3230). Without the `#` skip in counted_row_names this is a false
        # collision on a tree that configures, which is the phantom-duplicate trap this file's own docstring names.
        d = repo(tmpd, "commented_duplicate")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text()
                                  + "# add_python_test(parked_row ${CMAKE_SOURCE_DIR}/x.py)  -- parked until x exists\n"
                                  + "add_python_test(parked_row ${CMAKE_SOURCE_DIR}/x.py)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "a parked row and its live copy"], check=True)
        chk(0, check(d, "main")[0], "a commented-out copy beside the live row is not a second registration")

        # 4. a name computed from a variable is not compared
        d = repo(tmpd, "variable_name")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "add_test(NAME ${SOME_NAME} COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "a computed name"], check=True)
        chk(0, check(d, "main")[0], "a name computed from a variable is not guessed at")

        # 4b. engine#3018: a row REWRITTEN in place -- the same diff removes the name it re-adds. Not a duplicate.
        d = repo(tmpd, "rewritten_in_place")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text().replace(
            "add_test(NAME taken_row COMMAND true)", 'add_test(NAME taken_row COMMAND true "")'))
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "rewrite the row"], check=True)
        chk(0, check(d, "main")[0], "a row rewritten in place is not a duplicate (engine#3018: one registration, one name)")

        # 4c. engine#2989: the same name registered in a second `if()` branch -- mutually exclusive at configure time,
        # which a name-only diff cannot see, so this is cannot-tell rather than a red row on a correct merge.
        d = repo(tmpd, "in_two_if_branches")
        (d / ROW_FILE).write_text("# base\nif(ENABLE_X)\n    add_test(NAME step_users COMMAND true)\nendif()\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "register it inside a guard"], check=True)
        # the BASE is the commit that carries the first guard, not main: created from the branch that has it
        subprocess.run(["git", "-C", str(d), "branch", "-f", "base-if", "HEAD"], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", "-B", "feature2", "base-if"], check=True)
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "if(ENABLE_Y)\n    add_test(NAME step_users COMMAND true)\nendif()\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "second branch, same name"], check=True)
        chk(2, check(d, "base-if")[0], "the same name in a different if() branch is cannot-tell, not a red row")

        # 4d. the namespace is CMakeLists.txt PLUS cmake/tests-*.cmake, both directions (CONTRIBUTING.md sends a
        # family's row to its family file; the glob at the end of CMakeLists.txt include()s them). Measured on the
        # real tree: the tool said clean while `cmake -S . -B` died with `add_test given test NAME ... already exists`.
        d = repo(tmpd, "family_file_branch")
        (d / "cmake").mkdir()
        (d / "cmake/tests-pre-pr-checks.cmake").write_text("add_python_test(taken_row ${CMAKE_SOURCE_DIR}/x.py)\n")
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(d), "commit", "-qm", "duplicate row in a family file"], check=True)
        got, msgs = check(d, "main")
        chk(1, got, "a duplicate name registered in a cmake/tests-*.cmake family file is reported (one namespace)")
        chk(True, any("taken_row" in m for m in msgs), "...and the family-file message names the row")

        d = repo(tmpd, "family_file_base")
        subprocess.run(["git", "-C", str(d), "checkout", "-q", "main"], check=True)
        (d / "cmake").mkdir()
        (d / "cmake/tests-gpu-hold.cmake").write_text("add_test(NAME family_row COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(d), "commit", "-qm", "a family row on the base"], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", "feature"], check=True)
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "add_test(NAME family_row COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "re-registers a family row in CMakeLists.txt"], check=True)
        chk(1, check(d, "main")[0], "a name the BASE's family file registers, re-added in CMakeLists.txt, is reported")

        # 4e. ONE unconditional side is a duplicate, not cannot-tell: the guarded registration collides whenever that
        # guard is on. Measured on the real tree in both directions (`cmake -S . -B` rc 1 on the duplicate while the
        # `or` rule answered 2), so the guard test is `and`, not `or`.
        d = repo(tmpd, "base_unconditional")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text()
                                  + "if(ENABLE_Y)\n    add_test(NAME taken_row COMMAND true)\nendif()\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "a guarded duplicate of an unconditional row"], check=True)
        chk(1, check(d, "main")[0], "a guarded duplicate of an UNCONDITIONAL base row is a collision, not cannot-tell")
        subprocess.run(["git", "-C", str(d), "checkout", "-q", "main"], check=True)
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text().replace(
            "add_test(NAME taken_row COMMAND true)", "if(ENABLE_X)\n    add_test(NAME taken_row COMMAND true)\nendif()"))
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "the base row is guarded too"], check=True)
        subprocess.run(["git", "-C", str(d), "branch", "-f", "guarded-base", "HEAD"], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", "-B", "feature3", "guarded-base"], check=True)
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "add_test(NAME taken_row COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "an unconditional duplicate of a GUARDED base row"], check=True)
        chk(1, check(d, "guarded-base")[0], "an UNCONDITIONAL duplicate of a guarded base row is a collision, not cannot-tell")

        # 4f. and the depth is the one the TREE has, not the one its diff shows: a row added inside an `if()` block
        # that already exists has no `if(` line in its diff, so a diff-read depth calls this pair a collision.
        d = repo(tmpd, "preexisting_guard")
        (d / ROW_FILE).write_text("# base\nif(ENABLE_X)\n    add_test(NAME taken_row COMMAND true)\nendif()\n"
                                  "if(ENABLE_Y)\n    add_test(NAME other_row COMMAND true)\nendif()\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "both guards exist on the base"], check=True)
        subprocess.run(["git", "-C", str(d), "branch", "-f", "guarded", "HEAD"], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", "-B", "feature4", "guarded"], check=True)
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text().replace(
            "if(ENABLE_Y)\n    add_test(NAME other_row COMMAND true)",
            "if(ENABLE_Y)\n    add_test(NAME other_row COMMAND true)\n    add_test(NAME taken_row COMMAND true)"))
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "the same name inside a pre-existing second guard"], check=True)
        chk(2, check(d, "guarded")[0], "a row added inside an EXISTING if() reads its real depth (cannot-tell, not a collision)")

        # 5. an unresolvable base is cannot-tell, not a pass
        d = repo(tmpd, "no_base")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "add_test(NAME taken_row COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "duplicate against a base nobody has"], check=True)
        chk(2, check(d, "no-such-base")[0], "an unresolvable --base is cannot-tell, not a pass")

        # 6. a shallow clone cannot compare base...HEAD at all
        d = repo(tmpd, "shallow_src")
        (d / ROW_FILE).write_text((d / ROW_FILE).read_text() + "add_test(NAME second_commit COMMAND true)\n")
        subprocess.run(["git", "-C", str(d), "commit", "-qam", "second"], check=True)
        sh = tmpd / "shallow"
        subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{d}", str(sh)], check=True)
        rc_sh, _ = check(sh, "origin/main")
        chk(2, rc_sh, "a shallow clone is cannot-tell: the merge base is outside the truncated history")

    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", default=".", help="tree to check (default: .)")
    ap.add_argument("--base", default="origin/main", help="the ref whose tip the branch must not re-register a row of")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.check:
        ap.error("--check or --self-test")
    rc, msgs = check(pathlib.Path(a.root), a.base)
    for m in msgs:
        print(f"ctest_row_names: {m}", file=sys.stderr if rc else sys.stdout)
    if rc == 0:
        print(f"ctest_row_names: no row name the branch adds is already registered at {a.base}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
