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
                out.setdefault(m.group(1), (stripped.strip(), depth))
        # `if(`…`endif()` nesting, counted after the row so a registration inside a block is seen at that depth
        if re.match(r"(?i)if\s*\(", stripped):
            depth += 1
        elif re.match(r"(?i)endif\s*\(", stripped):
            depth = max(0, depth - 1)
    return out


def changed_row_lines(root: pathlib.Path, base: str):
    """(added, removed) lines of this branch's CMakeLists.txt diff, or (None, None) when git cannot answer.

    BOTH sides, since engine#3018: `-add_test(NAME X COMMAND t "")` / `+add_test(NAME X COMMAND t)` is one row
    rewritten in place -- a name the same diff also removes is not a duplicate, after a rebase or before it.
    """
    p = git(root, "diff", "--unified=0", f"{base}...HEAD", "--", ROW_FILE)
    if p.returncode != 0:
        return None, None
    added = [l[1:] for l in p.stdout.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in p.stdout.splitlines() if l.startswith("-") and not l.startswith("---")]
    return added, removed


def base_rows(root: pathlib.Path, base: str) -> dict[str, str] | None:
    p = git(root, "show", f"{base}:{ROW_FILE}")
    if p.returncode != 0:
        return None
    return row_names(p.stdout)  # name -> (line, if-depth)


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
    known = base_rows(root, base)
    if added is None or known is None:
        return 2, [f"cannot read {ROW_FILE} from {base} or diff against it (cannot tell, not a pass)"]
    taken = row_names("\n".join(added))
    freed = set(row_names("\n".join(removed)))
    if not taken:
        return 0, []
    clashes = sorted(n for n in taken if n in known and n not in freed)
    if not clashes:
        # Either a fresh name, or a row REWRITTEN in place (the same diff removed it) -- engine#3018's shape.
        return 0, []
    guarded = [n for n in clashes if taken[n][1] or known[n][1]]
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
