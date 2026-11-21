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
"""add_test_collision_check.py -- a branch must not add a ctest row name the base tip already has.

engine#3222. `CMakeLists.txt` is where every lane appends its registration row, and a merged row's NAME becomes a
namespace nobody reserves. Two lanes fixing the same hole from two directions therefore collide -- not as a patch
conflict, which a rebase resolves visibly, but as a CMake error at Configure, on the *first* branch to be rebased:

    add_test given test NAME "registry_build_self" which already exists in this

Measured twice on 2026-11-21. The branch that merged first is fine; the other one wears a red `build` for a change
that is correct, and its author spends a cycle finding out that nothing is wrong with their work (engine#3219 was
green with the identical hunk, because its base predated the merge).

The signal is NOT a patch conflict and NOT a duplicate claim: `pr_branch_stale_check` measures deletions and
`duplicate_fixes_check` groups claims by ISSUE, so a branch adding a row whose name the base already carries passes
both. This row compares NAMES:

  * names the branch ADDS  -- `git diff <base>...HEAD` lines matching `add_test(NAME X)` / `add_python_test(X`
  * names the base TIP has -- the same patterns in `git show <base>:CMakeLists.txt`

A name in both is a collision: rebasing cannot fix it by line motion, because the row is already there.

Exit: 0 no collision, 1 a collision, 2 cannot tell (no base ref, no CMakeLists.txt at the base, git failed).
The base is compared at its TIP, not at the merge base, because that is where the duplicate appears.
"""
import argparse
import re
import subprocess
import sys

BASE_DEFAULT = "origin/main"
CMAKE = "CMakeLists.txt"

# `add_test(NAME row COMMAND …)` and the repo's own `add_python_test(row …)` wrapper (CMakeLists.txt:551).
NAME_PATTERNS = (
    re.compile(r"add_test\s*\(\s*NAME\s+([A-Za-z0-9_.:-]+)"),
    re.compile(r"add_python_test\s*\(\s*([A-Za-z0-9_.:-]+)"),
)


def names_in(text):
    out = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for pat in NAME_PATTERNS:
            m = pat.search(line)
            if m:
                out.add(m.group(1))
    return out


def run(*args, root="."):
    return subprocess.run(list(args), cwd=root, capture_output=True, text=True)


def added_lines(root, base):
    """Lines this branch adds relative to its merge base with `base` (not a two-dot diff: engine#3222's own class)."""
    mb = run("git", "merge-base", base, "HEAD", root=root)
    if mb.returncode != 0 or not mb.stdout.strip():
        return None, f"no merge base between {base} and HEAD"
    diff = run("git", "diff", f"{mb.stdout.strip()}...HEAD", "--", CMAKE, root=root)
    if diff.returncode != 0:
        return None, f"git diff failed (rc {diff.returncode})"
    return [l[1:] for l in diff.stdout.splitlines() if l.startswith("+") and not l.startswith("+++")], None


def base_names(root, base):
    out = run("git", "show", f"{base}:{CMAKE}", root=root)
    if out.returncode != 0:
        return None, f"cannot read {BASE_DEFAULT if base is None else base}:{CMAKE}"
    return names_in(out.stdout), None


def check(root=".", base=BASE_DEFAULT):
    """(collisions, cannot) -- collisions is {name: None} sorted later; cannot is a reason string."""
    added, why = added_lines(root, base)
    if added is None:
        return None, why
    have, why = base_names(root, base)
    if have is None:
        return None, why
    mine = names_in("\n".join(added))
    return {n: True for n in sorted(mine & have)}, None


def self_test():
    """Fixtures through the same `names_in` the check uses, so the table cannot drift from the code."""
    ok = 0
    bad = 0

    def chk(cond, what):
        nonlocal ok, bad
        print(("ok   " if cond else "FAIL ") + what)
        if cond:
            ok += 1
        else:
            bad += 1

    adds = """
    add_test(NAME brand_new COMMAND true)
    add_python_test(existing_row --self-test)
"""
    base = """
    add_test(NAME existing_row COMMAND true)
    # add_test(NAME commented_out COMMAND true)
"""
    common = names_in(adds) & names_in(base)
    chk(common == {"existing_row"}, "a name the branch adds and the base has is a collision")
    chk("brand_new" not in common, "…and a genuinely new name is not")
    chk("commented_out" not in names_in(base), "a commented-out registration is not a name")
    chk(names_in("add_test(NAME a1 COMMAND x)\nadd_test(NAME a2 COMMAND x)") == {"a1", "a2"},
        "both plain add_test rows are read")
    chk(names_in("add_python_test(only_wrapper --self-test)") == {"only_wrapper"},
        "the add_python_test wrapper is read too (it is how most rows are registered here)")
    print(f"add_test_collision_check self-test: {ok} ok, {bad} failed")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.check:
        ap.error("one of --check or --self-test")
    hits, why = check(a.root, a.base)
    if hits is None:
        print(f"add_test_collision_check: {why} -- cannot tell, not a pass", file=sys.stderr)
        return 2
    if not hits:
        print(f"add_test_collision_check: no row this branch adds is already named in {a.base}")
        return 0
    for name in hits:
        print(f"add_test_collision_check: this branch adds {name!r}, and {a.base} already registers it -- "
              f"the merge would fail at Configure with \"add_test given test NAME … already exists\"; "
              f"drop the row (your base has it) or rename it", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
