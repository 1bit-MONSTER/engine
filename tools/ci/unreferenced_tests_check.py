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
"""A standalone tests/*.sh that nothing references never runs, and nothing says so.

Why this exists (engine#1820, behind #1800's withdrawn claim): three PRs in one
night added or touched a shell test that nothing ran -- #1806's
`base_modes_test.sh` (fixed by that PR), #1777's
`release_check_hrx_init_reason_test.sh` (reached main unregistered) and #1809's
unbuilt `add_test`. `ci_targets_check.py` owns "a test is declared and never
runs" but reasons about **targets**, and a standalone `.sh` has none, so it is
blind to the shell half -- which was the majority.

The rule is one line: every `tests/*.sh` must be named by something that could
run it (the build files, the workflows, `scripts/`, `tools/`, or another test).
A script named only by **itself** does not count: that is the false negative the
first hand-run missed, because the test's own `usage:` line names it.

Deliberate exclusions live in `tools/ci/unreferenced_tests.txt`, one path and
one reason per line, and are checked in the **other** direction too: an entry
whose file is now referenced, or was deleted, fails rather than rotting there.

    tools/ci/unreferenced_tests_check.py --check [--root .]
    tools/ci/unreferenced_tests_check.py --self-test
"""
import argparse
import glob
import os
import sys

ALLOW = os.path.join("tools", "ci", "unreferenced_tests.txt")
SELF = os.path.join("tools", "ci", "unreferenced_tests_check.py")  # this file names tests in its docstring by design

# What counts as "something that could run it". CMakeLists covers the add_test rows, cmake/ the
# split test files, .github/ the workflows, and scripts/tools/tests everything else that shells out.
SEARCH_TREES = ["cmake", os.path.join(".github", "workflows"), "scripts", "tools", "tests"]
SEARCH_FILES = ["CMakeLists.txt"]


def load_allow(root):
    """path -> reason, from the allow-list; blank lines and comments are ignored."""
    path = os.path.join(root, ALLOW)
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry, _, reason = line.partition("#")
        out[entry.strip()] = reason.strip()
    return out


def referencing_files(root, need, skips):
    """Every file under the search trees that names `need`'s basename.

    `skips` holds the file itself and the allow-list: a test naming its own basename (a `usage:` line) is not
    referenced by itself, and the allow-list names every test it excuses, which would otherwise make every
    excuse look like a reference and the "stale excuse" arm fire on all of them."""
    base = os.path.basename(need)
    skip_norm = {os.path.normpath(os.path.join(root, s)) for s in skips}
    # shortcut: a *comment* line naming a test is not a reference, which all these file types spell with `#`. It
    # catches the case that made this checker blind to the very test it was written for: its own docstring names
    # `release_check_hrx_init_reason_test.sh`, and that counted. Prose in another file's comment can still mask a
    # dangling test; upgrade by requiring the name to be on a line that also names a runner, if that ever bites.
    hits = set()
    files = list(SEARCH_FILES)
    for tree in SEARCH_TREES:
        for pattern in ("**/*",):
            files.extend(glob.glob(os.path.join(root, tree, pattern), recursive=True))
    for rel in files:
        full = os.path.join(root, rel)
        if not os.path.isfile(full) or os.path.normpath(full) in skip_norm:
            continue
        try:
            text = open(full, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for line in text.splitlines():
            if base in line and not line.lstrip().startswith("#"):
                hits.add(rel)
                break
    return hits


def check(root):
    tests = sorted(glob.glob(os.path.join(root, "tests", "*.sh")))
    allow = load_allow(root)
    unreferenced = []
    for t in tests:
        rel = os.path.relpath(t, root)
        if not referencing_files(root, t, skips=[t, ALLOW, SELF]):
            unreferenced.append(rel)
    problems = []

    # Direction 1: a test nothing names and nobody has explicitly excused.
    for rel in unreferenced:
        if rel not in allow:
            problems.append("%s: referenced by nothing and not in %s -- register it (add_test) or "
                            "say why not there with the reason" % (rel, ALLOW))

    # Direction 2: an excuse that no longer excuses anything.
    for rel, reason in allow.items():
        if rel not in [os.path.relpath(t, root) for t in tests]:
            problems.append("%s: listed in %s but no such test -- delete the entry" % (rel, ALLOW))
        elif rel not in unreferenced:
            problems.append("%s: listed in %s but it IS referenced now -- delete the entry" % (rel, ALLOW))
        elif not reason:
            problems.append("%s: listed in %s with no reason -- the reason is the point" % (rel, ALLOW))

    if problems:
        print("unreferenced tests (engine#1820):")
        for p in problems:
            print("  " + p)
        return 1
    print("ok  %d shell tests, all referenced or excused with a reason (%d excused)" % (len(tests), len(allow)))
    return 0


def self_test():
    """A fixture tree: the three outcomes must all be distinguished."""
    import tempfile
    import shutil

    def write(root, rel, text):
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w", encoding="utf-8").write(text)

    fails = 0
    with tempfile.TemporaryDirectory() as d:
        write(d, "CMakeLists.txt", 'add_test(NAME good COMMAND tests/good.sh)\n')
        write(d, "tests/good.sh", "#!/bin/bash\nexit 0\n")
        write(d, "tests/dangling.sh", "#!/bin/bash\n# usage: dangling.sh <arg>\nexit 0\n")
        write(d, ALLOW, "# comment\ntests/dangling.sh   # excused: this fixture says so\n")
        if check(d) != 0:
            print("FAIL self-test: a dangling test with a reason next to it must pass")
            fails += 1

        # direction 1: drop the excuse -> the dangling test must be reported
        write(d, ALLOW, "# nothing excused\n")
        if check(d) == 0:
            print("FAIL self-test: a dangling test with no excuse must fail")
            fails += 1
        else:
            print("ok  self-test: a dangling test with no excuse is reported")

        # the self-name trap: a file naming only itself must still count as dangling
        write(d, ALLOW, "# nothing excused\n")
        write(d, "tests/selfnamed.sh", "#!/bin/bash\n# selfnamed.sh runs nothing else\nexit 0\n")
        if check(d) == 0:
            print("FAIL self-test: a test naming only itself must still be dangling")
            fails += 1
        else:
            print("ok  self-test: a test naming only itself is still dangling")

        # direction 2: an excuse for something referenced, and an excuse for nothing
        os.remove(os.path.join(d, "tests/selfnamed.sh"))
        write(d, ALLOW, "tests/good.sh   # but it is registered\n")
        if check(d) == 0:
            print("FAIL self-test: an excuse for a registered test must fail")
            fails += 1
        else:
            print("ok  self-test: a stale excuse for a registered test is reported")
        write(d, ALLOW, "tests/deleted.sh   # gone\n")
        if check(d) == 0:
            print("FAIL self-test: an excuse for a missing test must fail")
            fails += 1
        else:
            print("ok  self-test: an excuse for a missing test is reported")

        # and the all-clear -- the earlier fixture's dangling test must be gone, or this case is not clean
        for leftover in ("dangling.sh", "selfnamed.sh"):
            p = os.path.join(d, "tests", leftover)
            if os.path.exists(p):
                os.remove(p)
        write(d, ALLOW, "# nothing excused\n")
        write(d, "tests/referenced_elsewhere.sh", "#!/bin/bash\nexit 0\n")
        write(d, "scripts/run_it.sh", "#!/bin/bash\nexec tests/referenced_elsewhere.sh\nexit 0\n")
        write(d, "tests/dangling2.sh", "#!/bin/bash\nexit 0\n")
        write(d, ALLOW, "tests/dangling2.sh   # fixture\n")
        if check(d) != 0:
            print("FAIL self-test: the clean tree must pass")
            fails += 1
        else:
            print("ok  self-test: the clean tree passes")

    print("%s" % ("self-test FAILED" if fails else "self-test ok"))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.check:
        return check(os.path.abspath(a.root))
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
