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
"""Does this branch still carry the registrations `main` has? (a stale-base detector)

A PR's diff is computed against its **merge base**, so a branch that forked before other lanes landed their rows shows those
rows as *removals* -- and a merge that resolves the conflict by taking the branch side deletes them. Measured three times on
2026-11-11/12:

  * engine#2091 resolved naively would have dropped eight `policy_verdict` references;
  * `main-whisper-tool` (#2183) is 5 commits behind, conflicts on `CMakeLists.txt` and `docs/npu.md`, and its diff removes
    **57** registrations -- `tool_help_check` (2 -> 0), `kernel_set_check` (1 -> 0), `dupdef_check` (2 -> 0),
    `hrx_f16_narrowing` (7 -> 4), all merged the same day;
  * a branch of mine carried another PR's four files because it was committed on a worktree whose HEAD already had them.

None of those is visible in `mergeable=true` or in the diff. This is the question that is:

    tools/ci/registration_loss_check.py --check --base origin/main
    tools/ci/registration_loss_check.py --check --base origin/main --head origin/some-branch

It parses the registration shapes that make a test exist -- `add_test(NAME X`, `add_python_test(X`, `add_executable(X` -- out of
`CMakeLists.txt` and `cmake/**/*.cmake` at both revisions, and reports the ones the head does not have. Lost rows exit 1,
because a row disappearing is either an intended deletion (say so in the commit) or the merge trap.

usage: registration_loss_check.py --check --base REF [--head REF] [--root DIR]
       registration_loss_check.py --self-test
"""
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

# The three shapes that create something runnable, and their names. `add_executable` is included because a target without a
# row is still a build product other rows depend on (`add_dependencies`).
SHAPES = (
    (re.compile(r"add_test\(\s*NAME\s+([A-Za-z0-9_.+-]+)"), "test"),
    (re.compile(r"add_python_test\(\s*([A-Za-z0-9_.+-]+)"), "py"),
    (re.compile(r"add_executable\(\s*([A-Za-z0-9_.+-]+)"), "exe"),
)

FILES = ("CMakeLists.txt",)


def cmake_files(root: pathlib.Path) -> list[str]:
    out = list(FILES)
    cm = root / "cmake"
    if cm.is_dir():
        out += sorted(str(p.relative_to(root)) for p in cm.glob("*.cmake"))
    return out


def registrations(text: str) -> set[tuple[str, str]]:
    """{(kind, name)} from one CMake file's text. Comments are stripped so a commented-out row is not counted as present."""
    live = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    out = set()
    for pattern, kind in SHAPES:
        out.update((kind, m.group(1)) for m in pattern.finditer(live))
    return out


def files_at_rev(root: pathlib.Path, rev: str) -> list[str]:
    """The CMake files AS THEY EXIST AT THAT REVISION, not as they exist in the working tree.

    Enumerating the working tree is how this check first lied to itself: a branch that dropped `cmake/tests-a.cmake` still has
    that path in nobody's tree, so the base revision was never asked for it and its rows came back invisible.
    """
    p = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "--name-only", rev], capture_output=True, text=True, check=False)
    if p.returncode != 0:
        return []
    return [rel for rel in p.stdout.splitlines()
            if rel == "CMakeLists.txt" or (rel.startswith("cmake/") and rel.endswith(".cmake"))]


def at_rev(root: pathlib.Path, rev: str) -> dict[tuple[str, str], str]:
    """{(kind, name): file} for one revision."""
    found: dict[tuple[str, str], str] = {}
    for rel in files_at_rev(root, rev):
        p = subprocess.run(["git", "-C", str(root), "show", f"{rev}:{rel}"], capture_output=True, text=True, check=False)
        if p.returncode != 0:
            continue
        for entry in registrations(p.stdout):
            found.setdefault(entry, rel)
    return found


def at_tree(root: pathlib.Path) -> dict[tuple[str, str], str]:
    found: dict[tuple[str, str], str] = {}
    for rel in cmake_files(root):
        p = root / rel
        if not p.is_file():
            continue
        for entry in registrations(p.read_text(errors="replace")):
            found.setdefault(entry, rel)
    return found


def check(root: pathlib.Path, base: str, head: str | None = None) -> int:
    before = at_rev(root, base)
    after = at_rev(root, head) if head else at_tree(root)
    if not before:
        print(f"registration_loss: nothing parsed at {base} - cannot tell, and that is not a pass")
        return 2
    lost = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    for kind, name in lost:
        print(f"registration_loss: {kind} {name} ({before[(kind, name)]}) exists at {base} and NOT at {head or 'the working tree'}")
    if added:
        print(f"registration_loss: {len(added)} registration(s) added ({', '.join(n for _, n in added[:5])}"
              f"{', ...' if len(added) > 5 else ''})")
    if lost:
        print(f"registration_loss: {len(lost)} of {len(before)} registration(s) at {base} are missing. Either this branch "
              f"means to delete them (say so in the commit message) or its base is stale: a diff against the merge base "
              f"shows other lanes' rows as removals, and resolving the conflict by taking this side deletes them. "
              f"Rebase onto {base} and keep every row it has, adding yours on top.")
        return 1
    print(f"registration_loss: all {len(before)} registration(s) at {base} are still present"
          f"{f' ({len(added)} added)' if added else ''}")
    return 0


def self_test() -> int:
    """Synthetic repositories: the check must fail on a stale branch and pass on a current one, and the fixtures have to be
    COMMITS rather than only files -- a working tree keeps every file an earlier commit wrote, which is exactly the illusion
    that hides a stale base."""
    import tempfile
    bad = 0

    def check_that(ok: bool, what: str) -> None:
        nonlocal bad
        print(("  ok   " if ok else "  FAIL ") + what)
        if not ok:
            bad += 1

    def repo(trees: list[dict[str, str | None]]) -> tuple[pathlib.Path, list[str]]:
        """A repo with one commit per tree; a None value deletes that path. Returns (dir, [shas])."""
        d = pathlib.Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q", str(d)], check=False)
        for cmd in (["config", "user.email", "t@t"], ["config", "user.name", "t"]):
            subprocess.run(["git", "-C", str(d), *cmd], check=False)
        shas = []
        for i, tree in enumerate(trees):
            for rel, body in tree.items():
                p = d / rel
                if body is None:
                    if p.exists():
                        p.unlink()
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(body)
            subprocess.run(["git", "-C", str(d), "add", "-A"], check=False)
            subprocess.run(["git", "-C", str(d), "commit", "-qm", f"c{i}"], check=False)
            shas.append(subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip())
        return d, shas

    # 1. the #2183 shape: main has a row the branch never had -> failure, and the row is named
    d, shas = repo([{"CMakeLists.txt": "add_test(NAME keep COMMAND x)\n"},
                    {"CMakeLists.txt": "add_test(NAME keep COMMAND x)\nadd_test(NAME later COMMAND y)\n"}])
    check_that(check(d, shas[1], shas[0]) == 1, "a row that exists at base and not at head is a failure")
    # 2. base compared to itself -> clean
    check_that(check(d, shas[1], shas[1]) == 0, "...and base against itself passes")
    # 3. an added row is reported, not a failure
    check_that(check(d, shas[0], shas[1]) == 0, "an added row is reported but does not fail")
    # 4. a commented-out row is not a row
    check_that(registrations("# add_test(NAME gone COMMAND z)\nadd_test(NAME here COMMAND z)\n") == {("test", "here")},
               "a commented-out registration is not counted as present")
    # 5. family files count, and a branch that dropped one loses its rows
    d2, shas2 = repo([{"CMakeLists.txt": "", "cmake/tests-a.cmake": "add_test(NAME fam1 COMMAND x)\n"},
                      {"cmake/tests-a.cmake": None, "cmake/tests-b.cmake": "add_test(NAME fam2 COMMAND y)\n"}])
    check_that(check(d2, shas2[0], shas2[1]) == 1, "rows in cmake/*.cmake count, and a branch that dropped the file lost them")
    # 6. nothing parsed at the base is 'cannot tell', not a pass
    d3, shas3 = repo([{"CMakeLists.txt": "# nothing\n"}])
    check_that(check(d3, shas3[0], shas3[0]) == 2, "a base with no registrations is exit 2, not a pass")

    print("registration_loss_check self-test:", "ok" if bad == 0 else f"{bad} FAILED")
    return 0 if bad == 0 else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--self-test", action="store_true")
    ap.add_argument("--base", help="the revision this branch should be compared against, usually origin/main")
    ap.add_argument("--head", help="the revision to check; default is the working tree")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.base:
        ap.error("--check needs --base")
    return check(pathlib.Path(args.root).resolve(), args.base, args.head)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
