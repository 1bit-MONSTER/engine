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
"""A hardcoded `/tmp` path is RAM on this box, not disk.

`/tmp` is tmpfs on both development boxes, so a dump, a staging copy or a server log written there is
anonymous memory: it cannot be reclaimed to disk under pressure, it goes to swap, and when swap fills every
lane on the box stalls. Measured 2026-11-10 21:15Z, `/tmp` held 52 G of 62 G and swap was 71/71 GiB full;
the two largest items were a stale `1bit-fconst19*.fconst` scratch copy (3.5 G) and five `q17-*.json` dumps
(9.2 G). engine#1907/#1952 bounded the fconst scratch; the *paths* are still literals in the tree.

This check keeps the literals from spreading. It is a ratchet, not a gate: `tools/ci/tmp_paths.txt` holds the
sites that exist today and may only shrink. A new site fails, and so does a baseline line whose site is gone -
that second arm matters because a file rename would otherwise move a site out of the baseline silently, which
is the `tools/ci/ci_targets_check.py` lesson from engine#1728 (a ratchet that only counts upward cannot see a
rename).

Usage:
  tools/ci/tmp_path_check.py --check [--root .]      # rc 1 if a site was added or a baseline line went stale
  tools/ci/tmp_path_check.py --self-test             # rc 1 unless the arms below behave
  tools/ci/tmp_path_check.py --write-baseline        # rewrite tools/ci/tmp_paths.txt from the tree
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

# A `/tmp` path in C, C++ or shell. The first version of this required a LEADING QUOTE, and a reviewer measured
# what that missed: `echo y > /tmp/x.log` -- a bare path, the plain shell idiom, and arguably the most likely form
# in `scripts/*.sh` -- was invisible, while the same path in quotes was caught. So the leading quote is gone and the
# anchor is "not the tail of another path": `(?<![A-Za-z0-9_./-])` keeps `/var/tmp/x` out (the `a` before `tmp`
# blocks it) and lets `/tmp/x` in whether it follows a space, `>`, `=`, a quote or nothing at all. A trailing quote
# is included when present so the baseline names the literal as a reader finds it.
# engine#2027 review: the directive this ratchet exists for says "never `/tmp` **or `/dev/shm`**", and a pattern that
# guards one of the two reads as "the RAM paths are handled". `/dev/shm` is the same failure mode with a worse ceiling --
# capped, always RAM, never reclaimable -- so it is one alternation, and the anchor is unchanged: the character before
# the leading slash is what must not be part of another path, so `/var/dev/shm/x` stays out for the same reason
# `/var/tmp/x` does. There are no `/dev/shm` sites on main today, so this lands as a pure guard with no baseline lines.
LITERAL = re.compile(r"""(?<![A-Za-z0-9_./-])/(?:tmp|dev/shm)/[^"'\s;|)]*["']?""")

# The paths this check owns. `npu/` and `app/` are where a dump lands; `tools/` and `scripts/` are where a
# benchmark log lands. Everything else is out of scope on purpose: a test may use a temp dir (and pytest's
# `tmp_path` is the right instrument there), and `docs/` quotes paths rather than writing them.
SCAN = ("npu", "app", "tools", "scripts")
SUFFIXES = (".c", ".cc", ".cpp", ".h", ".hpp", ".sh", ".py")

# This checker's own fixture text is not a site: its self-test writes `"/tmp/..."` literals into a scratch tree to
# prove they are found. Excluded BY NAME rather than by pattern so the exclusion is one line a reader can check, and
# the run prints it (--check counts excluded files) so a second such file cannot appear unnoticed.
SELF = "tools/ci/tmp_path_check.py"

# Test files are excluded: a test that writes to a temp dir it created is not the defect, and the fixtures would
# otherwise have to be baselined too.
def is_test(rel: str) -> bool:
    parts = rel.split(os.sep)
    if any(p in ("tests", "test") for p in parts[:-1]):
        return True
    name = parts[-1]
    return name.endswith(("_test.c", "_test.cpp", "_test.py", "_test.sh")) or name.startswith("test_")


# A commented-out path is not a path. engine#1983's review took `# was include(cmake/...)` for an include and
# lost seven registrations with every gate green; the same substring trap applies here, so comments are removed
# before the match rather than filtered after it.
def strip_comments(text: str, suffix: str) -> str:
    if suffix == ".py":
        return re.sub(r"#[^\n]*", "", text)
    if suffix == ".sh":
        # A `#` starts a comment when it begins a word; `$#`, `${x#y}` and a `#` inside a string are not.
        return re.sub(r"(?<![$\w{])#(?!\{)[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"(?<!:)//[^\n]*", "", text)


def rel_literal(lit: str, rel: str) -> str:
    """The baseline form: the literal as written, so a line stays recognisable next to its file."""
    return f"{rel}\t{lit}"


def scan(root: pathlib.Path) -> dict[str, list[str]]:
    """{file: [literal, ...]} for every in-scope site, sorted and de-duplicated."""
    found: dict[str, list[str]] = {}
    excluded_self: set[str] = set()
    scan.excluded_self = excluded_self          # read by --check so the exclusion is printed, not implicit
    for top in SCAN:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            rel = path.relative_to(root).as_posix()
            if rel == SELF:
                excluded_self.add(rel)
            if is_test(rel) or rel == SELF:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # A trailing quote is stripped so the baseline records the PATH: re-quoting a site (`"/tmp/x"` ->
            # `/tmp/x`) is not a new site, and the two spellings of one path would otherwise be two baseline lines.
            hits = [m.group(0).rstrip("\"'") for m in LITERAL.finditer(strip_comments(text, path.suffix))]
            if hits:
                found[rel] = sorted(set(hits))
    return found


def read_baseline(path: pathlib.Path) -> set[str]:
    if not path.is_file():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def write_baseline(path: pathlib.Path, found: dict[str, list[str]]) -> None:
    lines = [
        "# Hardcoded /tmp paths under npu/, app/, tools/, scripts/ - the sites that exist today.",
        "# A ram on this box is tmpfs, so each of these is anonymous memory when it is written.",
        "# This list may only SHRINK: fix a site, then delete its line. A new site fails",
        "# tools/ci/tmp_path_check.py --check, and so does a line whose site is gone.",
        "# Regenerate with: tools/ci/tmp_path_check.py --write-baseline",
        "",
    ]
    for rel in sorted(found):
        for lit in found[rel]:
            lines.append(rel_literal(lit, rel))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args(argv)
    root = pathlib.Path(args.root).resolve()
    baseline_path = root / "tools" / "ci" / "tmp_paths.txt"

    found = scan(root)
    have = {rel_literal(lit, rel) for rel, lits in found.items() for lit in lits}

    if args.write_baseline:
        write_baseline(baseline_path, found)
        print(f"tmp_path_check: wrote {len(have)} site(s) to {baseline_path}")
        return 0

    if args.self_test:
        return self_test()

    base = read_baseline(baseline_path)
    new = sorted(have - base)
    gone = sorted(base - have)
    for line in new:
        print(f"tmp_path_check: NEW hardcoded /tmp path: {line.replace(chr(9), '  ')}")
    for line in gone:
        print(
            "tmp_path_check: baseline line has no site any more (fixed or renamed): "
            f"{line.replace(chr(9), '  ')} - delete it if fixed"
        )
    if new or gone:
        print(
            f"tmp_path_check: FAIL ({len(new)} new, {len(gone)} stale of {len(base)} baselined) - /tmp is tmpfs "
            "on this box, so route the path through ONEBIT_DUMP_DIR, $XDG_CACHE_HOME/1bit/scratch or a "
            "$TMPDIR the caller set, instead"
        )
        return 1
    excluded = len(getattr(scan, "excluded_self", ()))
    print(f"tmp_path_check: {len(base)} baselined site(s), none new, none stale "
          f"({excluded} file(s) excluded: this checker's own fixtures)")
    return 0


def self_test() -> int:
    import tempfile

    bad = 0

    def check(ok: bool, what: str) -> None:
        nonlocal bad
        print(("  ok   " if ok else "  FAIL ") + what)
        if not ok:
            bad += 1

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "npu" / "forward").mkdir(parents=True)
        (root / "tools" / "ci").mkdir(parents=True)
        (root / "tests").mkdir()
        # A live site, a site in a comment, a test file, and a path that merely contains "tmp".
        (root / "npu" / "forward" / "dump.cpp").write_text(
            'void f() { open("/tmp/npu_layer.bin"); }\n'
            "// was: open(\"/tmp/old.bin\");\n"
            '/* open("/tmp/block.bin"); */\n'
            'const char* other = "/var/tmp/not_ours.bin";\n',
            encoding="utf-8",
        )
        (root / "tests" / "dump_test.cpp").write_text('void t() { open("/tmp/x.bin"); }\n', encoding="utf-8")
        (root / "tools" / "run.sh").write_text(
            'LOG="${TMPDIR:-/tmp}/run.log"\n' "echo hi > /tmp/run2.log\n" "echo $# # a comment /tmp/nope\n",
            encoding="utf-8",
        )
        found = scan(root)
        check(found.get("npu/forward/dump.cpp") == ["/tmp/npu_layer.bin"],
              "a live literal is found and the two commented copies are not")
        check("/tmp/old.bin" not in str(found), "a commented-out path is not a site")
        check("tests/dump_test.cpp" not in found, "a test file is out of scope")
        check("/var/tmp/not_ours.bin" not in str(found), "/var/tmp is not /tmp")
        # engine#2027 review: the bare shell spelling, which the first pattern missed
        (root / "tools" / "bare.sh").write_text('echo y > /tmp/bare.log\ncp a /tmp/b2\n', encoding="utf-8")
        bare = scan(root)
        check(bare.get("tools/bare.sh") == ["/tmp/b2", "/tmp/bare.log"],
              "a bare /tmp path is a site (no quotes): the engine#2027 review's hole")
        check(all('"' not in s and "'" not in s for s in bare.get("tools/bare.sh", [])),
              "and the bare form is recorded without a quote it never had")

        # engine#2027 review: `/dev/shm` is the other RAM path the directive names, so it is the same site rule -- and
        # the anchor has to keep `/var/dev/shm` out for the same reason it keeps `/var/tmp` out.
        (root / "npu" / "forward" / "shm.cpp").write_text(
            'void h() { open("/dev/shm/buf.bin"); }\nconst char* o2 = "/var/dev/shm/not_ours.bin";\n',
            encoding="utf-8",
        )
        shm = scan(root)
        check(shm.get("npu/forward/shm.cpp") == ["/dev/shm/buf.bin"],
              "a /dev/shm literal is a site too, and /var/dev/shm is not")
        (root / "npu" / "forward" / "shm.cpp").unlink()

        # The ratchet: a new site fails, a stale baseline line fails, and an unchanged tree passes.
        write_baseline(root / "tools" / "ci" / "tmp_paths.txt", found)
        base = read_baseline(root / "tools" / "ci" / "tmp_paths.txt")
        have = {rel_literal(lit, rel) for rel, lits in found.items() for lit in lits}
        check(base == have, f"the baseline round-trips ({len(base)} site(s))")
        check(not (have - base) and not (base - have), "an unchanged tree has no drift")

        (root / "npu" / "forward" / "added.cpp").write_text('void g() { open("/tmp/new.bin"); }\n', encoding="utf-8")
        added = {rel_literal(lit, rel) for rel, lits in scan(root).items() for lit in lits}
        check(bool(added - base), "a new site is drift")
        check(not (base - added), "and adding one leaves no stale line")

        (root / "npu" / "forward" / "dump.cpp").unlink()
        renamed = {rel_literal(lit, rel) for rel, lits in scan(root).items() for lit in lits}
        check(bool(base - renamed), "a renamed file leaves a stale baseline line (the engine#1728 lesson)")

    print("tmp_path_check self-test:", "ok" if bad == 0 else f"{bad} FAILED")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
