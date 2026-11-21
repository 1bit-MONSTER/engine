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
"""tool_selftest_wired_check.py — a tool with a self-test that nothing runs is a tool with no arms.

`tools/ci/tests_referenced_check.py` asks the same question about `tests/*.py`: a test file that no
ctest row, no script and no docs reference is a test nobody runs. The tool side of the tree has the same
hole and no check: measured on 2026-11-16, **71 tools carry a self-test and 4 of them appeared nowhere**
(`tools/template_bom.py`, `tools/moe_q4nx_ref.py`, `tools/pin_watch.py`, `tools/diag/iommu_fault_census.py`)
— each one's arms had been written, passed, and were never executed again. A self-test that is not wired
rots silently: it keeps printing the arms it had the day it was written, and the day the code it guards
changes, nothing says so.

Which tools are "wired" is deliberately the cheap question, but it is two conditions, not one: a reference
line must name the file's **basename** *and* carry a self-test marker on that same line. engine#3216 is why
the second half exists — matching the basename alone made `tools/registry_build.py` read as wired because
`CMakeLists.txt` names it for `--check-gaps`, while the eight arms #3214 added ran nowhere and this row
printed green. A tool named only for another flag is now reported separately from a tool named nowhere,
because the fix differs (add a `--self-test` row) and the class is exactly the one that hid.

Exit 0 when every self-test-carrying tool is referenced, 1 when one is not, 2 when the reference
sources cannot be read — a scan that could not look is not a clean scan.

  tools/ci/tool_selftest_wired_check.py [--root .] [--list]
  tools/ci/tool_selftest_wired_check.py --self-test
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

# FOUR places a self-test can be run from, and review of engine#2675 found two I had missed:
#   cmake/*.cmake            -- engine#2456 already put four such rows in cmake/tests-unreferenced-selftest-rows.cmake
#   .github/workflows/*.yml  -- ci.yml RUNS `template_bom.py --self-test` and `pin_watch.py --self-test` on every
#                               push, in a `run:` block. A checker that reads only CMakeLists.txt and the pre-pr list
#                               reports them unwired, which is a false positive on the class this file exists for.
# The tuple below was `("CMakeLists.txt", "scripts/pre-pr-checks.sh")` on the first pass: engine#2456
# already put four such rows in `cmake/tests-unreferenced-selftest-rows.cmake`, so a checker that read only
# CMakeLists.txt and the pre-pr list reported them as unwired when they had been wired for a day.
REFERENCE_SOURCES = ("CMakeLists.txt", "scripts/pre-pr-checks.sh")
REFERENCE_GLOBS = ("cmake/*.cmake", ".github/workflows/*.yml", ".github/workflows/*.yaml")
# The three spellings the tree uses. Kept as a tuple so the self-test can exercise one at a time.
SELFTEST_MARKERS = ("--self-test", "--selftest", "selftest")
MARKER_RE = re.compile(r"(?<![\w-])(" + "|".join(re.escape(m) for m in SELFTEST_MARKERS) + r")(?![\w-])")


def carries_self_test(path: pathlib.Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(MARKER_RE.search(text))


def discover(root: pathlib.Path, tool_dirs=("tools",)) -> list[pathlib.Path]:
    found = []
    for d in tool_dirs:
        base = root / d
        if not base.is_dir():
            continue
        found += sorted(p for p in base.rglob("*.py") if carries_self_test(p))
    return found


def reference_lines(root: pathlib.Path, sources=REFERENCE_SOURCES) -> list[tuple[str, str]] | None:
    """Every non-comment line of every reference source, as `(source, line)`; None when one cannot be read.

    Lines rather than one blob since engine#3216: whether a line WIRES a tool depends on the tool's name and a
    self-test marker being on the *same* line, which a concatenated blob cannot answer.
    """
    out: list[tuple[str, str]] = []
    paths = [root / rel for rel in sources]
    for g in REFERENCE_GLOBS:
        paths += sorted(root.glob(g))
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        # A COMMENT is not a wiring: this file's own fix mentions the tools it wired, and a mention in prose
        # would otherwise mark a tool wired the moment someone wrote a note about it. Only code counts.
        for line in text.splitlines():
            if not line.lstrip().startswith("#"):
                out.append((str(p.relative_to(root)), line))
    return out


def _name_in(name: str, line: str) -> bool:
    # A reference is a path, so the character before the basename is usually `/`: the lookbehind must not
    # exclude it. It must exclude `_`, `.` and `-`, so `pre_check.py` is not read as a reference to `check.py`.
    return bool(re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])", line))


def classify(tools: list[pathlib.Path], lines: list[tuple[str, str]]):
    """-> (wired, named_for_another_flag, unwired). `unwired` keeps this function's original output.

    wired                 some line names the tool AND carries a self-test marker: something runs its arms.
    named_for_another_flag some line names it, none carries a marker -- the engine#3216 class, where a tool
                          wired for `--check-gaps` looked wired while its `--self-test` ran nowhere.
    unwired               no line names it at all.
    `--self-test` alone is not a wiring either: the tool's own flag spelling appears in every file that has
    one, so marker-only matching would mark the whole tree wired.
    """
    wired, other, missing = [], [], []
    for t in tools:
        hits = [(src, line) for src, line in lines if _name_in(t.name, line)]
        if not hits:
            missing.append(t)
        elif any(MARKER_RE.search(line) for _src, line in hits):
            wired.append(t)
        else:
            other.append((t, hits))
    return wired, other, missing


def unwired(tools: list[pathlib.Path], lines: list[tuple[str, str]]) -> list[pathlib.Path]:
    """Tools no reference line names at all (the class this check has always reported)."""
    return classify(tools, lines)[2]


def self_test() -> int:
    import tempfile

    rc = 0

    def chk(want, got, what):
        nonlocal rc
        if want == got:
            print(f"ok   {what}")
        else:
            print(f"FAIL {what}: want {want}, got {got}")
            rc = 1

    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "tools" / "diag").mkdir(parents=True)
        (root / "scripts").mkdir()
        # three tools with an entry point: one wired by CMake, one by the pre-pr list, one by nothing.
        # A fourth only DEFINES `def self_test()` and has no way to run it, so it must not be discovered at all:
        # a self-test with no entry point is not an arm anything can execute.
        (root / "tools" / "by_cmake.py").write_text('if "--self-test" in sys.argv: pass\n')
        (root / "tools" / "by_prepr.py").write_text('if "--selftest" in sys.argv: pass\n')
        (root / "tools" / "diag" / "orphan.py").write_text('if "--self-test" in sys.argv: pass\n')
        (root / "tools" / "only_a_function.py").write_text('def self_test(): pass\n')
        # and one whose only mention of the flag is its own definition -- not a wiring
        (root / "tools" / "selfonly.py").write_text('SPELLING = "--self-test"\n')
        (root / "CMakeLists.txt").write_text("add_python_test(x ${CMAKE_SOURCE_DIR}/tools/by_cmake.py --self-test)\n")
        (root / "scripts" / "pre-pr-checks.sh").write_text('checks=(\n    "tools/by_prepr.py --selftest"\n)\n')

        tools = discover(root)
        names = sorted(p.name for p in tools)
        chk(["by_cmake.py", "by_prepr.py", "orphan.py", "selfonly.py"], names,
            "four tools carry a runnable self-test (and one that only defines the function does not)")

        lines = reference_lines(root)
        got = sorted(str(p.relative_to(root)) for p in unwired(tools, lines))
        chk(["tools/diag/orphan.py", "tools/selfonly.py"], got,
            "a tool no source names is reported (and a tool that only spells the flag is NOT wired)")

        # engine#2675 review: `.github/workflows/` is a fourth place, and it is the one where two of the four tools
        # already ran. A `run:` line is a wiring; a YAML comment naming the tool is not, exactly as for CMakeLists.
        wf = root / ".github" / "workflows"
        wf.mkdir(parents=True)
        # The `run:` line names the one tool that is otherwise unwired, so a reference set WITHOUT the workflows
        # cannot produce the expected answer -- an arm whose fixture is already wired elsewhere does not test this.
        (wf / "ci.yml").write_text("jobs:\n  t:\n    steps:\n      - run: python3 tools/diag/orphan.py --self-test\n")
        (wf / "ci2.yaml").write_text("# tools/selfonly.py --self-test   <- a YAML comment, not a wiring\n")
        got = sorted(str(p.relative_to(root)) for p in unwired(discover(root), reference_lines(root)))
        chk(["tools/selfonly.py"], got,
            "a workflow `run:` line counts as a wiring, and a workflow comment still does not")

        # engine#3216: the registry_build.py shape -- the name is on a line, the marker is not. Before this
        # change that tool read as wired; the arm below fails if it ever does again.
        (root / "CMakeLists.txt").write_text(
            "add_python_test(x ${CMAKE_SOURCE_DIR}/tools/by_cmake.py --self-test)\n"
            "add_test(NAME gaps COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/flag_only.py --check-gaps)\n")
        (root / "tools" / "flag_only.py").write_text('if "--self-test" in sys.argv: pass\nif "--check-gaps" in sys.argv: pass\n')
        wired, other, missing = classify(discover(root), reference_lines(root))
        chk(["flag_only.py"], [t.name for t, _hits in other],
            "a tool named only for ANOTHER flag is its own class, not wired (engine#3216)")
        chk(True, "by_cmake.py" in [t.name for t in wired], "...and a name WITH a marker on the line is wired")
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["--root", str(root), "--list"])
        chk(True, "NO-MARKER  tools/flag_only.py" in buf.getvalue(),
            "...and --list shows it as NO-MARKER, so a lane can see why the row is red")

        (root / "CMakeLists.txt").unlink()
        chk(None, reference_lines(root), "a reference source that cannot be read is None, not an empty string")

    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="tree to scan (default: .)")
    ap.add_argument("--list", action="store_true", help="list the wired tools too, then exit 0")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    root = pathlib.Path(a.root)
    tools = discover(root)
    lines = reference_lines(root)
    if lines is None:
        print("tool_selftest_wired_check: cannot read a reference source (CMakeLists.txt or "
              "scripts/pre-pr-checks.sh): a scan that could not look is not a clean scan", file=sys.stderr)
        return 2
    wired, other, bad = classify(tools, lines)
    if a.list:
        for t in tools:
            state = "wired      " if t in wired else ("NO-MARKER  " if any(t is o for o, _ in other) else "UNWIRED    ")
            print(f"{state}{t.relative_to(root)}")
        return 0
    rc = 0
    if bad:
        print(f"tool_selftest_wired_check: {len(bad)} of {len(tools)} self-test-carrying tool(s) are named by "
              f"neither CMakeLists.txt nor scripts/pre-pr-checks.sh, so nothing ever runs their arms:", file=sys.stderr)
        for t in bad:
            print(f"  {t.as_posix()}  (wire it: add_python_test(<name> ${{CMAKE_SOURCE_DIR}}/{t.as_posix()} --self-test))",
                  file=sys.stderr)
        rc = 1
    if other:
        # engine#3216: the class that hid -- a line names the tool, but for a different subcommand, so the
        # arms are written and nothing runs them. Reported on its own line because the fix is one row, not a
        # rewrite, and because "named" and "run" were the two things this check used to confuse.
        print(f"tool_selftest_wired_check: {len(other)} of {len(tools)} self-test-carrying tool(s) are named only "
              f"for ANOTHER flag, so their arms never run:", file=sys.stderr)
        for t, hits in other:
            print(f"  {t.as_posix()}  (named at {hits[0][0]} but with no --self-test there; wire it: "
                  f"add_python_test(<name> ${{CMAKE_SOURCE_DIR}}/{t.as_posix()} --self-test))", file=sys.stderr)
        rc = 1
    if rc:
        return 1
    print(f"tool_selftest_wired_check: {len(tools)} self-test-carrying tool(s), every one wired by a line that runs it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
