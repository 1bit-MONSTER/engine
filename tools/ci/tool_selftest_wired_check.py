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

Which tools are "wired" is deliberately the cheap question: the file's **basename** appears in
`CMakeLists.txt` (as an `add_python_test` row) or in `scripts/pre-pr-checks.sh` (as a `checks=(…)`
entry). Both are the two places a self-test can be run from, and either counts. A name that appears in
neither is reported with its path.

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


def references(root: pathlib.Path, sources=REFERENCE_SOURCES) -> str | None:
    """The concatenated text of every reference source, or None when one cannot be read."""
    parts = []
    paths = [root / rel for rel in sources]
    for g in REFERENCE_GLOBS:
        paths += sorted(root.glob(g))
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        # A COMMENT is not a wiring: this file's own fix mentions the four tools it wired, and a mention in
        # prose would otherwise mark a tool wired the moment someone wrote a note about it. Only code counts.
        parts.append("\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#")))
    return "\n".join(parts)


def reference_lines(refs: str) -> list[str]:
    """The reference sources, one line at a time, comments dropped.

    engine#3216: the rule is about a LINE, not about the concatenated text. A tool named by one row and mentioned
    beside `--self-test` in a different one is not wired: `registry_build.py` had `--check-gaps`, `--check-pins` and
    `--check` rows and an eight-arm `--self-test` that nothing invoked, and the whole-file match called it wired.
    """
    return [l for l in refs.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def named_beside_marker(tool: pathlib.Path, lines: list[str]) -> bool:
    """True when some reference line names this tool AND carries a self-test marker."""
    name = tool.name
    # A reference is a path, so the character before the basename is usually `/`: the lookbehind must not
    # exclude it. It must exclude `_`, `.` and `-`, so `pre_check.py` is not read as a reference to `check.py`.
    rx = re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])")
    return any(rx.search(l) and MARKER_RE.search(l) for l in lines)


def unwired(tools: list[pathlib.Path], refs: str) -> list[pathlib.Path]:
    """Tools whose arms nothing runs: no reference LINE names them beside a self-test marker.

    `--self-test` alone is not a reference: the tool's own flag spelling appears in every file that has one, so
    matching the marker anywhere would mark the whole tree wired. The marker has to sit on the line that names the
    tool (engine#3216).
    """
    lines = reference_lines(refs)
    return [t for t in tools if not named_beside_marker(t, lines)]


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

        refs = references(root)
        got = sorted(str(p.relative_to(root)) for p in unwired(tools, refs))
        chk(["tools/diag/orphan.py", "tools/selfonly.py"], got,
            "a tool no source names is reported (and a tool that only spells the flag is NOT wired)")

        # engine#3216: a row for ANOTHER subcommand is not a wiring. The live case was `registry_build.py` with
        # `--check-gaps`/`--check-pins`/`--check` rows and an eight-arm `--self-test` nothing invoked, reported wired.
        chk([], unwired([pathlib.Path("registry_build.py")],
                        "add_python_test(registry_build_self tools/registry_build.py --self-test)"),
            "a tool named beside a --self-test is wired")
        chk([pathlib.Path("registry_build.py")],
            unwired([pathlib.Path("registry_build.py")],
                    "add_test(NAME registry_gaps COMMAND tools/registry_build.py --check-gaps)"),
            "a tool named ONLY for another flag is unwired (engine#3216)")

        # engine#2675 review: `.github/workflows/` is a fourth place, and it is the one where two of the four tools
        # already ran. A `run:` line is a wiring; a YAML comment naming the tool is not, exactly as for CMakeLists.
        wf = root / ".github" / "workflows"
        wf.mkdir(parents=True)
        # The `run:` line names the one tool that is otherwise unwired, so a reference set WITHOUT the workflows
        # cannot produce the expected answer -- an arm whose fixture is already wired elsewhere does not test this.
        (wf / "ci.yml").write_text("jobs:\n  t:\n    steps:\n      - run: python3 tools/diag/orphan.py --self-test\n")
        (wf / "ci2.yaml").write_text("# tools/selfonly.py --self-test   <- a YAML comment, not a wiring\n")
        got = sorted(str(p.relative_to(root)) for p in unwired(discover(root), references(root)))
        chk(["tools/selfonly.py"], got,
            "a workflow `run:` line counts as a wiring, and a workflow comment still does not")

        (root / "CMakeLists.txt").unlink()
        chk(None, references(root), "a reference source that cannot be read is None, not an empty string")

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
    refs = references(root)
    if refs is None:
        print("tool_selftest_wired_check: cannot read a reference source (CMakeLists.txt or "
              "scripts/pre-pr-checks.sh): a scan that could not look is not a clean scan", file=sys.stderr)
        return 2
    bad = unwired(tools, refs)
    if a.list:
        for t in tools:
            print(f"{'UNWIRED' if t in bad else 'wired  '} {t.relative_to(root)}")
        return 0
    if bad:
        print(f"tool_selftest_wired_check: {len(bad)} of {len(tools)} self-test-carrying tool(s) are named by "
              f"neither CMakeLists.txt nor scripts/pre-pr-checks.sh, so nothing ever runs their arms:", file=sys.stderr)
        for t in bad:
            print(f"  {t.as_posix()}  (wire it: add_python_test(<name> ${{CMAKE_SOURCE_DIR}}/{t.as_posix()} --self-test))",
                  file=sys.stderr)
        return 1
    print(f"tool_selftest_wired_check: {len(tools)} self-test-carrying tool(s), all referenced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
