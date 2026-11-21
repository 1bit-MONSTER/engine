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


def references(root: pathlib.Path, sources=REFERENCE_SOURCES) -> list[str] | None:
    """The non-comment lines of every reference source, or None when one cannot be read.

    Lines rather than one joined blob, because engine#3216 has to ask whether a self-test MARKER is on the same
    line as the tool's name: a concatenated string cannot tell `foo.py --check` from `foo.py --check` followed on
    the next line by `bar.py --self-test`.
    """
    parts: list[str] = []
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
        parts += [l for l in text.splitlines() if not l.lstrip().startswith("#")]
    return parts


def name_re(name: str) -> re.Pattern:
    """A reference is a path, so the character before the basename is usually `/`: the lookbehind must not exclude
    it. It must exclude `_`, `.` and `-`, so `pre_check.py` is not read as a reference to `check.py`."""
    return re.compile(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])")


def classify(tools: list[pathlib.Path],
             refs: list[str]) -> tuple[list[pathlib.Path], list[pathlib.Path], list[pathlib.Path]]:
    """Split the tools into (wired, named for another flag, named by nothing).

    engine#3216: a NAME is not a wiring. `registry_build.py` was named in `CMakeLists.txt` for `--check-gaps` and
    `--check-pins`, so this checker called it wired while its eight self-test arms ran nowhere — the rule was "the
    basename appears somewhere", so the subcommand that actually runs the arms was never looked at. A tool is wired
    only when one reference LINE names it **and** carries a self-test marker **after** that name:

      `add_python_test(x ${CMAKE_SOURCE_DIR}/tools/foo.py --self-test)`          wired
      `add_test(NAME y COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/foo.py --check)` named for another flag
      `add_test(NAME z COMMAND tools/bar.py --self-test tools/foo.py)`            named for another flag — the
                                                                                 marker belongs to bar.py

    The after-the-name half is what the third case needs, and it is free on this tree: measured 2026-11-21 over the
    94 self-test-carrying tools, both spellings give 91 wired / 3 named-for-another-flag / 0 named by nothing.
    """
    wired, named_other, missing = [], [], []
    for t in tools:
        rx = name_re(t.name)
        pairs = [(l, m) for l in refs for m in rx.finditer(l)]
        if not pairs:
            missing.append(t)
        elif any(MARKER_RE.search(l, m.end()) for l, m in pairs):
            wired.append(t)
        else:
            named_other.append(t)
    return wired, named_other, missing


def unwired(tools: list[pathlib.Path], refs: list[str]) -> list[pathlib.Path]:
    """Tools no reference source names at all.

    `--self-test` alone is not a reference: the tool's own flag spelling appears in every file that has one, so
    matching it would mark the whole tree wired. A tool that IS named, but only for another subcommand, is the
    separate class above: tightening this one into a hard failure would red three rows the day it landed, so the
    count is reported instead — visible, and not a wall on day one.
    """
    return classify(tools, refs)[2]


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
        # engine#3216: two more tools WITH an entry point that are named by a reference line but not wired --
        # `by_other_flag.py` is named for a different subcommand, and `marker_before.py` shares a line whose marker
        # belongs to another tool. Both must land in the "named for another flag" class, not in `unwired` and not
        # in `wired`; the second is what the after-the-name half of the rule exists for.
        (root / "tools" / "by_other_flag.py").write_text('if "--self-test" in sys.argv: pass\n')
        (root / "tools" / "marker_before.py").write_text('if "--self-test" in sys.argv: pass\n')
        (root / "tools" / "marker_owner.py").write_text('if "--self-test" in sys.argv: pass\n')
        (root / "CMakeLists.txt").write_text(
            "add_python_test(x ${CMAKE_SOURCE_DIR}/tools/by_cmake.py --self-test)\n"
            "add_test(NAME other COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/by_other_flag.py --check)\n"
            "add_test(NAME owner COMMAND tools/marker_owner.py --self-test tools/marker_before.py)\n")
        (root / "scripts" / "pre-pr-checks.sh").write_text('checks=(\n    "tools/by_prepr.py --selftest"\n)\n')

        tools = discover(root)
        names = sorted(p.name for p in tools)
        # the fixture's tool set is a literal (engine#3216 added three: two for the named-for-another-flag class and
        # one to own the marker in the shared-line arm), so this arm names all seven rather than counting them --
        # a count would agree with itself after any edit that adds one tool and removes another.
        chk(["by_cmake.py", "by_other_flag.py", "by_prepr.py", "marker_before.py", "marker_owner.py", "orphan.py",
             "selfonly.py"], names,
            "seven tools carry a runnable self-test (and the one that only defines the function does not)")

        refs = references(root)
        got = sorted(str(p.relative_to(root)) for p in unwired(tools, refs))
        chk(["tools/diag/orphan.py", "tools/selfonly.py"], got,
            "a tool no source names is reported (and a tool that only spells the flag is NOT wired)")

        # engine#3216, the arm for the class that was invisible: a name without a marker beside it is not a wiring.
        wired, named_other, _ = classify(tools, refs)
        chk(["by_cmake.py", "by_prepr.py", "marker_owner.py"], sorted(p.name for p in wired),
            "a name WITH a self-test marker after it on the same line is wired (three of the six)")
        chk(["by_other_flag.py", "marker_before.py"], sorted(p.name for p in named_other),
            "a tool named only for another flag is reported as such, and a marker belonging to another tool on the "
            "same line does not vouch for it")
        # ... and the class must be counted without failing the run: the fixture's only unwired tool is selfonly.py.
        chk(0, sum(1 for t in named_other if t.name in ("selfonly.py", "only_a_function.py")),
            "a tool named for another flag is not silently promoted into the unwired class")

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
    wired, named_other, bad = classify(tools, refs)
    if a.list:
        for t in tools:
            cls = "UNWIRED " if t in bad else "OTHERFLG" if t in named_other else "wired   "
            print(f"{cls} {t.relative_to(root)}")
        return 0
    # engine#3216: a tool named only for another subcommand is reported, counted and NOT fatal on its own -- the
    # three rows that shape today were each wired for a `--check`, which is a real reference to the file and not a
    # wiring of its arms, and turning that into a red row the day the rule tightened would have taught lanes to
    # silence the check rather than wire the tool.
    if named_other:
        print(f"tool_selftest_wired_check: {len(named_other)} of {len(tools)} self-test-carrying tool(s) are named "
              f"only for another flag, so their arms still run nowhere (reported, not failed):", file=sys.stderr)
        for t in named_other:
            print(f"  {t.as_posix()}  (wire it: add_python_test(<name> ${{CMAKE_SOURCE_DIR}}/{t.as_posix()} --self-test))",
                  file=sys.stderr)
    if bad:
        print(f"tool_selftest_wired_check: {len(bad)} of {len(tools)} self-test-carrying tool(s) are named by "
              f"neither CMakeLists.txt nor scripts/pre-pr-checks.sh, so nothing ever runs their arms:", file=sys.stderr)
        for t in bad:
            print(f"  {t.as_posix()}  (wire it: add_python_test(<name> ${{CMAKE_SOURCE_DIR}}/{t.as_posix()} --self-test))",
                  file=sys.stderr)
        return 1
    print(f"tool_selftest_wired_check: {len(tools)} self-test-carrying tool(s): {len(wired)} wired, "
          f"{len(named_other)} named for another flag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
