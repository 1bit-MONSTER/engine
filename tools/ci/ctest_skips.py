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
"""ctest_skips.py -- make "green because it did not run" visible in a ctest log.

Two nights of reviewing turned up the same failure five times: a test that returns 77 without its fixture, a
registration above `find_package`, a row outside the guard CI configures, a `.py` nothing runs, an arm whose
`check(true, ...)` cannot fail. Each one is *green*, and each one is green because nothing executed.

`ctest` already knows: it prints `***Skipped` per skipped test and a `N% tests passed, 0 failed out of M` line that
counts skips as passes. This tool reads a ctest log and answers the one question a reviewer needs -- **which tests did
not run, and is that set the set we expect?** -- against a checked-in baseline.

    tools/ci/ctest_skips.py --log build/ctest.log                # report; exit 1 on a NEW skip
    tools/ci/ctest_skips.py --log build/ctest.log --write        # refresh the baseline
    tools/ci/ctest_skips.py --self-test

Exit: 0 every skip is in the baseline, 1 a test skipped that the baseline does not name (named), 2 cannot tell
(a log that exists but has no summary line), 77 the build never ran so there is no log at all (engine#2876).

The difference between 2 and 77 is the difference between "the suite ran and I cannot read it" and "the suite never
ran". The second is a consequence of an earlier failing step, and reporting it as a failure sent a reader down the
wrong path twice: the copyright step failing left no `build/ctest.log`, and the step below reported that as the
error.

The baseline is `tools/ci/ctest_skips.txt`: one test name per line with an optional `# reason`. An entry that no longer
skips is reported so the list shrinks instead of rotting -- the same shape as `unreferenced_tests.txt` (#1824).
"""

import argparse
import pathlib
import re
import sys

BASELINE = pathlib.Path("tools/ci/ctest_skips.txt")
# The repo this tool ships in: tools/ci/ctest_skips.py -> repo root. `--root` defaulted to "." until engine#2730's wiring, which made the
# answer depend on the caller's cwd -- running it from anywhere else reported every skipped test as unbaselined, i.e. a wrong
# `--root` looked exactly like a broken baseline. Every sibling tool resolves its root from __file__ for the same reason.
TOOL_ROOT = pathlib.Path(__file__).resolve().parents[2]
# ctest writes "did not run" in two shapes, and the second is the worse one:
#   " 16/192 Test  #16: npu_repack_path .......***Skipped   0.72 sec"     the test ran and returned 77
#   "\t182 - npu_private_route (Not Run)"                                  the binary was not built at all
# Only the first is a fixture deciding it cannot work here; the second is a build-list mistake (ci_targets_check's
# subject) and has no legitimate baseline entry.
SKIP_LINE = re.compile(r"^\s*\d+/\d+ Test\s+#\d+:\s+(\S+)\s+\.*\*\*\*Skipped", re.MULTILINE)
NOT_RUN_LINE = re.compile(r"^\s*\d+ - (\S+) \(Not Run\)", re.MULTILINE)
SUMMARY = re.compile(r"^\s*(\d+)% tests passed, (\d+) tests failed out of (\d+)")


def skipped(log_text: str):
    """{test name: 'skipped' or 'not-run'} -- every test that did not execute."""
    found = {name: "skipped" for name in SKIP_LINE.findall(log_text)}
    for name in NOT_RUN_LINE.findall(log_text):
        found[name] = "not-run"
    return found


def summary(log_text: str):
    """(failed, total) from ctest's summary line, or None when the log has no summary."""
    for line in log_text.splitlines():
        m = SUMMARY.match(line)
        if m:
            return int(m.group(2)), int(m.group(3))
    return None


def baseline(root: pathlib.Path):
    """{test name: reason} from the baseline file; comments and blanks skipped."""
    path = root / BASELINE
    out = {}
    if not path.is_file():
        return out
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition("#")
        name = name.strip()
        if name:
            out[name] = reason.strip()
    return out


def check(root: pathlib.Path, log_text: str) -> int:
    total = summary(log_text)
    if total is None:
        print("ctest_skips: cannot tell: the log has no 'N% tests passed' summary line", file=sys.stderr)
        return 2
    failed, count = total
    did_not_run = skipped(log_text)
    allow = baseline(root)
    new = sorted(n for n in did_not_run if n not in allow and did_not_run[n] == "skipped")
    not_run = sorted(n for n in did_not_run if did_not_run[n] == "not-run")
    gone = sorted(n for n in allow if n not in did_not_run)
    for name in not_run:
        print(f"ctest_skips: {name} did NOT RUN because its binary is missing -- that is a build-list mistake, not a "
              f"fixture decision: add the target to ci.yml's Build step (tools/ci/ci_targets_check.py is the same "
              f"class), and it needs no baseline entry")
    for name in new:
        print(f"ctest_skips: {name} SKIPPED and the baseline does not name it -- it did not run, and a test that does "
              f"not run cannot fail. Give it its fixture, move it where CI builds it, or add it to {BASELINE} with "
              f"the reason it cannot run here")
    for name in gone:
        print(f"ctest_skips: {name} is in {BASELINE} but it no longer skips -- drop the line, the list should shrink")
    if new or gone or not_run:
        return 1
    if did_not_run:
        print(f"ctest_skips: {len(did_not_run)} of {count} test(s) did not run, all named in the baseline "
              f"({failed} failed)")
    else:
        print(f"ctest_skips: all {count} test(s) ran ({failed} failed)")
    return 0


def _header(path: pathlib.Path) -> list[str]:
    """The file's own leading comment block, kept verbatim across a rewrite.

    `--write` regenerated the file from three hard-coded lines until engine#2750, which dropped the Apache notice every engine
    file carries, and it replaced every existing reason with a TODO placeholder. Both are the same mistake: a maintenance
    command that silently discards the evidence beside the names it keeps.
    """
    if not path.is_file():
        return ["# Tests that legitimately do not run in the CI configuration, one name per line with the reason.",
                "# Written by tools/ci/ctest_skips.py --write. A name here that stops skipping is reported, so the list",
                "# shrinks rather than rots (the shape of tools/ci/unreferenced_tests.txt, engine#1824).",
                "# A test whose BINARY is missing is never baselined here: that is a build-list mistake, not a fixture.",
                ""]
    head = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or not line.strip():
            head.append(line)
            continue
        break
    return head or [""]


def write_baseline(root: pathlib.Path, log_text: str, reason: str = "") -> int:
    did_not_run = skipped(log_text)
    # Only a test that RAN and returned 77 can be baselined. "Not Run" means the binary is missing, which is a build-list
    # mistake (ci_targets_check's subject) and would become invisible the moment this file named it.
    skipped_only = sorted(n for n, kind in did_not_run.items() if kind == "skipped")
    not_run = sorted(n for n, kind in did_not_run.items() if kind == "not-run")
    path = root / BASELINE
    known = baseline(root)
    new = [n for n in skipped_only if n not in known]
    # engine#2750: a NEW entry must carry the caller's reason. Writing a TODO placeholder and then passing the check is the
    # green tick over nothing this tool exists to remove -- the entry names a test nobody explained. Existing entries keep
    # the reason they already carry, so the routine refresh (`--write`) stays a single command.
    if new and not reason.strip():
        print(f"ctest_skips: {len(new)} test(s) would be baselined with no reason: {', '.join(new[:6])}"
              f"{' …' if len(new) > 6 else ''}", file=sys.stderr)
        print("ctest_skips: give the reason they cannot run here (--reason \"...\"), or give the test its fixture. "
              "Nothing was written.", file=sys.stderr)
        return 1
    lines = _header(path)
    for name in skipped_only:
        kept = known.get(name) or (reason.strip() if name in new else "")
        lines.append(f"{name}   # {kept}" if kept else name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ctest_skips: wrote {path} with {len(skipped_only)} skipped test(s)")
    if not_run:
        print(f"ctest_skips: {len(not_run)} test(s) did NOT RUN (binary missing) and are deliberately not baselined: "
              f"{', '.join(not_run[:6])}{' …' if len(not_run) > 6 else ''}", file=sys.stderr)
    return 0


def self_test() -> int:
    """A check that only ever sees a clean log proves nothing: drive all four verdicts."""
    import tempfile

    fails = 0

    def ok(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        fails += 0 if cond else 1

    clean = "100% tests passed, 0 tests failed out of 3\n"
    with_skip = (" 12/13 Test #12: npu_pack_import ..................***Skipped   0.01 sec\n"
                 "100% tests passed, 0 tests failed out of 13\n")
    with_notrun = ("\t182 - npu_private_route (Not Run)\n"
                   "100% tests passed, 0 tests failed out of 13\n")
    root = pathlib.Path(tempfile.mkdtemp())
    (root / BASELINE).parent.mkdir(parents=True, exist_ok=True)
    (root / BASELINE).write_text("# reasons\nnpu_pack_import   # needs a container\n")

    ok(not skipped(clean), "a log with no skip lines reports none")
    ok(list(skipped(with_skip)) == ["npu_pack_import"], "a '***Skipped'-style line is read")
    ok(check(root, clean) == 1, "a baseline entry that stopped skipping is drift (rc 1)")
    ok(check(root, with_skip) == 0, "a skip the baseline names is accepted (rc 0)")
    ok(summary(with_skip) == (0, 13), "the summary counts skips as passes, which is the point")
    (root / BASELINE).write_text("# empty\n")
    ok(check(root, with_skip) == 1, "a skip the baseline does NOT name is refused (rc 1)")
    ok(check(root, "no summary here\n") == 2, "a log without a summary is 'cannot tell' (rc 2)")
    ok(list(skipped(with_notrun)) == ["npu_private_route"], "a '(Not Run)' line is read")
    ok(check(root, with_notrun) == 1, "a missing binary is refused with no way to baseline it (rc 1)")
    ok(skipped(with_skip) != skipped(with_notrun), "... and the two shapes are not the same verdict")
    ok(_parser().parse_args([]).root == str(TOOL_ROOT),
       "with no --root the baseline is the repo's own, so a run from another cwd cannot read a stranger's")
    # engine#2750: --write must not baseline a name with a placeholder reason, and must keep the file's own header.
    (root / BASELINE).write_text("# Copyright notice that must survive\nnpu_pack_import   # a real reason\n")
    with_new = with_skip + " 14/15 Test #14: brand_new ....................***Skipped   0.01 sec\n" \
                          + "100% tests passed, 0 tests failed out of 15\n"
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc_no_reason = write_baseline(root, with_new)
    ok(rc_no_reason == 1, "a new name with no --reason is refused (rc 1)")
    ok("brand_new" not in (root / BASELINE).read_text(), "... and nothing was written")
    ok("brand_new" in err.getvalue() and "--reason" in err.getvalue(),
       "... the refusal names the test and the flag that would allow it")
    ok(write_baseline(root, with_new, "needs a container") == 0, "with --reason the write is allowed")
    text = (root / BASELINE).read_text()
    ok("brand_new   # needs a container" in text, "... the caller's reason is the one recorded")
    ok("npu_pack_import   # a real reason" in text, "... an existing entry keeps its own reason")
    ok(text.startswith("# Copyright notice that must survive"), "... the header block survives a rewrite")
    print("self-test ok" if not fails else f"self-test FAILED ({fails})")
    return 1 if fails else 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default="build/ctest.log", help="a ctest log (default: build/ctest.log)")
    ap.add_argument("--root", default=str(TOOL_ROOT),
                    help="the tree that holds %s (default: the repo this tool ships in, never the caller's cwd)" % BASELINE)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--reason", default="",
                    help="why the tests a --write would ADD cannot run here (required for a new name)")
    ap.add_argument("--self-test", action="store_true")
    return ap


def main() -> int:
    a = _parser().parse_args()
    if a.self_test:
        return self_test()
    path = pathlib.Path(a.log)
    if not path.is_file():
        # No log at all means the build or the test step never ran: an earlier step already failed and
        # said why, so this one must not add a second, misleading error. 77 is the skip code the rest of
        # tools/ci uses for "the precondition was not there".
        print(
            f"ctest_skips: skipped: no log at {path}, so the suite never ran -- look at the step that "
            f"failed before this one (engine#2876)",
            file=sys.stderr,
        )
        return 77
    text = path.read_text(encoding="utf-8", errors="replace")
    root = pathlib.Path(a.root).resolve()
    return write_baseline(root, text, a.reason) if a.write else check(root, text)


if __name__ == "__main__":
    sys.exit(main())
