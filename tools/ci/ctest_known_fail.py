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
"""ctest_known_fail.py -- make "red, but only in ways we have written down" visible in a ctest log.

`ctest_skips.py` answers *did anything fail to run?* This answers the next question, which the fleet
needed the moment a suite went red at a pinned fork (engine#2761: 15 of 112 fail at `52dda9bc`, 13 of
them serially, and nothing in CI runs the suite at all). Turning such a job on is not possible while
any red fails it, and leaving it off keeps the fleet green while the suite is red. The ratchet makes
the job shippable: **the failures we have are written down, and a NEW one fails the job.**

    tools/ci/ctest_known_fail.py --log build/ctest.log            # report; exit 1 on a NEW failure
    tools/ci/ctest_known_fail.py --log build/ctest.log --write    # refresh the baseline
    tools/ci/ctest_known_fail.py --self-test

Exit: 0 every failure is in the baseline, 1 a test failed that the baseline does not name (named), 2
cannot answer (no log, or no summary line).

The baseline is `tools/ci/ctest_known_fail.txt`: one test name per line with an optional `# reason`
and the *reason ctest gave* after it, because `(SEGFAULT)` and `(Failed)` are different bugs and a
baseline that loses that distinction will be read as one. A name in the baseline that now passes is
reported, so the list shrinks instead of rotting -- the same shape as `ctest_skips.txt` and
`unreferenced_tests.txt`.

The baseline is a record of debt, not a permission slip: an entry is a test the suite cannot pass at
the fork the tree pins, and it says so in the reason. Anything else belongs in the suite.

Relational: `ctest_skips.txt` is the "did not run" list for the same suite. Both files are read by
`.github/workflows/ci.yml`; a test may appear in one or the other, never both, because a test that
neither ran nor failed is filed under the reason it did not get the chance.

`--root` defaults to the tree this file lives in, resolved from `__file__`. A tool whose answer
depends on the caller's cwd reports every known failure as a new one the moment it is run from
elsewhere, which reads exactly like a broken baseline (ctest_skips.py, 2026-11-15).
"""

import argparse
import pathlib
import re
import shutil
import sys
import tempfile

BASELINE = pathlib.Path("tools/ci/ctest_known_fail.txt")
def _tool_root() -> pathlib.Path:
    """The tree this tool ships in, or the caller's cwd when it has been copied somewhere else.

    `pathlib.Path(__file__).resolve().parents[2]` raises IndexError for any copy shallow enough to lack three
    parents (a copy dropped in a scratch directory, say), which is how a relocated copy died before printing anything -- and a raw IndexError
    reads as a broken tool rather than "this copy is not inside a tree". Collapsing every mutation-test run on this
    file into that crash is what hid the arms for an hour (engine#2825 review, mine).
    """
    here = pathlib.Path(__file__).resolve()
    return here.parents[2] if len(here.parents) > 2 else pathlib.Path.cwd()


TOOL_ROOT = _tool_root()
# ctest writes the failure list in two places and the second one has the reason:
#   "\t 18 - test-tokenizers-ggml-vocabs (Failed)"          the numbered list after "The following tests FAILED:"
#   "\t101 - test-hrx-mul-mat-id-k32 (Subprocess aborted)"
# A name can appear in the per-test progress lines too ("***Failed"), but those carry no reason and
# are a strictly smaller set, so the FAILED block alone is both sufficient and the honest source.
# engine#2836: ctest pads a LABELS column after the parens (`\t 85 - test-hrx-buffer (Failed)     hrx`), and
# requiring end-of-line after `)` dropped exactly those rows -- the count then disagreed with the summary and
# every labelled failure read as a truncated log ("cannot tell", rc 2). A label can be several words
# (`main behavior backend`), so everything after the parens is ignored, not one token.
FAILED_LINE = re.compile(r"^\s*\d+ - (\S+) \((Failed|SEGFAULT|Subprocess aborted|Timeout|[A-Z][^)]*)\)(?:[ \t]+.*)?$",
                         re.MULTILINE)
# engine#2825 review (mine): NO re.MULTILINE meant `^` matched only at position 0, and a real ctest log puts
# its summary at the END, after the FAILED block -- so summary() returned None for every real log and check()
# always answered "cannot tell". The self-test stayed green because its fixture puts the summary FIRST, which
# is the one shape ctest never emits. The arm below is the real shape, and it fails without this flag.
SUMMARY = re.compile(r"^\s*(\d+)% tests passed, (\d+) tests failed out of (\d+)", re.MULTILINE)
# "You have 3 tests; 2 are known failures" is a state, not an error -- but a summary that says N
# failed while the FAILED block lists fewer is a truncated log, and answering from it would hide the
# missing ones. Both numbers are read and compared rather than trusting either alone.
# A comment runs from a `#` at the start of the line too, not only from one preceded by whitespace.
# The first version required \s+ and so parsed every column-0 `#` header line as a test name; running
# the real workflow step shape against the real baseline reported the whole licence block as
# "now passing, remove from".
# engine#2825 review (mine): this line used to add "`ctest_skips.py` has the same regex and the same latent
# defect", which is not true of the code as it stands -- that parser strips the line and tests startswith("#"),
# so a column-0 licence block is handled there. Measured: baseline() over a file whose first two lines are
# `# Copyright` and `# SPDX-...` returns only the real entry. A note that sends the next reader hunting a bug
# that is not there costs more than no note.
COMMENT = re.compile(r"^\s*#.*$|\s+#.*$")
# ctest colourises its summary when the log is captured from a terminal, so the byte before "5%" can be
# an ANSI SGR sequence. `^\s*` cannot match that, and the first CI run of this tool answered "cannot
# tell: no summary line" from a log that plainly contained "5% tests passed, 106 tests failed out of
# 112" -- the line was there and the reader could not see it. A log parser has to tolerate colour:
# strip the escapes before matching rather than trusting the producer to have disabled them.
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def failed(log_text: str):
    """{name: reason} from ctest's FAILED block -- FAILURES only, never the skip list (engine#2825 review, mine).

    The catch-all alternative in FAILED_LINE (`[A-Z][^)]*`) was written for reason strings ctest may add, and it also
    matched `(Skipped)`. ctest prints a skipped-tests section in the same `  318 - name (Skipped)` shape, so every
    skipped row counted as a failure: measured on a real 383-test log the summary said `3 tests failed out of 383`
    while this function returned 16 names (3 failures + 13 skips), and check() then refused EVERY real log with
    "one of the two is truncated" -- the ratchet could never answer, which is the failure it exists to prevent.
    """
    log_text = ANSI.sub("", log_text)
    out = {}
    for name, reason in FAILED_LINE.findall(log_text):
        if reason.split()[0].lower() in ("skipped", "disabled"):
            continue          # a row that did not run is not a failure and must never enter the ratchet
        # `Not Run` is NOT in that list, on purpose (engine#2836): ctest puts it in the FAILED block and COUNTS it
        # in "M tests failed", so dropping it made count != len(names) and every log with an unbuilt binary read
        # as truncated -- "cannot tell", rc 2, the failure this tool exists to prevent. The baseline carries
        # `test-hrx-host-resident # Not Run` for exactly this reason: an unbuilt row is a failure to fix, not a skip.
        out[name] = reason
    return out


def summary(log_text: str):
    """(failed_count, total) from ctest's last summary line, or None."""
    hits = SUMMARY.findall(ANSI.sub("", log_text))
    if not hits:
        return None
    return int(hits[-1][1]), int(hits[-1][2])


def baseline(root: pathlib.Path):
    """{test name: reason} from the checked-in baseline."""
    path = root / BASELINE
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = COMMENT.sub("", line).strip()
        if line:
            out[line.split()[0]] = line
    return out


def check(log_text: str, known, want_total=None):
    """(new_failures, fixed, cannot) -- cannot is a string when the log cannot answer at all."""
    if not log_text.strip():
        return set(), set(), "the log is empty"
    sums = summary(log_text)
    if sums is None:
        return set(), set(), "no 'N% tests passed, M tests failed out of T' line -- a partial log or no ctest run"
    count, total = sums
    if want_total is not None and total != want_total:
        return set(), set(), (f"the summary says {total} tests, expected {want_total} -- this log is from a "
                             f"different suite, so its failures cannot be compared with the baseline")
    names = failed(log_text)
    if count and not names:
        return set(), set(), (f"the summary says {count} failed but the FAILED block lists none -- "
                             f"truncated output; rerun with the block intact")
    if count != len(names):
        return set(), set(), (f"the summary says {count} failed, the FAILED block lists {len(names)} -- "
                             f"one of the two is truncated")
    new = {n for n in names if n not in known}
    fixed = {n for n in known if n not in names}
    return new, fixed, None


def write_baseline(path: pathlib.Path, names, log_text: str):
    """Rewrite the baseline by MERGING, never regenerating (engine#2825 review).

    Regenerating from `failed(log_text)` alone deleted two things the file must keep: the Apache block, which
    `tools/copyright.py --check` requires and which this file is registered in LINE_STYLES for, and the hand-written
    annotations (`-- behavior`, `-- main backend`, the measurement note) that a reader uses to judge whether an entry
    is still true. Following the file's own documented regeneration command therefore broke CI, and the self-test
    asserted the opposite ("a round-trip through baseline() loses nothing") because only `baseline()` was armed.

    So: keep the leading comment block verbatim, keep every surviving name's existing line verbatim, and emit a
    fresh `<name>    # <reason>` only for names the file did not already carry.
    """
    names = sorted(names or [])
    reasons = failed(log_text)
    existing = {}
    header = []
    if path.exists():
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                if not existing:          # still inside the leading block
                    header.append(line)
                continue
            existing[line.split()[0]] = line
    while header and not header[-1].strip():
        header.pop()
    body = list(header)
    if body:
        body.append("")
    for name in names:
        body.append(existing.get(name) or f"{name}    # {reasons.get(name, '')}".rstrip())
    path.write_text("\n".join(body) + "\n")


def self_test():
    import shutil, tempfile   # bound here because a later local import makes the name local to this function
    cases = 0
    bad = 0
    # The suite this baseline exists for, as engine#2761 reported it.
    real = """\
97% tests passed, 4 tests failed out of 112
The following tests FAILED:
\t 18 - test-tokenizers-ggml-vocabs (Failed)
\t 19 - test-kv-zero (Subprocess aborted)
\t 37 - test-speculative-rejection-sampling (SEGFAULT)
\t101 - test-hrx-mul-mat-id-k32 (Subprocess aborted)
Errors while running CTest
"""
    known = {"test-tokenizers-ggml-vocabs", "test-kv-zero",
             "test-speculative-rejection-sampling", "test-hrx-mul-mat-id-k32"}
    new, fixed, cannot = check(real, known)
    cases += 1
    if (new, fixed, cannot) != (set(), set(), None):
        print(f"  SELF-TEST FAIL: the issue's own 15-failure log vs a matching baseline -> {new, fixed, cannot}")
        bad += 1
    # A NEW failure is the whole point: exit 1 and name it.
    new2, _, cannot2 = check(real, known - {"test-kv-zero"})
    cases += 1
    if (new2, cannot2) != ({"test-kv-zero"}, None):
        print(f"  SELF-TEST FAIL: a baseline missing one failure -> {new2, cannot2}")
        bad += 1
    # engine#2825 review (mine): a SKIPPED row is not a failure. ctest prints its skipped rows in the same
    # `  318 - name (Skipped)` shape, and FAILED_LINE's catch-all counted every one of them, so a real log whose
    # summary said "3 failed" produced 16 names and check() refused as "truncated". Measured on a real 383-test log.
    with_skips = """\
97% tests passed, 4 tests failed out of 112
The following tests FAILED:
\t 19 - test-kv-zero (Subprocess aborted)
\t318 - onnx_graph (Skipped)
\t319 - audio_exec (Skipped)
\t326 - whisper_transcribe_jfk (Skipped)
Errors while running CTest
"""
    skips = failed(with_skips)
    cases += 1
    if set(skips) != {"test-kv-zero"}:
        print(f"  SELF-TEST FAIL: a skipped row was counted as a failure -> {sorted(skips)}")
        bad += 1
    # engine#2825 review (mine): the fixture above puts the summary FIRST and ctest puts it LAST. With the summary in
    # its real place a regex without re.MULTILINE cannot see it, so every real log answered "cannot tell" while this
    # self-test stayed green -- the arm that was supposed to cover the shape covered the one shape ctest never emits.
    # Both shapes are pinned now, so neither can regress in silence.
    real_tail = real.replace("97% tests passed, 4 tests failed out of 112\n", "") + "\n97% tests passed, 4 tests failed out of 112\n"
    new_t, fixed_t, cannot_t = check(real_tail, known)
    # ...and --write must MERGE, not regenerate: the licence block and the hand annotations are what a reader and
    # tools/copyright.py --check rely on, and the first documented regeneration deleted both (engine#2825 review).
    write_root = pathlib.Path(tempfile.mkdtemp())
    (write_root / "tools/ci").mkdir(parents=True)
    seeded = (write_root / "tools/ci/ctest_known_fail.txt")
    seeded.write_text('# Copyright 2026 x\n# SPDX-License-Identifier: Apache-2.0\n#\n'
                      'test-kv-zero    # Subprocess aborted -- main\n')
    write_baseline(seeded, {"test-kv-zero", "brand-new"}, real_tail)
    kept = seeded.read_text()
    cases += 1
    if "SPDX-License-Identifier" not in kept:
        print("  SELF-TEST FAIL: --write dropped the licence block that copyright.py --check requires")
        bad += 1
    elif "Subprocess aborted -- main" not in kept:
        print("  SELF-TEST FAIL: --write overwrote an existing annotation")
        bad += 1
    elif "brand-new" not in kept:
        print("  SELF-TEST FAIL: --write did not add a newly-failed name")
        bad += 1
    shutil.rmtree(write_root, ignore_errors=True)

    cases += 1
    if (new_t, fixed_t, cannot_t) != (set(), set(), None):
        print(f"  SELF-TEST FAIL: ctest's real log shape (FAILED block first, summary LAST) -> {new_t, fixed_t, cannot_t}")
        bad += 1
    # A known failure that now passes is reported, not failed.
    green = """\
100% tests passed, 0 tests failed out of 112
"""
    new3, fixed3, cannot3 = check(green, known)
    cases += 1
    if (new3, fixed3, cannot3) != (set(), known, None):
        print(f"  SELF-TEST FAIL: all-green vs a stale baseline -> {new3, fixed3, cannot3}")
        bad += 1
    # THE case this tool exists to get right: a log that cannot answer must not read as clean.
    for log, why in (("", "the log is empty"),
                     ("-- Configuring done\n-- Generating done\n", "tests passed"),
                     ("100% tests passed, 0 tests failed out of 112\nThe following tests FAILED:\n"
                      "\t 18 - a (Failed)\n", "one of the two is truncated")):
        cases += 1
        got_new, _, got_cannot = check(log, known)
        if got_new or not got_cannot or why not in got_cannot:
            print(f"  SELF-TEST FAIL: an unanswerable log -> {got_new, got_cannot}")
            bad += 1
    # The tool's first real catch was its own author's fixture: a log claiming 15 failures while
    # listing 4. Answering from it would silently ignore the 11 it did not print.
    cases += 1
    _, _, cannot5 = check(real.replace("97% tests passed, 4 tests failed", "87% tests passed, 15 tests failed"),
                          known)
    if not cannot5 or "the summary says 15 failed" not in cannot5:
        print(f"  SELF-TEST FAIL: a 15-vs-4 log -> {cannot5}")
        bad += 1
    # A log from a different suite must not be compared at all.
    cases += 1
    _, _, cannot4 = check(real, known, want_total=999)
    if not cannot4 or "different suite" not in cannot4:
        print(f"  SELF-TEST FAIL: a 112-test log checked against want_total=999 -> {cannot4}")
        bad += 1
    # Colour: the first CI run's log had the summary wrapped in SGR codes and the reader could not see
    # it. This arm is the one that would have caught that before the run did.
    ansi = "\x1b[0;32m5% tests passed, 1 tests failed out of 112\x1b[0m\nThe following tests FAILED:\n" \
           "\x1b[31m\t 19 - test-kv-zero (Subprocess aborted)\x1b[0m\n"
    cases += 1
    if summary(ansi) != (1, 112) or failed(ansi).get("test-kv-zero") != "Subprocess aborted":
        print(f"  SELF-TEST FAIL: an ANSI-coloured log -> {summary(ansi)} {failed(ansi)}")
        bad += 1
    # The reason is kept, because (SEGFAULT) and (Failed) are different bugs.
    cases += 1
    if failed(real).get("test-kv-zero") != "Subprocess aborted":
        print(f"  SELF-TEST FAIL: the reason was lost: {failed(real)}")
        bad += 1
    # baseline() against a real file with a column-0 licence header -- the shape the workflow reads.
    # This arm was missing, which is why the regex above survived until a human ran the step.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / BASELINE).parent.mkdir(parents=True, exist_ok=True)
        (root / BASELINE).write_text(
            "# Copyright 2026 bong-water-water-bong\n# SPDX-License-Identifier: Apache-2.0\n#\n"
            "# Known failures follow.\n"
            "test-kv-zero                  # Subprocess aborted -- main\n"
            "test-hrx-fa-masked-v  # Failed\n")
        # the value is the NAME: the `# reason` is for humans reading the file, and `--write`
        # re-derives it from the log, so a round-trip through baseline() loses nothing.
        want = {"test-kv-zero": "test-kv-zero", "test-hrx-fa-masked-v": "test-hrx-fa-masked-v"}
        got = baseline(root)
        cases += 1
        if got != want:
            print(f"  SELF-TEST FAIL: baseline() from a file with a licence header -> {sorted(got)}")
            bad += 1
    print(f"  self-test: {cases} case(s), {bad} failure(s)")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default="build/ctest.log")
    ap.add_argument("--root", default=str(TOOL_ROOT))
    ap.add_argument("--write", action="store_true", help="refresh the baseline from this log")
    ap.add_argument("--tests", type=int, default=None, help="expected suite size; a mismatch is exit 2")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return 1 if self_test() else 0
    root = pathlib.Path(args.root)
    log_path = pathlib.Path(args.log)
    if not log_path.exists():
        print(f"ctest_known_fail: cannot tell: no log at {args.log}", file=sys.stderr)
        return 2
    text = log_path.read_text()
    known = baseline(root)
    if args.write:
        sums = summary(text)
        if sums is None:
            print("ctest_known_fail: refusing to write a baseline from a log with no summary line",
                  file=sys.stderr)
            return 2
        write_baseline(root / BASELINE, failed(text), text)
        print(f"ctest_known_fail: wrote {len(failed(text))} known failure(s) to {BASELINE}")
        return 0
    new, fixed, cannot = check(text, known, args.tests)
    if cannot:
        print(f"ctest_known_fail: cannot tell: {cannot}", file=sys.stderr)
        return 2
    for name in sorted(fixed):
        print(f"  now passing, remove from {BASELINE}: {name}")
    if new:
        for name in sorted(new):
            print(f"  NEW failure: {name} -- not in {BASELINE}", file=sys.stderr)
        return 1
    print(f"ctest_known_fail: {len(known)} known failure(s), no new ones")
    return 0


if __name__ == "__main__":
    sys.exit(main())
