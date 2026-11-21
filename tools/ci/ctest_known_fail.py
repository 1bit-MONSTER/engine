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

Exit: 0 every failure is in the baseline, 1 a test failed that the baseline does not name (named) or a
row did not run on a device, 2 cannot answer (no log, or no summary line), 77 a row did not run on a
device and the caller said a device-less box may skip (`--no-device skip`).

**A row that did not run is neither a pass nor a failure** (engine#2912). Two shapes, both counted in
their own column and never folded into the other two:

  * ctest skipped it (`(Skipped)` in the log: a model or fixture is absent). It is not a failure, and it
    is not "now passing" either -- a baselined row that was skipped stays in the baseline.
  * it ran with no device. The fork's HRX rows print `test skipped: no device available` and ctest calls
    most of them `(Failed)` (engine#2393's FAIL_REGULAR_EXPRESSION; since fork #439/#441 the runner also
    exits 77). Read from the summary alone that row is either "a known failure" (when the baseline names
    it, so the run reads green having touched no device) or "a NEW failure" (a regression nobody
    introduced). `--outputs <build>/Testing/Temporary/LastTest.log` is where ctest keeps each row's
    output; with it the rows that printed the line are reported as "did not run on a device", and the
    run fails (or skips with 77 under `--no-device skip`) for that reason by name.

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


# engine#2912: the line every HRX row of the fork prints when it cannot see a device. It is the fork's
# `kSkipMessagePrefix` (tests/testing_suite.h) and the text of engine#2393's FAIL_REGULAR_EXPRESSION, and it is
# what a reader of the exit code cannot see: a whole-run skip exits 77, but a run whose host cases ran and whose
# device cases did not still exits 0, and its hand-written siblings (test-hrx-q2_0, ...) exit 1.
DEVICE_SKIP = "test skipped: no device available"
# One row's block in ctest's Testing/Temporary/LastTest.log starts `86/111 Testing: hrx-buffer-test`.
OUTPUT_BLOCK = re.compile(r"^\d+/\d+ Testing: (.+)$", re.MULTILINE)
SKIP_RC = 77


def skipped(log_text: str):
    """{name: reason} for the rows ctest did not run: `(Skipped)` and `(Disabled)`. Never failures, never passes."""
    out = {}
    for name, reason in FAILED_LINE.findall(ANSI.sub("", log_text)):
        if reason.split()[0].lower() in ("skipped", "disabled"):
            out[name] = reason
    return out


def outputs(last_test_log: str):
    """{name: output} from ctest's LastTest.log -- only what the row printed, between `Output:` and `<end of output>`.

    The `Command:` line above the output quotes the row's arguments, so searching the whole block would find the
    skip line in a row that merely mentions it (measured: `sh -c "echo test skipped: ..."` is echoed there).
    """
    text = ANSI.sub("", last_test_log)
    marks = list(OUTPUT_BLOCK.finditer(text))
    out = {}
    for i, mark in enumerate(marks):
        block = text[mark.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        start = block.find("\nOutput:\n")
        end = block.find("<end of output>")
        out[mark.group(1).strip()] = block[start:end] if start != -1 and end != -1 else ""
    return out


# engine#2987: ctest reason prefixes that mean the row stopped abnormally, i.e. it RAN and then died. A row can print
# the fork's device-skip line for its device cases and still segfault in its host half -- the hand-written rows do
# exactly that -- so the skip line alone is not evidence that nothing ran.
HARD_OUTCOMES = ("SEGFAULT", "Subprocess aborted", "Timeout", "Exception")


def hard_failures(log_text: str):
    """The rows ctest says stopped abnormally, whatever their own output printed."""
    return {n for n, reason in failed(log_text).items() if reason.split()[0] in HARD_OUTCOMES}


def device_skipped(last_test_log: str, log_text: str = ""):
    """The rows whose own output says they did not run on a device, whatever ctest called them.

    A HARD outcome is never "did not run" (engine#2987): the test ran and then crashed. Before this, one `in` test on
    the output text took such a row out of both the failure and the new-failure columns, and with `--no-device skip`
    the step then exited 77 -- a segfault reported as "this box has no device". Reachable in this fleet, because the
    hand-written HRX rows print the skip line for their device cases and exit non-zero for their host half.
    """
    return {name for name, text in outputs(last_test_log).items()
            if DEVICE_SKIP in text and name not in hard_failures(log_text)}


def exit_code(new, no_device, policy="fail"):
    """The row's verdict. A new failure outranks a skip: 77 must never hide a red row behind an absent device."""
    if new:
        return 1
    if no_device:
        return SKIP_RC if policy == "skip" else 1
    return 0


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
    """{test name: the entry as written} from the checked-in baseline.

    The value keeps the row's whole line, comment included, because the `# reason` after the name is where the
    recorded CLASS lives (`# Failed` vs `# Not Run`) and `reason_class` reads it from there (engine#3129). The name
    is still the key, so every caller that only asks "is this row known" is unaffected.
    """
    path = root / BASELINE
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        stripped = COMMENT.sub("", line).strip()
        if stripped:
            out[stripped.split()[0]] = line.strip()
    return out


def reason_class(reason: str) -> str:
    """What KIND of failure the FAILED block reports: 'not-run' when there was no executable, else 'failed'.

    ctest puts `Not Run` -- the binary was never built -- in the same FAILED block as assertion failures (see
    failed()), and the baseline was compared by NAME only, so a baselined `Failed` row whose target stopped compiling
    moved class and stayed "known": engine#3129 was that month. `hrx-backend-test` was baselined `Failed` on
    2026-11-17, stopped compiling with fork #438 (one stale call, `moe_router_descriptor_layout`), and its
    `Not Run` never printed a new failure anywhere. The reason after `#` is free text, so only its first words are
    read; everything that is not `Not Run` is the assertion-failure class.
    """
    text = reason or ""
    if "#" in text:
        # a baseline line is "<name>    # <reason>"; the name is not the reason (and is never "Not Run")
        text = text.split("#", 1)[1]
    return "not-run" if text.strip().lower().startswith("not run") else "failed"


def class_changes(names, known):
    """{name: (baseline_class, log_class)} for baselined rows whose class changed.

    Empty when `known` is a set of names: the self-test's shorthand, and any caller with no baseline file, records no
    reason -- so nothing is compared and the answer is exactly what it was before this existed. Treating a set as
    "every class unknown" would make every arm a class change, which is the opposite of safe.
    """
    if not isinstance(known, dict):
        return {}
    out = {}
    for name in names:
        if name in known:
            was, now = reason_class(known[name]), reason_class(names[name])
            if was != now:
                out[name] = (was, now)
    return out


def check(log_text: str, known, want_total=None, changed=None):
    """(new_failures, fixed, cannot) -- cannot is a string when the log cannot answer at all.

    `fixed` is "in the baseline and did not fail", minus the rows ctest skipped (engine#2912): a baselined row
    that did not run was printed as "now passing, remove from the baseline", which is a pass nobody measured.

    A baselined row whose FAILED-block CLASS changed is a new failure too (engine#3129) -- pass a dict as `changed`
    to receive {name: (baseline_class, log_class)} for the report.
    """
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
    classes = class_changes(names, known)
    if changed is not None:
        changed.update(classes)
    new = {n for n in names if n not in known} | set(classes)
    did_not_run = skipped(log_text)
    fixed = {n for n in known if n not in names and n not in did_not_run}
    return new, fixed, None


def judge(log_text: str, known, last_test_log=None, want_total=None):
    """check(), with the rows that did not run on a device taken out of BOTH columns.

    Returns a dict: new, fixed, known_failed, skipped, no_device, not_in_run (sets), passed and total (ints),
    cannot (str|None). not_in_run is the baseline's names this run never started (not registered in this
    configuration, or filtered out): known only with the outputs, and not "now passing" either.
    `last_test_log` is the text of ctest's LastTest.log, or None when the caller has no outputs to offer -- then
    no_device is empty and this is check() with the columns counted.
    """
    changed = {}
    new, fixed, cannot = check(log_text, known, want_total, changed)
    out = {"new": new, "fixed": fixed, "known_failed": set(), "skipped": set(), "no_device": set(),
           "not_in_run": set(), "passed": 0, "total": 0, "cannot": cannot, "class_changed": changed}
    if cannot:
        return out
    names = set(failed(log_text))
    did_not_run = set(skipped(log_text))
    no_device = set()
    if last_test_log is not None:
        seen = outputs(last_test_log)
        # A LastTest.log from another run is worse than none: ctest keeps the file until the next run rewrites it,
        # so a run that died early would be judged by the previous run's outputs. Every row ctest called failed has
        # a block in the file of the run that failed it, `Not Run` rows included (measured on ctest 4.2.3).
        missing = sorted(names - set(seen))
        if missing:
            out["cannot"] = (f"the outputs file has no block for {len(missing)} failed row(s) ({', '.join(missing[:3])}"
                             f"{', ...' if len(missing) > 3 else ''}) -- it is from a different run than the log")
            return out
        # a row that crashed is a failure even if it also printed the device-skip line (engine#2987)
        hard = hard_failures(log_text)
        no_device = {n for n, text in seen.items() if DEVICE_SKIP in text} - hard
        out["not_in_run"] = set(known) - set(seen)
    _, total = summary(log_text)
    out["no_device"] = no_device
    out["new"] = new - no_device
    # a row that exits 0 while printing the skip line did not pass, and neither did one that was never started
    out["fixed"] = fixed - no_device - out["not_in_run"]
    # a row that changed class is a NEW failure, so it must not also be counted as a known one (engine#3129)
    out["known_failed"] = (names & set(known)) - no_device - set(new)
    out["skipped"] = did_not_run - no_device
    out["total"] = total
    out["passed"] = total - len(names | did_not_run | no_device)
    return out


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
    # engine#3129: the baseline was compared by NAME only, so a baselined `Failed` row whose target stopped
    # compiling moved to `Not Run` and stayed "known" -- `hrx-backend-test` did exactly that for a month. The class is
    # now part of the comparison, in both directions, and a set of names (how every arm above passes `known`) is
    # deliberately unaffected.
    not_built = """\
95% tests passed, 4 tests failed out of 112
The following tests FAILED:
\t 18 - test-tokenizers-ggml-vocabs (Failed)
\t 19 - test-kv-zero (Subprocess aborted)
\t 37 - test-speculative-rejection-sampling (SEGFAULT)
\t101 - test-hrx-mul-mat-id-k32 (Not Run)
Errors while running CTest
"""
    base_dict = {n: n + "    # Failed -- measured in a real run" for n in known}
    changed = {}
    new_c, fixed_c, cannot_c = check(not_built, base_dict, None, changed)
    cases += 1
    if (new_c, cannot_c) != ({"test-hrx-mul-mat-id-k32"}, None) or \
            changed.get("test-hrx-mul-mat-id-k32") != ("failed", "not-run"):
        print(f"  SELF-TEST FAIL: a baselined failure that stopped building is not a new failure -> {new_c, changed}")
        bad += 1
    # the row a baseline RECORDS as `Not Run` keeps working (#2836/#3129: `test-hrx-host-resident` really does not
    # build at this pin, and a guard that fired on it would be the same bug pointing the other way)
    recorded_not_run = dict(base_dict, **{"test-hrx-mul-mat-id-k32": "test-hrx-mul-mat-id-k32  # Not Run -- #2836"})
    new_n, _, cannot_n = check(not_built, recorded_not_run, None, {})
    cases += 1
    if (new_n, cannot_n) != (set(), None):
        print(f"  SELF-TEST FAIL: a row baselined AS Not Run is not a class change -> {new_n, cannot_n}")
        bad += 1
    # ... and the reverse direction: a row baselined "did not build" that now RUNS and fails is not covered by that
    # baseline either, so it is new (otherwise "we knew it could not build" silently becomes "we accept it failing")
    runs_again = not_built.replace("101 - test-hrx-mul-mat-id-k32 (Not Run)",
                                  "101 - test-hrx-mul-mat-id-k32 (Subprocess aborted)")
    new_r, _, cannot_r = check(runs_again, recorded_not_run, None, changed_r := {})
    cases += 1
    if (new_r, cannot_r) != ({"test-hrx-mul-mat-id-k32"}, None) or \
            changed_r.get("test-hrx-mul-mat-id-k32") != ("not-run", "failed"):
        print(f"  SELF-TEST FAIL: a row baselined Not Run that now fails is not a new failure -> {new_r, changed_r}")
        bad += 1
    # a SET of names carries no recorded reason, so nothing is compared: this is the behaviour every arm above uses,
    # and it must stay byte-for-byte what it was (a set read as "class unknown" would turn every arm into a change)
    new_set, _, cannot_set = check(not_built, known, None, {})
    cases += 1
    if (new_set, cannot_set) != (set(), None):
        print(f"  SELF-TEST FAIL: a set of known names must not be class-compared -> {new_set, cannot_set}")
        bad += 1
    # the report half: judge() carries the detail and does not double-count the row as a known failure
    got_c = judge(not_built, base_dict)
    cases += 1
    if sorted(got_c["new"]) != ["test-hrx-mul-mat-id-k32"] or \
            got_c["class_changed"].get("test-hrx-mul-mat-id-k32") != ("failed", "not-run") or \
            "test-hrx-mul-mat-id-k32" in got_c["known_failed"]:
        print(f"  SELF-TEST FAIL: judge() does not report the class change once -> "
              f"{sorted(got_c['new'])}, {got_c['class_changed']}, {sorted(got_c['known_failed'])}")
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
        # The value now KEEPS the row's own line, because the recorded class (`# Failed` vs `# Not Run`) after the
        # name is what engine#3129 showed the ratchet could not see -- and it is stripped before this function used
        # to store it. The NAME is still the key, so "is this row known" is unchanged, and the property this arm was
        # written for -- a column-0 licence block is not read as names -- is still what it asserts: two rows, not
        # four comment lines.
        want = {"test-kv-zero": "test-kv-zero                  # Subprocess aborted -- main",
                "test-hrx-fa-masked-v": "test-hrx-fa-masked-v  # Failed"}
        got = baseline(root)
        cases += 1
        if got != want:
            print(f"  SELF-TEST FAIL: baseline() from a file with a licence header -> {sorted(got)}")
            bad += 1
    # ── engine#2912: a row that did not run is neither a pass nor a failure ──────────────────────────────────────
    # Both fixtures are ctest 4.2.3's own output for eight throwaway rows (dates and directories cut), one per shape
    # the fork's HRX rows can take on a box with no device:
    #   suite-all-skipped  every case needs a device: exit 77 (fork #441), no SKIP_RETURN_CODE -> (Failed)
    #   suite-mixed        host cases ran, device cases did not: exit 0, the regex of engine#2393 -> (Failed)
    #   quiet-skip         prints the line, exits 0, no regex on the row -> ctest says Passed
    #   skipcode-wins      prints the line, exits 77, SKIP_RETURN_CODE set -> ctest says (Skipped)
    #   model-skip         77 for an absent model: a skip that has nothing to do with the device
    nodev_log = """\
1/8 Test #1: plain-pass .......................   Passed    0.00 sec
2/8 Test #2: plain-fail .......................***Failed    0.00 sec
3/8 Test #3: model-skip .......................***Skipped   0.00 sec
4/8 Test #4: suite-all-skipped ................***Failed  Error regular expression found in output. Regex=[test skipped: no device available]  0.00 sec
5/8 Test #5: suite-mixed ......................***Failed  Error regular expression found in output. Regex=[test skipped: no device available]  0.00 sec
6/8 Test #6: quiet-skip .......................   Passed    0.00 sec
7/8 Test #8: not-built ........................***Not Run   0.00 sec
8/8 Test #7: skipcode-wins ....................***Skipped   0.00 sec

50% tests passed, 4 tests failed out of 8

The following tests did not run:
\t  3 - model-skip (Skipped)
\t  7 - skipcode-wins (Skipped)

The following tests FAILED:
\t  2 - plain-fail (Failed)
\t  4 - suite-all-skipped (Failed)                        hrx
\t  5 - suite-mixed (Failed)                              hrx
\t  8 - not-built (Not Run)
Errors while running CTest
"""

    def block(index, name, command, printed, verdict):
        return (f"{index}/8 Testing: {name}\n{index}/8 Test: {name}\nCommand: {command}\nDirectory: /b\n"
                f"\"{name}\" start time: Nov 20 08:33 AST\nOutput:\n"
                "----------------------------------------------------------\n"
                f"{printed}<end of output>\nTest time =   0.00 sec\n"
                "----------------------------------------------------------\n"
                f"{verdict}\n\"{name}\" end time: Nov 20 08:33 AST\n\"{name}\" time elapsed: 00:00:00\n"
                "----------------------------------------------------------\n\n")

    regex_hit = "Error regular expression found in output. Regex=[test skipped: no device available]"
    nodev_last = "Start testing: Nov 20 08:33 AST\n----------------------------------------------------------\n" + "".join((
        block(1, "plain-pass", '"/usr/bin/sh" "-c" "echo fine"', "fine\n", "Test Passed."),
        # the Command line QUOTES the skip text and the row prints something else: a mention is not a skip
        block(2, "plain-fail", '"/usr/bin/sh" "-c" "echo broken # not: test skipped: no device available"',
              "broken\n", "Test Failed."),
        block(3, "model-skip", '"/usr/bin/sh" "-c" "echo no model; exit 77"', "no model\n", "Test Passed."),
        block(4, "suite-all-skipped", '"/b/bin/hrx-buffer-test"',
              "test skipped: no device available for HRX test (11 device case(s) skipped)\n",
              "Test Fail Reason:\n" + regex_hit),
        block(5, "suite-mixed", '"/b/bin/test-hrx-buffer"',
              "test skipped: no device available for HRX test (6 device case(s) skipped)\n"
              "HRX test summary: 2 passed, 0 failed, 0 known-fail, 0 unexpected-pass, 2 total, 3 ms\n",
              "Test Fail Reason:\n" + regex_hit),
        block(6, "quiet-skip", '"/b/bin/test-hrx-quiet"', "test skipped: no device available\n", "Test Passed."),
        block(8, "not-built", '""', "", "Test Passed."),
        block(7, "skipcode-wins", '"/b/bin/test-hrx-both"', "test skipped: no device available\n",
              "Test Pass Reason:\n" + regex_hit))) + "End testing: Nov 20 08:33 AST\n"
    absent = {"suite-all-skipped", "suite-mixed", "quiet-skip", "skipcode-wins"}

    cases += 1
    if set(skipped(nodev_log)) != {"model-skip", "skipcode-wins"} or set(skipped(nodev_log)) & set(failed(nodev_log)):
        print(f"  SELF-TEST FAIL: ctest's skipped rows -> {sorted(skipped(nodev_log))}, failed {sorted(failed(nodev_log))}")
        bad += 1
    # THE ARM for "now passing": a baselined row ctest skipped has not passed. Before, it was told to leave the baseline.
    cases += 1
    _, fixed_s, cannot_s = check(nodev_log, {"model-skip", "plain-fail"})
    if cannot_s or fixed_s:
        print(f"  SELF-TEST FAIL: a baselined row that was SKIPPED read as now passing -> {fixed_s, cannot_s}")
        bad += 1
    cases += 1
    if device_skipped(nodev_last) != absent:
        print(f"  SELF-TEST FAIL: rows whose output says no device -> {sorted(device_skipped(nodev_last))} "
              f"(plain-fail only QUOTES the line in its Command)")
        bad += 1
    # ── engine#2987: a row that printed the skip line and then CRASHED did not "not run" ────────────────────────
    # ctest 4.2.3's shape for the reachable case: the row prints the device-skip line for its device cases and its
    # host half segfaults. The skip line is in its output AND the FAILED block carries the outcome.
    crash_log = (
        "1/2 Test #1: hrx-crash .......................***Failed    0.10 sec   (SEGFAULT)\n"
        "2/2 Test #2: hrx-skip ........................***Failed  "
        "Error regular expression found in output. Regex=[test skipped: no device available]  0.00 sec\n"
        "\n0% tests passed, 2 tests failed out of 2\n\nThe following tests FAILED:\n"
        "\t  1 - hrx-crash (SEGFAULT)\n\t  2 - hrx-skip (Failed)\n\nErrors while running CTest\n")

    def block2(index, name, printed, verdict):
        return (f"{index}/2 Testing: {name}\n{index}/2 Test: {name}\nCommand: \"/b/bin/{name}\"\nDirectory: /b\n"
                f"\"{name}\" start time: Nov 20 08:33 AST\nOutput:\n"
                "----------------------------------------------------------\n"
                f"{printed}<end of output>\nTest time =   0.10 sec\n"
                "----------------------------------------------------------\n"
                f"{verdict}\n\"{name}\" end time: Nov 20 08:33 AST\n")
    crash_last = ("Start testing: Nov 20 08:33 AST\n----------------------------------------------------------\n" + "".join((
        block2(1, "hrx-crash", "test skipped: no device available\nFAIL hrx-crash: assertion after the skip line\n",
               "Test Failed.\n\nTest Fail Reason:\nTest Failed."),
        block2(2, "hrx-skip", "test skipped: no device available\n", "Test Failed.\n\nTest Fail Reason:\n" + regex_hit),
    )) + "End testing: Nov 20 08:33 AST\n")
    cases += 1
    if device_skipped(crash_last, crash_log) != {"hrx-skip"}:
        print(f"  SELF-TEST FAIL: a row that CRASHED after the skip line was counted as 'did not run' -> "
              f"{sorted(device_skipped(crash_last, crash_log))} (want ['hrx-skip'])")
        bad += 1
    cases += 1
    got = judge(crash_log, set(), crash_last)
    if got["new"] != {"hrx-crash"} or got["no_device"] != {"hrx-skip"}:
        print(f"  SELF-TEST FAIL: a crash after the skip line -> new {sorted(got['new'])}, "
              f"no_device {sorted(got['no_device'])} (want new ['hrx-crash'], no_device ['hrx-skip'])")
        bad += 1
    cases += 1
    if exit_code(got["new"], got["no_device"], "skip") != 1:
        print("  SELF-TEST FAIL: --no-device skip swallowed a crash into 77")
        bad += 1

    # THE ARM: a device-less run against a baseline that names one of the device rows. Read from the summary alone
    # suite-mixed is "a known failure" and suite-all-skipped "a NEW failure"; neither ran.
    cases += 1
    got = judge(nodev_log, {"plain-fail", "not-built", "suite-mixed"}, nodev_last)
    want = {"new": set(), "fixed": set(), "known_failed": {"plain-fail", "not-built"}, "skipped": {"model-skip"},
            "no_device": absent, "not_in_run": set(), "passed": 1, "total": 8, "cannot": None,
            "class_changed": {}}
    if got != want:
        print(f"  SELF-TEST FAIL: a device-less run, by column -> {got}")
        bad += 1
    # ...and what that does to the row: red where a device is required, 77 where a box may have none, never 0.
    cases += 1
    codes = (exit_code(got["new"], got["no_device"], "fail"), exit_code(got["new"], got["no_device"], "skip"),
             exit_code(set(), set(), "skip"), exit_code({"x"}, absent, "skip"))
    if codes != (1, SKIP_RC, 0, 1):
        print(f"  SELF-TEST FAIL: exit codes (required, allowed, nothing absent, new failure + absent) -> {codes}")
        bad += 1
    # quiet-skip is the worst of the four: ctest says Passed. With it baselined it would be "now passing".
    cases += 1
    got_q = judge(nodev_log, {"quiet-skip"}, nodev_last)
    if "quiet-skip" in got_q["fixed"] or got_q["new"] != {"plain-fail", "not-built"}:
        print(f"  SELF-TEST FAIL: a row that printed the skip line and exited 0 -> fixed {got_q['fixed']}, new {got_q['new']}")
        bad += 1
    # A baselined name this run never started (the real case: hrx-backend-test-loom exists only in an add-on
    # build) is not "now passing". plain-pass is: it is in the baseline, it ran, and it did not fail.
    cases += 1
    got_r = judge(nodev_log, {"row-of-another-configuration", "plain-pass"}, nodev_last)
    if got_r["fixed"] != {"plain-pass"} or got_r["not_in_run"] != {"row-of-another-configuration"}:
        print(f"  SELF-TEST FAIL: a baselined row that was never started -> fixed {got_r['fixed']}, "
              f"not_in_run {got_r['not_in_run']}")
        bad += 1
    # No outputs offered: nothing can be said about devices, and the answer is check()'s, unchanged.
    cases += 1
    got_n = judge(nodev_log, {"plain-fail", "not-built"})
    if got_n["no_device"] or got_n["new"] != {"suite-all-skipped", "suite-mixed"} or got_n["cannot"]:
        print(f"  SELF-TEST FAIL: no outputs file -> {got_n}")
        bad += 1
    # An outputs file from another run must not be believed: ctest keeps it until the next run rewrites it.
    cases += 1
    stale = judge(nodev_log, set(), nodev_last.replace("4/8 Testing: suite-all-skipped", "4/8 Testing: other-row"))
    if not stale["cannot"] or "different run" not in stale["cannot"] or "suite-all-skipped" not in stale["cannot"]:
        print(f"  SELF-TEST FAIL: an outputs file missing a failed row -> {stale['cannot']}")
        bad += 1
    # The same through main(), because the exit code is the contract: required -> 1, allowed -> 77, a missing
    # outputs file -> 2, and --write does not record a row that did not run on a device as debt.
    import contextlib, io
    with tempfile.TemporaryDirectory() as tmp:
        troot = pathlib.Path(tmp)
        (troot / BASELINE).parent.mkdir(parents=True)
        (troot / BASELINE).write_text("# header\nplain-fail    # Failed\nnot-built    # Not Run\nsuite-mixed    # Failed\n")
        (troot / "ctest.log").write_text(nodev_log)
        (troot / "LastTest.log").write_text(nodev_last)
        base = ["--log", str(troot / "ctest.log"), "--root", str(troot)]
        with_out = base + ["--outputs", str(troot / "LastTest.log")]
        sink_out, sink_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(sink_out), contextlib.redirect_stderr(sink_err):
            rcs = (main(with_out),)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rcs += (main(with_out + ["--no-device", "skip"]),
                    main(base + ["--outputs", str(troot / "absent.log")]), main(base))
        cases += 1
        if rcs != (1, SKIP_RC, 2, 1):
            print(f"  SELF-TEST FAIL: main() (device required, allowed, outputs missing, no outputs) -> {rcs}")
            bad += 1
        cases += 1
        said = sink_out.getvalue() + sink_err.getvalue()
        if ("DID NOT RUN ON A DEVICE: suite-mixed" not in said or "NEW failure" in said
                or "8 row(s): 1 passed, 2 known failure(s), 0 new, 1 skipped, 4 without a device" not in said):
            print(f"  SELF-TEST FAIL: the report does not keep the columns apart:\n{said}")
            bad += 1
        (troot / BASELINE).write_text("# header\nplain-fail    # Failed\n")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            main(with_out + ["--write"])
        cases += 1
        if set(baseline(troot)) != {"plain-fail", "not-built"}:
            print(f"  SELF-TEST FAIL: --write recorded a row that did not run on a device -> {sorted(baseline(troot))}")
            bad += 1
    # The skip line is the fork's, so it is checked against the fork when this tree has it checked out. A line
    # reworded there (fork #439's first version moved it) would leave every arm above green and the tool blind.
    header = TOOL_ROOT / "third_party" / "llama.cpp" / "tests" / "testing_suite.h"
    if header.exists():
        cases += 1
        if f'"{DEVICE_SKIP}' not in header.read_text(errors="replace"):
            print(f"  SELF-TEST FAIL: {header} no longer prints '{DEVICE_SKIP}' -- the fork changed its skip line")
            bad += 1
    print(f"  self-test: {cases} case(s), {bad} failure(s)")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", default="build/ctest.log")
    ap.add_argument("--root", default=str(TOOL_ROOT))
    ap.add_argument("--write", action="store_true", help="refresh the baseline from this log")
    ap.add_argument("--tests", type=int, default=None, help="expected suite size; a mismatch is exit 2")
    ap.add_argument("--outputs", default=None, metavar="LASTTEST_LOG",
                    help="ctest's Testing/Temporary/LastTest.log for the same run: rows that printed "
                         f"'{DEVICE_SKIP}' are reported as did-not-run, not as failures or passes")
    ap.add_argument("--no-device", choices=("fail", "skip"), default="fail",
                    help="what a row that did not run on a device does to the exit code: fail (1, the default -- "
                         "the suite's subject is the hardware) or skip (77, for a box that is allowed to have none)")
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
    last = None
    if args.outputs is not None:
        out_path = pathlib.Path(args.outputs)
        if not out_path.exists():
            print(f"ctest_known_fail: cannot tell: no outputs file at {args.outputs}, so a row that did not run "
                  f"on a device cannot be told from a failure", file=sys.stderr)
            return 2
        last = out_path.read_text(errors="replace")
    if args.write:
        sums = summary(text)
        if sums is None:
            print("ctest_known_fail: refusing to write a baseline from a log with no summary line",
                  file=sys.stderr)
            return 2
        # A row that did not run on a device is not debt: written down, it would make every later device-less run
        # read as "known failures, no new ones". A name the file already carries keeps its line (write merges).
        # ... and --write must not drop a crashed row from the baseline either (engine#2987)
        absent = (device_skipped(last, text) if last is not None else set()) - set(known)
        names = {n for n in failed(text) if n not in absent}
        write_baseline(root / BASELINE, names, text)
        print(f"ctest_known_fail: wrote {len(names)} known failure(s) to {BASELINE}")
        for name in sorted(absent & set(failed(text))):
            print(f"  not written, it did not run on a device: {name}")
        return 0
    got = judge(text, known, last, args.tests)
    if got["cannot"]:
        print(f"ctest_known_fail: cannot tell: {got['cannot']}", file=sys.stderr)
        return 2
    new, no_device = got["new"], got["no_device"]
    for name in sorted(got["fixed"]):
        print(f"  now passing, remove from {BASELINE}: {name}")
    for name in sorted(got["not_in_run"]):
        print(f"  not in this run (not registered in this configuration, or filtered out): {name}")
    for name in sorted(got["skipped"]):
        print(f"  did not run (ctest skipped it): {name}")
    for name in sorted(no_device):
        print(f"  DID NOT RUN ON A DEVICE: {name} -- it printed '{DEVICE_SKIP}'; not a pass, not a failure",
              file=sys.stderr)
    for name, (was, now) in sorted(got.get("class_changed", {}).items()):
        print(f"  CLASS CHANGED: {name} -- {BASELINE} records {was!r}, this run says {now!r}: a row that stopped "
              f"building is not the failure the baseline recorded (engine#3129)", file=sys.stderr)
    for name in sorted(new):
        if name in got.get("class_changed", {}):
            continue      # the CLASS CHANGED line above states it; "not in the baseline" would be untrue
        print(f"  NEW failure: {name} -- not in {BASELINE}", file=sys.stderr)
    print(f"ctest_known_fail: {got['total']} row(s): {got['passed']} passed, {len(got['known_failed'])} known "
          f"failure(s), {len(new)} new, {len(got['skipped'])} skipped, {len(no_device)} without a device")
    rc = exit_code(new, no_device, args.no_device)
    if new:
        return rc
    if no_device:
        verdict = ("SKIP (77): this box is allowed to have no device" if rc == SKIP_RC else
                   "this check needs a device, so it FAILS; nothing is known about those rows")
        print(f"ctest_known_fail: {len(no_device)} row(s) did not run on a device -- {verdict}", file=sys.stderr)
        return rc
    print(f"ctest_known_fail: {len(known)} known failure(s), no new ones")
    return rc


if __name__ == "__main__":
    sys.exit(main())
