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
#
# mtp_depth_pool.py -- pool the live-vs-offline per-depth table over several `mtp_acceptance --live --match`
# runs, and say whether the brief's clause ("live acceptance equals the offline protocol within 2 pp at every
# depth") holds on the pooled window.
#
# Why pooling exists at all: the clause is judged on 8 prompts x 64 tokens POOLED, but one `--live` run costs
# ~90 s of array time plus a model + head load per process, so an 8-prompt window is usually two or three
# npu-hold tickets of <= 3 min each.  Pooling those logs by eye is how this lane produced two wrong numbers,
# so the arithmetic lives here and the two traps are refusals rather than comments:
#   1. a ticket log REPEATS ITS OWN TAIL after the harness' exit line, so a cell can appear twice -- pooling
#      that once inflated a 12-cell window to 20 cells;
#   2. a cell is keyed by (prompt, depth), and a second row for the same key with different counts is a
#      disagreement between two tickets (or a format change), never an average to be taken.
# A ticket's own run-level row is the sum of its per-prompt rows, so it is checked too: a truncated log cannot
# then pass silently as a smaller window.
#
#   tools/mtp_depth_pool.py ticket-a.log ticket-b.log ...   # pooled table + the 2 pp verdict per depth
#   tools/mtp_depth_pool.py --selftest                      # the refusals and the arithmetic, no logs needed
#   tools/mtp_depth_pool.py --selftest <the ticket logs>    # ... and the recorded 8x64 window, to the count
#
# Exit: 0 the pool was computed (individual depths may still read FAIL), 1 a refusal, 2 no input.
"""tools/mtp_depth_pool.py -- the per-depth live-vs-offline verdict over several mtp_acceptance tickets."""
import re
import sys

BAR = 2.0  # the brief's bar, in percentage points

# The tool prints the per-depth table in two shapes, and confusing them is what this file exists not to repeat:
#   per PROMPT (keyed, poolable):
#     depth 2: live  55.00 % (11/20, reached)  offline conditional  52.63 % (10/19)  delta  +2.37 pp  > 2 pp
#   per RUN (this ticket's own pool over its prompts, used only as the read-completeness check):
#     depth 2: live  52.78 % (38/72)  offline  52.11 % (37/71)  delta  +0.67 pp  PASS
DETAIL = re.compile(r"depth (\d+):\s+live\s+-?[\d.]+ %\s+\((\d+)/(\d+), reached\)\s+"
                    r"offline conditional\s+-?[\d.]+ %\s+\((\d+)/(\d+)\)")
RUNLEVEL = re.compile(r"depth (\d+):\s+live\s+-?[\d.]+ %\s+\((\d+)/(\d+)\)\s+"
                      r"offline\s+-?[\d.]+ %\s+\((\d+)/(\d+)\)\s+delta")
PROMPT = re.compile(r"(?:per-depth p|prompt )(\d+)\b")

# The numbers this tool must reproduce when it is pointed at the recorded window (TMTPLOOP-STATE.md,
# 2026-11-05): live (num, den) and offline (num, den) per depth, over 8 prompts x 64 tokens.
RECORDED = {1: (136, 181, 136, 181), 2: (86, 136, 84, 134), 3: (63, 86, 60, 82), 4: (53, 63, 49, 59)}


def parse_ticket(path):
    """Return {prompt, depth: (live_n, live_d, off_n, off_d)} for one ticket log.

    Raises SystemExit on a log it must not pool: a depth row before any prompt header (nothing to key it to),
    a cell scored twice with different counts, or per-prompt rows that do not add up to the ticket's own
    run-level row (a short read)."""
    cells, runlevel, prompt = {}, {}, None
    for ln in open(path, errors="replace").read().splitlines():
        if "== ticket exited" in ln:  # trap 1: everything after this is the harness echoing the summary
            break
        m = PROMPT.search(ln)
        if m and "depth " not in ln:
            prompt = int(m.group(1))
            continue
        m = DETAIL.search(ln)
        if m:
            if prompt is None:
                raise SystemExit(f"mtp_depth_pool: {path}: a depth row before any prompt line -- "
                                 f"refusing to guess which prompt it belongs to")
            key = (prompt, int(m.group(1)))
            got = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
            if key in cells and cells[key] != got:
                raise SystemExit(f"mtp_depth_pool: {path}: prompt {key[0]} depth {key[1]} scored twice with "
                                 f"different counts {cells[key]} then {got}")
            cells.setdefault(key, got)
            continue
        m = RUNLEVEL.search(ln)
        if m:
            runlevel[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
    # The ticket's own run-level row is the pool of its per-prompt rows, so a short read cannot pass as a
    # smaller window: the two have to agree before this log is allowed into the pool at all.
    for d, rl in runlevel.items():
        got = tuple(sum(v[i] for (_, dd), v in cells.items() if dd == d) for i in range(4))
        if got != rl:
            raise SystemExit(f"mtp_depth_pool: {path}: depth {d} per-prompt rows pool to {got} but the ticket's "
                             f"own run-level row says {rl} -- the log was read incompletely, do not pool it")
    if runlevel and len(runlevel) != len({d for _, d in cells}):
        raise SystemExit(f"mtp_depth_pool: {path}: run-level row covers {sorted(runlevel)} but per-prompt rows "
                         f"cover {sorted({d for _, d in cells})}")
    return cells


def pool(paths):
    """Union the tickets' cells, refusing any disagreement on a key.  Returns (cells, sources)."""
    seen, sources = {}, 0
    for p in paths:
        rows = parse_ticket(p)
        if not rows:
            print(f"mtp_depth_pool: {p}: no matched rows (was it run with --live --match?)", file=sys.stderr)
            continue
        sources += 1
        for k, v in rows.items():
            if k in seen and seen[k] != v:
                raise SystemExit(f"mtp_depth_pool: {p} disagrees with an earlier ticket on prompt {k[0]} "
                                 f"depth {k[1]}: {seen[k]} then {v}")
            seen.setdefault(k, v)
    return seen, sources


def table(seen, sources):
    """Print the pooled per-depth table; return (inside, scored, totals per depth)."""
    depths = sorted({d for _, d in seen})
    prompts = sorted({p for p, _ in seen})
    print(f"pooled {len(seen)} scored cells = {len(prompts)} prompts x {len(depths)} depths "
          f"(prompts {prompts[0]}..{prompts[-1]}, {sources} ticket(s))")
    print(f"\n{'depth':>5}  {'live':>18}  {'offline (same positions)':>24}  {'delta':>9}  verdict")
    inside = scored = 0
    totals = {}
    for d in depths:
        live_n = sum(v[0] for (_, dd), v in seen.items() if dd == d)
        live_d = sum(v[1] for (_, dd), v in seen.items() if dd == d)
        off_n = sum(v[2] for (_, dd), v in seen.items() if dd == d)
        off_d = sum(v[3] for (_, dd), v in seen.items() if dd == d)
        totals[d] = (live_n, live_d, off_n, off_d)
        if not live_d or not off_d:
            # 0/0 is not agreement: a depth no chain reached is unmeasured, and scoring it as +0.00 pp PASS
            # is how a phantom depth was once reported as a passing one (engine#916).
            print(f"{d:>5}  {'no denominator':>18}")
            continue
        delta = 100.0 * live_n / live_d - 100.0 * off_n / off_d
        ok = abs(delta) <= BAR
        scored += 1
        inside += ok
        print(f"{d:>5}  {f'{live_n}/{live_d} = {100.0 * live_n / live_d:.2f}%':>18}  "
              f"{f'{off_n}/{off_d} = {100.0 * off_n / off_d:.2f}%':>24}  {f'{delta:+.2f} pp':>9}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"\n{inside} of {scored} scored depths within {BAR:g} pp at the same positions")
    return inside, scored, totals


def refusal_selftest():
    """The refusals are why this tool exists, so they are tested and not only the happy path."""
    import os
    import tempfile

    d2 = "depth 2: live  72.22 % (13/18, reached)  offline conditional  72.22 % (13/18)\n"
    cases = [
        # a cell scored twice with different counts (two tickets disagreeing, or a format change)
        ("differing duplicate", "prompt 0: TEXT IDENTICAL\n" + d2
         + "depth 2: live  50.00 % (9/18, reached)  offline conditional  50.00 % (9/18)\n", "scored twice"),
        # a log whose per-prompt rows do not add up to the ticket's own run-level row (short read)
        ("incomplete read", "prompt 0: TEXT IDENTICAL\n" + d2
         + "depth 2: live  50.00 % (9/18)  offline  50.00 % (9/18)  delta  +0.00 pp  PASS\n",
         "read incompletely"),
        # a depth row with no prompt header to key it to
        ("unkeyed depth row", d2, "before any prompt line"),
    ]
    for name, body, want in cases:
        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write(body)
            path = fh.name
        try:
            parse_ticket(path)
        except SystemExit as e:
            if want not in str(e):
                print(f"selftest FAIL {name}: refused with {str(e)!r}, wanted {want!r}", file=sys.stderr)
                return 1
            print(f"selftest PASS {name}: refused -- {str(e).split(': ', 1)[-1][:96]}")
        else:
            print(f"selftest FAIL {name}: accepted a log it must refuse", file=sys.stderr)
            return 1
        finally:
            os.unlink(path)

    # The tail trap: the same cells repeated after the exit line must not double the window.
    d1 = "depth 1: live  80.00 % (8/10, reached)  offline conditional  80.00 % (8/10)\n"
    run = "depth 1: live  80.00 % (8/10)  offline  80.00 % (8/10)  delta  +0.00 pp  PASS\n"
    tail = ("== ticket exited rc=0 (hold reached)\n" + "prompt 0: TEXT IDENTICAL\n" + d1
            + "depth 1: live  40.00 % (4/10, reached)  offline conditional  40.00 % (4/10)\n")
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
        fh.write("prompt 0: TEXT IDENTICAL\n" + d1 + run + tail)
        path = fh.name
    try:
        cells = parse_ticket(path)
        if cells != {(0, 1): (8, 10, 8, 10)}:
            print(f"selftest FAIL tail repeat: parsed {cells}, wanted only the first row", file=sys.stderr)
            return 1
        print("selftest PASS tail repeat: the echoed summary after the exit line is ignored")
    finally:
        os.unlink(path)
    return 0


def main(argv):
    selftest = "--selftest" in argv
    paths = [a for a in argv if a != "--selftest"]
    if not paths:
        if selftest:
            return refusal_selftest()
        print(__doc__)
        return 2
    seen, sources = pool(paths)
    if not seen:
        print("mtp_depth_pool: nothing to pool", file=sys.stderr)
        return 1
    _, _, totals = table(seen, sources)
    if not selftest:
        return 0
    bad = {d: (totals.get(d), RECORDED[d]) for d in RECORDED if totals.get(d) != RECORDED[d]}
    if bad:
        for d, (g, e) in bad.items():
            print(f"selftest FAIL depth {d}: pooled {g} != recorded {e}", file=sys.stderr)
        return 1
    print(f"selftest PASS: {len(RECORDED)} depths reproduce the recorded 8x64 window to the count")
    return refusal_selftest()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
