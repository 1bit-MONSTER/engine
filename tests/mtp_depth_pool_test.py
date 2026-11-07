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
"""tools/mtp_depth_pool.py: the pooled per-depth live-vs-offline verdict, against fixture tickets.

The tool exists because the brief judges "live acceptance equals the offline protocol within 2 pp at every
depth" on 8 prompts x 64 tokens POOLED, while a device ticket can only carry two or three prompts, and because
pooling `mtp_acceptance --live --match` logs by eye produced two wrong numbers in this lane.  Both traps are
refusals in the tool, so they are tested here with fake tickets instead of only on the happy path:

1. two tickets that agree pool; a ticket whose per-prompt rows disagree on a shared cell, or that does not add
   up to its own run-level row (a short read), is refused rather than averaged;
2. the echoed summary a harness prints after its exit line is ignored, so a cell is never counted twice;
3. a depth with no denominator on either side is reported as unmeasured, never as a +0.00 pp pass;
4. the numbers the MTP leaderboard records publish (all-chains rates, hit rate, both tokens-per-pass figures) are
   re-derived from the same fixtures and compared to the record JSON, so a record cannot drift from its evidence --
   and every MTP record in the directory has to be one the guard names, so a new one cannot arrive uncovered.

usage: tests/mtp_depth_pool_test.py        (pure python, no model, no device)
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "mtp_depth_pool.py")

# prompt 0: depth1 8/10 (offline 8/10), depth2 5/9 (5/9)
# prompt 1: depth1 7/10 (offline 6/10), depth2 4/9 (4/9)
TICKET_A = """forward ready: vocab 248320, max_batch_rows 16, hidden source post-final-norm
prompt 0: TEXT IDENTICAL, 64 tokens, accepted 6/16, passes/token 0.500, stream 72 rows, 0 mismatch
  match p0: 4 offline chains at this loop's own positions (stream 72 rows)
depth 1: live  80.00 % (8/10, reached)  offline conditional  80.00 % (8/10)
depth 2: live  55.56 % (5/9, reached)  offline conditional  55.56 % (5/9)
prompt 1: TEXT IDENTICAL, 64 tokens, accepted 5/16, passes/token 0.500, stream 72 rows, 0 mismatch
  match p1: 4 offline chains at this loop's own positions (stream 72 rows)
depth 1: live  70.00 % (7/10, reached)  offline conditional  60.00 % (6/10)
depth 2: live  44.44 % (4/9, reached)  offline conditional  44.44 % (4/9)
depth 1: live  75.00 % (15/20)  offline  70.00 % (14/20)  delta  +5.00 pp  FAIL
depth 2: live  50.00 % (9/18)  offline  50.00 % (9/18)  delta  +0.00 pp  PASS
== ticket exited rc=0 (hold reached)
prompt 0: TEXT IDENTICAL, 64 tokens, accepted 6/16, passes/token 0.500, stream 72 rows, 0 mismatch
depth 1: live  40.00 % (4/10, reached)  offline conditional  40.00 % (4/10)
"""

# prompt 2: depth1 9/10 (offline 7/10), depth2 3/9 (3/9)
TICKET_B = """prompt 2: TEXT IDENTICAL, 64 tokens, accepted 4/16, passes/token 0.500, stream 72 rows, 0 mismatch
  match p2: 4 offline chains at this loop's own positions (stream 72 rows)
depth 1: live  90.00 % (9/10, reached)  offline conditional  70.00 % (7/10)
depth 2: live  33.33 % (3/9, reached)  offline conditional  33.33 % (3/9)
depth 1: live  90.00 % (9/10)  offline  70.00 % (7/10)  delta  +20.00 pp  FAIL
depth 2: live  33.33 % (3/9)  offline  33.33 % (3/9)  delta  +0.00 pp  PASS
"""

# pooled over A+B: depth1 live 24/30 = 80.00 % vs offline 21/30 = 70.00 % -> +10.00 pp FAIL
#                 depth2 live 12/27 = 44.44 % vs offline 12/27 = 44.44 % -> +0.00 pp PASS
POOLED_DEPTH1 = "24/30 = 80.00%"
POOLED_DEPTH2 = "12/27 = 44.44%"

fails = 0


def check(cond, what):
    global fails
    print(f"{'PASS' if cond else 'FAIL'}  {what}")
    if not cond:
        fails += 1


def run(args, logs):
    paths = []
    for name, body in logs:
        p = os.path.join(TMP, name)
        with open(p, "w") as fh:
            fh.write(body)
        paths.append(p)
    r = subprocess.run([sys.executable, TOOL] + args + paths, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


TMP = tempfile.mkdtemp(prefix="mtp-depth-pool-test.")

# 1. two tickets that agree: the pool is their union, keyed by (prompt, depth), and the tail echo is ignored.
rc, out, err = run([], [("a.log", TICKET_A), ("b.log", TICKET_B)])
check(rc == 0, "two agreeing tickets pool without a refusal")
check("pooled 6 scored cells = 3 prompts x 2 depths" in out, "3 prompts x 2 depths = 6 cells (the echoed tail is not counted)")
check(POOLED_DEPTH1 in out, f"depth 1 pools to {POOLED_DEPTH1}")
check(POOLED_DEPTH2 in out, f"depth 2 pools to {POOLED_DEPTH2}")
check("+10.00 pp" in out and "FAIL" in out, "depth 1 is 10.00 pp out, so the depth itself reads FAIL")
check("1 of 2 scored depths within 2 pp" in out, "the verdict line counts one pass of two scored depths")

# 2. the same ticket twice is not double counting: identical cells are the same cells.
rc2, out2, _ = run([], [("a1.log", TICKET_A), ("a2.log", TICKET_A)])
check(rc2 == 0 and "pooled 4 scored cells" in out2, "the same ticket twice pools to the same 4 cells, not 8")

# 3. two tickets disagreeing on one shared cell -- prompt 1 depth 2, which both A and C score -- is refused.
# C is internally consistent (its own run-level row matches its rows), so only the cross-ticket check can catch it.
TICKET_C = """prompt 1: TEXT IDENTICAL, 64 tokens, accepted 5/16, passes/token 0.500, stream 72 rows, 0 mismatch
depth 1: live  70.00 % (7/10, reached)  offline conditional  60.00 % (6/10)
depth 2: live  44.44 % (4/9, reached)  offline conditional  11.11 % (1/9)
depth 1: live  70.00 % (7/10)  offline  60.00 % (6/10)  delta  +10.00 pp  FAIL
depth 2: live  44.44 % (4/9)  offline  11.11 % (1/9)  delta  +33.33 pp  FAIL
"""
rc3, _, err3 = run([], [("a.log", TICKET_A), ("c.log", TICKET_C)])
check(rc3 == 1, f"a disagreement between tickets is refused (rc={rc3})")
check("disagrees" in err3 or "scored twice" in err3, f"the refusal says what disagrees: {err3.strip()[:120]}")

# 4. a short read: the per-prompt rows no longer add up to the ticket's own run-level row.
short = TICKET_A.replace("depth 2: live  44.44 % (4/9, reached)  offline conditional  44.44 % (4/9)\n", "")
rc4, _, err4 = run([], [("short.log", short)])
check(rc4 == 1, f"a log missing a cell is refused (rc={rc4})")
check("read incompletely" in err4, f"the refusal names the short read: {err4.strip()[:120]}")

# 5. a depth with no denominator is unmeasured, not a pass.
floor0 = """prompt 0: TEXT IDENTICAL, 64 tokens, accepted 1/16, passes/token 0.500, stream 72 rows, 0 mismatch
depth 1: live  100.00 % (1/1, reached)  offline conditional  100.00 % (1/1)
depth 2: live   0.00 % (0/0, reached)  offline conditional   0.00 % (0/0)
depth 1: live  100.00 % (1/1)  offline  100.00 % (1/1)  delta  +0.00 pp  PASS
depth 2: live   0.00 % (0/0)  offline   0.00 % (0/0)  delta  +0.00 pp  PASS
"""
rc5, out5, _ = run([], [("floor0.log", floor0)])
check(rc5 == 0, "a depth with an empty denominator does not fail the run")
check("no denominator" in out5 and "1 of 1 scored depths" in out5,
      "the empty depth is reported as unmeasured and excluded from the scored count")

# 6. a depth the LOG reported as `not scored` is carried through as unmeasured, not dropped.
#
# The tool prints `depth 4: n/a (live 0/0 reached, offline 15/20)  not scored` instead of a `%` row whenever one
# side has no denominator (npu/mtp/mtp_head.h:96), and at tau>0 the chain gate stops before depths 3-4.  A pool
# that only kept the rows it could score dropped the depth entirely, so the table and the verdict line read as a
# complete pass over a table that had quietly lost a depth -- the phantom-depth read (#916) one layer up.
NOTSCORED = """prompt 0: TEXT IDENTICAL, 64 tokens, accepted 20/40, passes/token 0.500, stream 72 rows, 0 mismatch
  per-depth p0: 1 18/20 2 13/20 3 2/20 4 0/20
  match p0: 20 offline chains at this loop's own positions (stream 72 rows)
depth 1: live  90.00 % (18/20, reached)  offline conditional  90.00 % (18/20)
depth 2: live  65.00 % (13/20, reached)  offline conditional  65.00 % (13/20)
depth 3: live  10.00 % (2/20, reached)  offline conditional  10.00 % (2/20)
depth 4: n/a (live 0/0 reached, offline 15/20)  not scored
depth 1: live  90.00 % (18/20)  offline  90.00 % (18/20)  delta  +0.00 pp  PASS
depth 2: live  65.00 % (13/20)  offline  65.00 % (13/20)  delta  +0.00 pp  PASS
depth 3: live  10.00 % (2/20)  offline  10.00 % (2/20)  delta  +0.00 pp  PASS
depth 4: n/a (live 0/0 reached, offline 15/20)  not scored
"""
rc6, out6, _ = run([], [("notscored.log", NOTSCORED)])
check(rc6 == 0, "a log with a not-scored depth still pools")
check("unmeasured -- no chain reached this depth" in out6, "the not-scored depth is printed as unmeasured")
check("4  unmeasured" in out6, "... and it is depth 4, the one the log named")
check("3 of 3 scored depths" in out6 and "1 of 4 depths unmeasured: 4" in out6,
      f"the verdict says 3 of 4 depths, not a bare '3 of 3': {out6.strip().splitlines()[-1]}")

# 7. a pooled prompt set with a hole says so, instead of printing the range it spans.
#
# This lane runs one or two prompts per ticket, so prompts 0,1,5,6,7 is a normal pool -- but printed as
# "prompts 0..7" it reads as the brief's whole window.
def ticket(p, tokens=64):
    return (f"prompt {p}: TEXT IDENTICAL, {tokens} tokens, accepted 4/16, passes/token 0.500, stream 72 rows, 0 mismatch\n"
            f"depth 1: live  80.00 % (8/10, reached)  offline conditional  80.00 % (8/10)\n"
            f"depth 1: live  80.00 % (8/10)  offline  80.00 % (8/10)  delta  +0.00 pp  PASS\n")


rc7, out7, _ = run([], [(f"p{p}.log", ticket(p)) for p in (0, 1, 5)])
check(rc7 == 0, "a gapped prompt set pools without a refusal (per-prompt tickets are the normal case)")
check("NOT a contiguous window: 2,3,4 missing of 0..5" in out7,
      f"the header names the hole instead of spanning it: {out7.splitlines()[0]}")
check("64 tokens/prompt" in out7, "a uniform budget is stated in the header")
rc8, out8, _ = run([], [(f"q{p}.log", ticket(p)) for p in (0, 1, 2)])
check("prompts 0..2" in out8 and "NOT a contiguous" not in out8, "a contiguous pool keeps the plain range form")

# 8. the generation budget is part of the window: 8 prompts x 64 tokens is the brief's, so a pool that mixes
# budgets says so instead of inheriting the shape of whichever ticket it read first.
rc9, out9, _ = run([], [("t64.log", ticket(0)), ("t128.log", ticket(1, tokens=128))])
check(rc9 == 0, "mixed budgets still pool (a longer prompt is a legitimate measurement)")
check("MIXED token budgets" in out9 and "64 on prompts 0" in out9 and "128 on prompts 1" in out9,
      f"the mixed budget is named per prompt: {[l for l in out9.splitlines() if 'MIXED' in l]}")

# 9. the same prompt at two lengths is a contradiction about the same cell, not a wider window.
rc10, _, err10 = run([], [("same64.log", ticket(3)), ("same32.log", ticket(3, tokens=32))])
check(rc10 == 1 and "two lengths of the same prompt" in err10,
      f"prompt 3 at 64 and at 32 tokens is refused: {err10.strip()[:110]}")

# 10. a log whose per-prompt rows predate the `reached` column is a DIFFERENT quantity, and says so: its live
# rate is marginal, not conditional, so "the log was read incompletely" would send a reader to re-run a complete
# ticket.  (2 of the 16 cell-bearing logs in this lane's own corpus are this shape.)
OLD_SHAPE = """prompt 0: TEXT IDENTICAL, 64 tokens, accepted 44/80, passes/token 0.625, stream 72 rows, 0 mismatch
  match p0: 20 offline chains at this loop's own positions (stream 72 rows)
    depth 1: live  90.00 % (18/20)  offline  90.00 % (18/20)  delta  +0.00 pp
    depth 2: live  65.00 % (13/20)  offline  65.00 % (13/20)  delta  +0.00 pp
    == live vs offline at the SAME positions (the 2 pp bar):
      depth 1: live  90.00 % (18/20)  offline  90.00 % (18/20)  delta  +0.00 pp  PASS
      depth 2: live  65.00 % (13/20)  offline  65.00 % (13/20)  delta  +0.00 pp  PASS
"""
rc11, _, err11 = run([], [("oldshape.log", OLD_SHAPE)])
check(rc11 == 1, "a pre-`reached` log is refused rather than pooled as if it were the current shape")
check("pre-`reached` shape" in err11 and "read incompletely" not in err11,
      f"the refusal names the real cause, not a short read: {err11.strip()[-120:]}")

# 12. k is part of the window too: a k=6 ticket pooled with k=4 ones builds a six-depth table whose depths 5-6
# exist on the wider prompts only, and `N of N scored depths` then reads as a pass over depths the rest never tried.
def ticket_k(p, k):
    body = [f"prompt {p}: TEXT IDENTICAL, 64 tokens, accepted 4/16, passes/token 0.500, stream 72 rows, 0 mismatch"]
    body += [f"    depth {d}: live  50.00 % (10/20, reached)  offline conditional  50.00 % (10/20)" for d in range(1, k + 1)]
    body.append("    == live vs offline at the SAME positions (the 2 pp bar):")
    body += [f"      depth {d}: live  50.00 % (10/20)  offline  50.00 % (10/20)  delta  +0.00 pp  PASS"
             for d in range(1, k + 1)]
    return "\n".join(body) + "\n"


rc12, out12, _ = run([], [("k4.log", ticket_k(0, 4)), ("k6.log", ticket_k(1, 6))])
check(rc12 == 0, "a pool of runs at different k still pools (the cells themselves are not in conflict)")
check("MIXED chain width (k)" in out12 and "6 on prompts 1" in out12 and "4 on prompts 0" in out12,
      f"the differing chain width is named per prompt: {[l for l in out12.splitlines() if 'MIXED chain' in l]}")
check("the depths above 4 come from the wider prompts only" in out12,
      "... and says what it means for the table (depths 5-6 are not the whole pool)")
rc13, out13, _ = run([], [("k4a.log", ticket_k(0, 4)), ("k4b.log", ticket_k(1, 4))])
check("k=4" in out13 and "MIXED chain width" not in out13, "a uniform chain width is stated once in the header")

# 13a. tau is part of the window too, and it is the setting the draft route actually ships with (`Options::tau = 0.6`,
# npu/mtp/draft.h:162). At tau>0 the chain gate stops the chain after the first low-confidence draft, so a tau=0.6
# ticket offers depths 1-2 while a tau=0 one offers 1-4: pooled together, the deeper rows read as a full pass over a
# window where half the tickets never drafted them. That is the k hole one setting over, and it was real until the tool
# started printing `== N prompts x M tokens, k=K tau=T, hidden=H` on the live path as well as the offline one.
def ticket_tau(p, tau, settings=True):
    body = []
    if settings:
        body.append(f"== 8 prompts x 64 tokens, k=4 tau={tau:.2f}, hidden=post-norm")
    body += [f"prompt {p}: TEXT IDENTICAL, 64 tokens, accepted 4/16, passes/token 0.500, stream 72 rows, 0 mismatch",
             "depth 1: live  80.00 % (8/10, reached)  offline conditional  80.00 % (8/10)",
             "depth 1: live  80.00 % (8/10)  offline  80.00 % (8/10)  delta  +0.00 pp  PASS"]
    return "\n".join(body) + "\n"


rc16, out16, _ = run([], [("t0a.log", ticket_tau(0, 0.0)), ("t0b.log", ticket_tau(1, 0.0))])
check(rc16 == 0 and "tau=0" in out16 and "MIXED chain gate" not in out16,
      f"a uniform chain gate is stated in the header: {[l for l in out16.splitlines() if 'tau=' in l][:1]}")
rc17, out17, _ = run([], [("t06a.log", ticket_tau(0, 0.6)), ("t06b.log", ticket_tau(1, 0.6))])
check(rc17 == 0 and "tau=0.6" in out17, "a tau=0.6 window says so")
rc18, out18, _ = run([], [("t0.log", ticket_tau(0, 0.0)), ("t06.log", ticket_tau(1, 0.6))])
check(rc18 == 0, "a pool across chain gates still pools (the cells themselves are not in conflict)")
check("MIXED chain gate (tau)" in out18 and "0, 0.6" in out18,
      f"pooling tau=0 with tau=0.6 is named rather than averaged: {[l for l in out18.splitlines() if 'MIXED chain' in l]}")
rc19, out19, _ = run([], [("old.log", ticket_tau(0, 0.0, settings=False))])
check("tau unstated" in out19 and "cannot be told apart" in out19,
      "a log without the settings line says its chain gate is unstated instead of assuming tau=0")
rc20, out20, _ = run([], [("mixed1.log", ticket_tau(0, 0.6)), ("mixed2.log", ticket_tau(1, 0.6, settings=False))])
check("is stated by 1 of 2 ticket(s)" in out20,
      f"a partly-stated window says which tickets it read the gate from: {[l for l in out20.splitlines() if 'stated by' in l]}")

# 14. the run identity (vocab, batch width, hidden source) is part of the window: 2.1 vs 3.19 tokens/pass on the
# hidden-source switch alone is more than the bar, so pooling across it would average two measurements.
READY_NORM = "forward ready: vocab 248320, max_batch_rows 16, hidden source post-final-norm\n"
READY_RSD = "forward ready: vocab 248320, max_batch_rows 16, hidden source residual\n"
rc14, out14, _ = run([], [("norm.log", READY_NORM + ticket_k(0, 4)), ("rsd.log", READY_RSD + ticket_k(1, 4))])
check(rc14 == 0, "a pool across hidden sources still pools (the cells themselves are not in conflict)")
check("MIXED run identity" in out14 and "post-final-norm" in out14 and "residual" in out14,
      f"a differing hidden source is named rather than averaged: {[l for l in out14.splitlines() if 'MIXED run' in l]}")
rc15, out15, _ = run([], [("n1.log", READY_NORM + ticket_k(0, 4)), ("n2.log", READY_NORM + ticket_k(1, 4))])
check("vocab 248320, max_batch_rows 16, hidden post-final-norm" in out15 and "MIXED run identity" not in out15,
      "a uniform run identity is stated in the header")

# 15. the recorded 8x64 window, from the three tickets shipped under tests/fixtures/mtp_clause55 (see its README).
# Without this the known answer is only checkable on the machine that holds the original `npu-hold` logs, i.e. the tool
# that produces the deliverable's published verdict could break it and CI would not notice.
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "mtp_clause55")
CL = [os.path.join(FIXTURES, n) for n in ("p0.txt", "p1-4.txt", "p5-7.txt")]
# A fixture that is present on the machine but not COMMITTED is the worse failure mode: the test passes locally and the
# CI checkout has nothing to read. It happened with `.log` (`.gitignore:19`), so being tracked is checked here when the
# tree is a git checkout, and the files carry a suffix the ignore rules do not match.
if os.path.isdir(os.path.join(ROOT, ".git")) or os.path.exists(os.path.join(ROOT, ".git")):
    listed = subprocess.run(["git", "-C", ROOT, "ls-files", "--"] + [os.path.relpath(p, ROOT) for p in CL],
                            capture_output=True, text=True)
    tracked = len([l for l in listed.stdout.splitlines() if l.strip()])
    check(tracked == len(CL), f"the three fixtures are tracked by git ({tracked} of {len(CL)})")
if all(os.path.exists(p) for p in CL):
    r16 = subprocess.run([sys.executable, TOOL, "--selftest"] + CL, capture_output=True, text=True)
    check(r16.returncode == 0, f"the shipped fixtures reproduce the recorded 8x64 window (rc={r16.returncode})")
    check("32 scored cells = 8 prompts x 4 depths" in r16.stdout, "the fixtures pool to the recorded 8 x 4 cells")
    for want in ("136/181 = 75.14%", "86/136 = 63.24%", "63/86 = 73.26%", "53/63 = 84.13%"):
        check(want in r16.stdout, f"fixture pool reports {want}")
    check("4 of 4 scored depths within 2 pp" in r16.stdout, "all four depths are inside the bar on the fixtures")
    check("selftest PASS: 4 depths reproduce the recorded 8x64 window" in r16.stdout,
          "the tool's own known-answer check passes on them")
else:
    check(False, f"tests/fixtures/mtp_clause55 is incomplete: {CL}")

# 16. the tool's own refusals, and (no logs) nothing else: this is what runs on a box without the ticket corpus.
r = subprocess.run([sys.executable, TOOL, "--selftest"], capture_output=True, text=True)
check(r.returncode == 0 and "selftest PASS tail repeat" in r.stdout, "the tool's refusal self-test passes")

# 17. the MTP leaderboard records are reproducible from the tree.
#
# The records name tools/mtp_depth_pool.py as their instrument, but the tool computed the CONDITIONAL per-depth table
# only: the all-chains rates, the hit rate and both tokens-per-pass figures were typed from a side parser, so the
# record named a tool that could not produce its own number -- the generated-artifact hole (#1213) one layer down, in
# a number rather than a file.  Every value is re-derived here from the three shipped tickets and compared to the
# record JSON, and the comparison is proved to bite before it is trusted.
import importlib.util  # noqa: E402  (section-local: the records are the last thing this file checks)
import json  # noqa: E402

RECORDS = os.path.join(ROOT, "bench", "npu-leaderboard")
FIXTURE_ROOT = os.path.join(ROOT, "tests", "fixtures")
# One entry per measured WINDOW, because a record's value only means anything against the tickets measured at its own
# setting: the window is (tau, k, tokens, prompts) and pooling two settings is the error the pool tool itself refuses
# for k and for the generation budget.  A new row in a different regime adds a window here, not a lambda to a flat dict.
WINDOWS = [
    {
        "name": "tau=0.6 (the default), k=4, 8 prompts x 64 tokens",
        "dir": "mtp_tau06",
        "files": ("p0.txt", "p1.txt", "p2.txt", "p3.txt", "p4.txt", "p5.txt", "p6.txt", "p7.txt"),
        "records": {
            "27b-mtp-tau06-depth1-20261106": lambda e: e["depth_rate"][1],
            "27b-mtp-tau06-depth2-20261106": lambda e: e["depth_rate"][2],
            "27b-mtp-tau06-depth3-20261106": lambda e: e["depth_rate"][3],
            "27b-mtp-tau06-hit-rate-20261106": lambda e: e["hit"],
            "27b-mtp-tau06-tokens-per-pass-20261106": lambda e: e["tpp_live"],
            "27b-mtp-tau06-offline-tokens-per-pass-20261106": lambda e: e["tpp_offline"],
        },
    },
    {
        "name": "tau=0, k=4, 8 prompts x 64 tokens",
        "dir": "mtp_clause55",
        "files": ("p0.txt", "p1-4.txt", "p5-7.txt"),
        "records": {
            "27b-mtp-live-depth1-20261105": lambda e: e["depth_rate"][1],
            "27b-mtp-live-depth2-20261105": lambda e: e["depth_rate"][2],
            "27b-mtp-live-depth3-20261105": lambda e: e["depth_rate"][3],
            "27b-mtp-live-depth4-20261105": lambda e: e["depth_rate"][4],
            "27b-mtp-hit-rate-20261105": lambda e: e["hit"],
            "27b-mtp-live-tokens-per-pass-20261105": lambda e: e["tpp_live"],
            "27b-mtp-offline-tokens-per-pass-20261105": lambda e: e["tpp_offline"],
        },
    },
]


def record_mismatches(expect):
    """Every record whose published value is not the derived one, as one line each (empty list = all agree)."""
    out = []
    for rid, want in expect.items():
        path = os.path.join(RECORDS, rid + ".json")
        if not os.path.exists(path):
            out.append(f"{rid}: no record at {os.path.relpath(path, ROOT)}")
            continue
        with open(path) as fh:
            rec = json.load(fh)
        if rec["metric"]["value"] != round(want, 2):
            out.append(f"{rid}: record says {rec['metric']['value']} {rec['metric']['unit']}, "
                       f"the fixtures derive {round(want, 2)}")
        elif "mtp_depth_pool.py" not in rec.get("instrument", ""):
            out.append(f"{rid}: value agrees but the instrument it names is not the tool that derived it")
    return out


spec = importlib.util.spec_from_file_location("mtp_depth_pool", TOOL)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
covered = set()
for win in WINDOWS:
    paths = [os.path.join(FIXTURE_ROOT, win["dir"], f) for f in win["files"]]
    covered |= set(win["records"])
    if not all(os.path.exists(p) for p in paths):
        check(False, f"the {win['name']} window has its tickets on disk ({win['dir']})")
        continue
    _seen, _un, _src, _bud, _wid, _idt, _offered, _taus = mod.pool(paths)
    econ = mod.economics(_seen, _offered)
    check("error" not in econ, f"{win['name']}: the tickets support the derived block: {econ.get('error', 'ok')}")
    if "error" in econ:
        continue
    derived = {rid: fn(econ) for rid, fn in win["records"].items()}
    bad = record_mismatches(derived)
    check(not bad, f"{win['name']}: all {len(derived)} records equal what its tickets derive: {bad[:2]}")
    # anti-vacuity: a record one hundredth off must be caught, or the check above passes on anything.
    wobbled = dict(derived)
    first = sorted(wobbled)[0]
    wobbled[first] += 0.01
    check(record_mismatches(wobbled) != [],
          f"{win['name']}: the record comparison bites: {first} +0.01 is reported, not accepted")

# ... and the guard covers the DIRECTORY, not just the ids some window happens to name: an MTP record added later that
# nobody adds here would look exactly like a covered one, which is how the "one table row per record" hole was closed
# for the page (#1213) and how a check ends up reporting success on the records it was told about rather than the ones
# that exist.  A record from a regime nobody measured in-tree is therefore a red test, not a silent row.
on_disk = sorted(f[:-5] for f in os.listdir(RECORDS) if f.startswith("27b-mtp-") and f.endswith(".json"))
check(len(on_disk) >= len(covered), f"the guard's record directory is readable ({len(on_disk)} MTP record(s))")
uncovered = [rid for rid in on_disk if rid not in covered]
check(not uncovered, f"every MTP record on disk is checked by a window: uncovered {uncovered}")

print(f"\n{'all checks passed' if not fails else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
