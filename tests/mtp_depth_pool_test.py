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
3. a depth with no denominator on either side is reported as unmeasured, never as a +0.00 pp pass.

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
def ticket(p):
    return (f"prompt {p}: TEXT IDENTICAL, 64 tokens, accepted 4/16, passes/token 0.500, stream 72 rows, 0 mismatch\n"
            f"depth 1: live  80.00 % (8/10, reached)  offline conditional  80.00 % (8/10)\n"
            f"depth 1: live  80.00 % (8/10)  offline  80.00 % (8/10)  delta  +0.00 pp  PASS\n")


rc7, out7, _ = run([], [(f"p{p}.log", ticket(p)) for p in (0, 1, 5)])
check(rc7 == 0, "a gapped prompt set pools without a refusal (per-prompt tickets are the normal case)")
check("NOT a contiguous window: 2,3,4 missing of 0..5" in out7,
      f"the header names the hole instead of spanning it: {out7.splitlines()[0]}")
rc8, out8, _ = run([], [(f"q{p}.log", ticket(p)) for p in (0, 1, 2)])
check("prompts 0..2" in out8 and "NOT a contiguous" not in out8, "a contiguous pool keeps the plain range form")

# 8. the tool's own refusals, and (with logs) the recorded 8x64 known answer.
r = subprocess.run([sys.executable, TOOL, "--selftest"], capture_output=True, text=True)
check(r.returncode == 0 and "selftest PASS tail repeat" in r.stdout, "the tool's refusal self-test passes")

print(f"\n{'all checks passed' if not fails else f'{fails} CHECK(S) FAILED'}")
sys.exit(1 if fails else 0)
