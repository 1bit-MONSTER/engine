#!/usr/bin/env bash
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
# engine#315: in-regime acceptance campaign runner + verdict summary.
#
# Pre-registered settings (dev/315/315-preregistration.md sections 3, 6, 7, 18b):
#   * regime: REGIME_LO/HI tuned so the arm lands in [95, 110) GiB with the resident inventory; the arm's
#     own gtt_start is what decides membership (gate >= 95 GiB).
#   * EXTRA_ENV is EMPTY: the campaign measures the shipped configuration, so the sentinel fill, the copy
#     probe and the fingerprints are OFF (section 18b). Instrumented runs are diagnostics, not acceptance.
#   * stop at the first fix-cell fault (NOSTOP unset), target MINCLEAN clean fix-cell arms, cap PAIRS.
#
# Usage: FIX=<fix bin dir> [CTRL=<control bin dir>] [TAG=<tag>] [PAIRS=80] [MINCLEAN=60] 315-campaign.sh
set -uo pipefail

FIX=${FIX:?fix bin dir}
CTRL=${CTRL:-$HOME/wt/315instr3-build}
TAG=${TAG:-315fix}
PAIRS=${PAIRS:-80}
MINCLEAN=${MINCLEAN:-60}

test -x "$FIX/bin/llama-server" || { echo "ABORT: no $FIX/bin/llama-server"; exit 2; }
test -x "$CTRL/bin/llama-server" || { echo "ABORT: no $CTRL/bin/llama-server"; exit 2; }
echo "fix  binary: $("$FIX/bin/llama-server" --version 2>&1 | head -1)"
echo "ctrl binary: $("$CTRL/bin/llama-server" --version 2>&1 | head -1)"
echo "campaign: tag=$TAG pairs=$PAIRS target_clean=$MINCLEAN regime_LO=${REGIME_LO:-80} regime_HI=${REGIME_HI:-92} EXTRA_ENV=<empty>"

REGIME_LO=${REGIME_LO:-80} REGIME_HI=${REGIME_HI:-92} MINCLEAN="$MINCLEAN" \
  bash "$HOME/wt/315-paired.sh" "$PAIRS" 2 2048 "$CTRL/bin" "$FIX/bin" "$TAG" 2>&1 | tail -12
# The "2" above is the driver's <neighbours> argument and it is INERT: the driver reads it only to print it
# in the session header, and composition is done by 315-regime.sh, which starts neighbours one at a time
# against live GTT until it reaches REGIME_LO and never past REGIME_HI (1..6). So the pre-registration's
# "3 detached neighbour processes" is the typical outcome at LO=80, not a parameter, and the literal here is
# not a criterion. Recorded because a header line reading "neighbours=2" next to a document saying three
# would otherwise look like a criteria change after the fact.

CONSOLE="$HOME/wt/$TAG-console.log"
echo "=== verdict summary (from the driver's own per-arm lines, which implement section 5)"
python3 - "$CONSOLE" <<'PY'
import re, sys
from math import comb
path = sys.argv[1]
arm = re.compile(r"^\s*pair\s+(\d+)\s+arm\s+([cl]):\s+(\S+)\s+nan=(\d+)\s+req=(\d+)\s+ret3=\d+\s+parse=(\d+)\s+gtt_start=(\d+)G(.*)$")
rows = []
for line in open(path, errors="replace"):
    m = arm.match(line)
    if not m:
        continue
    tail = m.group(8)
    utf8 = int((re.search(r"utf8=(\d+)", tail) or [0, 0])[1])
    alloc = int((re.search(r"alloc=(\d+)", tail) or [0, 0])[1])
    # engine#315: the disc rig's arm line carries the fields Face B actually sets; the paired rig's line does
    # not. Parsed here when present so a disc-rig console summarises correctly, and recorded as absent (None)
    # otherwise so an unattributed fault is never silently read as "not Face B".
    def tail_int(name):
        m2 = re.search(name + r"=(\d+)", tail)
        return int(m2.group(1)) if m2 else None
    http  = tail_int("http")
    s500  = tail_int("s500")
    unparsed = tail_int("unparsed")
    rows.append(dict(pair=int(m.group(1)), cell=m.group(2), verdict=m.group(3), nan=int(m.group(4)),
                     req=int(m.group(5)), parse=int(m.group(6)), gtt=int(m.group(7)), utf8=utf8, alloc=alloc,
                     http=http, s500=s500, unparsed=unparsed))

# engine#315: Face B is Content-only / non-JSON output answered with HTTP 500 (the recorded form), or a body
# that fails the full-body UTF-8 check (the objective's form). Both are tested, per the user's ruling that
# either manifestation counts. A field must be PRESENT to contribute, so a paired-rig line without these
# fields cannot manufacture a false Face B.
def is_face_b(r):
    return (r["parse"] > 0 or r["utf8"] > 0
            or (r["http"] or 0) > 0 or (r["s500"] or 0) > 0 or (r["unparsed"] or 0) > 0)

def summarise(cell):
    sel = [r for r in rows if r["cell"] == cell]
    inb = [r for r in sel if r["gtt"] >= 95]
    oor = [r for r in sel if r["gtt"] < 95]
    clean = [r for r in inb if r["verdict"] == "OK"]
    faults = [r for r in inb if r["verdict"] == "FAULT"]
    invalid = [r for r in inb if r["verdict"] == "INVALID"]
    print(f"cell {cell}: arms={len(sel)} in-regime={len(inb)} out-of-regime={len(oor)} "
          f"clean={len(clean)} fault={len(faults)} invalid={len(invalid)}")
    for r in faults[:6]:
        faces = []
        if r["nan"] > 0:
            faces.append("FaceA(nan=%d)" % r["nan"])
        if is_face_b(r):
            faces.append("FaceB(parse=%d,utf8=%d,http=%s,s500=%s,unparsed=%s)"
                         % (r["parse"], r["utf8"], r["http"], r["s500"], r["unparsed"]))
        if not faces:
            print("   NOTE: this line carries no http/s500/unparsed field, so Face B's recorded form cannot"
                  " be tested from it - the paired rig does not emit them. Take the face attribution from a"
                  " disc-rig console, or add the fields to the paired rig's verdict printer.")
            faces.append("unattributed - FAULT with nan=0 and no parse/utf8 counter; read the server log"
                          " (an HSA_STATUS_ERROR_MEMORY_FAULT is Face A and is not on this line)")
        print(f"   FAULT pair={r['pair']} {' '.join(faces)} req={r['req']} gtt={r['gtt']}G")
    face_a = sum(1 for r in faults if r["nan"] > 0)
    face_b = sum(1 for r in faults if is_face_b(r))
    unattr = sum(1 for r in faults if r["nan"] == 0 and not is_face_b(r))
    if faults:
        print(f"   face tally: FaceA={face_a} FaceB={face_b} unattributed={unattr}"
              f"  (the criterion is zero on BOTH faces; HSA faults are unattributed here by construction)")
    # The acceptance statistic runs over ALL in-regime arms, not over the clean ones. The first version of
    # this summary divided by the clean count and hardcoded a zero numerator, so it printed "rate=0/N = 0.0%"
    # and a rule-of-three bound no matter how many arms had faulted - a fault was visible only in the
    # separate count. For an acceptance decision that is exactly backwards: a faulted arm is evidence about
    # the rate and has to be in the denominator.
    n = len(inb)
    k = len(faults)
    if n == 0:
        print("   rate=undefined (no in-regime arms)")
    elif k == 0:
        print(f"   rate=0/{n} = 0.0%  one-sided 95% upper bound (rule of three, 3/n) = {3.0/n*100:.2f}%"
              f"  (0.95^{n} = {0.95**n*100:.2f}%)")
    else:
        # Exact Clopper-Pearson 95% interval by bisection on the binomial tail, so no scipy is needed.
        # P(X >= k | p) increases in p and P(X <= k | p) decreases in p, so each bound is the crossing of
        # its tail probability with alpha/2.
        def cdf(p, kk):
            return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, kk + 1))
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if 1.0 - cdf(mid, k - 1) > 0.025: hi = mid
            else: lo = mid
        lower = 0.5 * (lo + hi)
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if cdf(mid, k) > 0.025: lo = mid
            else: hi = mid
        upper = 0.5 * (lo + hi)
        print(f"   rate={k}/{n} = {k/n*100:.2f}%  exact 95% CI [{lower*100:.2f}%, {upper*100:.2f}%]")
        print(f"   ACCEPTANCE FAILS: {k} in-regime arm(s) faulted; the criterion is zero faults across both faces")
    if invalid:
        print(f"   {len(invalid)} in-regime arm(s) were INVALID, i.e. neither clean nor faulted - they are not")
        print(f"   clean evidence and have to be resolved before the criterion can be read as met")
    return clean, faults, invalid

for cell in ("l", "c"):
    summarise(cell)
PY
