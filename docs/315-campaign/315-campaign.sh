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
# Pre-registered settings (docs/315-preregistration.md sections 3, 6, 7, 18b):
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
echo "campaign: tag=$TAG pairs=$PAIRS target_clean=$MINCLEAN regime_LO=${REGIME_LO:-70} regime_HI=${REGIME_HI:-85} EXTRA_ENV=<empty>"

REGIME_LO=${REGIME_LO:-70} REGIME_HI=${REGIME_HI:-85} MINCLEAN="$MINCLEAN" \
  bash "$HOME/wt/315-paired.sh" "$PAIRS" 2 2048 "$CTRL/bin" "$FIX/bin" "$TAG" 2>&1 | tail -12

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
    rows.append(dict(pair=int(m.group(1)), cell=m.group(2), verdict=m.group(3), nan=int(m.group(4)),
                     req=int(m.group(5)), parse=int(m.group(6)), gtt=int(m.group(7)), utf8=utf8, alloc=alloc))

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
        print(f"   FAULT pair={r['pair']} nan={r['nan']} req={r['req']} parse={r['parse']} utf8={r['utf8']} gtt={r['gtt']}G")
    if clean:
        n = len(clean)
        # rule of three: 0 faults in n arms => 95% upper bound on the rate
        upper = 1 - 0.05 ** (1.0 / n)
        print(f"   rate=0/{n} = 0.0%  one-sided 95% upper bound (rule of three, 3/n) = {3.0/n*100:.2f}%"
              f"  (0.95^{n} = {0.95**n*100:.2f}%)")
    return clean, faults, invalid

for cell in ("l", "c"):
    summarise(cell)
PY
