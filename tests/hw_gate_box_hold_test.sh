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
#
# tests/hw_gate_box_hold_test.sh -- on the 1bit OS appliance the gate queues with box-hold, not gpu-hold.
#
# The appliance has no gpu-hold and no ~/GPU-TOKEN protocol; its lanes take `box-hold gpu|npu <cmd>`. The repo's
# scripts/gpu-hold sits beside hw-gate.sh, so a gate that did not know about box-hold queued on a private token and ran
# its GPU steps beside the lanes. Hermetic: box-hold and gpu-hold are stubs on PATH, the helpers are EXTRACTED FROM THE
# SCRIPT and run (so this pins the shipped code), no device, no build.
#
#   1. box-hold on PATH: the whole-run gpu-hold wrapper is skipped (gpu-hold called 0 times)
#   2. HW_GATE_BOX_HOLD=none: the old gpu-hold wrapper is back (called once)
#   3. box_ticket: box-hold gets the device and a labelled BOX_HOLD_WHO, the step runs under `timeout`, the lock
#      descriptors 8 and 9 are CLOSED in the step (a survivor cannot keep the box), and the step's rc comes back
#   4. box_ticket passes box-hold's "lock never came round" rc 4 through
#   5. box_hold_ancestor: true under a box-hold parent, false otherwise (a gate inside a ticket must not re-queue)
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
gate=$root/scripts/hw-gate.sh
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
fails=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fails=$((fails + 1)); }

cat > "$tmp/bin/gpu-hold" <<'STUB'
#!/bin/sh
printf 'gpu-hold %s\n' "$*" >> "$STUBLOG"
exit 0
STUB
cat > "$tmp/bin/box-hold" <<'STUB'
#!/bin/bash
# like the real one: lock descriptors open, then run the command (not exec) and pass its rc
printf 'box-hold %s who=%s wait=%s prio=%s\n' "$1" "$BOX_HOLD_WHO" "$BOX_HOLD_WAIT" "${BOX_HOLD_PRIORITY:-}" >> "$STUBLOG"
[ -n "${STUB_RC:-}" ] && exit "$STUB_RC"
exec 8> "$STUBDIR/gpu.lock" 9> "$STUBDIR/npu.lock"
shift; "$@"
STUB
chmod +x "$tmp/bin/gpu-hold" "$tmp/bin/box-hold"
export STUBDIR=$tmp

# 1 + 2: the whole-run wrapper
log=$tmp/log1; : > "$log"
STUBLOG=$log PATH="$tmp/bin:$PATH" HOME=$tmp timeout 60 bash "$gate" --phases placement > "$tmp/out1" 2>&1
n=$(grep -c '^gpu-hold' "$log"); [ "$n" = 0 ] && ok "box-hold on PATH: no gpu-hold wrapper" || bad "box-hold on PATH but gpu-hold called $n time(s)"
log=$tmp/log2; : > "$log"
STUBLOG=$log PATH="$tmp/bin:$PATH" HOME=$tmp HW_GATE_BOX_HOLD=none timeout 60 bash "$gate" --phases placement > "$tmp/out2" 2>&1
n=$(grep -c '^gpu-hold' "$log"); [ "$n" = 1 ] && ok "HW_GATE_BOX_HOLD=none: gpu-hold wrapper as before" || bad "HW_GATE_BOX_HOLD=none: gpu-hold called $n time(s), wanted 1"

# the helpers, verbatim
fn=$tmp/fn.sh
export GPU_STEP_REAP="$(cd "$(dirname "$gate")" && pwd)/gpu-step-reap"   # box_ticket runs its step through it (the extracted functions do not carry the default)
for f in box_hold_ancestor box_ticket; do sed -n "/^$f() {/,/^}/p" "$gate"; done > "$fn"
grep -q '^box_ticket() {' "$fn" && grep -q '^box_hold_ancestor() {' "$fn" || { echo "FAIL: the helpers are no longer where this test finds them"; exit 1; }

# gpu() runs its one step call inside the held ticket when box-hold is present (text-level: gpu() needs a build to run;
# tests/hw_gate_held_ticket_test.sh drives the held ticket itself)
body=$(sed -n '/^gpu() {/,/^}/p' "$gate")
case "$body" in *'hold_for "${HW_GATE_STEP_EST_S:-600}"'*'ticket=("$GPU_STEP_REAP" "$budget")'*'"${ticket[@]}" "$0" --cool'*) ok "gpu() runs its step inside the held box-hold ticket, under the ticket's time left" ;; *) bad "gpu() does not route its step through hold_for" ;; esac
grep -q 'hold_args=(box_ticket npu hw-gate-npu ' "$gate" && grep -q 'hold_args=(box_ticket npu hw-gate-npuserve ' "$gate" \
  && ok "npu and npuserve phases take a box-hold npu ticket" || bad "an NPU phase has no box-hold ticket"

# 3: a ticket
log=$tmp/log3; : > "$log"
cat > "$tmp/step" <<'STEP'
#!/bin/bash
for fd in 8 9; do [ -e /proc/$$/fd/$fd ] && echo "fd $fd open" >> "$STUBLOG"; done
ps -o args= -p $PPID >> "$STUBLOG"
exit 7
STEP
chmod +x "$tmp/step"
STUBLOG=$log BOX_HOLD=$tmp/bin/box-hold BOX_HOLD_WAIT_S=1234 GITHUB_RUN_ID=42 bash -c ". '$fn'; box_ticket gpu hw-gate 45 '$tmp/step'"; rc=$?
[ "$rc" = 7 ] && ok "step rc passes through the ticket" || bad "ticket rc $rc, wanted 7"
grep -q "^box-hold gpu who=hw-gate run 42 pid [0-9]* wait=1234 prio=ci$" "$log" && ok "box-hold gpu, labelled, with the wait, CI class under CI" || bad "box-hold call: $(grep box-hold "$log")"
grep -q "fd [89] open" "$log" && bad "lock descriptor open in the step: $(grep 'fd ' "$log")" || ok "lock descriptors closed in the step"
grep -q "^timeout -k 30 2700 " "$log" && ok "step runs under timeout 2700 s (45 min)" || bad "step parent: $(tail -1 "$log")"

# 4: lock never came round -- and, run by hand (no GITHUB_RUN_ID; CI itself sets one), not in the CI class
log=$tmp/log4; : > "$log"
STUB_RC=4 STUBLOG=$log BOX_HOLD=$tmp/bin/box-hold BOX_HOLD_WAIT_S=1 env -u GITHUB_RUN_ID -u HW_GATE_BOX_HOLD_PRIORITY bash -c ". '$fn'; box_ticket npu hw-gate-npu 20 true"; rc=$?
[ "$rc" = 4 ] && grep -q "^box-hold npu who=hw-gate-npu pid [0-9]* wait=1 prio=$" "$log" && ok "rc 4 (no lock) passes through, npu ticket" || bad "no-lock rc $rc: $(cat "$log")"

# 5: ancestor detection
cat > "$tmp/probe" <<PROBE
#!/bin/bash
. '$fn'; box_hold_ancestor && echo inside || echo outside
PROBE
chmod +x "$tmp/probe"
got=$(STUBLOG=/dev/null "$tmp/bin/box-hold" gpu bash -c "'$tmp/probe'"); [ "$got" = inside ] && ok "under box-hold: ancestor found" || bad "under box-hold: $got"
got=$(bash -c "'$tmp/probe'"); [ "$got" = outside ] && ok "no box-hold: no ancestor" || bad "no box-hold: $got"

[ "$fails" = 0 ] && { echo "hw_gate_box_hold_test: pass"; exit 0; }
echo "hw_gate_box_hold_test: $fails failure(s)"; exit 1
