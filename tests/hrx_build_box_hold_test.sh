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
# tests/hrx_build_box_hold_test.sh -- hrx-build's device rows take a box-hold gpu ticket on the appliance.
#
# Since 2026-11-26 the runner's account can open /dev/kfd, so `fork_ctest_known_fail` uses the GPU; on the appliance it
# must queue with the lanes. The snippet is EXTRACTED FROM .github/workflows/hrx-build.yml and run with stub box-hold and
# ctest, so this pins the shipped step. Arms: box-hold on PATH -> one gpu ticket, labelled, the ctest env passes through,
# lock fds closed, under timeout 2700; rc 4 -> the step fails saying no ticket; no box-hold -> ctest runs bare; a failing
# row fails the step.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
wf=$root/.github/workflows/hrx-build.yml
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fails=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fails=$((fails + 1)); }
snip=$tmp/snip.sh
sed -n '/^          ticket=()$/,/^          \[ "\$rc" = 0 \]/p' "$wf" | sed 's/^          //' > "$snip"
grep -q 'box-hold gpu' "$snip" && grep -q '^\[ "\$rc" = 0 \]' "$snip" || { echo "FAIL: the device-row snippet is no longer where this test finds it"; exit 1; }
mkdir -p "$tmp/bh" "$tmp/plain"
cat > "$tmp/bh/box-hold" <<'STUB'
#!/bin/bash
echo "box-hold $1 who=$BOX_HOLD_WHO wait=$BOX_HOLD_WAIT" >> "$STUBLOG"
[ -n "${STUB_RC:-}" ] && exit "$STUB_RC"
exec 8> "$STUBDIR/l8" 9> "$STUBDIR/l9"; shift; "$@"
STUB
cat > "$tmp/ctest" <<'STUB'
#!/bin/bash
for fd in 8 9; do [ -e /proc/$$/fd/$fd ] && echo "fd $fd open" >> "$STUBLOG"; done
echo "ctest fork=${ONEBIT_FORK_CTEST:-} parent=$(ps -o args= -p $PPID)" >> "$STUBLOG"
exit "${CTEST_RC:-0}"
STUB
chmod +x "$tmp/bh/box-hold" "$tmp/ctest"; cp "$tmp/ctest" "$tmp/bh/"; cp "$tmp/ctest" "$tmp/plain/"
run() { ( set -euo pipefail; GITHUB_RUN_ID=77 HRX_BUILD=$tmp; . "$snip"; echo "step-ok" ) > "$tmp/out" 2>&1; }
export STUBDIR=$tmp

STUBLOG=$tmp/l1 PATH="$tmp/bh:$PATH" run; rc=$?
[ "$rc" = 0 ] && grep -q step-ok "$tmp/out" && ok "box-hold present: step passes" || bad "step rc $rc: $(cat "$tmp/out")"
grep -q "^box-hold gpu who=hrx-build run 77 wait=5400$" "$tmp/l1" && ok "one gpu ticket, labelled, 90-min wait" || bad "ticket: $(cat "$tmp/l1")"
grep -q "ctest fork=1 parent=timeout -k 30 2700 ctest" "$tmp/l1" && ok "ctest under timeout 2700 with ONEBIT_FORK_CTEST=1" || bad "ctest: $(grep ctest "$tmp/l1")"
grep -q "fd [89] open" "$tmp/l1" && bad "lock fd open in the rows" || ok "lock fds closed in the rows"

STUB_RC=4 STUBLOG=$tmp/l2 PATH="$tmp/bh:$PATH" run; rc=$?
[ "$rc" != 0 ] && grep -q "no GPU ticket within 90 min" "$tmp/out" && ok "rc 4: the step fails, saying no ticket" || bad "no-ticket: rc $rc $(cat "$tmp/out")"

STUBLOG=$tmp/l3 PATH="$tmp/plain:/usr/bin:/bin" run; rc=$?
[ "$rc" = 0 ] && ! grep -q box-hold "$tmp/l3" && grep -q "ctest fork=1" "$tmp/l3" && ok "no box-hold: ctest runs bare" || bad "bare: rc $rc $(cat "$tmp/l3")"

CTEST_RC=8 STUBLOG=$tmp/l4 PATH="$tmp/bh:$PATH" run; rc=$?
[ "$rc" != 0 ] && ! grep -q step-ok "$tmp/out" && ok "a failing row fails the step" || bad "failing row passed: $(cat "$tmp/out")"

[ "$fails" = 0 ] && { echo "hrx_build_box_hold_test: pass"; exit 0; }
echo "hrx_build_box_hold_test: $fails failure(s)"; exit 1
