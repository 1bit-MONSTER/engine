#!/bin/sh
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
# scripts/thermal-run.sh, the governor both hardware workflows run their builds under (engine#3518):
#   1. the job's exit code passes through;
#   2. a job frozen at THERMAL_HI is thawed (SIGCONT) when the sensor becomes unreadable, and the read failure is
#      reported. Before the fix the loop skipped empty readings, so the job stayed frozen until the job timeout.
# A fake sensor file (THERMAL_TCTL) and a short poll: no hardware, no root, a few seconds.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
tr_sh=$here/scripts/thermal-run.sh
t=$(mktemp -d) || exit 1
trap 'rm -rf "$t"' EXIT
bad=0
check() { if [ "$1" = 0 ]; then echo "ok   $2"; else echo "FAIL $2"; bad=$((bad + 1)); fi; }

# 1. exit code passthrough on a cool box
echo 50000 > "$t/temp"
THERMAL_TCTL=$t/temp THERMAL_POLL=0.1 bash "$tr_sh" sh -c 'exit 7' 2> "$t/err1"
rc=$?
[ "$rc" = 7 ]; check $? "the job's exit code passes through (rc $rc)"

# 2. frozen at 85 C, then the sensor disappears: the job must be thawed and finish
echo 85000 > "$t/temp"
: > "$t/count"
THERMAL_TCTL=$t/temp THERMAL_POLL=0.1 timeout 20 bash "$tr_sh" \
    sh -c 'i=0; while [ $i -lt 20 ]; do i=$((i + 1)); echo $i >> "$1"; sleep 0.05; done' sh "$t/count" 2> "$t/err2" &
pid=$!
for _ in $(seq 1 50); do grep -q "frozen" "$t/err2" 2> /dev/null && break; sleep 0.1; done
grep -q "frozen" "$t/err2"; check $? "the job is frozen at 85 C"
n1=$(wc -l < "$t/count"); sleep 0.5; n2=$(wc -l < "$t/count")
[ "$n1" = "$n2" ]; check $? "a frozen job makes no progress ($n1 -> $n2 lines)"
rm -f "$t/temp"
wait "$pid"; rc=$?
[ "$rc" = 0 ]; check $? "after the sensor disappears the job is thawed and finishes (rc $rc; 124 = still frozen at the timeout)"
n=$(wc -l < "$t/count")
[ "$n" = 20 ]; check $? "the job ran to the end ($n of 20 lines)"
grep -q "cannot read" "$t/err2"; check $? "the read failure is reported"
grep -q "thawed" "$t/err2"; check $? "the thaw is reported"

echo "thermal_run_test: $([ $bad = 0 ] && echo ok || echo FAIL)"
[ $bad = 0 ]
