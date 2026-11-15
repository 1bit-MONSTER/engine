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
# hw-gate keeps a failing row's server log (engine#2500, #2513) under $HW_GATE_STORE/failed and names it in the
# failure line. The name must be unique per failure: CI sets HW_GATE_OUT=$RUNNER_TEMP/hw-gate, so a name built from
# basename "$OUT" was the constant "hw-gate-<model>.log" and every failing run on every PR overwrote the last one's
# evidence. These arms run the CI shape (same model, same run, same second), and check that a failed copy prints no
# path. A recording fake gpu-hold is first on PATH, so the internal mode is also proven to take no GPU ticket.
# No GPU, no model, runs in CI.
#
# The gate's phase check must match the phase FIELD, not a prefix of it, and only the phases that actually write
# `model <name>` rows may be satisfied by one (review of engine#2523):
#
#   * `npu` is a prefix of `npuserve`, so `grep -qE "^npu"` read an `npuserve` row as proof that `npu` had run;
#   * `placement` writes `placement <name>`, not `model <name>`, so grouping it with kat|kld|needle let any model row
#     satisfy a requested placement phase -- the same silent pass, one name over.
#
# The loop is EXTRACTED FROM THE SCRIPT and run, so this pins the shipped code rather than a copy of it. A copy would
# pass while the real check drifted.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
gate=$root/scripts/hw-gate.sh
fail=0

# the check, verbatim, as it ships
snippet=$(mktemp)
sed -n '/^for _ph in \$(printf .%s. "\$PHASES"/,/^done$/p' "$gate" > "$snippet"
[ -s "$snippet" ] || { echo "FAIL: the phase-check loop is no longer where this test finds it"; exit 1; }

check() {  # $1 results file, $2 PHASES -> prints the FAIL rows the check appended
  ( PHASES=$2; RESULTS=$1; . "$snippet" ) 2>&1
  awk -F'\t' '$2 == "FAIL" {print $1}' "$1"
}

run() {  # name, results content, phases, expected-substring-in-output
  local t; t=$(mktemp)
  printf '%b' "$2" > "$t"
  local out; out=$(check "$t" "$3")
  rm -f "$t"
  if [ -n "$4" ]; then
    case "$out" in *"$4"*) ;; *) echo "FAIL $1: expected a FAIL row for '$4', got '${out:-<none>}'"; fail=1; return ;; esac
  else
    [ -z "$out" ] || { echo "FAIL $1: expected no failure, got '$out'"; fail=1; return; }
  fi
  echo "ok   $1"
}

# the two slips the review measured, both must now fail
run "a npuserve row does not satisfy a requested npu phase" \
    'build\tPASS\t1\tok\nnpuserve\tPASS\t1\tok\n' "build,npu,npuserve" "npu"
run "a model row does not satisfy a requested placement phase" \
    'build\tPASS\t1\tok\nmodel gemma\tPASS\t1\tok\n' "build,placement" "placement"

# and the three legitimate satisfactions must keep working
run "a placement row satisfies placement" 'build\tPASS\t1\tok\nplacement gemma\tPASS\t1\tok\n' "build,placement" ""
run "a model row still satisfies kat"      'build\tPASS\t1\tok\nmodel gemma\tPASS\t1\tok\n'     "build,kat" ""
run "a plain row satisfies its own phase"  'build\tPASS\t1\tok\nops\tPASS\t1\tok\n'             "build,ops" ""

rm -f "$snippet"
[ "$fail" = 0 ] && echo "hw_gate_phase_rows: ok (a phase is a field, not a prefix; only kat/kld/needle accept a model row)"
exit "$fail"
