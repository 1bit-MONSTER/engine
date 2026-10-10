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
# engine#315 dose-response for the graph sync. Runs ONLY if 315gsync2's fix cell came back CLEAN.
#
# Why this exists (section 57's pre-commitment, written before gsync2's result): a binary cell can say "the sync
# removes the fault" but not "the fault scales with how often the consumer waits". If the binary cell is clean,
# the honest next question is whether the effect is monotone in the dose - because a clean cell with no dose
# response is more consistent with the sync perturbing the *inducer's* timing than with the sync repairing an
# ordering defect.
#
# `tools/patch-sync-dose.py` adds GGML_HRX_DIAGNOSTIC_GRAPH_SYNC_EVERY=N (synchronise every Nth replay; N=1 is
# the existing binary knob, a large N is effectively the no-knob cell). This script runs three doses, each in its
# own session so each takes and releases the box lock rather than holding it for the whole sweep.
#
# Readout:
#   fault rate falls monotonically as N falls   -> the measured quantity tracks the wait: the ordering is the
#                                                  carrier and the mechanism is named
#   rate flat across N while gsync2's cell was clean -> the sync's *effect* is not dose-dependent, so the clean
#                                                  cell is not evidence that waiting repairs the fault
#
# Diagnostic, not acceptance: no registered criterion changes and no arm from it is eligible.
#
# Usage: 315-dose.sh [arms-per-dose] [doses...]
set -uo pipefail
ARMS=${1:-3}; shift || true
DOSES=("$@"); [ ${#DOSES[@]} -eq 0 ] && DOSES=(2 8 64)
SRC=$HOME/wt/llama-315copy
B=$HOME/wt/315dose-build
HRX=$HOME/wt/pin-fork-hrx/third_party/hrx-system
LOG=$HOME/wt/315dose-build.log

# Pre-committed gate: this run is for the branch gsync2 did NOT close. gsync2 may still be queued behind another
# session when this script starts, so wait for its runner to exit before reading its console - evaluating the
# gate on a half-written file would abort the branch that may still be needed.
for _i in $(seq 1 300); do
  pgrep -f "315-attrib.sh .* 315gsync2" >/dev/null 2>&1 || break
  [ "$_i" -eq 1 ] && echo "dose: waiting for the gsync2 runner to finish before evaluating the gate"
  sleep 60
done
C=$HOME/wt/315gsync2-console.log
if [ -f "$C" ]; then
  fixok=$(grep -c 'arm l: OK' "$C" || true)
  fixf=$(grep -c 'arm l: \(FAULT\|INVALID\)' "$C" || true)
  ctlok=$(grep -c 'arm c: OK' "$C" || true)
  ctlf=$(grep -c 'arm c: \(FAULT\|INVALID\)' "$C" || true)
  echo "dose: gsync2 - control OK=$ctlok fault=$ctlf | fix OK=$fixok fault=$fixf"
  if [ "${ctlok:-0}" -eq 0 ] && [ "${ctlf:-0}" -eq 0 ]; then
    echo "dose: ABORT - gsync2 produced no scored arms; nothing to follow up"
    exit 0
  fi
  if [ "${fixf:-0}" -ne 0 ] || [ "${fixok:-0}" -lt 3 ]; then
    echo "dose: ABORT - gsync2's fix cell faulted or is too small; the ordering family is already closed there"
    echo "dose: the gated ordered locator (315-fp2.sh) is the run for that branch"
    exit 0
  fi
  echo "dose: gsync2's fix cell was CLEAN with a faulting control; running the dose sweep"
else
  echo "dose: ABORT - no gsync2 console; the gate cannot be checked"
  exit 0
fi

grep -q "diagnostic_sync_dose" "$SRC/ggml/src/ggml-hrx/runtime/command-program-executor.cpp" \
  || { echo "dose: ABORT - the dose patch is not in the source (tools/patch-sync-dose.py)"; exit 1; }

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${DOSE_LOCK_WAIT:-10800}" 9; then
  echo "dose: box lock held for the build $(date -Is)"
else
  echo "dose: lock unobtainable - aborting rather than building concurrently"
  exit 3
fi

echo "dose: configuring $B $(date -Is)"
rm -rf "$B"
cmake -S "$SRC" -B "$B" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/rocm-therock/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/opt/rocm-therock/bin/amdclang++ \
  -DGGML_HIP=OFF -DGGML_HRX=ON -DGGML_VULKAN=OFF \
  -DHRX_SOURCE_DIR="$HRX" \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DBUILD_TESTING=OFF \
  > "$LOG" 2>&1 || { echo "dose: cmake failed"; tail -20 "$LOG"; exit 1; }
echo "dose: building -j 6 $(date -Is)"
cmake --build "$B" -j 6 >> "$LOG" 2>&1 || { echo "dose: build failed"; tail -30 "$LOG"; exit 1; }
"$B/bin/llama-server" --version || exit 1
grep -qa "diagnostic_sync_dose" "$B/bin/llama-server" || { echo "dose: ABORT - the dose branch is not in the binary"; exit 1; }
echo "dose: dose branch present; binary sha256 $(sha256sum "$B/bin/llama-server" | cut -c1-16)"
flock -u 9

# Each dose is its own session: it takes the lock, runs, and releases, so the box is never held for the sweep.
for N in "${DOSES[@]}"; do
  tag="315dose-n$N"
  echo "=== dose session N=$N arms=$ARMS $(date -Is) ==="
  CBIN="$B/bin" FBIN="$B/bin" \
  REGIME_LO=${REGIME_LO:-80} REGIME_HI=${REGIME_HI:-92} \
  ARM_TIMEOUT=${ARM_TIMEOUT:-1800} LOCK_WAIT=${DOSE_LOCK_WAIT:-10800} \
    bash "$HOME/wt/315-attrib.sh" "$ARMS" "$tag" 999 \
    "GGML_HRX_DIAGNOSTIC_GRAPH_SYNC_EVERY=$N GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1"
  c=$HOME/wt/$tag-console.log
  echo "  N=$N clean=$(grep -c 'arm l: OK' "$c" || true) fault=$(grep -c 'arm l: \(FAULT\|INVALID\)' "$c" || true) oor=$(grep -c 'arm l: OOR' "$c" || true)"
  grep -hE "^  pair" "$c" | tail -3
done

echo "=== dose readout $(date -Is) ==="
for N in "${DOSES[@]}"; do
  c=$HOME/wt/315dose-n$N-console.log
  [ -f "$c" ] || continue
  echo "N=$N  clean=$(grep -c 'arm l: OK' "$c" || true)  fault=$(grep -c 'arm l: \(FAULT\|INVALID\)' "$c" || true)  oor=$(grep -c 'arm l: OOR' "$c" || true)  gtt=$(grep -oE 'gtt_start=[0-9]+G' "$c" | sort -u | tr '\n' ' ')"
done
echo "Read the rate against N: monotone means the wait is the carrier; flat means the clean binary cell was not."
