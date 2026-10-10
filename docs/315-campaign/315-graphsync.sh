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
# engine#315: the visibility/ordering falsifier.
#
# Theory under test. The logits allocation is host-visible but NOT host-coherent (`hv52=1 hv56=0` in every
# driver line), so a host read of it depends on an invalidate/sync happening after the device's write. The
# two faces are what a reader sees when that ordering is wrong: NaN if the write has not landed (Face A,
# caught by the sampler guard) or the buffer's previous finite contents (Face B, which reaches the sampler,
# emits degenerate text, and fails at the chat parser). Making the *allocation* coherent did not remove the
# fault (315accept, 9 clean + 1 Face-B in 10 fix-cell arms, with hv56=1 proving the patch was live), so if
# the family is right at all, the defect is in the *timing* of the sync and not in the coherence bit.
#
# `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC` is the direct test: in command-program-executor.cpp:2034 it calls
# `hrx_stream_synchronize(context.stream)` after each graph replay, so every later read - including the
# host's read of the logits - must follow the device's writes. One sync per graph, so the cost is small.
#
# Readout, comparing against the arms already on disk from the same harness with no knob (the 315chain and
# 315nan sessions, which faulted with nan=1 and with nan=0/req<66 respectively):
#   faults fall to zero here -> the mechanism is a missing/early sync on the consumer side, and a scoped
#     variant of this sync is the minimal forward-only fix
#   the rate is unchanged      -> visibility/ordering is dead as a mechanism *for the producer*, and the
#     corruption is computed: a kernel/shape defect, which is the opposite direction to look in
#
# Deliberately NO other instrument in this run: the tensor fingerprint adds 16 device-to-host reads per
# graph and the copy probe prints from the submit path, either of which could perturb the very timing under
# test. Keep it clean so the comparison is against the no-knob baseline.
#
# Diagnostic, not acceptance: no registered criterion changes and no arm from it is eligible.
#
# Usage: 315-graphsync.sh [pairs] [tag]
set -uo pipefail
PAIRS=${1:-6}; TAG=${2:-315gsync}

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${GSYNC_LOCK_WAIT:-5400}" 9; then
  echo "gsync: box lock held $(date -Is)"
else
  echo "gsync: lock unobtainable after ${GSYNC_LOCK_WAIT:-5400}s - aborting rather than running concurrently"
  exit 3
fi

# Both slots get the same build and the knob applies to both cells, so a control cell would be a duplicate:
# run fix-slot arms only and compare against the no-knob arms already recorded by the same harness.
cd "$HOME/wt" || exit 1
REGIME_LO=${REGIME_LO:-70} REGIME_HI=${REGIME_HI:-85} \
CONTROL_EVERY=999 NOSTOP=1 MINCLEAN=999 NOLOCK=1 ARM_TIMEOUT=${ARM_TIMEOUT:-1800} \
EXTRA_ENV="GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1" \
  bash "$HOME/wt/315-paired.sh" "$PAIRS" "${NB:-2}" "${NCTX:-2048}" \
       "$HOME/wt/315instr3-build/bin" "$HOME/wt/315instr3-build/bin" "$TAG"

echo "=== gsync readout $(date -Is) ==="
# The driver's own verdict is authoritative per arm, so just show the arms and the tallies.
grep -hE "^  pair" "$HOME/wt/$TAG-console.log" || true
echo "clean: $(grep -c 'arm l: OK' "$HOME/wt/$TAG-console.log" || true)"
echo "fault-or-invalid: $(grep -c 'arm l: \(FAULT\|INVALID\)' "$HOME/wt/$TAG-console.log" || true)"
echo "out-of-regime: $(grep -c 'arm l: OOR' "$HOME/wt/$TAG-console.log" || true)"
echo "compare with the same harness and no knob: 315chain pair 1 faulted (nan=0, req<66) and 315nan"
echo "faulted 16/17 in the fix cell, so a zero-fault result here is the discriminating outcome."
