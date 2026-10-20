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
# engine#315 Face-B probe. Diagnostic, never acceptance-eligible.
#
# Face B's landscape from the 142 recorded arms: 7 events, all in-band at gtt=98, all aborting early
# (req 13-47), **none** divergent, and zero of the 29 out-of-band arms - so it shows no evidence of
# tracking the regime, and no arm that produced one had the logits fingerprint enabled. What Face B is
# for certain is the server's own chat-template parse failing: "Content-only" is the parser name
# (common/chat.cpp:812) and the 500 comes from the parse, i.e. the model emitted text that does not fit
# the format the engine expects.
#
# The discriminating question is whether that text is downstream of NaN logits. `libllama-common.so`
# implements both switches needed to ask it:
#   GGML_HRX_LOGITS_FP=1     logits fingerprint at every sample: nan= and sentinel= counts
#   GGML_HRX_NAN_CONTINUE=1  keep sampling instead of erroring when the logits contain NaN
#
# So this run makes the continuation *induced* - a Face-A precursor is forced to continue and we watch
# what the engine's parser does with the result. Read it as a causal chain, not as a natural Face-B
# reproduction:
#   - 500s appear only in arms that also logged NaN logits, and after them  -> one mechanism, two
#     observables: the guard's choice (abort vs continue) decides which face you see
#   - 500s appear in arms whose logits were NaN-free                        -> Face B is a separate
#     finite-corruption path, downstream of the LM head, and must be named on its own
set -uo pipefail
PAIRS=${1:-8}; TAG=${2:-315faceb}

exec 9>"$HOME/.cache/lax-decode/box.lock"
if flock -w "${FACE_KEY:-5400}" 9; then
  echo "faceb: box lock held $(date -Is)"
else
  echo "faceb: lock unobtainable after ${FACE_KEY:-5400}s - aborting rather than running concurrently"
  exit 3
fi

cd "$HOME/wt" || exit 1
REGIME_LO=${REGIME_LO:-80} REGIME_HI=${REGIME_HI:-92} \
CONTROL_EVERY=999 NOSTOP=1 MINCLEAN=999 NOLOCK=1 ARM_TIMEOUT=${ARM_TIMEOUT:-1200} \
EXTRA_ENV="GGML_HRX_LOGITS_FP=1 GGML_HRX_NAN_CONTINUE=1 GGML_HRX_SENTINEL=1" \
  bash "$HOME/wt/315-paired.sh" "$PAIRS" "${NB:-2}" "${NCTX:-2048}" \
       "$HOME/wt/315instr3-build/bin" "$HOME/wt/315fp-build/bin" "$TAG"

echo "=== faceb readout $(date -Is) ==="
printf "%-10s %-6s %-8s %-10s %-8s %-8s\n" arm verdict "nan_logits" "500_format" "sentinel" "divergent"
for d in "$HOME"/wt/rt-det/"$TAG"-*/; do
  [ -d "$d" ] || continue
  a=$(basename "$d" | sed "s/${TAG}-//;s/-GLM.*//")
  nl=$(grep -c "hrx-logits-fp" "$d/server.log" 2>/dev/null || echo 0)
  nln=$(grep "hrx-logits-fp" "$d/server.log" 2>/dev/null | grep -vc " nan=0 " || echo 0)
  fb=$(grep -c "Content-only" "$d/server.log" 2>/dev/null || echo 0)
  sn=$(grep -c "hrx-sentinel" "$d/server.log" 2>/dev/null || echo 0)
  idn=$(python3 -c "import json;print(json.load(open('$d/detector.json'))['identical_distinct'])" 2>/dev/null || echo "?")
  printf "%-10s %-6s %-8s %-10s %-8s %-8s\n" "$a" "-" "$nln/$nl" "$fb" "$sn" "$idn"
done
echo "NaN-logits arms and format-500 arms, side by side: a 500 in an arm with zero NaN-logits lines is"
echo "evidence for a separate Face-B path; a 500 only where NaN was logged is evidence they are one."
