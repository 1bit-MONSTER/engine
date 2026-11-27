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
# tests/hw_gate_inputs_test.sh -- hw-gate.yml's prepare step points the gate at the appliance's read-only inputs
# (/data/ci/hw-gate: models + text, KLD bases). The snippet is EXTRACTED FROM THE WORKFLOW and run with HW_GATE_INPUTS
# and HOME in a temp dir. Arms: models dir -> HW_GATE_MODELS; an empty $STORE/kld -> a link to the store; an existing
# link is refreshed; a $STORE/kld with files in it is LEFT ALONE (a lane's own bases are never deleted); no inputs ->
# nothing written.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
wf=$root/.github/workflows/hw-gate.yml
T=$(mktemp -d) || exit 1
trap 'rm -rf "${T:?}"' EXIT
fails=0; ok() { echo "ok   $*"; }; bad() { echo "FAIL $*"; fails=$((fails + 1)); }
snip=$T/snip.sh
# the block is two `if ... fi`: take it from the inputs= line through the SECOND top-level fi
awk '/^          fi$/ {n++} {print} n==2 {exit}' <(sed -n '/^          inputs=\${HW_GATE_INPUTS/,$p' "$wf") | sed 's/^          //' > "$snip"
grep -q 'HW_GATE_MODELS=' "$snip" && grep -q 'ln -s "$inputs/kld" "$store/kld"' "$snip" || { echo "FAIL: the inputs snippet is no longer where this test finds it"; exit 1; }
run() { ( set -euo pipefail; GITHUB_ENV=$T/env HOME=$T/home HW_GATE_INPUTS=$T/inputs; unset HW_GATE_STORE; . "$snip" ) > "$T/out" 2>&1; }
K=$T/home/.cache/hw-gate/kld

mkdir -p "$T/inputs/models" "$T/inputs/kld" "$K"; : > "$T/env"
run; rc=$?
[ "$rc" = 0 ] && grep -qx "HW_GATE_MODELS=$T/inputs/models" "$T/env" && ok "models dir -> HW_GATE_MODELS" || bad "models: rc $rc env [$(cat "$T/env")] $(cat "$T/out")"
[ -L "$K" ] && [ "$(readlink "$K")" = "$T/inputs/kld" ] && ok "an empty \$STORE/kld becomes a link to the store" || bad "kld not linked: $(ls -la "$T/home/.cache/hw-gate")"
: > "$T/env"; run; rc=$?
[ "$rc" = 0 ] && [ "$(readlink "$K")" = "$T/inputs/kld" ] && ok "an existing link is refreshed (idempotent)" || bad "second run: rc $rc"
rm -f "${K:?}"; mkdir "$K"; echo mine > "$K/lane.kld"
run; [ -d "$K" ] && [ ! -L "$K" ] && [ -f "$K/lane.kld" ] && ok "a \$STORE/kld with files in it is left alone" || bad "a non-empty kld dir was replaced"
rm -rf "${T:?}/inputs" "${T:?}/home"; : > "$T/env"; run; rc=$?
[ "$rc" = 0 ] && [ ! -s "$T/env" ] && [ ! -e "$K" ] && ok "no inputs on this runner: nothing written" || bad "no-inputs: rc $rc env [$(cat "$T/env")]"

[ "$fails" = 0 ] && { echo "hw_gate_inputs_test: pass"; exit 0; }
echo "hw_gate_inputs_test: $fails failure(s)"; exit 1
