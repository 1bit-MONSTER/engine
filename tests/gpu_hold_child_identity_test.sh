#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
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
# SPDX-License-Identifier: Apache-2.0
#
# gpu_hold_child_identity_test.sh - the holder can PROVE it holds the ticket, and a bystander cannot.
#
# Why: `[ -e ~/GPU-TOKEN ]` is true whenever ANY lane holds the box, so a guard written with it passes exactly when
# starting work is most harmful. Measured 2026-11-14: a lane's runner used that check, another lane held a 45-minute
# ticket, and the runner served on the GPU for 40 s inside someone else's hold. gpu-hold hands its child the exact line
# it wrote into the token (GPU_HOLD_TICKET) and the label (GPU_HOLD_LABEL); this test asserts both, on a sandboxed HOME so
# the real ~/GPU-QUEUE and ~/GPU-TOKEN are never touched.
#
# usage: gpu_hold_child_identity_test.sh <path-to-gpu-hold>
set -u
HOLD=${1:?usage: gpu_hold_child_identity_test.sh <path-to-gpu-hold>}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ghid.XXXXXX") || exit 77
trap 'rm -rf "$tmp"' EXIT
fails=0

out=$(HOME="$tmp" ONEBIT_HOLD_DIR="$tmp/hold" GPU_HOLD_COOLDOWN=0 GPU_HOLD_RELEASE_WAIT=0 \
      GPU_HOLD_WAIT=20 GPU_HOLD_MINUTES=1 GPU_HOLD_THERMAL_FLAG="$tmp/none" \
      timeout 60 "$HOLD" -l child-identity -- \
      bash -c 'printf "TICKET=%s\nLABEL=%s\nINFILE=%s\n" "${GPU_HOLD_TICKET:-}" "${GPU_HOLD_LABEL:-}" "$(cat "$1" 2>/dev/null)"' _ "$tmp/GPU-TOKEN" 2>&1)
rc=$?

ticket=$(printf '%s\n' "$out" | sed -n 's/^TICKET=//p' | head -1)
label=$(printf '%s\n' "$out" | sed -n 's/^LABEL=//p' | head -1)
infile=$(printf '%s\n' "$out" | sed -n 's/^INFILE=//p' | head -1)

printf '  rc=%d ticket=%s\n' "$rc" "${ticket:-<unset>}"
if [ "$rc" != 0 ]; then echo "  FAIL the hold did not run (rc=$rc)"; exit 1; fi
if [ -z "$ticket" ]; then
    echo "  FAIL GPU_HOLD_TICKET is unset: a child cannot tell holding from being a bystander"
    fails=$((fails + 1))
elif [ "$ticket" != "$infile" ]; then
    echo "  FAIL the exported ticket does not match the token file"
    echo "    exported: $ticket"
    echo "    in file:  $infile"
    fails=$((fails + 1))
else
    echo "  ok   the child's GPU_HOLD_TICKET is exactly the line in the token"
fi
if [ "$label" != "child-identity" ]; then
    echo "  FAIL GPU_HOLD_LABEL is '$label', want 'child-identity'"
    fails=$((fails + 1))
else
    echo "  ok   GPU_HOLD_LABEL carries the sanitised label"
fi
# And the bystander direction: outside any hold the variables must be absent, which is what makes the check meaningful.
env -u GPU_HOLD_TICKET -u GPU_HOLD_LABEL bash -c '[ -z "${GPU_HOLD_TICKET:-}" ] && [ -z "${GPU_HOLD_LABEL:-}" ]' \
    && echo "  ok   outside a hold neither variable is set (so a bare presence check cannot pass)" \
    || { echo "  FAIL leaked into this shell"; fails=$((fails + 1)); }

echo "gpu_hold_child_identity: $fails failure(s)"
exit $((fails > 0))
