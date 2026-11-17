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
# The tree that carries scripts/npu-kernel-bitx.sh — resolved from THIS TEST's location, not from the copy under test.
# Measured 2026-11-16: run against the installed copy (`~/.local/bin/gpu-hold`), `dirname(HOLD)/..` is `~/.local`, so
# every arm that needs the helper failed with `~/.local/scripts/npu-kernel-bitx.sh: No such file or directory` and the
# test reported four failures for a copy that is byte-identical to the repo's. Validating the DEPLOYED copy is the
# point of running it there, so the root has to travel with the test.
SRC=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$SRC/scripts/npu-kernel-bitx.sh" ] || SRC=$(cd "$(dirname "$HOLD")/.." && pwd)
tmp=$(mktemp -d "${TMPDIR:-/tmp}/ghid.XXXXXX") || exit 77
trap 'rm -rf "$tmp"' EXIT
# The take-time refusal reads the kernel log and refuses a ticket after a recent fault line (#2480/#2498). On a box
# whose NPU is faulting every few minutes (#2542) that refuses EVERY ticket, so this test would fail with rc 78 for a
# reason that is not the code -- measured 2026-11-16 16:4xZ. The fixture is the same override the fault-abort test
# uses: a log with no fault lines in it, so the refusal runs for real and finds nothing.
printf 'kernel: gpu-hold test fixture: no fault lines in this log\n' > $tmp/klog
export GPU_HOLD_KLOG_FILE=$tmp/klog
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
# THE ARM THAT MATTERS, and the first version of this file did not have it: a token present but NOT ours must be refused by a
# real consumer. The old arm here ran `env -u X bash -c '[ -z "$X" ]'` - it unset the variables and then asserted they were
# unset, a tautology that printed ok unconditionally and pinned nothing (engine#2447 review). The incident this file exists for
# is "another lane holds the token and the guard still passes", so that is the case to drive.
foreign="gpu-hold another-lane pid 999999 2026-11-14T00:00:00-04:00"
printf '%s\n' "$foreign" > "$tmp/NPU-TOKEN"
# The bystander is simulated by EMPTYING the ticket variables, and `NPU_HOLD_TICKET=` alone cannot do it: the consumer
# resolves `_want=${NPU_HOLD_TICKET:-${GPU_HOLD_TICKET:-}}` (an explicit NPU_HOLD_TICKET wins, GPU_HOLD_TICKET is the
# helper path's name), so emptying the first falls through to the second -- which npu-hold's chain exports, because
# npu-hold EXECS gpu-hold. Lanes run this suite INSIDE a gpu-hold ticket, so the inherited line reached the consumer,
# which answered "the device ticket is not ours" (rc 2, a different message than this arm asserts) and the arm failed
# for exactly the lanes that follow the one-GPU-job rule (measured 2026-11-16 19:2xZ). Clear every name the consumer
# consults, so the fixture simulates absence rather than renaming it.
out=$(env -u GPU_HOLD_TICKET -u GPU_HOLD_LABEL ONEBIT_BIN=/bin/true NPU_TOKEN_FILE="$tmp/NPU-TOKEN" NPU_HOLD_TICKET= \
      bash "$SRC/scripts/npu-kernel-bitx.sh" /x /y 2>&1)
rc=$?
if [ "$rc" = 2 ] && printf '%s' "$out" | grep -q 'cannot prove it owns the device'; then
    echo "  ok   a foreign token with no ticket line is REFUSED by the consumer (rc 2, also when the caller's own\
         gpu-hold ticket exported GPU_HOLD_TICKET; mutant = drop the `env -u` and this fails inside a ticket)"
else
    echo "  FAIL the consumer did not refuse a foreign token: rc=$rc"
    fails=$((fails + 1))
fi
out=$(ONEBIT_BIN=/bin/true NPU_TOKEN_FILE="$tmp/NPU-TOKEN" NPU_HOLD_TICKET="$foreign" \
      bash "$SRC/scripts/npu-kernel-bitx.sh" /x /y 2>&1)
# engine#2447 review: this arm used to assert only the ABSENCE of the refusal text, so a run that died for any
# other reason (bad usage, missing binary, a reworded message) passed it. Assert POSITIVE progress instead: the
# script prints the device line only after the guard, so a refused run cannot produce it. Mutant: give the
# consumer a ticket line that does NOT match the token and this arm must fail.
if printf '%s' "$out" | grep -q '^device' && ! printf '%s' "$out" | grep -q 'cannot prove it owns the device'; then
    echo "  ok   a matching ticket line gets past the consumer guard (reached the device line)"
else
    echo "  FAIL the consumer refused a ticket line that matches the token, or never got past the guard"
    printf '%s\n' "$out" | sed -n '1,3p' | sed 's/^/       /'
    fails=$((fails + 1))
fi
# engine#2447 review: the CONTENT comparison must be exercised by a NON-EMPTY line that differs. The arms above
# cover "no ticket line at all" (refused for absence) and "a line equal to the token" (accepted); neither reaches
# `[ "$(cat $_tok)" != "$_want" ]`, so dropping that comparison passed 0 failures. This is the arm for it: a
# well-formed ticket line for a DIFFERENT hold, with the token present.
other="gpu-hold some-other-lane pid 424242 2026-11-14T00:00:00-04:00"
out=$(ONEBIT_BIN=/bin/true NPU_TOKEN_FILE="$tmp/NPU-TOKEN" NPU_HOLD_TICKET="$other" \
      bash "$SRC/scripts/npu-kernel-bitx.sh" /x /y 2>&1)
if printf '%s' "$out" | grep -q 'the device ticket is not ours'; then
    echo "  ok   a NON-EMPTY ticket line that differs from the token is refused ('not ours')"
else
    echo "  FAIL a well-formed ticket line for a different hold was not refused as 'not ours'"
    printf '%s\n' "$out" | sed -n '1,3p' | sed 's/^/       /'
    fails=$((fails + 1))
fi
# and the variable the helper chain ACTUALLY exports must be the one accepted - the P0: the first version required
# NPU_HOLD_TICKET, which no helper sets, so every run was refused.
out=$(ONEBIT_BIN=/bin/true NPU_TOKEN_FILE="$tmp/NPU-TOKEN" GPU_HOLD_TICKET="$foreign" NPU_HOLD_TICKET= \
      bash "$SRC/scripts/npu-kernel-bitx.sh" /x /y 2>&1)
if printf '%s' "$out" | grep -q '^device' && ! printf '%s' "$out" | grep -q 'cannot prove it owns the device'; then
    echo "  ok   GPU_HOLD_TICKET (what the helper chain exports) is accepted (reached the device line)"
else
    echo "  FAIL GPU_HOLD_TICKET is not accepted although it is what npu-hold's chain exports"
    printf '%s\n' "$out" | sed -n '1,3p' | sed 's/^/       /'
    fails=$((fails + 1))
fi

echo "gpu_hold_child_identity: $fails failure(s)"
exit $((fails > 0))
