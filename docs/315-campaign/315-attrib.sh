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
# engine#315 diagnostic: attribute the fault rate the instrumented arm hunt showed. NOT an acceptance
# run - it changes no registered criterion, produces no eligible arms, and must never be cited as one.
#
# Why it exists: the instrumented hunt (evidence/315-instrumented-arm-hunt.md) faulted 16 of 17 fix-cell
# arms and its one control arm, against a recorded 2/37. Before any stage name from that run is treated
# as a mechanism, the rate has to be attributed. This runs the *same binary pair* with CONTROL_EVERY=1
# so a stock-build arm and an instrumented arm alternate inside one composition and one box state:
#   - stock cell faults rarely, fix cell often  -> the environment (SENTINEL / TENSOR_FP) induces it,
#     and the stage names are where those reads land, not where the production fault is born
#   - both cells fault often                    -> the build or the box state is the cause, not the env
#   - neither faults                            -> the 16/17 is not reproducible and must be re-derived
#
# Usage: 315-attrib.sh <pairs> <tag> <control_every> <extra_env ...>
# Holds the box lock for the whole session and passes NOLOCK=1 to the driver, which then inherits this
# session's fd 9 instead of racing for its own (the driver's fallback is to proceed UNLOCKED, which
# would put two arm runners on the box at once).
set -uo pipefail
PAIRS=${1:-6}; TAG=${2:-315att}; CE=${3:-1}; shift 3 || true
EV="$*"
CBIN=${CBIN:-$HOME/wt/315instr3-build/bin}   # stock: implements none of the instruments
FBIN=${FBIN:-$HOME/wt/315fp-build/bin}       # carries SENTINEL / COPY_PROBE / TENSOR_FP
D=/home/bcloud/wt

# NOLOCK=1 means a caller already holds the lock on an inherited fd 9 (see 315-chain.sh). Taking it again
# here would open a second file description in the same process and deadlock against our own lock.
if [ -z "${NOLOCK:-}" ]; then
  exec 9>"$HOME/.cache/lax-decode/box.lock"
  if flock -w "${LOCK_WAIT:-3600}" 9; then
    echo "attrib: box lock held $(date -Is)"
  else
    echo "attrib: box lock unobtainable after ${LOCK_WAIT:-3600}s - aborting rather than running concurrently"
    exit 3
  fi
else
  echo "attrib: NOLOCK=1 - inheriting the caller's fd 9 lock"
fi

echo "=== attrib session tag=$TAG pairs=$PAIRS control_every=$CE $(date -Is) ==="
echo "    extra_env='${EV}'"
echo "    control_bin=$CBIN fix_bin=$FBIN"

REGIME_LO=${REGIME_LO:-70} REGIME_HI=${REGIME_HI:-85} \
CONTROL_EVERY="$CE" NOSTOP=1 MINCLEAN=999 NOLOCK=1 \
EXTRA_ENV="$EV" ARM_TIMEOUT="${ARM_TIMEOUT:-1200}" \
  bash "$D/315-paired.sh" "$PAIRS" "${NB:-2}" "${NCTX:-2048}" "$CBIN" "$FBIN" "$TAG"

echo "=== attrib session done $(date -Is) ==="
echo "stock-cell (control) arms:"; grep -c 'arm c: OK' "$D/$TAG-console.log" 2>/dev/null || true
echo "stock-cell (control) faults:"; grep -c 'arm c: \(FAULT\|INVALID\)' "$D/$TAG-console.log" 2>/dev/null || true
echo "fix-cell arms OK:"; grep -c 'arm l: OK' "$D/$TAG-console.log" 2>/dev/null || true
echo "fix-cell faults:"; grep -c 'arm l: \(FAULT\|INVALID\)' "$D/$TAG-console.log" 2>/dev/null || true
