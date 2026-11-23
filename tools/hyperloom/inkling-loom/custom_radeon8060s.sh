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
# Hyperloom 1.2.0 custom-workload entrypoint: Inkling-2-Mini on the Loom route, measured on the Strix Halo box.
#
# Hyperloom (on the orchestration host, ryzen) resolves this file by board name (`--gpu-type radeon8060s`) and runs it
# once per round with RESULT_DIR / RESULT_FILENAME and the round's lever environment. ONE round = ONE ticket on the
# box's device lock:
#
#   1. the lever environment is validated here against a fixed list (names AND values); nothing else crosses the ssh;
#   2. box_inkling.sh + gauntlet.py are copied to $HL_BOX:$HL_BOX_DIR/hl/<content hash>/;
#   3. `ssh $HL_BOX box-hold gpu sudo -n timeout <ticket> env <levers> bash box_inkling.sh <out>` runs the measurement
#      (box-hold serialises lanes on the GPU; the timeout caps the held part at HL_TICKET_MIN, default 40 < 45 minutes);
#   4. the ticket's output is fetched into $RESULT_DIR/box and hl_score.py writes $RESULT_DIR/$RESULT_FILENAME.json.
# A signal (Hyperloom stopping the round) kills the remote ticket by its id, so no measurement outlives its round.
# The ssh/ticket mechanics are shared with ../custom/ in ../box_ticket.sh.
#
# Exit 0 whenever a report was written -- a rejected candidate is quality_gate.passed=false; non-zero only when no
# report could be written at all.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_TAG=hl-inkling
# Levers: name -> allowed value pattern. Anything else in the round's environment stays on this side.
declare -A LEVERS=(
    [HL_DRAFT_DEPTH]='^[0-8]$'
    [HL_ROWS]='^([1-9]|1[0-6])$'
    [HL_CONTEXT]='^[0-9]{4,8}$'
    [HL_PREFILL_CAP]='^[0-9]{3,6}$'
    [HL_MAX_TOKENS]='^[0-9]{3,6}$'
    [HL_SPIN_NS]='^([0-9]{1,10}|default)$'
    [HL_WAIT_MODE]='^(active|blocking)$'
    [HL_POOL_BACKING]='^(fixed|elastic)$'
    [HL_WORKERS]='^([1-9]|[12][0-9]|3[0-2])$'
    [HL_RUNS]='^[1-5]$'
    [HL_GAUNTLET_ROWS]='^[a-z0-9,]+$'
    [HL_RAM_NEED_GB]='^[0-9]{1,3}$'
)
. "$HERE/../box_ticket.sh"
HL_STATE=${HL_STATE:-$HOME/hyperloom-ws/state/inkling-loom}
mkdir -p "$HL_STATE" || exit 1

hl_levers
hl_ship box_inkling.sh gauntlet.py
hl_ticket --sudo bash "$RDIR/box_inkling.sh" "$ROUT"   # sudo: loom-serve's fixed pools need ulimit -l unlimited
hl_fetch card
hl_exit_if_unreachable
python3 "$HERE/hl_score.py" --box "$RESULT_DIR/box" --state "$HL_STATE" --levers "$LEVERS_JSON" \
    --out "$RESULT_DIR/$RESULT_FILENAME.json" || exit 1
exit 0
