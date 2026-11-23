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
#
# Exit 0 whenever a report was written -- a rejected candidate is quality_gate.passed=false; non-zero only when no
# report could be written at all.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${RESULT_DIR:?Hyperloom sets RESULT_DIR}"
RESULT_FILENAME=${RESULT_FILENAME:-inferencex_result}
HL_BOX=${HL_BOX:-strixhalo}
HL_BOX_DIR=${HL_BOX_DIR:-/data/dev/cl-hyperloom}
HL_SSH=${HL_SSH:-ssh}
HL_STATE=${HL_STATE:-$HOME/hyperloom-ws/state/inkling-loom}
TICKET_MIN=${HL_TICKET_MIN:-40}
HOLD_WAIT=${HL_HOLD_WAIT:-3600}
mkdir -p "$RESULT_DIR/box" "$HL_STATE" || exit 1

# --- levers: name -> allowed value pattern. Anything else in the environment stays on this side. ---------------------
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
pass=()
levers_json='{'
for k in $(printf '%s\n' "${!LEVERS[@]}" | sort); do
    v=${!k-}
    [ -n "$v" ] || continue
    if ! [[ $v =~ ${LEVERS[$k]} ]]; then
        python3 - "$RESULT_DIR/$RESULT_FILENAME.json" "$k" "$v" <<'EOF'
import json, sys
json.dump({"framework": "custom", "workload_kind": "scriptable", "output_throughput": 0.0,
           "quality_gate": {"passed": False, "reason": "lever %s=%r refused: not an allowed value" % (sys.argv[2], sys.argv[3])}},
          open(sys.argv[1], "w"), indent=1)
EOF
        echo "[hl-inkling] refused lever $k=$v" >&2
        exit 0
    fi
    pass+=("$k=$v")
    levers_json+="\"$k\": \"$v\","
done
levers_json="${levers_json%,}}"
echo "[hl-inkling] levers: ${pass[*]:-(card defaults)}"

# --- ship the box side (content-addressed, so a running ticket never sees its script change) ------------------------
REV=$(cat "$HERE/box_inkling.sh" "$HERE/gauntlet.py" | sha256sum | cut -c1-12)
RDIR=$HL_BOX_DIR/hl/$REV
TICKET=hl-$(date -u +%Y%m%dT%H%M%SZ)-$$
ROUT=$HL_BOX_DIR/out/$TICKET
tar -C "$HERE" -cf - box_inkling.sh gauntlet.py | $HL_SSH "$HL_BOX" "mkdir -p $RDIR $HL_BOX_DIR/out && tar -C $RDIR -xf -" \
    || { echo "[hl-inkling] cannot copy the box side to $HL_BOX:$RDIR" >&2; exit 1; }

# The pattern is bracketed so it does not match the shell that runs this very pkill (its command line holds the
# pattern text, and "[o]ut/..." as text does not match "[o]ut/..." as a regex).
kill_remote() {
    $HL_SSH "$HL_BOX" "sudo -n pkill -TERM -f -- '[o]ut/$TICKET'; sleep 5; p=\$(cat $ROUT/serve.pid 2>/dev/null) && sudo -n kill -KILL -\$p 2>/dev/null; true" \
        </dev/null >/dev/null 2>&1
}
trap 'echo "[hl-inkling] stopped: killing ticket $TICKET" >&2; kill_remote; exit 143' TERM INT

# --- the ticket ---------------------------------------------------------------------------------------------------
echo "[hl-inkling] ticket $TICKET on $HL_BOX (box-hold gpu, held part <= ${TICKET_MIN} min)"
$HL_SSH -o ServerAliveInterval=30 "$HL_BOX" \
    "BOX_HOLD_WHO='cl-hyperloom $TICKET' BOX_HOLD_WAIT=$HOLD_WAIT box-hold gpu sudo -n timeout -k 30 $((TICKET_MIN * 60)) env ${pass[*]} bash $RDIR/box_inkling.sh $ROUT" \
    </dev/null > "$RESULT_DIR/box/ticket.log" 2>&1 &
sp=$!
wait $sp
rc=$?
echo "[hl-inkling] ticket rc=$rc"
$HL_SSH "$HL_BOX" "tar -C $ROUT -cf - --exclude=card . 2>/dev/null" </dev/null | tar -C "$RESULT_DIR/box" -xf - 2>/dev/null
[ -f "$RESULT_DIR/box/status" ] || echo "ticket rc=$rc, no status from the box (ticket.log)" > "$RESULT_DIR/box/status"
python3 "$HERE/hl_score.py" --box "$RESULT_DIR/box" --state "$HL_STATE" --levers "$levers_json" \
    --out "$RESULT_DIR/$RESULT_FILENAME.json" || exit 1
exit 0
