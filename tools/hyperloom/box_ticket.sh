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
# box_ticket.sh -- sourced by the Hyperloom 1.2.0 custom-workload entrypoints (custom/, inkling-loom/). Hyperloom runs on
# the orchestration host; the device is the Strix Halo box, shared by several lanes. One Hyperloom round = one ticket:
#
#   hl_levers   validate the round's environment against the caller's LEVERS map (name -> value regex); only those
#               names cross the ssh, and a bad value is a failing report (exit 0), never a ticket
#   hl_ship     copy the box-side files to $HL_BOX:$HL_BOX_DIR/hl/<content hash>/ (a running ticket never sees a change)
#   hl_ticket   `box-hold gpu [sudo -n] timeout -k 30 <HL_TICKET_MIN*60> env <levers> <command>` over ssh; a signal to
#               the entrypoint kills the remote ticket by its id, so no measurement outlives its round
#   hl_fetch    bring the ticket's output directory back into $RESULT_DIR/box
#
# Caller sets HERE, LOG_TAG and LEVERS (declare -A) before sourcing; exports TICKET, ROUT, RDIR, PASS, LEVERS_JSON.
: "${RESULT_DIR:?Hyperloom sets RESULT_DIR}"
RESULT_FILENAME=${RESULT_FILENAME:-inferencex_result}
HL_BOX=${HL_BOX:-strixhalo}
HL_BOX_DIR=${HL_BOX_DIR:-/data/dev/cl-hyperloom}
HL_SSH=${HL_SSH:-ssh}
HL_TICKET_MIN=${HL_TICKET_MIN:-40}      # the fleet's ticket cap is 45 minutes
HL_HOLD_WAIT=${HL_HOLD_WAIT:-3600}
case "$HL_TICKET_MIN" in ''|*[!0-9]*) echo "[$LOG_TAG] HL_TICKET_MIN must be minutes" >&2; exit 1 ;; esac
[ "$HL_TICKET_MIN" -le 45 ] || { echo "[$LOG_TAG] HL_TICKET_MIN=$HL_TICKET_MIN is above the fleet's 45-minute ticket cap" >&2; exit 1; }
mkdir -p "$RESULT_DIR/box" || exit 1
TICKET=hl-$(date -u +%Y%m%dT%H%M%SZ)-$$
ROUT=$HL_BOX_DIR/out/$TICKET
PASS=()
LEVERS_JSON='{}'

hl_fail_report() {  # hl_fail_report REASON: a failing InferenceX report, for a round that must not run
    python3 - "$RESULT_DIR/$RESULT_FILENAME.json" "$1" <<'EOF'
import json, sys
json.dump({"framework": "custom", "workload_kind": "scriptable", "output_throughput": 0.0,
           "quality_gate": {"passed": False, "reason": sys.argv[2]}}, open(sys.argv[1], "w"), indent=1)
EOF
}

hl_levers() {
    local k v j='{'
    for k in $(printf '%s\n' "${!LEVERS[@]}" | sort); do
        v=${!k-}
        [ -n "$v" ] || continue
        if ! [[ $v =~ ${LEVERS[$k]} ]]; then
            hl_fail_report "lever $k=$v refused: not an allowed value"
            echo "[$LOG_TAG] refused lever $k=$v" >&2
            exit 0
        fi
        PASS+=("$k=$v")
        j+="\"$k\": \"$v\","
    done
    LEVERS_JSON="${j%,}}"
    [ "$LEVERS_JSON" = '}' ] && LEVERS_JSON='{}'
    echo "[$LOG_TAG] levers: ${PASS[*]:-(defaults)}"
}

hl_ship() {  # hl_ship FILE... (relative to $HERE)
    local rev
    rev=$(cd "$HERE" && cat "$@" | sha256sum | cut -c1-12)
    RDIR=$HL_BOX_DIR/hl/$rev
    tar -C "$HERE" -cf - "$@" | $HL_SSH "$HL_BOX" "mkdir -p $RDIR $HL_BOX_DIR/out && tar -C $RDIR -xf -" \
        || { echo "[$LOG_TAG] cannot copy the box side to $HL_BOX:$RDIR" >&2; exit 1; }
}

# The pattern is bracketed so it does not match the shell that runs this very pkill (its command line holds the pattern
# text, and "[o]ut/..." as text does not match "[o]ut/..." as a regex). serve.pid is the process group a box script
# started its server in, if it wrote one.
hl_kill_remote() {
    $HL_SSH "$HL_BOX" "sudo -n pkill -TERM -f -- '[o]ut/$TICKET'; sleep 5; p=\$(cat $ROUT/serve.pid 2>/dev/null) && sudo -n kill -KILL -\$p 2>/dev/null; true" \
        </dev/null >/dev/null 2>&1
}

hl_ticket() {  # hl_ticket [--sudo] COMMAND...  (COMMAND words are box-side; ROUT is the ticket's output directory)
    local sudo=
    [ "${1:-}" = --sudo ] && { sudo='sudo -n'; shift; }
    trap 'echo "[$LOG_TAG] stopped: killing ticket $TICKET" >&2; hl_kill_remote; exit 143' TERM INT
    echo "[$LOG_TAG] ticket $TICKET on $HL_BOX (box-hold gpu, held part <= ${HL_TICKET_MIN} min)"
    $HL_SSH -o ServerAliveInterval=30 "$HL_BOX" \
        "mkdir -p $ROUT && BOX_HOLD_WHO='cl-hyperloom $TICKET' BOX_HOLD_WAIT=$HL_HOLD_WAIT box-hold gpu $sudo timeout -k 30 $((HL_TICKET_MIN * 60)) env ${PASS[*]} $*" \
        </dev/null > "$RESULT_DIR/box/ticket.log" 2>&1 &
    local sp=$!
    wait $sp
    TICKET_RC=$?
    trap - TERM INT
    echo "[$LOG_TAG] ticket rc=$TICKET_RC"
}

hl_fetch() {  # hl_fetch [tar --exclude patterns...]
    local ex=() p
    for p in "$@"; do ex+=("--exclude=$p"); done
    $HL_SSH "$HL_BOX" "tar -C $ROUT -cf - ${ex[*]} . 2>/dev/null" </dev/null | tar -C "$RESULT_DIR/box" -xf - 2>/dev/null
    [ -f "$RESULT_DIR/box/status" ] || echo "ticket rc=$TICKET_RC, no status from the box (ticket.log)" > "$RESULT_DIR/box/status"
}
