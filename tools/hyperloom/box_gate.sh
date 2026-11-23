#!/bin/bash
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
# box_gate.sh -- the ONLY thing the Hyperloom agent account's ssh key may run on the box. Installed on the box and named
# by the key's authorized_keys entry:
#
#   command="/data/dev/cl-hyperloom/gate/box_gate.sh",no-pty,no-port-forwarding,no-agent-forwarding,no-X11-forwarding ssh-ed25519 ...
#
# sshd then runs this for every connection with the client's command in SSH_ORIGINAL_COMMAND. The client is
# box_ticket.sh, which sends exactly four command shapes; each is matched here with a strict regular expression and
# re-run as an argv (never through a shell), anything else is refused with exit 126:
#
#   ship    accepted only as a no-op for a box-side directory that is already installed AND allowlisted (the restricted
#           key never writes scripts onto the box: a script it could write, it could run as root through `sudo -n`)
#   ticket  `box-hold gpu [sudo -n] timeout -k 30 N env K=V... bash <allowlisted rev>/box_{inkling,bench}.sh <out>` with
#           N <= 2700 s, every K one of ALLOWED_KEYS, every V in a safe character set, and the rev directory's content
#           re-hashed against its name before it runs
#   fetch   `tar -C <out> -cf - [--exclude=card] .`
#   kill    the kill-by-ticket of box_ticket.sh, performed here
#
# The allowlist ($GATE_DIR/allowed-revs, one 12-hex rev per line) is written by the operator, not through this gate.
# Log: $GATE_DIR/gate.log (one line per request, accepted or refused, the command's first 300 characters).
set -u
ROOT=${HL_GATE_ROOT:-/data/dev/cl-hyperloom}
GATE_DIR=${HL_GATE_DIR:-$ROOT/gate}
CMD=${SSH_ORIGINAL_COMMAND-}
[ -n "$CMD" ] || CMD="$*"
log() { printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$1" "${CMD:0:300}" >> "$GATE_DIR/gate.log" 2>/dev/null; }
refuse() { log "REFUSED($1)"; echo "box_gate: refused: $1" >&2; exit 126; }

R=${ROOT//./\\.}
TK='hl-[0-9]{8}T[0-9]{6}Z-[0-9]{1,10}'
OUTRE="$R/out/($TK)"
REVRE="$R/hl/([0-9a-f]{12})"
# The names box_ticket.sh may pass (the union of both workloads' LEVERS maps, minus anything that names a binary).
ALLOWED_KEYS=" HL_DRAFT_DEPTH HL_ROWS HL_CONTEXT HL_PREFILL_CAP HL_MAX_TOKENS HL_SPIN_NS HL_WAIT_MODE HL_POOL_BACKING
 HL_WORKERS HL_RUNS HL_GAUNTLET_ROWS HL_RAM_NEED_GB HL_B HL_UB HL_FA HL_CTK HL_CTV HL_MTP_MAX HL_MTP_HEAD HL_DFLASH
 HL_DFLASH_GGUF HL_PARALLEL HL_CTX HL_LONGCTX HL_LONGTOK HL_REPS HL_TOKENS GGML_HRX_Q8_PREFILL_RELAX
 GGML_HRX_GRAPH_PROGRAM_CACHE ONEBIT_HRX_DECODE_SPLIT MODEL HL_WIKI HL_STATE HL_GTT_NEED_GIB "
ALLOWED_KEYS=${ALLOWED_KEYS//$'\n'/ }

rev_ok() {  # rev_ok REV SCRIPT: allowlisted, and the directory still hashes to its name
    local rev=$1 script=$2 files
    grep -qxF "$rev" "$GATE_DIR/allowed-revs" 2>/dev/null || return 1
    case $script in box_inkling.sh) files="box_inkling.sh gauntlet.py" ;; box_bench.sh) files="box_bench.sh benchmark.py" ;; *) return 1 ;; esac
    [ "$(cd "$ROOT/hl/$rev" 2>/dev/null && cat $files 2>/dev/null | sha256sum | cut -c1-12)" = "$rev" ]
}

if [[ $CMD =~ ^mkdir\ -p\ $REVRE\ $R/out\ \&\&\ tar\ -C\ $REVRE\ -xf\ -$ ]]; then
    rev=${BASH_REMATCH[1]}
    [ "$rev" = "${BASH_REMATCH[2]}" ] || refuse "ship: two different revs"
    grep -qxF "$rev" "$GATE_DIR/allowed-revs" 2>/dev/null && [ -d "$ROOT/hl/$rev" ] || refuse "ship: rev $rev is not installed+allowlisted (the operator installs box sides)"
    cat > /dev/null
    log "ship(no-op)"
    exit 0
fi

if [[ $CMD =~ ^mkdir\ -p\ $OUTRE\ \&\&\ BOX_HOLD_WHO=\'cl-hyperloom\ ($TK)\'\ BOX_HOLD_WAIT=([0-9]{1,5})\ box-hold\ gpu\ (sudo\ -n\ )?timeout\ -k\ 30\ ([0-9]{1,4})\ env\ (.*)\ ?bash\ $REVRE/(box_inkling\.sh|box_bench\.sh)\ $OUTRE$ ]]; then
    t1=${BASH_REMATCH[1]} t2=${BASH_REMATCH[2]} wait=${BASH_REMATCH[3]} sudo=${BASH_REMATCH[4]} secs=${BASH_REMATCH[5]}
    kvs=${BASH_REMATCH[6]} rev=${BASH_REMATCH[7]} script=${BASH_REMATCH[8]} t3=${BASH_REMATCH[9]}
    [ "$t1" = "$t2" ] && [ "$t1" = "$t3" ] || refuse "ticket: ticket ids differ"
    [ "$secs" -le 2700 ] || refuse "ticket: $secs s is above the 45-minute ticket cap"
    rev_ok "$rev" "$script" || refuse "ticket: $rev/$script is not an allowlisted, unmodified box side"
    envs=()
    for kv in $kvs; do
        [[ $kv =~ ^([A-Z][A-Z0-9_]*)=([A-Za-z0-9._/+,:-]+)$ ]] || refuse "ticket: malformed $kv"
        [[ $ALLOWED_KEYS == *" ${BASH_REMATCH[1]} "* ]] || refuse "ticket: ${BASH_REMATCH[1]} is not an allowed name"
        [[ ${BASH_REMATCH[2]} == *..* ]] && refuse "ticket: '..' in ${BASH_REMATCH[1]}"
        envs+=("$kv")
    done
    out=$ROOT/out/$t1
    mkdir -p "$out" || refuse "ticket: cannot create $out"
    log "ticket"
    pre=(); [ -n "$sudo" ] && pre=(sudo -n)
    BOX_HOLD_WHO="cl-hyperloom $t1" BOX_HOLD_WAIT=$wait exec box-hold gpu "${pre[@]}" timeout -k 30 "$secs" \
        env "${envs[@]}" bash "$ROOT/hl/$rev/$script" "$out"
fi

if [[ $CMD =~ ^tar\ -C\ $OUTRE\ -cf\ -\ (--exclude=card\ )?\.\ 2\>/dev/null$ ]]; then
    log "fetch"
    if [ -n "${BASH_REMATCH[2]}" ]; then exec tar -C "$ROOT/out/${BASH_REMATCH[1]}" -cf - --exclude=card . 2>/dev/null; fi
    exec tar -C "$ROOT/out/${BASH_REMATCH[1]}" -cf - . 2>/dev/null
fi

if [[ $CMD =~ ^sudo\ -n\ pkill\ -TERM\ -f\ --\ \'\[o\]ut/($TK)\'\;\ sleep\ 5\;\ p=\$\(cat\ $OUTRE/serve\.pid\ 2\>/dev/null\)\ \&\&\ sudo\ -n\ kill\ -KILL\ -\$p\ 2\>/dev/null\;\ true$ ]]; then
    t=${BASH_REMATCH[1]}
    [ "$t" = "${BASH_REMATCH[2]}" ] || refuse "kill: ticket ids differ"
    log "kill"
    sudo -n pkill -TERM -f -- "[o]ut/$t"
    sleep 5
    p=$(cat "$ROOT/out/$t/serve.pid" 2>/dev/null)
    [[ $p =~ ^[0-9]+$ ]] && sudo -n kill -KILL -- "-$p" 2>/dev/null
    exit 0
fi

refuse "not one of ship/ticket/fetch/kill"
