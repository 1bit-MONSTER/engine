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
# install-coder-llm-ticketed.sh [--diff | --status | --install]
#
# Switches the coder-llm user unit (the PR-Agent model, :8090) to scripts/coder-llm-ticketed: the server command is
# kept exactly as the live drop-in has it, and only gets the wrapper in front of it (docs/ci-runners.md, "The PR-Agent
# model runs in gpu-hold windows"; engine#2494).
#
#   --diff     (default) print the drop-in change, apply nothing
#   --status   is the wrapper installed, and does the RUNNING unit execute it? exit 0 when both, 1 otherwise
#   --install  back up the drop-in as <name>.bak-<YYYYMMDD>, write the new one, install the wrapper into ~/.local/bin,
#              daemon-reload, restart the unit, print --status. Already ticketed: nothing to do, exit 0.
#
# Environment (tests): CODER_LLM_UNIT_DIR (default ~/.config/systemd/user/coder-llm.service.d), CODER_LLM_DROPIN
# (default zz-c22-selfcontained.conf), CODER_LLM_BIN_DIR (default ~/.local/bin), CODER_LLM_SYSTEMCTL (default
# "systemctl --user").
set -u
here=$(cd "$(dirname "$0")" && pwd)
UNIT_DIR=${CODER_LLM_UNIT_DIR:-$HOME/.config/systemd/user/coder-llm.service.d}
DROPIN=$UNIT_DIR/${CODER_LLM_DROPIN:-zz-c22-selfcontained.conf}
BIN_DIR=${CODER_LLM_BIN_DIR:-$HOME/.local/bin}
read -r -a SYSTEMCTL <<< "${CODER_LLM_SYSTEMCTL:-systemctl --user}"
WRAPPER=$BIN_DIR/coder-llm-ticketed
mode=${1:---diff}

# the new drop-in, from the live one: env lines before the final "ExecStart=" reset, wrapper in front of the command
render() {
    awk -v w="$WRAPPER" -v b="$BIN_DIR" '
        /^ExecStart=$/ && !done {
            print "# 2026-11-15 CODER-LLM-TICKETED (engine#2494): the model runs ONLY inside gpu-hold tickets, in on-demand windows"
            print "# (label coder-llm, max hold 25 min, re-queued FIFO by Restart=always); the server command below is unchanged."
            print "# `systemctl --user start coder-llm` now means QUEUED, not \":8090 answers\" (docs/ci-runners.md)."
            print "Environment=CODER_LLM_GH_POLL=1"
            print "Environment=CODER_LLM_GPU_HOLD=" b "/gpu-hold"
            print "Environment=PATH=" b ":/usr/local/bin:/usr/bin:/bin"
            print "Environment=CODER_LLM_MAX_HOLD_MIN=25"
            print "Environment=CODER_LLM_IDLE_MIN=5"
            print "Environment=GPU_HOLD_LOG=" ENVIRON["HOME"] "/GPU-HOLD.log"
            print "Restart=always"
            print "RestartSec=15"
            print "TimeoutStopSec=90"
            print; reset = 1; done = 1; next
        }
        reset && /^ExecStart=./ { sub(/^ExecStart=/, "ExecStart=" w " -- "); print; reset = 0; next }
        { print }' "$DROPIN"
}

is_ticketed_file() { grep -q '^ExecStart=.*coder-llm-ticketed' "$DROPIN" 2> /dev/null; }

status() {
    local ok=0 exec_line
    if [ -x "$WRAPPER" ] && cmp -s "$WRAPPER" "$here/coder-llm-ticketed"; then
        echo "wrapper: $WRAPPER installed, same as the tree"
    elif [ -x "$WRAPPER" ]; then
        echo "wrapper: $WRAPPER installed, DIFFERS from $here/coder-llm-ticketed"; ok=1
    else
        echo "wrapper: NOT installed at $WRAPPER"; ok=1
    fi
    if is_ticketed_file; then echo "drop-in: $DROPIN runs the wrapper"; else echo "drop-in: $DROPIN does NOT run the wrapper"; ok=1; fi
    exec_line=$("${SYSTEMCTL[@]}" show coder-llm -p ExecStart --value 2> /dev/null | head -1)
    case "$exec_line" in
        *coder-llm-ticketed*) echo "running unit: executes coder-llm-ticketed" ;;
        *) echo "running unit: does NOT execute coder-llm-ticketed (${exec_line:0:120})"; ok=1 ;;
    esac
    echo "token: $(head -c 80 "${GPU_TOKEN_FILE:-$HOME/GPU-TOKEN}" 2> /dev/null | head -1 || true)"
    [ "$ok" = 0 ] && echo "STATUS: ticketed" || echo "STATUS: NOT ticketed"
    return "$ok"
}

[ -f "$DROPIN" ] || { echo "no drop-in at $DROPIN" >&2; exit 2; }
case "$mode" in
    --diff)
        if is_ticketed_file; then echo "already ticketed: $DROPIN"; exit 0; fi
        diff -u "$DROPIN" <(render) | sed "2s|/dev/fd/[0-9]*|$DROPIN (new)|"
        exit 0 ;;
    --status)
        status; exit $? ;;
    --install)
        if is_ticketed_file; then echo "already ticketed: $DROPIN"; status; exit 0; fi
        grep -q '^ExecStart=$' "$DROPIN" || { echo "no 'ExecStart=' reset line in $DROPIN: refusing to guess" >&2; exit 2; }
        bak=$DROPIN.bak-$(date +%Y%m%d)
        [ -e "$bak" ] && bak=$bak-$(date +%H%M%S)
        cp -p "$DROPIN" "$bak" || exit 1
        render > "$DROPIN.new" && mv "$DROPIN.new" "$DROPIN" || exit 1
        mkdir -p "$BIN_DIR" && install -m 0755 "$here/coder-llm-ticketed" "$WRAPPER" || exit 1
        echo "backed up: $bak"
        "${SYSTEMCTL[@]}" daemon-reload && "${SYSTEMCTL[@]}" restart coder-llm || { echo "systemctl failed; roll back with: cp -p $bak $DROPIN" >&2; exit 1; }
        status; exit $? ;;
    *)
        echo "usage: $0 [--diff | --status | --install]" >&2; exit 2 ;;
esac
