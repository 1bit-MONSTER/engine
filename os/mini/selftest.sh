#!/bin/sh
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
# 1bit-selftest: one command that proves the engine answers from THIS boot.
#
# 1bit OS is live: everything runs from RAM, no internal disk is mounted and nothing is installed,
# so whatever this proves is gone at the next reboot (only /data on the stick persists). Run it from
# the console, or over SSH from another machine.
set -u
PORT=$(sed -n 's/.*"port"[^0-9]*\([0-9]\+\).*/\1/p' /data/lemonade/config/config.json 2>/dev/null | head -1)
PORT=${PORT:-13305}
KEY=$(sed -n 's/^LEMONADE_API_KEY=//p' /data/lemonade/lemond.conf 2>/dev/null | head -1)
# Lemonade listens on the LAN (config.json "host"), so it wants the key - and it must be set HERE,
# before the first request: the models call below used to go out unauthenticated and get 401.
AUTH=""; [ -n "$KEY" ] && AUTH="Authorization: Bearer $KEY"
LOG=/data/selftest.log
say() { echo "$@"; [ -w /data ] && echo "$@" >> "$LOG" 2>/dev/null; }

say "1bit OS self-test  $(date -u +%FT%TZ)"
say "  mode      : LIVE - RAM only, nothing installed, internal disks never mounted"
say "  kernel    : $(uname -r)"
say "  GPU / NPU : $(ls /dev/dri/renderD* /dev/kfd 2>/dev/null | tr '\n' ' ')$(ls /dev/accel/accel* 2>/dev/null | tr '\n' ' ')"
say "  NPU fw    : $([ -e /lib/firmware/amdnpu/17f0_11 ] && echo present || echo MISSING)"
say "  USB4/TB   : $(find /lib/modules -name 'thunderbolt.ko*' | head -1 | sed 's|.*/||')$([ -n "$(find /lib/modules -name 'thunderbolt.ko*' | head -1)" ] && echo '' || echo ' MISSING')"
say "  IP        : $(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | tr '\n' ' ')"

MODELS=$(wget -q -O - ${AUTH:+--header="$AUTH"} "http://127.0.0.1:$PORT/api/v1/models" 2>/dev/null)
if [ -z "$MODELS" ]; then
    say "  Lemonade  : no answer on 127.0.0.1:$PORT (see /tmp/lemond.log)"
    say "  hint      : if lemond is up this is usually the API key (401): grep LEMONADE_API_KEY /data/lemonade/lemond.conf"
    say "              otherwise run 'lemonade list'; models go in /data/models"
    exit 1
fi
pick() { printf '%s' "$MODELS" | tr ',' '\n' | sed -n "s/.*\"$1\":\"\($2\)\".*/\1/p" | head -1; }
# An NPU model directory first (that is what this image ships), then any plain GGUF.
NAME=$(pick id '[^"]*-1bit')
[ -n "$NAME" ] || NAME=$(pick model_name '[^"]*-1bit')
ANY=$(pick id '[^"]*')
[ -n "$NAME" ] || NAME=$ANY
[ -n "$NAME" ] || NAME=$(pick id '[^"]*')
[ -n "$NAME" ] || NAME=$(pick model_name '[^"]*')
[ -n "$NAME" ] || { say "  models    : Lemonade answers but lists none - put a GGUF or an NPU model dir in /data/models"; say "RESULT: FAIL - no model to test"; exit 1; }
say "  Lemonade  : answers; testing model '$NAME'"

BODY="{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: NPU OK\"}],\"max_tokens\":48,\"temperature\":0}"
RESP=$(wget -q -O - \
        --header='Content-Type: application/json' ${AUTH:+--header="$AUTH"} \
        --post-data="$BODY" "http://127.0.0.1:$PORT/api/v1/chat/completions" 2>/dev/null)
case "$RESP" in
    *'"content"'*)
        say "  answer    : $(printf '%s' "$RESP" | sed -n 's/.*"content":"\([^"]*\)".*/\1/p' | head -1)"
        say "  timings   : $(printf '%s' "$RESP" | sed -n 's/.*"predicted_per_second":\([0-9.]*\).*/\1/p' | head -1 | awk '{printf "decode %.1f tok/s", $1}')"
        say "RESULT: PASS - the engine answered from this live boot"
        ;;
    *)  say "  answer    : none (${RESP:-empty response})"
        say "RESULT-LM  : FAIL - Lemonade is up but the chat did not answer (the NPU route is tested below anyway)" ;;
esac
# THE NPU PROOF, THROUGH THE PRODUCT PATH. Lemonade runs the 1bit engine as its onebit backend, and the
# engine says where it put the model: "1bit unified: loading <model> on the NPU" then "on the private route
# dx: ...". The chat above already triggered that load, so read the engine's own words rather than starting
# a SECOND engine: the NPU takes one engine context at a time, and a second one poisons it - "npu runtime:
# invalid call (code 614): state 8" - which left Lemonade unable to load a model until a reboot. Proving it
# through Lemonade is also the thing the owner actually uses.
ONNPU=$(grep -a "1bit unified: loading .* on the NPU" /tmp/lemond.log 2>/dev/null | tail -1)
PLACE=$(grep -a "1bit unified: on the private route" /tmp/lemond.log 2>/dev/null | tail -1)
if [ -n "$ONNPU" ] && [ -n "$PLACE" ]; then
    say "  NPU model  : $(printf '%s' "$ONNPU" | sed 's/.*1bit unified: loading //')"
    say "  placement  : $(printf '%s' "$PLACE" | sed 's/.*1bit unified: //' | cut -c1-100)"
    say "  RESULT-NPU : PASS - Lemonade ran the engine on the NPU (the engine's own log lines)"
else
    say "  RESULT-NPU : FAIL - the engine never reported loading on the NPU (see /tmp/lemond.log)"
fi
say "  evidence  : appended to $LOG"
