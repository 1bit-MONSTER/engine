#!/bin/sh
# 1bit-selftest: one command that proves the engine answers from THIS boot.
#
# 1bit OS is live: everything runs from RAM, no internal disk is mounted and nothing is installed,
# so whatever this proves is gone at the next reboot (only /data on the stick persists). Run it from
# the console, or over SSH from another machine.
set -u
PORT=$(sed -n 's/.*"port"[^0-9]*\([0-9]\+\).*/\1/p' /data/lemonade/config/config.json 2>/dev/null | head -1)
PORT=${PORT:-13305}
KEY=$(sed -n 's/^LEMONADE_API_KEY=//p' /data/lemonade/lemond.conf 2>/dev/null | head -1)
LOG=/data/selftest.log
say() { echo "$@"; [ -w /data ] && echo "$@" >> "$LOG" 2>/dev/null; }

say "1bit OS self-test  $(date -u +%FT%TZ)"
say "  mode      : LIVE - RAM only, nothing installed, internal disks never mounted"
say "  kernel    : $(uname -r)"
say "  GPU / NPU : $(ls /dev/dri/renderD* /dev/kfd 2>/dev/null | tr '\n' ' ')$(ls /dev/accel/accel* 2>/dev/null | tr '\n' ' ')"
say "  NPU fw    : $([ -e /lib/firmware/amdnpu/17f0_11 ] && echo present || echo MISSING)"
say "  USB4/TB   : $(find /lib/modules -name 'thunderbolt.ko*' | head -1 | sed 's|.*/||')$([ -n "$(find /lib/modules -name 'thunderbolt.ko*' | head -1)" ] && echo '' || echo ' MISSING')"
say "  IP        : $(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | tr '\n' ' ')"

MODELS=$(wget -q -O - "http://127.0.0.1:$PORT/api/v1/models" 2>/dev/null)
if [ -z "$MODELS" ]; then
    say "  Lemonade  : no answer on 127.0.0.1:$PORT (see /tmp/lemond.log)"
    say "  hint      : run 'lemonade list' once lemond is up; models go in /data/models"
    exit 1
fi
pick() { printf '%s' "$MODELS" | tr ',' '\n' | sed -n "s/.*\"$1\":\"\($2\)\".*/\1/p" | head -1; }
NAME=$(pick id '[^"]*-1bit')
[ -n "$NAME" ] || NAME=$(pick model_name '[^"]*-1bit')
[ -n "$NAME" ] || NAME=$(pick id '[^"]*')
[ -n "$NAME" ] || NAME=$(pick model_name '[^"]*')
[ -n "$NAME" ] || { say "  models    : Lemonade answers but lists none - put a GGUF or an NPU model dir in /data/models"; say "RESULT: FAIL - no model to test"; exit 1; }
say "  Lemonade  : answers; testing model '$NAME'"

BODY="{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: NPU OK\"}],\"max_tokens\":48,\"temperature\":0}"
AUTH=""; [ -n "$KEY" ] && AUTH="Authorization: Bearer $KEY"
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
        say "RESULT: FAIL - Lemonade is up but the chat did not answer"
        exit 1 ;;
esac
say "  evidence  : appended to $LOG"
