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
#
# 1bit-release-check: ONE command that says PASS or FAIL for a booted 1bit OS image.
#
# This is the release row as a command, not a judgement call. Run it on the live stick from the
# console, or over SSH (`ssh root@<box> 1bit-release-check`), and again after installing to disk.
# Every check prints PASS, FAIL or SKIP(reason) on its own line; any FAIL sets a non-zero exit and
# the transcript is appended to /data/release-check.log.
#
# Rows that cannot PASS without the hardware are marked SKIP in a VM (Lemonade hides its onebit
# entries on a machine with no AMD GPU or NPU, and there is no HRX or NPU device to run them on):
# gpu, npu, models, chat-hrx, chat-npu, chat-npu-followup, chat-npu-stream, chat-npu-ladder, chat-npu-queue, chat-npu-queue-plain. Everything else - mode, kernel,
# Secure Boot, Lemonade up with its key, the stick's GGUF answering on vulkan, `lemonade bench`, the sealed NPU
# sets and the signed manifests over them, the licences and the image identity - is checked in the VM as well.
set -u

LOG=/data/release-check.log
FAILS=0
ROWS=0
say() { printf '%s\n' "$*"; [ -w /data ] 2>/dev/null && printf '%s\n' "$*" >> "$LOG" 2>/dev/null; return 0; }
pass() { ROWS=$((ROWS + 1)); say "PASS  $1 : $2"; }
fail() { ROWS=$((ROWS + 1)); FAILS=$((FAILS + 1)); say "FAIL  $1 : $2"; }
HW_SKIPPED=   # the hardware-only rows that actually skipped, for the footer; empty means none did
skip() {
    ROWS=$((ROWS + 1))
    case "$1" in gpu|npu|npu-power|hrx-placement|models|chat-hrx|chat-npu|chat-npu-followup|chat-npu-stream|chat-npu-ladder|chat-npu-queue|chat-npu-queue-plain) HW_SKIPPED="${HW_SKIPPED:+$HW_SKIPPED, }$1" ;; esac
    say "SKIP  $1 : $2"
}
# A check that DIES must not take the evidence - or the NPU - with it. Measured on the c27 board (5 Nov 2026 21:52Z):
# bash killed this script at `PPID=$!` (a readonly builtin), so five rows of a later run were refused with
# `code 102: No such device` and the appliance's own u4-release-check row reded as "no RESULT line; rc=1" with nothing
# to act on. The turn that was mid-flight when the script died kept a hwctx in the driver; the next backend's
# `Init hwctx` was refused (ret -19, 21:55:37Z) and 20 s later the watchdog TDR'd and power-cycled the device
# (21:55:52Z). So: the footer announces itself (a death before it says so in the log, so the RESULT line is never
# silently missing), and a death ends the NPU turn (unload -> lemond stops the backend -> the context is released).
RC_FOOTER=
rc_on_exit() {
    [ -n "$RC_FOOTER" ] && return 0        # a normal run already printed its RESULT and owns the device state
    say "RESULT: FAIL (this check died before its footer: $ROWS row(s) ran, $FAILS failed - read the line above)"
    [ -n "${NPU:-}" ] && [ -n "${API:-}" ] || return 0
    got --header='Content-Type: application/json' --post-data='{}' "$API/api/v1/unload" >/dev/null 2>&1
    return 0
}
trap rc_on_exit EXIT
# A row that holds the NPU can hold it for a minute or two (a fresh backend per ladder rung, a 512-token turn),
# and on the owner's console that is indistinguishable from a hung check - which is what the board operator
# reported. So each NPU row announces itself before it starts. Not a row: the prefix is `info` so a harness that
# greps PASS/FAIL/SKIP is unaffected, and ROWS does not move.
# The GUI banner the brief also asked for ("if cheap") is not cheap (a status endpoint plus a poller), so the
# console line is what ships and the hand-off says the banner did not.
# Two guards are pi-release's, from their npu-console-notice row, and both are measured facts rather than caution:
# an X session owning the VT is graphics mode, where a text write corrupts what the person sees (the appliance can
# have a desktop session - the u2-desktop-session row exists for exactly that), and a headless boot may have no
# writable tty1 at all, so /dev/console is the fallback. Nothing here can fail a row.
npu_note() {  # <row>
    [ -e /tmp/.X11-unix/X0 ] && return 0
    for _d in /dev/tty1 /dev/console; do
        [ -c "$_d" ] && [ -w "$_d" ] || continue
        printf '\n1bit-release-check: %s is holding the NPU (this can take a minute)\n' "$1" > "$_d" 2>/dev/null && break
    done
    say "info  $1 : holding the NPU (this can take a minute)"
}
# Every fetch is capped. Without that, a Lemonade that accepts the connection and never answers hangs
# the whole check with no output - measured on the release candidate: 1bit-release-check sat in the
# lemonade row until an external 20-minute limit killed the run, which on the owner's console would
# look like a hung machine rather than a failed check.
got() { wget -q -T 15 -t 1 -O - "$@" 2>/dev/null ||
        curl -sf --connect-timeout 5 --max-time 20 "$@" 2>/dev/null; }

# The FIRST HRX request on a fresh stick is not a 20-second operation: the one-hop backend JIT-compiles
# its Loom kernels before it can answer, so chat-hrx reported "no answer" on candidate 10 while the exec
# chain was working perfectly (lemond: "one hop: /opt/1bit-llama/llama-server on port 8002").
# MEASURED on the board once the runtime was right: cold 2 s (66 cache files), warm 80 ms, then 120 tokens
# in 383 ms. So this is not a minutes-long operation and the cap need not be minutes either: 120 s is 60x
# the measured cold start - enough that a first-run JIT cannot fail the row, short enough that a genuinely
# hung backend still fails this row instead of stalling the whole check. Only the HRX chat row uses it.
got_slow() { wget -q -T 120 -t 1 -O - "$@" 2>/dev/null ||
             curl -sf --connect-timeout 5 --max-time 120 "$@" 2>/dev/null; }

# HTTP status of a request: 000 when nothing answered at all. A refusal has to be *proved* by its
# status, not inferred from an empty body: the unprepared stick returns an empty body to the keyless
# probe and the old row counted that as a refusal (it false-PASSED).
status() {
    c=
    if command -v curl >/dev/null 2>&1; then
        c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$@" 2>/dev/null)
    fi
    if [ -z "$c" ] || [ "$c" = 000 ]; then
        if command -v wget >/dev/null 2>&1; then
            c=$(wget -S -O /dev/null --timeout=10 "$@" 2>&1 | sed -n 's/.*HTTP\/1\.[01] \([0-9][0-9][0-9]\).*/\1/p' | head -1)
        fi
    fi
    printf '%s' "${c:-000}"
}

# The Authorization header, and ONLY when this machine has a key (cycle 21, item 9a's other half, from the c20
# board). Blank mode is the shipped state and its contract is stock Lemonade's: a request with NO header at all.
# Cycle 20 sent `Authorization: Bearer none` on every call to keep one code path; on the board a header-less curl
# answered 200 and the NPU answered long then short with 0 TDR, so the header-less shape is the one that worked.
# Defined next to the fetchers so the keyed and blank shapes cannot drift apart.
auth_get() {  # <url>
    if [ -n "$KEY" ]; then got --header="Authorization: Bearer $KEY" "$1"; else got "$1"; fi
}
auth_status() {  # <url> -> HTTP status
    if [ -n "$KEY" ]; then status --header="Authorization: Bearer $KEY" "$1"; else status "$1"; fi
}
# The fetcher for a POST. `got`/`got_slow` try wget first and pass their arguments through unchanged, but
# a POST body is spelled differently by the two tools: wget takes `--post-data`, curl takes `--data`.
# Measured on this box while exercising the laya row: with wget absent and curl present, every POST row
# read an empty body and reported "no answer" for a server that had answered - a false FAIL on exactly the
# installed-to-disk run the release check is meant to prove. So the POST path picks its own fetcher.
# RELEASE_CHECK_FETCHER=wget|curl forces one, which is how both branches get exercised.
post_fetch() {  # <json body> <url> [slow]
    _cap=20; [ "${3:-}" = slow ] && _cap=120
    _f=${RELEASE_CHECK_FETCHER:-}
    if [ -z "$_f" ]; then
        if command -v wget >/dev/null 2>&1; then _f=wget; else _f=curl; fi
    fi
    if [ -n "$KEY" ]; then _auth="Authorization: Bearer $KEY"; else _auth=; fi
    if [ "$_f" = curl ]; then
        if [ -n "$_auth" ]; then
            curl -sf --connect-timeout 5 --max-time "$_cap" --header='Content-Type: application/json' \
                 --header="$_auth" --data "$1" "$2" 2>/dev/null
        else
            curl -sf --connect-timeout 5 --max-time "$_cap" --header='Content-Type: application/json' \
                 --data "$1" "$2" 2>/dev/null
        fi
    else
        if [ -n "$_auth" ]; then
            wget -q -T "$_cap" -t 1 -O - --header='Content-Type: application/json' \
                 --header="$_auth" --post-data="$1" "$2" 2>/dev/null
        else
            wget -q -T "$_cap" -t 1 -O - --header='Content-Type: application/json' \
                 --post-data="$1" "$2" 2>/dev/null
        fi
    fi
}

# ── `code 102` is a WAIT, not a verdict (TOS3-C29 item 2; c28 board, 5 Nov 2026) ─────────────────
# The NPU driver gives the accelerator to ONE client. When that client exits, 0011v2 frees the partition about
# 2 s later, and for those seconds every new client is refused with
#   npu runtime: device error (code 102): No such device
# Measured on the c28 board: this check ran straight after its own NPU rows released the device and 8 of 25 rows
# reded on that window, so a healthy appliance read as broken. (Worse, the failed run left a turn in flight and
# the driver TDR'd 15 s later - see PR #1177.) So a reply carrying code 102 is retried with a 1 s back-off, up to
# NPU_WAIT seconds, and the footer says how long was spent waiting. A 102 that outlives the window is reported
# unchanged: the wait shortens a transient, it does not excuse a device that stays gone.
NPU_WAIT=${NPU_WAIT:-10}
# The counter has to be a FILE: every reply is read through $( ), so the retry runs in a subshell and a plain variable
# set in there never reaches the footer (measured while proving this block - the wait worked and the footer said 0 s).
NPU_WAITFILE=${NPU_WAITFILE:-/tmp/.1bit-npu-waited.$$}
_npu_102() {  # <reply> -> 0 when the backend named the refusal, with its own words
    case "$1" in *'code 102'*) return 0 ;; esac
    return 1
}
_npu_waited() {  # <seconds spent backing off>
    [ "${1:-0}" -gt 0 ] 2>/dev/null && printf '%s\n' "$1" >> "$NPU_WAITFILE" 2>/dev/null
    return 0
}

auth_post() {  # <json body> <url> [slow] -> the reply
    _r=$(post_fetch "$1" "$2" "${3:-}")
    _w=0
    while _npu_102 "$_r" && [ "$_w" -lt "$NPU_WAIT" ]; do
        sleep 1; _w=$((_w + 1))
        _r=$(post_fetch "$1" "$2" "${3:-}")
    done
    _npu_waited "$_w"
    printf '%s' "$_r"
}

# A POST whose *status and headers* are the evidence (engine#1024's non-streamed arm answers 503 + Retry-After, so the
# body alone says nothing and got/get_slow hide both). Writes the body to $3 and prints "<status> <retry-after>".
# The same back-off, for the arm whose evidence is the status and headers: a non-streamed client refused code 102
# must be asked again rather than reported as "no HTTP status at all", which is what the c28 board said about a
# request the partition had simply not freed for yet.
post_watch_once() {  # <json body> <url> <file for the body>
    : > "$3"
    if command -v curl >/dev/null 2>&1; then
        if [ -n "$KEY" ]; then
            curl -s -m 200 -D "$3.headers" -o "$3" -w '%{http_code}' \
                 --header='Content-Type: application/json' --header="Authorization: Bearer $KEY" \
                 --post-data="$1" "$2" 2>/dev/null
        else
            curl -s -m 200 -D "$3.headers" -o "$3" -w '%{http_code}' \
                 --header='Content-Type: application/json' --post-data="$1" "$2" 2>/dev/null
        fi
    else
        # No curl on this image: wget can still prove a refusal, but it discards the body, so say what is missing.
        wget -S -O "$3" -T 200 --header='Content-Type: application/json' --post-data="$1" "$2" >"$3.headers" 2>&1
        printf '%s' "$(sed -n 's/.*HTTP\/1\.[01] \([0-9][0-9][0-9]\).*/\1/p' "$3.headers" | head -1)"
    fi
    printf ' '
    sed -n 's/^[Rr]etry-[Aa]fter: *//p' "${3}.headers" 2>/dev/null | tr -d '\r' | head -1
}

auth_post_watch() {  # <json body> <url> <file for the body>
    _s=$(post_watch_once "$1" "$2" "$3")
    _w=0
    while _npu_102 "$(cat "$3" 2>/dev/null)" && [ "$_w" -lt "$NPU_WAIT" ]; do
        sleep 1; _w=$((_w + 1))
        _s=$(post_watch_once "$1" "$2" "$3")
    done
    _npu_waited "$_w"
    printf '%s' "$_s"
}

# Readiness: on a cold stick lemond takes tens of seconds to answer its first request (catalog, web
# app, engine spawn). The capped fetches above then see "status 000" and the row fails although the
# server is fine - candidate 4 (25 Oct) failed exactly so, where candidate 3's uncapped wget had simply
# waited. So the port is polled with 2 s probes until anything answers, up to READY_WAIT seconds; only
# then do the capped assertions run. A server that never answers gets its own row (lemonade_ready
# false), distinct from a wrong key. The caps on the assertions stay: they are what keep the
# 20-minute hang from coming back.
READY_WAIT=${READY_WAIT:-120}
lemonade_ready() {
    i=0
    while [ "$i" -lt "$READY_WAIT" ]; do
        if command -v curl >/dev/null 2>&1; then
            c=$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "$API/api/v1/models" 2>/dev/null)
        else
            c=$(wget -S -O /dev/null --timeout=2 -t 1 "$API/api/v1/models" 2>&1 | sed -n 's/.*HTTP\/1\.[01] \([0-9][0-9][0-9]\).*/\1/p' | head -1)
        fi
        case "${c:-000}" in 000) ;; *) say "      lemond answered $API after ${i}s (status $c)"; return 0 ;; esac
        sleep 2; i=$((i + 2))
    done
    return 1
}

# The lemonade row, decided from what the two probes returned. Pure, so tests/release_check_row.sh can
# drive it (os/mini/release-check.sh --self-test): <keyed status> <keyed body> <keyless status>.
lemonade_verdict() {
    ws=$1; body=$2; ns=$3
    case "$ws" in
        200) ;;
        000) printf 'FAIL no answer with the key from /data/lemonade/lemond.conf (status 000; see /tmp/lemond.log)'; return 0 ;;
        *) printf 'FAIL the server answered %s with the key (wanted 200)' "$ws"; return 0 ;;
    esac
    case "$body" in
        *'"data"'*|*'"models"'*) ;;
        *) printf 'FAIL the keyed request answered 200 but not a model list: %.60s' "$body"; return 0 ;;
    esac
    case "$ns" in
        401|403) printf 'PASS answers on %s with the API key, and refuses it without (status %s)' "$API" "$ns" ;;
        200) printf 'FAIL answers on %s, but also answers WITHOUT the key (status 200)' "$API" ;;
        *) printf 'FAIL the keyless probe answered %s, not a refusal: the key is not proved to be enforced' "$ns" ;;
    esac
}

# The same row for BLANK mode (cycle 20, the shipped default): no key on the machine, so what it asserts is the
# other contract - the server answers 200 and a model list WITHOUT any Authorization header. Pure as well:
# <keyless status> <keyless body>.
lemonade_verdict_blank() {
    ns=$1; body=$2
    case "$ns" in
        200) ;;
        401|403) printf 'FAIL no key is configured, but %s refuses a keyless request (status %s): the appliance is locked out of its own server' "$API" "$ns"; return 0 ;;
        000) printf 'FAIL no answer from %s without a key (status 000; see /tmp/lemond.log)' "$API"; return 0 ;;
        *) printf 'FAIL the keyless probe answered %s, not 200' "$ns"; return 0 ;;
    esac
    case "$body" in
        *'"data"'*|*'"models"'*) printf 'PASS answers on %s without a key (blank mode: anyone on this network can use it)' "$API" ;;
        *) printf 'FAIL answered 200 but not a model list: %.60s' "$body" ;;
    esac
}

# The model list the rows below read (engine #945). In blank mode the keyless body IS the list; in keyed mode
# the authenticated one is. Pure and chosen in ONE place, so a mode that sets $BODY and forgets $MODELS cannot
# report a healthy server as dead and skip every chat/bench row behind it. <keyed body> <keyless body>
models_list() {
    if [ -n "${1:-}" ]; then printf '%s' "$1"; else printf '%s' "${2:-}"; fi
}

# The sealed-set manifest rule (engine #574). Sealing is confidentiality: the kernel files are
# 1BPKCRYP ciphertext and the loader opens them with the kernel key. A set re-sealed by another key,
# or swapped for a different set that also decrypts, is indistinguishable from ours by that alone.
# So every set ships manifest.json - every file it carries, with the sha256 of its ciphertext - and
# manifest.sig, an Ed25519 signature over those bytes by the release signing key (tools/seal_manifest.py
# write). This function is the half that needs no key: the set's files and its manifest must be the
# same set of files. Pure, so --self-test drives it and the row below uses the same code.
seal_manifest_problems() {  # <set dir>: one line per problem, nothing and status 0 when sound
    _sd=$1 _bad=0 _rel= _f= _sz= _h= _m=0 _n=
    if [ ! -f "$_sd/manifest.json" ]; then
        printf 'no manifest.json: the set was never signed\n'
        return 1
    fi
    [ -f "$_sd/manifest.sig" ] || { printf 'no manifest.sig: the manifest is not signed\n'; _bad=1; }
    for _f in $(find "$_sd" -type f 2>/dev/null); do
        case "${_f##*/}" in manifest.json|manifest.sig) continue ;; esac
        _m=$((_m + 1)); _rel=${_f#"$_sd"/}
        _sz=$(wc -c <"$_f" | tr -d ' ')
        _h=$(sha256sum "$_f" 2>/dev/null | cut -c1-64)
        # the manifest is one file per line, keys in this order, so a plain grep can read it
        if ! grep -qF "{\"path\": \"$_rel\", \"size\": $_sz, \"sha256\": \"$_h\"}" "$_sd/manifest.json"; then
            printf '%s is not in the set manifest with the sha256 of its bytes\n' "$_rel"
            _bad=1
        fi
    done
    _n=$(grep -c '"sha256":' "$_sd/manifest.json" 2>/dev/null || true)
    [ "${_n:-0}" = "$_m" ] || {
        printf 'the manifest lists %s file(s), the set carries %s\n' "${_n:-0}" "$_m"
        _bad=1
    }
    return $_bad
}

# The NPU chat row's follow-up decision (engine #911). The board's cycle-17 demo lost the second
# turn after a long first one - 0 chunks, then one amdxdna TDR - while this check's single short
# prompt stayed green, so a row is needed that asks twice and looks at the kernel log. Pure, so
# --self-test drives it and the row below uses the same code:
#   <turn 1 body> <turn 2 body> <TDRs before> <TDRs after>
followup_verdict() {
    _r1=$1 _r2=$2 _t0=$3 _t1=$4
    case "$_r1" in
        *'"error"'*) printf 'FAIL the long turn answered an error: %.100s' "$_r1"; return 0 ;;
        *'"finish_reason"'*) ;;
        *) printf 'FAIL the long turn did not finish: %.100s' "${_r1:-no body}"; return 0 ;;
    esac
    case "$_r2" in
        *'"error"'*) printf 'FAIL the follow-up answered an error: %.100s' "$_r2"; return 0 ;;
    esac
    if [ -z "$(json_str "$_r2" content)" ] && [ -z "$(json_str "$_r2" reasoning_content)" ]; then
        printf 'FAIL the follow-up gave no content and no reasoning: %.100s' "${_r2:-no body}"; return 0
    fi
    case "$_r2" in
        *'"finish_reason"'*) ;;
        *) printf 'FAIL the follow-up came back without a finish_reason, i.e. a cut reply: %.100s' "$_r2"; return 0 ;;
    esac
    # An unreadable kernel log (empty count) is said out loud rather than counted as clean.
    if [ -z "$_t0" ] || [ -z "$_t1" ]; then
        printf 'PASS the follow-up answered; the kernel log was not readable, so no TDR was ruled out'
        return 0
    fi
    if [ "$_t1" -gt "$_t0" ]; then
        printf 'FAIL the NPU logged a TDR during the two turns (%s -> %s): see dmesg' "$_t0" "$_t1"; return 0
    fi
    printf 'PASS the follow-up answered after a long turn, no TDR (still %s)' "$_t1"
}

# The NPU row's STREAMING decision (engine #911, c21 board 2026-11-04). The page streams, and a stream can
# fail after its 200: the engine answers a mid-generation failure with `data: {"error":...}` inside the body,
# so the HTTP status says nothing at all - on the board two such 200s carried "npu runtime: invalid call (code
# 614): state 8, run 0 of 28" while every row that looks at a status stayed green. Pure, so --self-test drives
# it and the row below uses the same code: <sse body> <TDRs before> <TDRs after>.
stream_verdict() {
    _b=$1 _t0=$2 _t1=$3
    case "$_b" in
        '') printf 'FAIL nothing came back at all'; return 0 ;;
    esac
    case "$_b" in
        *'"error"'*) printf 'FAIL the reply carried an error inside the stream: %.100s' \
                          "$(printf '%s' "$_b" | sed -n 's/.*"message"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
                      return 0 ;;
    esac
    case "$_b" in
        # '[[]' outside the quotes is a literal '['; inside them the shell would take the text as-is and the
        # pattern would never match, and unquoted '[DONE]' is a character class (the 'perfopt+' trap again)
        *'data: '[[[]DONE]*) ;;
        *) printf 'FAIL the stream ended without data: [DONE], so the backend stopped mid-reply: %.80s' "$_b"
           return 0 ;;
    esac
    _n=$(printf '%s' "$_b" | grep -oE '"(content|reasoning_content)":"[^"]' | wc -l | tr -d ' ')
    [ "${_n:-0}" -gt 0 ] || { printf 'FAIL the stream carried no text chunk at all'; return 0; }
    if [ -z "$_t0" ] || [ -z "$_t1" ]; then
        printf 'PASS %s text chunk(s) and [DONE]; the kernel log was not readable, so no TDR was ruled out' "$_n"
        return 0
    fi
    [ "$_t1" -gt "$_t0" ] && { printf 'FAIL the NPU logged a TDR during the turn (%s -> %s): see dmesg' "$_t0" "$_t1"; return 0; }
    printf 'PASS %s text chunk(s) and [DONE] over the stream, no TDR (still %s)' "$_n" "$_t1"
}

# The NPU row's BUSY decision (engine#1024, c23 board 2026-11-04). A single-user appliance's second chat has to
# queue behind the first, not be refused. On the board the owner's chat got
# `{"message":"the NPU is busy with another request"}` inside a 200, because the serve waited --queue-ms (5 s) for
# the device and gave up while the check's own 1000-token ladder row held it for 14 s. The refusal is a frame in
# the body, so a row that looks at the status passes it. Pure, so --self-test drives it and the row below uses the
# same code: <the long chat's body> <the queued chat's body> <TDRs before> <TDRs after>.
queue_verdict() {
    _q1=$1 _q2=$2 _t0=$3 _t1=$4
    case "$_q1" in
        '') printf 'FAIL the long chat, which owns the device, came back empty'; return 0 ;;
        *'"error"'*) printf 'FAIL the long chat answered an error: %.100s' "$_q1"; return 0 ;;
    esac
    case "$_q2" in
        ''|'000') printf 'FAIL the queued chat came back empty: a refusal still has a body, so this is a dropped request'
                   return 0 ;;
    esac
    case "$_q2" in
        *'"error"'*) printf 'FAIL the second chat was refused while the device was busy: %.120s' \
                          "$(printf '%s' "$_q2" | sed -n 's/.*"message"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)"
                     return 0 ;;
    esac
    case "$_q2" in
        *'data: '[[[]DONE]*) ;;
        *) printf 'FAIL the queued chat ended without data: [DONE], so it was cut rather than answered: %.80s' "$_q2"
           return 0 ;;
    esac
    _qn=$(printf '%s' "$_q2" | grep -oE '"(content|reasoning_content)":"[^"]' | wc -l | tr -d ' ')
    [ "${_qn:-0}" -gt 0 ] || { printf 'FAIL the queued chat carried no text chunk at all: a queue that ends in silence is the #911 shape'
                               return 0; }
    if [ -z "$_t0" ] || [ -z "$_t1" ]; then
        printf 'PASS the second chat queued behind a 1000-token turn and answered (%s chunk(s), [DONE]); the kernel log was not readable, so no TDR was ruled out' "$_qn"
        return 0
    fi
    [ "$_t1" -gt "$_t0" ] && { printf 'FAIL the NPU logged a TDR while the second chat queued (%s -> %s): see dmesg' "$_t0" "$_t1"
                               return 0; }
    printf 'PASS the second chat queued behind a 1000-token turn and answered (%s chunk(s), [DONE]), no TDR (still %s)' "$_qn" "$_t1"
}

# The same row's OTHER arm (engine#1024): a **non-streamed** request cannot be told it is queueing, so it must be
# REFUSED at its own short bound (`--queue-ms-plain`, 5000 ms on the appliance) with `503 + Retry-After` - not parked
# silently for the whole queue. Without this arm the row cannot see the path the fix's second half is about, because
# both of its chats stream. Pure, so --self-test drives it: <long chat body> <status> <refusal body> <Retry-After>
# <TDRs before> <TDRs after>.
plain_queue_verdict() {
    _p1=$1 _ps=$2 _pb=$3 _pr=$4 _t0=$5 _t1=$6
    case "$_p1" in
        *'data: '[[[]DONE]*) ;;
        *) printf 'FAIL the long (1000-token) chat, which owns the device, came back without data: [DONE]: %.120s' "${_p1:-no body}"
           return 0 ;;
    esac
    case "$_ps" in
        503) ;;
        000|'') printf 'FAIL the non-streamed chat got no HTTP status at all, so nothing can be said about its bound'
                return 0 ;;
        200) printf 'FAIL the non-streamed chat was ANSWERED while the device was still busy: the short bound did not apply'
             return 0 ;;
        *) printf 'FAIL the non-streamed chat answered %s, not the 503 + Retry-After a client can act on' "$_ps"
           return 0 ;;
    esac
    case "$_pb" in
        *device_busy*|*'the NPU is busy'*) ;;
        *) printf 'FAIL the 503 did not say the device is busy, so a client cannot tell it from a broken model: %.120s' "${_pb:-empty body}"
           return 0 ;;
    esac
    case "$_pr" in
        ''|*[!0-9]*) printf 'FAIL the 503 carried no numeric Retry-After (got "%s"), so the refusal names no wait' "${_pr:-none}"
                     return 0 ;;
    esac
    if [ -z "$_t0" ] || [ -z "$_t1" ]; then
        printf 'PASS the non-streamed chat was refused at its short bound (503, Retry-After %s) instead of waiting silently; the kernel log was not readable, so no TDR was ruled out' "$_pr"
        return 0
    fi
    [ "$_t1" -gt "$_t0" ] && { printf 'FAIL the NPU logged a TDR during the non-streamed arm (%s -> %s): see dmesg' "$_t0" "$_t1"
                               return 0; }
    printf 'PASS the non-streamed chat was refused at its short bound (503, Retry-After %s) instead of waiting silently, no TDR (still %s)' "$_pr" "$_t1"
}

# The first value of a JSON key - quoted or not, since usage counts are bare numbers - for the row's
# small bodies (one line, one message: the same assumption chat() already makes).
json_str() { printf '%s' "$1" | sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*/\\1/p" | head -1; }

# Text on its way into a JSON string: no control characters (a newline would end the line) and the
# two characters JSON escapes. Used to replay the long turn's own answer as the follow-up's history,
# as the web page does.
json_text() { printf '%s' "$1" | tr '\n\r\t' '   ' | sed 's/\\/\\\\/g; s/"/\\"/g'; }


laya_class_model() {  # <class> -> the model the shipped table names for it
    case "$1" in
        short)    printf 'Qwen3-0.6B-1bit' ;;
        prose)    printf 'ZAYA1-8B-1bit' ;;
        code)     printf 'Qwen3-Coder-30B-A3B-Instruct-1bit' ;;
        long_doc) printf 'Qwen3.8-27B-1bit' ;;
        *)        printf '' ;;
    esac
}

# The laya-auto row, decided from the router's own decision (docs/laya-first.md §7). Pure, so
# --self-test drives it: <response body carrying x_lemonade_route> <class> <model>. All four have to
# hold: the class's Laya rule matched, the request went to that class's model, the request did NOT fall
# open to default_model, and Laya's score is in the trace. A request that landed on the right model by
# accident - a deterministic rule, or the default - is a FAIL: the row exists to prove Laya decided.
#
# Spaces are stripped before matching because the decision is embedded JSON whose spacing is the
# server's business; none of the field names or values matched here carry spaces.
laya_verdict() {
    _b=$(printf '%s' "${1:-}" | tr -d ' \n\r\t')
    _cls=${2:-} _want=${3:-}
    # A gate fails closed on input it cannot interpret: a class this script has no model for would
    # otherwise skip both the class and the model check and PASS on shape alone (a row added without
    # touching laya_class_model is exactly how that happens).
    if [ -n "$_cls" ] && [ -z "$(laya_class_model "$_cls")" ]; then
        printf 'FAIL no model is named for class %s in this script' "$_cls"; return
    fi
    printf '%s' "$_b" | grep -q '"matched_rule":"laya-' || {
        printf 'FAIL no Laya rule matched the request'; return; }
    _rule=$(printf '%s' "$_b" | sed -n 's/.*"matched_rule":"laya-\([^"]*\)".*/\1/p')
    if [ -n "$_cls" ] && [ "$_rule" != "$_cls" ]; then
        printf 'FAIL matched the %s rule, not the %s one' "$_rule" "$_cls"; return
    fi
    printf '%s' "$_b" | grep -q '"default_used":true' && {
        printf 'FAIL fell open to default_model, so Laya did not decide'; return; }
    printf '%s' "$_b" | grep -q '"condition":"classifier:laya"' || {
        printf 'FAIL no Laya score in the trace'; return; }
    if [ -n "$_want" ]; then
        printf '%s' "$_b" | grep -q "\"route_to\":\"$_want\"" || {
            printf 'FAIL routed to another model than %s' "$_want"; return; }
    fi
    printf 'PASS'
}

# The NPU must be held awake (owner decision 2026-11-05 23:4x ADT: "keep the npu awake on the stick"; this is c30's
# `code 614` fix). The driver's runtime PM suspends the array between requests, and on the c29 take the request that
# followed such a window died `npu runtime: invalid call (code 614): state 8, run 0 of 28` - the picture page's prompt
# turn - and the fallback model wrote the prompt, so the client saw a clean answer and the demo lost the beat. The image
# sets `power/control=on` with a udev rule plus an early-boot fallback (a module loaded before udev misses the rule), so
# what this row reads is the state THIS boot has, not the intent of a rule file. `runtime_status` is asserted as well
# because `control=on` alone can be set while the array is not running at all - two different faults, two different
# messages. Pure, so --self-test drives it: <power/control> <runtime_status>
npu_power_verdict() {
    _c=$(printf '%s' "${1:-}" | tr -d ' \n\r\t') _s=$(printf '%s' "${2:-}" | tr -d ' \n\r\t')
    if [ -z "$_c" ]; then
        printf 'SKIP no power/control to read'; return 0
    fi
    if [ "$_c" != on ]; then
        printf 'FAIL power/control=%s (runtime_status=%s): the array may runtime-suspend between requests, which is the state the demo'"'"'s prompt turn died in' "$_c" "${_s:-unreadable}"
        return 0
    fi
    case "$_s" in
        active) printf 'PASS power/control=on, runtime_status=active' ;;
        '')     printf 'FAIL power/control=on but runtime_status is unreadable, so the array'"'"'s state is not proven' ;;
        *)      printf 'FAIL power/control=on but runtime_status=%s: held awake and not running' "$_s" ;;
    esac
}

# The HRX weight placement (engine #1339's mitigation), decided from two measurements of the same model.
# Importing the loader's mapping instead of copying it to the device (fork #226, engine #856) saves GTT - Qwen3-4B goes
# from 3925 to 1243 MiB of the process's own GTT - but on large files the import never finishes: a dense 16.4 GiB 27B and
# a 76.3 GiB Flash-Next sat in it for 72 minutes with no error, while the device copy loads in 20-75 s (#1339). Fork
# #231 (pinned since engine #1343) makes the import OPT-IN: `GGML_HRX_WEIGHT_HOST_PLACEMENT=1` asks for it, unset or 0
# keeps the device copy, which is what a stick needs. Nothing prints that switch - not `--list-devices` (name and size
# only), not the backend (no log at the decision), not `1bit serve --print-backend-env` - so the row measures the thing
# itself, the way the issue did: load one model twice and weigh each serve process's own GTT.
# Pure, so --self-test drives it: <default arm MiB> <opt-in arm MiB> [required % more].
hrx_placement_verdict() {
    _d=${1:-} _i=${2:-} _f=${3:-15}
    case "$_d" in ''|*[!0-9]*) printf 'FAIL the default arm'"'"'s own GTT is unreadable (%s)' "${_d:-empty}"; return 0 ;; esac
    case "$_i" in ''|*[!0-9]*) printf 'FAIL the opt-in arm'"'"'s own GTT is unreadable (%s)' "${_i:-empty}"; return 0 ;; esac
    if [ "$_d" -le 0 ] || [ "$_i" -le 0 ]; then
        printf 'FAIL an arm reported %s / %s MiB of own GTT: that is not a measurement' "$_d" "$_i"; return 0
    fi
    # the issue's own arms are 1.89x (4B, 5692 -> 3010 MiB) and 3.16x (4B, docs); 15 % is a conservative floor that
    # cannot be reached by noise, and being relative it needs no absolute MiB threshold to drift with ctx or slots.
    if [ "$_d" -ge $(( _i + _i * _f / 100 )) ]; then
        printf 'PASS the default arm costs %s MiB of its own GTT against %s MiB with GGML_HRX_WEIGHT_HOST_PLACEMENT=1 (%s%% more): the shipped default takes the device copy, and the import still works when it is asked for' "$_d" "$_i" "$_f"
    else
        printf 'FAIL the default arm used %s MiB and the opt-in arm %s MiB - the default is not the device copy that loads a large model (#1339); check that this build carries fork #231' "$_d" "$_i"
    fi
}

# The payload's modes (engine #1352, fixed by #1361). Until that fix `package-linux.sh` staged with `cp -a`, so every mode was
# inherited from the build tree: 134 of the 195 files under the payload were group- or other-writable (1bit and the HRX libs at
# 0775, LICENSE/NOTICE at 0664). All root:root, so nothing non-root could reach those bits, and no appliance user is in group
# root - the defect is that a mode in a release artifact was nobody's decision. #1361 makes the packer normalise (dirs 0755,
# files 0755 if executable else 0644) and assert it, so THIS row is the board-side proof that it did: a payload packed after that
# commit has no writable file, and a payload packed before it has 134. Pure, so --self-test drives it:
#   <group/other-writable file count> [<first few paths>]
# ── the identity file must be TRUE, not merely present ─────────────────────────────────────────
# /etc/1bit/release-identity states which payload and which engine the image carries. Nothing checked that the statement was true,
# and on 8 Nov it was not: c32 and c33 both shipped `payload=lemonade-onebit-rc-2026-11-05-c26c` with `engine_sha256=89a6c386…`
# (cycle 26's launcher) while the payloads carried `80aceca1…` and `7ba21108…`. The cause is per-path writers: os/v3/puppy/
# container-build.sh writes the file when it builds a base SFS, os/mini/mkimage.sh writes a different schema, and
# os/v3/puppy/mkimage.sh - what a payload roll like c33 runs - writes neither, so a cut reusing a base SFS inherits whatever the
# SFS was born with. This row used to `pass identity "$(tr '\n' ' ' < $ID)"`: it printed the lie and called it a PASS. It now
# compares the claims with the shipped bytes. Pure, so --self-test drives it: <claimed engine> <actual launcher> <claimed libhsa> <actual libhsa>
# Which file's bytes does an `engine_sha256` claim have to match? Two schemas, two layouts (engine#1481/#1541):
#   * v3 (`container-build.sh`): the payload launcher at /opt/1bit-package/1bit/1bit;
#   * mini (`os/mini/mkimage.sh`): the engine the image RUNS, /bin/1bit - a symlink to the image's own copy. Inside the
#     booted image that symlink resolves to the image copy, which is why reading it here is right; on the build host the
#     same expression escapes to the build tree, so the WRITER hashes $root$bin instead of hashing the symlink.
# An empty first argument means "no checkable claim"; then the git short sha (if any) decides whether the row SKIPs.
# Pure, so --self-test drives it: <claimed engine_sha256|""> <git sha|"">
identity_launcher_path() {
    [ -n "${1:-}" ] || return 0
    if [ -n "${2:-}" ]; then printf '%s\n' "${MINI_LAUNCHER:-/bin/1bit}"
    else printf '%s\n' "${PAYLOAD_DIR:-/opt/1bit-package}/1bit/1bit"; fi
}

# The payload's own identity (engine#1557). `payload=` in release-identity is a LABEL - the tarball stem, which the console
# shows and which no fact inside the image can equal (the payload directory is /opt/1bit-package, and nothing in the payload
# records the stem) - so it is printed and NOT asserted. What IS checkable is the version string the payload's own launcher
# reports; the writer reads it from the payload's bytes at build time and stores it as `payload_lemond=`, and this row
# compares the claim with what the payload on this machine says about itself. Two different claims, never mixed: the label is
# presented as a label, and the version is a real check. Pure, so --self-test drives it: <claimed> <actual>
payload_identity_verdict() {   # <payload_lemond claimed> <what the launcher reports>
    _pc=${1:-} _pa=${2:-}
    if [ -z "$_pc" ]; then
        printf 'SKIP this image'"'"'s identity carries no payload_lemond=, so the payload it names cannot be checked (engine#1557: only images built after this row have it)'
        return 0
    fi
    if [ -z "$_pa" ]; then
        printf 'FAIL the payload'"'"'s launcher reported no version, so the identity'"'"'s claim (%s) cannot be checked against it' "$_pc"
        return 0
    fi
    # COMPARE THE PIN, NOT THE WHOLE STRING. `lemond --version` is `2026.47.0~1.52a2a6b3`, and the `~N` is the commit count of
    # the CHECKOUT IT WAS BUILT IN (docs/releases.md: a depth-1 clone reports ~1, a full clone ~1752 for the same commit; c18
    # printed ~5). So the whole string is not a fact about the payload: an image built from a depth-1 clone and a machine payload
    # built from a full clone carry the same code and print different strings, and comparing them would red on a CORRECT pair.
    # What is a fact about the payload is the week pin and the trailing sha - `2026.47.0` and `52a2a6b3` - so the count is dropped
    # on both sides before comparing, and the verdict says so.
    _pc_pin=$(printf '%s' "$_pc" | sed 's/~[0-9]*\././')
    _pa_pin=$(printf '%s' "$_pa" | sed 's/~[0-9]*\././')
    if [ -n "$_pc_pin" ] && [ "$_pc_pin" = "$_pa_pin" ]; then
        printf 'PASS the identity claims payload_lemond=%s and the payload'"'"'s own launcher reports the same pin (%s, clone count ignored: the ~N is commits in the checkout that built it)' "$_pc" "$_pa_pin"
    else
        printf 'FAIL the identity claims payload_lemond=%s (pin %s) but the payload here reports %s (pin %s) - the name is stale, or the payload is not the one that was packed (engine#1557)' \
            "$_pc" "${_pc_pin:-?}" "$_pa" "${_pa_pin:-?}"
    fi
}

identity_verdict() {
    _ce=${1:-} _ae=${2:-} _ch=${3:-} _ah=${4:-}
    case "$_ce" in ''|*[!0-9a-f]*) printf 'FAIL the identity file'"'"'s engine_sha256 is not a hex digest (%s) - malformed, not merely old' "${_ce:-empty}"; return 0 ;; esac
    [ "${#_ce}" = 64 ] || { printf 'FAIL the identity file'"'"'s engine_sha256 is %s characters, not 64 - malformed, not merely old' "${#_ce}"; return 0; }
    # `libhsa_sha256=none` is a legitimate claim - "this image ships no runtime of its own" - and it is checkable, not a pass by
    # absence: it is true only if nothing /etc/1bit/libhsa.env names on the loader path is actually there. A cut that writes `none`
    # while libhsa.env still names a runtime has written a false statement, and the board's u5-hrxenv row pins its comparison to this
    # same field, so a `none` there fails that row too - with a message that blames the runtime instead of the file.
    if [ -z "$_ae" ]; then
        printf 'FAIL the identity file claims engine_sha256=%.16s but there is no payload launcher to compare it with - an uncheckable claim is not evidence' "$_ce"
        return 0
    fi
    if [ "$_ce" != "$_ae" ]; then
        printf 'FAIL the identity file says engine_sha256=%.16s but the payload'"'"'s launcher is %.16s - this cut does not write the file (os/v3/puppy/mkimage.sh writes it not at all), so it is an inherited base-SFS value naming an older engine' "$_ce" "$_ae"
        return 0
    fi
    if [ -n "$_ah" ] && [ "$_ch" != "$_ah" ]; then
        printf 'FAIL the identity file says libhsa_sha256=%.16s but the shipped runtime is %.16s - the image would name a runtime it does not carry' "$_ch" "$_ah"
        return 0
    fi
    case "$_ch" in
        not-stated)   # the mini schema writes tag/lemonade/engine/engine_sha256/built and no libhsa_sha256 at all, so there is
                      # nothing to check in that field. "Not stated" is not "none" (which claims the image ships no runtime) and
                      # it is not malformed - but it is only acceptable when the engine half IS checkable, or the row would PASS
                      # on a file with nothing in it.
            printf 'PASS consistent: engine_sha256=%.16s is the shipped engine; this schema states no libhsa_sha256, so that half is not claimed' "$_ce"
            return 0 ;;
        none)
            if [ -n "$_ah" ]; then
                printf 'FAIL the identity file says libhsa_sha256=none but the image loads a runtime libhsa.env names, sha256 %.16s - either write that sha or say why the two disagree' "$_ah"
            else
                printf 'PASS consistent: engine_sha256=%.16s is the shipped launcher and libhsa_sha256=none is true (libhsa.env names no runtime that exists here)' "$_ce"
            fi
            return 0 ;;
        ''|*[!0-9a-f]*) printf 'FAIL the identity file'"'"'s libhsa_sha256 is neither a hex digest nor none (%s) - malformed, not merely old' "${_ch:-empty}"; return 0 ;;
    esac
    printf 'PASS consistent: engine_sha256=%.16s and libhsa_sha256=%.16s are the shipped bytes, read off the payload rather than out of the file' "$_ce" "$_ch"
}

# ── the NPU's locked memory, read from the process that locks it ─────────────────────────────────
# Two mechanisms set this limit and either can silently override the other: /etc/security/limits.d/*.conf (pam_limits, files
# applied in LEXICAL order, the last assignment for an item wins) covers anything PAM starts, and
# /etc/profile.d/90onboard-limits sets it for a console session. A row that reads one of them passes on a machine where the
# other is overridden - measured 2026-11-08: this fleet's own box carries a hand-written
# /etc/security/limits.d/99-amdnpu-memlock.conf, and a higher-numbered user file wins over the 90- one our package ships
# (scripts/mk-kernel-packages.sh), so the value we intend is not necessarily the value in force.
# So this row reads neither file: it reads the effective soft limit of the process that actually locks command buffers
# (/proc/<pid>/limits, "Max locked memory") and, when it is not unlimited, names the highest-numbered limits file that sets
# memlock - the file whose value won, whether or not we shipped it.
# Pure, so --self-test drives it: <effective soft limit> <process name> [<winning file>]
memlock_verdict() {
    _v=${1:-} _p=${2:-} _f=${3:-}
    if [ -z "$_p" ]; then
        printf 'SKIP no lemond/1bit process is running, so the locked-memory limit in force cannot be read (this row measures the process that locks, not the files that configure it)'
        return 0
    fi
    case "$_v" in
        '') printf 'FAIL %s is running but its /proc limits could not be read, so the locked-memory limit in force is unknown' "$_p"; return 0 ;;
        unlimited) printf 'PASS %s may lock unlimited memory (soft limit read from /proc/.../limits, the value in force, not the value a file asks for)' "$_p"; return 0 ;;
        *[!0-9]*) printf 'FAIL %s'"'"'s locked-memory soft limit reads %s, which is neither a byte count nor unlimited' "$_p" "$_v"; return 0 ;;
    esac
    printf 'FAIL %s'"'"'s locked-memory soft limit is %s bytes, not unlimited, so a large NPU buffer allocation fails with "Lock limit exceeded"%s - that is the limit IN FORCE in the process; the files that ask for one are the highest-numbered limits.d files and /etc/profile.d, and a process can also fail to inherit an unlimited file (measured 2026-11-08: a lemond started outside the appliance'\''s sessions had 8388608 while this box'\''s own 99- file says unlimited)' \
        "$_p" "$_v" "${_f:+ (highest-numbered memlock file: $_f)}"
}

# ── the payload= label against the payload the image ships (engine#1557, containment) ───────────────
# `payload=` is a LABEL (the tarball stem, which no in-image fact equals) and the row above checks the payload's own version. Neither
# settles the incident: an image carrying payload=…c26c with a correctly computed payload_lemond passes both. The one in-image fact that
# DOES name the cycle is the payload's own README - `opt/1bit-package/1bit/README.txt` line 1 states `1bit engine rc-YYYY-MM-DD-cNN` - and
# `payload=` CONTAINS that token (it never equals it: the label carries the stem). So the comparison is containment, on the first line only.
payload_name_verdict() {
    _pn=${1:-} _pt=${2:-}
    case "$_pn" in
        '')
            printf 'FAIL the identity file states no payload at all (payload= is empty), so nothing says which payload this image carries' ;;
        unknown)
            printf 'FAIL the identity file states payload=unknown - the value is an environment input (container-build.sh: PAYLOAD_NAME), so "unknown" means the build never named what it shipped' ;;
        *)
            if [ -z "$_pt" ]; then
                printf 'SKIP the payload ships no release tag in its README, so payload=%s cannot be checked in the image (the tarball name lives beside the bundle, not in the image)' "$_pn"
            else
                # the label must CONTAIN the token with a BOUNDARY after it: `c33x` must not match `c33`, `c3` must not match `c33`.
                # Unanchored containment passed `…-c26b` against a `c26c` payload, and those letter-suffixed names are exactly the
                # incident's (this box carries payload-c17b, -c26b and -c26c).
                case "$_pn" in
                    *"$_pt"|*"$_pt"[!0-9a-z]*) printf 'PASS consistent: the identity says payload=%s and the payload ships %s' "$_pn" "$_pt" ;;
                    *) printf 'FAIL the identity says payload=%s but the payload ships %s - payload= is a declared environment input, so this image names a release it does not carry (c32 and c33 shipped exactly this)' "$_pn" "$_pt" ;;
                esac
            fi ;;
    esac
}

payload_name_from_image() {  # <identity file> <payload dir>
    _pi=${1:-} _pd=${2:-}
    _pval=$(sed -n 's/^payload=//p' "$_pi" 2>/dev/null | head -1)
    _ptag=
    if [ -n "$_pd" ] && [ -f "$_pd/1bit/README.txt" ]; then
        _ptag=$(sed -n '1s/.*\(rc-[0-9][0-9]*-[0-9][0-9]*-[0-9][0-9]*-c[0-9][0-9]*[a-z]*\).*/\1/p' "$_pd/1bit/README.txt" 2>/dev/null)
    fi
    payload_name_verdict "$_pval" "$_ptag"
}

# ── the shipped kernel carries the ip_options UAF fix (CVE-2026-98432) ─────────────────────────
# CVE-2026-98432: a remote, unauthenticated use-after-free in IPv4 `ip_options` (the request-socket path). The fix is to free
# `ireq_opt` with `kfree_rcu` in `tcp_v4_reqsk_destructor`; upstream fixed it in 5.10.272, 5.15.223, 6.1.190, 6.6.159, 6.12.113,
# 6.18.56, 7.2.10, 7.3 - and 7.4-rc1 carries it.
#
# Two kinds of evidence, and the STRONGER one decides where it exists:
#   * a kernel source tree on the image (`net/ipv4/tcp_ipv4.c`): the function must free with `kfree_rcu`, which is the fix itself
#     and outranks a version number - a backport can carry the fix below the floor (this box's 7.3-rc4 perfopt tree does), and a
#     vendor tree can lack it above the floor;
#   * otherwise the VERSION FLOOR for the shipped kernel's own line, which is what an appliance with no sources can offer.
# Pure, so --self-test drives it: <kernel release> <source verdict: yes|no|absent>
kernel_cve_verdict() {
    _v=${1:-} _src=${2:-absent}
    case "$_src" in
        yes) printf 'PASS %s'"'"'s tcp_v4_reqsk_destructor frees ireq_opt with kfree_rcu (read from its own net/ipv4/tcp_ipv4.c, which outranks the version floor)' "$_v"; return 0 ;;
        no)  printf 'FAIL %s'"'"'s tcp_v4_reqsk_destructor frees ireq_opt WITHOUT kfree_rcu - the CVE-2026-98432 use-after-free, whatever the version says' "$_v"; return 0 ;;
    esac
    # version floor per line, from the advisory: a patch equal to the floor is the fix; a PRE-release at that patch is not.
    # The advisory writes "7.4-rc1" and "5.10.272"; `uname -r` writes "7.4.0-rc1-1bit". Both must parse, so the
    # numeric part is taken from the text BEFORE the first '-': without this, `_min` kept the suffix ("4-rc1") and
    # every `X.Y-rcN` release - the advisory's own 7.4-rc1 included - fell through to "cannot be parsed".
    _vbase=${_v%%-*}
    _maj=${_vbase%%.*}; _rest=${_vbase#*.}
    _min=${_rest%%.*}; _rest2=${_rest#*.}
    if [ "$_rest2" = "$_rest" ]; then _pat=0; else _pat=${_rest2%%[!0-9]*}; fi   # "7.3" is 7.3.0, not 7.3.3
    _rc=$(printf '%s' "$_v" | sed -n 's/.*-rc\([0-9][0-9]*\).*/\1/p')
    case "$_maj$_min" in ''|*[!0-9]*) printf 'FAIL the kernel release (%s) cannot be parsed, so whether it carries the CVE-2026-98432 fix is unknown' "${_v:-empty}"; return 0 ;; esac
    case "$_pat" in ''|*[!0-9]*) printf 'FAIL the kernel release (%s) has no patch number, so it is a pre-release of %s.%s and cannot be compared with the fix floor' "$_v" "$_maj" "$_min"; return 0 ;; esac
    if [ "$_maj" -gt 7 ] || { [ "$_maj" = 7 ] && [ "$_min" -ge 4 ]; }; then
        # the 7.4 line: 7.4-rc1 is the advisory's own "carries it"; earlier rc of 7.4 is unknown, and 7.4 release is fixed
        if [ -z "$_rc" ] || [ "$_rc" -ge 1 ]; then
            printf 'PASS %s is at or after 7.4%s, the advisory'"'"'s own floor for this fix (no kernel source on the image to read the function from)' "$_v" "$([ -n "$_rc" ] && echo "-rc1")"
        else
            printf 'FAIL %s predates 7.4-rc1, the advisory'"'"'s floor for this fix' "$_v"
        fi
        return 0
    fi
    case "$_maj.$_min" in
        5.10) _floor=272 ;; 5.15) _floor=223 ;; 6.1) _floor=190 ;; 6.6) _floor=159 ;;
        6.12) _floor=113 ;; 6.18) _floor=56 ;; 7.2) _floor=10 ;;
        7.3)  _floor=0 ;;
        *) printf 'FAIL no fix floor is known for the %s.%s line (the advisory lists 5.10/5.15/6.1/6.6/6.12/6.18/7.2/7.3/7.4), so this kernel cannot be confirmed either way' "$_maj" "$_min"; return 0 ;;
    esac
    if [ "$_pat" -gt "$_floor" ]; then
        printf 'PASS %s is after %s.%s.%s, the advisory'"'"'s floor for its line (no kernel source on the image to read the function from)' "$_v" "$_maj" "$_min" "$_floor"
    elif [ "$_pat" -eq "$_floor" ] && [ -z "$_rc" ]; then
        printf 'PASS %s is the advisory'"'"'s floor for its line (no kernel source on the image to read the function from)' "$_v"
    elif [ "$_pat" -eq "$_floor" ] && [ -n "$_rc" ]; then
        printf 'FAIL %s is a pre-release AT the floor patch %s.%s.%s - the fix landed in the release, not the rc' "$_v" "$_maj" "$_min" "$_floor"
    else
        printf 'FAIL %s is before %s.%s.%s, the advisory'"'"'s floor for its line' "$_v" "$_maj" "$_min" "$_floor"
    fi
}

payload_modes_noexec() {   # <payload dir> -> the binaries present but not executable, space-separated and payload-relative
    # engine#2422: the row prints the modes, and printing is not asserting. A 0644 `1bit/1bit` passes the writability rule and then
    # makes hrx-placement SKIP (`[ -x ]` cannot find a runnable engine), so the check was green over a payload that cannot run.
    # Absence is deliberately NOT an offender: a binary that is not here is hrx-device's or identity's claim, and a payload still
    # being extracted is caught by the entry floor above.
    #
    # `[ -x ]` is the right predicate and NOT a stricter "is it 0755" (engine#2424's phrasing, worth keeping): the appliance runs as
    # root, and access(2) documents that a real UID of zero passes X_OK "if execute permission is enabled for any of the file owner,
    # group, or other". So the property is *can it execute*, which is exactly what the row that would start the engine tests.
    _pe_dir=${1:-} _pe_out=
    for _pe_b in "$_pe_dir/1bit/1bit" "$_pe_dir/bin/lemond" "$_pe_dir/lemond-onebit"; do
        [ -f "$_pe_b" ] || continue
        { [ -x "$_pe_b" ] || _pe_out="$_pe_out ${_pe_b#"$_pe_dir"/}"; } 2>/dev/null || true
    done
    printf '%s' "$_pe_out"
}

payload_modes_verdict() {   # <writable files> <writable directories> [<sample>]
    _f=${1:-} _d=${2:-} _sample=${3:-}
    case "$_f" in ''|*[!0-9]*) printf 'FAIL the payload'"'"'s file modes could not be counted (%s)' "${_f:-empty}"; return 0 ;; esac
    case "$_d" in ''|*[!0-9]*) printf 'FAIL the payload'"'"'s directory modes could not be counted (%s)' "${_d:-empty}"; return 0 ;; esac
    if [ "$_f" = 0 ] && [ "$_d" = 0 ]; then
        printf 'PASS no path in the payload is group- or other-writable - neither a file nor a directory (engine #1352: the packer normalises, this is the proof)'
    else
        # FILES AND DIRECTORIES BOTH COUNT. The first version of this row counted files only, which was the same "too narrow" shape as
        # the /home/-only and 1bit-only checks: measured 8 Nov, c32's shipped payload had **134 writable files AND 49 writable
        # directories** (183 paths, not 134), so a files-only row would pass a payload that had re-grown writable directories - and
        # the c33 cut's first image did exactly that (0 files, 1 directory: the payload root at 0775, because `tar -x` creates the
        # top directory with the caller's umask while every entry below it carries the archive's stored modes).
        printf 'FAIL %s file(s) and %s dir(s) in the payload are group- or other-writable%s - a packing site is not normalising (engine #1352: the fix is in scripts/package-linux.sh / pack-elf.sh, and for an unpacked payload the extraction root is the usual one)' \
            "$_f" "$_d" "${_sample:+: $_sample}"
    fi
}

# The BASE layer's modes (engine#1474). The payload row above closes the same class for opt/1bit-package; this one is one
# layer down, where woof-CE's build left 2,176 group/other-writable paths (1,814 of them files) inherited from the
# builder's umask. The appliance runs as root, so none of them was reachable by anything that should not have it - the
# defect is the same one: a mode in a release artifact that nobody decided.
#
# The keep-list is the point of the row as much as the count: eight classes of path DO need to stay writable (tmp,
# var/tmp, var/run, var/lock, var/spool, var/log, run, dev, proc, sys), and a cycle that silently changes that list is
# exactly what a row should make visible. So the row PRINTS it, and the verdict takes its size.
# Pure, so --self-test drives it: <writable count> <keep-list entries> [<sample>]
base_modes_verdict() {   # <writable> <keeplist entries> [<sample>]
    _b=${1:-} _k=${2:-} _bs=${3:-}
    case "$_b" in ''|*[!0-9]*) printf 'FAIL the base layer'"'"'s modes could not be counted (%s)' "${_b:-empty}"; return 0 ;; esac
    case "$_k" in ''|*[!0-9]*) printf 'FAIL the base keep-list could not be read (%s)' "${_k:-empty}"; return 0 ;; esac
    if [ "$_b" = 0 ]; then
        printf 'PASS no path in the BASE layer is group- or other-writable outside the %s keep-list entr(y/ies) - the packer normalises them and this is the proof (engine #1474)' "$_k"
    else
        printf 'FAIL %s path(s) in the BASE layer are group- or other-writable outside the keep-list%s - a packing site is not normalising (engine #1474: scripts/base-modes.sh / pack-elf.sh)' \
            "$_b" "${_bs:+: $_bs}"
    fi
}

# The payload's RUNPATH (the orchestrator's rule of 2026-11-08, after an installed engine's RUNPATH into a build tree made a
# rebuild swap the libraries under two running services: coder-llm on :8090 crash-looped 154 times and 1bit-canonical on :8095
# 755 times). An installed package must carry `$ORIGIN` ONLY: it finds its own libraries beside itself, and a path into a build
# tree is a live wire to whatever the next build there leaves behind. This is the shipped half of that rule - the appliance has no
# readelf/objdump, so the row reads DT_RUNPATH/DT_RPATH out of the dynamic section itself (python3, which the image does ship) and
# falls back to readelf on a host. Pure, so --self-test drives it: <offending count> [<sample>]
payload_runpath_verdict() {
    _n=${1:-} _sample=${2:-}
    case "$_n" in ''|*[!0-9]*) printf 'FAIL the payload'"'"'s RUNPATHs could not be read (%s)' "${_n:-empty}"; return 0 ;; esac
    if [ "$_n" = 0 ]; then
        printf 'PASS every RUNPATH in the payload is $ORIGIN: the package finds its libraries beside itself and points at no build tree'
    else
        printf 'FAIL %s payload ELF(s) carry a RUNPATH or RPATH that is not $ORIGIN%s - an installed package must not point into a build tree (a rebuild there swaps the libraries under a running service)' \
            "$_n" "${_sample:+: $_sample}"
    fi
}

# ── the payload's home-path LAYOUT (engine#1543, the fourth leak class) ─────────────────────────────────────────
# The first three classes are about CONTENT - a path compiled into a map or a debug string, a build tree packaged as
# bytes, a symbol table. The fourth is the path delivered as the FILESYSTEM LAYOUT: `bin/1bit` was an ABSOLUTE symlink
# to /home/<builder>/wt/issue-709-build/1bit, so the image shipped the builder's home as a directory and every content
# check passed. Discriminator measured on the shipped c33 SFS and on the leaking mini image: an absolute symlink whose
# target RESOLVES TO A FILE under home|Users|root (0 in c33, 1 in the leaking image). Targets that are a directory or
# absent are named but not failed - `root/spot -> /home/spot` is legitimate, and a payload scanned on a build host
# cannot tell an image directory from a missing one.
# The sealed NPU sets live on the stick and the engine looks for them BESIDE ITSELF, so `init.sh` links
# /data/npu/kernels to <engine dir>/npu/kernels at boot. engine#1543 moved the engine from its builder's path to
# /opt/1bit and had to move this link with it; a link left behind (or a guard that skips because the engine dir is
# absent) is not cosmetic: the engine finds no set, registers no device, and the boot is otherwise green.
# Pure, so --self-test drives it: <sets present 0|1> <link target or empty>
npu_link_verdict() {
    _sets=${1:-} _link=${2:-}
    case "$_sets" in
        0) printf 'PASS no sealed NPU set on this boot, so there is no link to check beside the engine'; return 0 ;;
        1) ;;
        *) printf 'FAIL whether the sealed NPU sets are present could not be read (%s)' "${_sets:-empty}"; return 0 ;;
    esac
    if [ -z "$_link" ]; then
        printf 'FAIL the boot carries sealed NPU sets but nothing is linked beside the engine: the engine loads them from <engine dir>/npu/kernels, so it would register NO device on an otherwise green boot (engine#1543)'
    elif [ "$_link" = "${NPU_SETS_DIR:-/data/npu/kernels}" ]; then
        printf 'PASS the sealed NPU sets are linked beside the engine (%s -> %s)' "$_link" "${ENGINE_DIR:-/opt/1bit}/npu/kernels"
    else
        printf 'FAIL %s/npu/kernels points at %s, not at the set directory of this boot (%s): the engine would load the wrong sets or none (engine#1543)' "${ENGINE_DIR:-/opt/1bit}" "$_link" "${NPU_SETS_DIR:-/data/npu/kernels}"
    fi
}

payload_home_layout_bad() {   # <payload dir> -> "relpath|TARGET|file|dir|dangling" per absolute home-targeting link
    python3 - "$1" 2>/dev/null <<'PYSCAN'
import os, sys
root = sys.argv[1]
HOMES = ('/home/', '/Users/', '/root/')
for dp, dns, fns in os.walk(root):
    for n in sorted(dns + fns):
        p = os.path.join(dp, n)
        if not os.path.islink(p):
            continue
        t = os.readlink(p)
        if not t.startswith('/') or not t.startswith(HOMES):
            continue
        kind = 'file' if os.path.isfile(p) else ('dir' if os.path.isdir(p) else 'dangling')
        print('%s|%s|%s' % (os.path.relpath(p, root), t, kind))
PYSCAN
}

payload_home_layout_verdict() {   # <links at a FILE or at NOTHING> <links at a DIRECTORY> [<sample>]
    _bad=${1:-} _dir=${2:-} _s=${3:-}
    case "${_bad}:${_dir}" in
        *[!0-9:]*) printf 'FAIL the home-path links in the payload could not be counted (%s:%s)' "${_bad:-empty}" "${_dir:-empty}"; return 0 ;;
    esac
    # FAIL on a FILE target AND on a target that is NOT THERE. A dangling `bin/1bit -> /home/<builder>/wt/build/onebit`
    # is the leak twice over: the disclosure IS the path (the image still ships the builder's username and workdir as its
    # layout), and it is the worse failure to stand up - nothing to load, so the engine registers no device on an
    # otherwise fine boot. Only a DIRECTORY target is a different, legitimate thing (`root/spot -> /home/spot`, a home
    # on /data): measured on the shipped c33 image, the whole rootfs has exactly one of those and the payload has none.
    if [ "$_bad" != 0 ]; then
        printf 'FAIL %s absolute symlink(s) in the payload point into a home directory at a file or at nothing%s - the image ships the builder path as its layout and no content check can see it' \
            "$_bad" "${_s:+: $_s}"
    elif [ "$_dir" != 0 ]; then
        printf 'PASS no payload symlink points at a builder file or at nothing (0); %s name a home DIRECTORY%s, which is a home on /data and not this class' \
            "$_dir" "${_s:+: $_s}"
    else
        printf "PASS no absolute symlink in the payload points into a home directory: the payload's layout carries no builder path"
    fi
}

xhci_capture_verdict() {   # <1 when /bin/1bit-xhci-capture is present and executable, 0 when not>
    case "${1:-}" in
        1) printf 'PASS /bin/1bit-xhci-capture is installed and executable: a boot can capture the xHCI state for engine#823, which is the only place a plug can be observed' ;;
        0) printf 'FAIL /bin/1bit-xhci-capture is not installed: engine#823 asks whether a rear USB-A or Type-C port sees a device, and nothing on this medium can read that' ;;
        *) printf 'FAIL whether the xHCI capture is installed could not be read (%s)' "${1:-empty}" ;;
    esac
}

# --self-test: assert the lemonade verdict on the cases that matter (tests/release_check_row.sh runs
# this in CI). The unprepared-stick case is the one that used to false-PASS: 200 with the key, an empty
# body and no status without it.
if [ "${1:-}" = --self-test ]; then
    API=http://127.0.0.1:13305
    rc=0
    check() {  # <want> <got>
        want=$1; g=${2:-}
        case "$g" in
            "$want"*) printf 'ok   %s\n' "$(printf '%s' "$g" | cut -c1-78)" ;;
            *) printf 'FAIL want %s, got: %s\n' "$want" "$g"; rc=1 ;;
        esac
    }
    check PASS "$(lemonade_verdict 200 '{"data":[{"id":"m"}]}' 401)"
    check PASS "$(lemonade_verdict 200 '{"models":[]}' 403)"
    check FAIL "$(lemonade_verdict 200 '{"data":[{"id":"m"}]}' 200)"      # answers without the key
    check FAIL "$(lemonade_verdict 200 '{"data":[{"id":"m"}]}' 000)"      # empty body / no answer
    check FAIL "$(lemonade_verdict 000 '' 000)"                           # no server at all
    check FAIL "$(lemonade_verdict 200 '' 401)"                           # 200 with no model list
    check FAIL "$(lemonade_verdict 500 'error' 401)"                      # not a 200
    # cycle 20: the blank-mode row (the shipped default) - answers without a key, and says so when it does not
    check PASS "$(lemonade_verdict_blank 200 '{"data":[{"id":"m"}]}')"
    check PASS "$(lemonade_verdict_blank 200 '{"models":[]}')"
    check FAIL "$(lemonade_verdict_blank 401 'error')"                    # blank mode that demands a key
    check FAIL "$(lemonade_verdict_blank 200 '')"                         # 200 with no model list
    check FAIL "$(lemonade_verdict_blank 000 '')"                         # nothing answered
    # engine #945: whichever body this mode has is the model list the rows below read
    check '{"data":[{"id":"m"}]}' "$(models_list '' '{"data":[{"id":"m"}]}')"   # blank mode: the keyless body
    check '{"data":[{"id":"k"}]}' "$(models_list '{"data":[{"id":"k"}]}' '{"data":[]}')"  # keyed mode
    check '' "$(models_list '' '')"                                         # nothing answered
    # engine #911: the NPU follow-up row. The board's shape is the second case: a long turn that
    # finished, then a follow-up with no content at all and a TDR in the kernel log.
    check PASS "$(followup_verdict '{"choices":[{"finish_reason":"length"}]}' '{"choices":[{"message":{"content":"1BIT OK"},"finish_reason":"length"}]}' 0 0)"
    check FAIL "$(followup_verdict '{"choices":[{"finish_reason":"length"}]}' '{"error":{"message":"dx: position out of range"}}' 0 0)"
    check FAIL "$(followup_verdict '{"choices":[{"finish_reason":"length"}]}' '{"choices":[{"message":{"content":""},"finish_reason":"length"}]}' 0 0)"
    check FAIL "$(followup_verdict '{"choices":[{"finish_reason":"length"}]}' '{"choices":[{"message":{"content":"1BIT OK"},"finish_reason":"length"}]}' 0 1)"
    check FAIL "$(followup_verdict '{"error":{"message":"no answer"}}' '{"choices":[{"message":{"content":"1BIT OK"},"finish_reason":"length"}]}' 0 0)"
    check FAIL "$(followup_verdict '{"choices":[{"finish_reason":"length"}]}' '{"choices":[{"message":{"content":"1BIT"}}]}' 0 0)"
    check PASS "$(followup_verdict '{"choices":[{"finish_reason":"length"}]}' '{"choices":[{"message":{"content":"1BIT OK"},"finish_reason":"length"}]}' '' '')"
    # the RUNPATH rule (2026-11-08 incident): $ORIGIN only, and RPATH is always a fail.
    check PASS "$(payload_home_layout_verdict 0 0 '')"
    check FAIL "$(payload_home_layout_verdict 1 0 'bin/1bit|/home/x/wt/issue-709-build/1bit|file')"
    check FAIL "$(payload_home_layout_verdict 1 0 'bin/1bit|/home/x/wt/build/onebit|dangling')"
    check PASS "$(payload_home_layout_verdict 0 1 'root/spot|/home/spot|dir')"
    _hl_st=$(mktemp -d)
    mkdir -p "$_hl_st/bin" "$_hl_st/opt"
    ln -s /home/nonexistent-xyz/wt/build/1bit "$_hl_st/bin/leak"   # the class: absolute, target absent
    ln -s ../opt/1bit/1bit "$_hl_st/bin/rel"                       # relative: not this class
    ln -s /home/nonexistent-xyz "$_hl_st/opt/home"                 # a home directory target: named, not failed
    check "bin/leak|/home/nonexistent-xyz/wt/build/1bit|dangling" "$(payload_home_layout_bad "$_hl_st" | grep '^bin/leak|')"
    rm -rf "$_hl_st"
    check PASS "$(xhci_capture_verdict 1)"
    check FAIL "$(xhci_capture_verdict 0)"
    check FAIL "$(xhci_capture_verdict '')"
    check PASS "$(payload_runpath_verdict 0)"
    check FAIL "$(payload_runpath_verdict 3 '/opt/1bit-package/1bit/1bit runs /home/x/wt/release-rc/work/build')"
    check FAIL "$(payload_runpath_verdict 1)"
    check FAIL "$(payload_runpath_verdict '')"
    # engine#1543's other half: the sets are linked BESIDE the engine, so moving the engine without the link
    # registers no device on a green boot
    check PASS "$(npu_link_verdict 0 '')"
    check PASS "$(npu_link_verdict 1 /data/npu/kernels)"
    check FAIL "$(npu_link_verdict 1 '')"
    check FAIL "$(npu_link_verdict 1 /home/builder/wt/npupkg-build/npu/kernels)"
    check FAIL "$(npu_link_verdict '')"

    # engine #1352: the payload's modes. c31/c32 shipped 134 writable files; a payload packed after #1361 has none.
    check PASS "$(identity_verdict aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)"
    check FAIL "$(identity_verdict aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 7ba2110811111111111111111111111111111111111111111111111111111111 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)"       # the c26-in-a-c32-image shape: engine claim vs shipped launcher
    check FAIL "$(identity_verdict aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc)"       # libhsa claim vs the runtime the image carries
    check FAIL "$(identity_verdict 89a6c386 bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)"    # malformed claim is FAIL, not unknown
    check FAIL "$(identity_verdict aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa '' bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)"          # nothing to check the claim against
    check PASS "$(identity_verdict aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa none '')"           # `none` and nothing on the loader path: consistent
    check FAIL "$(identity_verdict aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa none bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)"        # `none` while the image loads a runtime: false statement
    # `check` matches by PREFIX, so a path helper is asserted by giving the expected path as <want>
    # 64 'a's generated POSIX-ly: `{1..60}` is a BASH brace expansion, and this file is #!/bin/sh - under dash it does not
    # expand, printf formats the literal, the claim arrives 5 characters long and this PASS case fails as "malformed". That is
    # what CI saw while a `bash release-check.sh --self-test` here passed: the self-test was run with a different shell than the
    # one ctest uses, so the run was not evidence about the artifact.
    a64=$(printf '%064s' '' | tr ' ' a)
    check PASS "$(identity_verdict "$a64" "$a64" not-stated '')"   # mini: no libhsa claim
    check FAIL "$(identity_verdict '' '' not-stated '')"                                                          # nothing checkable: still a FAIL
    check /opt/1bit-package/1bit/1bit "$(PAYLOAD_DIR= identity_launcher_path aaaa '')"      # v3: no git sha -> the payload launcher
    check /bin/1bit "$(MINI_LAUNCHER= identity_launcher_path aaaa bbbb)"                   # mini: both keys -> /bin/1bit
    check /x/1bit "$(MINI_LAUNCHER=/x/1bit identity_launcher_path aa bb)"                  # the seam is honoured
    check /y/1bit/1bit "$(PAYLOAD_DIR=/y identity_launcher_path aa '')"                    # PAYLOAD_DIR is honoured
    if [ -z "$(identity_launcher_path '' bbbb)" ]; then printf 'ok   no claim yields no path (the row SKIPs)\n'
    else printf 'FAIL no claim should yield no path\n'; rc=1; fi
    check FAIL "$(payload_name_verdict lemonade-onebit-rc-2026-11-05-c26 rc-2026-11-05-c26c)"       # c26 must NOT match c26c
    check FAIL "$(payload_name_verdict lemonade-onebit-rc-2026-11-05-c26b rc-2026-11-05-c26c)"      # nor c26b
    check FAIL "$(payload_name_verdict lemonade-onebit-rc-2026-11-08-c33x rc-2026-11-08-c33)"        # the token must END there, not continue
    check PASS "$(payload_name_verdict lemonade-onebit-rc-2026-11-08-c33.tar rc-2026-11-08-c33)"    # a non-alphanumeric after it is fine
    check PASS "$(payload_name_verdict lemonade-onebit-rc-2026-11-05-c26c rc-2026-11-05-c26c)"      # the real c26c pair still passes
    check PASS "$(payload_name_verdict lemonade-onebit-rc-2026-11-08-c33 rc-2026-11-08-c33)"        # the label CONTAINS the token
    check FAIL "$(payload_name_verdict lemonade-onebit-rc-2026-11-05-c26c rc-2026-11-08-c33)"        # the c26 incident: must red
    check SKIP "$(payload_name_verdict lemonade-onebit-rc-2026-11-08-c33 '')"                        # no README tag: not a pass by absence
    check FAIL "$(payload_name_verdict unknown rc-2026-11-08-c33)"                                   # PAYLOAD_NAME unset
    check FAIL "$(payload_name_verdict '' rc-2026-11-08-c33)"                                        # nothing stated
    check PASS "$(payload_identity_verdict 2026.47.0~1.52a2a6b3 2026.47.0~1.52a2a6b3)"
    check PASS "$(payload_identity_verdict 2026.47.0~1.52a2a6b3 2026.47.0~1752.52a2a6b3)"   # the clone-count case: must NOT red
    check FAIL "$(payload_identity_verdict 2026.47.0~1.52a2a6b3 2026.47.0~1.abcdef01)"      # a different sha IS a different pin
    check SKIP "$(payload_identity_verdict '' 2026.47.0~1.52a2a6b3)"
    check FAIL "$(payload_identity_verdict 2026.47.0~1.52a2a6b3 '')"
    check PASS "$(memlock_verdict unlimited lemond)"
    check FAIL "$(memlock_verdict 65536 lemond /etc/security/limits.d/99-amdnpu-memlock.conf)"   # a user's 99- winning over our 90-
    check FAIL "$(memlock_verdict 4294967296 1bit)"                                             # finite is not what the driver asks for
    check SKIP "$(memlock_verdict '' '')"                                                       # nothing running: say so, do not pass
    check FAIL "$(memlock_verdict '' lemond)"                                                   # running but unreadable: unknown is not a pass
    check FAIL "$(memlock_verdict 'lots' lemond)"                                               # nonsense is a fail, not a skip
    check PASS "$(payload_modes_verdict 0 0)"
    check FAIL "$(payload_modes_verdict 134 49)"        # c31/c32's real shape: 134 files AND 49 directories
    check FAIL "$(payload_modes_verdict 0 1)"           # the c33 first cut: no files, the payload ROOT writable
    check FAIL "$(payload_modes_verdict 1 0 '/opt/1bit-package/1bit/1bit')"
    check FAIL "$(payload_modes_verdict '' 0)"
    check FAIL "$(payload_modes_verdict 0 '')"
    check FAIL "$(payload_modes_verdict x 0)"
    # engine#1557: the payload's own identity - a claim that matches is a PASS, a stale one is a FAIL, and an image without
    # the field is a SKIP that says why rather than a pass by absence
    check PASS "$(payload_identity_verdict '2026.47.0~8.f44ff953' '2026.47.0~8.f44ff953')"
    check FAIL "$(payload_identity_verdict '2026.47.0~8.f44ff953' '2026.48.0~1.aaaaaaaa')"
    check SKIP "$(payload_identity_verdict '' '2026.47.0~8.f44ff953')"
    check FAIL "$(payload_identity_verdict '2026.47.0~8.f44ff953' '')"
    # engine #1474: the base layer's row - clean only when the count is 0 AND a keep-list was read
    check PASS "$(base_modes_verdict 0 11)"   # 11: the entry count in os/v3/puppy/base-modes-keep today
    check FAIL "$(base_modes_verdict 2166 10 'drwxrwxr-x usr/share')"   # c33's base layer, the shape this closes
    check FAIL "$(base_modes_verdict 1 10 'drwxrwxr-x usr')"
    check FAIL "$(base_modes_verdict '' 10)"                            # unreadable count
    check FAIL "$(base_modes_verdict 0 '')"                             # no keep-list to judge against
    check FAIL "$(base_modes_verdict 0 x)"                              # a non-numeric keep count
    # a ZERO keep count is not a verdict failure: the row SKIPs when the keep-list file is absent, and a present-but-empty
    # one is a repository error rather than a machine state, so it is covered at the call site (documented here so the
    # split is not mistaken for a gap)
    # c32: the HRX weight placement. The measured arms from #1339 are the first two; the failure shapes are a default
    # that already imported (2), an unreadable arm, and a zero.
    check PASS "$(hrx_placement_verdict 5692 3010)"        # 4B perplexity, #1339
    check PASS "$(hrx_placement_verdict 3925 1243)"        # 4B, docs/hrx.md
    check FAIL "$(hrx_placement_verdict 3010 5692)"        # default imported: the wrong way round
    check FAIL "$(hrx_placement_verdict 1000 1000)"        # no difference at all
    check FAIL "$(hrx_placement_verdict 0 3010)"           # a zero is not a measurement
    check FAIL "$(hrx_placement_verdict '' 3010)"          # unreadable default
    check FAIL "$(hrx_placement_verdict 5692 '')"          # unreadable opt-in
    check PASS "$(hrx_placement_verdict 2000 1000)"        # exactly the floor is enough
    check FAIL "$(hrx_placement_verdict 1140 1000)"        # just under it is not
    # c30 (owner decision 2026-11-05): the NPU's power state. The c29 take's shape is the second case - the array was
    # allowed to suspend between requests, and the prompt turn that followed it came back `code 614`.
    check PASS "$(npu_power_verdict on active)"
    check FAIL "$(npu_power_verdict auto suspended)"                     # the c29 take's state
    check FAIL "$(npu_power_verdict auto active)"                        # suspended as soon as the work stops
    check FAIL "$(npu_power_verdict on suspended)"                       # held awake but not running
    check FAIL "$(npu_power_verdict on '')"                              # status unreadable
    check SKIP "$(npu_power_verdict '' 'active')"                        # not an NPU boot
    check SKIP "$(npu_power_verdict '' '')"
    # engine #911 / c21 board: the streaming row. The board's own failure is the second case - the status was
    # 200, the body carried the error and the stream still closed with [DONE], so only the body can tell.
    check PASS "$(stream_verdict 'data: {"choices":[{"delta":{"content":"1BIT"}}]}
data: {"choices":[{"delta":{"content":" OK"}}]}
data: [DONE]' 0 0)"
    check FAIL "$(stream_verdict 'data: {"error":{"message":"npu runtime: invalid call (code 614): state 8, run 0 of 28"}}
data: [DONE]' 0 0)"
    check FAIL "$(stream_verdict 'data: {"choices":[{"delta":{"content":"1BI"}}]}
data: [DONE]' 0 1)"
    check FAIL "$(stream_verdict 'data: {"choices":[{"delta":{"content":"1BI"}}]}' 0 0)"      # no [DONE]
    check FAIL "$(stream_verdict 'data: [DONE]' 0 0)"                                              # no text at all
    check FAIL "$(stream_verdict '' 0 0)"
    check PASS "$(stream_verdict 'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}
data: [DONE]' 0 0)"
    check PASS "$(stream_verdict 'data: {"choices":[{"delta":{"content":"1BIT OK"}}]}
data: [DONE]' '' '')"
    # engine #1024 / c23 board: the busy row. The board's shape is the second case - the refusal is an `error`
    # frame inside a 200, so a row that looks at a status stays green on it.
    check PASS "$(queue_verdict 'data: {"choices":[{"delta":{"content":"long"}}]}
data: [DONE]' 'data: {"choices":[{"delta":{"content":""}}]}
data: {"choices":[{"delta":{"content":"1BIT OK"}}]}
data: [DONE]' 0 0)"
    check FAIL "$(queue_verdict 'data: [DONE]' 'data: {"error":{"message":"the NPU is busy with another request"}}
data: [DONE]' 0 0)"
    check FAIL "$(queue_verdict 'data: [DONE]' 'data: {"choices":[{"delta":{"content":""}}]}
data: [DONE]' 0 0)"                                                                                   # queued, then nothing
    check FAIL "$(queue_verdict 'data: [DONE]' 'data: {"choices":[{"delta":{"content":"OK"}}]}' 0 0)"   # no [DONE]
    check FAIL "$(queue_verdict 'data: [DONE]' '' 0 0)"                                                  # no body at all
    check FAIL "$(queue_verdict 'data: [DONE]' 'data: {"choices":[{"delta":{"content":"OK"}}]}
data: [DONE]' 0 1)"                                                                                    # a TDR while queueing
    check PASS "$(queue_verdict 'data: [DONE]' 'data: {"choices":[{"delta":{"reasoning_content":"OK"}}]}
data: [DONE]' '' '')"                                                                                  # dmesg unreadable
    # engine #1024, the non-streamed arm: it cannot be told it is queueing, so a 503 + Retry-After at its own bound is
    # the pass, and being *answered* while the device is busy is the failure (that would mean the bound did not apply).
    check PASS "$(plain_queue_verdict 'data: [DONE]' 503 '{"error":{"message":"the NPU is busy with another request","type":"device_busy"}}' 1 0 0)"
    check FAIL "$(plain_queue_verdict 'data: [DONE]' 200 '{"choices":[{"message":{"content":"OK"}}]}' '' 0 0)"
    check FAIL "$(plain_queue_verdict 'data: [DONE]' 503 '{"error":{"message":"the NPU is busy with another request","type":"device_busy"}}' '' 0 0)"   # no Retry-After
    check FAIL "$(plain_queue_verdict 'data: [DONE]' 503 '{"error":{"message":"model failed to load"}}' 1 0 0)"                                        # no device_busy
    check FAIL "$(plain_queue_verdict 'data: [DONE]' 000 '' '' 0 0)"                                                                                   # nothing answered
    check FAIL "$(plain_queue_verdict 'data: {"choices":[{"delta":{"content":"long"}}]}' 503 '{"type":"device_busy"}' 1 0 0)"                           # long chat cut
    check FAIL "$(plain_queue_verdict 'data: [DONE]' 503 '{"type":"device_busy"}' 1 0 1)"                                                               # a TDR
    check PASS "$(plain_queue_verdict 'data: [DONE]' 503 '{"type":"device_busy"}' 1 '' '')"                                                             # dmesg unreadable
    # the history that goes back into the request has to survive a JSON string
    jt=$(json_text 'a "quoted" \\ line
next')
    if printf '{"x":"%s"}' "$jt" | grep -q '\\"quoted\\"'; then printf 'ok   json_text escapes quotes and drops the newline\n'
    else printf 'FAIL json_text left the text unsafe for JSON: %s\n' "$jt"; rc=1; fi
    # engine #574: the sealed set's manifest, on sets built here (a manifest whose file list and
    # the set's files are not the same set of files is exactly what the packaging and this check
    # have to refuse)
    ST=$(mktemp -d)
    mkdir -p "$ST/set/layer1_1"
    printf 'ciphertext' > "$ST/set/layer1_1/layer1.elf"
    hs=$(sha256sum "$ST/set/layer1_1/layer1.elf" | cut -c1-64)
    printf '{\n  "format": 1,\n  "set": "set",\n  "files": [\n    {"path": "layer1_1/layer1.elf", "size": 10, "sha256": "%s"}\n  ]\n}\n' \
        "$hs" > "$ST/set/manifest.json"
    printf 'abababab\n' > "$ST/set/manifest.sig"
    st() {  # <want> <label> <set dir>
        _got=FAIL; seal_manifest_problems "$3" >/dev/null 2>&1 && _got=PASS
        if [ "$_got" = "$1" ]; then printf 'ok   %s -> %s\n' "$2" "$_got"
        else printf 'FAIL want %s, got %s: %s\n' "$1" "$_got" "$2"; rc=1; fi
    }
    st PASS "a sealed set whose manifest lists exactly its files" "$ST/set"
    printf 'more' > "$ST/set/layer1_1/extra.elf"
    st FAIL "a file the manifest does not list" "$ST/set"
    rm -f "$ST/set/layer1_1/extra.elf" "$ST/set/manifest.sig"
    st FAIL "a manifest with no signature" "$ST/set"
    printf 'abababab\n' > "$ST/set/manifest.sig"; printf 'tampered' > "$ST/set/layer1_1/layer1.elf"
    st FAIL "a ciphertext byte that changed after it was hashed" "$ST/set"
    rm -f "$ST/set/manifest.json"
    st FAIL "a sealed set with no manifest at all" "$ST/set"
    rm -rf "$ST"
    # docs/laya-first.md §7: the laya-auto row. The second case is the one a status-only row misses - the
    # request answered 200 on default_model, which looks like a healthy route until you read the decision.
    LV='{"x_lemonade_route":{"route_to":"Qwen3-Coder-30B-A3B-Instruct-1bit","matched_rule":"laya-code","default_used":false,"trace":[{"condition":"classifier:laya","label":"code","score":0.46}]}}'
    check PASS "$(laya_verdict "$LV" code Qwen3-Coder-30B-A3B-Instruct-1bit)"
    check PASS "$(laya_verdict "$LV" '' '')"
    check FAIL "$(laya_verdict '{"x_lemonade_route":{"route_to":"Qwen3-0.6B-1bit","matched_rule":"laya-code","default_used":true,"trace":[{"condition":"classifier:laya","score":0.46}]}}' code Qwen3-0.6B-1bit)"   # fell open
    check FAIL "$(laya_verdict '{"x_lemonade_route":{"route_to":"Qwen3-0.6B-1bit","matched_rule":"long-input","default_used":false,"trace":[]}}' code Qwen3-0.6B-1bit)"   # a deterministic rule decided
    check FAIL "$(laya_verdict "$LV" prose '')"                                        # the wrong class's rule matched
    check FAIL "$(laya_verdict "$LV" code Qwen3-0.6B-1bit)"                            # the wrong model
    check FAIL "$(laya_verdict '{"x_lemonade_route":{"route_to":"m","matched_rule":"laya-code","default_used":false,"trace":[]}}' code m)"   # no Laya score
    check 'FAIL no model is named for class not-a-class' "$(laya_verdict "$LV" not-a-class '')"   # an unmapped class must not skip the checks
    check FAIL "$(laya_verdict '' code m)"                                             # nothing answered
    # the row's class -> model mapping mirrors config/laya-first.json; assert it here so a table change is
    # a red self-test rather than a row that checks agreement with a mapping nobody updated
    map_check() {  # <class> <want model>
        _got=$(laya_class_model "$1")
        if [ "$_got" = "$2" ]; then printf 'ok   %s -> %s\n' "$1" "$_got"
        else printf 'FAIL %s -> %s (want %s)\n' "$1" "$_got" "$2"; rc=1; fi
    }
    map_check short Qwen3-0.6B-1bit
    map_check prose ZAYA1-8B-1bit
    map_check code Qwen3-Coder-30B-A3B-Instruct-1bit
    map_check long_doc Qwen3.8-27B-1bit
    map_check not-a-class ''
    # a body with the answer text in front of the decision still decides on the decision
    check PASS "$(laya_verdict '{"choices":[{"message":{"content":"Here you go"}}],"x_lemonade_route":{"route_to":"m","matched_rule":"laya-prose","default_used":false,"trace":[{"condition":"classifier:laya","score":0.77}]}}' prose m)"

    [ $rc = 0 ] && echo "release-check self-test PASS" || echo "release-check self-test FAIL"
    # ── kernel_cve_verdict: the CVE-2026-98432 fix standing of a kernel release ──────────────────────────
    # The advisory's own spelling, `X.Y-rcN` and `X.Y` with no patch component, is driven first: the first
    # version of the parse rejected those as unparsable while the code had a 7.4 branch waiting for them.
    check PASS "$(kernel_cve_verdict 7.4-rc1 absent)"          # the advisory's floor for this fix
    check PASS "$(kernel_cve_verdict 7.4.0-rc1-1bit absent)"   # the same kernel as `uname -r` writes it
    check PASS "$(kernel_cve_verdict 7.4 absent)"              # the release
    check FAIL "$(kernel_cve_verdict 7.4-rc0 absent)"          # before the advisory's rc1
    check PASS "$(kernel_cve_verdict 7.3 absent)"              # a release with no patch component
    check FAIL "$(kernel_cve_verdict 7.3-rc4 absent)"          # a pre-release AT the floor patch
    check FAIL "$(kernel_cve_verdict 7.3.0-rc4-1bit absent)"   # the same, in `uname -r` form
    check PASS "$(kernel_cve_verdict 7.3-rc4 yes)"             # source evidence outranks the floor
    check FAIL "$(kernel_cve_verdict 7.4-rc1 no)"              # ... and can convict above it
    check FAIL "$(kernel_cve_verdict 5.10.272-rc1 absent)"     # pre-release at a patch floor
    check PASS "$(kernel_cve_verdict 5.10.272 absent)"
    check FAIL "$(kernel_cve_verdict 6.6.158 absent)"
    check PASS "$(kernel_cve_verdict 6.6.159 absent)"
    check FAIL "$(kernel_cve_verdict 7.2.9 absent)"
    check PASS "$(kernel_cve_verdict 7.2.10 absent)"
    check FAIL "$(kernel_cve_verdict 4.19.0 absent)"           # a line the advisory does not list
    check FAIL "$(kernel_cve_verdict '' absent)"               # nothing may read as safe
    check FAIL "$(kernel_cve_verdict 7.x.0 absent)"
    # A self-test is not a check run: without this the EXIT trap above appended "RESULT: FAIL (this check died
    # before its footer)" to every green self-test, which reads exactly like the death the trap exists to report.
    RC_FOOTER=1
    exit $rc
fi


# Where this image keeps the Lemonade package. os/mini installs it at /opt/lemonade; the Puppy appliance
# (os/v3/puppy) embeds the same release package at /opt/1bit-package. Found, not configured, so one tool
# answers for both artifacts instead of drifting into two copies.
LEMDIR=
for _d in /opt/lemonade /opt/1bit-package; do
    if [ -d "$_d/bin" ]; then LEMDIR=$_d; break; fi
done
LEMDIR=${LEMDIR:-/opt/lemonade}

PORT=$(sed -n 's/.*"port"[^0-9]*\([0-9]\+\).*/\1/p' /data/lemonade/config/config.json 2>/dev/null | head -1)
PORT=${PORT:-13305}
# The key the running lemond actually holds, read from the environment of the process that listens on
# the Lemonade port (via the socket inode in /proc/net/tcp). Candidate 10 on the board (26 Oct): the conf
# held two active key lines for a while (a staged one glued to the previous line, and the one init then
# minted); lemond, which SOURCES the file, ran on the last one, and this check, taking the first, sent the
# other and read 401 as "the key is wrong". The file is only the fallback, and then the last assignment
# wins, as it does for the shell that started lemond.
lemond_key() {
    ino=$(awk -v p="$(printf ':%04X' "$PORT")" '$2 ~ p"$" && $4 == "0A" { print $10; exit }' /proc/net/tcp 2>/dev/null)
    [ -n "$ino" ] || return 1
    for d in /proc/[0-9]*; do
        if ls -l "$d/fd" 2>/dev/null | grep -q "socket:\[$ino\]"; then
            tr '\0' '\n' < "$d/environ" 2>/dev/null | sed -n 's/^LEMONADE_API_KEY=//p' | head -n 1
            return 0
        fi
    done
    return 1
}
KEY=$(lemond_key)
[ -n "$KEY" ] || KEY=$(sed -n 's/^LEMONADE_API_KEY=//p' /data/lemonade/lemond.conf 2>/dev/null | tail -n 1)
# Cycle 20, the owner's "blank key": blank mode is the SHIPPED state. Cycle 21: a request in blank mode carries no
# Authorization header at all (auth_get/auth_status/auth_post above), and on a keyed machine the same calls carry the
# real key (a wrong one is still a 401, which the auth row asserts).
API="http://127.0.0.1:$PORT"
CATALOG=$LEMDIR/bin/resources/server_models.json
PLAIN_TOTAL=$(find /data/npu/kernels -name 'layer1.elf' 2>/dev/null | wc -l)

say "1bit OS release check   $(date -u +%FT%TZ)"

# ── mode ─────────────────────────────────────────────────────────────────────────────────────────
i=0
while [ ! -e /etc/1bit/live ] && [ ! -e /data/.installed ] && [ "$i" -lt 10 ]; do sleep 1; i=$((i + 1)); done
if [ -e /data/.installed ] || [ -e /etc/1bit/installed ]; then
    MODE=INSTALLED
elif [ -e /etc/1bit/live ]; then
    MODE=LIVE
elif [ -e /data/lemonade ] && [ ! -e /data/.installed ]; then
    MODE=LIVE      # init.sh writes /etc/1bit/live late; the data partition is still the live one
else
    MODE=
fi
if [ -n "$MODE" ]; then pass mode "$MODE - $( [ "$MODE" = LIVE ] && echo 'RAM only, /data on the stick' || echo '/data on the internal disk')"
else fail mode "neither /etc/1bit/live nor /etc/1bit/installed: not a 1bit OS boot"; fi

# ── kernel + Secure Boot ────────────────────────────────────────────────────────────────────────
REL=$(uname -r)
if [ -d "/lib/modules/$REL" ]; then pass kernel "$REL with its modules"
else fail kernel "$REL but /lib/modules/$REL is missing (the image booted a kernel it does not carry)"; fi

if command -v mokutil >/dev/null 2>&1; then
    SB=$(mokutil --sb-state 2>&1 | head -1)
    case "$SB" in
        *enabled*) pass secureboot "$SB (the MOK-signed kernel is what booted)";;
        *disabled*|*"not supported"*) pass secureboot "$SB";;
        *) skip secureboot "mokutil answered: $SB";;
    esac
elif [ -e /sys/firmware/efi ]; then
    # mokutil is not shipped, but the firmware's own variable says it plainly: SecureBoot-<vendor guid>,
    # 5 bytes - attributes(4) then the value. Reading it needs no new binary, and the previous wording
    # ("no mokutil in the image") reported a SKIP on a machine that had Secure Boot ON and had just
    # booted our MOK-signed kernel, which is the pass the owner actually cares about.
    SBVAR=/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c
    if [ -r "$SBVAR" ]; then
        sbv=$(od -An -tu1 -j4 -N1 "$SBVAR" 2>/dev/null | tr -d ' \n')
        case "${sbv:-}" in
            1) pass secureboot "Secure Boot enabled; the MOK-signed kernel booted";;
            0) pass secureboot "Secure Boot disabled (the firmware reports it off)";;
            *) skip secureboot "Secure Boot state unreadable from $SBVAR";;
        esac
    else
        skip secureboot "no mokutil and no readable $SBVAR; firmware reports EFI but not the Secure Boot state"
    fi
else
    fail secureboot "no /sys/firmware/efi: booted in legacy mode, not the signed UEFI chain"
fi

# ── devices ─────────────────────────────────────────────────────────────────────────────────────
if [ -e /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1; then
    pass gpu "$(ls /dev/dri/renderD* 2>/dev/null | tr '\n' ' ')/dev/kfd present (amdgpu up)"
else
    skip gpu "no AMD GPU in this VM: /dev/kfd and /dev/dri/renderD* only appear on the machine"
fi
if ls /dev/accel/accel* >/dev/null 2>&1 && [ -e /lib/firmware/amdnpu/17f0_11 ]; then
    pass npu "$(ls /dev/accel/accel* 2>/dev/null | tr '\n' ' ')and the NPU firmware present"
else
    skip npu "no NPU in this VM: /dev/accel and /lib/firmware/amdnpu only appear on the machine"
fi
# The NPU's power state (owner decision 2026-11-05, c30's `code 614` fix - see npu_power_verdict above). The device is
# matched on its PCI id `17f0` - the same XDNA2 device the image's own udev rule matches (`pci 1022:17f0`) and the same
# one the firmware row above tests for - rather than on a hardcoded 0000:c6:00.1, so a different board's NPU answers the
# same row and a machine with no NPU says so in the image's own words (04onboard-npu-keepawake prints "no XDNA2 NPU,
# nothing to keep awake").
NPUDIR=
for ctl in /sys/bus/pci/devices/*/power/control; do
    [ -e "$ctl" ] || continue
    [ "$(cat "${ctl%/power/control}/device" 2>/dev/null)" = 0x17f0 ] && NPUDIR=${ctl%/power/control}
done
if [ -z "$NPUDIR" ]; then
    skip npu-power "no XDNA2 NPU (PCI device id 0x17f0) in this boot: there is no power state to read"
else
    # A device the udev rule just resumed can take a moment to report `active`, and that race is not a fault: wait for
    # it rather than red a boot for the driver's own workqueue. Ten seconds, then the row says what it sees.
    _i=0
    while :; do
        _c=$(cat "$NPUDIR/power/control" 2>/dev/null)
        _s=$(cat "$NPUDIR/power/runtime_status" 2>/dev/null)
        [ "$_c" = on ] && [ "$_s" = active ] && break
        _i=$((_i + 1))
        [ "$_i" -lt 10 ] || break
        sleep 1
    done
    VP=$(npu_power_verdict "$_c" "$_s")
    case "$VP" in
        PASS*) pass npu-power "the NPU is held awake ($(basename "$NPUDIR")): ${VP#PASS }" ;;
        SKIP*) skip npu-power "${VP#SKIP }" ;;
        *)     fail npu-power "$(basename "$NPUDIR") ${VP#FAIL }" ;;
    esac
fi

# ── the HRX weight placement, measured (engine #1339's mitigation - see hrx_placement_verdict) ────────────
# The model is NAMED in the row on purpose: it has to be UNTIED. Fork #233 (engine #1316) changes where a TIED
# embedding lands, so a tied model would move between the two arms for a second reason and prove nothing about this
# switch. The shipped demo 4B carries both token_embd.weight and output.weight and no tie_word_embeddings key.
# Seams for measurement off a stick: HRXP_SERVE (a command that stands in for `1bit serve`), HRXP_GTT (a command that
# prints the MiB of own GTT for the pids given), HRXP_MODEL (a model path), HRXP_WAIT (seconds to wait for /health).
hrxp_model() {  # the staged 4B GGUF, if this medium has one
    if [ -n "${HRXP_MODEL:-}" ]; then printf '%s' "$HRXP_MODEL"; return 0; fi
    for _g in /data/lemonade/huggingface/models--unsloth--Qwen3-4B-Instruct-2507-GGUF/snapshots/*/Qwen3-4B-Instruct-2507-Q4_K_M.gguf; do
        [ -f "$_g" ] && { printf '%s' "$_g"; return 0; }
    done
    return 1
}
hrxp_gtt() {  # <pid>... -> that process tree's own drm-total-gtt, in MiB
    if [ -n "${HRXP_GTT:-}" ]; then "$HRXP_GTT" "$@"; return 0; fi
    _k=0
    for _p in "$@"; do
        for _fd in /proc/"$_p"/fdinfo/*; do
            [ -r "$_fd" ] || continue
            _b=$(sed -n 's/^drm-total-gtt:[[:space:]]*\([0-9]*\).*/\1/p' "$_fd" 2>/dev/null | head -1)
            [ -n "$_b" ] && _k=$((_k + _b))
        done
    done
    [ "$_k" -gt 0 ] && printf '%s' $(( _k / 1024 / 1024 )) || printf '0'
}
# Which mechanism does the SHIPPED engine have? Fork #231 (engine #1343) added the import switch and made it opt-in;
# fork #234 (engine #1351, which also closed #1339) REMOVED it - that build reads the weights into their buffers and drops
# their page cache instead. On the second kind the env var is ignored, both arms take the same path and their GTT readings
# are identical, so a verdict would red a correct image. The row therefore asks the shipped code, and SKIPs with the reason
# when the switch is not in it: a gate that fails a correct build is its own defect. Measuring the read+DONTNEED mechanism
# needs its own row (#1351 measured page-cache residency 16.5 GiB -> nothing, peak RSS 16.4 -> 2.9 GiB, load 14.3 -> 18.6 s),
# and it must be built and measured on a pin that has it before it can gate one.
hrxp_has_switch() {  # <engine binary> -> 0 when the shipped libs carry GGML_HRX_WEIGHT_HOST_PLACEMENT
    case "${HRXP_SWITCH:-}" in 1) return 0 ;; 0) return 1 ;; esac
    _d=$(dirname "${1:-}" 2>/dev/null) _l=
    for _cand in "$_d/libggml-hrx.so" "$_d"/libggml-hrx.so.* "$_d/../lib/ggml/libggml-hrx.so" "$_d"/../lib/ggml/libggml-hrx.so.* "$_d"/hrx/llama/bin/libggml-hrx.so; do
        [ -f "$_cand" ] || continue
        if grep -q 'GGML_HRX_WEIGHT_HOST_PLACEMENT' "$_cand" 2>/dev/null; then _l=$_cand; break; fi
    done
    [ -n "$_l" ]
}

hrxp_arm() {  # <port> <optin 0|1> -> "<own GTT MiB>" on stdout; non-zero when the arm did not come up
    if [ -n "${HRXP_SERVE:-}" ]; then
        _g=$($HRXP_SERVE "$1" "$2" 2>/dev/null | tail -1)
        case "${_g:-}" in ''|*[!0-9]*) return 1 ;; esac
        printf '%s' "$_g"; return 0
    fi
    set -- "$1" "$2"
    _port=$1 _optin=$2
    if [ "$_optin" = 1 ]; then export GGML_HRX_WEIGHT_HOST_PLACEMENT=1; else unset GGML_HRX_WEIGHT_HOST_PLACEMENT; fi
    setsid "$ONEBIT_BIN" serve -m "$HRXP_MODEL_PATH" --device hrx --ctx-size 512 --host 127.0.0.1 --port "$_port" \
        >/tmp/1bit-hrxp-$_port.log 2>&1 &
    _pgid=$!
    _i=0; _w=${HRXP_WAIT:-120}
    while [ "$_i" -lt "$_w" ]; do
        if command -v curl >/dev/null 2>&1; then
            curl -sf -m 2 "http://127.0.0.1:$_port/health" >/dev/null 2>&1 && break
        else
            wget -q -O /dev/null --timeout=2 -t 1 "http://127.0.0.1:$_port/health" 2>/dev/null && break
        fi
        kill -0 "$_pgid" 2>/dev/null || break
        sleep 2; _i=$((_i + 2))
    done
    if ! kill -0 "$_pgid" 2>/dev/null; then
        kill -TERM -- -"$_pgid" 2>/dev/null; return 1
    fi
    # the child (llama-server) holds the device; sample the group, then stop it and let the GTT go
    _pids=$(pgrep -g "$_pgid" 2>/dev/null | tr '\n' ' ')
    _g=$(hrxp_gtt "$_pgid" $_pids)
    kill -TERM -- -"$_pgid" 2>/dev/null; sleep 2; kill -KILL -- -"$_pgid" 2>/dev/null
    case "${_g:-}" in ''|*[!0-9]*) return 1 ;; esac
    printf '%s' "$_g"
}
if [ ! -e "${HRXP_KFD:-/dev/kfd}" ]; then
    skip hrx-placement "no /dev/kfd: this machine has no AMD GPU, so there is no HRX placement to measure"
elif ! HRXP_MODEL_PATH=$(hrxp_model); then
    skip hrx-placement "no staged Qwen3-4B-Instruct-2507-Q4_K_M.gguf under /data/lemonade/huggingface: the row measures one named model twice and will not substitute another"
else
    ONEBIT_BIN=${ONEBIT_BIN:-}
    if [ -z "$ONEBIT_BIN" ]; then
        for _b in $(command -v 1bit 2>/dev/null) /opt/1bit-package/1bit/1bit /opt/1bit-engine/1bit /opt/1bit/1bit; do
            [ -x "$_b" ] && { ONEBIT_BIN=$_b; break; }
        done
    fi
    if [ -z "$ONEBIT_BIN" ] && [ -z "${HRXP_SERVE:-}" ]; then
        skip hrx-placement "no 1bit engine binary in the image, so the two arms cannot be started"
    elif ! hrxp_has_switch "$ONEBIT_BIN"; then
        skip hrx-placement "this build does not carry the weight-mapping import switch (fork #234 / engine #1351 removed it: weights are read into their buffers and their page cache is dropped), so both arms would take the same path and the two-arm GTT measurement cannot distinguish anything - #1339's current mechanism needs its own row, measured on a pin that has it"
    else
        export HRXP_MODEL_PATH
        DFLT=$(hrxp_arm 18301 0) || _drc=$?
        OPTIN=$(hrxp_arm 18302 1) || _orc=$?
        if [ -n "${_drc:-}" ] || [ -n "${_orc:-}" ]; then
            # an arm that never answered /health is not a measurement of the placement: name which one, and why, rather
            # than reporting "unreadable GTT" and sending the reader after the wrong fault
            VM="FAIL an arm did not come up (default${_drc:+ rc=$_drc} ${DFLT:-no-answer} / opt-in${_orc:+ rc=$_orc} ${OPTIN:-no-answer}); see /tmp/1bit-hrxp-1830{1,2}.log"
        else
            VM=$(hrx_placement_verdict "${DFLT:-}" "${OPTIN:-}")
        fi
        case "$VM" in
            PASS*) pass hrx-placement "$(basename "$HRXP_MODEL_PATH") (untied): ${VM#PASS }" ;;
            *)     fail hrx-placement "$(basename "$HRXP_MODEL_PATH"): ${VM#FAIL }" ;;
        esac
        unset HRXP_MODEL_PATH GGML_HRX_WEIGHT_HOST_PLACEMENT
    fi
fi

# Placed BEFORE the chat rows on purpose: lemond loads a model for those, and this row wants the device to itself -
# two arms that load while another serve holds the NPU/GPU would differ for the wrong reason.
# ── fans (EC fan curve; only a board we know has one we drive) ──────────────────────────────────
if [ ! -x /usr/sbin/1bit-board ]; then
    skip fans "no /usr/sbin/1bit-board in this image: the fans stay on the firmware curve"
elif ! /usr/sbin/1bit-board known; then
    skip fans "this board has no EC we drive: the firmware fan curve stays"
elif [ ! -d /sys/class/ec_su_axb35 ]; then
    skip fans "ec_su_axb35 did not load: the firmware fan curve stays"
else
    # level 4 (the boost level) for 5 s, then every CPU fan must report rpm > 0. A fan that reads 0 rpm
    # at 80 % is exactly the fault this row exists for - it is how the system fan failed on 2026-10-04.
    for f in fan1 fan2; do
        printf 'fixed\n' > "/sys/class/ec_su_axb35/$f/mode" 2>/dev/null
        printf '4\n' > "/sys/class/ec_su_axb35/$f/level" 2>/dev/null
    done
    sleep 5
    RPMS=; BAD=0
    for f in fan1 fan2; do
        r=$(cat "/sys/class/ec_su_axb35/$f/rpm" 2>/dev/null || echo 0)
        RPMS="$RPMS$f $r rpm, "
        [ "$r" -gt 0 ] 2>/dev/null || BAD=1
    done
    for f in fan1 fan2; do printf 'curve\n' > "/sys/class/ec_su_axb35/$f/mode" 2>/dev/null; done
    RPMS=$(printf '%s' "$RPMS" | sed 's/, $//')
    if [ "$BAD" = 0 ]; then pass fans "controlled ($RPMS at level 4)"
    else fail fans "the driver loaded but a fan reads 0 rpm at level 4 ($RPMS): the curve and the boost cannot cool this board"
    fi
fi

# ── Lemonade, with and without the key ──────────────────────────────────────────────────────────
if [ -z "$KEY" ]; then
    # Blank mode: the shipped state (cycle 20). Assert what THAT means instead of failing for a key nobody asked for.
    if ! lemonade_ready; then
        fail lemonade "lemond did not answer $API within ${READY_WAIT}s (see /tmp/lemond.log)"
    else
        BODY=$(got "$API/api/v1/models")
        WITHOUT=$(status "$API/api/v1/models")
        v=$(lemonade_verdict_blank "$WITHOUT" "$BODY")
        case "$v" in
            PASS*) pass lemonade "$(printf '%s' "$v" | sed 's/^PASS //')";;
            *) fail lemonade "$(printf '%s' "$v" | sed 's/^FAIL //')";;
        esac
    fi
elif ! lemonade_ready; then
    # Not the same row as a wrong key: nothing answered on the port at all within the wait.
    fail lemonade "lemond did not answer $API within ${READY_WAIT}s (see /tmp/lemond.log)"
else
    MODELS=$(auth_get "$API/api/v1/models")
    WITH=$(auth_status "$API/api/v1/models")
    BODY=$(got "$API/api/v1/models")
    WITHOUT=$(status "$API/api/v1/models")
    case "$(lemonade_verdict "$WITH" "$MODELS" "$WITHOUT")" in
        PASS*) pass lemonade "$(printf '%s' "$(lemonade_verdict "$WITH" "$MODELS" "$WITHOUT")" | sed 's/^PASS //')";;
        *) fail lemonade "$(printf '%s' "$(lemonade_verdict "$WITH" "$MODELS" "$WITHOUT")" | sed 's/^FAIL //')";;
    esac
fi

# The rows below read only $MODELS, so it is chosen here for BOTH modes (engine #945): the keyed probe's answer
# when there is a key, else the keyless body - which is the list in blank mode.
MODELS=$(models_list "${MODELS:-}" "$BODY")

# Lemonade registers the onebit entries under a "user." prefix: candidate 8's hardware boot logged
# "lemond DID register user.Qwen3-0.6B-Q4_K_M-1bit". The manifest holds the bare catalogue id, so a
# plain ""$1"" match found nothing and every entry counted as missing - which is how the row ended up
# claiming the machine had no GPU. Match either form. -F, because ids contain dots that would otherwise
# be regex wildcards.
have_model() {
    printf '%s' "${MODELS:-}" | grep -qF "\"$1\"" && return 0
    printf '%s' "${MODELS:-}" | grep -qF "\"user.$1\"" && return 0
    return 1
}
# Case-insensitive sibling, for comparing a FILE name against an id. The appliance registers a .gguf under
# the shipped catalogue id, which is lower-cased (05onboard-data: name[:-5].lower() + "-1bit"), while the file
# on the partition keeps the model's own spelling - so on the 2026-11-02 board boot
# /data/models/Qwen3-0.6B-Q4_K_M.gguf WAS registered and served as qwen3-0.6b-q4_k_m-1bit, and this row
# failed it with a case-sensitive grep. The row still asks the same question (does Lemonade list this file);
# it just no longer answers it by spelling.
have_model_ci() {
    printf '%s' "${MODELS:-}" | grep -qiF "$1" && return 0
    return 1
}
GGUF=$(ls /data/models/*.gguf 2>/dev/null | head -1)
GGUF_NAME=$(basename "${GGUF:-none}" .gguf)

# ── the onebit entries, against the manifest the build wrote from the shipped catalog ───────────
MAN=/etc/1bit/onebit-models.txt
if [ -z "${MODELS:-}" ]; then
    fail models "Lemonade did not answer, so the model list cannot be checked"
elif [ ! -f "$MAN" ] && ! ( [ -e /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1 ) && ! ls /dev/accel/accel* >/dev/null 2>&1; then
    CAT=$LEMDIR/bin/resources/server_models.json
    WANT_HRX=$(grep -oE '"recipe"[[:space:]]*:[[:space:]]*"onebit"' "$CAT" 2>/dev/null | wc -l)
    WANT_NPU=$(grep -oE '"onebit_backend"[[:space:]]*:[[:space:]]*"npu"' "$CAT" 2>/dev/null | wc -l)
    skip models "no GPU or NPU here, so Lemonade hides the onebit entries ($WANT_HRX HRX + $WANT_NPU NPU in the catalog; $MAN records the shipped subset when the build writes it)"
elif [ ! -f "$MAN" ]; then
    fail models "$MAN is missing from the image (the build writes it from server_models.json)"
else
    HRX_WANT=$(grep -c '^hrx ' "$MAN")
    NPU_WANT=$(grep -c '^npu ' "$MAN")
    HRX_GOT=0; NPU_GOT=0; MISSING=
    while read -r kind id; do
        [ -n "${id:-}" ] || continue
        if have_model "$id"; then
            [ "$kind" = hrx ] && HRX_GOT=$((HRX_GOT + 1)) || NPU_GOT=$((NPU_GOT + 1))
        else
            MISSING="$MISSING $id"
        fi
    done < "$MAN"
    # Decide from the HARDWARE, not from whether the count happened to be zero. On candidate 8's hardware
    # boot this row said "the machine has no AMD GPU or NPU ... Lemonade hides them here" on a box with a
    # Radeon 8060S: two bugs made that happen. First, $MODE is LIVE/INSTALLED while the manifest this build
    # writes says "mode=live" / "mode=installed", so the mismatch test was always true - a case difference,
    # invisible and permanent. Second, the skip was gated on HRX_GOT = 0, so any failure to match a single
    # id was reported as "this machine has no GPU" instead of as a count. Now: a machine with a device never
    # gets the no-hardware skip (it gets a PASS or a count), and the mode comparison is case-folded.
    man_mode=$(sed -n 's/^mode=//p' "$MAN" 2>/dev/null | head -1 | tr '[:lower:]' '[:upper:]')
    have_gpu=0
    [ -e /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1 && have_gpu=1
    have_npu=0
    ls /dev/accel/accel* >/dev/null 2>&1 && have_npu=1
    # "listed by Lemonade" means LISTED AND DOWNLOADED, not "known". On candidate 10's stick this row
    # reported "0/22 HRX and 0/4 NPU listed" and looked like a build defect, while every id was in the
    # catalogue the image ships: the stick simply carries one model, so Lemonade offered only that one.
    # The question this row can actually answer on any machine is whether the manifest and the shipped
    # catalogue agree; how many are downloaded is context, and chat-hrx is the row that proves serving.
    CAT=$LEMDIR/bin/resources/server_models.json
    HRX_REG=0; NPU_REG=0; NOTCAT=
    while read -r kind id; do
        [ -n "${id:-}" ] || continue
        if grep -qF "\"$id\"" "$CAT" 2>/dev/null; then
            [ "$kind" = hrx ] && HRX_REG=$((HRX_REG + 1)) || NPU_REG=$((NPU_REG + 1))
        else
            NOTCAT="$NOTCAT $id"
        fi
    done < "$MAN"
    SERVED="$HRX_GOT HRX + $NPU_GOT NPU downloaded"
    if [ -n "$NOTCAT" ]; then
        fail models "the manifest names ids the shipped catalogue ($CAT) does not contain:$NOTCAT"
    elif [ "$HRX_REG" = "$HRX_WANT" ] && [ "$NPU_REG" = "$NPU_WANT" ]; then
        pass models "$HRX_REG HRX and $NPU_REG NPU onebit entries are in the shipped catalogue ($SERVED here)"
    elif [ "$have_gpu" = 0 ] && [ "$have_npu" = 0 ]; then
        skip models "this machine has no AMD GPU or NPU, so Lemonade hides the onebit entries ($HRX_WANT HRX + $NPU_WANT NPU are registered in $MAN)"
    elif [ "${man_mode:-}" != "$MODE" ] && [ "$HRX_REG" = 0 ]; then
        skip models "a $MODE boot lists no onebit entries; the manifest records the ${man_mode:-<unknown>} set ($HRX_WANT HRX + $NPU_WANT NPU): checked in the registry by vmtest.sh"
    else
        fail models "$HRX_REG/$HRX_WANT HRX and $NPU_REG/$NPU_WANT NPU entries found in the shipped catalogue; not in it:$NOTCAT"
    fi
fi
# Three different situations used to share one message. On candidate 10's first hardware run the gguf row
# said "no /data/models/*.gguf on the data partition" while `ls /data/models/*.gguf` plainly found the
# file: the row never looked at the partition at all. It took this else branch because $MODELS was empty -
# Lemonade's list had come back 401 with a stale key - so a missing model list was reported as a missing
# file, and the person reading it went looking at the data partition instead of the key.
if [ "$GGUF_NAME" = none ]; then
    skip gguf "no /data/models/*.gguf on the data partition"
elif [ -z "${MODELS:-}" ]; then
    skip gguf "'$GGUF_NAME.gguf' is on the data partition, but Lemonade listed no models, so it cannot be checked (see the lemonade row)"
elif [ ! -e /dev/kfd ] && ! ls /dev/accel/accel* > /dev/null 2>&1; then
    # The file is on the partition and the registry names it, but Lemonade hides every onebit entry on a
    # machine with no AMD GPU and no NPU - so "not listed" here says nothing about the image. Measured in
    # QEMU (cycle 12): /api/v1/models carried no onebit id at all on a VM, while the same image on the
    # board lists the model. Same hardware test the gpu/npu/models rows use.
    skip gguf "'$GGUF_NAME.gguf' is on the data partition; this machine has no GPU or NPU, so Lemonade hides the onebit entries that would list it"
elif have_model_ci "$GGUF_NAME"; then
    # the local registry names a GGUF <name>-1bit (os/mini/init.sh and /etc/init.d/05onboard-data both do,
    # the appliance under the catalogue's lower-cased id), and llama.cpp lists the bare name; either answer
    # means Lemonade knows this file
    pass gguf "the stick's own model '$GGUF_NAME' is listed"
else
    fail gguf "/data/models/$GGUF_NAME.gguf is on the data partition but Lemonade does not list it"
fi

# ── one chat per backend that this machine can actually run ─────────────────────────────────────
chat() {   # chat <model id> [slow] -> sets ANSWER and TPS
    ANSWER=; TPS=
    R=$(auth_post "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: 1BIT OK\"}],\"max_tokens\":48,\"temperature\":0}" \
        "$API/api/v1/chat/completions" "${2:-}")
    case "$R" in
        *'"content"'*)
            ANSWER=$(printf '%s' "$R" | sed -n 's/.*"content":"\([^"]*\)".*/\1/p' | head -1)
            TPS=$(printf '%s' "$R" | sed -n 's/.*"predicted_per_second":\([0-9.]*\).*/\1/p' | head -1) ;;
        *) ANSWER="no answer: $(printf '%s' "${R:-}" | head -c 120)" ;;
    esac
}
pick() { printf '%s' "${MODELS:-}" | tr '{' '\n' | sed -n "s/.*\"id\":\"\($1\)\".*/\1/p" | head -1; }
# The ids Lemonade has already downloaded, one per line. The model object carries "downloaded": true|false,
# and the entries that are false would have to be pulled from the Hub before they can answer - so a chat row
# that picked one would either download 400 MB during a gate or fail offline on a machine with no network.
# /v1/models defaults to show_all=false, i.e. exactly this list; /api/v1/models carries the same records.
# Falls back to pick() when nothing matches (an older Lemonade without the flag reads as "none downloaded").
listed() {   # <ere over the id>
    printf '%s' "${MODELS:-}" | tr '{' '\n' |
        grep -E '"downloaded"[[:space:]]*:[[:space:]]*true' |
        sed -n "s/.*\"id\"[[:space:]]*:[[:space:]]*\"\($1\)\".*/\1/p"
}

VMODEL=$(pick '[^"]*')
if [ -n "$VMODEL" ]; then
    chat "$VMODEL"
    if [ -n "$TPS" ]; then pass chat "$VMODEL answered \"$ANSWER\" at ${TPS} tok/s"
    else fail chat "$VMODEL did not answer ($ANSWER)"; fi
else
    skip chat "Lemonade lists no model this machine can run"
fi
# ── the HRX runtime itself, before any chat row that depends on it ─────────────────────────────
# A chat-hrx failure is useless on its own: it says an entry did not answer, not whether the runtime
# below it registered a device at all. Candidate 5 shipped an HRX runtime from before the pin that
# carries HRX_ALLOW_NO_SVM, so it registered no HRX0 while every other row stayed green - the engine
# sets HSA_USE_SVM=0 on a CWSR-on kernel and needs that gate to still see the device. This row asks
# llama-server directly, which is the smallest question with a yes/no answer.
# The same device test the gpu row uses: /dev/kfd alone is not enough. In the QEMU gate /dev/kfd is
# present but /dev/dri/renderD* is not, so a guard on /dev/kfd alone runs this row on a machine with no
# usable GPU, gets an honest "no device", and FAILS a candidate that is fine - a false FAIL in the row
# written to prevent false results. Require both, exactly as `gpu` does.
if [ -e /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1; then
    LSRV=${ONEBIT_HRX_LLAMA_SERVER:-/opt/1bit-llama/llama-server}
    if [ ! -x "$LSRV" ]; then
        fail hrx-device "no llama-server at $LSRV"
    else
        # The runtime resolves libhsa through IREE_HAL_AMDGPU_LIBHSA_PATH (iree/.../util/libhsa.c) -
        # that is the only libhsa knob in libggml-hrx/libhrx, and init.sh exports it from
        # /etc/1bit/libhsa. Set the same variable here so this row tests the runtime the engine
        # actually loads, not whatever the loader would find first. It accepts a directory or an exact
        # library path; /etc/1bit/libhsa holds the exact path, which is passed through unchanged.
        if [ -z "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ] && [ -r /etc/1bit/libhsa ]; then
            HRXLIB=$(cat /etc/1bit/libhsa 2>/dev/null)
            [ -n "$HRXLIB" ] && export IREE_HAL_AMDGPU_LIBHSA_PATH="$HRXLIB"
        fi
        # Cycle 14, measured on the cycle-13 board: this row printed "no HRX0 listed ... (Available devices: (none))",
        # which reads as a GPU fault. The GPU was fine. The appliance names the runtime in /etc/1bit/libhsa.env as
        # ONEBIT_HRX_LIBHSA - a build-time CMake name that no code reads from the environment - and ships that runtime
        # ALONE, with five NEEDED libraries missing, so the dlopen fails whoever names it. So: take the path from either
        # file and either key, then ask the loader whether the file can be loaded BEFORE asking llama-server for devices,
        # and give the two failures different words.
        if [ -z "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ] && [ -r /etc/1bit/libhsa.env ]; then
            HRXLIB=$(sed -n 's/^IREE_HAL_AMDGPU_LIBHSA_PATH=//p;s/^ONEBIT_HRX_LIBHSA=//p' /etc/1bit/libhsa.env 2>/dev/null | head -1)
            [ -n "$HRXLIB" ] && export IREE_HAL_AMDGPU_LIBHSA_PATH="$HRXLIB"
        fi
        # A runtime that LOADS is not a runtime that initialises: libhsa can be present and still lack a symbol the
        # backend needs (`hsa_amd_queue_create`, measured on a distro ROCr), and ldd cannot see that - it resolves NEEDED
        # libraries, never symbols. So the row is told the reason by the backend's own log line, and repeating it is the
        # difference between "no HRX0 listed" (which reads as a GPU fault) and "the HRX runtime did not initialise:
        # NOT_FOUND: symbol 'hsa_amd_queue_create' not found" (engine#1766).
        hrx_init_reason() {
            printf '%s\n' "$1" | grep -m1 'hrx_gpu_initialize(0) failed:' | sed 's/.*failed: *//'
        }
        HRXUNLOADABLE=
        if [ -n "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ]; then
            # ldd where it exists (it lists EVERY missing library); the loader itself where it does not - and the loader
            # refuses a library with missing dependencies without printing any "=> not found" line at all:
            #   <path>: error while loading shared libraries: librocm_sysdeps_elf.so.1: cannot open shared object file
            # so both shapes are read here. (Measured on the cycle-13 runtime: the loader form is the only one it gives.)
            if command -v ldd > /dev/null 2>&1; then HRXLOAD=$(ldd "$IREE_HAL_AMDGPU_LIBHSA_PATH" 2>&1)
            else HRXLOAD=$(/lib64/ld-linux-x86-64.so.2 --list "$IREE_HAL_AMDGPU_LIBHSA_PATH" 2>&1); fi
            HRXMISS=$(printf '%s\n' "$HRXLOAD" | sed -n 's/^[[:space:]]*\([^[:space:]]*\)[[:space:]]*=>[[:space:]]*not found.*/\1/p; s/.*error while loading shared libraries: \([^:]*\):.*/\1/p' | sort -u | tr '\n' ' ')
            if [ -n "$HRXMISS" ]; then
                fail hrx-device "the HRX runtime $IREE_HAL_AMDGPU_LIBHSA_PATH cannot be dlopened: it needs ${HRXMISS}- the image must ship them flat beside it (os/mini/mkimage.sh does), so this is a wiring defect, not a GPU one"
                HRXUNLOADABLE=1
            fi
        fi
        if [ -n "$HRXUNLOADABLE" ]; then
            :   # already reported above, with the cause; do not ask llama-server and blame the device
        else
            # --list-devices prints the devices ggml can see; an HRX backend that registered appears as
            # "HRX0: <name>". Capture both streams: a runtime that refuses prints its INCOMPATIBLE line
            # on stderr and that text is the diagnostic.
            LD=$(timeout 120 "$LSRV" --list-devices 2>&1)
            # A device that registered but says INCOMPATIBLE still prints "HRX0: INCOMPATIBLE (...)", so it
            # must be rejected BEFORE the match is accepted - otherwise this row passes on the exact failure
            # it exists to catch (candidate 5 printed HRX0: INCOMPATIBLE and had no usable device).
            HRXDEV=$(printf '%s\n' "$LD" | sed -n 's/^[[:space:]]*\(HRX[0-9][0-9]*\): *\(.*\)$/\1: \2/p' | grep -vi 'INCOMPATIBLE' | head -1)
            if [ -n "$HRXDEV" ]; then
                pass hrx-device "$HRXDEV"
            elif printf '%s' "$LD" | grep -qi "INCOMPATIBLE\|only systems with SVM"; then
                fail hrx-device "no HRX0 listed: $(printf '%s\n' "$LD" | grep -i 'INCOMPATIBLE\|only systems with SVM' | head -1)"
            else
                HRXREASON=$(hrx_init_reason "$LD")
                if [ -n "$HRXREASON" ]; then
                    fail hrx-device "the HRX runtime did not initialise: $HRXREASON (it loaded, so this is not a missing library; a symbol or config the runtime needs is absent)"
                else
                    fail hrx-device "no HRX0 listed by $LSRV (${LD:+$(printf '%s' "$LD" | tr '\n' ' ' | cut -c1-120)})"
                fi
            fi
        fi
    fi
else
    skip hrx-device "no /dev/kfd or no /dev/dri/renderD*: this machine has no AMD GPU, so no HRX device can register"
fi

# A downloaded entry first, and never an NPU one for the HRX row: the ids say which backend they are for
# ("-NPU-1bit"), and a machine can have both downloaded at once. Without this, the first "-1bit" id in the
# catalogue order was picked and the HRX row could be answered by the NPU model - a green row that proves
# nothing about the GPU path the row exists for.
HRX=$(listed '[^"]*-1bit' | grep -vi 'npu' | head -1)
[ -n "$HRX" ] || HRX=$(pick '[^"]*-1bit')
if [ -e /dev/kfd ] && [ -n "$HRX" ]; then
    # slow: the first HRX call compiles Loom kernels (see got_slow)
    chat "$HRX" slow
    if [ -n "$TPS" ]; then pass chat-hrx "$HRX (onebit, HRX) answered \"$ANSWER\" at ${TPS} tok/s"
    else fail chat-hrx "$HRX (onebit, HRX) did not answer ($ANSWER)"; fi
elif [ ! -e /dev/kfd ]; then
    skip chat-hrx "no /dev/kfd: this machine has no AMD GPU, so no HRX entry can run"
else
    skip chat-hrx "no onebit HRX entry is listed by Lemonade, so there is nothing to ask"
fi
NPU=$(listed '[^"]*-1bit' | grep -i 'npu' | head -1)
[ -n "$NPU" ] || NPU=$(pick '[^"]*-NPU-1bit')
if ls /dev/accel/accel* >/dev/null 2>&1 && [ -n "$NPU" ]; then
    npu_note chat-npu
    chat "$NPU"
    if [ -n "$TPS" ]; then pass chat-npu "$NPU (onebit, NPU) answered \"$ANSWER\" at ${TPS} tok/s"
    else fail chat-npu "$NPU (onebit, NPU) did not answer ($ANSWER)"; fi
elif ! ls /dev/accel/accel* >/dev/null 2>&1; then
    skip chat-npu "no /dev/accel/accel*: this machine has no NPU, so no NPU entry can run"
else
    skip chat-npu "no onebit NPU entry is listed by Lemonade, so there is nothing to ask"
fi

# ── NPU chat, twice: a 512-token turn, then a short prompt like the page's (engine #911) ────────
# The cycle-17 board demo lost the second turn of a chat - the page itself printed "done: 0 chunks in
# 5.9 s", Lemonade logged "Backend did not send [DONE] marker", and the kernel logged one amdxdna TDR -
# while this check's NPU row, one short question on a freshly loaded model, stayed green all evening.
# A fault that needs a long context first cannot appear in that row.
#
# The page is not a multi-turn client: index.html posts `messages: [{role: user, content: text}]`,
# the latest message only, so the demo's second turn was a SHORT PROMPT alone after a long one, on
# the same device state. That is arm A here. Arm B sends the stricter shape - the long turn's answer
# replayed as history - which the page never does, so the row covers both. dmesg's TDR line is
# counted between the arms, so a TDR during any of the three requests fails the row.
#
# The prompt is ~70 words with the answer space left open and the first turn is allowed the page's
# 512 tokens, so turn 2 starts from the context the demo had. Each request keeps got_slow's 120 s cap:
# 512 tokens is ~8 s on the 0.6B entry, and the same row runs on the 4B entry at 17 tok/s (30 s).
tdr_count() {  # the kernel's TDR lines so far; empty when the kernel log cannot be read
    _d=$(dmesg 2>/dev/null) || return 0
    [ -n "$_d" ] && printf '%s\n' "$_d" | grep -c 'aie2_legacy_tdr_detect'
    return 0
}
if ls /dev/accel/accel* >/dev/null 2>&1 && [ -n "$NPU" ]; then
    npu_note chat-npu-followup
    TDR0=$(tdr_count)
    Q_LONG='/no_think Explain in detail how tides work: the moon and the sun, the shape of ocean basins, why some coasts have two high tides a day and others one, and what an amphidromic point is'
    _post() {  # <json body> -> the reply
        auth_post "$1" "$API/api/v1/chat/completions" slow
    }
    R1=$(_post "{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":$(json_text "$Q_LONG" | sed 's/.*/"&"/')}],\"max_tokens\":512,\"temperature\":0}")
    A1=$(json_str "$R1" content)
    N1=$(json_str "$R1" completion_tokens)
    TDR1=$(tdr_count)
    # Arm A is the page's own shape: index.html posts `messages: [{role: user, content: text}]`
    # - the latest message only, no history - so the demo's second turn was a SHORT PROMPT after a
    # long one, on the same device state. Arm B is the stricter shape the web page does not send:
    # the long turn's own answer replayed as history.
    RA=$(_post "{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: 1BIT OK\"}],\"max_tokens\":48,\"temperature\":0}")
    TDRA=$(tdr_count)
    RB=$(_post "{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":$(json_text "$Q_LONG" | sed 's/.*/"&"/')},{\"role\":\"assistant\",\"content\":\"$(json_text "$A1")\"},{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: 1BIT OK\"}],\"max_tokens\":48,\"temperature\":0}")
    TDRB=$(tdr_count)
    VA=$(followup_verdict "$R1" "$RA" "$TDR1" "$TDRA")
    VB=$(followup_verdict "$R1" "$RB" "$TDRA" "$TDRB")
    _tdrs() { if [ -n "$TDR0" ] && [ -n "$TDRB" ]; then
                  printf 'TDRs %s -> %s -> %s -> %s' "$TDR0" "$TDR1" "$TDRA" "$TDRB"
              else printf 'dmesg not readable, so the TDR count is unknown'; fi; }
    if [ -n "$TDR0" ] && [ "$TDR1" -gt "$TDR0" ]; then
        fail chat-npu-followup "$NPU: a TDR landed during the 512-token turn ($(_tdrs))"
    elif [ "${VA%% *}" != PASS ] || [ "${VB%% *}" != PASS ]; then
        fail chat-npu-followup "$NPU: short follow-up -> $VA | with history -> $VB ($(_tdrs))"
    else
        pass chat-npu-followup "$NPU answered a 512-token turn (${N1:-?} tokens), then \"$(json_str "$RA" content | cut -c1-24)\" alone (as the page sends it) and again with that turn as history ($(_tdrs))"
    fi
elif ! ls /dev/accel/accel* >/dev/null 2>&1; then
    skip chat-npu-followup "no /dev/accel/accel*: this machine has no NPU, so no NPU entry can run"
else
    skip chat-npu-followup "no onebit NPU entry is listed by Lemonade, so there is nothing to ask"
fi

# ── the first request, decided by Laya (docs/laya-first.md §7) ───────────────────────────────────
# A cold Lemonade loads no model; the first request must go through Laya's router, which classifies it
# and picks one of OUR models. The row sends one prompt per class through the Laya-first router
# (`1bit laya-first` emits that collection's policy) and reads the router's own decision back. What
# makes it a row rather than a chat test: the decision has to say Laya decided - the class's own rule,
# no fall-through to default_model, and Laya's score in the trace. A request that lands on the right
# model by accident (a deterministic rule, or the default) fails laya_verdict, because the whole point
# of the appliance's first request is who chose.
#
# The class -> model pairs below mirror config/laya-first.json. If the table changes, this row fails
# loudly instead of quietly checking yesterday's promise; the engine's `1bit laya-first --check`
# validates the table itself.
# The router's name comes from the engine itself: `1bit laya-first` prints the collection Lemonade
# registers (user.Laya-First in the shipped table). Hardcoding it here would break the row the day the
# table's name changes - and a name is not a detail, the server only registers `user.*` definitions.
LAYA_ROUTER=user.Laya-First
LAYA_BIN=
for _b in $(command -v 1bit 2>/dev/null) /opt/1bit-engine/1bit /opt/1bit/1bit /opt/1bit-package/bin/1bit; do
    [ -x "$_b" ] && { LAYA_BIN=$_b; break; }
done
if [ -n "$LAYA_BIN" ]; then
    _n=$("$LAYA_BIN" laya-first 2>/dev/null | sed -n 's/.*"model_name": *"\([^"]*\)".*/\1/p' | head -1)
    [ -n "$_n" ] && LAYA_ROUTER=$_n
fi
if ! printf '%s' "${MODELS:-}" | grep -q "\"$LAYA_ROUTER\""; then
    skip laya-auto "no $LAYA_ROUTER router is registered (it needs a Laya checkpoint - scripts/fetch-laya.sh - and the desugared policy registered: 1bit laya-first --policy FILE)"
else
    LAYA_FAILS=0
    LAYA_SEEN=
    # auth_post_watch, not auth_post: the latter spells its body as wget's `--post-data`, so on an image
    # with curl but no wget it returns nothing and every post row reads as "no answer" (measured while
    # exercising this row off the stick). This helper handles both fetchers and hands back the body.
    LAYA_BODY=/tmp/1bit-laya-body.$$
    : > "$LAYA_BODY"
    laya_probe() {  # <class> <prompt>
        _cls=$1
        _want=$(laya_class_model "$_cls")
        _body="{\"model\":\"$LAYA_ROUTER\",\"messages\":[{\"role\":\"user\",\"content\":$(json_text "$2" | sed 's/.*/"&"/')}],\"max_tokens\":16,\"temperature\":0,\"route_trace\":true}"
        _st=$(auth_post_watch "$_body" "$API/api/v1/chat/completions" "$LAYA_BODY")
        case "${_st%% *}" in
            200) ;;
            *)   printf '      %s: the router did not answer (status %s)\n' "$_cls" "${_st%% *}"
                 LAYA_FAILS=$((LAYA_FAILS + 1)); return ;;
        esac
        _v=$(laya_verdict "$(cat "$LAYA_BODY" 2>/dev/null)" "$_cls" "$_want")
        case "$_v" in
            PASS) LAYA_SEEN="$LAYA_SEEN ${_cls}->${_want}";;
            *)    printf '      %s: %s\n' "$_cls" "$_v"; LAYA_FAILS=$((LAYA_FAILS + 1));;
        esac
    }
    # one prompt per class, each short enough to answer on a cold 0.6B and unmistakable to Laya
    laya_probe short "Say hello in five words."
    laya_probe prose "Summarise what tides are and why they happen, in two sentences."
    laya_probe code "Write a Python function that reverses a linked list."
    if [ "$LAYA_FAILS" = 0 ]; then
        pass laya-auto "Laya decided all three:${LAYA_SEEN}"
    else
        fail laya-auto "$LAYA_FAILS of 3 classes were not decided by Laya (see the lines above)"
    fi
fi

# ── NPU chat the way a person uses it: after a reload, streaming, thinking on (engine #911, c21 board) ──────
# The board's green check proved nothing about the shape the release onebit page actually uses: the page
# loads the model on demand (after this check's own rows evicted it), leaves thinking ON and streams the
# reply. On the c21 board exactly that failed twice, "npu runtime: invalid call (code 614): state 8, run 0 of
# 28", inside two 200s, while every row above stayed green - and the same boot logged one amdxdna TDR. So a
# 200 is not evidence here: the body is read (stream_verdict) and the kernel log is counted.
if ls /dev/accel/accel* >/dev/null 2>&1 && [ -n "$NPU" ]; then
    npu_note chat-npu-stream
    TDRS0=$(tdr_count)
    _stream() {  # <json body> -> the whole SSE body
        got_slow --header='Content-Type: application/json' --header="Authorization: Bearer $KEY" \
            --post-data="$1" "$API/api/v1/chat/completions"
    }
    # Evict first: that is what makes the next request auto-load the model, and an auto-load is when Lemonade's
    # ctx auto-tune runs (the board's reload resolved 32768 where the rows above load at 4096).
    got --header='Content-Type: application/json' --post-data='{}' "$API/api/v1/unload" >/dev/null 2>&1
    SS1=$(_stream "{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":\"Give me the history of New Brunswick in several paragraphs.\"}],\"max_tokens\":300,\"temperature\":0.7,\"stream\":true}")
    TDRS1=$(tdr_count)
    VSS1=$(stream_verdict "$SS1" "$TDRS0" "$TDRS1")
    SS2=$(_stream "{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: 1BIT OK\"}],\"max_tokens\":48,\"temperature\":0,\"stream\":true}")
    TDRS2=$(tdr_count)
    VSS2=$(stream_verdict "$SS2" "$TDRS1" "$TDRS2")
    _tdrs_s() { if [ -n "$TDRS0" ] && [ -n "$TDRS2" ]; then printf 'TDRs %s -> %s -> %s' "$TDRS0" "$TDRS1" "$TDRS2"
                else printf 'dmesg not readable, so the TDR count is unknown'; fi; }
    if [ "${VSS1%% *}" != PASS ] || [ "${VSS2%% *}" != PASS ]; then
        fail chat-npu-stream "$NPU: reload + thinking on -> $VSS1 | follow-up -> $VSS2 ($(_tdrs_s))"
    else
        pass chat-npu-stream "$NPU after an evict and a reload, streaming: $VSS1; then $VSS2 ($(_tdrs_s))"
    fi
elif ! ls /dev/accel/accel* >/dev/null 2>&1; then
    skip chat-npu-stream "no /dev/accel/accel*: this machine has no NPU, so no NPU entry can run"
else
    skip chat-npu-stream "no onebit NPU entry is listed by Lemonade, so there is nothing to ask"
fi

# ── NPU length ladder: the position count a real chat reaches (cycle 23; c22 board, 4 Nov 2026) ──────────────
# The c22 board made the wedge deterministic. Fresh backend each time, temperature 0, /no_think, a long-essay
# prompt: 400 max_tokens (429 positions) answered, 480 (509) answered, and 600 - the first step that crosses 512 -
# produced `npu runtime: invalid call (code 614): state 8` plus one amdxdna TDR and a power-cycle, and both board
# chat failures that day (12:56, 13:13 AST) sat ~7 s into a thinking-on reply, at that same boundary. The image
# now ships `options amdxdna tdr_timeout_ms=15000`, so the same ladder has to answer at 1000 with no new TDR line.
# The model is unloaded before every step: a warm context would let one step hide the next.
if ls /dev/accel/accel* >/dev/null 2>&1 && [ -n "$NPU" ]; then
    npu_note chat-npu-ladder
    TDRL0=$(tdr_count)
    TDRL1=$TDRL0
    LAD_OK=
    LAD_BAD=
    for n in 400 600 1000; do
        got --header='Content-Type: application/json' --post-data='{}' "$API/api/v1/unload" > /dev/null 2>&1
        RL=$(auth_post "{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Write a long essay about the history of cartography, and about what map-making changed in how people understood the world.\"}],\"max_tokens\":$n,\"temperature\":0}" "$API/api/v1/chat/completions" slow)
        AL=$(json_str "$RL" content)
        CL=$(json_str "$RL" completion_tokens)
        PL=$(json_str "$RL" prompt_tokens)
        TDRL1=$(tdr_count)
        if [ -z "$AL" ]; then
            LAD_BAD="$LAD_BAD ${n}->no-answer"
        elif [ -n "$TDRL1" ] && [ -n "$TDRL0" ] && [ "$TDRL1" -gt "$TDRL0" ]; then
            LAD_BAD="$LAD_BAD ${n}->TDR(${TDRL0}->${TDRL1})"
        else
            LAD_OK="$LAD_OK $n ok (${CL:-?} tokens${PL:+, ${PL} prompt})"
        fi
        [ -z "$LAD_BAD" ] || break
    done
    _tdrs_l() { if [ -n "$TDRL0" ] && [ -n "$TDRL1" ]; then printf 'TDRs %s -> %s' "$TDRL0" "$TDRL1"
                else printf 'dmesg not readable, so the TDR count is unknown'; fi; }
    if [ -n "$LAD_BAD" ]; then
        fail chat-npu-ladder "$NPU ladder 400/600/1000:$LAD_BAD; clean steps:$LAD_OK ($(_tdrs_l))"
    else
        pass chat-npu-ladder "$NPU answered every rung of 400/600/1000 max_tokens with a fresh backend, no TDR:$LAD_OK ($(_tdrs_l))"
    fi
elif ! ls /dev/accel/accel* >/dev/null 2>&1; then
    skip chat-npu-ladder "no /dev/accel/accel*: this machine has no NPU, so no NPU entry can run"
else
    skip chat-npu-ladder "no onebit NPU entry is listed by Lemonade, so there is nothing to ask"
fi

# ── NPU chat while the device is busy: two concurrent chats, both complete (engine#1024; c23 board, 4 Nov 2026) ──
# The c23 board refused the owner's chat with `the NPU is busy with another request` because THIS check's own
# 1000-token ladder row held the device for 14 s: the serve waited --queue-ms (5 s) and gave up. On a single-user
# appliance the second request has to queue, so a streamed 16-token chat started 1 s into a streamed 1000-token one
# must be ANSWERED, and the long one must stay whole. The refusal arrives inside a 200 on the stream path, so the
# bodies are read (queue_verdict), not the statuses.
if ls /dev/accel/accel* >/dev/null 2>&1 && [ -n "$NPU" ]; then
    TDRQ0=$(tdr_count)
    QDIR=$(mktemp -d /tmp/npu-queue.XXXXXX)
    QLONG="{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Write a long essay about the history of cartography, and about what map-making changed in how people understood the world.\"}],\"max_tokens\":1000,\"temperature\":0,\"stream\":true}"
    QSHORT="{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: 1BIT OK\"}],\"max_tokens\":16,\"temperature\":0,\"stream\":true}"
    ( auth_post "$QLONG" "$API/api/v1/chat/completions" slow > "$QDIR/long.sse" 2>&1 ) &
    QPID=$!
    sleep 1
    QSHORT_BODY=$(auth_post "$QSHORT" "$API/api/v1/chat/completions" slow)
    wait "$QPID" 2>/dev/null
    QLONG_BODY=$(cat "$QDIR/long.sse" 2>/dev/null)
    TDRQ1=$(tdr_count)
    # Name the arm that failed: the long chat is the one in the background, so a refusal or a cut there would otherwise
    # be reported under the short one's name and cost a round trip on the board.
    case "$QLONG_BODY" in
        *'data: '[[[]DONE]*) VQ=$(queue_verdict "$QLONG_BODY" "$QSHORT_BODY" "$TDRQ0" "$TDRQ1") ;;
        *) VQ="FAIL the long (1000-token) chat, which owns the device, came back without data: [DONE]: $(printf '%.120s' "${QLONG_BODY:-no body}")" ;;
    esac
    rm -rf "$QDIR"
    case "$VQ" in
        PASS*) pass chat-npu-queue "$NPU: $VQ" ;;
        *)     fail chat-npu-queue "$NPU: $VQ" ;;
    esac
elif ! ls /dev/accel/accel* >/dev/null 2>&1; then
    skip chat-npu-queue "no /dev/accel/accel*: this machine has no NPU, so no NPU entry can run"
else
    skip chat-npu-queue "no onebit NPU entry is listed by Lemonade, so there is nothing to ask"
fi

# ── NPU chat from a NON-STREAMED client while the device is busy (engine#1024; measured 4 Nov 2026) ────────────
# A non-streamed request has no way to say "queueing": no chunk, no comment, no header - so a long wait would be the
# #911 shape again (nothing on the wire while the client's own no-data timer runs out). The serve therefore keeps a
# short bound for it (`--queue-ms-plain`, 5000 ms) and answers `503 + Retry-After: 1` with `type: device_busy`, which a
# client can act on; Lemonade's agentic tool loop is exactly such a call site. This arm exists because
# `chat-npu-queue`'s two chats both stream, so it cannot see this path at all.
if ls /dev/accel/accel* >/dev/null 2>&1 && [ -n "$NPU" ]; then
    TDRP0=$(tdr_count)
    PDIR=$(mktemp -d /tmp/npu-queue-plain.XXXXXX)
    PLONG="{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Write a long essay about the history of cartography, and about what map-making changed in how people understood the world.\"}],\"max_tokens\":1000,\"temperature\":0,\"stream\":true}"
    PSHORT="{\"model\":\"$NPU\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: 1BIT OK\"}],\"max_tokens\":16,\"temperature\":0}"
    ( auth_post "$PLONG" "$API/api/v1/chat/completions" slow > "$PDIR/long.sse" 2>&1 ) &
    # NOT `PPID=$!`: PPID is a readonly bash builtin, so this line was a fatal error in the image's own shell (it
    # printed "PPID: readonly variable" and exited 1, with no RESULT line - measured on the c27 board, and the row
    # before this one had already passed). busybox ash does not have PPID readonly, which is why a shell that never
    # ships the check could not catch it.
    QPLPID=$!
    sleep 1
    PW=$(auth_post_watch "$PSHORT" "$API/api/v1/chat/completions" "$PDIR/short.json")
    wait "$QPLPID" 2>/dev/null
    PLONG_BODY=$(cat "$PDIR/long.sse" 2>/dev/null)
    PSHORT_BODY=$(cat "$PDIR/short.json" 2>/dev/null)
    TDRP1=$(tdr_count)
    VP=$(plain_queue_verdict "$PLONG_BODY" "${PW%% *}" "$PSHORT_BODY" "${PW#* }" "$TDRP0" "$TDRP1")
    rm -rf "$PDIR"
    case "$VP" in
        PASS*) pass chat-npu-queue-plain "$NPU: $VP" ;;
        *)     fail chat-npu-queue-plain "$NPU: $VP" ;;
    esac
elif ! ls /dev/accel/accel* >/dev/null 2>&1; then
    skip chat-npu-queue-plain "no /dev/accel/accel*: this machine has no NPU, so no NPU entry can run"
else
    skip chat-npu-queue-plain "no onebit NPU entry is listed by Lemonade, so there is nothing to ask"
fi

# ── lemonade bench, one chat-short row on the default model ─────────────────────────────────────
BENCH=$LEMDIR/bin/lemonade
if [ ! -x "$BENCH" ]; then
    skip bench "no $BENCH in the image"
elif [ -z "$VMODEL" ]; then
    skip bench "no model to bench"
else
    BLF=/tmp/bench.json
    # `lemonade bench` prints one human row per run: "Run 1/1... TTFT=30.6ms PP=881.6t/s TPS=120.2".
    # That line is the evidence; --json/--output is only a fallback because its shape moves between
    # Lemonade versions (and it wrote nothing to stdout in the 2026-10-23 candidate image).
    # Blank mode hands the CLI NO key at all: the variable is unset rather than set to a placeholder, so the
    # command runs exactly as a person's would on a keyless stick.
    unset LEMONADE_API_KEY
    [ -n "$KEY" ] && export LEMONADE_API_KEY="$KEY"
    BL=$(timeout 900 "$BENCH" bench "$VMODEL" --scenarios chat-short --runs 1 --warmup 0 \
            --json --output "$BLF" 2>/tmp/bench.err) || BL=
    BTT=$(printf '%s\n' "$BL" | sed -n 's/.*TTFT=\([0-9.]*\)ms.*/\1/p' | head -1)
    BPP=$(printf '%s\n' "$BL" | sed -n 's/.*PP=\([0-9.]*\)t\/s.*/\1/p' | head -1)
    BTG=$(printf '%s\n' "$BL" | sed -n 's/.* TPS=\([0-9.]*\).*/\1/p' | head -1)
    [ -n "$BTG" ] || BTG=$(sed -n 's/.*"tg_tps":[[:space:]]*\([0-9.]*\).*/\1/p' "$BLF" 2>/dev/null | head -1)
    [ -n "$BPP" ] || BPP=$(sed -n 's/.*"pp_tps":[[:space:]]*\([0-9.]*\).*/\1/p' "$BLF" 2>/dev/null | head -1)
    if [ -n "$BTG" ]; then pass bench "lemonade bench chat-short on $VMODEL: TTFT ${BTT:-?} ms, prefill ${BPP:-?} tok/s, decode ${BTG} tok/s"
    else
        # Say what bench actually printed, so the row diagnoses itself instead of needing a second run.
        say "      bench stderr: $(head -c 200 /tmp/bench.err 2>/dev/null | tr '\n' ' ')"
        say "      bench stdout: $(head -c 200 "$BLF" 2>/dev/null | tr '\n' ' ')"
        fail bench "lemonade bench produced no chat-short row for $VMODEL (its own output is printed above, /tmp/bench.json)"
    fi
fi

# ── sealed NPU sets: the method never ships in the clear, and every file is in its set's manifest ────────
SETS=$(find /data/npu/kernels -name layout.json 2>/dev/null | wc -l)
if [ "$SETS" = 0 ]; then
    fail sealed "no NPU kernel set on the data partition (find /data/npu/kernels -name layout.json)"
else
    BAD=0
    for f in $(find /data/npu/kernels -type f 2>/dev/null); do
        case "${f##*/}" in layout.json|manifest.json|manifest.sig) continue ;; esac
        [ "$(head -c 8 "$f" 2>/dev/null)" = 1BPKCRYP ] && continue
        BAD=$((BAD + 1)); say "      not sealed: $f"
    done
    [ "$(find /data/npu/kernels -name '*.mlir' -o -name 'layer1.work' 2>/dev/null | wc -l)" = 0 ] || { BAD=$((BAD + 1)); say "      plaintext build tree on the data partition"; }
    # engine #574: sealing is confidentiality, the manifest is authenticity. Every set has to carry a
    # manifest listing every file it ships with the sha256 of its ciphertext, and a signature over it.
    for d in $(find /data/npu/kernels -name layout.json 2>/dev/null | sed 's|/layout.json$||'); do
        P=$(seal_manifest_problems "$d") && continue
        BAD=$((BAD + 1))
        printf '%s\n' "$P" | sed "s|^|      ${d#/data/npu/kernels/}: |" | while IFS= read -r l; do say "$l"; done
    done
    if [ "$BAD" = 0 ]; then pass sealed "$SETS sealed sets, $PLAIN_TOTAL kernels, none in the clear, every file in its set's manifest"
    else fail sealed "$BAD problem(s) under /data/npu/kernels: not sealed, or not in the set's manifest"; fi
fi

# ── the signatures over those manifests (engine #574): the release public key is the only thing needed ──
# The loader - the private NPU runtime - holds the same key and verifies the signature before it opens
# a kernel. This row is the same judgement on the stick, where it can be made at all: it needs an
# interpreter (the image carries busybox, not python) and the release public key. Where either is
# missing it says so, rather than passing a row it did not check.
SEALPUB=${ONEBIT_SEAL_PUB:-/etc/1bit/seal-pub.txt}
SEALTOOL=
for t in "$(dirname "$0")/../../tools/seal_manifest.py" "$(dirname "$0")/../lib/1bit/seal_manifest.py" \
         /usr/lib/1bit/seal_manifest.py /opt/1bit/tools/seal_manifest.py; do
    [ -f "$t" ] && { SEALTOOL=$t; break; }
done
if [ "$SETS" = 0 ]; then
    skip sealed-sig "no NPU kernel set on the data partition, so there is no manifest to verify"
elif ! command -v python3 >/dev/null 2>&1; then
    skip sealed-sig "no python3 in this image, so the signatures can only be verified by the loader (the private NPU runtime holds the release public key)"
elif [ -z "$SEALTOOL" ]; then
    skip sealed-sig "no verifier (tools/seal_manifest.py) in this image; the loader verifies the signatures here"
elif [ ! -f "$SEALPUB" ]; then
    skip sealed-sig "no release public key at $SEALPUB: install the key the loader was built with to check the signatures here"
else
    SIGBAD=0; SIGS=0
    for d in $(find /data/npu/kernels -name layout.json 2>/dev/null | sed 's|/layout.json$||'); do
        SIGS=$((SIGS + 1))
        _o=$(python3 "$SEALTOOL" verify "$d" --pub "$SEALPUB" 2>&1) || {
            SIGBAD=$((SIGBAD + 1)); say "      ${d#/data/npu/kernels/}: $(printf '%s\n' "$_o" | head -1)"
        }
    done
    if [ "$SIGBAD" = 0 ]; then pass sealed-sig "$SIGS manifest.sig verified with the release public key ($SEALPUB)"
    else fail sealed-sig "$SIGBAD of $SIGS set manifests are not signed by the release public key"; fi
fi

# ── licences and image identity ────────────────────────────────────────────────────────────────
LIC=; NOT=; LOOKED=
for d in /usr/share/licenses/1bit-os "$LEMDIR/bin/resources/../../.." "$LEMDIR" "$LEMDIR/1bit" "$LEMDIR/bin" /; do
    LOOKED="$LOOKED $d"
    [ -f "$d/LICENSE-BINARY" ] && LIC=$d/LICENSE-BINARY
    [ -f "$d/NOTICE-BINARY" ] && NOT=$d/NOTICE-BINARY
done
if [ -n "$LIC" ] && [ -n "$NOT" ]; then pass licences "$LIC and $NOT present"
elif [ -n "$LIC" ]; then fail licences "$LIC present but no NOTICE-BINARY beside it (looked in:$LOOKED)"
elif [ -n "$LIC" ]; then fail licences "$LIC present but no NOTICE-BINARY beside it"
else fail licences "no LICENSE-BINARY/NOTICE-BINARY in the image (looked in:$LOOKED)"; fi

ID=${RELEASE_IDENTITY:-/etc/1bit/release-identity}
if [ -f "$ID" ]; then
    _id_name=$(sed -n 's/^payload=//p' "$ID" 2>/dev/null | head -1)
    _id_built=$(sed -n 's/^built=//p' "$ID" 2>/dev/null | head -1)
    # BOTH separators: os/v3/puppy/container-build.sh writes `engine_sha256=<sha>` (key=value) and os/mini/mkimage.sh
    # writes `engine_sha256: <sha>` (key: value) beside its `engine:` git sha (engine#1541). Reading only `=` made a mini
    # image's claim arrive empty, which sent it down the SKIP branch for ever - found by a fixture built the way the
    # writer writes it, not the way the reader expected it.
    _id_engine=$(sed -n 's/^engine_sha256 *[=:] *//p' "$ID" 2>/dev/null | head -1 | tr -d '[:space:]')
    # Two schemas write this file and only one of them is checkable against a payload (engine#1481): os/v3/puppy/
    # container-build.sh writes `engine_sha256=<sha256 of the launcher>` (key=value), while os/mini/mkimage.sh writes
    # `engine: <git short sha>` (key: value). Without this branch the mini schema reaches identity_verdict with an empty
    # claim and reds as "malformed, not merely old" - a false red on every mini image. A git short sha is not a file
    # hash, so the honest answer there is a SKIP that says why, not a FAIL and not a pass by absence.
    _id_gitsha=$(sed -n 's/^engine: *//p' "$ID" 2>/dev/null | head -1)
    _id_hsasha=$(sed -n 's/^libhsa_sha256 *[=:] *//p' "$ID" 2>/dev/null | head -1 | tr -d '[:space:]')
    # the mini schema has no libhsa field: say "not stated" rather than letting the strict verdict call it malformed
    [ -z "$_id_hsasha" ] && [ -n "${_id_gitsha:-}" ] && _id_hsasha=not-stated
    _id_launcher=$(identity_launcher_path "${_id_engine:-}" "${_id_gitsha:-}")
    _id_act_engine=
    # sha256sum follows the symlink, so a mini image's /bin/1bit is hashed as the image copy it points at - which is the
    # file the writer hashed ($root$bin). The resolved path goes into the message, because "/bin/1bit" and the file it
    # names are different facts on a build host.
    [ -n "$_id_launcher" ] && [ -f "$_id_launcher" ] && _id_act_engine=$(sha256sum "$_id_launcher" 2>/dev/null | cut -d' ' -f1)
    _id_where=${_id_launcher:+$(readlink -f "$_id_launcher" 2>/dev/null || printf '%s' "$_id_launcher")}
    _id_hsadir=$(sed -n 's/^\(export \)\{0,1\}LD_LIBRARY_PATH=//p' "${LIBHSA_ENV:-/etc/1bit/libhsa.env}" 2>/dev/null | sed 's/\$.*//' | cut -d: -f1)
    _id_act_hsasha=; [ -n "$_id_hsadir" ] && [ -f "${_id_hsadir%/}/libhsa-runtime64.so.1" ] && _id_act_hsasha=$(sha256sum "${_id_hsadir%/}/libhsa-runtime64.so.1" 2>/dev/null | cut -d' ' -f1)
    _idinfo="payload=${_id_name:-?} built=${_id_built:-?}${_id_where:+ launcher=${_id_where}}"
    if [ -z "$_id_engine" ] && [ -n "$_id_gitsha" ]; then
        skip identity "this image's identity is the mini schema: it records engine: $_id_gitsha, a git short sha rather than the launcher's sha256, so the payload it carries cannot be checked from it (engine#1481)"
    else
        _idv=$(identity_verdict "$_id_engine" "$_id_act_engine" "$_id_hsasha" "$_id_act_hsasha")
        case "$_idv" in
            PASS*) pass identity "${_idv#PASS } (${_idinfo})" ;;
            *)     fail identity "${_idv#FAIL } (${_idinfo})" ;;
        esac
    fi
    # engine#1557: the label is printed, the payload's own version is CHECKED. `payload=` stays the human label (the tarball
    # stem, which no in-image fact equals) and payload_lemond= is the claim the launcher on this machine can settle.
    _pl_claim=$(sed -n 's/^payload_lemond=//p' "$ID" 2>/dev/null | head -1)
    _pl_actual=
    if [ -x "${PAYLOAD_DIR:-/opt/1bit-package}/lemond-onebit" ]; then
        _pl_actual=$("${PAYLOAD_DIR:-/opt/1bit-package}/lemond-onebit" --version 2> /dev/null | sed -n 's/^lemond version //p' | head -1)
    fi
    PLV=$(payload_identity_verdict "$_pl_claim" "$_pl_actual")
    case "$PLV" in
        PASS*) pass payload-identity "${PLV#PASS } (label: ${_id_name:-?})" ;;
        SKIP*) skip payload-identity "${PLV#SKIP }" ;;
        *)     fail payload-identity "${PLV#FAIL } (label: ${_id_name:-?})" ;;
    esac
    if [ -f /data/SHA256SUMS ]; then
        if (cd /data && sha256sum -c SHA256SUMS >/dev/null 2>&1); then pass sha256sums "/data/SHA256SUMS verifies"
        else fail sha256sums "/data/SHA256SUMS does not verify (cd /data && sha256sum -c SHA256SUMS)"; fi
    else
        skip sha256sums "no /data/SHA256SUMS on the data partition; the image hash lives in the release table beside the .img.zst"
    fi
else
    skip identity "no /etc/1bit/release-identity in this image build; the tag and the image sha256 are in the release table beside the .img.zst"
fi

# ── the payload's modes (engine #1352) ─────────────────────────────────────────────────────────────
# The board-side half of #1361: the packer now normalises and asserts at pack time, and this proves it on the shipped tree. It
# also prints the two binaries' modes, so the identity of a boot says 0755 for `1bit` and `bin/lemond` rather than leaving a reader
# to trust that a mode fix happened. A SKIP (not a failure) when the payload is not where an appliance keeps it, so the row can be
# run off a stick.
PDIR=${PAYLOAD_DIR:-/opt/1bit-package}
# A clean mode count over an EMPTY payload proves nothing - and a directory that exists but holds nothing is exactly what an
# interrupted extraction leaves behind (measured 2026-11-08: an `unsquashfs -d` with the wrong path argument produced 0 entries
# and this row printed PASS over it). So the row checks its subject before it checks the modes: a real payload carries hundreds
# of entries (c33: 263).
# Absence first, then vacuity: the SKIP below is the row's documented behaviour on a machine where the payload is not
# unpacked (a stick, a dev box), and the floor must not pre-empt it. A missing payload is a claim about THIS system; a
# payload that exists and holds nothing is a claim about the ARTIFACT - and only the second is a FAIL.
if [ ! -d "$PDIR" ]; then
    skip payload-modes "no $PDIR on this system: the payload is not unpacked where an appliance keeps it, so its modes cannot be read"
else
    _pm_entries=$(find "$PDIR" -mindepth 1 2>/dev/null | wc -l)
    if [ "${_pm_entries:-0}" -lt 100 ] 2>/dev/null; then
        fail payload-modes "the payload at $PDIR holds only ${_pm_entries:-0} entries - a payload has hundreds (c33: 263), so this is an interrupted extraction, not a clean payload"
    else
        _pm_f=$(find "$PDIR" -type f \( -perm -g+w -o -perm -o+w \) 2>/dev/null | wc -l)
        _pm_d=$(find "$PDIR" -mindepth 0 -type d \( -perm -g+w -o -perm -o+w \) 2>/dev/null | wc -l)
        # the sample is relative to the payload: an absolute path per file makes a line nobody can scroll, and the
        # prefix is identical for every one of them. It prints the mode too, because "writable" is not actionable
        # and `drwxrwxr-x opt/1bit-package` is.
        _pm_s=$(find "$PDIR" -mindepth 0 \( -type f -o -type d \) \( -perm -g+w -o -perm -o+w \) 2>/dev/null | head -3 |
            while IFS= read -r _p; do _r=${_p#"$PDIR"}; _r=${_r#/}; printf '%s %s ' "$(stat -c %A "$_p" 2>/dev/null)" "${_r:-.}"; done)
        PM=$(payload_modes_verdict "${_pm_f:-}" "${_pm_d:-}" "${_pm_s:-}")
        _pm_bits=
        # engine#2422: printing the bits is not asserting them. A 0644 `1bit/1bit` passes the writability rule and then makes
        # hrx-placement SKIP (`[ -x ]` cannot find a runnable engine), so the release check was green over a payload whose
        # engine cannot run - measured on a real 258-entry payload: `PASS payload-modes (modes: 644 1bit/1bit 644 bin/lemond)`
        # beside `SKIP hrx-placement`. Absence stays out of this row's claim: a binary that is not here is hrx-device's or
        # identity's business, and a payload mid-extraction is already caught by the floor above.
        for _b in "$PDIR/1bit/1bit" "$PDIR/bin/lemond" "$PDIR/lemond-onebit"; do
            [ -f "$_b" ] && _pm_bits="$_pm_bits $(stat -c '%a %n' "$_b" 2>/dev/null | sed "s|$PDIR/||")"
        done
        # The executability requirement is its own function so `tests/release_check_payload_modes_test.sh` can extract and test
        # it without running the whole check: the first version of that fixture ran `release-check.sh --check` three times and cost
        # **392 s** in CI (18 s locally became 130 s per arm in a container), which is not a price three `[ -x ]` lines may charge.
        _pm_noexec=$(payload_modes_noexec "$PDIR")
        if [ -n "$_pm_noexec" ]; then
            fail payload-modes "in the payload but not executable:${_pm_noexec} - a 0644 engine or lemond refuses to run, and hrx-placement then SKIPs instead of failing (modes:$_pm_bits)"
        else
            case "$PM" in
                PASS*) pass payload-modes "${PM#PASS }${_pm_bits:+ (modes:$_pm_bits)}" ;;
                *)     fail payload-modes "${PM#FAIL }${_pm_bits:+ (modes:$_pm_bits)}" ;;
            esac
        fi
    fi
fi

# ── the BASE layer's modes (engine#1474) ────────────────────────────────────────────────────────────────────────
# The scan is the root filesystem only, and the keep-list is the ONE source of truth for what may stay writable: the prune
# list is BUILT from it rather than copied beside it (the first version hardcoded the same eleven entries by hand, so the
# next cycle to edit the keep-list - the file whose whole purpose is to be edited - would have had no prune for its new
# entry, and the row would have failed on a path the keep-list explicitly allows). The four paths pruned for a DIFFERENT
# reason are named separately, because "not part of the OS image" is not "allowed to be writable".
#
# The scan root is overridable for the same reason PAYLOAD_DIR is above: a row that can only be judged by booting an image
# cannot be tested. `BASE_SCAN_ROOT=<tree>` points it at any tree of the same shape.
BKEEP=${BASE_KEEP:-/etc/1bit/base-modes-keep}
if [ ! -f "$BKEEP" ]; then
    skip base-modes "no $BKEEP on this system: this image predates the base-layer row (engine #1474), so there is no keep-list to judge against"
else
    _bm_keep=$(awk '{ sub(/#.*/, ""); gsub(/[[:space:]]/, ""); if ($0 != "") print }' "$BKEEP")
    _bm_n=$(printf '%s\n' "$_bm_keep" | grep -c . || true)
    BROOT=${BASE_SCAN_ROOT:-/}
    BROOT=${BROOT%/}
    # the rows that are not the OS image at all (a mounted data partition, the initramfs, removable media)
    _bm_excl='data initrd mnt media'
    _bm_prune=
    for _p in $_bm_excl; do
        _bm_prune="$_bm_prune -path $BROOT/$_p -prune -o"
    done
    # and everything the keep-list names, from the keep-list itself
    for _p in $_bm_keep; do
        _bm_prune="$_bm_prune -path $BROOT/${_p%/} -prune -o"
    done
    # COUNT the whole set, then sample from it: `find … | head -40` before the count capped the number at 40, so the one
    # sentence a reader uses to judge severity said "40 path(s)" for a base layer with 2,176 of them (review on #1544)
    _bm_list=${TMPDIR:-/tmp}/.1bit-base-modes.$$
    # shellcheck disable=SC2086  # the prune expression is built as separate words on purpose
    find "${BROOT:-/}" $_bm_prune \( -type f -o -type d \) \( -perm -g+w -o -perm -o+w \) -print 2> /dev/null > "$_bm_list" || true
    _bm_n_paths=$(wc -l < "$_bm_list" | tr -d ' ')
    _bm_s=$(head -3 "$_bm_list" | while IFS= read -r _p; do [ -n "$_p" ] && printf '%s %s ' "$(stat -c %A "$_p" 2>/dev/null)" "$_p"; done)
    rm -f "$_bm_list"
    BM=$(base_modes_verdict "$_bm_n_paths" "$_bm_n" "$_bm_s")
    case "$BM" in
        PASS*) pass base-modes "${BM#PASS } (keep-list: $(printf '%s' "$_bm_keep" | tr '\n' ' '))" ;;
        *)     fail base-modes "${BM#FAIL }" ;;
    esac
fi

# ── the payload's RUNPATH (orchestrator's rule, 2026-11-08) ─────────────────────────────────────────────────────
# An installed package must carry `$ORIGIN` ONLY. The incident: an installed engine whose RUNPATH pointed into
# ~/wt/release-rc/work/build had its libraries swapped by a rebuild there, and two running services crash-looped
# (coder-llm on :8090 154x, 1bit-canonical on :8095 755x). The appliance ships no readelf/objdump, so the dynamic section is read
# directly: python3 is in the image, with readelf as the host fallback. DT_RPATH is a fail on its own - it is deprecated and cannot
# be overridden by LD_LIBRARY_PATH, which is exactly the property that makes a stale path stick.
payload_runpath_bad() {   # <payload dir> -> "relpath|TAG|value" per offending entry
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$1" <<'PYSCAN' 2>/dev/null
import os, struct, sys
root = sys.argv[1]
def dyn(path):
    try:
        with open(path, 'rb') as f:
            d = f.read()
    except OSError:
        return None
    if len(d) < 64 or d[:4] != b'\x7fELF':
        return None
    is64 = d[4] == 2
    try:
        if is64:
            shoff, = struct.unpack_from('<Q', d, 0x28)
            shent, = struct.unpack_from('<H', d, 0x3A)
            shnum, = struct.unpack_from('<H', d, 0x3C)
        else:
            shoff, = struct.unpack_from('<I', d, 0x20)
            shent, = struct.unpack_from('<H', d, 0x2E)
            shnum, = struct.unpack_from('<H', d, 0x30)
        secs = []
        for i in range(shnum):
            o = shoff + i * shent
            if is64:
                nm, typ, fl, addr, off, size, link, info, al, ent = struct.unpack_from('<IIQQQQIIQQ', d, o)
            else:
                nm, typ, fl, addr, off, size, link, info, al, ent = struct.unpack_from('<IIIIIIIIII', d, o)
            secs.append((typ, off, size, link))
    except struct.error:
        return None
    out = []
    for (typ, off, size, link) in secs:
        if typ != 6 or link >= len(secs):
            continue
        stroff = secs[link][1]
        step = 16 if is64 else 8
        fmt = '<qQ' if is64 else '<iI'
        for j in range(0, size - step + 1, step):
            tag, val = struct.unpack_from(fmt, d, off + j)
            if tag == 0:
                break
            if tag in (15, 29):
                end = d.find(b'\0', stroff + val)
                out.append((tag, d[stroff + val:end].decode('utf-8', 'replace')))
    return out
for dirpath, _, files in os.walk(root):
    for name in files:
        p = os.path.join(dirpath, name)
        for (tag, value) in dyn(p) or []:
            if tag == 15:
                print('%s|RPATH|%s' % (os.path.relpath(p, root), value))
            elif value not in ('$ORIGIN', '$ORIGIN/'):
                print('%s|RUNPATH|%s' % (os.path.relpath(p, root), value))
PYSCAN
    elif command -v readelf >/dev/null 2>&1; then
        find "$1" -type f -perm /111 2>/dev/null | while IFS= read -r _f; do
            readelf -d "$_f" 2>/dev/null | sed -n "s|.*(\(RUNPATH\|RPATH\)).*\[\(.*\)\]|${_f#$1/}|\1|\2|p"
        done | awk -F'|' '($2 == "RPATH") || ($3 != "$ORIGIN" && $3 != "$ORIGIN/")'
    else
        return 2
    fi
}
RUNDIR=${PAYLOAD_DIR:-/opt/1bit-package}
# A payload with no ELF in it is an empty or interrupted unpack, not a clean RUNPATH set: the row would print
# "ELF=0 non-$ORIGIN=0" and read as a pass over nothing. Same floor as payload-modes, for the same reason.
# Same order as payload-modes, and one chain rather than two `if`s: with the floor checked first, a missing payload
# took the FAIL branch and its own SKIP was unreachable - and in this row both ran, recording a FAIL and a SKIP for the
# same path, which also double-counted in the summary.
if [ ! -d "$RUNDIR" ]; then
    skip payload-runpath "no $RUNDIR on this system: the payload is not unpacked where an appliance keeps it, so its RUNPATHs cannot be read"
else
    # Count ELFs by magic rather than by name, and judge only after: a RUNPATH scan over a tree holding no ELF prints
    # the same PASS whether it read a payload or an empty directory.
    _rp_elfs=$(find "$RUNDIR" -type f -size +1k 2>/dev/null | head -400 | while IFS= read -r _f; do
        head -c4 "$_f" 2>/dev/null | od -An -tx1 | tr -d ' \n' | grep -q '^7f454c46' && echo x; done | wc -l)
    if [ "${_rp_elfs:-0}" -lt 5 ] 2>/dev/null; then
        fail payload-runpath "the payload at $RUNDIR holds ${_rp_elfs:-0} ELF(s) - a payload carries dozens (c33: 19 real ELFs), so this is an interrupted extraction, not a clean payload"
    else
        _rp_bad=$(payload_runpath_bad "$RUNDIR"); _rp_rc=$?
        if [ "$_rp_rc" = 2 ]; then
            skip payload-runpath "neither python3 nor readelf is here, so the payload's RUNPATHs cannot be read - that is no verdict either way"
        else
            _rp_n=$(printf '%s\n' "${_rp_bad:-}" | sed '/^$/d' | wc -l)
            _rp_s=$(printf '%s\n' "${_rp_bad:-}" | sed '/^$/d' | head -3 | tr '\n' ' ')
            RP=$(payload_runpath_verdict "${_rp_n:-0}" "${_rp_s:-}")
            case "$RP" in
                PASS*) pass payload-runpath "${RP#PASS }" ;;
                *)     fail payload-runpath "${RP#FAIL }" ;;
            esac
        fi
    fi
fi

# The layout row gets the same subject check as the other payload rows (engine#1599 measured why): existence first,
# so a host that simply has no payload SKIPs, then the subject, so an empty or interrupted unpack FAILs rather than
# passing over nothing, and only then the row itself - once, in the else, so a subject failure never falls through.
# ── the NPU sets are linked BESIDE the engine (engine#1543's coupling) ──────────────────────────────
# init.sh links the stick's /data/npu/kernels to <engine dir>/npu/kernels at boot, because that is where the
# engine looks. The failure this row exists for is silent: the boot is green and no NPU device registers.
_npu_sets=0; [ -d "${NPU_SETS_DIR:-/data/npu/kernels}" ] && _npu_sets=1
_npu_link=$(readlink "${ENGINE_DIR:-/opt/1bit}/npu/kernels" 2>/dev/null || true)
if [ "$_npu_sets" = 1 ]; then
    _nv=$(npu_link_verdict 1 "${_npu_link:-}")
    case "$_nv" in PASS*) pass npu-kernels-beside-engine "${_nv#PASS }" ;; *) fail npu-kernels-beside-engine "${_nv#FAIL }" ;; esac
else
    skip npu-kernels-beside-engine "no ${NPU_SETS_DIR:-/data/npu/kernels} on this boot, so there is no set to link beside the engine"
fi

_HL_DIR=${PAYLOAD_DIR:-/opt/1bit-package}
if [ ! -d "$_HL_DIR" ]; then
    skip payload-home-layout "no $_HL_DIR on this system: the payload is not unpacked where an appliance keeps it, so its symlinks cannot be judged here - that is not evidence"
elif [ "$(find "$_HL_DIR" -mindepth 1 2>/dev/null | wc -l)" -lt 100 ]; then
    fail payload-home-layout "the payload at $_HL_DIR holds fewer than 100 entries - an empty or interrupted unpack has no layout to judge, and a green row here would be a pass over nothing (c33: 263 entries)"
elif ! command -v python3 >/dev/null 2>&1; then
    skip payload-home-layout "no python3 here, so the payload's symlinks cannot be read - that is not evidence"
else
    _hl_bad=$(payload_home_layout_bad "$_HL_DIR" 2>/dev/null)
    _hl_nf=$(printf '%s\n' "${_hl_bad:-}" | sed '/^$/d' | awk -F'|' '$3 != "dir"' | wc -l)
    _hl_nd=$(printf '%s\n' "${_hl_bad:-}" | sed '/^$/d' | awk -F'|' '$3 == "dir"' | wc -l)
    _hl_s=$(printf '%s\n' "${_hl_bad:-}" | sed '/^$/d' | awk -F'|' '$3 != "dir"' | head -3 | tr '\n' ' ')
    [ "$_hl_nf" = 0 ] && [ -n "$(printf '%s\n' "${_hl_bad:-}" | sed '/^$/d')" ] && _hl_s=$(printf '%s\n' "${_hl_bad:-}" | sed '/^$/d' | head -2 | tr '\n' ' ')
    HL=$(payload_home_layout_verdict "$_hl_nf" "$_hl_nd" "${_hl_s:-}")
    case "$HL" in
        PASS*) pass payload-home-layout "${HL#PASS }" ;;
        *)     fail payload-home-layout "${HL#FAIL }" ;;
    esac
fi


# engine#823: the instrument the plug test needs must be on the medium (see os/mini/mkimage.sh).
if [ -x /bin/1bit-xhci-capture ]; then
    XC=$(xhci_capture_verdict 1)
else
    XC=$(xhci_capture_verdict 0)
fi
case "$XC" in
    PASS*) pass xhci-capture "${XC#PASS }" ;;
    *)     fail xhci-capture "${XC#FAIL }" ;;
esac

# ── the NPU's locked memory in force (the driver asks for unlimited; this fleet's own 99- file can override ours) ──
MEMLOCK_PID=${MEMLOCK_PID:-}
MEMLOCK_NAME=${MEMLOCK_NAME:-}
if [ -z "$MEMLOCK_PID" ]; then
    for _n in lemond 1bit; do
        _p=$(pgrep -x "$_n" 2>/dev/null | head -1)
        [ -n "$_p" ] && { MEMLOCK_PID=$_p; MEMLOCK_NAME=$_n; break; }
    done
fi
MEMLOCK_LIMITS=${MEMLOCK_LIMITS:-/proc/${MEMLOCK_PID:-0}/limits}
MLV=
if [ -n "$MEMLOCK_PID" ] && [ -r "$MEMLOCK_LIMITS" ]; then
    # the last field is the UNIT ("bytes"), the value is the field before it - $NF here reads "bytes" and looks like a
    # value that is not a number, which is the same $NF confusion as reading a path off a symlink line
    MLV=$(awk '/^Max locked memory/ {print $(NF-1)}' "$MEMLOCK_LIMITS" 2>/dev/null | head -1)
    MLV=${MLV:-}
fi
# the file whose value won: the LAST limits.d file in lexical order that mentions memlock at all
MLF=
for _c in $(ls "${LIMITS_D:-/etc/security/limits.d}"/*.conf 2>/dev/null | sort); do
    grep -qs 'memlock' "$_c" && MLF=$_c
done
if [ "$MLV" != unlimited ]; then
    _pf=${PROFILE_D:-/etc/profile.d}/90onboard-limits
    if [ -f "$_pf" ] && grep -qs 'ulimit -l unlimited' "$_pf"; then
        MLF="${MLF}${MLF:+, }$_pf (console sessions only)"
    fi
fi
ML=$(memlock_verdict "${MLV:-}" "${MEMLOCK_NAME:-}" "${MLF:-}")
case "$ML" in
    PASS*) pass memlock "${ML#PASS }" ;;
    SKIP*) skip memlock "${ML#SKIP }" ;;
    *)     fail memlock "${ML#FAIL }" ;;
esac


# ── the payload= label is consistent with the payload the image ships (engine#1557) ──────────────
if [ ! -f "$ID" ]; then
    skip payload-name "no $ID in this image build, so there is no payload label to read (an image older than the identity file is not wrong, it is silent)"
elif [ -n "${_id_gitsha:-}" ]; then
    skip payload-name "this image's identity is the mini schema (tag/engine/engine_sha256/built), which carries no payload= label and no ${PAYLOAD_DIR:-/opt/1bit-package}: there is nothing here to check"
elif [ ! -d "${PAYLOAD_DIR:-/opt/1bit-package}" ]; then
    skip payload-name "no ${PAYLOAD_DIR:-/opt/1bit-package} in this image, so the payload's own README cannot be read"
else
    PNV=$(payload_name_from_image "$ID" "${PAYLOAD_DIR:-/opt/1bit-package}")
    case "$PNV" in
        PASS*) pass payload-name "${PNV#PASS }" ;;
        SKIP*) skip payload-name "${PNV#SKIP }" ;;
        *)     fail payload-name "${PNV#FAIL }" ;;
    esac
fi

# ── the shipped kernel carries the CVE-2026-98432 ip_options use-after-free fix ──────────────────────────
# Remote, unauthenticated UAF in the IPv4 request-socket path; the fix frees ireq_opt with kfree_rcu in
# tcp_v4_reqsk_destructor. An appliance is exactly the thing this CVE is aimed at: a listening socket someone
# else can reach - so the released kernel's standing on it is a release fact, not a distro footnote.
_kver=$(uname -r 2>/dev/null)
# Kernel-headers packages do NOT carry net/ipv4/*.c, so on a normal appliance this is absent and the version
# floor decides. When a source tree IS here it outranks the floor: a backport can carry the fix below the floor
# and a vendor tree can lack it above it.
_ksrc=""
for _d in "/usr/src/linux-headers-$_kver" "/usr/src/linux-source-$_kver" "/usr/src/linux-$_kver" /usr/src/linux "/lib/modules/$_kver/build"; do
    if [ -f "$_d/net/ipv4/tcp_ipv4.c" ]; then _ksrc="$_d/net/ipv4/tcp_ipv4.c"; break; fi
done
if [ -n "$_ksrc" ]; then
    _kfn=$(sed -n '/^static void tcp_v4_reqsk_destructor/,/^}/p' "$_ksrc" 2>/dev/null)
    case "$_kfn" in
        *kfree_rcu*) _ksrckind=yes ;;
        '')          _ksrckind=absent ;;   # the function is not in that file: not a v4-TCP kernel, so no claim from it
        *)           _ksrckind=no ;;
    esac
else
    _ksrckind=absent
fi
KC=$(kernel_cve_verdict "$_kver" "$_ksrckind")
case "$KC" in
    PASS*) pass kernel-cve-2026-98432 "${KC#PASS }" ;;
    *)     fail kernel-cve-2026-98432 "${KC#FAIL }" ;;
esac
say ""
# The back-off leaves a trace: a run that only passed because the partition freed in time must say so, or the next
# person reads a clean transcript and believes nothing was ever refused.
NPU_WAITED=$(awk '{s += $1} END {print s + 0}' "$NPU_WAITFILE" 2>/dev/null)
[ "${NPU_WAITED:-0}" -gt 0 ] 2>/dev/null && say "info  npu-wait : $NPU_WAITED s spent backing off on code 102 (a client had just released the partition; the window is ~2 s)"
RC_FOOTER=1
if [ "$FAILS" = 0 ]; then
    say "RESULT: PASS ($ROWS checks, no failures)"
    # Name only the rows that actually skipped. On the 2026-11-02 live boot four of these five PASSED on the board and
    # this line still listed all five as skipped "here", which puts a false statement in the release evidence file.
    [ -z "$HW_SKIPPED" ] || say "Hardware-only rows (SKIP here): $HW_SKIPPED - they need the Strix Halo GPU and NPU."
    exit 0
fi
say "RESULT: FAIL ($FAILS of $ROWS checks failed)"
exit 1
