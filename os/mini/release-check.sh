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
# gpu, npu, models, chat-hrx, chat-npu. Everything else - mode, kernel, Secure Boot, Lemonade up
# with its key, the stick's GGUF answering on vulkan, `lemonade bench`, the sealed NPU sets and
# the signed manifests over them, the licences and the image identity - is checked in the VM as well.
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
    case "$1" in gpu|npu|models|chat-hrx|chat-npu|chat-npu-followup) HW_SKIPPED="${HW_SKIPPED:+$HW_SKIPPED, }$1" ;; esac
    say "SKIP  $1 : $2"
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

# The first value of a JSON key - quoted or not, since usage counts are bare numbers - for the row's
# small bodies (one line, one message: the same assumption chat() already makes).
json_str() { printf '%s' "$1" | sed -n "s/.*\"$2\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\)\"\{0,1\}.*/\\1/p" | head -1; }

# Text on its way into a JSON string: no control characters (a newline would end the line) and the
# two characters JSON escapes. Used to replay the long turn's own answer as the follow-up's history,
# as the web page does.
json_text() { printf '%s' "$1" | tr '\n\r\t' '   ' | sed 's/\\/\\\\/g; s/"/\\"/g'; }

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
    [ $rc = 0 ] && echo "release-check self-test PASS" || echo "release-check self-test FAIL"
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
# Cycle 20, the owner's "blank key": blank mode is the SHIPPED state, and its contract is stock Lemonade's - the
# server answers without a key and any placeholder is accepted. So every request below sends ${KEY:-none}: one code
# path, and on a keyed machine it is the real key (a wrong one there is still a 401, which the auth row asserts).
KEYH=${KEY:-none}
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
    MODELS=$(got --header="Authorization: Bearer $KEYH" "$API/api/v1/models")
    WITH=$(status --header="Authorization: Bearer $KEYH" "$API/api/v1/models")
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
    _get=got; [ "${2:-}" = slow ] && _get=got_slow
    R=$($_get --header='Content-Type: application/json' --header="Authorization: Bearer $KEYH" \
        --post-data="{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"/no_think Reply with exactly: 1BIT OK\"}],\"max_tokens\":48,\"temperature\":0}" \
        "$API/api/v1/chat/completions")
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
                fail hrx-device "no HRX0 listed by $LSRV (${LD:+$(printf '%s' "$LD" | tr '\n' ' ' | cut -c1-120)})"
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
    TDR0=$(tdr_count)
    Q_LONG='/no_think Explain in detail how tides work: the moon and the sun, the shape of ocean basins, why some coasts have two high tides a day and others one, and what an amphidromic point is'
    _post() {  # <json body> -> the reply
        got_slow --header='Content-Type: application/json' --header="Authorization: Bearer $KEYH" \
            --post-data="$1" "$API/api/v1/chat/completions"
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
    BL=$(LEMONADE_API_KEY="$KEYH" timeout 900 "$BENCH" bench "$VMODEL" --scenarios chat-short --runs 1 --warmup 0 \
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

ID=/etc/1bit/release-identity
if [ -f "$ID" ]; then
    pass identity "$(tr '\n' ' ' < "$ID")"
    if [ -f /data/SHA256SUMS ]; then
        if (cd /data && sha256sum -c SHA256SUMS >/dev/null 2>&1); then pass sha256sums "/data/SHA256SUMS verifies"
        else fail sha256sums "/data/SHA256SUMS does not verify (cd /data && sha256sum -c SHA256SUMS)"; fi
    else
        skip sha256sums "no /data/SHA256SUMS on the data partition; the image hash lives in the release table beside the .img.zst"
    fi
else
    skip identity "no /etc/1bit/release-identity in this image build; the tag and the image sha256 are in the release table beside the .img.zst"
fi

say ""
if [ "$FAILS" = 0 ]; then
    say "RESULT: PASS ($ROWS checks, no failures)"
    # Name only the rows that actually skipped. On the 2026-11-02 live boot four of these five PASSED on the board and
    # this line still listed all five as skipped "here", which puts a false statement in the release evidence file.
    [ -z "$HW_SKIPPED" ] || say "Hardware-only rows (SKIP here): $HW_SKIPPED - they need the Strix Halo GPU and NPU."
    exit 0
fi
say "RESULT: FAIL ($FAILS of $ROWS checks failed)"
exit 1
