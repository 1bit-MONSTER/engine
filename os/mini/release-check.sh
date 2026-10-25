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
# with its key, the stick's GGUF answering on vulkan, `lemonade bench`, the sealed NPU sets, the
# licences and the image identity - is checked in the VM as well.
set -u

LOG=/data/release-check.log
FAILS=0
ROWS=0
say() { printf '%s\n' "$*"; [ -w /data ] 2>/dev/null && printf '%s\n' "$*" >> "$LOG" 2>/dev/null; return 0; }
pass() { ROWS=$((ROWS + 1)); say "PASS  $1 : $2"; }
fail() { ROWS=$((ROWS + 1)); FAILS=$((FAILS + 1)); say "FAIL  $1 : $2"; }
skip() { ROWS=$((ROWS + 1)); say "SKIP  $1 : $2"; }
got() { wget -q -O - "$@" 2>/dev/null || curl -sf "$@" 2>/dev/null; }

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
    [ $rc = 0 ] && echo "release-check self-test PASS" || echo "release-check self-test FAIL"
    exit $rc
fi


PORT=$(sed -n 's/.*"port"[^0-9]*\([0-9]\+\).*/\1/p' /data/lemonade/config/config.json 2>/dev/null | head -1)
PORT=${PORT:-13305}
KEY=$(sed -n 's/^LEMONADE_API_KEY=//p' /data/lemonade/lemond.conf 2>/dev/null | head -1)
API="http://127.0.0.1:$PORT"
CATALOG=/opt/lemonade/bin/resources/server_models.json
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
    skip secureboot "no mokutil in the image; firmware reports EFI but not the Secure Boot state"
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
    fail lemonade "no LEMONADE_API_KEY in /data/lemonade/lemond.conf: a network-facing server without a key"
else
    MODELS=$(got --header="Authorization: Bearer $KEY" "$API/api/v1/models")
    WITH=$(status --header="Authorization: Bearer $KEY" "$API/api/v1/models")
    BODY=$(got "$API/api/v1/models")
    WITHOUT=$(status "$API/api/v1/models")
    case "$(lemonade_verdict "$WITH" "$MODELS" "$WITHOUT")" in
        PASS*) pass lemonade "$(printf '%s' "$(lemonade_verdict "$WITH" "$MODELS" "$WITHOUT")" | sed 's/^PASS //')";;
        *) fail lemonade "$(printf '%s' "$(lemonade_verdict "$WITH" "$MODELS" "$WITHOUT")" | sed 's/^FAIL //')";;
    esac
    : "$BODY"
fi

have_model() { printf '%s' "${MODELS:-}" | grep -q "\"$1\""; }
GGUF=$(ls /data/models/*.gguf 2>/dev/null | head -1)
GGUF_NAME=$(basename "${GGUF:-none}" .gguf)

# ── the onebit entries, against the manifest the build wrote from the shipped catalog ───────────
MAN=/etc/1bit/onebit-models.txt
if [ -z "${MODELS:-}" ]; then
    fail models "Lemonade did not answer, so the model list cannot be checked"
elif [ ! -f "$MAN" ] && ! ( [ -e /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1 ) && ! ls /dev/accel/accel* >/dev/null 2>&1; then
    CAT=/opt/lemonade/bin/resources/server_models.json
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
    if [ "$MODE" != "$(sed -n 's/^mode=//p' "$MAN" 2>/dev/null | head -1)" ] && [ "$HRX_GOT" = 0 ]; then
        skip models "the machine has no AMD GPU or NPU ($HRX_WANT HRX + $NPU_WANT NPU entries are registered, Lemonade hides them here): checked in the registry by vmtest.sh"
    elif [ "$HRX_GOT" = "$HRX_WANT" ] && [ "$NPU_GOT" = "$NPU_WANT" ]; then
        pass models "$HRX_WANT onebit HRX entries and $NPU_WANT NPU entries listed"
    else
        fail models "$HRX_GOT/$HRX_WANT HRX and $NPU_GOT/$NPU_WANT NPU entries listed; missing:$MISSING"
    fi
fi
if [ -n "${MODELS:-}" ] && [ "$GGUF_NAME" != none ]; then
    if have_model "$GGUF_NAME"; then pass gguf "the stick's own model '$GGUF_NAME' is listed"
    else fail gguf "/data/models/$GGUF_NAME.gguf is on the data partition but Lemonade does not list it"; fi
else
    skip gguf "no /data/models/*.gguf on the data partition"
fi

# ── one chat per backend that this machine can actually run ─────────────────────────────────────
chat() {   # chat <model id> -> sets ANSWER and TPS
    ANSWER=; TPS=
    R=$(got --header='Content-Type: application/json' --header="Authorization: Bearer $KEY" \
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

VMODEL=$(pick '[^"]*')
if [ -n "$VMODEL" ]; then
    chat "$VMODEL"
    if [ -n "$TPS" ]; then pass chat "$VMODEL answered \"$ANSWER\" at ${TPS} tok/s"
    else fail chat "$VMODEL did not answer ($ANSWER)"; fi
else
    skip chat "Lemonade lists no model this machine can run"
fi
HRX=$(pick '[^"]*-1bit')
if [ -e /dev/kfd ] && [ -n "$HRX" ]; then
    chat "$HRX"
    if [ -n "$TPS" ]; then pass chat-hrx "$HRX (onebit, HRX) answered \"$ANSWER\" at ${TPS} tok/s"
    else fail chat-hrx "$HRX (onebit, HRX) did not answer ($ANSWER)"; fi
else
    skip chat-hrx "no AMD GPU in this VM: the onebit HRX entries only run on the machine"
fi
NPU=$(pick '[^"]*-NPU-1bit')
if ls /dev/accel/accel* >/dev/null 2>&1 && [ -n "$NPU" ]; then
    chat "$NPU"
    if [ -n "$TPS" ]; then pass chat-npu "$NPU (onebit, NPU) answered \"$ANSWER\" at ${TPS} tok/s"
    else fail chat-npu "$NPU (onebit, NPU) did not answer ($ANSWER)"; fi
else
    skip chat-npu "no NPU in this VM: the onebit NPU entries only run on the machine"
fi

# ── lemonade bench, one chat-short row on the default model ─────────────────────────────────────
BENCH=/opt/lemonade/bin/lemonade
if [ ! -x "$BENCH" ]; then
    skip bench "no $BENCH in the image"
elif [ -z "$VMODEL" ]; then
    skip bench "no model to bench"
else
    BLF=/tmp/bench.json
    # `lemonade bench` prints one human row per run: "Run 1/1... TTFT=30.6ms PP=881.6t/s TPS=120.2".
    # That line is the evidence; --json/--output is only a fallback because its shape moves between
    # Lemonade versions (and it wrote nothing to stdout in the 2026-10-23 candidate image).
    BL=$(LEMONADE_API_KEY="$KEY" timeout 900 "$BENCH" bench "$VMODEL" --scenarios chat-short --runs 1 --warmup 0 \
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

# ── sealed NPU sets: the method never ships in the clear ────────────────────────────────────────
SETS=$(find /data/npu/kernels -name layout.json 2>/dev/null | wc -l)
if [ "$SETS" = 0 ]; then
    fail sealed "no NPU kernel set on the data partition (find /data/npu/kernels -name layout.json)"
else
    BAD=0
    for f in $(find /data/npu/kernels -type f 2>/dev/null); do
        [ "$(basename "$f")" = layout.json ] && continue
        [ "$(head -c 8 "$f" 2>/dev/null)" = 1BPKCRYP ] && continue
        BAD=$((BAD + 1)); say "      not sealed: $f"
    done
    [ "$(find /data/npu/kernels -name '*.mlir' -o -name 'layer1.work' 2>/dev/null | wc -l)" = 0 ] || { BAD=$((BAD + 1)); say "      plaintext build tree on the data partition"; }
    if [ "$BAD" = 0 ]; then pass sealed "$SETS sealed sets, $PLAIN_TOTAL kernels, none in the clear"
    else fail sealed "$BAD file(s) under /data/npu/kernels are not sealed artifacts"; fi
fi

# ── licences and image identity ────────────────────────────────────────────────────────────────
LIC=; NOT=; LOOKED=
for d in /usr/share/licenses/1bit-os /opt/lemonade/bin/resources/../../.. /opt/lemonade /opt/lemonade/bin /; do
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
    say "Hardware-only rows (SKIP here): gpu, npu, models, chat-hrx, chat-npu - they need the Strix Halo GPU and NPU."
    exit 0
fi
say "RESULT: FAIL ($FAILS of $ROWS checks failed)"
exit 1
