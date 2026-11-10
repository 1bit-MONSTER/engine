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

# The gate must refuse for each FATAL condition, warn for each WARN condition, and promote warnings
# under --strict -- because a gate that passes everything is indistinguishable from no gate, and a
# gate that refuses everything gets removed.
#
# Each case fakes the world under $tmp and runs the real script with its roots pointed there, so this
# runs anywhere: no GPU, no model, no hold.
set -uo pipefail

tool=$(dirname "$0")/../bench-preflight.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fake() {  # fake <uptime_s> <busy> <mem_avail_gib> [gtt_used_b] [gtt_total_b] -> prints the root
    local root="$tmp/$1_$2_$3_${4:-0}"
    mkdir -p "$root/proc" "$root/sys/class/drm/card0/device/hwmon/hwmon4"
    printf '%s 1000\n' "$1" > "$root/proc/uptime"
    printf 'MemTotal: 131072000 kB\nMemAvailable: %s kB\n' "$(( $3 * 1048576 ))" > "$root/proc/meminfo"
    printf '%s\n' "$2" > "$root/sys/class/drm/card0/device/gpu_busy_percent"
    printf '%s\n' "${4:-0}" > "$root/sys/class/drm/card0/device/mem_info_gtt_used"
    printf '%s\n' "${5:-107374182400}" > "$root/sys/class/drm/card0/device/mem_info_gtt_total"
    printf '50000\n' > "$root/sys/class/drm/card0/device/hwmon/hwmon4/temp1_input"
    printf '%s\n' "$root"
}

patched() {  # patched <root> [args...] -> runs the real script against the world in <root>
    local root=$1; shift
    sed -e "s#/proc/uptime#$root/proc/uptime#g" \
        -e "s#/proc/meminfo#$root/proc/meminfo#g" \
        -e "s#/sys/class/drm/card0/device#$root/sys/class/drm/card0/device#g" \
        -e 's/^busy_window=10$/busy_window=1/' \
        -e "s#^token=.*#token=$root/GPU-TOKEN#" \
        "$tool" > "$root/run.sh"
    # No server and no ticket on a fake box: drive the process loop from /dev/null and let the pipeline
    # be a no-op. The loop's own heredoc and pipeline are two lines; replace both by exact line number
    # pattern so nothing else in the file is touched.
    python3 - "$root/run.sh" <<'PYFIX'
import re, sys
path = sys.argv[1]
src = open(path).read().splitlines()
out, i = [], 0
while i < len(src):
    if src[i].startswith('done < <(pgrep'):
        out.append('done < /dev/null')
        i += 1
        # the next line is the pipeline feeding that loop
        if i < len(src) and 'bench-preflight' in src[i]:
            i += 1
        continue
    out.append(src[i]); i += 1
open(path, 'w').write('\n'.join(out) + '\n')
PYFIX
    sed -i '/prov-libs/,/^fi$/d' "$root/run.sh"
    sed -i '/^for lib in llama-bench/,/^done$/d' "$root/run.sh"
    # A syntax error in the patched script exits non-zero, which the test would read as "the gate
    # refused". Prove it parses before asserting on its exit code.
    bash -n "$root/run.sh" || { echo "FAIL patched script does not parse: $root/run.sh" >&2; exit 99; }
    bash "$root/run.sh" "$@"
}

pass=0; fail=0
expect() {  # expect <want: ok|no> <name> <root> [args...]
    local want=$1 name=$2 root=$3; shift 3
    out=$(patched "$root" "$@" 2>&1); rc=$?
    got=ok
    [ $rc -ne 0 ] && got=no
    if [ "$got" = "$want" ]; then
        printf 'ok   %-38s rc=%d\n' "$name" "$rc"; pass=$((pass+1))
    else
        printf 'FAIL %-38s rc=%d want=%s\n%s\n' "$name" "$rc" "$want" "$out"; fail=$((fail+1))
    fi
}

want_text() {  # want_text <name> <root> <needle> [args...] -> exit 0 iff <needle> appears
    local name=$1 root=$2 needle=$3; shift 3
    out=$(patched "$root" "$@" 2>&1)
    if printf '%s' "$out" | grep -q -- "$needle"; then
        printf 'ok   %-38s saw %s\n' "$name" "$needle"; pass=$((pass+1))
    else
        printf 'FAIL %-38s missing %s\n%s\n' "$name" "$needle" "$out"; fail=$((fail+1))
    fi
}

GIB=1073741824
# FATAL conditions
expect ok  "fresh boot is a WARNING, not fatal"     "$(fake 300 0 60)"
expect no  "--strict makes fresh boot fatal"        "$(fake 300 0 60)" --strict
expect no  "die busy (38%)"                         "$(fake 3600 38 60)"
expect no  "RAM below the floor"                    "$(fake 3600 0 12)"
expect no  "--need-gtt 40 with 16 GiB free"         "$(fake 3600 0 60 $((112*GIB)) $((128*GIB)))" --need-gtt 40
expect ok  "--need-gtt 4 with 16 GiB free"          "$(fake 3600 0 60 $((112*GIB)) $((128*GIB)))" --need-gtt 4
expect ok  "--need-gtt 40 with 100 GiB free"        "$(fake 3600 0 60 $((28*GIB))  $((128*GIB)))" --need-gtt 40

# WARN conditions and strict
expect ok  "clean box"                              "$(fake 3600 0 60)"
expect ok  "clean box under --strict"               "$(fake 3600 0 60)" --strict
want_text  "the warning is printed, not hidden"     "$(fake 300 0 60)" "WARNING(S)"
want_text  "uptime warning names its reason"        "$(fake 300 0 60)" "still settling"

# --dispatch must fail when the bench prints no dispatch lines, rather than print an empty tally.
cat > "$tmp/fakebench" <<'SH'
#!/bin/sh
echo "| model | pp512 | 3000.00 t/s |" >&2
exit 0
SH
chmod +x "$tmp/fakebench"
mkdir -p "$tmp/models"; : > "$tmp/models/Qwen3-4B-Q4_K_M.gguf"
PREFLIGHT_MODEL=$tmp/models/Qwen3-4B-Q4_K_M.gguf expect no "--dispatch with silent bench" \
    "$(fake 3600 0 60)" --dispatch "$tmp/fakebench" 512

# --dispatch must PIN the load mode (engine #1668): `-lm none` is +17.5 % prefill at -1.6 % decode for +604 MiB of GTT, so a tally
# taken with the mode unstated is a tally of two variables. The bench below answers like a live one and records its own argv, so the
# assertions are about what was EXECUTED rather than about what the script printed.
cat > "$tmp/argbench" <<'SH'
#!/bin/sh
printf '%s\n' "$*" >> "$ARGV_FILE"
echo "ggml_hrx dispatch: common.mul_mat.q4_k_q8_1_x4_prefill" >&2
echo "| qwen3-4b | 2.30 GiB | 4B | HRX | 99 | 1 | pp512 | 3181.97 t/s |" >&2
exit 0
SH
chmod +x "$tmp/argbench"

argv_case() {  # argv_case <name> <want-needle> [PREFLIGHT_LM]
    local name=$1 needle=$2 lm=${3:-} root
    root=$(fake 3600 0 60)                      # the same clean box for every case; patched() rewrites run.sh in place
    : > "$tmp/argv"
    out=$(ARGV_FILE=$tmp/argv PREFLIGHT_LM=$lm PREFLIGHT_MODEL=$tmp/models/Qwen3-4B-Q4_K_M.gguf \
          patched "$root" --dispatch "$tmp/argbench" 512 2>&1)
    if grep -q -- "$needle" "$tmp/argv"; then
        printf 'ok   %-38s bench argv has %s\n' "$name" "$needle"; pass=$((pass+1))
    else
        printf 'FAIL %-38s bench argv lacks %s (argv: %s)\n%s\n' "$name" "$needle" "$(cat "$tmp/argv")" "$out"; fail=$((fail+1))
    fi
    printf '%s' "$out" > "$tmp/lastout"
}

argv_case "load mode is pinned to the default" "-lm mmap"
argv_case "PREFLIGHT_LM=none reaches the bench" "-lm none" none
# A mode nobody pinned must be refused, not forwarded: a typo would silently mean something else.
root=$(fake 3600 0 60); : > "$tmp/argv"
out=$(ARGV_FILE=$tmp/argv PREFLIGHT_LM=bogus PREFLIGHT_MODEL=$tmp/models/Qwen3-4B-Q4_K_M.gguf \
      patched "$root" --dispatch "$tmp/argbench" 512 2>&1)
if [ -s "$tmp/argv" ] || ! printf '%s' "$out" | grep -q 'is not a llama-bench load mode'; then
    printf 'FAIL %-38s a bogus PREFLIGHT_LM was not refused before the bench ran\n%s\n' "bogus load mode is refused" "$out"; fail=$((fail+1))
else
    printf 'ok   %-38s refused before the bench ran\n' "bogus load mode is refused"; pass=$((pass+1))
fi
# The header must name the mode, so a capture says which one it was.
if grep -q -- "-lm none" "$tmp/lastout"; then   # the last argv_case ran with PREFLIGHT_LM=none
    printf 'ok   %-38s header names the load mode\n' "tally header states -lm"; pass=$((pass+1))
else
    printf 'FAIL %-38s header does not name the load mode\n%s\n' "tally header states -lm" "$(cat "$tmp/lastout")"; fail=$((fail+1))
fi

printf '\nbench-preflight test: %d passed, %d failed\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
