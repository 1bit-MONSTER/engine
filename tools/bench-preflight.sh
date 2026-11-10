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

# Whether a throughput number taken right now would mean anything.
#
# Why this exists: on 2026-11-08 a chunk-size sweep ranked five configurations from numbers that
# differed by up to 15% between two runs of the SAME configuration. Three llama-servers were resident
# on HRX0 the whole time. Every ranking in that sweep was noise, and it took a second idle-box sweep
# to publish a real one. The check is cheap and the write-up is expensive, so it runs first.
#
#   tools/bench-preflight.sh                    check the box; FATAL and exit 1 if a number would lie
#   tools/bench-preflight.sh --strict           every warning becomes fatal
#   tools/bench-preflight.sh --need-gtt GiB     also fatal if the die has less than GiB free GTT
#   tools/bench-preflight.sh --dispatch B N     then one GGML_HRX_LOG_DISPATCH=1 pass of `B -p N` and
#                                               the matcher each MUL_MAT took
#
# FATAL (exit 1, reason on stderr) -- each of these corrupts the number itself:
#   gpu_busy_percent > 5 for 10 s   somebody else is using the die; llama-bench cannot see their cost
#   MemAvailable < 20 GiB           the box's floor, so a sweep cannot push another lane into swap
#   free GTT < --need-gtt           the model will not load, or loads and thrashes
#
# WARN (exit 0, listed with pid/cmdline/age) -- each of these is the caller's call, not the tool's:
#   uptime < 10 min                 clock and page cache still settling after a boot
#   a resident llama-server/1bit serve/max serve
#   a live gpu-hold ticket that is not the caller's
#
# --strict is for a lane that needs a clean die and would rather be told no. Splitting the two exists
# because a gate that refuses for conditions which are permanent on a shared box is a gate nobody runs,
# and a warning nobody can miss is worth more than a refusal everybody works around.
#
# Run a benchmark under it, so the check and the measurement share one hold:
#   gpu-hold -l mybench -t 14 -- tools/bench-preflight.sh --dispatch llama-bench 512
set -uo pipefail

busy_pct=5
busy_window=10
ram_floor_gib=${BENCH_RAM_FLOOR_GIB:-20}
uptime_floor=600

die=/sys/class/drm/card0/device
here=$(dirname "$0")
strict=0
need_gtt=0
watch=0

while [ $# -gt 0 ]; do
    case $1 in
        --strict) strict=1; shift ;;
        --need-gtt) need_gtt=${2:?--need-gtt needs a GiB value}; shift 2 ;;
        --watch) watch=1; shift; [ "${1:-}" = "--" ] && shift; break ;;
        --dispatch) break ;;
        *) echo "bench-preflight: unknown option $1" >&2; exit 2 ;;
    esac
done

warned=""
warn() { warned+="$1"$'\n'; }
fatal() {
    printf 'bench-preflight: FAIL -- %s\n' "$1" >&2
    if [ -n "$warned" ]; then
        printf 'bench-preflight: also worth knowing:\n%s' "$warned" >&2
    fi
    printf 'bench-preflight: a number measured now would not be attributable to the code.\n' >&2
    exit 1
}

# gpu_busy over a window. The counter is per-sample, so one idle reading proves nothing.
if [ -r "$die/gpu_busy_percent" ]; then
    for i in $(seq "$busy_window"); do
        busy=$(cat "$die/gpu_busy_percent")
        [ "$busy" -gt "$busy_pct" ] &&
            fatal "gpu_busy_percent read ${busy}% (limit ${busy_pct}%) on sample ${i}-of-${busy_window}"
        sleep 1
    done
else
    fatal "no $die/gpu_busy_percent; cannot tell whether the die is free"
fi

avail_gib=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo)
[ "$avail_gib" -lt "$ram_floor_gib" ] &&
    fatal "MemAvailable is ${avail_gib} GiB, below the ${ram_floor_gib} GiB floor"

if [ "$need_gtt" != 0 ]; then
    [ -r "$die/mem_info_gtt_used" ] && [ -r "$die/mem_info_gtt_total" ] ||
        fatal "--need-gtt was given but $die/mem_info_gtt_used is not readable"
    gtt_free_gib=$(( ($(cat "$die/mem_info_gtt_total") - $(cat "$die/mem_info_gtt_used")) / 1073741824 ))
    if [ "$gtt_free_gib" -lt "$need_gtt" ]; then
        holders=$(pgrep -af 'llama-server|1bit .*serve|max serve' | cut -c1-120 | sed 's/^/      /')
        fatal "GTT free is ${gtt_free_gib} GiB, below the ${need_gtt} GiB asked for; holding processes:
$holders"
    fi
fi

up_s=$(cut -d' ' -f1 /proc/uptime | cut -d. -f1)
[ "$up_s" -lt "$uptime_floor" ] &&
    warn "  uptime ${up_s}s < ${uptime_floor}s (clock and page cache still settling)"

now=$(date +%s)
while read -r pid args; do
    [ -n "$pid" ] || continue
    age=$(( (now - $(stat -c %Y "/proc/$pid" 2>/dev/null || echo "$now")) / 60 ))
    warn "  resident ${pid} (${age}m): ${args:0:120}"
done < <(pgrep -af 'llama-server|1bit .*serve|max serve' 2>/dev/null |
         grep -v 'bench-preflight' | grep -v 'llama-bench ' || true)

token=${GPU_TOKEN_FILE:-$HOME/GPU-TOKEN}
if [ -r "$token" ]; then
    holder=$(cat "$token")
    case $holder in
        *"$$"*) ;;                                   # this process already holds it
        *) warn "  gpu-hold ticket held by someone else: $holder" ;;
    esac
fi

if [ -n "$warned" ] && [ "$strict" = 1 ]; then
    fatal "the box is usable but not clean, and --strict was asked for"
fi

# --watch: the start gate says nothing about the rest of the run. engine#572's second arm started at 85 GiB
# available, another load arrived, MemAvailable fell to 1-3 GiB with swap full, and the arm timed out at 540 s
# -- so the number that came back (a timeout) and the number that did not (the arm) were both produced on a box
# nobody had described. Sample for the whole command and report start/min/end, so an arm is either clean or
# visibly contaminated where its number is printed.
if [ "$watch" = 1 ]; then
    # engine#572: the start gate describes the wrong moment -- a mid-arm load can take MemAvailable from 85 GiB to
    # 1-3 GiB and the arm times out, so the number is produced on a box nobody described. The sampler lives in its own
    # tool so it can be tested on a busy box (this gate would refuse to reach it).
    [ $# -gt 0 ] || { echo "bench-preflight: --watch needs a command after --" >&2; exit 2; }
    exec "$here/bench-arm.sh" --floor "$ram_floor_gib" -- "$@"
fi

printf 'bench-preflight: OK\n'
printf '  kernel     %s\n' "$(uname -r)"
printf '  uptime     %ss  gpu_busy %s%%  mem_avail %s GiB\n' "$up_s" "$busy" "$avail_gib"
printf '  temp       %s mC\n' "$(cat "$die"/hwmon/hwmon*/temp1_input 2>/dev/null | sort -n | tail -1)"
if [ -n "$warned" ]; then
    printf '\nbench-preflight: %d WARNING(S) -- a number taken now can still be real, but say what was\n' \
           "$(printf '%s' "$warned" | grep -c .)"
    printf 'running when you post it, because a reader cannot check what you did not write down:\n%s\n' "$warned"
fi
for lib in llama-bench llama-perplexity; do
    b=$(command -v "$lib" || echo "$here/../build-hrx/bin/$lib")
    [ -x "$b" ] || continue
    printf '  %-10s %s  md5 %s\n' "$lib" "$b" "$(md5sum "$b" | cut -c1-12)"
done
if [ -x "$HOME/lb/prov-libs.sh" ]; then
    bin=$(command -v llama-bench || echo "$here/../build-hrx/bin/llama-bench")
    "$HOME/lb/prov-libs.sh" "$(dirname "$bin")" | sed 's/^/  /'
fi

[ "${1:-}" = "--dispatch" ] || exit 0

# --dispatch <bench binary> <prompt tokens>: one prefill pass with the dispatch log on, then the tally.
bench=${2:?usage: bench-preflight.sh --dispatch <llama-bench> <prompt tokens>}
tokens=${3:?usage: bench-preflight.sh --dispatch <llama-bench> <prompt tokens>}
model=${PREFLIGHT_MODEL:-$HOME/models/Qwen3-4B-Q4_K_M.gguf}
[ -x "$bench" ] || fatal "no such binary: $bench"
[ -r "$model" ] || fatal "no model at $model (set PREFLIGHT_MODEL)"

log=$(mktemp)
trap 'rm -f "$log"' EXIT
# The LOAD MODE is part of the measurement (engine #1668): `-lm none` is +17.5 % prefill at -1.6 % decode for +604 MiB of GTT on this
# box, so a dispatch tally taken with the mode unstated is a tally of two variables. It is pinned here, printed in the header below,
# and overridable only through PREFLIGHT_LM so a comparison sets both arms deliberately instead of inheriting a default. The value
# set is llama-bench's own (`-lm, --load-mode <none|mmap|mlock|mmap+mlock|dio>`, tools/llama-bench/llama-bench.cpp at the pin).
lm=${PREFLIGHT_LM:-mmap}
case "$lm" in
    none|mmap|mlock|mmap+mlock|dio) ;;
    *) fatal "PREFLIGHT_LM='$lm' is not a llama-bench load mode (none|mmap|mlock|mmap+mlock|dio); the mode is part of the measurement, so it is not guessed (engine #1668)" ;;
esac
GGML_HRX_LOG_DISPATCH=1 "$bench" -m "$model" -p "$tokens" -n 0 -ngl 99 --device HRX0 -r 3 -lm "$lm" \
    >/dev/null 2>"$log"

lines=$(grep -c 'ggml_hrx dispatch:' "$log" || true)
# A silent bench is the failure this flag exists to catch: an empty tally looks like "nothing declined"
# and would be read as a clean bill of health.
[ "$lines" -eq 0 ] &&
    fatal "GGML_HRX_LOG_DISPATCH=1 produced no 'ggml_hrx dispatch:' lines; an empty tally would be a fiction"
printf '\n  matcher decisions at pp%s (one run, -r 3, -lm %s):\n' "$tokens" "$lm"
grep 'ggml_hrx dispatch:' "$log" |
    sed -E 's/^.*ggml_hrx dispatch: //; s/ declined.*/ [declined]/' |
    sort | uniq -c | sort -rn | sed 's/^/    /' | head -40
grep -q 't/s' "$log" || fatal "the bench produced no throughput line; see $log"
