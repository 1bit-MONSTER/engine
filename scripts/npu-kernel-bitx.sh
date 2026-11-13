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
#
#
#
# npu-kernel-bitx.sh: is a kernel family BIT-EXACT against the host, on the device? (engine#1783, engine#1808)
#
#   npu-kernel-bitx.sh <container.q4nx> <kernels dir> [--repeats N] [prompt ...]
#
# A kernel family's contract is not "it runs": it is bit-exact against the host reference, on the device, over more than one
# input. The engine carries the reference the device can be exact against -- NPU_INFER_HOST_GEMM=i8, whose int32 accumulation
# is the only thing an int8 kernel family can match by construction (engine#1808) -- so the check is: serve the same container
# twice, once on the NPU and once through that host arm, dump every position's raw logits (--dump-logits), and compare bytes.
#
# Why this is a script and not a ctest row: it needs the NPU, a container and a ticket. It therefore REFUSES to run outside an
# npu-hold ticket (no ~/NPU-TOKEN) rather than starting a bare serve on the device, and it releases what it starts.
#
# --repeats N runs the DEVICE arm N times and requires every run's dumps to be byte-identical to each other as well as to the host:
# the family contract is "bit-exact against the host reference, and 3 repeats identical", and doing the repeats by hand in a second
# script is how the second half gets skipped.
#
# Measured with it (whisper's 768 family, 2026-11-12): four prompts -- a single token, a three-token, a five-token and an
# eight-token one, so position-0 and repeated-token edges are covered -- all IDENTICAL, 4096 B each, and three repeats identical.
set -u
bin=${ONEBIT_BIN:-}
if [ -z "$bin" ]; then
    for c in "$(dirname "$0")/../build/1bit" "$HOME/.local/share/1bit/1bit" "$(command -v 1bit 2>/dev/null)"; do
        [ -n "$c" ] && [ -x "$c" ] && { bin=$c; break; }
    done
fi
[ -n "$bin" ] && [ -x "$bin" ] || { echo "npu-kernel-bitx: no 1bit binary (set ONEBIT_BIN=)" >&2; exit 2; }
if [ ! -e "$HOME/NPU-TOKEN" ]; then
    echo "npu-kernel-bitx: not inside an npu-hold ticket (no ~/NPU-TOKEN); refusing to start a serve on the device" >&2
    exit 2
fi
container=${1:-}; kernels=${2:-}
[ -n "$container" ] && [ -n "$kernels" ] || {
    echo "usage: $0 <container.q4nx> <kernels dir> [prompt ...]" >&2; exit 2; }
shift 2
repeats=1
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --repeats) repeats=${2:-1}; shift 2 ;;
        *) args+=("$1"); shift ;;
    esac
done
if ! [ "$repeats" -ge 1 ] 2>/dev/null; then echo "npu-kernel-bitx: --repeats needs a count >= 1" >&2; exit 2; fi
prompts=(${args[@]+"${args[@]}"})
[ ${#prompts[@]} -gt 0 ] || prompts=('<1>' '<7> <9> <11>' '<300> <301> <302> <303> <304>' '<1000> <1000> <2> <3> <4> <5> <6> <7>')

out=${BITX_OUT:-$HOME/.cache/scratch-bitx/$(basename "$container" .q4nx)}
rm -rf "$out"; mkdir -p "$out/dev" "$out/host"

serve() {   # <dir> <port> [env ...]
    local dir=$1 port=$2; shift 2
    env "$@" "$bin" forward-serve -m "$container" --kernels "$kernels" --prefix-slots 0 \
        -p "$port" --dump-logits "$dir" > "$dir.log" 2>&1 &
    local pid=$!
    local up=0
    for _ in $(seq 1 60); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" = 0 ]; then
        echo "  $dir: serve did not come up -- $(grep -E 'init_elf failed|missing|error' "$dir.log" | tail -1)" >&2
        kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 1
    fi
    local i=0 rc=0
    for p in "${prompts[@]}"; do
        i=$((i + 1))
        code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$port/v1/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"prompt\":\"$p\",\"max_tokens\":1,\"temperature\":0}" -m 240)
        [ "$code" = 200 ] || { echo "  $dir: prompt $i -> http $code" >&2; rc=1; }
    done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    return $rc
}

echo "device :"
for r in $(seq 1 "$repeats"); do
    d=$( [ "$repeats" = 1 ] && printf '%s' "$out/dev" || printf '%s/dev%s' "$out" "$r" )
    mkdir -p "$d"
    serve "$d" $((18150 + r - 1)) || exit 1
done
echo "host   :"; serve "$out/host" 18151 NPU_INFER_HOST_GEMM=i8 || exit 1

bad=0 n=0
# Every device run must match the host; then the repeats must match each other. Comparison loops over the run dirs actually made.
devdirs=""
for r in $(seq 1 "$repeats"); do devdirs="$devdirs $( [ "$repeats" = 1 ] && printf '%s' "$out/dev" || printf '%s/dev%s' "$out" "$r" )"; done
first=$(printf '%s' "$devdirs" | awk '{print $1}')
for d in $devdirs; do
    for f in "$d"/step_*_0.bin; do
        [ -e "$f" ] || continue
        b=$(basename "$f"); n=$((n + 1))
        if [ ! -f "$out/host/$b" ]; then echo "  $d/$b: the host run wrote no counterpart"; bad=$((bad + 1)); continue; fi
        if cmp -s "$f" "$out/host/$b"; then echo "  $d/$b IDENTICAL to host ($(stat -c%s "$f") B)"
        else echo "  $d/$b DIFFERS from host ($(cmp -l "$f" "$out/host/$b" | wc -l) bytes)"; bad=$((bad + 1)); fi
    done
done
if [ "$repeats" -gt 1 ]; then
    drift=0 total=0
    for f in "$first"/step_*_0.bin; do
        [ -e "$f" ] || continue
        b=$(basename "$f"); total=$((total + 1))
        for d in $devdirs; do
            [ "$d" = "$first" ] && continue
            if ! cmp -s "$f" "$d/$b"; then
                echo "  $b: $first and $d differ -- the family is not deterministic on device" >&2
                drift=$((drift + 1)); break
            fi
        done
    done
    [ "$drift" = 0 ] && echo "  all $total position(s) identical across $repeats device run(s)" \
                     || { echo "npu-kernel-bitx: $drift of $total position(s) differ between repeats" >&2; bad=$((bad + drift)); }
fi
[ "$n" -gt 0 ] || { echo "npu-kernel-bitx: no logits were dumped -- the run proved nothing" >&2; exit 1; }
if [ "$bad" = 0 ]; then echo "npu-kernel-bitx: $n/$n prompt(s) bit-exact against the host integer arm"; exit 0; fi
echo "npu-kernel-bitx: $bad of $n prompt(s) differ from the host" >&2; exit 1
