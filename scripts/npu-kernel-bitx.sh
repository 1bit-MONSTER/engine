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
# OWNERSHIP, not presence. `[ -e ~/NPU-TOKEN ]` is true whenever ANY lane holds the device, so the old check let this
# script start a serve inside someone else's ticket while telling itself it was "inside an npu-hold" (engine#2434's
# family; measured 2026-11-14 on a sibling runner that did exactly that for 40 s). npu-hold/gpu-hold now hand the child
# the line they wrote into the token, so compare against that. A caller that holds the device in a shell created some
# other way can pass NPU_HOLD_TICKET explicitly; without it, refuse rather than guess.
_tok=${NPU_TOKEN_FILE:-$HOME/NPU-TOKEN}
# WHICH VARIABLE: the one the helper that actually runs this script exports. `npu-hold` execs `gpu-hold`, and gpu-hold hands
# its child GPU_HOLD_TICKET - nobody exports NPU_HOLD_TICKET. The first version of this check required NPU_HOLD_TICKET, so it
# refused EVERY run and told the reader "the helper exports NPU_HOLD_TICKET for you", which was false (engine#2447 review,
# measured: inside a real npu-hold the script refused with exactly that message). NPU_HOLD_TICKET stays as an override name
# for a caller that sets it by hand; GPU_HOLD_TICKET is the one that exists on the helper path.
_want=${NPU_HOLD_TICKET:-${GPU_HOLD_TICKET:-}}
if [ -z "$_want" ]; then
    echo "npu-kernel-bitx: refusing: no ticket line in the environment (NPU_HOLD_TICKET or GPU_HOLD_TICKET), so this run" >&2
    echo "  cannot prove it owns the device. Run it under: npu-hold -l <label> -t <min> -- $0 ..." >&2
    exit 2
fi
if [ ! -e "$_tok" ] || [ "$(cat "$_tok" 2>/dev/null)" != "$_want" ]; then
    echo "npu-kernel-bitx: refusing: the device ticket is not ours." >&2
    echo "  ours:   $_want" >&2
    echo "  in file: $(cat "$_tok" 2>/dev/null || echo '(absent)')" >&2
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
# engine#3196: logits alone cannot say whether two arms differ in the BODY or in the HEAD. On a container whose
# `lm_head` the device does not claim, the device falls through to an f32 host matvec while the integer arm takes its
# int8 ladder, so a logits verdict there is about two heads. The residual stream (`NPU_INFER_DUMP_DIR`, one
# `layer_<ll>.f32` per layer) is taken before the head, so the same runs can answer the body question too.
# (the per-run dump directory is created in serve(), which covers dev, dev1..R and host alike)

serve() {   # <dir> <port> [env ...]
    local dir=$1 port=$2; shift 2
    # engine#3196 review: the dump directory has to exist for the CHILD (`docs/npu.md`: "The directory must exist"),
    # and `$dir` is not always `$out/dev` -- `--repeats R` runs the device into `$out/dev1..R`, where nothing created
    # it, so the device arm dumped nothing and the arm that exists to measure determinism lost the comparison.
    # The residual comparison reads ONE device run (`$first`) and the host arm, so `--repeats R>1` arms the dump only
    # on the first: the repeats exist to test logit determinism, and that needs no residuals -- arming them there
    # would write ~1.1 GB per extra run on the 14B family for data nothing reads.
    local dumpargs=()
    if [ "${dump_layers:-1}" = 1 ]; then
        mkdir -p "$dir/layers"
        dumpargs=(NPU_INFER_DUMP_DIR="$dir/layers")
    fi
    # engine#3194: 60 s is a whisper-sized budget. A 14B host-GEMM arm spends minutes warming first
    # (`forward-total` pays the weight cache up front: "forward-serve: warming (weight cache; first request would
    # otherwise pay it)"), so a fixed 60 s kills it mid-warm and files the result as a failure. The budget is
    # configurable, defaults to something a 14B arm survives, and the failure message below always names it.
    local budget=${BITX_SERVE_WAIT_S:-300}
    case "$budget" in
        ''|*[!0-9]*|0) echo "  $dir: BITX_SERVE_WAIT_S must be a positive integer of seconds (got '$budget')" >&2; return 2 ;;
    esac
    env "$@" ${dumpargs[@]+"${dumpargs[@]}"} "$bin" forward-serve -m "$container" --kernels "$kernels" --prefix-slots 0 \
        -p "$port" --dump-logits "$dir" > "$dir.log" 2>&1 &
    local pid=$!
    local up=0
    for _ in $(seq 1 "$budget"); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { up=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if [ "$up" = 0 ]; then
        # engine#3194: the old line grepped only `init_elf failed|missing|error`, so the most common real reason --
        # a log ending in "warming (weight cache ...)" -- printed as `serve did not come up -- ` with nothing after it,
        # and the reader was left to guess. Print the matched error when there is one, and otherwise (or as well) the
        # child's own last line, plus the budget that expired.
        local why last
        why=$(grep -E 'init_elf failed|missing|error' "$dir.log" 2>/dev/null | tail -1)
        last=$(tail -1 "$dir.log" 2>/dev/null)
        echo "  $dir: serve did not come up within ${budget}s${why:+ -- $why}" >&2
        [ -n "$last" ] && [ "$last" != "$why" ] && echo "  $dir: last line of $dir.log: $last" >&2
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
    # only the first device run's residuals are compared, so only it pays the dump
    dump_layers=$([ "$r" = 1 ] && echo 1 || echo 0)
    serve "$d" $((18150 + r - 1)) || exit 1
done
dump_layers=1   # the host arm always dumps: it is the reference the comparison reads
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
# engine#3196 fix 2: WHERE the two arms part, on the same runs and before the head. ADVISORY on purpose -- the
# threshold at which a body counts as agreeing is a decision #3196 has not made yet, and turning it into a second
# gate criterion here would red the arm for a difference the issue is still measuring. What it does settle is the
# question the logits verdict cannot: a logits difference with the residuals inside the band is a HEAD difference.
residual_tool="$(dirname "$0")/../tools/ci/npu_residual_diff.py"
if [ ! -f "$residual_tool" ]; then
    echo "  residuals: not compared (no tools/ci/npu_residual_diff.py beside this script)"
elif ! command -v python3 >/dev/null 2>&1; then
    echo "  residuals: not compared (no python3)"
else
    # `$first` is the first device run's directory (`--repeats R` writes dev1..R, and `$out/dev` exists only at R=1).
    python3 "$residual_tool" "$first/layers" "$out/host/layers" 2>&1 | sed 's/^/  /'
    # The dumps are M x H x 4 B per layer per arm: 1366 tokens x 5120 f32 x 4 B = 28.0 MB per layer, so ~1.1 GB per
    # arm over 40 layers and ~2.2 GB for the pair. BITX_KEEP_DUMPS=0 drops them after the comparison, as the logits
    # dumps are dropped by the next run of the same container.
    if [ "${BITX_KEEP_DUMPS:-1}" = 0 ]; then
        rm -rf "$out"/dev*/layers "$out/host/layers"   # `dev*` also matches plain `dev` at --repeats 1
        echo "  residuals: layer dumps removed (BITX_KEEP_DUMPS=0)"
    fi
fi
if [ "$bad" = 0 ]; then echo "npu-kernel-bitx: $n/$n prompt(s) bit-exact against the host integer arm"; exit 0; fi
echo "npu-kernel-bitx: $bad of $n prompt(s) differ from the host" >&2; exit 1
