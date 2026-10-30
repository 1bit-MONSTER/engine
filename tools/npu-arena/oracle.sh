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
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# oracle.sh -- score ONE submission for ONE phase.
#
#   oracle.sh <GU|D|QKV|O|Z> <submission-dir> [--iters N] [--no-leader]
#
# A submission dir is a copy of tasks/<phase>/kernel/ with your edits; it must
# contain build.sh (writes out/<PHASE>.elf) and be the leader's code plus your
# change (rule 3 of the arena).  The oracle:
#
#   1. refuses to start unless the NPU is free (no forward-serve, device lock
#      uncontended) -- the box is shared, and a score measured next to someone
#      else's stream is not a score;
#   2. copies the submission into a scratch dir and rebuilds it FROM SOURCE with
#      the pinned toolchain (TOOLCHAIN.lock), so a stale or hand-copied ELF
#      cannot be scored;
#   3. replays the current leader first, then the submission, under gpu-hold;
#   4. checks BYTE-EXACT int32 against the frozen harness on seeds 1..3 and
#      times 5 launches (median) on seed 1;
#   5. prints ACCEPT / REJECT and, on accept, the line to paste into
#      LEADERBOARD.md.
set -u
ARENA=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$ARENA/lib/env.sh"
TASK=${1:?usage: oracle.sh TASK SUBMISSION_DIR [--iters N] [--no-leader]}
SUB=${2:?usage: oracle.sh TASK SUBMISSION_DIR [--iters N] [--no-leader]}
shift 2
ITERS=5
NO_LEADER=0
while [ $# -gt 0 ]; do
    case "$1" in
        --iters) ITERS=$2; shift 2 ;;
        --no-leader) NO_LEADER=1; shift ;;
        *) echo "oracle: unknown arg $1" >&2; exit 2 ;;
    esac
done
TD=$ARENA/tasks/$TASK
[ -d "$TD/harness" ] || { echo "oracle: no harness for $TASK"; exit 2; }
arena_require "$ARENA/runner/runner" "$TD/harness/SHA256SUMS" || exit 2
K=$(sed -n 's/^K //p' "$TD/harness/geometry.txt")
N=$(sed -n 's/^N //p' "$TD/harness/geometry.txt")
[ -n "$K" ] && [ -n "$N" ] || { echo "oracle: no geometry"; exit 2; }
[ -f "$SUB/build.sh" ] || { echo "oracle: $SUB has no build.sh"; exit 2; }

# ---- 1. the device must be ours ------------------------------------------
# Only an NPU forward-serve is disqualifying; a plain `1bit serve` (a GGUF HTTP
# server on another lane's GPU/CPU) does not touch the array, so warn instead of
# refusing -- otherwise another lane's LLM server could block the arena forever.
if pgrep -f "[1]bit f""orward-serve" >/dev/null; then
    echo "oracle: an NPU forward-serve is running ($(pgrep -af '[1]bit f''orward-serve' | head -1))"
    echo "oracle: stop it first (contract: ask the orchestrator) -- not scoring a shared device"
    exit 9
fi
if pgrep -f "[1]bit serve" >/dev/null; then
    echo "oracle: note: another lane's plain serve is up; it does not use the NPU, the device check decides"
fi
if [ -n "$(fuser /dev/accel/accel0 2>/dev/null)" ]; then
    echo "oracle: /dev/accel/accel0 is held by $(fuser /dev/accel/accel0 2>/dev/null)"; exit 9
fi
exec 7> "$DEVICE_LOCK"
flock -w 900 7 || { echo "oracle: device lock busy"; exit 9; }

# ---- harness integrity ----------------------------------------------------
( cd "$TD/harness" && sha256sum -c --quiet SHA256SUMS ) || { echo "oracle: harness modified"; exit 3; }

# ---- 2. rebuild from source in a scratch copy ----------------------------
tc_id=$(arena_toolchain_id | tr '\n' ' ')
sha=$(cd "$SUB" && find . -type f -name '*.py' -o -type f -name '*.cc' -o -type f -name '*.sh' \
        -o -name GEOMETRY | sort | xargs sha256sum 2>/dev/null | sha256sum | cut -c1-16)
work=$ARENA/scratch/${TASK}-${sha}
rm -rf "$work"; mkdir -p "$work"
cp -a "$SUB"/. "$work"/
rm -rf "$work/out"
echo "== build $TASK submission=$SUB tree=$sha"
ARENA=$(cd "$ARENA" && pwd)
export ARENA
if ! ( cd "$work" && ~/lb/thermal-run.sh ./build.sh ) > "$work/build.log" 2>&1; then
    echo "oracle: REJECT (build failed)"; tail -5 "$work/build.log"; exit 4
fi
ELF=$work/out/$TASK.elf
[ -s "$ELF" ] || { echo "oracle: REJECT (build produced no out/$TASK.elf)"; exit 4; }
echo "== built $(basename "$ELF") $(stat -c%s "$ELF") B  [$tc_id]"

# ---- 3. leader replay, then the submission, on the device ----------------
# gpu-hold is the box-wide GPU fair FIFO.  If the caller already holds it (one
# batch of arena work under a single `gpu-hold ...` wrapper), nesting would put
# us behind ourselves, so ARENA_INSIDE_HOLD=1 runs the runner directly.
if [ "${ARENA_INSIDE_HOLD:-0}" = 1 ]; then
    RUN=()
else
    RUN=("$GPU_HOLD" -l "npu arena" -t 15 --)
fi
# A submission MAY move the weight bytes differently than the harness stores them:
# `layout.sh <in_B> <out_B> <K> <N>` (optional) writes its device-side weight buffer
# once, and `LAYOUT` declares `W_BYTES=<n>` for it.  The buffer is built outside the
# timed region -- the engine likewise packs weights once per process, not per token --
# and the score is those bytes over the launch time.  What it may NOT do is change
# the maths: the expected C is still A @ B of the semantic matrix, so a reordering or
# a repack must produce the identical int32 result.
layout_for() {   # layout_for <dir> <out-B> <seed> ; echoes the --w args or nothing
    local dir=$1 out=$2 seed=$3 decl
    [ -f "$dir/layout.sh" ] || return 0
    decl=$(sed -n 's/^W_BYTES=//p' "$dir/LAYOUT" 2>/dev/null | head -1)
    if ! bash "$dir/layout.sh" "$TD/harness/seed${seed}_B.bin" "$out" "$K" "$N" >/dev/null 2>&1; then
        echo "oracle: layout.sh failed for seed $seed" >&2; return 1
    fi
    echo "--w $out --w-bytes ${decl:-0}"
}
run_one() {   # run_one <elf> <label> <layout-dir> <seeds...>
    local elf=$1 label=$2 ldir=$3; shift 3
    local ok=1 s wargs out
    echo "== $label"
    for s in "$@"; do
        wargs=$(layout_for "$ldir" "$work/layout_s$s.B" "$s") || return 1
        out=$("${RUN[@]}" "$ARENA/runner/runner" --elf "$elf" --task "$TD" --seed "$s" \
                  --iters 2 $wargs) || ok=0
        echo "$out" | sed "s/^/   /"
        echo "$out" | grep -q "verdict=PASS" || ok=0     # every seed must be byte-exact
        rm -f "$work/layout_s$s.B"
    done
    wargs=$(layout_for "$ldir" "$work/layout_t.B" 1) || return 1
    out=$("${RUN[@]}" "$ARENA/runner/runner" --elf "$elf" --task "$TD" --seed 1 \
              --iters "$ITERS" $wargs) || ok=0
    echo "$out" | tee -a /tmp/arena-$TASK-$$.last | sed 's/^/   /'
    echo "$out" | grep -q "verdict=PASS" || ok=0
    return $((1 - ok))
}

LEADER_ELF=$ARENA/submissions/$TASK/leader.elf
if [ "$NO_LEADER" = 0 ] && [ -s "$LEADER_ELF" ]; then
    run_one "$LEADER_ELF" "leader (submissions/$TASK/leader.elf)" \
        "$ARENA/submissions/$TASK/leader-src" 1 2 3 || echo "   (leader FAILED its own gate!)"
fi
: > /tmp/arena-$TASK-$$.last    # never let a leader line be read as the submission's
run_one "$ELF" "submission tree=$sha" "$work" 1 2 3
rc=$?   # 0 only if every seed AND the timing run were byte-exact

line=$(grep '^RESULT' /tmp/arena-$TASK-$$.last 2>/dev/null | tail -1)
rm -f /tmp/arena-$TASK-$$.last
if [ -z "$line" ]; then echo "oracle: REJECT (no RESULT line)"; exit 5; fi
verdict=$(echo "$line" | sed -n 's/.*verdict=\([A-Z]*\).*/\1/p')
gbs=$(echo "$line" | sed -n 's/.*gbsB=\([0-9.]*\).*/\1/p')
med=$(echo "$line" | sed -n 's/.*med_ms=\([0-9.]*\).*/\1/p')
if [ "$verdict" != PASS ] || [ "$rc" != 0 ]; then
    echo "oracle: REJECT (not byte-exact on every seed + the timed run): $line"; exit 6
fi
echo "oracle: ACCEPT  task=$TASK tree=$sha gbs=$gbs med_ms=$med [$tc_id]"
echo "oracle: leaderboard row ->"
echo "| $(date -u +%Y-%m-%d) | \`$sha\` | $gbs | $med | $tc_id |"
[ -f "$ARENA/submissions/$TASK/leader.elf" ] || {
    echo "oracle: first accepted entry -- promote it:"
    echo "   mkdir -p $ARENA/submissions/$TASK && cp $ELF $ARENA/submissions/$TASK/leader.elf"
    echo "   cp -a $work $ARENA/submissions/$TASK/leader-src"
}
exit 0
