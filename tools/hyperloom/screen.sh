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
#
# screen.sh -- measure a batch of Hyperloom lever sets in ONE box-hold ticket (the operator's side of the screen cache in
# box_ticket.sh). The box queue is 30-40 tickets deep; one ticket per Hyperloom round would take days per model, so the
# operator screens the seed grid (baseline + variants) back to back with the SAME box side a round would ship, and each
# measurement is filed under $SCREEN_DIR/<hl_screen_key>/ for the rounds to read.
#
#   screen.sh WORKLOAD SCREEN_DIR "COMMON" "SET" ["SET"...]
#     WORKLOAD    inkling-loom | custom (the directory under tools/hyperloom whose box side runs)
#     COMMON      pins every round of the run carries, K=V,K=V (e.g. HL_CARD_NAME=qwen38-27b-3e43d41d,HL_RUNS=3)
#     SET         the lever set of one measurement, K=V,K=V; "-" = the baseline (COMMON only)
#   env: HL_BOX, HL_BOX_DIR, HL_SSH (as box_ticket.sh), SCREEN_BUDGET (seconds of measurement in the ticket, 2400),
#        SCREEN_EST (seconds one set is expected to take, 420; a set starts only if it fits). No BOX_HOLD_WAIT: since
#        box-hold v6 (28 Nov) lane tickets never expire, and box_ticket.sh's HL_HOLD_WAIT default (3600) made screens
#        give up after 1 h in a deep queue
#
# Values are restricted to [A-Za-z0-9._:/+-]; the workload's own entrypoint still validates every lever when a round
# reads the measurement (an invalid one is refused there, exactly as without the cache).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WL=${1:?workload}; SD=${2:?screen dir}; COMMON=${3?common pins}; shift 3
[ $# -gt 0 ] || { echo "screen: no lever sets" >&2; exit 2; }
case $WL in
    inkling-loom) FILES=(box_inkling.sh gauntlet.py); BOXSCRIPT=box_inkling.sh; EXCL=card ;;
    custom) FILES=(box_bench.sh benchmark.py); BOXSCRIPT=box_bench.sh; EXCL= ;;
    *) echo "screen: unknown workload $WL" >&2; exit 2 ;;
esac
HL_BOX=${HL_BOX:-strixhalo} HL_BOX_DIR=${HL_BOX_DIR:-/data/dev/cl-hyperloom} HL_SSH=${HL_SSH:-ssh}
BUDGET=${SCREEN_BUDGET:-2400} EST=${SCREEN_EST:-420}
[ "$BUDGET" -le 2580 ] || { echo "screen: SCREEN_BUDGET $BUDGET is above the 45-minute ticket cap" >&2; exit 2; }
SAFE='^[A-Z][A-Z0-9_]*=[A-Za-z0-9._:/+-]+$'
REV=$(cd "$HERE/$WL" && cat "${FILES[@]}" | sha256sum | cut -c1-12)
RDIR=$HL_BOX_DIR/hl/$REV
T=hl-screen-$(date -u +%Y%m%dT%H%M%SZ)-$$
B=$HL_BOX_DIR/out/$T
mkdir -p "$SD" || exit 1

kvs() { local s=$1 out=() kv; [ "$s" = - ] && s=; IFS=, read -ra out <<< "$s"; for kv in "${out[@]}"; do [ -n "$kv" ] && printf '%s\n' "$kv"; done; }
# hl_screen_key from box_ticket.sh itself, so the screen and the rounds can never compute different keys
SCRATCH=$(mktemp -d); trap 'rm -rf "$SCRATCH"' EXIT
RESULT_DIR=$SCRATCH LOG_TAG=screen . "$HERE/box_ticket.sh" || exit 1

# the box-side runner: one line per set (index + env words); a set starts only while it fits the budget
runner="t0=\$(date +%s); i=0"
declare -a SETS KEYS
n=0
for s in "$@"; do
    mapfile -t L < <(kvs "$COMMON"; kvs "$s")
    for kv in "${L[@]}"; do [[ $kv =~ $SAFE ]] || { echo "screen: refused $kv" >&2; exit 2; }; done
    SETS[n]="${L[*]}"; KEYS[n]=$(hl_screen_key "$REV" "${L[@]}")
    runner+="; if [ \$(( \$(date +%s) - t0 + $EST )) -le $BUDGET ]; then sudo -n timeout -k 30 $(( BUDGET + 120 )) env ${L[*]} HL_OUT_OWNER=\$(id -un) bash $RDIR/$BOXSCRIPT $B/s$n > $B/s$n.log 2>&1; echo \"set $n rc=\$? \$(( \$(date +%s) - t0 ))s\"; else echo \"set $n skipped (budget)\"; fi"
    n=$((n + 1))
done

echo "[screen] $T: $n sets, workload $WL rev $REV, budget ${BUDGET}s"
tar -C "$HERE/$WL" -cf - "${FILES[@]}" | $HL_SSH "$HL_BOX" "mkdir -p $RDIR $B && tar -C $RDIR -xf -" || exit 1
# The box-side runner logs to a FILE on the box, never to the ssh channel: on 28 Nov a dropped ssh client (the driver
# died with its session) closed the runner's stdout, and its first echo after set 0 killed it with SIGPIPE -- the
# remaining sets never ran. Now the ticket finishes whatever happens to this side, and the log is read back at the end.
$HL_SSH -o ServerAliveInterval=30 "$HL_BOX" "BOX_HOLD_WHO='cl-hyperloom $T' box-hold gpu bash -c '$runner' > $B/runner.log 2>&1 < /dev/null; cat $B/runner.log" \
    < /dev/null | tee "$SD/$T.log"
for i in $(seq 0 $((n - 1))); do
    d=$SD/${KEYS[i]}
    rm -rf "$d.tmp"; mkdir -p "$d.tmp"
    $HL_SSH "$HL_BOX" "tar -C $B/s$i -cf - ${EXCL:+--exclude=$EXCL} . 2>/dev/null" < /dev/null | tar -C "$d.tmp" -xf - 2>/dev/null
    if [ -f "$d.tmp/status" ] && grep -qx ok "$d.tmp/status"; then
        printf '{"ticket": "%s", "rev": "%s", "levers": "%s", "measured_at": "%s"}\n' "$T" "$REV" "${SETS[i]}" \
            "$(date -u +%FT%TZ)" > "$d.tmp/screen.json"
        rm -rf "$d"; mv "$d.tmp" "$d"
        echo "[screen] set $i (${SETS[i]}) -> $d"
    else
        echo "[screen] set $i (${SETS[i]}): not filed ($(cat "$d.tmp/status" 2>/dev/null || echo no status))"
        rm -rf "$d.tmp"
    fi
done
