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
# hw-gate keeps a failing row's server log (engine#2500, #2513) under $HW_GATE_STORE/failed and names it in the
# failure line. The name must be unique per failure: CI sets HW_GATE_OUT=$RUNNER_TEMP/hw-gate, so a name built from
# basename "$OUT" was the constant "hw-gate-<model>.log" and every failing run on every PR overwrote the last one's
# evidence. These arms run the CI shape (same model, same run, same second), and check that a failed copy prints no
# path. A recording fake gpu-hold is first on PATH, so the internal mode is also proven to take no GPU ticket.
# No GPU, no model, runs in CI.
#
# The gate's phase check must match the phase FIELD, not a prefix of it, and only the phases that actually write
# `model <name>` rows may be satisfied by one (review of engine#2523):
#
#   * `npu` is a prefix of `npuserve`, so `grep -qE "^npu"` read an `npuserve` row as proof that `npu` had run;
#   * `placement` writes `placement <name>`, not `model <name>`, so grouping it with kat|kld|needle let any model row
#     satisfy a requested placement phase -- the same silent pass, one name over.
#
# The loop is EXTRACTED FROM THE SCRIPT and run, so this pins the shipped code rather than a copy of it. A copy would
# pass while the real check drifted.
# engine#2542: attribute the amdxdna IO_PAGE_FAULT bursts by sampling WHICH PROCESS HOLDS /dev/accel/accel0 open, and
# correlating each fault with the samples around it.
#
# Read-only by construction: it scans /proc for open-file symlinks and reads journald. It does not open the accel node,
# does not talk to XRT, never touches :18097, and needs no sudo (so no `lsof` loops). Because it takes no device action it
# needs no npu-hold ticket -- but it also must never *cause* a fault, which is why every read is /proc or the journal.
#
# Two loops in one unit:
#   sample   every SAMPLE_S seconds: the PIDs holding the accel node, with cmdline, cgroup scope and the npu-hold ticket
#            (from ~/NPU-TOKEN) as they stood at that moment
#   watch    journalctl -k -f, and for each IO_PAGE_FAULT line, the samples within +-WINDOW_S seconds, written as one
#            attribution record
#
# Inputs (env, all optional): ACCEL_SAMPLER_DIR, SAMPLE_S (2), WINDOW_S (5), MAX_BYTES (50 MB), ACCEL_NODE.
set -u
DIR=${ACCEL_SAMPLER_DIR:-$HOME/.cache/accel-sampler}
SAMPLE_S=${SAMPLE_S:-2}
FULL_SCAN_S=${FULL_SCAN_S:-30}   # a full /proc sweep every 2 s cost 62% of a core (measured); holders are long-lived
                                 # servers, so re-checking the known ones each sample is enough between sweeps
WINDOW_S=${WINDOW_S:-5}
MAX_BYTES=${MAX_BYTES:-52428800}   # 50 MB, rotate rather than grow without bound
NODE=${ACCEL_NODE:-/dev/accel/accel0}
mkdir -p "$DIR" || exit 1
SAMP=$DIR/samples.log
ATTR=$DIR/attributed.log
mkdir -p "$DIR"

rotate() {  # $1 = file: hard cap, one previous kept
  local f=$1 sz
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$sz" -gt "$MAX_BYTES" ]; then
    mv -f "$f" "$f.1" 2>/dev/null
    : > "$f"
  fi
}

ticket() {  # the npu-hold ticket that was live, if one is recorded
  for p in "$HOME/NPU-TOKEN" "$HOME/.npu-hold" "$HOME/lb/NPU-TOKEN"; do
    [ -f "$p" ] && { head -c 200 "$p" | tr '\n' ' '; return; }
  done
  echo "-"
}

holders() {  # print "pid|cmd|scope|fds" for every process holding $NODE open
  local pid fd tgt cmd scope n
  for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    [ -d "/proc/$pid/fd" ] || continue                 # kernel threads: no fd table, skip cheaply
    n=0
    for fd in /proc/"$pid"/fd/*; do
      tgt=$(readlink "$fd" 2>/dev/null) || continue
      [ "$tgt" = "$NODE" ] && n=$((n + 1))
    done
    [ "$n" -eq 0 ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)
    scope=$(sed -n 's/^0:://p' "/proc/$pid/cgroup" 2>/dev/null | head -1)
    [ -z "$scope" ] && scope=$(awk -F/ '/^0::/{print $NF}' "/proc/$pid/cgroup" 2>/dev/null)
    printf '%s|%s|%s|%s\n' "$pid" "${cmd:--}" "${scope:--}" "$n"
  done
}

KNOWN=$DIR/known.pids
sample_once() {  # one line per (moment, holder); "none" when nothing holds the node
  local ts seen=0 now last=0
  ts=$(date +%s)
  now=$ts
  [ -f "$KNOWN" ] && last=$(cat "$KNOWN.last" 2>/dev/null || echo 0)
  if [ $((now - last)) -ge "$FULL_SCAN_S" ]; then
    holders | cut -d'|' -f1 | sort -u > "$KNOWN.new"
    mv -f "$KNOWN.new" "$KNOWN"; echo "$now" > "$KNOWN.last"
  fi
  # only the processes known to hold the node: count their accel fds (cheap) and record them
  while read -r pid; do
    [ -z "$pid" ] && continue
    local n=0 fd
    for fd in /proc/"$pid"/fd/*; do
      [ "$(readlink "$fd" 2>/dev/null)" = "$NODE" ] && n=$((n + 1))
    done
    [ "$n" -eq 0 ] && continue      # it closed it; the next full sweep drops it from KNOWN
    seen=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ts" "$pid" "$n" \
      "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)" \
      "$(awk -F/ '/^0::/{print $NF}' "/proc/$pid/cgroup" 2>/dev/null)" \
      "$(ticket)" >> "$SAMP"
  done < "$KNOWN" 2>/dev/null
  if [ "$seen" = 0 ]; then
    printf '%s\t-\t-\t-\t-\t%s\n' "$ts" "$(ticket)" >> "$SAMP"
  fi
  rotate "$SAMP"
}

correlate() {  # $1 lo $2 hi -> "pid:cmd:scope" for every sample in [lo,hi], deduplicated
  awk -F'\t' -v lo="$1" -v hi="$2" '$1 >= lo && $1 <= hi' "$SAMP" 2>/dev/null | cut -f2,4,5,6 | sort -u | tr '\n' ';'
}

selftest() {  # prove the correlation before an event needs it: the window, the fields, and the miss case
  local d fail=0 out
  d=$(mktemp -d "$DIR/selftest.XXXXXX")   # never /tmp: this lane's rule, and tmpfs pressure has bitten the box
  printf '1000\t111\t1\t/serve A\t/scope/a\tticket1\n1010\t222\t2\t/serve B\t/scope/b\tticket2\n' > "$d/samples.log"
  SAMP=$d/samples.log
  out=$(correlate 1005 1015)                       # +-5 s around a fault at 1010
  case "$out" in *222*) ;; *) echo "FAIL: the holder at the fault instant is missing"; fail=1 ;; esac
  # the ticket is the field the whole issue is about -- a record without it cannot say which ticket was live. This arm
  # is here because a refactor dropped exactly that field and the other arms stayed green (review of engine#2591).
  case "$out" in *ticket2*) ;; *) echo "FAIL: the npu-hold ticket was dropped from the record"; fail=1 ;; esac
  case "$out" in *111*) echo "FAIL: a sample outside the window was included"; fail=1 ;; esac
  out=$(correlate 2000 2010)
  [ -z "$out" ] || { echo "FAIL: a window with no samples produced '$out'"; fail=1; }
  # a "none" sample (nobody holding the node) must be reported as such, not as a holder
  printf '1000\t-\t-\t-\t-\t-\n' > "$d/samples.log"
  out=$(correlate 995 1005)
  case "$out" in *'-:-:-'*|*'-'*) ;; *) echo "FAIL: a no-holder sample was not passed through"; fail=1 ;; esac
  # rotation: a 24 h run WILL hit it, so it gets an arm rather than a hand test
  MAX_BYTES=10; printf 'line one is long enough to exceed the cap\n' > "$d/rot.log"
  rotate "$d/rot.log"
  [ -f "$d/rot.log.1" ] || { echo "FAIL: rotation did not keep a previous file"; fail=1; }
  [ "$(stat -c %s "$d/rot.log")" -eq 0 ] || { echo "FAIL: the live file was not truncated"; fail=1; }
  rm -rf "$d"
  [ "$fail" = 0 ] && echo "accel-sampler self-test: ok (window, fields+ticket, empty window, no-holder sample, rotation)"
  return "$fail"
}

watch_faults() {  # correlate each fault with the samples around it
  local line ts lo hi hits
  journalctl -k -f -o short-unix 2>/dev/null | grep --line-buffered 'IO_PAGE_FAULT' |
  while IFS= read -r line; do
    ts=${line%% *}                       # short-unix: "<epoch.frac> host kernel: ..."
    ts=${ts%%.*}
    lo=$((ts - WINDOW_S)); hi=$((ts + WINDOW_S))
    hits=$(correlate "$lo" "$hi")
    printf '%s\t%s\t%s\n' "$ts" "${hits:--}" "$(printf '%s' "$line" | cut -c1-200)" >> "$ATTR"
    rotate "$ATTR"
  done
}

summary() {  # what the run observed, so a 24 h expiry is not silence
  local out=$DIR/summary.txt n attr
  if [ -f "$SAMP" ]; then n=$(wc -l < "$SAMP"); else n=0; fi
  if [ -f "$ATTR" ]; then attr=$(wc -l < "$ATTR"); else attr=0; fi
  {
    echo "# accel sampler summary $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "samples: $n   attributed faults: $attr"
    echo
    echo "## holders seen (pid, cmd, scope, ticket, sample count)"
    awk -F'\t' '$2 != "-" {print $2"\t"$4"\t"$5"\t"$6}' "$SAMP" 2>/dev/null | sort | uniq -c | sort -rn | head -20
    echo
    if [ "$attr" -gt 0 ]; then
      echo "## attribution records"
      cut -c1-300 "$ATTR" | tail -20
    else
      echo "## no IO_PAGE_FAULT was seen in this window"
      echo "The sampler cannot attribute a burst that did not happen. What it does establish: the holders above were"
      echo "live for the whole window, so if a burst had occurred, its record would name them."
    fi
  } > "$out"
  echo "summary written: $out"
}

case "${1:-both}" in
  sample) sample_once ;;
  watch)  watch_faults ;;
  both)
    # the fault reader first, so no fault is missed between start and the first sample
    watch_faults &
    W=$!
    trap 'kill "$W" 2>/dev/null; summary; exit 0' TERM INT
    trap 'kill "$W" 2>/dev/null' EXIT
    LAST=0
    while :; do
      # gap record: a stall is data. Measured need: the loop lost 93 s at the peak of a fault storm, and those minutes became
      # "no holder" attribution records -- silence that looked like a finding. Now the gap is written (and kept) instead.
      NOW=$(date +%s)
      if [ "$LAST" -gt 0 ] && [ $((NOW - LAST)) -gt $((SAMPLE_S * 3)) ]; then
        printf '%s\tGAP\t%s\t-	-	-\n' "$NOW" "$((NOW - LAST))" >> "$SAMP"
      fi
      LAST=$NOW
      sample_once
      # refresh the summary every ~15 min (450 samples at 2 s): a reboot kills the unit and SIGTERM never arrives, so a
      # summary written only on exit can be hours stale exactly when it is needed
      SAMP_N=$(( ${SAMP_N:-0} + 1 ))
      [ $(( SAMP_N % 450 )) -eq 0 ] && summary
      sleep "$SAMPLE_S"
    done
    ;;
  holders) holders ;;
  selftest|--self-test) selftest ;;
  summary) summary ;;
  *) echo "usage: $0 [both|sample|watch|holders|summary|selftest]" >&2; exit 2 ;;
esac
