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
# window-run.sh -- run one GPU job as a fleet memory window, with the housekeeping the rules ask for,
# so a lane does not have to remember it. Every lane wrote its own version of this (six of them in
# ~/lb by 11-14), and the failure mode is always the same one: a window that ends with coder-llm still
# stopped, which fails the PR-agent check on every open PR for as long as it lasts.
#
#   scripts/window-run.sh -l LABEL [-i ISSUE] [-m MINUTES] -c 'command ...'
#   scripts/window-run.sh --self-test
#
# What it enforces, and why each one is here rather than in a lane's memory:
#
#   -m 1..15        coder-llm stays down at most 15 minutes, so the window cannot starve the fleet LLM
#   the 45-min gap  at least 45 minutes between one window's END and the next one's start
#   gpu-hold        the job runs INSIDE a ticket, and queues behind other lanes rather than racing them
#   --now           refuses instead of queueing, for the cases where waiting is the wrong answer
#   coder-llm       stopped only after the ticket is ours, restarted by a trap on ANY exit, and :8090
#                   is polled until it answers (the restart is verified, not asserted)
#   WINDOWS.log     one start line and one end line, carrying lane, issue, label, minutes, the exact
#                   command and the rc -- the command is what makes a window auditable afterwards
#   the child       gets its own ticket pid in GPU_HOLD_TICKET, so a measurement script can prove
#                   ownership instead of guessing it from the global token file
set -uo pipefail

SELF=${0##*/}
LOG=${WINDOWS_LOG:-$HOME/briefs/WINDOWS.log}
# gpu-hold next to this script, not "scripts/gpu-hold" relative to wherever the caller stands: run from another directory,
# the relative path failed AFTER the start line was written, so a window that never took a ticket logged a start/end pair
# and moved every lane's 45-minute gap (11-15 03:31Z).
GPU_HOLD=${GPU_HOLD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu-hold}
export GPU_HOLD

# epoch of the newest END line in the log (0 when there is none); read fresh every time it is asked, because another
# lane can open and close a window while this one waits (11-15 04:14-04:17Z, inside my -w wait)
last_end_s() {
  local e
  [ -f "$LOG" ] || { echo 0; return; }
  e=$(sed -n 's/^\([0-9T:-]*Z\) end .*/\1/p' "$LOG" | sort | tail -1)
  [ -n "$e" ] && date -u -d "$e" +%s 2>/dev/null || echo 0
}
die() { printf '%s: %s\n' "$SELF" "$*" >&2; exit 2; }

self_test() {
    local fail=0
    chk() { local label=$1 got=$2 want=$3; if [ "$got" = "$want" ]; then echo "ok   $label"; else echo "FAIL $label (got $got, want $want)"; fail=$((fail+1)); fi; }
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' RETURN
    # 1. a duration outside 1..15 is refused, with the reason
    out=$(WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 0 -l x -c true 2>&1); rc=$?
    chk "a zero-minute window is refused" "$rc" 2
    chk "  ... and says why" "$(printf '%s' "$out" | grep -c 'fails the PR-agent check')" 1
    out=$(WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 16 -l x -c true 2>&1); chk "a sixteen-minute window is refused" "$?" 2
    # 2. no label / no command
    WINDOWS_LOG=$tmp/w.log "$0" --dry-run -c true >/dev/null 2>&1; chk "no -l is refused" "$?" 2
    WINDOWS_LOG=$tmp/w.log "$0" --dry-run -l x >/dev/null 2>&1; chk "no -c is refused" "$?" 2
    # 3. a window inside the 45-minute gap is refused, and writes nothing to the log
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u +%FT%TZ)" > "$tmp/w.log"
    WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 5 -l x -c true >/dev/null 2>&1; chk "a window inside the gap is refused" "$?" 2
    chk "  ... and a refusal is not written to the log" "$(wc -l < "$tmp/w.log")" 1
    # 3b. no gpu-hold where the script expects it: refused before the start line, so nothing is logged
    : > "$tmp/g.log"
    GPU_HOLD=$tmp/none WINDOWS_LOG=$tmp/g.log "$0" --dry-run -m 5 -l x -c true >/dev/null 2>&1; chk "a missing gpu-hold is refused" "$?" 2
    chk "  ... and writes nothing to the log" "$(wc -l < "$tmp/g.log")" 0
    # 3c. --gap-ok reads the log fresh: a window that ended 5 minutes ago refuses, one 90 minutes ago does not
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '5 minutes ago' +%FT%TZ)" > "$tmp/k.log"
    WINDOWS_LOG=$tmp/k.log "$0" --gap-ok >/dev/null 2>&1; chk "--gap-ok refuses 5 min after a window" "$?" 1
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '90 minutes ago' +%FT%TZ)" > "$tmp/k.log"
    WINDOWS_LOG=$tmp/k.log "$0" --gap-ok >/dev/null 2>&1; chk "--gap-ok passes 90 min after one" "$?" 0
    # 3d. TERM while the ticket is still queued: the queued job dies with the runner and nothing is logged
    #     (11-15 04:24Z: a TERM to the runner left coder-llm down and a llama-server outliving its ticket)
    printf '#!/bin/bash\n[ "${1:-}" = --status ] && exit 0\necho $$ > %s/fake.pid; exec sleep 60\n' "$tmp" > "$tmp/fake-gpu-hold"
    chmod +x "$tmp/fake-gpu-hold"; : > "$tmp/q.log"
    GPU_HOLD=$tmp/fake-gpu-hold WINDOWS_LOG=$tmp/q.log "$0" -m 1 -l x -c true >/dev/null 2>&1 & local runner=$!
    for _ in $(seq 1 50); do [ -s "$tmp/fake.pid" ] && break; sleep 0.1; done
    kill -TERM "$runner" 2>/dev/null; wait "$runner" 2>/dev/null; local rrc=$?
    chk "a TERM'd runner exits 143" "$rrc" 143
    kill -0 "$(cat "$tmp/fake.pid" 2>/dev/null)" 2>/dev/null; chk "  ... and its queued job is gone" "$?" 1
    chk "  ... and nothing is logged" "$(wc -l < "$tmp/q.log")" 0
    # 3e. coder-llm not back after the restart fails the run (69) unless the job already failed
    chk "--final-rc 200 0 -> 0" "$("$0" --final-rc 200 0)" 0
    chk "--final-rc 000 0 -> 69 (coder-llm did not come back)" "$("$0" --final-rc 000 0)" 69
    chk "--final-rc 503 3 -> 3 (the job's own failure is kept)" "$("$0" --final-rc 503 3)" 3
    # 4. an old enough end opens the gate again (the refusal above is about the gap, not about the file)
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '90 minutes ago' +%FT%TZ)" > "$tmp/w.log"
    out=$(WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 5 -l x -c true 2>&1); rc=$?
    chk "  ... and 90 minutes later the gap no longer refuses" "$(printf '%s' "$out" | grep -c 'not yet')" 0
    chk "  ... and the run is only PLANNED, no ticket taken" "$(printf '%s' "$out" | grep -c 'would take gpu-hold')" 1
    # 5. the gap check says when it could not check: a stale log, a log with no END line, and no log at
    #    all. Twenty lane scripts in ~/lb stop coder-llm without writing this file, so silence here is
    #    not evidence that no window happened - each branch has to be able to say its own name.
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '20 hours ago' +%FT%TZ)" > "$tmp/w.log"
    out=$(WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 5 -l x -c true 2>&1)
    chk "a stale log is called out, not silently trusted" "$(printf '%s' "$out" | grep -c 'invisible to this check')" 1
    : > "$tmp/w.log"
    out=$(WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 5 -l x -c true 2>&1)
    chk "a log with no END line says the gap is UNCHECKED" "$(printf '%s' "$out" | grep -c 'UNCHECKED')" 1
    rm -f "$tmp/w.log"
    out=$(WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 5 -l x -c true 2>&1)
    chk "and no log at all says so too" "$(printf '%s' "$out" | grep -c 'no .*UNCHECKED\|UNCHECKED')" 1
    chk "  ... while a fresh window still plans normally" "$(printf '%s' "$out" | grep -c 'would take gpu-hold')" 1

    # 6. --wait holds the lane instead of refusing, and it refuses to *shorten* the rule: a window 5 min
    #    inside the gap still has ~40 min to go, which the message must say. (The loop itself is not
    #    exercised here - an arm that slept 40 minutes is not a self-test.)
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '40 minutes ago' +%FT%TZ)" > "$tmp/w.log"
    out=$(WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 5 -l x -c true 2>&1)
    chk "5 min inside the gap: a plain run still refuses" "$(printf '%s' "$out" | grep -c 'not yet')" 1
    chk "  ... and names -w as the way to wait" "$(printf '%s' "$out" | grep -c '\-w waits for it')" 1

    printf '\n%s self-test: %s\n' "$SELF" "$([ "$fail" = 0 ] && echo ok || echo "$fail failure(s)")"
    return "$([ "$fail" = 0 ] && echo 0 || echo 1)"
}
[ "${1:-}" = --self-test ] && { self_test; exit $?; }

# The run's final status once coder-llm was restarted: the job's rc when :8090 answers 200, else 69 (EX_UNAVAILABLE) for
# a job that itself succeeded. A window that leaves the fleet LLM down is a failed window, not a WARNING line.
final_rc() { if [ "$1" = 200 ] || [ "$2" != 0 ]; then echo "$2"; else echo 69; fi; }
if [ "${1:-}" = "--final-rc" ]; then final_rc "${2:-}" "${3:-0}"; exit 0; fi
# --gap-ok: rc 0 when the 45-minute gap is open now, rc 1 (with the reason) when it is not. The ticket's child runs it
# right before it stops coder-llm, so a window another lane ran while this one queued still counts.
if [ "${1:-}" = "--gap-ok" ]; then
  e=$(last_end_s); g=$(( $(date -u +%s) - e ))
  if [ "$e" -gt 0 ] && [ "$g" -lt 2700 ]; then
    printf '%s: the gap closed while queued: a window ended %s min ago (45 needed); coder-llm left running\n' "$SELF" "$(( g / 60 ))" >&2
    exit 1
  fi
  exit 0
fi
LABEL=""; ISSUE=""; MINS=15; CMD=""; NOW=${NOW:-0}; DRY=0; WAIT=0
while [ $# -gt 0 ]; do case "$1" in
  -n|--now) NOW=1; shift;;
  --dry-run) DRY=1; shift;;
  -w|--wait) WAIT=1; shift;;
  -l) LABEL=${2:-}; shift 2;;
  -i) ISSUE=${2:-}; shift 2;;
  -m) MINS=${2:-}; shift 2;;
  -c) CMD=${2:-}; shift 2;;
  -h|--help) sed -n '17,40p' "$0"; exit 0;;
  *) die "unknown argument $1";; esac; done
[ -n "$LABEL" ] && [ -n "$CMD" ] || die "need -l LABEL and -c 'command'"
case "$MINS" in ''|*[!0-9]*) die "-m takes whole minutes";; esac
[ "$MINS" -ge 1 ] && [ "$MINS" -le 15 ] || die "-m $MINS is outside 1..15: every minute coder-llm is down fails the PR-agent check on every PR"

# the 45-minute gap, from the log's END lines only (a start line is not a window end)
if [ -f "$LOG" ]; then
  last_end=$(sed -n 's/^\([0-9T:-]*Z\) end .*/\1/p' "$LOG" | sort | tail -1)
  if [ -n "$last_end" ]; then
    end_s=$(date -u -d "$last_end" +%s 2>/dev/null || echo 0)
    now_s=$(date -u +%s)
    gap=$(( now_s - end_s ))
    if [ "$end_s" -gt 0 ] && [ "$gap" -lt 2700 ]; then
      left=$(( (2700 - gap + 59) / 60 ))
      if [ "$WAIT" = 1 ] && [ "$DRY" != 1 ]; then
        # --wait: hold the lane here instead of making it come back. The rule is unchanged - this only
        # stops *running*, it never shortens the 45 minutes - and it is the same service gpu-hold does
        # for a ticket, which is why a lane that waits out a window by hand is doing the queue's job.
        printf '%s: waiting %s min for the 45-minute gap (last window ended %s)\n' "$SELF" "$left" "$last_end" >&2
        while :; do
          end_s=$(last_end_s); gap=$(( $(date -u +%s) - end_s ))
          { [ "$end_s" -eq 0 ] || [ "$gap" -ge 2700 ]; } && break
          sleep 30
        done
        printf '%s: the gap is open after waiting %s min\n' "$SELF" "$left" >&2
      else
        die "not yet: the last window ended $last_end ($(( gap / 60 )) min ago of 45 needed; -w waits for it)"
      fi
    fi
    # Honesty about what this check actually covered: on 2026-11-15 twenty lane scripts in ~/lb were
    # found to stop and start coder-llm WITHOUT writing this file, so the gap above is only as good as
    # the runs that log. A stale (or empty) log is not evidence that no window happened.
    if [ "$end_s" -gt 0 ] && [ "$gap" -gt 14400 ]; then
      printf '%s: note: the last logged window ended %s (%s h ago); windows that do not write %s are invisible to this check\n' \
        "$SELF" "$last_end" "$(( gap / 3600 ))" "$LOG" >&2
    fi
  else
    printf '%s: note: %s exists but has no END line, so the 45-minute gap is UNCHECKED here\n' "$SELF" "$LOG" >&2
  fi
else
  printf '%s: note: no %s, so the 45-minute gap is UNCHECKED here\n' "$SELF" "$LOG" >&2
fi

# a held ticket is not a refusal: gpu-hold is a FIFO queue, and queueing is how "one GPU job at a
# time" is honoured. --now keeps the refuse-instead-of-wait case.
tick=$("$GPU_HOLD" --status 2>/dev/null | head -1 || true)
if [ "$NOW" = 1 ]; then
  case "$tick" in ""|*NONE*|*none*) : ;; *) die "--now was asked for, and a ticket is held: $tick";; esac
fi

[ -x "$GPU_HOLD" ] || die "no executable gpu-hold at $GPU_HOLD (set GPU_HOLD): refusing before anything is logged"
if [ "$DRY" = 1 ]; then
  printf '%s: would take gpu-hold -t %s -l %s, stop coder-llm, run: %s\n' "$SELF" "$MINS" "$LABEL" "$CMD"
  exit 0
fi
[ -d "$(dirname "$LOG")" ] || mkdir -p "$(dirname "$LOG")"
# Markers the ticket's child writes: STARTED once it logged the start line, STOPPED once it stopped coder-llm. A run that
# never got that far (refused, killed in the queue, gpu-hold missing) logs nothing and restarts nothing.
WR_DIR=$(mktemp -d); export WR_STARTED=$WR_DIR/started WR_STOPPED=$WR_DIR/stopped
export WR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}" WR_LOG=$LOG WR_LANE=${LANE:-$SELF} \
       WR_ISSUE=$ISSUE WR_LABEL=$LABEL WR_MINS=$MINS WR_CMD=$CMD WINDOWS_LOG=$LOG
JOB=""
restore() {
  local rc=${1:-0} i code=000
  trap - EXIT TERM INT
  # the job runs in its own process group: a signal to this runner stops the whole tree, not just gpu-hold
  # (killing gpu-hold alone leaves its child running, 11-12; 11-15 04:24Z a llama-server outlived its ticket)
  if [ -n "$JOB" ] && kill -0 "$JOB" 2>/dev/null; then
    kill -TERM -- "-$JOB" 2>/dev/null; sleep 3; kill -KILL -- "-$JOB" 2>/dev/null
  fi
  if [ ! -e "$WR_STARTED" ]; then
    printf '%s: no window was opened (rc %s): coder-llm untouched, nothing logged\n' "$SELF" "$rc" >&2
    rm -rf "$WR_DIR"; exit "$rc"
  fi
  if [ -e "$WR_STOPPED" ]; then
    systemctl --user start coder-llm 2>/dev/null
    for i in $(seq 1 40); do
      code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/v1/models 2>/dev/null || true)
      [ "$code" = 200 ] && break; sleep 3
    done
    rc=$(final_rc "$code" "$rc")
    printf '%s end   lane=%s issue=%s label=%s coder-llm=%s :8090=%s rc=%s%s\n' "$(date -u +%FT%TZ)" \
      "$WR_LANE" "$ISSUE" "$LABEL" "$(systemctl --user is-active coder-llm)" "$code" "$rc" \
      "$([ "$code" = 200 ] || echo ' restart=FAILED')" >> "$LOG"
    [ "$code" = 200 ] || printf '%s: ERROR coder-llm was restarted but :8090 answers %s: exit %s\n' "$SELF" "$code" "$rc" >&2
  else
    printf '%s end   lane=%s issue=%s label=%s coder-llm=was-not-running :8090=n/a rc=%s\n' "$(date -u +%FT%TZ)" \
      "$WR_LANE" "$ISSUE" "$LABEL" "$rc" >> "$LOG"
  fi
  rm -rf "$WR_DIR"; exit "$rc"
}
trap 'restore $?' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
setsid "$GPU_HOLD" -t "$MINS" -l "$LABEL" -- bash -c '
  "$WR_SELF" --gap-ok || exit 75
  if systemctl --user is-active --quiet coder-llm; then systemctl --user stop coder-llm && touch "$WR_STOPPED"; fi
  printf "%s start lane=%s issue=%s label=%s mins=%s cmd=%s\n" "$(date -u +%FT%TZ)" "$WR_LANE" "$WR_ISSUE" "$WR_LABEL" "$WR_MINS" "$WR_CMD" >> "$WR_LOG"
  touch "$WR_STARTED"
  holder=$(sed -n "s/.*pid \([0-9]*\).*/\1/p" <<<"$("$GPU_HOLD" --status 2>/dev/null | head -1)" | head -1)
  [ -n "$holder" ] && export GPU_HOLD_TICKET="${GPU_HOLD_TICKET:-$holder}"
  '"$CMD"'
' &
JOB=$!
wait "$JOB"
