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
    # 4. an old enough end opens the gate again (the refusal above is about the gap, not about the file)
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '90 minutes ago' +%FT%TZ)" > "$tmp/w.log"
    out=$(WINDOWS_LOG=$tmp/w.log "$0" --dry-run -m 5 -l x -c true 2>&1); rc=$?
    chk "  ... and 90 minutes later the gap no longer refuses" "$(printf '%s' "$out" | grep -c 'not yet')" 0
    chk "  ... and the run is only PLANNED, no ticket taken" "$(printf '%s' "$out" | grep -c 'would take gpu-hold')" 1
    printf '\n%s self-test: %s\n' "$SELF" "$([ "$fail" = 0 ] && echo ok || echo "$fail failure(s)")"
    return "$([ "$fail" = 0 ] && echo 0 || echo 1)"
}
[ "${1:-}" = --self-test ] && { self_test; exit $?; }

LABEL=""; ISSUE=""; MINS=15; CMD=""; NOW=${NOW:-0}; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  -n|--now) NOW=1; shift;;
  --dry-run) DRY=1; shift;;
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
    [ "$end_s" -gt 0 ] && [ "$gap" -lt 2700 ] && die "not yet: the last window ended $last_end ($(( gap / 60 )) min ago of 45 needed)"
  fi
fi

# a held ticket is not a refusal: gpu-hold is a FIFO queue, and queueing is how "one GPU job at a
# time" is honoured. --now keeps the refuse-instead-of-wait case.
tick=$(scripts/gpu-hold --status 2>/dev/null | head -1 || true)
if [ "$NOW" = 1 ]; then
  case "$tick" in ""|*NONE*|*none*) : ;; *) die "--now was asked for, and a ticket is held: $tick";; esac
fi

if [ "$DRY" = 1 ]; then
  printf '%s: would take gpu-hold -t %s -l %s, stop coder-llm, run: %s\n' "$SELF" "$MINS" "$LABEL" "$CMD"
  exit 0
fi
saw_up=0; systemctl --user is-active --quiet coder-llm && saw_up=1
[ -d "$(dirname "$LOG")" ] || mkdir -p "$(dirname "$LOG")"
printf '%s start lane=%s issue=%s label=%s mins=%s cmd=%s\n' "$(date -u +%FT%TZ)" "${LANE:-$SELF}" "$ISSUE" "$LABEL" "$MINS" "$CMD" >> "$LOG"
restore() {
  local rc=${1:-0} i code=000
  if [ "$saw_up" = 1 ]; then
    systemctl --user start coder-llm 2>/dev/null
    for i in $(seq 1 40); do
      code=$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8090/v1/models 2>/dev/null || true)
      [ "$code" = 200 ] && break; sleep 3
    done
    printf '%s end   lane=%s issue=%s label=%s coder-llm=%s :8090=%s rc=%s\n' "$(date -u +%FT%TZ)" \
      "${LANE:-$SELF}" "$ISSUE" "$LABEL" "$(systemctl --user is-active coder-llm)" "$code" "$rc" >> "$LOG"
    [ "$code" = 200 ] || printf '%s: WARNING coder-llm restarted but :8090 answers %s\n' "$SELF" "$code" >&2
  else
    printf '%s end   lane=%s issue=%s label=%s coder-llm=was-not-running :8090=n/a rc=%s\n' "$(date -u +%FT%TZ)" \
      "${LANE:-$SELF}" "$ISSUE" "$LABEL" "$rc" >> "$LOG"
  fi
}
trap 'restore $?' EXIT
scripts/gpu-hold -t "$MINS" -l "$LABEL" -- bash -c '
  if systemctl --user is-active --quiet coder-llm; then systemctl --user stop coder-llm; fi
  holder=$(sed -n "s/.*pid \([0-9]*\).*/\1/p" <<<"$(scripts/gpu-hold --status 2>/dev/null | head -1)" | head -1)
  [ -n "$holder" ] && export GPU_HOLD_TICKET="${GPU_HOLD_TICKET:-$holder}"
  '"$CMD"'
'
