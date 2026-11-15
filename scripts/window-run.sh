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
#
# Exit status (what a caller should do with it):
#   0     the window ran, the job succeeded, and coder-llm is back (:8090 answered 200)
#   N     the job's own rc, whenever the job failed (a restart failure never masks it)
#   2     refused before anything happened: bad arguments, inside the 45-min gap without -w, no gpu-hold / setsid
#   69    the job succeeded but THIS window stopped coder-llm and :8090 did not answer 200 after the restart: the fleet
#         LLM is down because of this run; look now (a window that found coder-llm already stopped never returns 69,
#         its end line says coder-llm=was-not-running and it does not start it)
#   75    the ticket came, but another lane's window had ended less than 45 min before: nothing stopped, nothing logged;
#         queue again later (-w waits only before the ticket, not after a refusal)
#   130 / 143   INT / TERM: the job group was killed, coder-llm restored if this window stopped it
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
    mkdir -p "${WR_TMPDIR:-$HOME/.cache}"; tmp=$(mktemp -d "${WR_TMPDIR:-$HOME/.cache}/window-run-test.XXXXXX"); trap 'rm -rf "$tmp"' RETURN
    # 0. THE RESTART PROBE ASKS FOR A COMPLETION. No device is touched here, so this arm reads the artifact the
    # gate ships. /v1/models is a static registry listing that answers 200 while the weights load (app/serve.cpp
    # :1909) -- a false green; /v1/health read 503 twice while the service was serving (2026-11-15 07:09Z,
    # 07:15Z) -- a false red. A 1-token completion is the only probe that cannot be right while the LLM is
    # unusable, and it is what the caller needs anyway.
    local me; me=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/${BASH_SOURCE[0]##*/}   # WR_SELF is exported later, not yet here
    # The patterns carry cu[r]l rather than curl so that this arm's OWN text cannot satisfy them: a "/v1/health" count
    # that includes the line asserting it is an arm that can pass for the wrong reason.
    # The pattern spells the digit as a class so that this arm's own text cannot satisfy it.
    # The patterns spell one letter as a class so that the arm's own text cannot satisfy it, and they assert the
    # URL the probe hits -- a payload-only arm was verified vacuous: pointing the probe back at /v1/models left
    # every arm green (mutation, 2026-11-15).
    chk "the restart probe hits /v1/chat/completions" "$(grep -c '8090/v1/chat/completio[n]s' "$me")" 1
    chk "  ... and never the static model list" "$(grep -c '8090/v1/model[s]' "$me")" 0
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
    # 3g. the restart verdict, four rows (engine#2496: a ticketed coder-llm is `active` while :8090 is silent, waiting
    #     for demand). The third row is the one that was missing: asleep-on-purpose is not a failed restart.
    chk "--restart-rc 200 0 active waiting -> 0"        "$("$0" --restart-rc 200 0 1 1)" 0
    chk "--restart-rc 200 0 active not-waiting -> 0"    "$("$0" --restart-rc 200 0 1 0)" 0
    chk "--restart-rc 000 0 active waiting -> 0 (asleep on purpose)" "$("$0" --restart-rc 000 0 1 1)" 0
    chk "--restart-rc 000 0 active NOT waiting -> 69 (a real failed restart)" "$("$0" --restart-rc 000 0 1 0)" 69
    chk "--restart-rc 000 0 inactive waiting -> 69 (inactive is never healthy)" "$("$0" --restart-rc 000 0 0 1)" 69
    chk "--restart-rc 000 3 active waiting -> 3 (the job's failure is still kept)" "$("$0" --restart-rc 000 3 1 1)" 3
    chk "--restart-rc 503 0 active waiting -> 0 (a 503 with a waiting unit is still asleep)" "$("$0" --restart-rc 503 0 1 1)" 0
    # 3h. the journal reader itself (review on this PR): the crash-on-load sequence must NOT read as "waiting for demand",
    #     even though a stale "waiting for demand" from the PREVIOUS invocation is still in the tail.
    cat > "$tmp/j-crash.log" <<'JLOG'
coder-llm-ticketed[100] waiting for demand (demand file touched within 1800s)
Stopping coder-llm.service...
Started coder-llm.service...
coder-llm-ticketed[200] demand: queueing for the GPU (gpu-hold -l coder-llm -t 25 -w 900)
coder-llm-ticketed[201] window open: server pid 202
E hrx: device lost
coder-llm-ticketed[201] server exited by itself, rc 1
coder-llm-ticketed[200] window ended rc 1
JLOG
    chk "a crash-on-load after a stale 'waiting for demand' is NOT asleep" "$("$0" --asleep "$tmp/j-crash.log")" 0
    chk "  ... so the restart verdict for it is 69" "$("$0" --restart-rc 000 0 1 "$("$0" --asleep "$tmp/j-crash.log")")" 69
    printf '%s\n' 'coder-llm-ticketed[300] waiting for demand (demand file touched within 1800s)' > "$tmp/j-wait.log"
    chk "the newest wrapper line saying 'waiting for demand' IS asleep" "$("$0" --asleep "$tmp/j-wait.log")" 1
    : > "$tmp/j-empty.log"
    chk "an empty journal is not asleep (the safe direction)" "$("$0" --asleep "$tmp/j-empty.log")" 0
    printf '%s\n' 'no wrapper lines here at all' > "$tmp/j-none.log"
    chk "a journal with no wrapper line at all is not asleep" "$("$0" --asleep "$tmp/j-none.log")" 0
    # 3f. the ticket's child really re-checks the gap: another window ends while this one is queued (the fake gpu-hold
    #     appends that end line, then runs the real child) -> rc 75, coder-llm never stopped, nothing else logged.
    #     systemctl is a recording fake first on PATH, so even a regression here cannot stop the real coder-llm.
    mkdir -p "$tmp/fakebin"
    printf '#!/bin/bash\necho "$*" >> %s/systemctl.calls\n[ "$*" = "--user is-active --quiet coder-llm" ] && exit 0\nexit 0\n' "$tmp" > "$tmp/fakebin/systemctl"
    printf '#!/bin/bash\n[ "${1:-}" = --status ] && exit 0\nwhile [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done; shift\nprintf "%%s end   lane=other issue=0 label=other coder-llm=active :8090=200 rc=0\\n" "$(date -u +%%FT%%TZ)" >> "$WR_LOG"\nexec "$@"\n' > "$tmp/fakebin/gpu-hold"
    chmod +x "$tmp/fakebin/systemctl" "$tmp/fakebin/gpu-hold"; : > "$tmp/r.log"; : > "$tmp/systemctl.calls"
    PATH=$tmp/fakebin:$PATH GPU_HOLD=$tmp/fakebin/gpu-hold WINDOWS_LOG=$tmp/r.log "$0" -m 1 -l x -c true >/dev/null 2>&1
    chk "a window that ended while queued refuses the ticket (75)" "$?" 75
    chk "  ... coder-llm is never stopped" "$(grep -c 'stop coder-llm' "$tmp/systemctl.calls")" 0
    chk "  ... and the log holds only the other lane's end line" "$(wc -l < "$tmp/r.log")" 1
    # 3i. a TICKETED coder-llm is never stopped by a window (it is queued, not resident): the fake systemctl reports an
    #     ExecStart that runs coder-llm-ticketed and an active unit; the window runs for real through a pass-through
    #     gpu-hold, and must make no stop call and say why in its start line.
    mkdir -p "$tmp/tbin"
    printf '#!/bin/bash\necho "$*" >> %s/t.calls\ncase "$*" in *"show coder-llm -p ExecStart"*) echo "{ path=/x/coder-llm-ticketed ; argv[]=/x/coder-llm-ticketed -- serve }";; esac\nexit 0\n' "$tmp" > "$tmp/tbin/systemctl"
    printf '#!/bin/bash\n[ "${1:-}" = --status ] && exit 0\nwhile [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done; shift\nexec "$@"\n' > "$tmp/tbin/gpu-hold"
    chmod +x "$tmp/tbin/systemctl" "$tmp/tbin/gpu-hold"; : > "$tmp/t.calls"
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '90 minutes ago' +%FT%TZ)" > "$tmp/t.log"
    PATH=$tmp/tbin:$PATH GPU_HOLD=$tmp/tbin/gpu-hold WINDOWS_LOG=$tmp/t.log "$0" -m 1 -l x -c true >/dev/null 2>&1
    chk "a ticketed coder-llm is never stopped by a window" "$(grep -c 'stop coder-llm' "$tmp/t.calls")" 0
    chk "  ... and the window's start line says so" "$(grep -c 'coder-llm ticketed: not stopped' "$tmp/t.log")" 1
    # 3j. a restart deferred to the NEXT lane's window is not a failure (engine: measured 2026-11-15 11:30Z/12:54Z)
    : > "$tmp/t.calls"
    # a NON-ticketed unit that is active: the window stops it (creating WR_STOPPED), which is what makes the restore
    # path -- and therefore the deferral -- reachable at all.
    # its own directory, so the ticketed arm's fake systemctl (which asserts the opposite) stays in place
    mkdir -p "$tmp/tbin2"
    printf '#!/bin/bash\necho "$*" >> %s/t.calls\ncase "$*" in *"show coder-llm -p ExecStart"*) echo "{ path=/x/plain-1bit ; argv[]= }";; *"is-active --quiet coder-llm"*) exit 0;; *"is-active coder-llm"*) echo active;; esac\n' "$tmp" > "$tmp/tbin2/systemctl"
    chmod +x "$tmp/tbin2/systemctl"
    printf 'gpu-hold cl-other 999999\n' > "$tmp/token"
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '90 minutes ago' +%FT%TZ)" > "$tmp/t.log"
    PATH=$tmp/tbin2:$tmp/tbin:$PATH GPU_HOLD=$tmp/tbin/gpu-hold GPU_TOKEN_FILE=$tmp/token WINDOWS_LOG=$tmp/t.log \
      "$0" -m 1 -l x -c true >/dev/null 2>&1
    chk "a GPU token held by the next lane defers the restart" "$(grep -c 'restart=deferred(gpu held by 999999)' "$tmp/t.log")" 1
    chk "  ... and the window still starts the unit, so it is not left down" "$(grep -c 'start coder-llm' "$tmp/t.calls")" 1
    chk "  ... and the deferral is not reported as a failed restart" "$(grep -c 'restart=FAILED' "$tmp/t.log")" 0
    # the loading-vs-wedged check needs a live child growing in CPU, which a self-test cannot fake without testing its
    # own fake; pin the reader instead so it cannot be deleted silently (the same technique the probe arm uses).
    # 3k. engine#2496's contract: a ticketed coder-llm that is QUEUED is not a failed restart
    : > "$tmp/t.calls"; mkdir -p "$tmp/q"
    # a LIVE pid: the arm used to write pid=999999, which does not exist -- so it demonstrated a STALE entry and
    # asserted "queued" as correct (review of this PR). $$ is this test's own live shell.
    printf 'label=coder-llm\npid=%s\n' "$$" > "$tmp/q/0179.coder-llm"
    printf '%s end   lane=t issue=0 label=t coder-llm=active :8090=200 rc=0\n' "$(date -u -d '90 minutes ago' +%FT%TZ)" > "$tmp/t.log"
    PATH=$tmp/tbin2:$tmp/tbin:$PATH GPU_HOLD=$tmp/tbin/gpu-hold GPU_TOKEN_FILE=$tmp/none-token GPU_QUEUE_DIR=$tmp/q WR_HEALTH_URL=http://127.0.0.1:9/x \
      WINDOWS_LOG=$tmp/t.log "$0" -m 1 -l x -c true >/dev/null 2>&1
    chk "a queued coder-llm is recorded as queued, not failed" "$(grep -c 'restart=queued(WAITING for a GPU ticket when this window ended, not restored)' "$tmp/t.log")" 1
    chk "  ... and it is not reported as a failed restart" "$(grep -c 'restart=FAILED' "$tmp/t.log")" 0
    # a DEAD pid is a stale entry, which gpu-hold drops -- it must not license "queued" forever
    : > "$tmp/t.log"; printf 'label=coder-llm\npid=999999\n' > "$tmp/q/0179.coder-llm"
    PATH=$tmp/tbin2:$tmp/tbin:$PATH GPU_HOLD=$tmp/tbin/gpu-hold GPU_TOKEN_FILE=$tmp/none-token GPU_QUEUE_DIR=$tmp/q WR_HEALTH_URL=http://127.0.0.1:9/x \
      WINDOWS_LOG=$tmp/t.log "$0" -m 1 -l x -c true >/dev/null 2>&1
    chk "a STALE queue entry (dead pid) does not count as queued" "$(grep -c 'restart=queued' "$tmp/t.log")" 0
    chk "the poll reads the largest child's CPU, not the idle embed child" "$(grep -c 'sort -k2 -n | tai[l] -1' "$me")" 1
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
# The restart's verdict, once the port did not answer (engine#2496's contract, and the reason final_rc is not enough):
#
#   code=200                        -> the job's rc            (the model is back)
#   the job itself failed           -> the job's rc            (a restart failure never masks it)
#   the unit is ACTIVE and its own journal says "waiting for demand"
#                                   -> the job's rc            (the model is asleep ON PURPOSE; this run did not break it)
#   anything else                   -> 69                       (we stopped it and it did not come back)
#
# The middle case is why this exists: a ticketed coder-llm (`scripts/coder-llm-ticketed`, engine#2496) waits for the
# demand file BEFORE it takes a ticket, so with stale demand it never serves and never exits - `is-active` stays
# `active` while `:8090` is silent. Without this, every window that ran more than CODER_LLM_DEMAND_TTL_MIN after the
# last PR-Agent call would return 69 and print "the fleet LLM is down because of this run", which is false and is the
# alert that gets ignored by the third occurrence.
# "Waiting for demand" is only the answer if it is what THIS invocation is doing NOW. Two things make that true, and my
# first version got both wrong (review on this PR):
#   * scope to the current invocation -- `journalctl --user _SYSTEMD_INVOCATION_ID=<id>` -- because the same unit's
#     PREVIOUS run also logged "waiting for demand", and the last 60 lines can easily still hold it;
#   * read the LAST wrapper state line, not any matching one. A crash on load right after this window logs
#     `demand: queueing …` -> `window open: server pid …` -> `device lost` -> `server exited by itself, rc 1`, all after
#     that stale "waiting for demand", and calling that "asleep on purpose" is exactly the failure 69 exists for.
# The journal is read through a seam so a test can feed a fixture: coder_llm_asleep [journal-file].
coder_llm_asleep() {   # [journal file] -> prints 1 when the unit is waiting for demand on THIS invocation, else 0
    local jid src
    if [ -n "${1:-}" ]; then
        src=$(cat "$1" 2>/dev/null)
    else
        jid=$(systemctl --user show -p InvocationID --value coder-llm 2>/dev/null)
        [ -n "$jid" ] || { echo 0; return; }
        src=$(journalctl --user --no-pager -q _SYSTEMD_INVOCATION_ID="$jid" 2>/dev/null)
    fi
    [ -n "$src" ] || { echo 0; return; }
    # the wrapper's own lines, newest last; only the newest one is a statement about now
    local last
    last=$(printf '%s\n' "$src" | grep -F 'coder-llm-ticketed[' | tail -1)
    case "$last" in
        *'waiting for demand'*) echo 1 ;;
        *)                      echo 0 ;;
    esac
}

restart_rc() {   # <code> <job rc> <unit active: 0|1> <journal says waiting for demand: 0|1>
    if [ "$1" = 200 ] || [ "$3" != 1 ] || [ "$4" != 1 ]; then final_rc "$1" "$2"; else echo "$2"; fi
}
if [ "${1:-}" = "--final-rc" ]; then final_rc "${2:-}" "${3:-0}"; exit 0; fi
if [ "${1:-}" = "--restart-rc" ]; then restart_rc "${2:-000}" "${3:-0}" "${4:-0}" "${5:-0}"; exit 0; fi
if [ "${1:-}" = "--asleep" ]; then coder_llm_asleep "${2:-}"; exit 0; fi
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
# setsid (util-linux) puts the job in its own process group, which is what lets a TERM kill the whole tree
command -v setsid >/dev/null 2>&1 || die "setsid not found (util-linux): the job could not be killed as one group; refusing"
if [ "$DRY" = 1 ]; then
  printf '%s: would take gpu-hold -t %s -l %s, stop coder-llm, run: %s\n' "$SELF" "$MINS" "$LABEL" "$CMD"
  exit 0
fi
[ -d "$(dirname "$LOG")" ] || mkdir -p "$(dirname "$LOG")"
# Markers the ticket's child writes: STARTED once it logged the start line, STOPPED once it stopped coder-llm. A run that
# never got that far (refused, killed in the queue, gpu-hold missing) logs nothing and restarts nothing.
mkdir -p "${WR_TMPDIR:-$HOME/.cache}"   # /tmp is tmpfs on strixhalo and the lane rules keep new files off it
WR_DIR=$(mktemp -d "${WR_TMPDIR:-$HOME/.cache}/window-run.XXXXXX"); export WR_STARTED=$WR_DIR/started WR_STOPPED=$WR_DIR/stopped
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
    # A DEFERRED RESTART IS NOT A FAILED ONE (measured 2026-11-15 11:30Z and 12:54Z): when the next lane's window
    # already holds the GPU token, coder-llm cannot load -- it needs the GTT that window is using -- so polling
    # here burns minutes and, worse, delays this window's END line, which is what every other lane reads as
    # "still holding". Say so in one line and let the unit come up when that window ends.
    _tok=${GPU_TOKEN_FILE:-$HOME/GPU-TOKEN}
    if [ -e "$_tok" ]; then
      _held=$(awk '{print $NF}' "$_tok" 2>/dev/null | head -1)
      printf '%s end   lane=%s issue=%s label=%s coder-llm=%s :8090=deferred restart=deferred(gpu held by %s) rc=%s\n' \
        "$(date -u +%FT%TZ)" "$WR_LANE" "$ISSUE" "$LABEL" "$(systemctl --user is-active coder-llm)" "${_held:-?}" "$rc" >> "$LOG"
      printf '%s: coder-llm restarts when the next lane releases the GPU (holder %s); not a failed restart\n' "$SELF" "${_held:-?}" >&2
      rm -rf "$WR_DIR"; exit "$rc"
    fi
    # QUEUED IS NOT DEAD EITHER (engine#2496's contract, live 2026-11-15): once `coder-llm` runs inside
    # `coder-llm-ticketed`, `systemctl start` means "queue for a GPU ticket" -- so with lanes ahead of it the unit
    # is `active` and `:8090` stays empty for minutes. A restart-then-fail there is a false signal on EVERY window,
    # which is how lanes learn to ignore the failure. Measured: the unit ran `coder-llm-ticketed`, `coder-llm` was
    # 4th in ~/GPU-QUEUE, and this restore reported failure for a window whose job had succeeded.
    # LIVENESS, not existence (review of this PR, and the same mistake #2547 made with the token file): an entry
    # left behind by a killed process outlives it, and `gpu-hold` itself calls such an entry stale and DROPS it
    # (stale(), nineteen lines in: "no pid, or whose pid is gone"). Matching on the label alone makes a dead
    # ticket licence a `queued` verdict for every later window -- a permanent false calm, which is worse than the
    # false alarm this PR removes.
    coder_llm_queued() {
        local q=${GPU_QUEUE_DIR:-$HOME/GPU-QUEUE} f pid
        [ -d "$q" ] || return 1
        for f in "$q"/*; do
            case "${f##*/}" in .*) continue ;; esac
            grep -qx 'label=coder-llm' "$f" 2>/dev/null || continue
            pid=$(sed -n 's/^pid=\([0-9][0-9]*\)$/\1/p' "$f" 2>/dev/null | head -1)
            [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && return 0
        done
        return 1
    }
    # LOADING IS NOT WEDGED (same windows): a child loading a 30B answers 503 while its CPU time advances;
    # restarting it there throws the load away. Only a child whose CPU time has not moved for ~30 s is dead.
    # The LARGEST llama-server is the one loading -- the embed service's idle child is ~15 MB.
    cpu_of() { ps -o time=,rss= -C llama-server 2>/dev/null | sort -k2 -n | tail -1 | awk '{print $1}'; }
    prev=""; froze=0
    _poll_s=${WR_POLL_S:-3}
    for i in $(seq 1 "${WR_POLL_MAX:-100}"); do
      # /v1/health, NOT /v1/models: /v1/models is a static registry listing (app/serve.cpp:1909) that
      # answers 200 from the moment the process binds, so it reports success while the weights are still
      # loading and every PR-agent check is still failing. /v1/health is 503 until `ready` and 503 when a
      # child is dead (app/serve.cpp:1892). Measured 2026-11-15: /v1/health went 503 -> 200 across a real
      # restart while /v1/models was already 200. The engine's own child poll uses /health first for the
      # same reason (app/serve.cpp:694).
      # A TINY COMPLETION, not a status endpoint. /v1/health is a readiness signal, but on this payload it also
      # read 503 while the service was serving (measured 2026-11-15 07:09Z and 07:15Z, then 200 on five consecutive
      # samples with a completion in between), so a status-only probe trades a false GREEN for a false RED -- and a
      # false 69 is how a lane learns to ignore 69. A 1-token completion is what the caller needs, so it is what the
      # check asks for: slower than a status code, and the only probe that cannot be right while the LLM is unusable.
      code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' \
                 -d '{"model":"qwen3-coder","messages":[{"role":"user","content":"ok"}],"max_tokens":1,"temperature":0}' \
                 "${WR_HEALTH_URL:-http://127.0.0.1:8090/v1/chat/completions}" 2>/dev/null || true)
      [ "$code" = 200 ] && break
      _cpu=$(cpu_of)
      if [ -n "$_cpu" ] && [ "$_cpu" = "$prev" ]; then froze=$((froze + 1)); else froze=0; fi
      prev=$_cpu
      [ "$froze" -ge 10 ] && break      # ~30 s without CPU progress: a wedge, not a load
      sleep "$_poll_s"
    done
    _state=$(systemctl --user is-active coder-llm 2>/dev/null)
    if [ "$code" != 200 ] && [ "$_state" = active ] && coder_llm_queued; then
        printf '%s end   lane=%s issue=%s label=%s coder-llm=active :8090=queued restart=queued(WAITING for a GPU ticket when this window ended, not restored) rc=%s\n' \
          "$(date -u +%FT%TZ)" "$WR_LANE" "$ISSUE" "$LABEL" "$rc" >> "$LOG"
        printf '%s: coder-llm is queued for a GPU ticket (engine#2496), not dead: :8090 answers when it gets one\n' "$SELF" >&2
        rm -rf "$WR_DIR"; exit "$rc"
    fi
    _waiting=0
    if [ "$code" != 200 ] && [ "$_state" = active ]; then
        _waiting=$(coder_llm_asleep ${WR_JOURNAL_FILE:+"$WR_JOURNAL_FILE"})
    fi
    rc=$(restart_rc "$code" "$rc" "$([ "$_state" = active ] && echo 1 || echo 0)" "$_waiting")
    # The end line keeps its shape for the gap parser (`Z end ` prefix) and names which case it took, so a reader does
    # not have to infer it from a port number: asleep-no-demand is not a failure, restart=FAILED is.
    printf '%s end   lane=%s issue=%s label=%s coder-llm=%s :8090=%s rc=%s%s\n' "$(date -u +%FT%TZ)" \
      "$WR_LANE" "$ISSUE" "$LABEL" "$_state" "$code" "$rc" \
      "$([ "$code" = 200 ] && echo '' || { [ "$_waiting" = 1 ] && echo ' asleep-no-demand' || echo ' restart=FAILED'; })" >> "$LOG"
    if [ "$code" != 200 ] && [ "$_waiting" != 1 ]; then
        printf '%s: ERROR coder-llm was restarted but :8090 answers %s: exit %s\n' "$SELF" "$code" "$rc" >&2
    elif [ "$_waiting" = 1 ]; then
        printf '%s: coder-llm is waiting for demand (ticketed, :8090 silent by design): keeping rc %s\n' "$SELF" "$rc" >&2
    fi
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
  # a TICKETED coder-llm (engine#2496) is not resident outside its own gpu-hold windows, and this window holds the
  # token, so it cannot be serving now; stopping it would only throw away its queued ticket (seen 2026-11-15 10:00Z:
  # a window stopped it and its place in the queue was lost). Stop the old always-on unit only.
  note=""
  if systemctl --user show coder-llm -p ExecStart --value 2>/dev/null | grep -q coder-llm-ticketed; then
    note=" (coder-llm ticketed: not stopped)"
  elif systemctl --user is-active --quiet coder-llm; then systemctl --user stop coder-llm && touch "$WR_STOPPED"; fi
  printf "%s start lane=%s issue=%s label=%s mins=%s cmd=%s%s\n" "$(date -u +%FT%TZ)" "$WR_LANE" "$WR_ISSUE" "$WR_LABEL" "$WR_MINS" "$WR_CMD" "$note" >> "$WR_LOG"
  touch "$WR_STARTED"
  holder=$(sed -n "s/.*pid \([0-9]*\).*/\1/p" <<<"$("$GPU_HOLD" --status 2>/dev/null | head -1)" | head -1)
  [ -n "$holder" ] && export GPU_HOLD_TICKET="${GPU_HOLD_TICKET:-$holder}"
  '"$CMD"'
' &
JOB=$!
wait "$JOB"
