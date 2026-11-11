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
# ci-runner-health.sh - say whether the self-hosted GitHub runners are RUNNING, and refuse to be silent about a dead one.
#
# Why this exists (engine#1504): both runners died on an OOM kill at 08:37 on 2026-11-08 and stayed dead for ~3h45m with
# `Result=oom-kill` and `NRestarts=0`. Every check that needs the box then read as **pending** - never green, never red -
# which is indistinguishable, at a glance, from "queued behind a busy runner". The issue's own words for the gap are
# "nothing notices", and this is the thing that notices: one line per runner, a printed verdict, and a non-zero exit.
#
# It reads systemd's view, which is the only place `Result=oom-kill` survives (the journal has it too, but the unit property
# is what a restart policy would consult). It never restarts anything: `systemctl start` needs root, and whether to restart a
# runner that OOM-killed twice in two minutes is a decision, not a check (engine#1504 measured that: the fourth restart
# survived only because the GPU ticket was empty).
#
#   scripts/ci-runner-health.sh                 # every actions.runner.* unit
#   scripts/ci-runner-health.sh <unit> [...]    # named units, e.g. for a specific runner
#   scripts/ci-runner-health.sh --self-test     # the verdict logic, with systemctl stubbed
#   scripts/ci-runner-health.sh --runs          # the RUN-CONCLUSION mix: a cancelled check is neither
#
# Exit: 0 when every runner is active/running with a restart policy that survives an OOM kill; 1 when a runner is not
# accepting jobs NOW; 2 when there is nothing to inspect (no such units on this host, so a silent pass would be the defect);
# 3 when every runner is accepting jobs but a restart policy would leave one down after the next OOM kill. 1 and 3 are
# separate because they call for different things (engine#1504, owner 2026-11-11): a dead runner blocks a release, since
# its checks would queue for ever, while a policy line is a box configuration change the owner applies
# (docs/ci-runners.md "The two drop-ins"), so scripts/weekly.sh warns on 3 and goes on. The policy lines print as WARN.
set -uo pipefail

# <restart> <oompolicy> <memorymax> -> one verdict line. The state checks above answer "is it alive", which is what the
# issue named; this answers "will it come back", which is what actually kept both runners dead for 3h45m (engine#1504).
# A unit with Restart=no/OOMPolicy=stop PASSes every state check right up to the OOM kill that ends it for good, so the
# configuration has to be read, not inferred from ActiveState. Expected values are docs/ci-runners.md's, and the reason
# for each is one line there (runsvc.sh exits 0 after a kill, so on-failure never fires; OOMPolicy=stop lets an OOM kill
# of a JOB stop the RUNNER).
policy_verdict() {
    _rst=${1:-} _oom=${2:-} _mmax=${3:-}
    if [ -z "$_rst" ]; then
        printf 'FAIL no restart policy could be read for this runner, so whether it would come back after an OOM kill is unknown'
        return 0
    fi
    _bad=""
    # always, not on-failure: runsvc.sh reports success when it is killed, so on-failure never fires.
    [ "$_rst" = always ] || _bad="$_bad Restart=$_rst(want always);"
    [ "${_oom:-}" = continue ] || _bad="$_bad OOMPolicy=${_oom:-?}(want continue);"
    # an infinite ceiling means the cgroup never hits its own limit, so the box's OOM killer chooses - and the bench runner
    # is the unit whose jobs are memory-heavy by definition.
    [ -n "${_mmax:-}" ] && [ "$_mmax" != infinity ] || _bad="$_bad MemoryMax=${_mmax:-?}(want a ceiling);"
    case "$_bad" in
        "") ;;
        *Restart=*|*OOMPolicy=*) ;;
        # Restart and OOMPolicy are right and only the ceiling is missing: the unit does come back, so do not say it will not.
        *) printf 'FAIL the runner would come back after an OOM kill, but its cgroup has no ceiling:%s the box-wide OOM killer picks the victim, not the limit of this unit' "$_bad"
           return 0 ;;
    esac
    if [ -n "$_bad" ]; then
        printf 'FAIL the restart policy cannot restart this runner after an OOM kill:%s one OOM kill and the unit stops for good, while jobs read as pending' "$_bad"
        return 0
    fi
    printf 'PASS Restart=always, OOMPolicy=continue, MemoryMax=%s - an OOM kill costs a job, not the runner' "$_mmax"
}

# <state> <substate> <result> <nrestarts> -> one verdict line
runner_verdict() {
    _st=${1:-} _sub=${2:-} _res=${3:-} _nr=${4:-}
    if [ -z "$_st" ]; then
        printf 'FAIL no systemd state could be read for this runner, so whether it is alive is unknown'
        return 0
    fi
    if [ "$_st" = active ]; then
        printf 'PASS active/running, last result %s, %s restart(s) - the runner is accepting jobs' "${_res:-?}" "${_nr:-0}"
        return 0
    fi
    # `activating (auto-restart)` is the transient where a FLAPPING runner looks busy: it is not accepting jobs, but a naive
    # check sees a unit that systemd is already dealing with. Say which it is, and say what a repeat means (pi-os3's addition).
    if [ "$_st" = activating ] || [ "${_sub:-}" = auto-restart ]; then
        printf 'FAIL %s/%s - the runner is in its RESTART WINDOW and not accepting jobs (%s restarts so far); one line is transient, a log full of them is a flapping runner' "$_st" "${_sub:-?}" "${_nr:-0}"
        return 0
    fi
    case "$_res" in
        oom-kill) printf 'FAIL %s/%s after an OOM kill (%s restarts) - the kernel killed it and nothing restarted it; jobs read as pending, not as failed' "$_st" "$_sub" "${_nr:-0}" ;;
        *)        printf 'FAIL %s/%s with last result %s (%s restarts) - the runner is not accepting jobs' "$_st" "${_sub:-?}" "${_res:-?}" "${_nr:-0}" ;;
    esac
}

# engine#2077: a `cancelled` run is a THIRD state, and it is the one a merge gate silently reads as neither
# green nor a verdict about the diff. Measured on 2026-11-11: `gh run list --workflow=pr-agent.yml --event
# pull_request --limit 40` was 23 cancelled / 17 success, because `concurrency.cancel-in-progress: true` cancels a
# minutes-long review whenever the next push arrives. So the count is a health number like any other: this
# summarises a `conclusion` per line and fails when the cancelled share is too large to call the workflow healthy.
#
# The threshold is a knob, not a law: `--max-cancelled-pct` (default 25) is where a workflow stops being a signal
# about the diff. 0 is legal and means "a cancelled run is already too many".
runs_summary() {  # <max_pct> ; conclusions on stdin, one per line
    awk -v maxpct="${1:-25}" '
        { n++; c[$0]++ }
        END {
            if (n == 0) { print "FAIL no runs read - an empty list is not a pass"; exit 2 }
            out = ""
            for (k in c) out = out sprintf("%s=%d ", k, c[k])
            canc = ("cancelled" in c) ? c["cancelled"] : 0
            pct = 100.0 * canc / n
            # verdict first: a caller (and the self-test) reads the conclusion from line 1, and the counts follow
            if (pct > maxpct)
                printf "FAIL %d of %d runs cancelled (%.0f%%, over %d%%) - a cancelled check is neither green nor a verdict about the diff; that is a throughput or concurrency defect, not a judgement\n", canc, n, pct, maxpct
            else
                printf "PASS %d of %d runs cancelled (%.0f%%, within %d%%)\n", canc, n, pct, maxpct
            printf "%s\n", out
            exit (pct > maxpct) ? 1 : 0
        }'
}

self_test() {
    rc=0
    chk() {  # <want prefix> <got>
        case "$2" in "$1"*) printf 'ok   %s\n' "$(printf '%s' "$2" | cut -c1-78)" ;;
                      *) printf 'FAIL want %s, got: %s\n' "$1" "$2"; rc=1 ;; esac
    }
    chk PASS "$(runner_verdict active running success 0)"
    chk FAIL "$(runner_verdict failed failed oom-kill 0)"      # the 08:37 shape
    chk FAIL "$(runner_verdict failed failed oom-kill 0)"
    chk FAIL "$(runner_verdict inactive dead success 0)"
    chk FAIL "$(runner_verdict '' '' '' '')"                   # nothing readable is not a pass
    chk FAIL "$(runner_verdict activating auto-restart success 4)"    # the window where a flapping runner looks busy
    chk FAIL "$(runner_verdict activating auto-restart oom-kill 2)"   # same window, last result a kill
    if printf '%s' "$(runner_verdict activating auto-restart success 4)" | grep -q 'RESTART WINDOW'; then
        printf 'ok   the restart window is named, not just failed\n'
    else printf 'FAIL the restart window should be named in the message\n'; rc=1; fi
    # a restart policy that works shows NRestarts>0 and result success - which is a PASS with the count visible
    chk PASS "$(runner_verdict active running success 3)"
    # engine#1504: the state checks cannot see a unit that is one OOM kill away from being gone for good, which is why the
    # bench runner PASSed this script while carrying Restart=no/OOMPolicy=stop.
    chk PASS "$(policy_verdict always continue 51539607552)"
    chk FAIL "$(policy_verdict no stop infinity)"          # the live strixhalo-bench unit on 2026-11-11
    chk FAIL "$(policy_verdict on-failure continue 51539607552)"   # on-failure never fires: runsvc.sh exits 0
    chk FAIL "$(policy_verdict always stop 51539607552)"    # an OOM kill of a JOB stops the RUNNER
    chk FAIL "$(policy_verdict always continue infinity)"   # no cgroup ceiling: the box's OOM killer chooses
    chk FAIL "$(policy_verdict '' '' '')"                   # nothing readable is not a pass
    # a missing ceiling alone must not be reported as "the unit stops for good" (the live strixhalo-engine unit, 2026-11-11)
    case "$(policy_verdict always continue infinity)" in
        *'would come back'*) printf 'ok   a missing ceiling alone says the runner comes back\n' ;;
        *) printf 'FAIL a missing ceiling alone should not say the unit stops for good\n'; rc=1 ;;
    esac
    # engine#2077: the run-conclusion arms. A cancelled run must never read as a pass, and an empty list must not
    # either - the same rule as the unit arms above (nothing readable is not a pass).
    chkP() { case "$2" in "$1"*) printf 'ok   %s\n' "$(printf '%s' "$2" | cut -c1-78)" ;;
                           *) printf 'FAIL want %s, got: %s\n' "$1" "$2"; rc=1 ;; esac; }
    chkP PASS "$(printf 'success\nsuccess\nsuccess\nsuccess\ncancelled\n' | runs_summary 25)"
    chkP FAIL "$(printf 'cancelled\ncancelled\ncancelled\nsuccess\n' | runs_summary 25)"
    chkP FAIL "$(printf 'cancelled\n' | runs_summary 25)"                 # 100% cancelled: the #2077 shape
    chkP FAIL "$(printf '' | runs_summary 25)"                             # no runs read is not a pass
    chkP PASS "$(printf 'cancelled\nsuccess\nsuccess\nsuccess\nsuccess\n' | runs_summary 25)"
    chkP FAIL "$(printf 'cancelled\nsuccess\nsuccess\nsuccess\nsuccess\n' | runs_summary 0)"
    [ "$rc" = 0 ] && echo "ci-runner-health self-test: ok (22 cases)" || echo "ci-runner-health self-test: FAIL"
    return $rc
}

[ "${1:-}" = --self-test ] && { self_test; exit $?; }

# engine#2077: `--runs [--workflow NAME] [--event EVENT] [--limit N] [--max-cancelled-pct P]` reads run conclusions, not units.
# Read-only: `gh run list` and nothing else. Exit 2 when there is nothing readable, so a lane that cannot reach the
# API cannot report a silent pass (the same rule as the unit path below).
if [ "${1:-}" = --runs ]; then
    shift
    wf=; lim=40; maxpct=25; ev=
    while [ $# -gt 0 ]; do
        case "$1" in
            --workflow) wf="${2:-}"; shift 2 ;;
            --workflow=*) wf="${1#*=}"; shift ;;
            --limit) lim="${2:-40}"; shift 2 ;;
            --limit=*) lim="${1#*=}"; shift ;;
            --max-cancelled-pct) maxpct="${2:-25}"; shift 2 ;;
            --max-cancelled-pct=*) maxpct="${1#*=}"; shift ;;
            # engine#2077: the EVENT filter changes the denominator. Without it `skipped` runs from non-PR events
            # dilute the cancelled share (measured: 10/40 with all events, 23/40 on pull_request), so a caller
            # comparing two numbers must say which population it counted.
            --event) ev="${2:-}"; shift 2 ;;
            --event=*) ev="${1#*=}"; shift ;;
            *) echo "usage: $0 --runs [--workflow NAME] [--event EVENT] [--limit N] [--max-cancelled-pct P]" >&2; exit 2 ;;
        esac
    done
    if ! command -v gh >/dev/null; then echo "FAIL no gh on PATH - cannot read run conclusions"; exit 2; fi
    args=(run list --limit "$lim" --json conclusion,workflowName,event)
    [ -n "$wf" ] && args+=(--workflow "$wf")
    [ -n "$ev" ] && args+=(--event "$ev")
    json=$(gh "${args[@]}" 2>/dev/null) || { echo "FAIL gh run list failed - cannot read run conclusions"; exit 2; }
    printf '%s' "$json" | python3 -c 'import json,sys;[print(r.get("conclusion") or "running") for r in json.load(sys.stdin)]' 2>/dev/null \
        | runs_summary "$maxpct"
    exit $?
fi

SC=${SYSTEMCTL:-systemctl}
units=("$@")
if [ ${#units[@]} -eq 0 ]; then
    mapfile -t units < <("$SC" list-units 'actions.runner.*' --all --no-legend --plain 2>/dev/null | awk '{print $1}')
fi
if [ ${#units[@]} -eq 0 ]; then
    echo "FAIL no actions.runner.* units on this host - either the runners were never installed here or systemd has dropped them entirely; a silent pass would be the defect this check exists for"
    exit 2
fi

rc=0 policy=0
for u in "${units[@]}"; do
    line=$("$SC" show "$u" -p ActiveState -p SubState -p Result -p NRestarts -p Restart -p OOMPolicy -p MemoryMax 2>/dev/null)
    st=$(printf '%s\n' "$line" | sed -n 's/^ActiveState=//p')
    sub=$(printf '%s\n' "$line" | sed -n 's/^SubState=//p')
    res=$(printf '%s\n' "$line" | sed -n 's/^Result=//p')
    nr=$(printf '%s\n' "$line" | sed -n 's/^NRestarts=//p')
    rst=$(printf '%s\n' "$line" | sed -n 's/^Restart=//p')
    oom=$(printf '%s\n' "$line" | sed -n 's/^OOMPolicy=//p')
    mmax=$(printf '%s\n' "$line" | sed -n 's/^MemoryMax=//p')
    v=$(runner_verdict "$st" "$sub" "$res" "$nr")
    case "$v" in
        PASS*) printf 'PASS %-52s %s\n' "$u" "${v#PASS }" ;;
        *)     printf 'FAIL %-52s %s\n' "$u" "${v#FAIL }"; rc=1 ;;
    esac
    # engine#1504: alive now is not the same as after the next OOM kill, so the policy is its own line.
    p=$(policy_verdict "$rst" "$oom" "$mmax")
    case "$p" in
        PASS*) printf 'PASS %-52s %s\n' "" "${p#PASS }" ;;
        *)     printf 'WARN %-52s %s\n' "" "${p#FAIL }"; policy=1 ;;
    esac
done
# a runner that is down outranks a policy line: 1 if any unit is not accepting jobs, else 3 if only a policy is short.
[ "$rc" = 0 ] && [ "$policy" = 1 ] && rc=3
exit "$rc"
