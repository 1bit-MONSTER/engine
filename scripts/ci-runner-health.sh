#!/bin/bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
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
#
# Exit: 0 when every runner is active/running with no failure result; 1 when any is not; 2 when there is nothing to inspect
# (no such units on this host, so a silent pass would be the defect).
set -uo pipefail

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
    case "$_res" in
        oom-kill) printf 'FAIL %s/%s after an OOM kill (%s restarts) - the kernel killed it and nothing restarted it; jobs read as pending, not as failed' "$_st" "$_sub" "${_nr:-0}" ;;
        *)        printf 'FAIL %s/%s with last result %s (%s restarts) - the runner is not accepting jobs' "$_st" "${_sub:-?}" "${_res:-?}" "${_nr:-0}" ;;
    esac
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
    # a restart policy that works shows NRestarts>0 and result success - which is a PASS with the count visible
    chk PASS "$(runner_verdict active running success 3)"
    [ "$rc" = 0 ] && echo "ci-runner-health self-test: ok (6 cases)" || echo "ci-runner-health self-test: FAIL"
    return $rc
}

[ "${1:-}" = --self-test ] && { self_test; exit $?; }

SC=${SYSTEMCTL:-systemctl}
units=("$@")
if [ ${#units[@]} -eq 0 ]; then
    mapfile -t units < <("$SC" list-units 'actions.runner.*' --all --no-legend --plain 2>/dev/null | awk '{print $1}')
fi
if [ ${#units[@]} -eq 0 ]; then
    echo "FAIL no actions.runner.* units on this host - either the runners were never installed here or systemd has dropped them entirely; a silent pass would be the defect this check exists for"
    exit 2
fi

rc=0
for u in "${units[@]}"; do
    line=$("$SC" show "$u" -p ActiveState -p SubState -p Result -p NRestarts 2>/dev/null)
    st=$(printf '%s\n' "$line" | sed -n 's/^ActiveState=//p')
    sub=$(printf '%s\n' "$line" | sed -n 's/^SubState=//p')
    res=$(printf '%s\n' "$line" | sed -n 's/^Result=//p')
    nr=$(printf '%s\n' "$line" | sed -n 's/^NRestarts=//p')
    v=$(runner_verdict "$st" "$sub" "$res" "$nr")
    case "$v" in
        PASS*) printf 'PASS %-52s %s\n' "$u" "${v#PASS }" ;;
        *)     printf 'FAIL %-52s %s\n' "$u" "${v#FAIL }"; rc=1 ;;
    esac
done
exit "$rc"
