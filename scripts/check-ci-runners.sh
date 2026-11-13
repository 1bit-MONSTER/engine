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
# check-ci-runners.sh [--repo OWNER/REPO] [--require NAME ...] [--list]
#
# engine #1504 item 3: something that NOTICES when a self-hosted runner is gone.
#
# On 2026-11-08 both runners were OOM-killed and stayed dead for 3h45m, and every check that needs this box read
# as *pending* - never green, never red - which is indistinguishable from "queued behind a busy runner". A restart
# policy (scripts/install-ci-runner.sh) shortens that outage; nothing but a row like this makes it visible, because
# a job whose runner does not exist never runs and therefore never fails.
#
# It asks the same API the release lanes already use. Two deliberate rules:
#   - a repository with NO self-hosted runners registered is not this check's business (a fresh box, a fork):
#     it says so and exits 0. A runner that exists and is not `online` is an outage, and that fails;
#   - names given with --require must BOTH exist and be online, because "both runners are online" is the claim,
#     and a deleted runner silently satisfies an "every runner is online" reading.
#
# It cannot be a ctest: it needs the network and a token. scripts/weekly.sh's preflight runs it; docs/ci-runners.md
# says where else it belongs and why a scheduled red needs a workflow change (an RFC under CONTRIBUTING).
set -uo pipefail

# usage(): the header comment as prose, anchored on the comment block rather than a line range -- two of these ranges had
# drifted past the end of their header and printed code to whoever asked for help (engine#2129).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

repo=1bit-MONSTER/engine
require=()
list=0
while [ $# -gt 0 ]; do
    case $1 in
        --repo) repo=$2; shift 2 ;;
        --require) shift; while [ $# -gt 0 ] && [ "${1#-}" = "$1" ]; do require+=("$1"); shift; done ;;
        --list) list=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "check-ci-runners.sh: unknown option $1" >&2; exit 2 ;;
    esac
done

command -v gh >/dev/null || { echo "check-ci-runners.sh: no gh on PATH (the runner list comes from the GitHub API)" >&2; exit 2; }

json=$(gh api "repos/$repo/actions/runners?per_page=100" 2>/dev/null) || {
    echo "::error::check-ci-runners: cannot read the runner list of $repo (gh api failed): an unchecked runner is the state this row exists to end" >&2
    exit 2; }

# name<TAB>status<TAB>busy, one per registered runner -- or the word "truncated", which is not a verdict at all.
# The list is one page of 100: if the repository ever has more, a runner that EXISTS beyond page one is reported
# missing, which is a false outage and a page at 3 a.m. for a pager that cannot see page two. A monitor that
# cannot tell has to say so (exit 2), same rule as an unreadable list.
rows=$(printf '%s' "$json" | python3 -c '
import json,sys
d=json.load(sys.stdin)
rs=d.get("runners",[])
total=d.get("total_count",len(rs))
if total>len(rs) or len(rs)>=100:
    print("truncated\t%s\t%s" % (len(rs), total))
    raise SystemExit(0)
for r in rs:
    print("%s\t%s\t%s" % (r.get("name","?"), r.get("status","?"), str(r.get("busy")).lower()))
') || { echo "check-ci-runners.sh: cannot parse the runner list" >&2; exit 2; }

case ${rows:-} in
    truncated*)
        have=$(printf '%s' "$rows" | cut -f2); total=$(printf '%s' "$rows" | cut -f3)
        echo "::error::check-ci-runners: the runner list is truncated ($have of $total registered, one page is 100), so a runner on the next page would read as missing: this row cannot answer" >&2
        exit 2 ;;
esac

if [ -z "$rows" ]; then
    echo "ok   no self-hosted runners registered for $repo: nothing for this row to watch"
    exit 0
fi

if [ "$list" = 1 ]; then
    printf '%s\n' "$rows" | while IFS=$'\t' read -r name status busy; do
        printf '%-22s %-8s busy=%s\n' "$name" "$status" "$busy"
    done
fi

bad=0
# every registered runner has to be online: one that is offline is an outage, whatever it is called
while IFS=$'\t' read -r name status busy; do
    printf '%s\n' ${require[@]+"${require[@]}"} | grep -qxF "$name" && continue   # required: checked below, with its own line
    [ "$status" = online ] && continue
    echo "::error::runner $name is $status (busy=$busy): every check that needs this box now reads as pending, not red"
    bad=$((bad + 1))
done < <(printf '%s\n' "$rows")

# ... and the names that must exist have to exist, then be online
for name in ${require[@]+"${require[@]}"}; do
    line=$(printf '%s\n' "$rows" | awk -F'\t' -v n="$name" '$1 == n')
    if [ -z "$line" ]; then
        echo "::error::runner $name is not registered for $repo: the runner that takes these jobs is gone, so they queue for ever"
        bad=$((bad + 1))
        continue
    fi
    status=$(printf '%s' "$line" | cut -f2)
    busy=$(printf '%s' "$line" | cut -f3)
    [ "$status" = online ] || {
        echo "::error::runner $name is $status (busy=$busy): every check that needs this box now reads as pending, not red"
        bad=$((bad + 1))
    }
done

if [ "$bad" = 0 ]; then
    echo "ok   $(printf '%s\n' "$rows" | wc -l) self-hosted runner(s) online${require[*]:+, including: ${require[*]}}"
    exit 0
fi
exit 1
