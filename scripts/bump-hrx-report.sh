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
# bump-hrx-report.sh -- build the body of the issue a failed scheduled sync files.
#
# Why this exists (engine#1945): the report step already files an issue so a broken sync cannot sit unnoticed
# for five days, but its body was a static template -- the run URL and instructions, never the failing step or
# the files in conflict. So the reader was told where to look and not what broke. This turns the log the run
# already has into the body, and lives in scripts/ so it is testable without running the workflow.
#
# usage: bump-hrx-report.sh --run <url> [--step <name>] [--log <file>|-] [--attempt <n>]
#   stdout = the issue body; rc 0 unless an argument is missing.
#   --log - reads stdin. No --log means "the log was not available": the body says so rather than claiming
#   there were no conflicts, which is the difference between "nothing conflicted" and "nobody looked".
set -uo pipefail

# usage(): the header comment as prose, anchored on the comment block rather than a line range -- two of these ranges had
# drifted past the end of their header and printed code to whoever asked for help (engine#2129).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

run="" step="" log="" attempt=""
while [ $# -gt 0 ]; do
    case "$1" in
        --run)     run=${2:?--run needs a URL}; shift 2 ;;
        --step)    step=${2:?--step needs a name}; shift 2 ;;
        --log)     log=${2:?--log needs a path or -}; shift 2 ;;
        --attempt) attempt=${2:?--attempt needs a number}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "bump-hrx-report: unknown argument: $1" >&2; exit 2 ;;
    esac
done
[ -n "$run" ] || { echo "bump-hrx-report: --run is required" >&2; exit 2; }

text=""
if [ "$log" = "-" ]; then
    text=$(cat)
elif [ -n "$log" ]; then
    if [ -r "$log" ]; then text=$(cat "$log"); else
        echo "bump-hrx-report: cannot read log: $log" >&2; exit 2
    fi
fi

# git names every conflicting path in a line of the form "CONFLICT (content): Merge conflict in <path>";
# "CONFLICT (modify/delete)" and friends use the same prefix. Paths can repeat across retries, so dedupe.
# git spells the same event two ways, and a report that knows only one of them silently lists half the conflict:
#   CONFLICT (content): Merge conflict in <path>
#   CONFLICT (modify/delete): <path> deleted in <ref> and modified in <ref>
#   CONFLICT (rename/delete): <path> renamed ...            (path is the source side)
conflicts=$(printf '%s\n' "$text" \
    | sed -n -e 's/.*CONFLICT ([^)]*): Merge conflict in \(.*\)$/\1/p' \
             -e 's/.*CONFLICT ([^)]*): \(.*\) \(deleted\|renamed\) in .*/\1/p' \
    | sort -u)

printf 'The bump-hrx run failed: %s\n' "$run"
if [ -n "$step" ]; then
    printf '\n**Failing step:** `%s`' "$step"
    [ -n "$attempt" ] && printf ' (attempt %s)' "$attempt"
    printf '\n'
fi

if [ -n "$conflicts" ]; then
    printf '\n**Conflicting files** (from the log):\n\n```\n%s\n```\n' "$conflicts"
    printf '\nResolve these by hand on `1bit/hrx-vulkan-patched` or on `1bit-MONSTER/hrx-system` `1bit/main`, then re-run the workflow.\n'
elif [ -z "$log" ]; then
    printf '\n**The log was not available to this report**, so the conflicting files are not listed here. Open the run above and read the failing step.\n'
else
    printf '\n**No conflicting files were named in the log** -- this failure is not a merge conflict. Open the run above and read the failing step.\n'
fi

printf '\nWhy this report exists: bump-hrx failed on every scheduled run from 2026-10-01 to 2026-10-05 without anyone\nnoticing, and the HRX pin silently froze. A failed run files this issue; it is deduped while open.\n'
