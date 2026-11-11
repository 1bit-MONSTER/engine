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
# The body of the issue `bump-hrx` files when a scheduled sync fails (engine#1945).
#
# The workflow already notices a failure and files one issue per episode (#332, #350, #372, #388, #406,
# #628, #1945). What it cannot do inline is *read the log*: the body it posts today says "Look at the
# sync steps", so the person who opens the issue starts by opening a run page and scrolling. This
# script turns the failing step's log into the three things that step actually needs — which step
# failed, which files are in conflict (the log names them), and what to do about it.
#
#   scripts/bump-hrx-report.sh --run-id 42839424667 [--repo OWNER/REPO] [--log FILE] [--step NAME]
#
# --log defaults to stdin; --step defaults to the step named in the log's own `##[error]` context. The
# body goes to stdout, so the caller decides whether it becomes an issue or a comment.
#
# Exit: 0 the body was produced; 2 the log named no failing step (a report about nothing is worse than
# no report: the caller then posts the workflow's plain body rather than a confident-looking empty one).

set -uo pipefail

run_id=""
repo="${GITHUB_REPOSITORY:-1bit-MONSTER/engine}"
log=""
step=""

while [ $# -gt 0 ]; do
    case "$1" in
        --run-id) run_id="${2:-}"; shift 2 ;;
        --repo) repo="${2:-}"; shift 2 ;;
        --log) log="${2:-}"; shift 2 ;;
        --step) step="${2:-}"; shift 2 ;;
        -h|--help) sed -n '19,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf 'bump-hrx-report: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

text=""
if [ -n "$log" ]; then
    [ -r "$log" ] || { printf 'bump-hrx-report: cannot read %s\n' "$log" >&2; exit 2; }
    text=$(cat -- "$log")
else
    text=$(cat)
fi

# The step name, in the order GitHub prints it: the `##[group]Run …`/`##[error]` wrapper first, then a bare
# banner, then a `Run <script>` line. Last one wins, because a log can name several steps and the failure
# belongs to the last that started.
# GitHub's own log format, in preference order: `##[section]Starting: <step name>` names the step;
# `##[group]Run <command>` names the command; the `##[error]Process completed with exit code N` trailer
# names neither. Taking the last `##[error]` line first (as this script did before its test existed) reports
# "Process completed with exit code 1." as the failing step, which is the one line the reader already knows.
if [ -z "$step" ]; then
    step=$(printf '%s\n' "$text" | grep -E '^##\[section\]Starting: ' | tail -1 | sed -E 's/^##\[section\]Starting: //')
fi
if [ -z "$step" ]; then
    step=$(printf '%s\n' "$text" | grep -E '^##\[group\]Run ' | tail -1 | sed -E 's/^##\[group\](Run )?//')
fi
if [ -z "$step" ]; then
    step=$(printf '%s\n' "$text" | grep -E '^##\[error\]' | grep -v 'Process completed with exit code' | tail -1 |
           sed -E 's/^##\[error\]//')
fi
if [ -z "$step" ] && [ -n "$(printf '%s\n' "$text" | tr -d '[:space:]')" ]; then
    # A log with content but no recognisable step banner: report the exit status line if there is one, else refuse.
    step=$(printf '%s\n' "$text" | grep -E 'exit code|exit status|Killed|Traceback' | tail -1 |
           sed -E 's/^##\[[a-z]+\]//; s/^[[:space:]]+//')
fi
if [ -z "$step" ]; then
    printf 'bump-hrx-report: the log names no failing step, so there is nothing specific to report\n' >&2
    exit 2
fi

# The exit status, when the log carries one: useful next to the step name, and never a substitute for it.
status=$(printf '%s\n' "$text" | grep -oE 'Process completed with exit code [0-9]+' | tail -1 | grep -oE '[0-9]+$')
status_line=""
if [ -n "$status" ]; then
    status_line=" (exit $status)"
fi

# Conflicting paths. git names them in four shapes, and all four appear in this workflow's history:
#   CONFLICT (content): Merge conflict in <path>
#   CONFLICT (modify/delete): <path> deleted in <ref> and modified in <ref>
#   CONFLICT (submodule): Merge conflict in <path>
#   error: Merging is not possible because you have unmerged files.  +  git ls-files output shape
files=$(
    printf '%s\n' "$text" |
        sed -nE \
            -e 's/^.*CONFLICT \([a-z\/-]+\): Merge conflict in (.*)$/\1/p' \
            -e 's/^.*CONFLICT \([a-z\/-]+\): (.*) deleted in .*$/\1/p' \
            -e 's/^.*CONFLICT \([a-z\/-]+\): (.*) created in .*$/\1/p' |
        sed -E 's/[[:space:]]+$//' | sort -u
)

{
    if [ -n "$run_id" ]; then
        printf 'The bump-hrx run %s failed: https://github.com/%s/actions/runs/%s\n\n' "$run_id" "$repo" "$run_id"
    else
        printf 'A scheduled bump-hrx run failed.\n\n'
    fi
    printf '**The failing step:** `%s`%s\n\n' "$step" "$status_line"

    if [ -n "$files" ]; then
        printf '**Conflicting files** (named in the log):\n\n```\n%s\n```\n\n' "$files"
        printf 'Resolve them by hand on `1bit/hrx-vulkan-patched` (for `third_party/llama.cpp`) or on '
        printf '`1bit-MONSTER/hrx-system` `1bit/main` (for `third_party/hrx-system`), then re-run this workflow.\n\n'
    else
        printf '**No conflicting files were named in the log.** This run failed for a reason other than a merge '
        printf 'conflict, so the step above is where to start: the log for that step is the record, and this body\n'
        printf 'deliberately does not guess at a cause it did not see.\n\n'
    fi

    printf 'The HRX pin is what this workflow moves, so until it is green the pin is frozen at whatever it was.\n'
    printf 'Why this step exists: bump-hrx failed on every scheduled run from 2026-10-01 to 2026-10-05 without\n'
    printf 'anyone noticing, and the HRX pin silently froze.\n'
}
