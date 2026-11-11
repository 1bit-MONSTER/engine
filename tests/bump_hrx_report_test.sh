#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# tests/bump_hrx_report_test.sh -- scripts/bump-hrx-report.sh against logs in the shapes this workflow
# has actually failed with (engine#1945). No network, no git: fixtures only.

set -uo pipefail

root=$(cd -- "$(dirname -- "$0")/.." && pwd)
script="$root/scripts/bump-hrx-report.sh"
fails=0
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT

check() {  # check <ok|fail> <what>
    if [ "$1" = ok ]; then
        printf 'ok   %s\n' "$2"
    else
        printf 'FAIL %s\n' "$2"
        fails=$((fails + 1))
    fi
}

has() {  # has <file> <literal>
    grep -qF -- "$2" "$1"
}

# 1. a merge conflict in the llama.cpp submodule: the step and the path, both from the log
cat > "$tmp/conflict.log" <<'LOG'
##[section]Starting: Set up job
##[section]Starting: Sync the fork and merge AMD's pin into our line
##[group]Run git merge --no-edit origin/1bit/hrx-vulkan-patched
Auto-merging ggml/src/ggml-hrx/dispatch/dispatch-scheduler.cpp
CONFLICT (content): Merge conflict in ggml/src/ggml-hrx/dispatch/dispatch-scheduler.cpp
Automatic merge failed; fix conflicts and then commit the result.
##[error]Process completed with exit code 1.
LOG
"$script" --run-id 42839424667 --repo 1bit-MONSTER/engine --log "$tmp/conflict.log" > "$tmp/out1.txt"
check $([ $? -eq 0 ] && echo ok || echo fail) "conflict log: a body is produced"
check $(has "$tmp/out1.txt" "42839424667" && has "$tmp/out1.txt" "actions/runs/42839424667" && echo ok || echo fail) \
      "conflict log: the run id and its URL are in the body"
check $(has "$tmp/out1.txt" "ggml/src/ggml-hrx/dispatch/dispatch-scheduler.cpp" && echo ok || echo fail) \
      "conflict log: the conflicting path is NAMED (the whole point of reading the log)"
check $(has "$tmp/out1.txt" "Resolve them by hand on \`1bit/hrx-vulkan-patched\`" && echo ok || echo fail) \
      "conflict log: the remedy names where to resolve it"
check $(has "$tmp/out1.txt" "**The failing step:** \`Sync the fork and merge AMD's pin into our line\`" && echo ok || echo fail) \
      "conflict log: the failing step is the SECTION name, not the exit-code trailer"
check $(has "$tmp/out1.txt" '**The failing step:** `Process completed with exit code' && echo fail || echo ok) \
      "conflict log: the exit-code trailer is not reported AS the step"
check $(has "$tmp/out1.txt" '(exit 1)' && echo ok || echo fail) \
      "conflict log: the exit status rides beside the step name, as context"

# 2. a modify/delete conflict, the second shape git prints
cat > "$tmp/modify-delete.log" <<'LOG'
##[section]Starting: Merge AMD's hrx-system pin into our line
CONFLICT (modify/delete): third_party/hrx-system/README.md deleted in 1bit/main and modified in HEAD.  Version HEAD of third_party/hrx-system/README.md left in tree.
##[error]Process completed with exit code 1.
LOG
"$script" --run-id 7 --log "$tmp/modify-delete.log" > "$tmp/out2.txt"
check $(has "$tmp/out2.txt" "third_party/hrx-system/README.md" && echo ok || echo fail) \
      "modify/delete log: the path is named, not just 'a conflict'"

# 3. two different paths in one log: both, deduplicated
cat > "$tmp/two.log" <<'LOG'
##[section]Starting: Sync
CONFLICT (content): Merge conflict in a/b.cpp
CONFLICT (content): Merge conflict in c/d.cpp
CONFLICT (content): Merge conflict in a/b.cpp
##[error]Process completed with exit code 1.
LOG
"$script" --run-id 8 --log "$tmp/two.log" > "$tmp/out3.txt"
check $([ "$(grep -c 'a/b.cpp' "$tmp/out3.txt")" -eq 1 ] && echo ok || echo fail) \
      "two conflicts: a repeated path appears once"
check $(has "$tmp/out3.txt" "c/d.cpp" && echo ok || echo fail) "two conflicts: both paths are named"

# 4. a failure that is NOT a conflict: the body says so instead of inventing a cause
cat > "$tmp/generic.log" <<'LOG'
##[section]Starting: Sync the fork and merge AMD's pin into our line
##[group]Run git fetch
fatal: unable to access 'https://github.com/AMD/...': Could not resolve host
##[error]Process completed with exit code 128.
LOG
"$script" --run-id 9 --log "$tmp/generic.log" > "$tmp/out4.txt"
check $(has "$tmp/out4.txt" "No conflicting files were named in the log" && echo ok || echo fail) \
      "generic log: the body refuses to name a conflict it did not see"
check $(has "$tmp/out4.txt" "(exit 128)" && echo ok || echo fail) \
      "generic log: the step line carries the exit status"
check $(has "$tmp/out4.txt" "a/b.cpp" && echo fail || echo ok) "generic log: no path is invented"

# 5. an empty log: refuse (exit 2) rather than post a confident-looking empty body
: > "$tmp/empty.log"
"$script" --run-id 10 --log "$tmp/empty.log" > "$tmp/out5.txt" 2> "$tmp/err5.txt"
check $([ $? -eq 2 ] && echo ok || echo fail) "empty log: exits 2"
check $([ ! -s "$tmp/out5.txt" ] && echo ok || echo fail) "empty log: prints no body"
check $(has "$tmp/err5.txt" "names no failing step" && echo ok || echo fail) "empty log: says why on stderr"

# 6. --step wins over what the log says, so a caller that knows better can say so
"$script" --run-id 11 --log "$tmp/generic.log" --step 'Sync the fork and merge AMD'"'"'s pin into our line' > "$tmp/out6.txt"
check $(has "$tmp/out6.txt" "Sync the fork and merge AMD's pin into our line" && echo ok || echo fail) \
      "--step: the caller's step name is used"

# 7. stdin works (the workflow pipes a step log), and no --run-id still produces a body
cat "$tmp/conflict.log" | "$script" > "$tmp/out7.txt"
check $([ $? -eq 0 ] && echo ok || echo fail) "stdin: a body is produced without --log"
check $(has "$tmp/out7.txt" "A scheduled bump-hrx run failed." && echo ok || echo fail) \
      "stdin: with no run id the body says so rather than printing an empty link"

if [ "$fails" -ne 0 ]; then
    printf 'bump_hrx_report: %d failure(s)\n' "$fails"
    exit 1
fi
printf 'bump_hrx_report: all cases ok\n'
