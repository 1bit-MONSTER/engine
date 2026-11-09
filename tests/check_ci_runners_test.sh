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
# check_ci_runners_test.sh: the row that notices a dead runner (engine #1504 item 3).
#
# The failure it guards against cannot be seen from this box: a job whose runner does not exist never runs, so
# GitHub never fails it - the check reads as *pending*, which looks exactly like a busy queue. On 2026-11-08 that
# state lasted 3h45m with 9 hrx-build and 15 pr-agent jobs behind it.
#
# Arms, each with a fake `gh` on PATH so no network and no token are involved:
#   1. both runners online and required -> 0, and it names them;
#   2. one runner offline -> 1, and the message names the dead one (not the healthy one);
#   3. a required runner not registered at all -> 1, and says it is gone rather than online;
#   4. no self-hosted runners registered -> 0: a fresh box or a fork is not an outage;
#   5. the API call fails -> 2: an unchecked runner is exactly the state this row exists to end.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
script=$here/../scripts/check-ci-runners.sh
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"

# a fake gh: `gh api <url>` prints $FAKE_RUNNERS, or fails when $FAKE_API_FAIL=1
cat > "$tmp/bin/gh" <<'SH'
#!/usr/bin/env bash
[ "${FAKE_API_FAIL:-0}" = 1 ] && { echo "gh: HTTP 401" >&2; exit 1; }
printf '%s\n' "${FAKE_RUNNERS:-}"
SH
chmod +x "$tmp/bin/gh"

run() {   # run <json> [args...]
    local json=$1; shift
    FAKE_RUNNERS=$json PATH=$tmp/bin:$PATH bash "$script" "$@" 2>&1
}

two_online='{"total_count":2,"runners":[{"name":"strixhalo-engine","status":"online","busy":false},{"name":"strixhalo-bench","status":"online","busy":true}]}'
one_dead='{"total_count":2,"runners":[{"name":"strixhalo-engine","status":"online","busy":false},{"name":"strixhalo-bench","status":"offline","busy":false}]}'
only_bench='{"total_count":1,"runners":[{"name":"strixhalo-bench","status":"online","busy":false}]}'
none='{"total_count":0,"runners":[]}'
req=(--require strixhalo-engine strixhalo-bench)

fail=0
check() {   # check <what> <expected-rc> <output> [must-contain]
    local rc=$2 out=$3 want=${4:-}
    if [ "$rc" != "$1" ]; then
        echo "FAIL $5: expected rc=$1, got rc=$rc"
        printf '%s\n' "$out" | sed 's/^/     | /'
        fail=$((fail + 1))
        return
    fi
    if [ -n "$want" ] && ! printf '%s' "$out" | grep -qF "$want"; then
        echo "FAIL $5: rc was right but the output does not say \"$want\""
        printf '%s\n' "$out" | sed 's/^/     | /'
        fail=$((fail + 1))
        return
    fi
    echo "ok   $5"
}

# 1. both online
out=$(run "$two_online" "${req[@]}"); check 0 $? "$out" "2 self-hosted runner(s) online" "both online -> 0, and it counts them"

# 2. one offline: the message must name the dead one, exactly once (a required-but-dead runner used to be reported
#    by both the loop and the --require pass, so the count is what pins the skip against the array-expansion bug)
out=$(run "$one_dead" "${req[@]}"); check 1 $? "$out" "runner strixhalo-bench is offline" "one offline -> 1, naming the dead runner"
n=$(printf '%s\n' "$out" | grep -c 'strixhalo-bench is offline')
[ "$n" = 1 ] || { echo "FAIL the dead runner was reported $n times (expected 1)"; fail=$((fail + 1)); }
printf '%s' "$out" | grep -q 'runner strixhalo-engine is' && { echo "FAIL the healthy runner was blamed too"; fail=$((fail + 1)); }

# 2b. an unrequired runner that is offline still fails: a dead runner is an outage whatever it is called
out=$(run "$one_dead"); check 1 $? "$out" "strixhalo-bench is offline" "an unrequired dead runner still fails"

# 3. a required runner that is not registered
out=$(run "$only_bench" "${req[@]}"); check 1 $? "$out" "runner strixhalo-engine is not registered" "a required runner missing -> 1, 'not registered'"
printf '%s' "$out" | grep -q 'strixhalo-engine is online' && { echo "FAIL a missing runner was reported as online"; fail=$((fail + 1)); }

# 4. no runners at all is not an outage
out=$(run "$none" "${req[@]}"); check 0 $? "$out" "no self-hosted runners registered" "no runners registered -> 0"

# 5. the API fails: 2, never a silent ok
out=$(FAKE_API_FAIL=1 FAKE_RUNNERS=$none PATH=$tmp/bin:$PATH bash "$script" "${req[@]}" 2>&1)
check 2 $? "$out" "cannot read the runner list" "API failure -> 2, explicitly"

# 5b. more runners than one page: the list cannot be trusted, so 2 -- NOT a "missing" verdict for a required name
long=$(python3 -c 'import json; print(json.dumps({"total_count":150,"runners":[{"name":"r%d"%i,"status":"online","busy":False} for i in range(100)]}))')
out=$(run "$long" "${req[@]}"); check 2 $? "$out" "runner list is truncated" "a truncated list -> 2, not 'strixhalo-engine is not registered'"
printf '%s' "$out" | grep -q 'is not registered' && { echo "FAIL a truncated list blamed a required name"; fail=$((fail + 1)); }
full=$(python3 -c 'import json; print(json.dumps({"total_count":100,"runners":[{"name":"r%d"%i,"status":"online","busy":False} for i in range(100)]}))')
out=$(run "$full"); check 2 $? "$out" "truncated" "exactly one page -> 2 (may be truncated), never a vacuous ok"

# 6. --list prints the table without changing the verdict
out=$(run "$two_online" --list); check 0 $? "$out" "strixhalo-engine" "--list prints runners and still exits 0"

if [ "$fail" != 0 ]; then
    echo "check_ci_runners_test: $fail failure(s)"
    exit 1
fi
echo "check_ci_runners_test: all arms pass"
