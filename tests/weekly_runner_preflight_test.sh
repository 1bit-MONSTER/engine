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
# weekly_runner_preflight_test.sh: the release preflight BLOCKS on a runner unit that is down and only WARNS on a restart
# policy that is short (engine#1504, owner 2026-11-11). Two halves, both the real code:
#   1. scripts/ci-runner-health.sh against a stub systemctl: exit 0 / 1 / 2 / 3, and a down unit outranks a policy line.
#   2. scripts/weekly.sh's preflight_runner_units(), cut out of the script and run against the real health script on the
#      same stub: returns 0 with a WARNING line on exit 3, returns 1 on exit 1.
# No systemd, no network, no build.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
fails=0
ok()  { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fails=$((fails + 1)); }

# stub systemctl: STUB_UNITS = lines "unit state sub result restart oompolicy memorymax"
cat > "$tmp/systemctl" <<'STUB'
#!/usr/bin/env bash
case "$1" in
    list-units) awk '{print $1, "loaded", $2, $3}' "$STUB_UNITS" ;;
    show) awk -v u="$2" '$1 == u {
              printf "ActiveState=%s\nSubState=%s\nResult=%s\nNRestarts=0\nRestart=%s\nOOMPolicy=%s\nMemoryMax=%s\n", $2, $3, $4, $5, $6, $7 }' "$STUB_UNITS" ;;
esac
STUB
chmod +x "$tmp/systemctl"
export SYSTEMCTL="$tmp/systemctl" STUB_UNITS="$tmp/units"
health="$root/scripts/ci-runner-health.sh"

good="actions.runner.a.service active running success always continue 51539607552"
policy="actions.runner.b.service active running success no stop infinity"      # the live bench unit, 2026-11-11
dead="actions.runner.c.service failed failed oom-kill no stop infinity"        # the 11-08 shape

expect_rc() {  # <name> <want rc> <unit lines...>
    local name=$1 want=$2; shift 2
    printf '%s\n' "$@" > "$STUB_UNITS"; [ $# -eq 0 ] && : > "$STUB_UNITS"
    out=$(bash "$health" 2>&1); rc=$?
    if [ "$rc" = "$want" ]; then ok "health: $name -> exit $rc"; else bad "health: $name: want exit $want, got $rc: $out"; fi
}
expect_rc "all healthy" 0 "$good"
expect_rc "a unit is down" 1 "$good" "$dead"
expect_rc "no units" 2
expect_rc "all up, one policy short" 3 "$good" "$policy"
expect_rc "down outranks policy" 1 "$policy" "$dead"
printf '%s\n' "$good" "$policy" > "$STUB_UNITS"
out=$(bash "$health" 2>&1)
if printf '%s\n' "$out" | grep -q '^WARN .*Restart=no(want always)' && ! printf '%s\n' "$out" | grep -q '^FAIL'; then
    ok "health: a short policy prints WARN with the setting, and no FAIL line"
else bad "health: a short policy should print WARN and no FAIL: $out"; fi

# half 2: the function weekly.sh really runs
sed -n '/^preflight_runner_units() {$/,/^}$/p' "$root/scripts/weekly.sh" > "$tmp/fn.sh"
grep -q 'health_rc' "$tmp/fn.sh" || { bad "could not cut preflight_runner_units() out of scripts/weekly.sh"; echo "weekly_runner_preflight: FAIL"; exit 1; }
grep -q 'preflight_runner_units "' "$root/scripts/weekly.sh" || bad "scripts/weekly.sh's preflight no longer calls preflight_runner_units"
say() { echo "$*"; }
# shellcheck disable=SC1091
. "$tmp/fn.sh"
row() {  # <name> <want return> <want text> <unit lines...>
    local name=$1 want=$2 text=$3; shift 3
    printf '%s\n' "$@" > "$STUB_UNITS"; [ $# -eq 0 ] && : > "$STUB_UNITS"
    out=$(preflight_runner_units "$health" 2>&1); rc=$?
    if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -q -- "$text"; then ok "weekly: $name -> return $rc"
    else bad "weekly: $name: want return $want with '$text', got $rc: $out"; fi
}
row "all healthy passes" 0 "PASS in systemd's view" "$good"
row "a short policy warns and does not block" 0 "WARNING, not blocking" "$good" "$policy"
row "the warning names the setting" 0 "OOMPolicy=stop(want continue)" "$good" "$policy"
row "a dead unit blocks" 1 "not accepting jobs" "$good" "$dead"
row "a dead unit blocks even beside a short policy" 1 "after an OOM kill" "$policy" "$dead"
row "no units is a note" 0 "nothing to inspect"
out=$(preflight_runner_units "$tmp/absent.sh" 2>&1); rc=$?
[ "$rc" = 0 ] && [ -z "$out" ] && ok "weekly: no health script, no row" || bad "weekly: a missing health script should be silent: $rc $out"

[ "$fails" = 0 ] && { echo "weekly_runner_preflight: ok"; exit 0; }
echo "weekly_runner_preflight: FAIL ($fails)"; exit 1
