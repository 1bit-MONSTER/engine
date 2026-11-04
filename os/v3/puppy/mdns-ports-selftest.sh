#!/bin/sh
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
# mdns-ports-selftest.sh [script] - the self-check for zz-onboard-ports' inbound rules, with a stub `iptables`.
#
# No root, no real ruleset, no device: the stub keeps a rule table in a file and answers `-C` from it, so this
# asserts what the shipped script actually issues. The mDNS block is the kind of code that fails silently (a rule
# with the wrong source, or one re-added on every boot), so it gets a check that runs anywhere:
#
#   * one insert per LAN range, four of them, with the four sources the file intends;
#   * a second run against the same ruleset inserts nothing (`-C` first, which is what keeps a re-run a no-op);
#   * the two pre-existing TCP ports are still opened - this script must not have moved them.
#
# The measured behaviour on a booted image (`vm-check`'s `U3 mdns rule` row, and the counter test in #1016's body)
# is what proves the appliance accepts the packet; this proves the script's own shape.
set -u
prog=${0##*/}
here=$(cd "$(dirname "$0")" && pwd)
script=${1:-$here/embed/etc/init.d/zz-onboard-ports}
[ -f "$script" ] || { echo "$prog: no $script" >&2; exit 2; }

work=$(mktemp -d "${TMPDIR:-/tmp}/mdns-ports.XXXXXX") || exit 2
trap 'rm -rf "$work"' EXIT INT TERM
mkdir -p "$work/bin"
export CALLS="$work/calls" STATE="$work/state"
: > "$CALLS"; : > "$STATE"

cat > "$work/bin/iptables" <<'STUB'
#!/bin/sh
# the rule's identity is "<chain> <spec...>", so `-I INPUT 1 <spec>` and `-C INPUT <spec>` compare directly
echo "$*" >> "$CALLS"
cmd=$1; shift
chain=$1; shift
case $cmd in
  -C) grep -qxF "$chain $*" "$STATE" 2>/dev/null && exit 0; exit 1 ;;
  -I) shift; echo "$chain $*" >> "$STATE"; exit 0 ;;
  -S) printf '%s\n' '-A INPUT -p tcp -m tcp --dport 13305 -j ACCEPT' '-A INPUT -p udp -m udp --dport 5353 -j ACCEPT'; exit 0 ;;
  *)  exit 0 ;;
esac
STUB
chmod +x "$work/bin/iptables"
PATH="$work/bin:$PATH" sh "$script" > "$work/run1.log" 2>&1
i1=$(grep -c -- '^-I ' "$CALLS")
c1=$(grep -c -- '-dport 5353' "$CALLS")
u1=$(grep -c -- '^-I .* -p udp --dport 5353 ' "$CALLS")
PATH="$work/bin:$PATH" sh "$script" > "$work/run2.log" 2>&1
i2=$(grep -c -- '^-I ' "$CALLS")
u2=$(grep -c -- '^-I .* -p udp --dport 5353 ' "$CALLS")
c2=$(grep -c -- '-dport 5353' "$CALLS")

echo "run 1: $i1 insert(s), $u1 of them udp/5353, $c1 udp/5353 command(s)"
echo "run 2 (same ruleset): $((i2 - i1)) insert(s), $((c2 - c1)) udp/5353 command(s) - all -C, none -I"
fail=
[ "$i1" -ge 6 ] || fail="$fail first-run-inserts=$i1(expected 6: two TCP ports, four udp/5353)"
[ "$u1" -eq 4 ] || fail="$fail udp5353-inserts=$u1(expected 4)"
[ "$u2" -eq 4 ] || fail="$fail re-run-inserted-udp5353=$u2(not idempotent)"
[ "$((c2 - c1))" -eq 4 ] || fail="$fail re-run-udp5353-commands=$((c2 - c1))(expected 4 -C and no -I)"
for src in 10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 169.254.0.0/16; do
    grep -qxF "INPUT -p udp --dport 5353 -s $src -j ACCEPT" "$STATE" || fail="$fail missing-rule-$src"
done
if [ -n "$fail" ]; then
    echo "RESULT: FAIL -$fail"
    exit 1
fi
echo "RESULT: PASS (four udp/5353 rules inserted once, the re-run inserts nothing, sources as intended)"
