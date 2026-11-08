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
# one_bit_network_conf_test.sh: the address validation and the /data/net.conf parse of
# os/v3/puppy/.../1bit-network, run as the real functions over real files. No root, no interface, no VM: the script is
# sourced with ONEBIT_NETWORK_LIB=1, which defines its functions and returns before it touches anything.
#
# This is the file that can make an installed machine unreachable, so what it must never do is accept a value the
# kernel would refuse or half-read a file: a bad address has to be a refusal *before* anything is written, and an
# unparsable file has to leave the network alone rather than apply half of it.
#
# usage: tests/one_bit_network_conf_test.sh <repo root>
set -u
root=${1:?usage: one_bit_network_conf_test.sh <repo root>}
script=$root/os/v3/puppy/embed/usr/local/bin/1bit-network
[ -f "$script" ] || { echo "no $script"; exit 1; }

fail=0
check() { if [ "$2" = 0 ]; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT

# shellcheck disable=SC1090
ONEBIT_NETWORK_LIB=1 . "$script"
check "the script sources as a library without running (ONEBIT_NETWORK_LIB=1)" 0

# 1. addresses the kernel would take
for a in 10.0.2.50/24 192.168.50.20/24 172.16.0.1/12 1.1.1.1/32 0.0.0.0/1 255.255.255.255/32; do
    valid_addr "$a"; check "valid_addr accepts $a" $?
done

# 2. and the ones it must refuse, each of which would otherwise be written into a boot-time file
for a in "" 10.0.2.50 10.0.2.50/ 10.0.2.50/33 10.0.2.50/0 10.0.2/24 10.0.2.256/24 10.0.2.-1/24 \
         10.0.2.50/24x abc/24 1.2.3.4.5/24 " 10.0.2.50/24" 10.0.2.50/24extra; do
    valid_addr "$a" && { check "valid_addr refuses '$a'" 1; continue; }
    check "valid_addr refuses '$a'" 0
done

# 3. gateways and DNS are plain addresses: a prefix there is a mistake, not a shortcut
valid_ip 10.0.2.2; check "valid_ip accepts a plain address" $?
valid_ip 10.0.2.2/24 && { check "valid_ip refuses a CIDR (a gateway is not a prefix)" 1; } || check "valid_ip refuses a CIDR (a gateway is not a prefix)" 0

# 4. the file itself: comments, blank lines, an unknown key (ignored with a warning), and the four keys it uses
cat > "$t/good.conf" <<'CONF'
# written by hand in this test
iface=lo
static=10.0.2.50/24
gateway=10.0.2.2
dns=1.1.1.1 8.8.8.8
CONF
read_conf "$t/good.conf"
[ "$iface" = lo ] && [ "$static" = 10.0.2.50/24 ] && [ "$gateway" = 10.0.2.2 ] && \
    [ "$dns" = "1.1.1.1 8.8.8.8" ]; check "a well-formed file is read into iface/static/gateway/dns" $?
conf_ok > /dev/null; check "and it validates (lo is an interface on any machine)" $?

printf 'iface=lo\nstatic=10.0.2.50/24\nbogus=1\n' > "$t/unknown.conf"
warn=$(read_conf "$t/unknown.conf" 2>&1)
case $warn in *"unknown key 'bogus'"*) true ;; *) false ;; esac
check "an unknown key is reported, not silently ignored [$warn]" $?
conf_ok > /dev/null; check "and it does not stop the file from validating" $?

# 5. the cases where conf_ok has to refuse, because the boot then leaves the network alone instead of applying it
for bad in "iface=lo\nstatic=10.0.2.50\n" "iface=lo\nstatic=10.0.2.50/24\ngateway=nope\n" \
           "iface=lo\nstatic=10.0.2.50/24\ndns=nope\n" "iface=nosuchiface0\nstatic=10.0.2.50/24\n"; do
    printf "$bad" > "$t/bad.conf"
    read_conf "$t/bad.conf"
    if conf_ok > "$t/why" 2>&1; then check "conf_ok refuses '$(printf "$bad" | tr '\n' ';')'" 1
    else check "conf_ok refuses '$(printf "$bad" | tr '\n' ';')' [$(cat "$t/why")]" 0; fi
done

# 6. an empty or absent file is "no opinion", not an error: the boot must treat it as DHCP
: > "$t/empty.conf"
read_conf "$t/empty.conf"
[ -z "$iface" ] && [ -z "$static" ]; check "an empty file parses to nothing (DHCP)" $?
read_conf "$t/does-not-exist"
[ -z "$static" ]; check "an absent file parses to nothing (DHCP)" $?

[ $fail = 0 ] && echo "one_bit_network_conf: ok" || echo "one_bit_network_conf: FAILED"
exit $fail
