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
# one_bit_update_slots_test.sh: the save-layer and target-slot decisions of os/v3/puppy/.../1bit-update, run as the
# real functions over fake /data layouts. No root, no mounts, no device, no VM: the script is sourced with
# ONEBIT_UPDATE_LIB=1, which defines its functions and returns before it touches anything.
#
# The case this exists for is the SECOND update (engine#1334 review): after update 1 the machine runs from /1bit-alt,
# the save layer still lives in /1bit (the alt slot only links to it), so update 2 targets /1bit and must keep that
# layer as the machine's own, not refuse it as "another machine's state". update-gate.sh proves the boots on a VM;
# this proves the decision on every CI run.
#
# usage: tests/one_bit_update_slots_test.sh <repo root>
# shellcheck disable=SC2319,SC2154  # $? after a [ ] test is the point; save_slot/savename come from the sourced script
set -u
root=${1:?usage: one_bit_update_slots_test.sh <repo root>}
script=$root/os/v3/puppy/embed/usr/local/bin/1bit-update
[ -f "$script" ] || { echo "no $script"; exit 1; }

fail=0
check() { if [ "$2" = 0 ]; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
t=$(mktemp -d); trap 'rm -rf "$t"' EXIT

# one fresh data partition per case: /1bit holds the real layer (where 1bit-install put it), with a marker in it
fresh() { rm -rf "$t/data"; mkdir -p "$t/data/1bit/upup64rsave/upper/root"; echo boot1 > "$t/data/1bit/upup64rsave/upper/root/marker"; }

# shellcheck disable=SC1090
ONEBIT_UPDATE_LIB=1 . "$script"
check "the script sources as a library without running (ONEBIT_UPDATE_LIB=1)" 0
DATA=$t/data

# 1. first update: /1bit runs, the target is /1bit-alt, which gets a relative link to the shared layer
fresh; mkdir -p "$DATA/1bit-alt"
active=1bit
find_save_layer "$DATA" "$active"
[ "$save_slot" = 1bit ] && [ "$savename" = upup64rsave ]; check "update 1: the layer is found in /1bit" $?
why=$(prepare_target_layer "$DATA" 1bit-alt); check "update 1: /1bit-alt is accepted as the target" $?
link_target_layer "$DATA" 1bit-alt
[ "$(readlink "$DATA/1bit-alt/upup64rsave")" = ../1bit/upup64rsave ]; check "update 1: /1bit-alt links ../1bit/upup64rsave" $?

# 2. second update: the machine now runs /1bit-alt; the layer is still the real dir in /1bit; the target is /1bit
active=1bit-alt
find_save_layer "$DATA" "$active"
[ "$save_slot" = 1bit ]; check "update 2: the layer is still found in /1bit (a link reaches it, it does not hold it)" $?
why=$(prepare_target_layer "$DATA" 1bit); rc=$?
check "update 2: /1bit is accepted although it holds the layer - it is this machine's own (was: REFUSED) [$why]" $rc
link_target_layer "$DATA" 1bit
[ -d "$DATA/1bit/upup64rsave" ] && [ ! -L "$DATA/1bit/upup64rsave" ]; check "update 2: the layer is still a real directory" $?
[ ! -e "$DATA/1bit/upup64rsave/upup64rsave" ] && [ ! -L "$DATA/1bit/upup64rsave/upup64rsave" ]; check "update 2: no link was made inside the layer" $?
[ "$(cat "$DATA/1bit/upup64rsave/upper/root/marker")" = boot1 ]; check "update 2: the machine's state (boot-1 marker) is intact" $?
[ "$(readlink "$DATA/1bit-alt/upup64rsave")" = ../1bit/upup64rsave ]; check "update 2: the running slot's link is untouched" $?

# 3. third update: the machine runs /1bit again; back to /1bit-alt, whose link is replaced by an equal one
active=1bit
find_save_layer "$DATA" "$active"
why=$(prepare_target_layer "$DATA" 1bit-alt); check "update 3: /1bit-alt is accepted again" $?
link_target_layer "$DATA" 1bit-alt
[ "$(readlink "$DATA/1bit-alt/upup64rsave")" = ../1bit/upup64rsave ]; check "update 3: /1bit-alt links the shared layer again" $?

# 4. another machine's layer in the target is still refused - and the scan must not mistake it for ours
fresh; mkdir -p "$DATA/1bit-alt/upup64rsave2"
active=1bit
find_save_layer "$DATA" "$active"
[ "$save_slot" = 1bit ] && [ "$savename" = upup64rsave ]; check "a foreign layer in the other slot is not taken for the running one" $?
why=$(prepare_target_layer "$DATA" 1bit-alt); rc=$?
[ $rc -ne 0 ] && case $why in *"two layers cannot be merged"*) true ;; *) false ;; esac; check "a foreign real layer in the target is refused, by name [$why]" $?

# 5. a stale link in the target is replaced, not refused
fresh; mkdir -p "$DATA/1bit-alt"; ln -s ../nowhere/upup64rsave "$DATA/1bit-alt/upup64rsave"
active=1bit
find_save_layer "$DATA" "$active"
why=$(prepare_target_layer "$DATA" 1bit-alt); check "a stale link in the target is accepted" $?
[ ! -L "$DATA/1bit-alt/upup64rsave" ]; check "and removed before the new link is made" $?
link_target_layer "$DATA" 1bit-alt
[ "$(readlink "$DATA/1bit-alt/upup64rsave")" = ../1bit/upup64rsave ]; check "and the new link points at the shared layer" $?

# 6. no layer at all leaves savename empty, which the install path refuses (pre-#1291 image)
rm -rf "$DATA"; mkdir -p "$DATA/1bit"
active=1bit
find_save_layer "$DATA" "$active"
[ -z "$savename" ]; check "a set with no real layer is found as none" $?

[ $fail = 0 ] && echo "one_bit_update_slots: ok" || echo "one_bit_update_slots: FAILED"
exit $fail
