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
# The boot-line rows at the end are engine#3374: the other slot's entry is built from the NEW release's own line
# (/1bit/cmdline, which mkimage.sh writes) with only this machine's keys from the running entry - not the running
# entry's line copied, which is what froze a release's kernel parameters at whatever the machine was installed with.
#
# The case the slots exist for is the SECOND update (engine#1334 review): after update 1 the machine runs from /1bit-alt,
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

# 6. a layer with THIS machine's NAME but a DIFFERENT directory is another machine's state, and must be refused.
# Regression guard for the identity test: `-ef` compares device+inode, so "same name" cannot pass for "same layer".
# (The older name-based form refused this too - it required the slot to match as well - so this arm guards the property,
# it does not separate the two implementations; measured by the #1385 review.)
fresh; mkdir -p "$DATA/1bit-alt/upup64rsave/upper/root"; echo other > "$DATA/1bit-alt/upup64rsave/upper/root/marker"
active=1bit
find_save_layer "$DATA" "$active"
[ "$save_slot" = 1bit ] && [ "$savename" = upup64rsave ]; check "a same-named layer in the other slot is not taken for ours" $?
why=$(prepare_target_layer "$DATA" 1bit-alt); rc=$?
[ $rc -ne 0 ] && case $why in *"two layers cannot be merged"*) true ;; *) false ;; esac
check "a same-named FOREIGN directory in the target is refused (name is not identity) [$why]" $?
[ "$(cat "$DATA/1bit-alt/upup64rsave/upper/root/marker")" = other ]; check "and its contents were not touched" $?

# 7. no layer at all leaves savename empty, which the install path refuses (pre-#1291 image)
rm -rf "$DATA"; mkdir -p "$DATA/1bit"
active=1bit
find_save_layer "$DATA" "$active"
[ -z "$savename" ]; check "a set with no real layer is found as none" $?

# 8. engine#3374: the other slot's boot line comes from the NEW release's own parameters, not from the running entry.
# The running entry is what 1bit-install wrote for the INSTALLED release: no printk.always_kmsg_dump=1, no pfix=, and
# this disk's UUID pin. The release carries the image's own line, whose UUIDs name the release stick's partition.
has() { case " $1 " in *" $2 "*) return 0 ;; esac; return 1; }
run='pmedia=usbhd pupsfs=UUID=disk 1bit.data=UUID=disk psubdir=1bit ttm.pages_limit=31457280 nvme.quirks=1e49:1081:dmapool_align_512 pfix=nocopy,nox console=ttyS0,115200 console=tty0'
rel='pmedia=usbhd pupsfs=UUID=stick 1bit.data=UUID=stick psubdir=1bit ttm.pages_limit=31457280 nvme.quirks=1e49:1081:dmapool_align_512 printk.always_kmsg_dump=1 pfix=copy,nox console=ttyS0,115200 console=tty0'
got=$(slot_params "$rel" "$run" 1bit-alt); rc=$?
[ "$rc" = 0 ] && has "$got" printk.always_kmsg_dump=1
check "the new release's ADDED parameter arrives on the other slot's line (printk.always_kmsg_dump=1)" $?
has "$got" pupsfs=UUID=disk && has "$got" 1bit.data=UUID=disk && ! has "$got" pupsfs=UUID=stick
check "and the machine's UUID pin replaces the release image's [$got]" $?
has "$got" psubdir=1bit-alt && ! has "$got" psubdir=1bit
check "and psubdir is the slot being written" $?
has "$got" pfix=nocopy,nox && ! has "$got" pfix=copy,nox
check "and pfix=copy (the live stick's) becomes pfix=nocopy, as 1bit-install writes for an installed disk" $?
has "$got" console=ttyS0,115200 && has "$got" console=tty0
check "the release's console order is taken as it is (the LAST console= is /dev/console)" $?

# a parameter a release DROPS must go: the running line still has ttm.pages_limit=31457280, the release's line does not
got=$(slot_params 'pmedia=usbhd psubdir=1bit nvme.quirks=1e49:1081:dmapool_align_512 pfix=copy,nox console=tty0' "$run" 1bit-alt)
! has "$got" ttm.pages_limit
check "a parameter the release DROPS does not survive from the running entry (no ttm.pages_limit) [$got]" $?

# the release's line may be the whole grub line, which is what a person copies out of an image's grub.cfg
printf 'linux /1bit/vmlinuz pmedia=usbhd psubdir=1bit log_buf_len=64M\r\n' > "$t/cmdline-full"
[ "$(release_headless "$t/cmdline-full")" = "pmedia=usbhd psubdir=1bit log_buf_len=64M" ]
check "release_headless reads a grub 'linux <kernel> <params>' line (CRLF included)" $?
printf '# the release'"'"'s boot line\n\npmedia=usbhd psubdir=1bit log_buf_len=64M\n' > "$t/cmdline-bare"
[ "$(release_headless "$t/cmdline-bare")" = "pmedia=usbhd psubdir=1bit log_buf_len=64M" ]
check "and a bare parameters line (blank and # lines skipped)" $?
[ -z "$(release_headless "$t/no-such-file")" ] && [ -z "$(release_headless "$t")" ]
check "a release that carries no /1bit/cmdline reads as no line" $?

# no release line: the running entry's parameters are copied (the pre-#3374 behaviour), and the update warns
[ "$(slot_params '' "$run" 1bit-alt)" = "pmedia=usbhd pupsfs=UUID=disk 1bit.data=UUID=disk psubdir=1bit-alt ttm.pages_limit=31457280 nvme.quirks=1e49:1081:dmapool_align_512 pfix=nocopy,nox console=ttyS0,115200 console=tty0" ]
check "a release with no line falls back to the running entry's parameters, psubdir rewritten" $?

# a release that pins the data partition and a running line without the pin: refused, with the reason named
why=$(slot_params 'pmedia=usbhd pupsfs=UUID=stick 1bit.data=UUID=stick psubdir=1bit console=tty0' 'pmedia=usbhd psubdir=1bit' 1bit-alt); rc=$?
[ "$rc" -ne 0 ] && case $why in *pupsfs=*) true ;; *) false ;; esac
check "a release pin naming a key the running entry cannot fill in is refused, by name [$why]" $?

# a release that drops the pin still gets this machine's (an installed disk boots by it)
got=$(slot_params 'pmedia=usbhd psubdir=1bit log_buf_len=64M console=tty0' "$run" 1bit-alt)
has "$got" pupsfs=UUID=disk && has "$got" 1bit.data=UUID=disk
check "a release that drops the pin still gets this machine's (the installed disk boots by it) [$got]" $?

[ $fail = 0 ] && echo "one_bit_update_slots: ok" || echo "one_bit_update_slots: FAILED"
exit $fail
