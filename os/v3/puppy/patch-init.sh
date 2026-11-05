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
# patch-init.sh <build-root>          patch the initrd's /init in that build root (idempotent)
# patch-init.sh --verify <initrd.gz>  fail unless a packed initrd carries the guard
#
# Why: the 2 Nov board boot (cycle 10) worked all the way through shim -> GRUB -> our signed kernel and then
# panicked at 26.7 s:
#     Kernel panic - not syncing: Attempted to kill init! exitcode=0x00000000    Comm: sh
# A clean exit code on PID 1 means Puppy's init script EXITED. Its only failure path is
#     exec /bin/sh >/dev/console 2>&1          (init:164, init:174, init:1519)
# and a shell that reads EOF - a serial console with no cable, a tty1 nobody is typing at - exits 0, taking
# PID 1 with it. So the failure was handled where no one could see it: no panic screen, no log, no console.
#
# The patch does two things to init and nothing else:
#   1. installs one_rescue() and `trap 'one_rescue $?' EXIT`, so the script ending for ANY reason (a fatal
#      error, `exit` from a sourced file, an `exec` that fails) reaches the rescue instead of exiting PID 1;
#   2. replaces the three `exec /bin/sh >/dev/console 2>&1` failure paths with `one_rescue 1` - exec does not
#      run EXIT traps, so the trap alone would not have covered the paths that actually fire.
# one_rescue writes /tmp/bootinit.log + dmesg + /proc/cmdline to `<LABEL=1BIT-DATA>/init-fail.log` (that
# partition is mountable by this init: it is the one the frugal install lives on) and then keeps a shell on
# /dev/console alive for ever - a shell that exits immediately (EOF) is re-spawned every 5 s, so PID 1 never
# goes away. The board's JetKVM showed the panic and nothing else; the next boot of a broken stick shows
# /init-fail.log on the data partition and a console.
#
# Exit: 0 patched (or already patched) and verified, 1 the init is not the shape this patch understands,
# 2 usage / the tree is not there.
set -uo pipefail
prog=$(basename "$0")
die() { echo "$prog: $*" >&2; exit 1; }

# the strings the patch inserts and the gate greps for
MARK='1bit: PID 1 must never exit'
FUNC_MARK='one_rescue()'

# The console's progress lines (cycle 14, item 6). The board boot sat on the kernel log for a long time with
# nothing saying why: with pfix=copy every SFS is copied into RAM at the medium's speed before the system is
# up, and that is the slow part. Two lines, timestamped from /proc/uptime, so the wait has a name on the screen
# and the duration is measurable afterwards (the VM rows read both).
PROG1='*** 1bit: loading the system'
PROG2='*** 1bit: system in RAM after'

# engine#865: Puppy's init resolves a partition by name or by a bare blkid token, but not by the `UUID=`/`LABEL=`
# form - and the installer writes `pupsfs=UUID=<uuid>` so an installed disk can find its own data partition once two
# 1BIT-DATA partitions exist (engine#539/#711). --verify greps the packed initrd for this marker.
UUID_MARK='1bit: pupsfs=UUID'

# engine#1159: a slow USB medium is not a failure. Puppy's wait_for_blkid() polls blkid 15 times, one second apart,
# and then gives up - which the cycle-26 board boot hit: the stick's own SCSI probe did not finish until 45.7 s
# (kernel log: `scsi 0:0:0:0: Direct-Access General UDisk` at 3.8 s, two `usb 3-2: reset high-speed USB device`
# cycles from the JetKVM's emulated media on the same xHCI controller, `sda: sda1 sda2` at 45.7 s), so the screen
# said `Waiting for partition [1BIT-DATA]............... failed` on a medium blkid had already called healthy - and
# every gate was green because QEMU answers in 3.7 s. The budget is now 120 s and the wait says out loud that it is
# still waiting, so a slow medium is a slow boot instead of a failure.
SLOW_MARK='1bit: a slow medium is not a failure'
PART_TRIES=120
PART_NEED=90

rescue_block() {
cat <<'ONEBLOCK'
# 1bit: say it on every console a person might be watching. Since cycle 15 the boot line ends with
# `console=ttyS0,115200 console=tty0`, and the LAST console= is what /dev/console resolves to - the screen. That
# fixed the owner's "it hangs hard here" (the progress lines had been going to the serial port only) and broke the
# other half: a userspace write to /dev/console no longer reaches ttyS0, so a headless box on a serial cable - and
# the VM harness, which drives the guest over ttyS0 - saw nothing. Measured on the cycle-15 image: `U8 boot
# progress` failed with "the console never said what it was doing" while the same lines were on tty0.
one_console() {
	for ONE_C in /dev/console /dev/ttyS0; do
		# the initramfs carries no /dev at all: the kernel creates /dev/console for init itself, and nothing
		# creates the serial node - so it is mknod'd here (ttyS0 is 4,64), which is what any initrd with a serial
		# console has to do. After switch_root the real devtmpfs replaces this directory.
		[ -e "$ONE_C" ] || { [ "$ONE_C" = /dev/ttyS0 ] && { mkdir -p /dev; mknod "$ONE_C" c 4 64 2>/dev/null; }; }
		[ -e "$ONE_C" ] && printf '%s\n' "$*" > "$ONE_C" 2>/dev/null
	done
	return 0
}
# 1bit: PID 1 must never exit. Measured on the 2026-11-02 board boot: this script's failure path is
# `exec /bin/sh`, a shell that reads EOF exits 0, and PID 1 exiting is a kernel panic
# ("Attempted to kill init! exitcode=0x00000000"). one_rescue is the only way out now.
ONE_RESCUE_LABEL='1BIT-DATA'
ONE_RESCUE_MT='/mnt/1bit-rescue'
one_rescue() {
	ONE_STATUS=${1:-0}
	one_console ""
	one_console "*** 1bit: init stopped (status $ONE_STATUS) - logging, and this console stays"
	ONE_P=$(blkid 2>/dev/null | grep -m1 "LABEL=\"$ONE_RESCUE_LABEL\"" | cut -d: -f1)
	[ "$ONE_P" ] || { sleep 3; ONE_P=$(blkid 2>/dev/null | grep -m1 "LABEL=\"$ONE_RESCUE_LABEL\"" | cut -d: -f1); }
	if [ "$ONE_P" ] && mkdir -p $ONE_RESCUE_MT && mount "$ONE_P" $ONE_RESCUE_MT 2>/dev/null; then
		{
			echo "=== 1bit init failure, status $ONE_STATUS, $(date 2>/dev/null)"
			echo "--- /proc/cmdline"; cat /proc/cmdline 2>/dev/null
			echo "--- /tmp/bootinit.log"; tail -n 300 /tmp/bootinit.log 2>/dev/null
			echo "--- dmesg"; dmesg 2>/dev/null
		} > $ONE_RESCUE_MT/init-fail.log 2>/dev/null
		sync
		one_console "*** 1bit: log written to $ONE_RESCUE_LABEL:/init-fail.log"
	else
		one_console "*** 1bit: $ONE_RESCUE_LABEL not mountable - the log is only in /tmp/bootinit.log"
	fi
	while : ; do
		one_console "*** 1bit rescue: this console stays alive. 'reboot -f' restarts; the log is on" \
			"$ONE_RESCUE_LABEL:/init-fail.log"
		/bin/sh </dev/console >/dev/console 2>&1
		sleep 5
	done
}
trap 'one_rescue $?' EXIT
ONEBLOCK
}

verify_initrd() {
    local gz tmp n
    gz=$1
    [ -f "$gz" ] || die "no such initrd: $gz"
    # absolute before the cd below: a relative initrd path plus `cd "$tmp"` extracts nothing and looks
    # exactly like an archive with no /init (measured - it refused a good build)
    gz=$(cd "$(dirname "$gz")" && pwd)/$(basename "$gz")
    tmp=$(mktemp -d) || exit 2
    ( cd "$tmp" && zcat "$gz" | cpio -idm --quiet init ) 2> /dev/null
    [ -f "$tmp/init" ] || { rm -rf "$tmp"; die "$gz carries no /init"; }
    n=$(grep -c 'one_rescue 1' "$tmp/init")
    if ! grep -q "$MARK" "$tmp/init" || ! grep -q "$FUNC_MARK" "$tmp/init" || ! grep -q "trap 'one_rescue" "$tmp/init"; then
        rm -rf "$tmp"; die "$gz: /init has no 1bit rescue guard (panic on any init failure) - patch-init.sh did not run"
    fi
    [ "$n" = 3 ] || { rm -rf "$tmp"; die "$gz: /init has $n of the 3 failure paths guarded"; }
    grep -qF "$PROG1" "$tmp/init" && grep -qF "$PROG2" "$tmp/init" || {
        rm -rf "$tmp"; die "$gz: /init has no 1bit boot-progress lines (the console says nothing while the SFS copy to RAM runs)"; }
    # and they (and the rescue messages) have to reach BOTH consoles: /dev/console is the screen now, and a
    # headless box on a serial cable - the VM rows, and the panic-repro harness - reads ttyS0
    grep -qF 'one_console()' "$tmp/init" || {
        rm -rf "$tmp"; die "$gz: /init has no one_console (the progress and rescue lines reach one console only)"; }
    grep -qF "one_console \"$PROG1" "$tmp/init" || {
        rm -rf "$tmp"; die "$gz: /init writes the progress lines to /dev/console only (nothing on ttyS0 for a headless box)"; }
    # engine#865: an installed disk boots with pupsfs=UUID=<uuid>; an init that cannot resolve that form drops to the
    # rescue console instead of booting, so the packed initrd has to carry the resolver too
    grep -qF "$UUID_MARK" "$tmp/init" || {
        rm -rf "$tmp"; die "$gz: /init cannot resolve pupsfs=UUID=<uuid> (engine#865: an installed disk would not boot)"; }
    # engine#1159: and its partition wait has to outlast a slow USB medium - see SLOW_MARK above
    grep -qF "$SLOW_MARK" "$tmp/init" && grep -qF 'one_console "*** 1bit: still waiting' "$tmp/init" || {
        rm -rf "$tmp"; die "$gz: /init's partition wait is the stock 15 s and says nothing while it waits (engine#1159: the board needed 45.7 s)"; }
    tries=$(sed -n 's/^[[:space:]]*for NUM in \$(seq 1 \([0-9]\+\)); do$/\1/p' "$tmp/init" | head -1)
    [ "${tries:-0}" -ge "$PART_NEED" ] || {
        rm -rf "$tmp"; die "$gz: /init waits ${tries:-0}s for the partition, less than the ${PART_NEED}s this board needs (engine#1159)"; }
    rm -rf "$tmp"
    echo "ok: $gz: /init carries the guard, all 3 failure paths reach it, both progress lines, the pupsfs=UUID resolver,"
    echo "    and a ${tries}s partition wait that says it is still waiting"
}

if [ "${1:-}" = "--verify" ]; then
    [ $# -eq 2 ] || { sed -n "/^# $prog --verify/,/^# 2 usage/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
    verify_initrd "$2"
    exit 0
fi
[ $# -eq 1 ] || { sed -n "/^# $prog <build-root>/,/^# 2 usage/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

root=$1
# 3builddistro packs ../initrd-progs/0initrd (woof-code/3builddistro:649-657), so that is the file to patch;
# the woof-CE master copy is patched too when it is there, because a fresh merge2out starts from it.
shopt -s nullglob
cands=("$root"/woof-out_*/initrd-progs/0initrd/init)
shopt -u nullglob
[ ${#cands[@]} -eq 1 ] || die "expected exactly one $root/woof-out_*/initrd-progs/0initrd/init (found ${#cands[@]})"
init=${cands[0]}
[ -f "$init" ] || die "no such init: $init"
master=$root/woof-CE/initrd-progs/0initrd/init
[ -f "$master" ] || master=''

# "Already patched" is only true for THIS version of the block: a block inserted by an older patch-init carries the
# old messages (cycle 15: the progress lines and the rescue text went to /dev/console alone, which the new console
# order turned into "the screen only"). The block is always inserted at line 2, so an older one is dropped from
# line 2 to its `trap ... one_rescue` line and the current one installed in its place. Measured twice: refusing
# instead of re-patching cost two whole builds, because the refusal happens after the ISO is written.
strip_old_block() {   # strip_old_block <file> -> the file without a previously inserted rescue block
    awk 'NR == 1 { print; next } $0 ~ /^trap .one_rescue/ && !seen { seen = 1; next } !seen { next } { print }' "$1"
}
if grep -q "$MARK" "$init" && grep -qF 'one_console()' "$init" && \
   [ "$(grep -c 'one_rescue 1' "$init")" = 3 ]; then
    echo "== init guard: already patched ($init)"
else
    if grep -q "$MARK" "$init"; then
        echo "== init guard: an OLDER block is in $init - replacing it with the current one"
        tmp=$(mktemp) || exit 2
        strip_old_block "$init" > "$tmp" && cat "$tmp" > "$init" && rm -f "$tmp"
        # the `&` in 2>&1 is sed's "the whole match" in a replacement, so it has to be escaped - unescaped it
        # rewrote the three paths into `... 2one_rescue 11` and the insert path below then refused the file it had
        # just produced (measured while re-patching cycle 15d's initrd)
        sed -i 's|one_rescue 1|exec /bin/sh >/dev/console 2>\&1|g' "$init"
    fi
    echo "== init guard into $init"
    grep -q '^#!' "$init" || die "$init is not a script"
    grep -q 'exec /bin/sh >/dev/console 2>&1' "$init" || die "$init has none of the 3 failure paths - upstream changed, re-measure"
    tmp=$(mktemp) || exit 2
    { sed -n '1p' "$init"; rescue_block; sed -n '2,$p' "$init"; } > "$tmp" || exit 1
    sed -i 's|exec /bin/sh >/dev/console 2>&1|one_rescue 1|g' "$tmp" || exit 1
    n=$(grep -c 'one_rescue 1' "$tmp")
    [ "$n" = 3 ] || { rm -f "$tmp"; die "the patch landed on $n of 3 failure paths"; }
    cat "$tmp" > "$init" || exit 1
    rm -f "$tmp"
    sh -n "$init" || die "the patched init does not parse (dash is the closest thing to busybox sh here)"
    echo "   three failure paths now reach one_rescue; trap installed on EXIT"
fi

# the two console progress lines, patched separately so the guard above stays greppable and idempotent
patch_progress() {
    local f=$1 tmp
    # Current form? Both progress lines written through one_console. Testing for "one_console" anywhere is not
    # enough - the rescue block's own messages use it, so a file with the OLD `echo ... > /dev/console` progress
    # lines passed that test and shipped with the lines on one console only (measured re-patching cycle 15d).
    if grep -qF "one_console \"$PROG1" "$f" && grep -qF "one_console \"$PROG2" "$f"; then return 0; fi
    # An older patch's two lines, upgraded in place: the redirect goes, one_console goes on. Done with awk so the
    # *** in the strings is not a regex, and so nothing else in the file is touched.
    if grep -qF "$PROG1" "$f" || grep -qF "$PROG2" "$f"; then
        tmp=$(mktemp) || return 2
        awk -v p1="$PROG1" -v p2="$PROG2" '
            index($0, p1) || index($0, p2) {
                line = $0
                sub(/^[ \t]*echo /, "", line)
                sub(/ > \/dev\/console$/, "", line)
                print "\tone_console " line
                next
            }
            { print }
        ' "$f" > "$tmp" && cat "$tmp" > "$f"
        rm -f "$tmp"
        grep -qF "one_console \"$PROG1" "$f" && grep -qF "one_console \"$PROG2" "$f" && return 0
    fi
    grep -q '^#- handle pfix=copy pfix=ram pfix=nocopy' "$f" || {
        echo "$prog: $f has no pfix=copy block - upstream changed, re-measure" >&2; return 1; }
    grep -q '^echo "PUPMODE=\$PUPMODE"' "$f" || {
        echo "$prog: $f has no PUPMODE line - upstream changed, re-measure" >&2; return 1; }
    tmp=$(mktemp) || return 2
    # the lines carry no single quotes on purpose: this awk program is single-quoted, and three of the last four
    # build-side bugs in this lane were a quote inside a quoted block. `read up _ < /proc/uptime` avoids the
    # -d' ' that would have needed one.
    awk '
        /^#- handle pfix=copy pfix=ram pfix=nocopy/ && !d1 {
            print "\t# 1bit: say what the console is waiting for (pfix=copy copies the SFS into RAM at the medium speed)"
            print "\tread up _ < /proc/uptime"
            print "\tone_console \"'"$PROG1"' at ${up}s: with pfix=copy the SFS are copied into RAM, which is the slow part on USB\""
            d1 = 1
        }
        /^echo "PUPMODE=\$PUPMODE"/ && !d2 {
            print "\tread up _ < /proc/uptime"
            print "\tone_console \"'"$PROG2"' ${up}s; PUPMODE=$PUPMODE\""
            d2 = 1
        }
        { print }
    ' "$f" > "$tmp" || { rm -f "$tmp"; return 1; }
    grep -qF "one_console \"$PROG1" "$tmp" && grep -qF "one_console \"$PROG2" "$tmp" || { rm -f "$tmp"; return 1; }
    cat "$tmp" > "$f" && rm -f "$tmp"
}
# engine#865: wait_for_blkid() builds its blkid match out of the token it is given, which is a device name or a bare
# label/UUID. With `UUID=<uuid>` the generated pattern was " UUID=.UUID=<uuid>" - it can never match, so an installed
# system could not find the partition it had just booted from: measured "Partition UUID=... not found.", init exits
# 1, rescue console. Prefixed tokens are matched against blkid's own UUID="..."/LABEL="..." fields (quotes optional:
# util-linux and busybox blkid both print them today).
patch_pupsfs_uuid() {
    local f=$1 tmp prog
    grep -qF "$UUID_MARK" "$f" && return 0
    grep -qF 'PART="$(blkid | grep "$1"' "$f" || {
        echo "$prog: $f has no wait_for_blkid blkid chain - upstream changed, re-measure" >&2
        return 1; }
    tmp=$(mktemp) || return 2
    prog=$(mktemp) || return 2
    # the awk program is a quoted here-doc: no shell expansion, so awk's own escaping is written as-is
    cat > "$prog" <<'ONEAWK'
index($0, "PART=\"$(blkid | grep \"$1\"") && !done {
    print "\t# " mark ": a prefixed token is a blkid FIELD, not a substring (engine#865)"
    print "\tcase \"$1\" in"
    print "\t UUID=*)  w=\"${1#UUID=}\""
    print "\t          PART=\"$(blkid 2>/dev/null | grep -E \" UUID=\\\"?${w}\\\"?( |$)\" | cut -f1 -d: | cut -f3 -d/)\" ;;"
    print "\t LABEL=*) w=\"${1#LABEL=}\""
    print "\t          PART=\"$(blkid 2>/dev/null | grep -E \" LABEL=\\\"?${w}\\\"?( |$)\" | cut -f1 -d: | cut -f3 -d/)\" ;;"
    print "\t *) PART=\"$(blkid | grep \"$1\" | grep -E \"^/dev/$1| LABEL=.$1| UUID=.$1\" | cut -f1 -d: | cut -f3 -d/)\" ;;"
    print "\tesac"
    done = 1
    next
}
{ print }
ONEAWK
    # the matched line is printed OVER by the case block above (print + next), so it is replaced, not duplicated
    awk -v mark="$UUID_MARK" -f "$prog" "$f" > "$tmp" || { rm -f "$tmp" "$prog"; return 1; }
    rm -f "$prog"
    grep -qF "$UUID_MARK" "$tmp" || { rm -f "$tmp"; return 1; }
    cat "$tmp" > "$f" && rm -f "$tmp"
    return 0
}
if ! patch_pupsfs_uuid "$init"; then
    die "could not teach init to resolve pupsfs=UUID=<uuid> (engine#865)"
fi
echo "   wait_for_blkid resolves UUID= and LABEL= tokens (engine#865)"

# engine#1159: the stock wait gives up after 15 s and says nothing while it waits. Measured on the cycle-26 board
# boot: healthy medium, partition published at 45.7 s, init gave up at 15 s -> `Waiting for partition [...]... failed`.
patch_slow_medium() {
    local f=$1 tmp
    grep -qF "$SLOW_MARK" "$f" && return 0
    grep -qF 'for NUM in $(seq 1 ' "$f" || {
        echo "$prog: $f has no wait_for_blkid poll loop - upstream changed, re-measure" >&2; return 1; }
    tmp=$(mktemp) || return 2
    awk -v tries="$PART_TRIES" -v mark="$SLOW_MARK" '
        index($0, "for NUM in $(seq 1 ") && !d1 {
            sub(/\(seq 1 [0-9]+\)/, "(seq 1 " tries ")")
            print
            d1 = 1
            next
        }
        index($0, "#yellow dot") && !d2 {
            print "\t# " mark ": the kernel can take tens of seconds to publish the partition on USB, and a wait"
            print "\t# that gives up quietly looks exactly like a hang (cycle-26 board boot: 45.7 s, engine#1159)"
            print "\t[ $NUM -eq 15 ] && one_console \"*** 1bit: still waiting for [$1] - a slow medium is not a failure (the kernel has not published the partition yet)\""
            print
            print
            d2 = 1
            next
        }
        { print }
    ' "$f" > "$tmp" || { rm -f "$tmp"; return 1; }
    grep -qF "$SLOW_MARK" "$tmp" && grep -qF 'one_console "*** 1bit: still waiting' "$tmp" || { rm -f "$tmp"; return 1; }
    cat "$tmp" > "$f" && rm -f "$tmp"
}
if ! patch_slow_medium "$init"; then
    die "could not make init's partition wait outlast a slow medium (engine#1159)"
fi
echo "   partition wait: ${PART_TRIES}s, and it says so at 15 s instead of looking hung (engine#1159)"

if ! patch_progress "$init"; then
    die "could not add the 1bit boot-progress lines to $init"
fi
echo "   two console progress lines: the copy to RAM, and its duration"
if [ "$master" ]; then
    if ! grep -q "$MARK" "$master"; then
        echo "== init guard into the woof-CE master copy ($master)"
        tmp=$(mktemp) || exit 2
        { sed -n '1p' "$master"; rescue_block; sed -n '2,$p' "$master"; } > "$tmp" || exit 1
        sed -i 's|exec /bin/sh >/dev/console 2>&1|one_rescue 1|g' "$tmp" || exit 1
        cat "$tmp" > "$master" && rm -f "$tmp"
    fi
fi
exit 0
