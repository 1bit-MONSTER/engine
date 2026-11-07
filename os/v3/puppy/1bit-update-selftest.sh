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
# 1bit-update-selftest.sh - the second-update logic of 1bit-update, on fake /data trees, with no VM and no disk.
#
# engine#1376: `1bit-update` let an installed machine take ONE update. The second was refused on the machine's own save
# layer, because after update 1 the machine runs the alt slot while the layer still lives in /1bit - so the second update's
# target IS the slot holding the real layer, and the "a slot that already owns a save layer is another machine's state"
# guard walked into it. `update-gate.sh` cannot see this: its boots are update -> boot alt -> rollback -> boot original, so it
# never runs a second update.
#
# The four cases, each on a tree built to be the shape the real machine has at that moment:
#
#   A  one layer, linked into the other slot (what update 1 leaves behind)  -> must PROCEED: the layer found in the target
#      slot is the same file as this machine's, so it is kept, the layer is not moved, and NO self-referencing link is created
#      (the latent defect #1334's review found: linking a layer into the slot it really lives in)
#   B  a genuinely different layer in the target slot                        -> must REFUSE, naming which is the machine's own
#   C  a foreign real layer in the OTHER slot, the machine's own in the active -> the scan must pick the ACTIVE slot's layer,
#      not the last real directory it happens to see
#   D  no layer at all                                                       -> must proceed (nothing to keep, nothing to link)
#
# The three blocks under test are lifted out of the script itself, between the section markers it already carries, so this
# tests the shipped code rather than a copy of it. If the markers ever move, the extraction is empty and the selftest fails
# loudly (case 0) instead of silently passing nothing.
#
# Usage: 1bit-update-selftest.sh [path to 1bit-update]     Exit: 0 all cases pass, 1 a case failed, 2 could not run.
set -u

here=$(cd "$(dirname "$0")" && pwd)
script=${1:-$here/embed/usr/local/bin/1bit-update}
[ -f "$script" ] || { echo "1bit-update-selftest: no such script: $script" >&2; exit 2; }
command -v mktemp > /dev/null 2>&1 || { echo "1bit-update-selftest: mktemp is missing" >&2; exit 2; }

work=$(mktemp -d "${TMPDIR:-/tmp}/1bu-selftest.XXXXXX") || exit 2
trap 'chmod -R u+w "$work" 2> /dev/null; rm -rf "$work"' EXIT

# ── lift the three blocks out of the script under test ────────────────────────────────────────────────────────────────
blocks=$work/blocks.sh
awk '
    # two contiguous blocks, each ending at the same marker: the scan, then the guard + write + link. Fragment-based
    # extraction (stopping at a line that appears twice) cut a for-loop mid-body and produced a syntax error - the failure
    # was loud, but it wasted a run; contiguous ranges cannot do that.
    /^save_slot=$/                                  { scan = 1 }
    scan && /^# ── GRUB/                            { scan = 0 }
    scan                                            { print }

    /^# a slot that already owns a save layer/      { act = 1 }
    act && /^# ── GRUB/                             { act = 0; print "" }
    act                                             { print }
' "$script" > "$blocks"

fail=0 ok=0
check() {  # <want: ok|die> <name> <data dir> <active> <target> [grep for a line that must appear]
    want=$1 name=$2 data=$3 active=$4 target=$5 say_grep=${6:-}
    out=$(env -i PATH=/usr/bin:/bin DATA="$data" active="$active" target="$target" ALT=1bit-alt \
          LOG="$work/log" BLOCKS="$blocks" SELFTEST_SRC="$SELFTEST_SRC" sh "$work/harness.sh" 2>&1)
    rc=$?
    case $want in
        ok)
            if [ "$rc" = 0 ] && { [ -z "$say_grep" ] || printf '%s' "$out" | grep -q "$say_grep"; }; then
                printf 'PASS %s\n' "$name"; ok=$((ok + 1))
            else
                printf 'FAIL %s: expected success%s, got rc=%s\n%s\n' "$name" "${say_grep:+ and '$say_grep'}" "$rc" "$out"
                fail=$((fail + 1))
            fi
            ;;
        die)
            if [ "$rc" = 2 ] && printf '%s' "$out" | grep -q 'DIE(rc 2)'; then
                printf 'PASS %s\n' "$name"; ok=$((ok + 1))
            else
                printf 'FAIL %s: expected the refusal (rc 2), got rc=%s\n%s\n' "$name" "$rc" "$out"
                fail=$((fail + 1))
            fi
            ;;
    esac
    [ -f "$work/log" ] && : > "$work/log"
}

# the harness the blocks run inside: the two commands they call, and a source tree with one file to copy
cat > "$work/harness.sh" <<'H'
say() { printf 'say: %s\n' "$*"; }
die() { printf 'DIE(rc 2): %s\n' "$*" >&2; exit 2; }
fail() { printf 'FAIL(rc 3): %s\n' "$*" >&2; exit 3; }
src=$SELFTEST_SRC
need="vmlinuz"
. "$BLOCKS"
H
SELFTEST_SRC=$work/src
export SELFTEST_SRC
mkdir -p "$SELFTEST_SRC"
# NOT empty: the script's own pre-check is `[ -s "$src/$f" ]` (size > 0), so an empty fixture fails a check that has
# nothing to do with the case under test - measured, that is how this line was found
printf 'vmlinuz fixture\n' > "$SELFTEST_SRC/vmlinuz"

mk() {  # <dir> <shape>
    d=$1
    rm -rf "$d"
    mkdir -p "$d/1bit" "$d/1bit-alt"
    case $2 in
        linked)   mkdir -p "$d/1bit/upup64rsave"; ln -s ../1bit/upup64rsave "$d/1bit-alt/upup64rsave" ;;
        foreign)  mkdir -p "$d/1bit/foreignsave"; ln -s ../1bit/foreignsave "$d/1bit-alt/foreignsave" ;;
        two)      mkdir -p "$d/1bit/foreignsave" "$d/1bit-alt/mysave" ;;
        none)     : ;;
    esac
    mkdir -p "$d/1bit"
}

# ── case 0: the extraction itself ───────────────────────────────────────────────────────────────────────────────────
if [ -s "$blocks" ] && grep -q 'savename=' "$blocks" && grep -q 'ln -sfn' "$blocks"; then
    printf 'PASS %s\n' 'the three blocks were found in the script under test'; ok=$((ok + 1))
else
    printf 'FAIL %s: the marker-based extraction found nothing in %s - the selftest would pass nothing\n' 'extraction' "$script"
    fail=$((fail + 1))
fi

# ── A: the shape update 1 leaves behind - must proceed, keep the layer, create no self-link ─────────────────────────
mk "$work/a" linked
check ok  'A: second update proceeds on one linked layer' "$work/a" 1bit-alt 1bit 'own layer'
if [ -e "$work/a/1bit/upup64rsave/upup64rsave" ]; then
    printf 'FAIL %s: a self-referencing upup64rsave/upup64rsave was created inside the layer\n' 'A: no self-link'
    fail=$((fail + 1))
else
    printf 'PASS %s\n' 'A: no self-referencing link inside the layer'; ok=$((ok + 1))
fi
if [ -f "$work/a/1bit/upup64rsave/.keep" ] || [ -d "$work/a/1bit/upup64rsave" ]; then
    printf 'PASS %s\n' 'A: the layer is still where it lives'; ok=$((ok + 1))
else
    printf 'FAIL %s\n' 'A: the layer directory is gone'; fail=$((fail + 1))
fi

# ── B: a genuinely different layer in the target slot - must refuse ──────────────────────────────────────────────
mk "$work/b" two
check die 'B: a foreign layer in the target slot is refused' "$work/b" 1bit-alt 1bit

# ── C: a foreign real layer in the other slot - the scan must pick the ACTIVE slot's layer ────────────────────────
mk "$work/c" two
out=$(env -i PATH=/usr/bin:/bin DATA="$work/c" active=1bit-alt target=1bit ALT=1bit-alt LOG="$work/log" \
      BLOCKS="$blocks" SELFTEST_SRC="$SELFTEST_SRC" sh "$work/harness.sh" 2>&1)
if printf '%s' "$out" | grep -q 'not this machine.s (1bit-alt/mysave)'; then
    printf 'PASS %s\n' "C: the scan picked the active slot's layer, not the foreign one"; ok=$((ok + 1))
else
    printf 'FAIL %s\n%s\n' 'C: the scan did not name 1bit-alt/mysave as the machine layer' "$out"
    fail=$((fail + 1))
fi

# ── D: no layer at all - must proceed ────────────────────────────────────────────────────────────────────────────
mk "$work/d" none
check ok 'D: an update with no layer anywhere proceeds' "$work/d" 1bit 1bit-alt

printf '\n1bit-update-selftest: %s passed, %s failed\n' "$ok" "$fail"
[ "$fail" = 0 ] || exit 1
echo "RESULT: PASS"
