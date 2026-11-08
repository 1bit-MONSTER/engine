#!/usr/bin/env bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# pack_check_generated_text_test.sh - build_path_strings must red a leaking generated text file, not only a leaking ELF.
#
# Why: the guard walked every packed file with `pack_is_elf ... || continue`, so compiler output was invisible to it. A
# shipped `devicetable-offsets.s` carrying 449 absolute build-tree paths passed the row that exists to stop exactly that
# (engine#1487). The extension is bounded on purpose - `.s`, `.cmd`, `.mod.c`, `Module.symvers`, `modules.order`, `*.d` -
# because a licence or a man page may legitimately name a path and a guard that reds those is a guard people switch off.
set -uo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
. "$here/scripts/pack-check.sh"

work=$(mktemp -d /tmp/pcheck.XXXXXX) || exit 1
trap 'rm -rf "$work"' EXIT

fails=0; checks=0
check() { checks=$((checks + 1)); if [ "$2" = "$3" ]; then printf 'ok   %s\n' "$1"; else printf 'FAIL %s: got %s want %s\n' "$1" "$2" "$3"; fails=$((fails + 1)); fi; }

mkdir -p "$work/red/scripts/mod" "$work/green/scripts/mod" "$work/licence"
# the shape a real one has: `.file "<abs path>"` directives, one per translation unit
printf ' .file "devicetable-offsets.c"\n# 13 "/home/builder/linux-7.4/scripts/mod/devicetable-offsets.c" 1\n# 0 "" 2\n' \
    > "$work/red/scripts/mod/devicetable-offsets.s"
printf ' .file "devicetable-offsets.c"\n# 13 "linux/scripts/mod/devicetable-offsets.c" 1\n# 0 "" 2\n' \
    > "$work/green/scripts/mod/devicetable-offsets.s"
# a text file OUTSIDE the bounded set: this is the case the list must not grow into, because licences name paths
printf 'This product includes software developed at /home/someone/project.\n' > "$work/licence/LICENSE"

check "a leaking .s is refused" "$(build_path_strings "$work/red" > /dev/null 2>&1; echo $?)" "1"
check "and it is named, with a path" \
    "$(build_path_strings "$work/red" 2>&1 | grep -c 'devicetable-offsets.s names a build-machine path')" "1"
check "a clean .s passes" "$(build_path_strings "$work/green" > /dev/null 2>&1; echo $?)" "0"
check "the pass line counts what it looked at" \
    "$(build_path_strings "$work/green" 2>&1 | grep -c '1 generated text')" "1"
check "a licence naming a path is NOT refused (the bounded set)" \
    "$(build_path_strings "$work/licence" > /dev/null 2>&1; echo $?)" "0"
check "gen_text accepts the six generated shapes" \
    "$(for n in a.s b.cmd c.mod.c Module.symvers modules.order d.d; do gen_text "$n" && printf y; done)" "yyyyyy"
# The two shapes a prefix map CANNOT fix are accepted on purpose: a red on them means a build tree was packaged, and the
# remedy is to stop packaging it. This check exists so that distinction is not "tidied away" by someone narrowing the list.
check "the shapes only 'do not package the tree' can fix are accepted, not narrowed out" \
    "$(for n in x.cmd y.d; do gen_text "$n" && printf y; done)" "yy"
check "gen_text refuses a licence and a man page" \
    "$(for n in LICENSE NOTICE 1bit.1 README.md; do gen_text "$n" && printf y; done)" ""

echo "pack_check_generated_text: $checks checks, $fails failed"
[ "$fails" = 0 ] || exit 1
