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
# pack_amdgpu_objects_test.sh: the binary drop's "HIP code objects are compiled in" check (scripts/pack-elf.sh,
# pack_amdgpu_objects -- engine#1443).
#
# make-binary-drop.sh used to count the string "amdgcn-amd-amdhsa" in libggml-hrx.so. The release build's library
# carries 54 embedded AMDGPU code objects and not that string, so the drop refused every real build; a 2-object
# build happened to carry it and passed. The check now counts embedded ELF64 headers whose e_machine is EM_AMDGPU
# (224). Fixtures, no build:
#
#   1. a host library with embedded AMDGPU code objects and no triple string counts them (the c31/release shape);
#   2. the triple string alone, with no code object, counts 0 (a string must not stand in for the property);
#   3. the outer x86-64 ELF is not counted, and a 32-bit header with e_machine 224 is not counted;
#   4. a code-object header whose skipped bytes hold a newline is still counted (grep -P without (?s) misses it);
#   5. make-binary-drop.sh really calls the helper and no longer greps the string.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=../scripts/pack-elf.sh
. "$here/../scripts/pack-elf.sh"

fails=0
check() { if [ "$2" = "$3" ]; then printf '  ok   %s = %s\n' "$1" "$3"; else printf '  FAIL %s = %s (want %s)\n' "$1" "$2" "$3"; fails=$((fails+1)); fi; }

d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
# 20 header bytes: magic, class, data, version, OSABI, then 10 pad/abi bytes, e_type (2), e_machine (2)
host()   { printf '\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\x00\x3e\x00'; }
amdgpu() { printf '\x7fELF\x02\x01\x01\x40\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\xe0\x00'; }
filler() { head -c "$1" /dev/zero; }

{ host; filler 4096; amdgpu; filler 300; amdgpu; filler 300; amdgpu; filler 100; } > "$d/kernels.so"
check "three embedded code objects, no triple string" "$(pack_amdgpu_objects "$d/kernels.so")" 3
check "  (the old string check would have refused it)" "$(strings -a "$d/kernels.so" | grep -c amdgcn-amd-amdhsa)" 0

{ host; filler 4096; printf 'hipv4-amdgcn-amd-amdhsa--gfx1151\0'; filler 100; } > "$d/string-only.so"
check "the triple string alone" "$(pack_amdgpu_objects "$d/string-only.so")" 0

{ host; filler 4096; printf '\x7fELF\x01\x01\x01\x40\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\xe0\x00'; filler 100; } > "$d/host-only.so"
check "the outer host ELF and a 32-bit e_machine-224 header" "$(pack_amdgpu_objects "$d/host-only.so")" 0

{ host; filler 64; printf '\x7fELF\x02\x01\x01\x40\x03\x0a\x0a\x00\x00\x00\x00\x00\x03\x00\xe0\x00'; filler 64; } > "$d/newline.so"
check "a code object whose header holds a newline byte" "$(pack_amdgpu_objects "$d/newline.so")" 1

check "a missing file counts 0" "$(pack_amdgpu_objects "$d/absent.so")" 0

drop=$here/../scripts/make-binary-drop.sh
check "make-binary-drop.sh calls pack_amdgpu_objects" \
    "$(grep -cE '^[[:space:]]*n=\$\(pack_amdgpu_objects ' "$drop")" 1
check "make-binary-drop.sh no longer greps the triple string" "$(grep -cE '^[^#]*grep -c amdgcn-amd-amdhsa' "$drop")" 0

if [ "$fails" -ne 0 ]; then printf '%d check(s) failed\n' "$fails"; exit 1; fi
echo "pack_amdgpu_objects: ok"
