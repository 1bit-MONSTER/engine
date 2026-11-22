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
# engine#3088: the 1bit kernel carries the YMTC PC41Q's NVME_QUIRK_DMAPOOL_ALIGN_512 row
# (third_party/patches/linux/0001-*), and os/v3/puppy/kernel-quirks-check.sh refuses a kernel stage without it.
# Offline, no kernel tree:
#   - the patch adds exactly the PCI_DEVICE(0x1e49, 0x1081) row with NVME_QUIRK_DMAPOOL_ALIGN_512, to pci.c, and
#     nothing else (two + lines, no - lines);
#   - the check passes a built-in stage that lists the alias, a module stage that lists it in modules.alias, and
#     refuses a stage without it, one where another driver lists it, and a missing module tree;
#   - mk-huge-kernel.sh runs the check.
#
# usage: tests/kernel_quirks_test.sh [repo-root]
set -uo pipefail
root=${1:-$(cd "$(dirname "$0")/.." && pwd)}
chk=$root/os/v3/puppy/kernel-quirks-check.sh
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

shopt -s nullglob
patches=("$root"/third_party/patches/linux/0001-*.patch)
check "the nvme patch exists" '[ ${#patches[@]} -eq 1 ]'
p=${patches[0]:-/dev/null}
body=$(sed -n '/^diff --git /,$p' "$p")
check "it touches only drivers/nvme/host/pci.c" '[ "$(printf "%s\n" "$body" | grep -c "^diff --git")" = 1 ] && printf "%s\n" "$body" | grep -q "^+++ b/drivers/nvme/host/pci.c$"'
check "it adds two lines and removes none" '[ "$(printf "%s\n" "$body" | grep -c "^+[^+]")" = 2 ] && [ "$(printf "%s\n" "$body" | grep -c "^-[^-]")" = 0 ]'
check "the row is PCI_DEVICE(0x1e49, 0x1081)" 'printf "%s\n" "$body" | grep -q "^+[[:space:]]*{ PCI_DEVICE(0x1e49, 0x1081),"'
check "its driver_data is NVME_QUIRK_DMAPOOL_ALIGN_512" 'printf "%s\n" "$body" | grep -q "^+[[:space:]]*\.driver_data = NVME_QUIRK_DMAPOOL_ALIGN_512, },$"'

t=$(mktemp -d); trap 'rm -rf "$t"' EXIT
k=7.4.0-rc1-1bit
mk() { mkdir -p "$t/$1/lib/modules/$k"; }
mk builtin; printf 'nvme.alias=pci:v00001E49d00000041sv*sd*bc*sc*i*\0nvme.alias=pci:v00001E49d00001081sv*sd*bc*sc*i*\0' \
    > "$t/builtin/lib/modules/$k/modules.builtin.modinfo"
mk module; printf 'alias pci:v00001E49d00001081sv*sd*bc*sc*i* nvme\n' > "$t/module/lib/modules/$k/modules.alias"
mk without; printf 'nvme.alias=pci:v00001E49d00000041sv*sd*bc*sc*i*\0' > "$t/without/lib/modules/$k/modules.builtin.modinfo"
mk other; printf 'alias pci:v00001E49d00001081sv*sd*bc*sc*i* vfio_pci\n' > "$t/other/lib/modules/$k/modules.alias"
run() { bash "$chk" "$t/$1" "$k" > /dev/null 2>&1; echo $?; }
check "a built-in nvme that lists 1e49:1081 passes" '[ "$(run builtin)" = 0 ]'
check "a modular nvme that lists it in modules.alias passes" '[ "$(run module)" = 0 ]'
check "a kernel without the row is refused (2)" '[ "$(run without)" = 2 ]'
check "another driver's alias does not count (2)" '[ "$(run other)" = 2 ]'
check "a stage with no module tree is refused (2)" '[ "$(run missing)" = 2 ]'
check "the refusal names the patch directory" '{ bash "$chk" "$t/without" "$k" 2>&1 || true; } | grep -q "third_party/patches/linux"'
check "mk-huge-kernel.sh runs the check" 'grep -q "kernel-quirks-check.sh\" \"\$stage\" \"\$kver\"" "$root/os/v3/puppy/mk-huge-kernel.sh"'
exit $fail
