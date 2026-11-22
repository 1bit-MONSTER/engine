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
# kernel-quirks-check.sh <kernel-stage> <kver>
#
# The device-table rows the 1bit kernel carries (third_party/patches/linux), checked on a built kernel stage
# (<stage>/lib/modules/<kver>/). A row is present when the driver lists the device's PCI alias:
#   built-in driver (=y):  modules.builtin.modinfo has  <driver>.alias=pci:v0000VVVVd0000DDDD...
#   module (=m):           modules.alias has           alias pci:v0000VVVVd0000DDDD... <driver>
# The alias proves the row exists, not its driver_data; each row below exists only for its flag.
#
# Exit: 0 every row present, 1 usage, 2 refused (a row is missing).
set -euo pipefail
prog=kernel-quirks-check
[ $# -eq 2 ] || { sed -n '/^# kernel-quirks-check.sh/,/^# Exit:/p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 1; }
mods=$1/lib/modules/$2
[ -d "$mods" ] || { printf '%s: REFUSED: no %s\n' "$prog" "$mods" >&2; exit 2; }

# driver vendor device why
rows='nvme 1e49 1081 YMTC PC41Q needs NVME_QUIRK_DMAPOOL_ALIGN_512 (engine#2478, engine#3088)'

missing=0
while read -r drv ven dev why; do
    [ -n "$drv" ] || continue
    alias=$(printf 'pci:v0000%sd0000%s' "${ven^^}" "${dev^^}")
    if { [ -f "$mods/modules.builtin.modinfo" ] &&
            tr '\0' '\n' < "$mods/modules.builtin.modinfo" | grep -q "^$drv\.alias=$alias"; } ||
        { [ -f "$mods/modules.alias" ] && grep -q "^alias $alias[^ ]* $drv$" "$mods/modules.alias"; }; then
        printf '   %s lists %s:%s (%s)\n' "$drv" "$ven" "$dev" "$why"
    else
        printf '%s: REFUSED: %s does not list %s:%s - %s; apply third_party/patches/linux and rebuild\n' \
            "$prog" "$drv" "$ven" "$dev" "$why" >&2
        missing=$((missing + 1))
    fi
done <<< "$rows"
[ "$missing" -eq 0 ] || exit 2
