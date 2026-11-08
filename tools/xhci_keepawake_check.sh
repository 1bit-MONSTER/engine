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
# xhci_keepawake_check.sh -- does the shipped xHCI keepawake rule apply to THIS machine, and are the controllers awake?
#
# engine#823: on the Bosgame BeyondMax (AXB35-02) three of the four xHCI functions sit in D3cold on a stock boot, so the rear
# USB-A and both Type-C ports never enumerate; the gate is /etc/udev/rules.d/91-1bit-xhci-keepawake.rules, and the one thing the
# issue still owed was a *plug* test on a board boot. That test has a precondition nobody could check in one command: is the rule
# applicable to the machine under my hands, and are the controllers actually awake right now? This script answers both, read-only,
# and refuses to pretend on a machine the rule does not cover.
#
#   xhci_keepawake_check.sh                 # report: DMI match, per-controller state, verdict; exit 0
#   xhci_keepawake_check.sh --require-awake # exit 1 if an applicable controller is NOT in D0 (plug test would be void)
#   xhci_keepawake_check.sh --sysfs-root DIR # check a tree other than /sys (what the test uses)
#
# Read-only by default. `--apply` / `--revert` perform the rule's own single action (`power/control = on|auto`) on the
# controllers the rule matches, which needs root; they exist so the mechanism can be verified without a reboot, and `--revert`
# is the documented way back.
#
# Exit: 0 = checked (the verdict can be "not this board", "all awake", or "some asleep"), 1 = --require-awake not met,
#       2 = usage/IO problem. Never 0 by accident: a machine where the rule does not apply says so.
set -u

SYSFS=${SYSFS:-/sys}
MODE=report
while [ $# -gt 0 ]; do
    case "$1" in
        --require-awake) MODE=require; shift;;
        --apply)         MODE=apply; shift;;
        --revert)        MODE=revert; shift;;
        --sysfs-root)    SYSFS=${2:?--sysfs-root needs a directory}; shift 2;;
        -h|--help)       sed -n '17,40p' "$0"; exit 0;;
        *) echo "xhci_keepawake_check: unknown argument '$1'" >&2; exit 2;;
    esac
done

# The rule's own predicates, in one place. board_name is the DMI gate; vendor/device are the AMD Strix Halo xHCI functions the
# rule names (0x1587/0x1588/0x1589/0x158b); DRIVER must be xhci_hcd.
BOARD=${BOARD:-AXB35-02}
VENDOR=${VENDOR:-0x1022}
DEVICES=${DEVICES:-0x1587 0x1588 0x1589 0x158b}
DRIVER=${DRIVER:-xhci_hcd}

dmi_board=$(cat "$SYSFS/class/dmi/id/board_name" 2>/dev/null || true)
dmi_bios=$(cat "$SYSFS/class/dmi/id/bios_version" 2>/dev/null || true)
dmi_product=$(cat "$SYSFS/class/dmi/id/product_name" 2>/dev/null || true)

echo "== machine: board_name='${dmi_board:-?}'  bios='${dmi_bios:-?}'  product='${dmi_product:-?}'"
if [ "$dmi_board" != "$BOARD" ]; then
    echo "== verdict: NOT THIS BOARD -- the keepawake rule (board_name == '$BOARD') does not apply here, so nothing about it"
    echo "            can be concluded from this machine. engine#823's port finding is AXB35-02-specific."
    exit 0
fi
echo "== board matches the rule's DMI gate ('$BOARD')"

# Walk every PCI function, not a hardcoded list: a BIOS update can move them, and a hardcoded list is how a check silently stops
# checking anything. Only AMD xHCI functions whose device id the rule names are "applicable".
applicable=0
asleep=0
printf '%-14s %-8s %-10s %-8s %-9s %s\n' function vendor device driver power control
for fn in "$SYSFS"/bus/pci/devices/*; do
    [ -e "$fn/class" ] || continue
    cls=$(cat "$fn/class" 2>/dev/null || true)
    case "$cls" in 0x0c0330*) ;; *) continue;; esac          # xHCI host controller
    v=$(cat "$fn/vendor" 2>/dev/null || true)
    d=$(cat "$fn/device" 2>/dev/null || true)
    drv=$(basename "$(readlink -f "$fn/driver" 2>/dev/null || echo none)" 2>/dev/null || echo none)
    ps=$(cat "$fn/power_state" 2>/dev/null || echo '?')
    ctl=$(cat "$fn/power/control" 2>/dev/null || echo '?')
    hit=""
    if [ "$v" = "$VENDOR" ]; then
        for want in $DEVICES; do [ "$d" = "$want" ] && hit=yes; done
    fi
    printf '%-14s %-8s %-10s %-8s %-9s %s%s\n' "$(basename "$fn")" "$v" "$d" "$drv" "$ps" "$ctl" \
        "$([ -n "$hit" ] && echo "   <- the rule matches this function")"
    if [ -n "$hit" ] && [ "$drv" = "$DRIVER" ]; then
        applicable=$((applicable + 1))
        [ "$ps" != "D0" ] && asleep=$((asleep + 1))
        if [ "$MODE" = apply ] || [ "$MODE" = revert ]; then
            want=$([ "$MODE" = apply ] && echo on || echo auto)
            if ! ( : > "$fn/power/control" ) 2>/dev/null; then
                echo "   $MODE needs write access to $fn/power/control (run as root)" >&2
            else
                printf '%s' "$want" > "$fn/power/control" && echo "   $MODE: power/control -> $want"
            fi
        fi
    fi
done

if [ "$applicable" -eq 0 ]; then
    echo "== verdict: the rule's DMI gate matches but no AMD xHCI function with the rule's device ids was found --"
    echo "            the rule would bind nothing here (BIOS changed the ids?). Reported, not guessed."
    exit 0
fi

if [ "$MODE" = apply ] || [ "$MODE" = revert ]; then
    sleep 2
    echo "== after $MODE:"
    for fn in "$SYSFS"/bus/pci/devices/*; do
        [ -e "$fn/class" ] || continue
        case "$(cat "$fn/class" 2>/dev/null)" in 0x0c0330*) ;; *) continue;; esac
        d=$(cat "$fn/device" 2>/dev/null || true)
        for want in $DEVICES; do
            if [ "$d" = "$want" ]; then
                printf '   %-14s power=%-8s control=%s\n' "$(basename "$fn")" \
                    "$(cat "$fn/power_state" 2>/dev/null || echo '?')" "$(cat "$fn/power/control" 2>/dev/null || echo '?')"
            fi
        done
    done
fi

echo "== verdict: $applicable controller(s) match the rule; $asleep of them are NOT in D0"
if [ "$asleep" -eq 0 ]; then
    echo "==           all awake: a plug test on this boot would be a valid test of engine#823's ports."
    exit 0
fi
echo "==           with $asleep asleep the rule is either absent (a non-1bit boot) or not yet applied; a plug into a rear USB-A"
echo "==           or a Type-C port would fail for the reason the rule exists, so it would test nothing."
if [ "$MODE" = require ]; then
    exit 1
fi
exit 0
