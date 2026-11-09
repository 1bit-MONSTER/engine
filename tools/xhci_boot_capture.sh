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
# xhci_boot_capture.sh - one bundle of evidence from a boot, for engine#823 (AXB35-02 USB ports).
#
# engine#823: on the Bosgame BeyondMax (AXB35-02) only the front USB-A port enumerates under Linux;
# rear USB-A and both Type-C ports read as nothing. The question a capture has to answer is narrow and
# has three parts, in this order:
#
#   1. what the firmware is (BIOS version and board), because the owner reflashed to 1.10 and any
#      before/after claim needs the version recorded next to it;
#   2. what power state each xHCI controller ended up in, and whether the image's keepawake rule
#      (etc/udev/rules.d/91-1bit-xhci-keepawake.rules) actually held them in D0;
#   3. what the root hubs report PER PORT - connect or not - because "nothing plugged in" and
#      "plugged in and not enumerated" are the two answers that look identical everywhere else.
#
# Written to run ON THE LIVE STICK, so: no python, no debugfs assumption (this kernel does not expose
# usb/xhci/*/ports/*/portsc), and the per-port status comes from `lsusb -v` on each root hub, which
# needs root and is therefore the one part that will say so if it cannot read.
#
# usage: tools/xhci_boot_capture.sh [outdir]
#   default outdir: /data/xhci-capture-<YYYYmmdd-HHMMSS> when /data is writable, else /tmp/...
# exit: 0 captured, 2 not root and sudo unavailable, 3 cannot write
set -u

say() { printf '%s\n' "$*"; }

if [ "$(id -u)" != 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        exec sudo -- "$0" "$@"
    fi
    say "xhci_boot_capture: needs root (reads per-port status and PCI config space); no sudo here" >&2
    exit 2
fi

stamp=$(date +%Y%m%d-%H%M%S)
out=${1:-}
if [ -z "$out" ]; then
    if [ -d /data ] && [ -w /data ]; then out=/data/xhci-capture-$stamp; else out=/tmp/xhci-capture-$stamp; fi
fi
mkdir -p "$out" 2>/dev/null || { say "xhci_boot_capture: cannot write $out" >&2; exit 3; }

# ── 1. the firmware, next to the claim it will be used to support ─────────────────────────────────
{
    echo "captured:  $(date '+%F %T %Z')"
    echo "uptime:    $(cut -d' ' -f1 /proc/uptime) s"
    echo "kernel:    $(uname -a)"
    echo "cmdline:   $(cat /proc/cmdline 2>/dev/null)"
    echo
    for f in bios-vendor bios-version bios-release-date baseboard-manufacturer baseboard-product-name \
             baseboard-version system-product-name system-version; do
        printf '%-24s %s\n' "$f:" "$(dmidecode -s "$f" 2>/dev/null)"
    done
    echo
    # A capture from a minimal appliance stick is only interpretable if it says what it could and could
    # not ask: an empty section then means "no tool" rather than "no problem".
    printf 'tools:'
    for t in dmidecode lsusb lspci dmesg tar; do
        if command -v "$t" >/dev/null 2>&1; then printf ' %s=yes' "$t"; else printf ' %s=NO' "$t"; fi
    done
    echo
} > "$out/summary.txt" 2>&1

# ── 2. controllers: what state are they in, and is the keepawake rule holding them? ───────────────
: > "$out/controllers.txt"
for d in /sys/bus/pci/drivers/xhci_hcd/0000:*; do
    [ -e "$d" ] || continue
    bdf=$(basename "$d")
    printf '%s  control=%s  power_state=%s  runtime_status=%s\n' \
        "$bdf" \
        "$(cat "$d/power/control" 2>/dev/null)" \
        "$(cat "$d/power_state" 2>/dev/null)" \
        "$(cat "$d/power/runtime_status" 2>/dev/null)" >> "$out/controllers.txt"
done
# The rule's own subject, as the running system sees it (the image ships it; a non-1bit boot does not).
{
    echo "== udev rule on this system:"
    ls -l /etc/udev/rules.d/91-1bit-xhci-keepawake.rules 2>/dev/null || echo "   absent (this is not a 1bit boot, or the rule did not ship)"
    echo
    echo "== interrupts taken per controller since boot:"
    grep -iE 'xhci' /proc/interrupts 2>/dev/null || echo "   none"
} > "$out/rule-and-irq.txt" 2>&1

# ── 3. per port: connect or not, which is the answer everything else hides ────────────────────────
: > "$out/ports.txt"
for hub in /sys/bus/usb/devices/usb*; do
    [ -e "$hub/devnum" ] || continue
    bus=$(cat "$hub/busnum" 2>/dev/null); dev=$(cat "$hub/devnum" 2>/dev/null)
    [ "$dev" = 1 ] || continue                       # root hubs only
    printf '\n== bus %s root hub (%s)\n' "$bus" "$(basename "$hub")" >> "$out/ports.txt"
    # lsusb -v on a root hub lists one "Port N: ...." line per port; "connect" present means a device
    # is seen on that port. Needs root, which is why the whole script escalates at the top.
    lsusb -v -s "$bus:1" 2>/dev/null | sed -n 's/^ *Port \([0-9]*\): \(.*\)$/  port \1: \2/p' >> "$out/ports.txt" \
        || echo "  (lsusb -v could not read bus $bus)" >> "$out/ports.txt"
done

lspci -vv 2>/dev/null | awk '/USB controller/{p=1} p{print} p&&/^$/{p=0}' > "$out/lspci-usb.txt"
lsusb -t > "$out/lsusb-tree.txt" 2>&1
dmesg > "$out/dmesg-full.txt" 2>&1
grep -iE 'xhci|usb|acpi.*(usb|UCSI)|pcieport|AER' "$out/dmesg-full.txt" > "$out/dmesg-filtered.txt" 2>&1

# The repo's own precondition check, if the tools are on the stick: it answers "would a plug here test
# anything", which is the question that decides whether the boot is worth asking the owner to repeat.
for c in "$(dirname "$0")/xhci_keepawake_check.sh" /usr/local/bin/xhci_keepawake_check /opt/1bit/tools/xhci_keepawake_check.sh; do
    [ -x "$c" ] && { "$c" > "$out/keepawake-check.txt" 2>&1; break; }
done

# ── the one-screen answer, so a capture can be read before it is sent ────────────────────────────
{
    echo "== xHCI capture $stamp"
    sed -n '1,10p' "$out/summary.txt"
    echo
    echo "== controllers"
    sed 's/^/   /' "$out/controllers.txt"
    awake=$(grep -c 'power_state=D0' "$out/controllers.txt" 2>/dev/null)
    total=$(wc -l < "$out/controllers.txt")
    echo "   $awake of $total in D0"
    echo
    echo "== ports seen with a device attached (connect), per bus"
    awk '/^== bus/{b=$3} /port [0-9]+:.*connect/{c[b]++} END{for (k in c) printf "   bus %s: %s port(s) with connect\n", k, c[k]}' "$out/ports.txt"
    echo
    echo "== files: $(ls "$out" | tr '\n' ' ')"
} > "$out/VERDICT.txt" 2>&1
cp "$out/VERDICT.txt" "$out/../xhci-capture-$stamp.txt" 2>/dev/null || true

tar -C "$(dirname "$out")" -czf "$out.tar.gz" "$(basename "$out")" 2>/dev/null
# The script escalates, so an outdir the caller owns would otherwise come back root-owned and the caller
# could not delete or open the bundle without sudo - a capture that is hard to send is a capture not sent.
if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
    chown -R "$SUDO_UID:$SUDO_GID" "$out" "$out.tar.gz" 2>/dev/null || true
fi
cat "$out/VERDICT.txt"
say ""
say "bundle: $out.tar.gz"
