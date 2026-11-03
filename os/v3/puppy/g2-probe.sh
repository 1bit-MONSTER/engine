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
# g2-probe.sh <iso> <work-dir> [seconds]
#
# Gate G2 of docs/os-v3-live-puppy-eval.md: boot the built Puppy ISO headless in QEMU and keep the init's
# transcript, so "it boots" or "it stops here" is a measurement rather than a reading of the code.
#
# <iso>       the image to test (upup64r-26.04.iso); the kernel and initrd are read OUT of it, so the run
#             measures the artifact and not a copy of it in the build tree
# <work-dir>  where the transcript, the QEMU stderr and the extracted kernel land
# [seconds]   how long to let it run, default 150 - the init drops to a console at ~10 s, so anything
#             past that is headroom for a slow first boot on a loaded box
#
# The kernel command line is the bootloader's own (`linux /vmlinuz pmedia=cd ...`) plus console=ttyS0:
# without it a headless run leaves no transcript at all. The VM is capped (2 GiB, 2 vCPU) and the whole
# QEMU process runs under a MemoryMax so a runaway guest cannot take the box with it.
#
# Exit 0 when the boot got past the sfs search into the layered filesystem, 1 when it dropped to the
# console with the sfs not found (G2 red: the kernel cannot see the medium), 2 on a usage or tool error.
# The `~ #` prompt is NOT the success signal: the init drops to that same prompt when it fails, so the
# verdict comes from which of the two it printed.
set -uo pipefail
prog=g2-probe
usage() { sed -n '/^# g2-probe.sh/,/^# Exit/p' "$0" | sed 's/^# \{0,1\}//'; }

[ $# -ge 2 ] || { usage >&2; exit 2; }
iso=$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")
work=$2
secs=${3:-150}
[ -f "$iso" ] || { echo "$prog: no such image: $iso" >&2; exit 2; }
for t in qemu-system-x86_64 xorriso python3 systemd-run; do
    command -v "$t" > /dev/null || { echo "$prog: $t is missing" >&2; exit 2; }
done
mkdir -p "$work" || exit 2
work=$(cd "$work" && pwd)

# The kernel and initrd come from the image, not from the build tree beside it.
[ -f "$work/vmlinuz" ] || xorriso -osirrox on -indev "$iso" \
    -extract /vmlinuz "$work/vmlinuz" -extract /initrd.gz "$work/initrd.gz" > /dev/null 2>&1
[ -s "$work/vmlinuz" ] && [ -s "$work/initrd.gz" ] || { echo "$prog: cannot extract the kernel from $iso" >&2; exit 2; }

rm -f "$work/ser" "$work/serial.log"
( timeout "$secs" systemd-run --user --scope -q -p MemoryMax=4G -- \
    qemu-system-x86_64 -m 2048 -smp 2 -machine q35 \
    -kernel "$work/vmlinuz" -initrd "$work/initrd.gz" \
    -append "console=ttyS0 pfix=nox pmedia=cd" \
    -cdrom "$iso" \
    -display none -vga std -no-reboot \
    -serial unix:"$work/ser",server,nowait ) > "$work/qemu.out" 2>&1 &
python3 - "$work/ser" "$work/serial.log" <<'EOF' > "$work/tee.out" 2>&1 &
import socket, sys, time
path, log = sys.argv[1], sys.argv[2]
while True:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(path); break
    except Exception:
        time.sleep(0.2)
with open(log, 'ab', buffering=0) as f:
    while True:
        d = s.recv(65536)
        if not d: break
        f.write(d)
EOF
wait

echo "== $iso"
echo "== $(wc -c < "$work/serial.log" 2> /dev/null || echo 0) bytes of transcript in $work/serial.log"
grep -a -E 'ResolutePup64|Finding puppy|HAVE_PARTS|not found|dropping out|PUPMODE|layered' "$work/serial.log" 2> /dev/null | tail -8
if grep -aqE 'not found|dropping out to console' "$work/serial.log" 2> /dev/null; then
    echo "== G2: no shell; the init did not find its sfs (G2 red until the kernel carries the medium drivers)"
    exit 1
fi
# The image booted if the init layered the main SFS and Puppy's own userspace then took over: the
# "main file" line is the init, "Loading kernel modules"/"Setting up services" are rc.sysinit, and the
# quicksetup dialog is the first thing a user sees. PUPMODE only appears once there is a save layer, so
# requiring it was a gate that could not pass on a fresh image - cycle 1 booted to the keyboard-setup
# dialog and this still answered "neither a layer nor the not-found error" (measured 2026-11-02).
if grep -aqE "main file|PUPMODE=|layered filesystem" "$work/serial.log" 2> /dev/null; then
    echo "== G2: the sfs was found and layered - the base booted on our kernel"
    grep -a -oE "ResolutePup64[^[]*|Loading the '[^']*' [a-z]+ file|Setting up services[^.]*|Set console keyboard layout" \
        "$work/serial.log" 2> /dev/null | sort -u | sed 's/^/   /'
    exit 0
fi
echo "== G2: neither a layer nor the not-found error: read $work/serial.log"
exit 2
