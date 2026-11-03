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
# install-probe.sh <img> <work-dir> [seconds]
#
# The installer's own gate: boot the appliance image, run `1bit-install` onto a second, empty disk, and then read
# the firmware's OWN state back — because the install's own log line has been wrong before (engine#866: step 7 took
# the first Boot#### of efibootmgr's output, which is Boot0000 BootManagerMenuApp, and pointed BootNext and the front
# of BootOrder at the UEFI menu; a headless machine then waits in that menu for ever).
#
# What it asserts, on the firmware state and not on the installer's word:
#   * exactly one entry is labelled "1bit OS" and its device path carries THIS disk's ESP partition GUID,
#   * BootNext is that entry's id,
#   * the first id in BootOrder is that same id,
#   * and (for engine#865) the installed grub.cfg pins the data partition by UUID while the boot line's pupsfs=
#     value is one the init can actually resolve — `--pupsfs=label|uuid` decides which answer is expected.
#
# Exit: 0 the installed disk is what the firmware will boot and what its grub.cfg says, 1 a check failed,
# 2 the harness could not run (no qemu, no image, no firmware vars).
#
# Env: PUPPY_PUPSFS=label|uuid (default label - what engine#865 says init can resolve), SBBOOT=1 (default) to run the
# firmware with Secure Boot on and this machine's MOK enrolled, like the board.
set -uo pipefail
prog=$(basename "$0")
[ $# -ge 2 ] || { sed -n "/^# $prog /,/^# Exit:/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
img=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
work=$2
secs=${3:-420}
pupsfs=${PUPPY_PUPSFS:-label}
[ -f "$img" ] || { echo "$prog: no such image: $img" >&2; exit 2; }
for t in qemu-system-x86_64 python3; do
    command -v "$t" > /dev/null || { echo "$prog: $t is missing" >&2; exit 2; }
done
mkdir -p "$work" || exit 2
work=$(cd "$work" && pwd)
target=$work/target.img
if [ ! -f "$target" ]; then
    truncate -s 8G "$target" || exit 2
fi

# Secure Boot with this machine's certificate: the installer writes an efibootmgr entry, and efivarfs only exists
# when the firmware actually boots us (a -kernel boot has none) - so the image is booted as a USB disk, exactly as
# the appliance is.
vars=$work/OVMF_VARS_SB.fd
"$(dirname "$0")/sb-vars.sh" "$vars" > "$work/sb-vars.txt" 2>&1 || {
    cat "$work/sb-vars.txt" >&2; echo "$prog: cannot build the Secure Boot variable store" >&2; exit 2; }

rm -f "$work/ser" "$work/serial.log"
( timeout "$secs" systemd-run --user --scope -q -p MemoryMax=6G -- \
    qemu-system-x86_64 -m 4096 -smp 4 \
    -drive file="$img",format=raw,if=none,id=d0 -device qemu-xhci -device usb-storage,drive=d0 \
    -drive file="$target",format=raw,if=virtio,id=d1 \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.secboot.fd \
    -drive if=pflash,format=raw,file="$vars" \
    -machine q35,smm=on -global driver=cfi.pflash01,property=secure,value=on \
    -display none -vga std -no-reboot \
    -serial unix:"$work/ser",server,nowait ) > "$work/qemu.out" 2>&1 &

prog_out=$(PUPSFS=$pupsfs python3 - "$work" "$img" <<'PY'
import os, re, socket, subprocess, sys, time
work, img = sys.argv[1], sys.argv[2]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
end = time.time() + 60
while True:
    try:
        s.connect(os.path.join(work, "ser")); break
    except OSError:
        if time.time() > end:
            print("FAIL boot           qemu never opened the serial socket"); sys.exit(2)
        time.sleep(0.2)
buf = b""
log = open(os.path.join(work, "serial.log"), "ab", buffering=0)

def read_until(token, timeout):
    """bytes up to and including token, or None."""
    global buf
    end = time.time() + timeout
    while time.time() < end:
        i = buf.find(token)
        if i >= 0:
            out, buf = buf[:i + len(token)], buf[i + len(token):]
            return out
        s.settimeout(0.5)
        try:
            d = s.recv(65536)
        except socket.timeout:
            continue
        except OSError:
            return None
        if not d:
            return None
        log.write(d); buf += d
    return None

# The grub menu is on the serial line and the serial-shell entry is its last one. Wait for the menu by its FOOTER
# ("Press enter to boot"), not by a menuentry title: the title text also appears in the header of every capture, and
# matching it early sends the arrow keys before grub is listening (measured: the harness then waited out its whole
# timeout on a shell it had never reached).
t0 = time.time()
if read_until(b"Press enter to boot the selected OS", 180) is None:
    print("FAIL boot           no grub menu on the serial line (no serial-shell entry?)"); sys.exit(2)
print("INFO boot           grub menu after %.1fs" % (time.time() - t0), flush=True)
time.sleep(1.0)
s.sendall(b"\x1b[B" * 3 + b"\r")
t0 = time.time()
ok = False
for token in (b"21serial-console: root shell", b"root@1bit-os", b"bash-5.3#"):
    if read_until(token, 120) is not None:
        ok = True
        break
if not ok:
    print("FAIL boot           no serial root shell; last transcript lines:")
    for l in (buf.decode("utf-8", "replace").splitlines()[-12:]):
        print("   " + l[:150])
    sys.exit(2)
print("INFO boot           serial shell after %.1fs" % (time.time() - t0), flush=True)
buf = b""
s.sendall(b"\n")
time.sleep(1)

def read_until_re(pattern, timeout):
    """like read_until, but a regex - and on the SENTINEL ONLY: a literal token in the command line is echoed by
    the terminal, so a plain token matched its own echo and every row came back empty (measured)."""
    global buf
    end = time.time() + timeout
    while time.time() < end:
        m = re.search(pattern, buf)
        if m:
            out, buf = buf[:m.end()], buf[m.end():]
            return out
        s.settimeout(0.5)
        try:
            d = s.recv(65536)
        except socket.timeout:
            continue
        except OSError:
            return None
        if not d:
            return None
        log.write(d); buf += d
    return None

def cmd(c, timeout=600):
    """run one line, return its output. `$(date +%s%N)` is expanded only when the command RUNS, so the sentinel
    cannot match the echo of the line; the exit code rides in front of it."""
    s.sendall((c + "; echo __E$?__$(date +%s%N)\n").encode())
    out = read_until_re(rb"__E[0-9]+__[0-9]{10,}", timeout)
    if out is None:
        return None
    return re.sub(r"__E[0-9]+__[0-9]{10,}\s*$", "", out.decode("utf-8", "replace"))

rows = []
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")
def clean(text):
    """the console's own noise: colours, and the prompt that prefixes the first line of output"""
    text = ANSI.sub("", text)
    text = re.sub(r"[a-z0-9_-]+@[a-z0-9_-]+:[^$#]*[$#]\s*", "", text)
    text = re.sub(r"bash-[0-9.]+#\s*", "", text)
    return text

def row(name, cmdline, timeout=600):
    out = cmd(cmdline, timeout)
    if out is None:
        rows.append(("FAIL", name, "no answer")); return ""
    body = clean(out)
    body = "\n".join(l.strip() for l in body.splitlines() if l.strip() and cmdline[:20] not in l)
    rows.append(("INFO", name, body.replace("\n", " | ")[:400]))
    return body

# 1. the firmware as it is BEFORE the install, so the "first Boot####" trap is visible by name
row("fw before", "efibootmgr | head -6")
# 2. install onto the second disk, non-interactively
install = row("install", "1bit-install /dev/vda --yes; echo rc=$?", 900)
# 3. the firmware AFTER - the whole question of #866
after = row("fw after", "efibootmgr; echo ---; efibootmgr -v | grep -i '1bit OS'", 120)
grub = row("grub.cfg", "grep -h '^ *linux ' /tmp/x 2>/dev/null; "
                       "ls /mnt 2>/dev/null >/dev/null; "
                       "mkdir -p /mnt/e; mount -o ro /dev/vda1 /mnt/e 2>/dev/null && "
                       "grep '^ *linux ' /mnt/e/EFI/BOOT/grub.cfg | head -1; umount /mnt/e 2>/dev/null", 120)
espuuid = (row("esp uuid", "blkid -s PARTUUID -o value /dev/vda1") or "").strip()
espuuid = [l.strip() for l in espuuid.splitlines() if re.match(r"^[0-9a-fA-F-]{8,}$", l.strip())]
espuuid = espuuid[-1] if espuuid else ""

# ---- the verdicts, on the firmware state rather than on the installer's log line
bootnext = re.search(r"BootNext: ([0-9A-Fa-f]{4})", after or "")
bootorder = re.search(r"BootOrder: ([0-9A-Fa-f,]+)", after or "")
entries = dict(re.findall(r"Boot([0-9A-Fa-f]{4})\* ([^\r\n]+)", after or ""))
onebit = [i for i, l in entries.items() if l.strip().startswith("1bit OS")]

print("== %s" % img)
for st, name, body in rows:
    print("%-4s %-12s %s" % (st, name, body))

ok = True
if "INSTALLED" not in (install or ""):
    print("FAIL install      the installer did not report INSTALLED"); ok = False
if len(onebit) != 1:
    print("FAIL fw entries   expected exactly one '1bit OS' entry, found %d: %s" % (len(onebit), entries))
    ok = False
else:
    want = onebit[0]
    if espuuid and espuuid.lower() not in (after or "").lower():
        print("FAIL fw entries   the 1bit OS entry's device path does not carry this ESP's PARTUUID %s" % espuuid)
        ok = False
    if not bootnext or bootnext.group(1) != want:
        print("FAIL BootNext     BootNext is %s, the 1bit OS entry is Boot%s" %
              (bootnext.group(1) if bootnext else "<absent>", want))
        ok = False
    if not bootorder or bootorder.group(1).split(",")[0].strip() != want:
        print("FAIL BootOrder    BootOrder starts with %s, the 1bit OS entry is Boot%s" %
              (bootorder.group(1).split(",")[0] if bootorder else "<absent>", want))
        ok = False
    if entries.get(want, "").startswith("BootManagerMenuApp"):
        print("FAIL fw entries   Boot%s is BootManagerMenuApp, not 1bit OS" % want); ok = False

# ---- engine#865's half: what the installed grub.cfg tells the init, and whether that is resolvable. Asserted only
# when PUPPY_ASSERT_PUPSFS=1, because it is #865's question: on a #866 run the installer legitimately writes
# pupsfs=UUID=... (that is what the install needs) and the init's ability to resolve it is the other issue.
assert_pupsfs = os.environ.get("PUPPY_ASSERT_PUPSFS") == "1"
line = ""
for l in (grub or "").splitlines():
    if "linux " in l and "vmlinuz" in l:
        line = l.strip(); break
if not line:
    print("FAIL grub.cfg     no linux line could be read from the installed ESP"); ok = False
else:
    m = re.search(r"pupsfs=([^ ]+)", line)
    got = m.group(1) if m else ""
    pu = os.environ.get("PUPSFS", "label")
    want = "1BIT-DATA" if pu == "label" else "UUID="
    if not assert_pupsfs:
        print("INFO grub.cfg     pupsfs=%s (not asserted: set PUPPY_ASSERT_PUPSFS=1 for the #865 question)" % got)
    elif pu == "label" and got != "1BIT-DATA":
        print("FAIL grub.cfg     pupsfs=%s, but the init can only resolve a label (#865)" % got); ok = False
    elif pu == "uuid" and not got.startswith("UUID="):
        print("FAIL grub.cfg     pupsfs=%s, expected UUID=<uuid>" % got); ok = False
    else:
        print("PASS grub.cfg     pupsfs=%s (and 1bit.data= stays on the UUID)" % got)
    if "1bit.data=UUID=" not in line:
        print("FAIL grub.cfg     the boot line lost 1bit.data=UUID= (#539/#711)"); ok = False

print("== %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
)
rc=$?
echo "$prog_out"
for p in $(pgrep -f "qemu-system-x86_64 -m 4096 -smp 4.*$img" 2> /dev/null); do kill "$p" 2> /dev/null; done
echo "== transcript $work/serial.log"
[ "$rc" = 0 ] && exit 0
[ "$rc" = 2 ] && exit 2
exit 1
