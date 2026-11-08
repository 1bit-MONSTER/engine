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

prog_out=$(PUPSFS=$pupsfs PROBE_VARS=$vars python3 - "$work" "$img" <<'PY'
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
# The menu selection needs the same protection a reboot gets (below): arrows CANCEL GRUB's countdown, so an Enter lost
# while the menu is still drawing leaves the machine at the menu (measured), and on a loaded box the boot to userspace
# has been measured at 155 s and past 300 s - so press again if nothing arrives, and give each attempt a budget the
# boot can actually fit in (120 s was inside the boot's own range on 2026-11-07/08, and reported "no shell" for a
# machine that was booting normally).
t0 = time.time()
ok = False
for attempt in (1, 2, 3):
    s.sendall(b"\x1b[B" * 3)
    time.sleep(0.5)
    s.sendall(b"\r")
    time.sleep(0.5)
    s.sendall(b"\r")
    for token in (b"21serial-console: root shell", b"root@1bit-os", b"bash-5.3#"):
        if read_until(token, 90 if attempt == 1 else 300) is not None:
            ok = True
            break
    if ok:
        break
    print("INFO boot           nothing booted after attempt %d; pressing Enter again at the menu" % attempt, flush=True)
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
# engine#709 item (4): the installer now ends with a 10 s countdown and a reboot. This gate reads the firmware
# state AFTER the install, so it needs the flag that keeps the old behaviour - and that makes `--no-reboot` one of
# the two paths the owner asked to have tested, not a convenience.
install = row("install", "1bit-install /dev/vda --yes --no-reboot; echo rc=$?", 900)
# 3. the firmware AFTER - the whole question of #866
after = row("fw after", "efibootmgr; echo ---; efibootmgr -v | grep -i '1bit OS'", 120)
grub = row("grub.cfg", "grep -h '^ *linux ' /tmp/x 2>/dev/null; "
                       "ls /mnt 2>/dev/null >/dev/null; "
                       "mkdir -p /mnt/e; mount -o ro /dev/vda1 /mnt/e 2>/dev/null && "
                       "grep '^ *linux ' /mnt/e/EFI/BOOT/grub.cfg | head -1; umount /mnt/e 2>/dev/null", 120)
stick_acct = row("stick acct", "id onebit 2>&1 | tail -1")
espuuid = (row("esp uuid", "blkid -s PARTUUID -o value /dev/vda1") or "").strip()
espuuid = [l.strip() for l in espuuid.splitlines() if re.match(r"^[0-9a-fA-F-]{8,}$", l.strip())]
espuuid = espuuid[-1] if espuuid else ""

# ---- engine#709's arm: a firmware entry whose partition does not exist must not survive an install.
# The issue was filed from exactly this shape: the wiped distribution's `\EFI\ubuntu\shimx64.efi` entry was still in NVRAM, the
# firmware walked it first, failed, and fell through to the stick. So: seed such an entry on the ESP we just wrote (it therefore
# carries that ESP's partition GUID), then run the installer AGAIN - the supported reinstall - which repartitions the disk and
# gives the ESP a new GUID, leaving the seeded entry pointing at a partition that exists nowhere. That is the dangling state the
# firmware cannot boot through, and the installer is expected to remove it while writing this disk.
seed = row("seed stale", "efibootmgr -c -d /dev/vda -p 1 -L ubuntu -l '\\EFI\\ubuntu\\shimx64.efi' > /dev/null 2>&1; "
                  "efibootmgr 2>/dev/null | grep -i ubuntu | head -2")
reinstall = row("reinstall", "1bit-install /dev/vda --yes --no-reboot; echo rc=$?", 900)
cleaned = row("fw cleaned", "efibootmgr 2>/dev/null; echo ---; efibootmgr -v 2>/dev/null | grep -iE '1bit OS|ubuntu'", 120)

# ---- engine#709 item (4)'s other path: the countdown, cancelled by a keypress. A serial gate cannot wait out a real
# reboot (it would lose its shell), so the arm takes the path a person takes - it watches for the countdown line and sends
# ONE byte, no Enter, which is what `read -n 1` takes.
s.sendall(b"1bit-install /dev/vda --yes; echo __E$?__$(date +%s%N)\n")
countdown = ""
if read_until(b"rebooting into the installed system in", 900) is None:
    rows.append(("FAIL", "countdown", "the installer never printed the countdown line"))
else:
    s.sendall(b"x")                       # the keypress
    raw = read_until_re(rb"__E[0-9]+__[0-9]{10,}", 120)
    countdown = re.sub(r"__E[0-9]+__[0-9]{10,}\s*$", "", (raw or b"").decode("utf-8", "replace")).strip()
    rows.append(("INFO", "countdown", (countdown or "no output").replace("\n", " | ")[-300:]))

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
if "uid=" in (stick_acct or ""):
    print("FAIL stick acct   the live stick carries a onebit account (%s): a machine that forgets should stay "
          "root-only" % stick_acct.strip()); ok = False
else:
    print("PASS stick acct   no onebit account on the live stick (%s)" % (stick_acct or "").strip()[:80])
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

# ---- engine#709 item (4): both endings. `--no-reboot` must skip the countdown entirely and leave the shell in place; a
# key during the countdown must cancel the reboot and do the same.
if "rebooting into the installed system in" in (install or ""):
    print("FAIL no-reboot    --no-reboot still counted down"); ok = False
elif "--no-reboot" in (install or "") and "not rebooting" not in (install or ""):
    print("FAIL no-reboot    the installer did not report that --no-reboot was honoured"); ok = False
else:
    print("PASS no-reboot    --no-reboot skipped the countdown and returned to the shell")
if "cancelled" not in (countdown or ""):
    print("FAIL countdown    a keypress during the countdown did not cancel the reboot"); ok = False
else:
    print("PASS countdown    a keypress during the countdown cancelled the reboot")

# ---- engine#709: the seeded dangling entry, and that the reinstall left a bootable order behind it
entries2 = dict(re.findall(r"Boot([0-9A-Fa-f]{4})\* ([^\r\n]+)", cleaned or ""))
stale_left = [i for i, l in entries2.items() if "ubuntu" in l.lower()]
onebit2 = [i for i, l in entries2.items() if l.strip().startswith("1bit OS")]
if "INSTALLED" not in (reinstall or ""):
    print("FAIL reinstall    the second install did not report INSTALLED"); ok = False
elif "ubuntu" not in (seed or "").lower():
    print("FAIL stale seed   the probe could not create the stale entry, so removal is unproven"); ok = False
elif stale_left:
    print("FAIL stale entry  a dangling firmware entry survived the install: Boot%s %s" %
          (stale_left[0], entries2.get(stale_left[0], "")))
    ok = False
else:
    print("PASS stale entry  the dangling 'ubuntu' entry is gone (its partition no longer exists)")
if len(onebit2) != 1:
    print("FAIL fw entries   after the reinstall there should be exactly one '1bit OS' entry, found %d" % len(onebit2))
    ok = False
else:
    order2 = re.search(r"BootOrder: ([0-9A-Fa-f,]+)", cleaned or "")
    next2 = re.search(r"BootNext: ([0-9A-Fa-f]{4})", cleaned or "")
    keep = onebit2[0]
    if not order2 or order2.group(1).split(",")[0].strip() != keep:
        print("FAIL BootOrder    after the reinstall BootOrder does not start with the 1bit OS entry Boot%s" % keep)
        ok = False
    if not next2 or next2.group(1) != keep:
        print("FAIL BootNext     after the reinstall BootNext is not the 1bit OS entry Boot%s" % keep)
        ok = False

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

# ---- engine#865's own question, when asked: boot the INSTALLED disk and see whether it reaches userspace. The
# init resolves the boot line's pupsfs=...; with `UUID=<uuid>` an init that cannot resolve that form exits 1 into the
# rescue console, which is what the issue measured. Only meaningful with PUPPY_ASSERT_PUPSFS=1 (the installer writes
# the UUID form on purpose), so the #866 run stops here.
if assert_pupsfs:
    print("== booting the installed disk (init must resolve pupsfs=%s)" % got)
    s.close()
    log.close()
    # the first VM still holds target.img (the installed disk) with a write lock, and this one would be refused with
    # "Failed to get write lock" (measured). Stop it by the STICK image on its command line - the new VM's line names
    # target.img instead, so this cannot kill what it is about to start.
    subprocess.run(["pkill", "-f", "qemu-system-x86_64.*" + img],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    b = os.path.join(work, "ser-installed")
    for f in (b, b + ".log"):
        try: os.unlink(f)
        except OSError: pass
    q = ["qemu-system-x86_64", "-m", "4096", "-smp", "4",
         "-netdev", "user,id=n0", "-device", "virtio-net-pci,netdev=n0",
         "-drive", "file=" + os.path.join(work, "target.img") + ",format=raw,if=none,id=d0",
         "-device", "qemu-xhci", "-device", "usb-storage,drive=d0",
         "-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.secboot.fd",
         "-drive", "if=pflash,format=raw,file=" + os.environ["PROBE_VARS"],
         "-machine", "q35,smm=on", "-global", "driver=cfi.pflash01,property=secure,value=on",
         "-display", "none", "-vga", "std", "-no-reboot",
         "-serial", "unix:" + b + ",server,nowait"]
    proc = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=6G", "--"] + q,
                            stdout=open(os.path.join(work, "qemu-installed.out"), "wb"),
                            stderr=subprocess.STDOUT)
    ok2 = False
    text = ""
    try:
        t = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        end = time.time() + 90
        while True:
            try:
                t.connect(b); break
            except OSError:
                if time.time() > end:
                    print("FAIL installed    the second VM never opened its serial socket"); break
                time.sleep(0.2)
        t.settimeout(0.5)
        end = time.time() + 420
        while time.time() < end:
            try:
                d = t.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not d:
                break
            text += d.decode("utf-8", "replace")
            open(os.path.join(work, "serial-installed.log"), "w").write(text)
            # from the initrd era to the running system: the 1bit progress line proves the SFS were found and layered
            if "1bit: system in RAM after" in text or "ResolutePup64" in text:
                ok2 = True
                break
            if "init stopped" in text or "not found." in text:
                break
    finally:
        subprocess.run(["pkill", "-f", "qemu-system-x86_64.*" + os.path.join(work, "target.img")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: proc.wait(timeout=10)
        except Exception: pass
    if ok2:
        print("PASS installed    the installed disk reached userspace on its own boot line")
    else:
        # name the three ways this leg fails, because they belong to different issues: the firmware went to the
        # boot-manager menu (engine#866: BootNext pointed at Boot0000), the init could not resolve the boot line
        # (engine#865), or init gave up and dropped to the rescue console
        why = "the firmware booted BootManagerMenuApp instead of 1bit OS (engine#866)" \
            if "BootManagerMenuApp" in text and "ResolutePup64" not in text else \
            ("the init could not resolve the boot line (engine#865)" if "not found." in text else \
             ("init dropped to the rescue console" if "init stopped" in text else "no boot output"))
        print("FAIL installed    the installed disk did not boot: %s (transcript %s)" %
              (why, os.path.join(work, "serial-installed.log")))
        ok = False

    # ---- engine#712: the installed machine must be a full persistent OS and not the live RAM image. Measured on
    # the installed disk's OWN boots, never on the installer's word: what PUPMODE the init settled on, where the
    # root's writable layer and the SFS actually are, and whether a write made on one boot is still there on the
    # next. The three ways an "installed" machine can still be the stick are: no save layer (PUPMODE=5, nothing
    # persists), the SFS copied into RAM (pfix=copy), and a save layer that points at the other disk.
    class Installed:
        """One boot of the installed disk, driven over its serial console. The installed grub.cfg is the stick's
        (headless console first, serial root shell last), so the shell is three arrows down - and this VM is
        started WITHOUT -no-reboot, because the last row reboots the guest."""
        def __init__(self, tag, shell):
            self.tag = tag
            self.s = None
            self.text = ""
            self.path = os.path.join(work, "ser-" + tag)
            for f in (self.path, self.path + ".log"):
                try: os.unlink(f)
                except OSError: pass
            q = ["qemu-system-x86_64", "-m", "4096", "-smp", "4",
                 "-netdev", "user,id=n0", "-device", "virtio-net-pci,netdev=n0",
                 "-drive", "file=" + os.path.join(work, "target.img") + ",format=raw,if=none,id=d0",
                 "-device", "qemu-xhci", "-device", "usb-storage,drive=d0",
                 "-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.secboot.fd",
                 "-drive", "if=pflash,format=raw,file=" + os.environ["PROBE_VARS"],
                 "-machine", "q35,smm=on", "-global", "driver=cfi.pflash01,property=secure,value=on",
                 "-display", "none", "-vga", "std",
                 "-serial", "unix:" + self.path + ",server,nowait"]
            self.proc = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=6G", "--"] + q,
                                         stdout=open(os.path.join(work, "qemu-%s.out" % tag), "wb"),
                                         stderr=subprocess.STDOUT)
            t = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            end = time.time() + 90
            while True:
                try:
                    t.connect(self.path); break
                except OSError:
                    if time.time() > end:
                        print("FAIL %-12s the installed disk's VM never opened its serial socket" % tag)
                        return
                    time.sleep(0.2)
            t.settimeout(0.5)
            self.s, self.buf = t, b""
            self.log = open(self.path + ".log", "ab", buffering=0)
            if not shell:
                return
            if self.until(b"Press enter to boot the selected OS", 180) is None:
                print("FAIL %-12s no grub menu on the installed disk" % tag); self.s = None; return
            time.sleep(1.0)
            self.s.sendall(b"\x1b[B" * 3 + b"\r")
            for token in (b"21serial-console: root shell", b"root@1bit-os", b"bash-5.3#"):
                if self.until(token, 150) is not None:
                    self.buf = b""
                    self.s.sendall(b"\n")
                    time.sleep(1)
                    return
            print("FAIL %-12s no root shell on the installed disk (transcript %s)" % (tag, self.path + ".log"))
            self.s = None

        def until(self, token, timeout):
            end = time.time() + timeout
            while time.time() < end:
                i = self.buf.find(token)
                if i >= 0:
                    out, self.buf = self.buf[:i + len(token)], self.buf[i + len(token):]
                    return out
                self.s.settimeout(0.5)
                try:
                    d = self.s.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return None
                if not d:
                    return None
                self.log.write(d); self.buf += d; self.text += d.decode("utf-8", "replace")
            return None

        def cmd(self, c, timeout=240):
            """one line, its output. Bracketed by __S__/__E...__ on the SENTINEL only (the terminal echoes the
            command back), and everything before the LAST __S__ is dropped: an installed boot runs services while
            this reads, and their one-line notices land in the middle of a transcript (measured on the first run:
            `24onboard-webapp: serving ...` arrived between the command and its output)."""
            self.s.sendall(("echo __S__; " + c + "; echo __E$?__$(date +%s%N)\n").encode())
            end = time.time() + timeout
            while time.time() < end:
                m = re.search(rb"__E[0-9]+__[0-9]{10,}", self.buf)
                if m:
                    out, self.buf = self.buf[:m.start()], self.buf[m.end():]
                    body = clean(out.decode("utf-8", "replace"))
                    i = body.rfind("__S__")
                    if i >= 0:
                        body = body[i + len("__S__"):]
                    return "\n".join(l.strip() for l in body.splitlines() if l.strip())
                self.s.settimeout(0.5)
                try:
                    d = self.s.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    return None
                if not d:
                    return None
                self.log.write(d); self.buf += d; self.text += d.decode("utf-8", "replace")
            return None

        def reboot(self):
            """the guest's own reboot - not a new VM. The firmware comes back up on the installed entry."""
            self.s.sendall(b"sync; reboot -f\n")
            self.text = ""
            if self.until(b"Press enter to boot the selected OS", 240) is None:
                return False
            time.sleep(1.0)
            # an arrow key CANCELS GRUB's countdown, so an Enter lost while the menu is still drawing leaves the
            # machine waiting at it for ever (measured in update-gate.sh, and again here): pause, press it twice, and
            # give it one more try before reporting a machine that "did not come back"
            for attempt in (1, 2):
                self.s.sendall(b"\x1b[B" * 3)
                time.sleep(0.5)
                self.s.sendall(b"\r")
                time.sleep(0.5)
                self.s.sendall(b"\r")
                # generous: a loaded box has been measured at 138 s to a shell and 6 min for a whole boot, and a
                # harness that loses that race reports "did not come back" for a machine that did
                for token in (b"21serial-console: root shell", b"root@1bit-os", b"bash-5.3#"):
                    if self.until(token, 120 if attempt == 1 else 300) is not None:
                        self.buf = b""
                        self.s.sendall(b"\n")
                        time.sleep(1)
                        return True
                if attempt == 1:
                    print("INFO reboot         nothing booted after the first Enter; pressing it again at the menu",
                          flush=True)
            return False

        def close(self):
            try: self.log.close()
            except Exception: pass
            try: self.s.close()
            except Exception: pass
            subprocess.run(["pkill", "-f", "qemu-system-x86_64.*" + os.path.join(work, "target.img")],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try: self.proc.wait(timeout=10)
            except Exception: pass

    print("== the installed disk, its own boots: is it a persistent OS? (#712)")
    v = Installed("persist", shell=True)
    try:
        if v.s is None:
            ok = False
        else:
            def val(token, cmdline, timeout=240):
                """the value behind `TOKEN=` in the command's output (last match). Every row prints its value
                with its own prefix, because the console is shared with the running system's notices."""
                out = v.cmd("echo %s=$(%s)" % (token, cmdline), timeout) or ""
                m = re.findall(r"\b%s=(.*)" % token, out)
                return (m[-1].strip() if m else "")

            pm = val("PM", "sed -n 's/^PUPMODE=//p' /etc/rc.d/PUPSTATE | head -1")
            rw = val("RW", "readlink /initrd/pup_rw")
            lay = val("LAY", "ls -d /data/1bit/*save* 2>/dev/null | head -1")
            # NOTE: one `%s`, not the escaped form - the cmdline is an ARGUMENT to val()'s "%s=$(%s)" formatting,
            # so a `%%` here reaches awk literally and the row prints "%s,%s,%s,%s," while asserting nothing
            # (measured on the previous run: it "passed" because that string cannot contain /mnt/tmpfs).
            sfs = val("SFS", "awk '$3 == \"squashfs\" { printf \"%s \", $1 }' /proc/mounts")
            room = val("ROOM", "df -h /data | tail -1")
            loop = val("LOOP", "losetup -a 2>/dev/null | head -1")
            print("INFO persist      PUPMODE=%s | root layer=%s | save layer=%s" % (pm, rw, lay))
            print("INFO persist      SFS mounted from: %s" % sfs[:200])
            print("INFO persist      backed by: %s" % loop[:200])
            print("INFO persist      /data room: %s" % room[:120])
            if pm != "12":
                print("FAIL persist      PUPMODE=%s, not 12: the installed machine boots the live RAM image "
                      "(no save layer, nothing persists)" % (pm or "<unread>")); ok = False
            elif "/1bit/" not in rw or not rw.endswith("save/upper"):
                print("FAIL persist      the root's writable layer is %s, not a save folder on the disk" % (rw or "<none>")); ok = False
            elif "/mnt/tmpfs" in sfs:
                print("FAIL persist      the SFS are copied into RAM (pfix=copy): %s" % sfs); ok = False
            elif not sfs:
                print("FAIL persist      no squashfs mount at all: cannot say where the OS came from"); ok = False
            else:
                print("PASS persist      PUPMODE=12, root writes through %s, SFS from %s" % (rw, sfs.strip()))
            # ---- engine#712's second half: a normal account with sudo, created by 07onboard-user on THIS boot,
            # read off the running machine (id, an actual `sudo -n` as that user, and the keys it was given).
            acct = val("ACCT", "id onebit 2>&1 | head -1")
            sugr = val("SUDO", "su -s /bin/sh onebit -c 'sudo -n id -un' 2>&1 | tail -1")
            keys = val("KEYS", "stat -c '%a %U' /home/onebit/.ssh/authorized_keys 2>&1")
            # the account is given the machine's keys from /data/ssh/authorized_keys, and from the image's own
            # /root/.ssh/authorized_keys when /data has none (a stock image ships only the host key there; a
            # prepared stick is the one that puts authorized_keys on /data). Compare against the same choice.
            same = val("SAME", "s=/data/ssh/authorized_keys; [ -s \"$s\" ] || s=/root/.ssh/authorized_keys; "
                               "cmp -s \"$s\" /home/onebit/.ssh/authorized_keys && echo same-from-$s || echo differ")
            nkey = val("NKEY", "grep -c . /home/onebit/.ssh/authorized_keys 2>/dev/null || echo 0")
            print("INFO account      %s | sudo -n -> %s | keys %s (%s line(s)) | %s"
                  % (acct, sugr, keys, nkey, same))
            if "uid=" not in acct or "onebit" not in acct:
                print("FAIL account      no onebit account on the installed machine (id: %s)" % (acct or "<none>")); ok = False
            elif sugr != "root":
                print("FAIL account      sudo as onebit answered %s, not root" % (sugr or "<nothing>")); ok = False
            elif not keys.startswith("600 onebit"):
                print("FAIL account      /home/onebit/.ssh/authorized_keys is %s, want 600 onebit" % (keys or "<missing>")); ok = False
            elif not same.startswith("same-from-"):
                print("FAIL account      the account's keys are not the machine's keys (cmp: %s)" % (same or "no answer")); ok = False
            elif nkey == "0":
                print("FAIL account      the account's key file is empty"); ok = False
            else:
                print("PASS account      %s, sudo -> root, keys 600 onebit, %s key line(s) from %s"
                      % (acct, nkey, same[len("same-from-"):]))
            # ---- engine#712 item 3 (the server-shaped rest), measured where it matters: on the installed machine.
            # The claim is "present or named", so each row is a fact a server has: a firewall default that DROPS and
            # opens only what we serve, the v6 stack too, a time source, the machine's name, and logs that survive on
            # the data partition. What is NOT measured here (and stays named in the issue) is a *persisted* static
            # address: the console's static action is session-only on purpose, and the persistent form needs a connman
            # service config - a product decision, not a row to add.
            #
            # WAIT for the firewall first: /etc/rc.d/rc.sysinit starts rc.services in the BACKGROUND, and this shell is
            # itself one of those scripts ("21serial-console"), so it exists before the last script - the one that opens
            # the ports - has run. Measured: the policies are ACCEPT and there are no port rules at the moment the shell
            # appears, and 10 s later they are DROP with 7 port rules. A row that fires immediately would "find" a
            # missing firewall that is not missing (and did, once: the first version of this row).
            wait_s = val("WAIT", "n=0; while [ $n -lt 24 ]; do iptables -S INPUT 2>/dev/null | grep -q 'dport 13305' && "
                                 "iptables -S INPUT 2>/dev/null | grep -q 'dport 22' && break; sleep 5; n=$((n+1)); done; "
                                 "echo $((n*5))", timeout=180)
            fw4 = val("FW4", "iptables -S 2>/dev/null | awk '/^-P/ { printf \"%s \", $0 }'")
            # each service port counted on its own: `dport (22|13305)` was satisfied by two rules for the same port,
            # and the row's claim is that BOTH services are reachable (review nit on #1402)
            fw22 = val("FW22", "iptables -S INPUT 2>/dev/null | grep -cE '\\-p tcp (\\-m tcp )?--dport 22 .*-j ACCEPT'")
            fw13 = val("FW13", "iptables -S INPUT 2>/dev/null | grep -cE '\\-p tcp (\\-m tcp )?--dport 13305 .*-j ACCEPT'")
            fw6 = val("FW6", "ip6tables -S 2>/dev/null | awk '/^-P/ { printf \"%s \", $0 }'")
            # the exempts, matched separately and order-insensitively: rc.firewall writes
            # `-m conntrack --ctstate RELATED,ESTABLISHED`, and a pattern demanding ESTABLISHED,RELATED in that order
            # counted 1 instead of 2 and failed a firewall that was correct (measured)
            fwbase = val("FWBASE", """sh -c 'l=$(iptables -S INPUT 2>/dev/null | grep -c -- "-i lo -j ACCEPT"); e=$(iptables -S INPUT 2>/dev/null | grep -cE "[-][-]ctstate [A-Z,]*ESTABLISHED"); echo $((l+e))'""")
            fwrules = val("FWRULES", "iptables -S INPUT 2>/dev/null | head -12 | tr '\n' ';'")
            stray = val("STRAY", "iptables -S INPUT 2>/dev/null | grep -c 'dport 9999'")
            ntp = val("NTP", "grep -h '^FallbackTimeservers' /etc/connman/main.conf 2>/dev/null")
            hn = val("HN", "hostname 2>/dev/null")
            logs = val("LOGS", "ls /data/logs 2>/dev/null | wc -l")
            print("INFO server       in place %s s after the shell: %s | %s | %s"
                  % (wait_s, fw4.strip(), fw6.strip(), ntp.strip()))
            print("INFO server       hostname=%s, /data/logs %s file(s), ACCEPT rules: tcp/22=%s tcp/13305=%s"
                  % (hn, logs, fw22, fw13))
            print("INFO server       INPUT rules: %s" % fwrules[:400])
            if "INPUT DROP" not in fw4:
                print("FAIL server       the IPv4 firewall has no default DROP on INPUT: %s" % (fw4 or "<no iptables>")); ok = False
            elif "INPUT DROP" not in fw6:
                print("FAIL server       the IPv6 firewall has no default DROP on INPUT: %s" % (fw6 or "<no ip6tables>")); ok = False
            elif not fw22.isdigit() or int(fw22) < 1:
                print("FAIL server       ssh tcp/22 is not allowed inbound (rules: %s)" % (fw22 or "<none>")); ok = False
            elif not fw13.isdigit() or int(fw13) < 1:
                print("FAIL server       Lemonade tcp/13305 is not allowed inbound (rules: %s)" % (fw13 or "<none>")); ok = False
            elif not fwbase.isdigit() or int(fwbase) < 2:
                print("FAIL server       the DROP policy has no loopback and established/related exempts (found %s)"
                      % (fwbase or "<none>")); ok = False
            elif stray != "0":
                print("FAIL server       a port nothing serves (9999) is in the INPUT rules: %s" % (stray or "<unread>")); ok = False
            elif "FallbackTimeservers" not in ntp:
                print("FAIL server       no time source configured (connman FallbackTimeservers: %s)" % (ntp or "<absent>")); ok = False
            elif not hn or hn == "(none)":
                print("FAIL server       the machine has no hostname (%s)" % (hn or "<empty>")); ok = False
            elif not logs.isdigit() or int(logs) < 1:
                print("FAIL server       nothing on /data/logs: the boot's logs do not survive (count: %s)" % (logs or "<none>")); ok = False
            else:
                print("PASS server       after %s s: firewall DROP v4+v6, the inbound services it serves (ssh tcp/22, "
                      "Lemonade tcp/13305, mDNS udp/5353) the only ACCEPT rules, loopback+established exempted; a time "
                      "source (%s), hostname %s, /data/logs %s file(s)"
                      % (wait_s, ntp.strip(), hn, logs))
            # ---- engine#712 the persistent network identity, and this row has to DISCRIMINATE (review on #1513): the
            # address it persists is one DHCP will never hand out (slirp leases from 10.0.2.15), so "the interface
            # carries it after a reboot" cannot be true by accident - and connman's own view of the service has to say
            # Method=manual, because `connmanctl config` exiting 0 proves nothing about what connman did with it.
            cur = val("CUR", "ip -4 -o addr show scope global 2>/dev/null | awk '{ print $2 \" \" $4; exit }'")
            parts = cur.split()
            if len(parts) == 2 and "/" in parts[1]:
                dev, dhcp_addr = parts[0], parts[1]
                gw = val("GW", "ip route show default 2>/dev/null | awk '/^default/ { for (i=1;i<=NF;i++) if ($i==\"via\") { print $(i+1); exit } }'")
                # an address in the same /24 that no DHCP server here will lease
                prefix = dhcp_addr.split("/")[1]
                static_addr = ".".join(dhcp_addr.split("/")[0].split(".")[:3] + ["250"]) + "/" + prefix
                setr = val("SETSTATIC", "1bit-network set-static %s %s 2>&1 | tr '\\n' '; '" % (static_addr, gw or ""), timeout=240)
                print("INFO net          set-static %s %s -> %s" % (static_addr, gw or "-", setr[:300]))
                if "persisted to /data/net.conf" not in setr:
                    print("FAIL net          set-static did not persist the address: %s" % (setr[-200:] or "<no output>")); ok = False
                else:
                    print("PASS net          %s (not the DHCP lease %s) persisted to /data/net.conf and applied now" % (static_addr, dhcp_addr))
                    now = val("NETNOW", "ip -4 -o addr show dev %s 2>/dev/null | awk '{ print $4 }'" % dev)
                    if static_addr not in now:
                        print("FAIL net          the interface does not carry %s right after set-static (%s)" % (static_addr, now or "<none>")); ok = False
                if not v.reboot():
                    print("FAIL net          the machine did not come back after the static-address reboot"); ok = False
                    v.close()
                else:
                    # zz-onboard-net runs AFTER connman (which sorts under "c"), so it lands later than the shell: wait for
                    # it rather than sampling the instant the shell appears (measured: "<none>" at the shell, the address
                    # minutes later from connman's own stored config - which is what "applied" must never look like)
                    val("WAITNET", "n=0; while [ $n -lt 18 ]; do "
                                   "ip -4 -o addr show dev %s 2>/dev/null | grep -q '%s' && break; "
                                   "sleep 5; n=$((n+1)); done; echo waited" % (dev, static_addr), timeout=180)
                    per = val("PERSIST", "ip -4 -o addr show dev %s 2>/dev/null | awk '{ print $4 }'" % dev)
                    f = val("CONF", "grep -c '^static=' /data/net.conf 2>/dev/null")
                    # `connmanctl services --properties` does not print the IPv4 configuration on this build (measured), so
                    # the proof is the applier's own log line from THIS boot plus an address DHCP would never hand out:
                    # an address cannot survive a reboot, so .250 on the interface means something applied it after boot.
                    bootline = val("BOOTLOG", "grep -c 'zz-onboard-net: .*carries 10.0.2.250' /var/log/1bit-network.log 2>/dev/null")
                    print("INFO net          after the reboot: %s | net.conf static lines=%s | applier log lines for it=%s"
                          % (per or "<none>", f, bootline or "0"))
                    if static_addr not in per:
                        print("FAIL net          after the reboot the interface does not carry the static %s (%s)" % (static_addr, per or "<none>")); ok = False
                    elif f != "1":
                        print("FAIL net          /data/net.conf lost its static line across the reboot (%s)" % (f or "0")); ok = False
                    elif not bootline.isdigit() or int(bootline) < 1:
                        print("FAIL net          the applier did not log that it applied the file after the reboot (%s lines),"
                              " so the address has no explanation on this boot" % (bootline or "<none>")); ok = False
                    else:
                        print("PASS net          after a reboot %s carries %s (which DHCP never hands out) and"
                              " /var/log/1bit-network.log has this boot's application from /data/net.conf"
                              % (dev, static_addr))
                    val("RESETNET", "1bit-network reset 2>&1 | tr '\\n' '; '", timeout=180)
                    left = val("LEFT", "ls /data/net.conf 2>&1 | head -1")
                    # and DHCP has to come back on its own: a reset that leaves the box on the static address is a
                    # half-recovery, so the row waits for the lease rather than just for the file to disappear
                    dhcp_back = val("BACK", "n=0; while [ $n -lt 24 ]; do "
                                            "ip -4 -o addr show dev %s 2>/dev/null | grep -q '%s' && break; "
                                            "sleep 5; n=$((n+1)); done; "
                                            "ip -4 -o addr show dev %s 2>/dev/null | awk '{ print $4 }'"
                                            % (dev, dhcp_addr.split('/')[0], dev), timeout=180)
                    if "No such file" not in left:
                        print("FAIL net          /data/net.conf is still there after reset (%s)" % left); ok = False
                    elif dhcp_addr.split("/")[0] not in (dhcp_back or ""):
                        print("FAIL net          after reset the link did not get its DHCP address back (%s)" % (dhcp_back or "<none>")); ok = False
                    else:
                        print("PASS net          reset removed /data/net.conf and the link is back on DHCP (%s)"
                              % dhcp_addr.split("/")[0])
            else:
                print("INFO net          no global IPv4 address on this boot (%s): the static-address rows are a SKIP" % (cur or "<none>"))
            wrote = val("WROTE", "sh -c 'echo 712 > /root/712-persist; sync; cat /root/712-persist'")
            if wrote != "712":
                print("FAIL persist      could not write to the installed root (/root/712-persist: %s)" % (wrote or "<no answer>")); ok = False
                v.close()
            elif not v.reboot():
                print("FAIL persist      the installed disk did not come back after its own reboot (transcript %s)" % (v.path + ".log")); ok = False
                v.close()
            else:
                back = val("BACK", "cat /root/712-persist")
                pm2 = val("PM", "sed -n 's/^PUPMODE=//p' /etc/rc.d/PUPSTATE | head -1")
                acct2 = val("ACCT", "id -u onebit 2>&1 | head -1")
                nacc = val("NACC", "awk -F: '$1 == \"onebit\" { n++ } END { print n+0 }' /etc/passwd")
                if back == "712" and pm2 == "12":
                    print("PASS persist      /root/712-persist still there after a reboot, PUPMODE=12 on both boots")
                    if acct2.isdigit() and nacc == "1":
                        print("PASS account      onebit still uid %s after the reboot, and one account in /etc/passwd" % acct2)
                    else:
                        print("FAIL account      after the reboot onebit=%s, %s entries in /etc/passwd "
                              "(the account did not persist, or was created twice)" % (acct2 or "<gone>", nacc or "<unread>"))
                        ok = False
                else:
                    print("FAIL persist      after a reboot /root/712-persist=%s, PUPMODE=%s "
                          "(the installed machine lost the write)" % (back or "<gone>", pm2 or "<unread>"))
                    ok = False
    finally:
        v.close()

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
