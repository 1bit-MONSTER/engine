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
# vm-check.sh <iso|img> <work-dir> [seconds]
#
# <iso>    booted the way g2-probe.sh does: -kernel/-initrd out of the image, entry 0
# <img>    the shipped artifact (os/v3/puppy/mkimage.sh): GPT + ESP + 1BIT-DATA, booted through UEFI
#          firmware as a USB disk - shim, grub, then the kernel and initrd off the data partition - and
#          the serial menu entry selected over the serial line, which is how a headless box is driven
#          With Secure Boot ON, from `sb-vars.sh`'s variable store (this machine's MOK in db): that is the
#          one check a `-kernel` boot can never make, and it is the defect the 2 Nov board boot found -
#          `error: bad shim lock signature` on a kernel image nothing had signed. The `sb state` row is the
#          guest saying so from inside.
#
# The TPUPPY-RELEASE rows that can be answered without the board: U1 (a real shell, sudo, apt install),
# U3 (lemond up at boot with its key, reachable over the network) and the software half of U4 (the engine
# on PATH, the EC driver in Puppy's ADDLIST). Runs inside QEMU over a serial root shell, because Puppy's
# console is tty1 and nothing reads the serial port it writes to: the image's /etc/init.d/21serial-console
# opens one when the kernel command line carries 1bit.serial=1.
#
# Secure Boot is read as userspace reads it: the `efivarfs` row is /proc/mounts BEFORE anything is touched
# (the appliance has to have mounted it), then `mokutil --sb-state`, then `efibootmgr` - U4's "Secure Boot
# state readable and ON" and the tool U6's installer runs. Rows that need the hardware (HRX chat on the 8060S,
# the NPU through Lemonade, the installer writing an entry for a second disk, boot time and idle RAM) are
# printed as SKIP with the reason.
#
# exit 0 = every row that can run here passed, 1 = a row failed, 2 = the VM never came up.
set -uo pipefail
prog=$(basename "$0")
[ $# -ge 2 ] || { sed -n "/^# $prog/,/^# exit 0/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
iso=$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")
work=$2
secs=${3:-600}
[ -f "$iso" ] || { echo "$prog: no such image: $iso" >&2; exit 2; }
for t in qemu-system-x86_64 xorriso python3 systemd-run curl; do
    command -v "$t" > /dev/null || { echo "$prog: $t is missing" >&2; exit 2; }
done
mkdir -p "$work" || exit 2
work=$(cd "$work" && pwd)
case $iso in
*.img) : ;;   # the artifact boots itself; its kernel and initrd live on the data partition
*)
    [ -f "$work/vmlinuz" ] || xorriso -osirrox on -indev "$iso" \
        -extract /vmlinuz "$work/vmlinuz" -extract /initrd.gz "$work/initrd.gz" > /dev/null 2>&1
    [ -s "$work/vmlinuz" ] && [ -s "$work/initrd.gz" ] || { echo "$prog: cannot extract the kernel from $iso" >&2; exit 2; }
    ;;
esac

rm -f "$work/ser" "$work/serial.log" "$work/qemu.out"
# The host port is picked, not assumed: 13305 here belongs to another lane's lemond, and the previous
# run's qemu can still hold the forward for a moment after its timeout, which fails qemu outright
# ("Could not set up host forwarding rule") before firmware runs - measured on the second cycle-4 run.
hostport=21305
i=0
while [ $i -lt 60 ]; do
    ss -lnt 2> /dev/null | grep -q ":$hostport " || break
    hostport=$((hostport + 2))
    i=$((i + 1))
done
# 22 is forwarded too: U7 is "the stick authorises the owner's key", and the only honest way to check it
# is to log in with that key over the network rather than to look at the file.
sshport=2222
i=0
while [ $i -lt 60 ]; do
    ss -lnt 2> /dev/null | grep -q ":$sshport " || break
    sshport=$((sshport + 2))
    i=$((i + 1))
done
# shellcheck disable=SC2054  # qemu arguments carry commas inside a single word
net=(-netdev user,id=n0,hostfwd=tcp:127.0.0.1:"$hostport"-:13305,hostfwd=tcp:127.0.0.1:"$sshport"-:22
     -device virtio-net-pci,netdev=n0)
case $iso in
*.img)
    # Cycle 17 (pi-release, 3 Nov 10:4x): the artifact that gets handed off must NOT be the file this harness
    # boots. A boot writes to the image (a journal, /data, the minted API key), so the sha256 published in a
    # hand-off stops being the sha256 on disk the moment a VM has touched it - measured exactly that way: an
    # image was hashed, then booted here, and the gate was right to refuse the pair. Boot a COPY, always.
    bootimg=$work/artifact-under-test.img
    if [ "$(cd "$(dirname "$iso")" && pwd -P)/$(basename "$iso")" != "$bootimg" ]; then
        cp --sparse=always "$iso" "$bootimg" || { echo "$prog: cannot copy $iso for the boot" >&2; exit 2; }
        echo "INFO artifact      booting a copy ($bootimg); the hand-off file $iso is left untouched"
    fi
    # the shipped artifact: firmware boots the ESP, grub reads the data partition. With Secure Boot on and
    # this machine's certificate enrolled, an unsigned kernel stops at grub exactly as it did on the board.
    "$(dirname "$0")/sb-vars.sh" "$work/OVMF_VARS_SB.fd" > "$work/sb-vars.txt" 2>&1 || {
        cat "$work/sb-vars.txt" >&2
        echo "$prog: cannot build the Secure Boot variable store (os/v3/puppy/sb-vars.sh) - refusing to
       run the VM without Secure Boot, which is how a signing defect reached the board" >&2
        exit 2; }
    grep -E '^   ' "$work/sb-vars.txt"
    # shellcheck disable=SC2054  # same: commas belong to the qemu arguments
    boot=(-drive file="$bootimg",format=raw,if=none,id=d0 -device qemu-xhci -device usb-storage,drive=d0
          -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.secboot.fd
          -drive if=pflash,format=raw,file="$work/OVMF_VARS_SB.fd"
          -machine q35,smm=on -global driver=cfi.pflash01,property=secure,value=on)
    export SBBOOT=1
    ;;
*)
    boot=(-machine q35 -kernel "$work/vmlinuz" -initrd "$work/initrd.gz" -append "console=ttyS0 1bit.serial=1"
          -cdrom "$iso")
    echo "SKIP sb state      -kernel boots verify nothing; the Secure Boot row needs the .img (boot the artifact)"
    ;;
esac
# systemd-run --scope runs qemu in place, so the timeout wrapper is what has to end: when the rows are done
# the VM is stopped here by name instead of being left to its timeout. Three runs once overlapped and the
# next one could not even open its image ("Failed to get write lock"), while the previous VM still held the
# port the host probe was aimed at.
# the pattern has to name the file qemu was actually given: with the .img branch booting a copy, matching on
# $iso would find nothing and leave the VM holding its image and its forwarded ports until its own timeout.
qemuimg=${bootimg:-$iso}
vmqemu() { for p in $(pgrep -f "qemu-system-x86_64 -m.*$qemuimg" 2> /dev/null); do kill "$p" 2> /dev/null; done; }
# A host-side row (U3 lan, U7 ssh) measures a FORWARD, and a forward exists only while qemu does. The cycle-18 run
# had both of them fail "connection refused" while every in-guest row passed: the rows run after the row driver, the
# qemu budget (${secs}s) had expired, and the table read that as an image defect. Ask whether the VM is still there
# and say so, so a budget overrun can never be mistaken for a broken image again.
vm_alive() { pgrep -f "qemu-system-x86_64 -m.*$qemuimg" > /dev/null 2>&1; }
vm_gone() { echo "the VM was no longer running when this row ran (qemu budget ${secs}s) - not an image finding"; }
( timeout "$secs" systemd-run --user --scope -q -p MemoryMax=6G -- \
    qemu-system-x86_64 -m 4096 -smp 4 "${boot[@]}" -display none -vga std -no-reboot \
    "${net[@]}" -serial unix:"$work/ser",server,nowait ) > "$work/qemu.out" 2>&1 &

ARTIFACT=$iso python3 - "$work" "$iso" > "$work/rows.txt" 2>&1 <<'PY'
import socket, sys, time, os, re, subprocess, shutil
work = sys.argv[1]
artifact = sys.argv[2]
log = open(os.path.join(work, "serial.log"), "ab", buffering=0)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
stop = time.time() + 60
while True:
    try:
        s.connect(os.path.join(work, "ser")); break
    except OSError:
        if time.time() > stop:
            print("FAIL boot           qemu never opened the serial socket", flush=True); sys.exit(2)
        time.sleep(0.2)

buf = b""
def read_until(token, timeout):
    """Read the serial stream until token, keeping everything in a rolling buffer."""
    global buf
    stop = time.time() + timeout
    while time.time() < stop:
        i = buf.find(token)
        if i >= 0:
            out, buf = buf[:i + len(token)], buf[i + len(token):]
            return out.decode("utf-8", "replace")
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

def menu_downs(img, work):
    """How many entries precede the serial root shell one, read out of the IMAGE's own grub.cfg.

    The harness used to send two downs, because the menu was headless, desktop, serial. Cycle 14 added the
    owner's "install to a disk" entry BEFORE the serial one, and a fixed count then booted the installer:
    the run reported "no serial shell" while ssh answered, i.e. the machine was up and the harness was on the
    wrong entry. Counting from the artifact means the rows follow the menu instead of a constant.
    """
    if not shutil.which("mcopy"):
        return 2
    try:
        out = os.path.join(work, "grub.cfg.from-image")
        subprocess.run(["mcopy", "-n", "-o", "-i", img + "@@1048576", "::/EFI/BOOT/grub.cfg", out],
                       check=True, capture_output=True)
        for i, line in enumerate(l for l in open(out) if l.startswith("menuentry ")):
            if "serial root shell" in line:
                return i
    except Exception as e:                     # no mcopy, no FAT, a different layout: the old constant
        print("INFO boot           grub.cfg not readable from the image (%s); assuming 2" % e, flush=True)
    return 2

# A disk image boots through grub, whose menu is also on the serial line. The serial entry carries pfix=nox,
# so the rows below measure the headless boot the appliance defaults to.
if artifact.endswith(".img"):
    if read_until(b"serial root shell", 120) is not None:
        downs = menu_downs(artifact, work)
        s.sendall(b"\x1b[B" * downs + b"\r")
        print("INFO boot           grub menu on serial, selected the serial-shell entry after %d down(s)" % downs,
              flush=True)

def read_until_re(pattern, timeout):
    """Like read_until, but for a regex; returns the text up to and including the match."""
    global buf
    stop = time.time() + timeout
    while time.time() < stop:
        m = re.search(pattern, buf)
        if m:
            out, buf = buf[:m.end()], buf[m.end():]
            return out.decode("utf-8", "replace")
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

# Wait for the shell itself, not only for the service's banner: on the cycle-4 image the shell came up and
# answered (root@puppypc31992:/$) while the banner never reached ttyS0, because rc.services' stdout at that
# point is not the console. Either evidence is enough, and a bare newline coaxes a prompt out.
def wait_shell(timeout):
    global buf
    stop = time.time() + timeout
    last_nudge = 0.0
    while time.time() < stop:
        if b"21serial-console: root shell on ttyS0" in buf or \
                re.search(rb"root@[^\r\n]{0,80}[$#] ", buf):
            return True
        if time.time() - last_nudge > 10:
            s.sendall(b"\n"); last_nudge = time.time()
        s.settimeout(0.5)
        try:
            d = s.recv(65536)
        except socket.timeout:
            continue
        except OSError:
            return False
        if not d:
            return False
        log.write(d); buf += d
    return False

if not wait_shell(240):
    # name the firmware refusal instead of only "no shell": with Secure Boot on, an unsigned kernel stops at
    # GRUB before init, which is exactly the failure that shipped to the board on 2 Nov
    reason = ""
    for line in open(os.path.join(work, "serial.log"), "rb").read().decode("utf-8", "replace").splitlines():
        if "bad shim lock signature" in line or "need to load the kernel first" in line or \
                "Access Denied" in line:
            reason = line.strip()
    if reason:
        print("FAIL sb chain      firmware refused the kernel: %s" % reason, flush=True)
    else:
        print("FAIL boot           no serial shell: no 21serial-console in the image, or it never ran", flush=True)
    sys.exit(2)
buf = b""   # the menu and the boot log are not row output

# The shell has to be READY FOR INPUT, not merely started: 21serial-console spawns it and prints its banner, and
# bash's readline setup then flushes the tty's pending input (TCSAFLUSH). A row sent in that window is simply
# discarded - measured on the cycle-15 image, where the first row sat waiting its whole 180 s and every row after
# it read one exchange behind (that shift is what a silent lost line looks like in a row table). So: a handshake.
# Each attempt sends a unique token and waits for it; only a shell that answers has finished initialising.
def handshake(tries=30):
    for i in range(tries):
        token = "1BIT-READY-%d" % i
        s.sendall(("echo %s\n" % token).encode())
        if read_until(token.encode(), 12) is not None:   # the echo alone proves the shell is reading
            return True
    return False

if not handshake():
    print("FAIL boot           the serial shell never answered a handshake (the harness would lose every row)",
          flush=True)
    sys.exit(2)

rows = [
    # Secure Boot is what the board runs: the chain shim -> grub -> kernel has to verify, and the guest has to
    # agree that it did. Only present when the VM booted with sb-vars.sh's store - a -kernel boot cannot.
    # Secure Boot as USERSPACE reads it: `mokutil` and `efibootmgr` both need efivarfs, and the appliance has
    # to have mounted it itself (03onboard-efivars) - a row that mounts it first proves nothing about the
    # image, which is the mistake pi-release's board script caught on cycle 9. So: read /proc/mounts BEFORE
    # touching anything, then ask mokutil. Measured on cycle 9: unmounted at row time (0 in /proc/mounts, 0
    # SecureBoot- entries) - and the very next rows then printed `SecureBoot enabled` and OVMF's boot entries
    # anyway, because **mokutil mounts efivarfs itself** when it cannot read efivars. That is precisely why
    # the mount is its own row and runs first: a tool repairing the state must not make the image look whole.
    *([("efivarfs", "grep -c ' /sys/firmware/efi/efivars ' /proc/mounts; "
        "ls /sys/firmware/efi/efivars 2>/dev/null | grep -c '^SecureBoot-'; "
        "grep -q ' /sys/firmware/efi/efivars ' /proc/mounts && echo efivarfs-mounted",
        "efivarfs-mounted"),
       ("sb state", "mokutil --sb-state 2>&1 | head -2; dmesg | grep -i 'secure boot' | head -2; "
        "dmesg | grep -i 'integrity: Loaded X.509' | head -1", "SecureBoot enabled"),
       # U6's "a firmware boot entry written": efibootmgr is the tool the installer runs, and it needs the
       # same efivars. BootCurrent is the firmware saying which entry it booted - the appliance's own name
       # only appears once an installer writes one, so that half stays the second-disk run.
       ("U6 efibootmgr", "efibootmgr 2>&1 | head -6; "
        "efibootmgr 2>&1 | grep -c '^BootCurrent:'; efibootmgr 2>&1 | grep -c '^Boot0'",
        "BootCurrent")] if os.environ.get("SBBOOT") == "1" else []),
    # The boot path the 2 Nov board panic made into a gate. Both rows read the RUNNING boot, not the tree:
    # (1) the boot media is PINNED on the command line, so init never scans another drive - the board's scan
    #     had the JetKVM's USB mass storage and the internal NVMe's ESP in it; (2) the initrd's own /init
    #     carries the rescue guard, so a failed boot writes /init-fail.log to 1BIT-DATA and keeps a console
    #     instead of exiting PID 1 (kernel panic, "Attempted to kill init! exitcode=0x00000000"). init copies
    #     /init* into /pup_new/initrd before switch_root, so /initrd/init is the file that actually ran.
    ("boot pin",     "tr ' ' '\\n' < /proc/cmdline | grep '^pupsfs='; "
                     "tr ' ' '\\n' < /proc/cmdline | grep -c '^pupsfs=1BIT-DATA$'", "pupsfs=1BIT-DATA"),
    ("init guard",   "grep -c 'one_rescue' /initrd/init 2>/dev/null; "
                     "grep -c '1bit: PID 1 must never exit' /initrd/init 2>/dev/null; "
                     "n=$(grep -c 'one_rescue 1' /initrd/init); echo paths-guarded=$n; "
                     "[ \"$n\" = 3 ] && grep -q 'trap .one_rescue' /initrd/init && echo init-guard-3-paths",
                     "init-guard-3-paths"),
    # Cycle 14, item 1 - the GPU blocker the board found: HRX registered no device because nothing ever opened
    # libhsa-runtime64.so.1. `ONEBIT_HRX_LIBHSA` is a CMake compile definition, not a variable (measured in
    # CMakeLists.txt:279 and app/serve.cpp:201), so the only runtime route on an appliance is the engine"'"'s
    # scan of /opt/rocm-therock - and even handed the file, the pinned libhsa cannot load without its
    # $ORIGIN siblings. This row asserts all three halves on the artifact: the set at the path the scan walks,
    # no unresolved dependency, and a dlopen by BARE soname inside the guest (what libggml-hrx does when the
    # engine has not exported IREE_HAL_AMDGPU_LIBHSA_PATH).
    ("U3 hsa runtime", "ls -l /opt/rocm-therock/lib/libhsa-runtime64.so.1; "
                     "echo not-found=$(LD_LIBRARY_PATH=/opt/rocm-therock/lib ldd "
                     "/opt/rocm-therock/lib/libhsa-runtime64.so.1 2>&1 | grep -c 'not found'); "
                     "ls -ld /opt/1bit/libhsa; ls /opt/rocm-therock/lib | wc -l; "
                     "python3 -c 'import ctypes; ctypes.CDLL(\"/opt/rocm-therock/lib/libhsa-runtime64.so.1\"); "
                     "print(\"hsa-abs-ok\")' 2>&1 | tail -1; "
                     "LD_LIBRARY_PATH=/opt/rocm-therock/lib python3 -c 'import ctypes; "
                     "ctypes.CDLL(\"libhsa-runtime64.so.1\"); print(\"hsa-soname-ok\")' 2>&1 | tail -1; "
                     # the file's lines reach a CHILD only if they are exported - the cycle-13 file assigned
                     # them as plain variables in a sourced file, which exports nothing. This is the same
                     # question 10lemonade asks of it, answered by a subshell.
                     "sh -c '. /etc/1bit/libhsa.env; echo exported LD_LIBRARY_PATH=$LD_LIBRARY_PATH "
                     "SVM=$HSA_USE_SVM HRX=$HRX_ALLOW_NO_SVM' | grep -q 'LD_LIBRARY_PATH=/opt/rocm-therock/lib' "
                     "&& echo libhsa-env-exported",
                     "libhsa-env-exported"),
    # Cycle 14, item 1 second half: the GTT ceiling and CWSR are on the boot line (the reference Ubuntu install
    # runs ttm.pages_limit=31457280 ttm.page_pool_size=31457280 amdgpu.cwsr_enable=0; the cycle-13 image had
    # none of them). Read off the RUNNING boot, so it is the cmdline the kernel got, not the tree.
    ("boot gtt",     "tr ' ' '\\n' < /proc/cmdline | grep -E '^(ttm\\.|amdgpu\\.)'; "
                     "tr ' ' '\\n' < /proc/cmdline | grep -c '^ttm.pages_limit=31457280$'",
                     "ttm.pages_limit=31457280"),
    # Cycle 14, item 2: who owns tty1. `U2 tty1` reads what a person sees; this says which process put it
    # there, because the board run had a prompt in QEMU and none on the hardware and only this line names the
    # difference. 1bit-console is the appliance's own entry (inittab), plogin is what it replaced.
    ("U2 tty1 owner", "for p in /proc/[0-9]*; do t=$(readlink \"$p/fd/0\" 2>/dev/null); "
                     "case \"$t\" in /dev/tty1) echo \"tty1-owner: $(tr -d '\\000' < $p/cmdline | tr '\\000' ' ')\";; esac; done | head -2; "
                     "pgrep -af 'getty|plogin|1bit-console' | head -3; "
                     "grep -c '^tty1::respawn:/usr/local/bin/1bit-console$' /etc/inittab", None),
    # Cycle 14, item 5: release-check SKIPped `fans` on the board ("no /usr/sbin/1bit-board in this image").
    # The tool is os/mini/board.sh, installed as the path that row reads. On a machine whose EC it does not
    # know it must write nothing and say so - which is what this QEMU boot is.
    ("U4 board tool", "ls -l /usr/sbin/1bit-board | tail -1; /usr/sbin/1bit-board known; echo known-rc=$?; "
                     "/usr/sbin/1bit-board status 2>&1; /usr/sbin/1bit-board power-status 2>&1; "
                     "[ -x /usr/sbin/1bit-board ] && echo board-tool-present", "board-tool-present"),
    # Cycle 14, item 7 (owner): the "Install 1bit OS to a disk" entry, in the image"'"'s own grub.cfg - read off
    # the ESP of the disk it booted, not from the tree.
    ("U6 install entry", "mkdir -p /tmp/esp1; mount -o ro /dev/sda1 /tmp/esp1 2>/dev/null && { "
                     "grep -c 'install to a disk' /tmp/esp1/EFI/BOOT/grub.cfg; "
                     "grep -q 'install to a disk' /tmp/esp1/EFI/BOOT/grub.cfg && echo install-entry-grub; "
                     "umount /tmp/esp1; }", "install-entry-grub"),
    # ... and the guard, which is the part that must not be trusted to a person"'"'s typing: the boot disk is
    # refused both ways (--check, and the interactive path with the disk name fed in), with nothing written.
    ("U6 install guard", "1bit-console-install --check /dev/sda > /tmp/ci1.txt 2>&1; echo check-rc=$?; "
                     "tail -2 /tmp/ci1.txt; "
                     "printf '%s\\n' /dev/sda | 1bit-console-install > /tmp/ci2.txt 2>&1; "
                     "echo typed-rc=$?; grep -c REFUSED /tmp/ci2.txt; "
                     "1bit-console-install --list 2>&1 | head -4", "typed-rc=2"),
    # U1 as the board asked for it (owner, cycle 12): the LOGIN shell is bash, not only the bash binary
    # being present. /etc/passwd is what plogin -> getty -n -l /bin/autologin ends in.
    ("U1 login shell", "awk -F: '{print $1, $7}' /etc/passwd | head -4; "
                     "rootsh=$(awk -F: '{if ($1 == \"root\") print $7}' /etc/passwd); echo root=$rootsh; "
                     "[ \"$rootsh\" = /bin/bash ] && echo login-shell-bash", "login-shell-bash"),
    ("U1 shell",     "echo SHELL=$SHELL; command -v bash; bash --version | head -1", "bash"),
    # U2 headless (owner, cycle 12 item 7): the default entry boots with no X, and tty1 is a login prompt.
    # Puppy's own chain does the second half - /etc/inittab has tty1::respawn:plogin and plogin execs
    # `getty -n -l /bin/autologin ... tty1` - so the row asserts the getty, which is what a person sees.
    # Cycle 13 (pi-release, 18:39): a headless boot has to reach tty1's prompt BY ITSELF. ttyS0 is not the
    # question - it is a root shell this image opens on purpose - so this row reads tty1's OWN screen through
    # the image's own /usr/local/bin/1bit-tty1 (the kernel's VGA text buffer, /dev/vcsa1: what a person in
    # front of the box sees). rc.country's first-boot dialogs (keyboard, locale, timezone) print there, so any
    # of their words on that screen is a FAIL - nothing answers a console dialog on an appliance - and a
    # screen with nothing on it is its own verdict, because then the prompt never came either.
    ("U2 tty1",      "1bit-tty1 1 | head -10; "
                     "if 1bit-tty1 1 | grep -qiE 'keyboard|locale|timezone|quicksetup|country'; then "
                     "echo tty1-wizard-text; elif [ -n \"$(1bit-tty1 1 | tr -d ' \\n')\" ]; then "
                     "echo tty1-no-wizard; else echo tty1-screen-empty; fi", "tty1-no-wizard"),
    ("U2 headless",  "tr ' ' '\\n' < /proc/cmdline | grep '^pfix='; "
                     "pgrep -x Xorg > /dev/null && echo X-RUNNING || echo no-X-server; "
                     "pgrep -af 'getty.*tty1' | head -1; "
                     "pgrep -af 'Xorg|X ' | head -1", "no-X-server"),
    # Cycle 15, item C, read off the RUNNING boot: the LAST console= wins, and it has to be tty0 or the screen
    # shows nothing while the SFS are copied into RAM - the owner's "it hangs hard here" on the c14 board boot,
    # where /dev/console was ttyS0 and every userspace progress line went to the serial port instead.
    ("boot console", "tr ' ' '\\n' < /proc/cmdline | grep '^console='; "
                     "last=$(tr ' ' '\\n' < /proc/cmdline | sed -n 's/^console=//p' | tail -1); "
                     "[ \"$last\" = tty0 ] && echo console-is-tty0", "console-is-tty0"),
    # Cycle 15, item D: the board's only non-fault dmesg lines were a pcspkr double registration (snd_pcsp and
    # pcspkr both bind the PC speaker; the ALSA one is blacklisted in the image) and an HDMI infoframe error that
    # amdgpu prints on an unconnected output and that is left alone. This row reads the file that does it and
    # what actually bound, so a later edit that drops the file is visible here rather than in a boot log nobody reads.
    ("boot quiet",   "grep -H 'blacklist' /etc/modprobe.d/1bit-quiet.conf 2>&1 | head -2; "
                     "echo snd_pcsp-loaded=$(lsmod | grep -c '^snd_pcsp'); "
                     "dmesg | grep -ci 'pcspkr' ; echo quiet-boot-checked", "quiet-boot-checked"),
    # Cycle 15, item B: the appliance is 1bit-os. rc.sysinit only invents puppypcNNNNN while /etc/hostname still
    # holds the stock word, so this reads the running hostname AND the file that decides the next boot.
    ("U1 hostname",  "hostname; cat /etc/hostname; grep -c 1bit-os /etc/hosts; "
                     "[ \"$(hostname)\" = 1bit-os ] && [ \"$(cat /etc/hostname)\" = 1bit-os ] && echo hostname-1bit-os",
                     "hostname-1bit-os"),
    # Cycle 15, item A - the headline. Three rows, because "a menu exists" and "a person can use it" are two
    # different claims: (1) the SCREEN a person sees carries the menu (tty1's own VGA text buffer via the image's
    # own 1bit-tty1, the reader the U2 tty1 row already uses); (2) the status block behind it answers every line
    # the owner asked for, as data; (3) every item runs without a keyboard (--selftest) - an item that only works
    # when a human presses it is exactly what a later run cannot re-check.
    ("U2 menu",      "1bit-tty1 1 | head -30; "
                     "if 1bit-tty1 1 | grep -q 'Show the API key' && 1bit-tty1 1 | grep -q 'Reboot / power off'; then "
                     "echo tty1-menu; elif 1bit-tty1 1 | grep -q 'root@'; then echo tty1-bare-shell; "
                     "else echo tty1-nothing; fi", "tty1-menu"),
    # every line the owner asked for, by name and present rather than at column 0 (two of them share a line);
    # `Secure` matches "Secure Boot", and the four-letter words are unambiguous in this block
    ("U2 menu status", "1bit-console-menu --status 2>&1 | head -10; "
                     "for w in Lemonade GPU NPU Secure Fans Data Models; do "
                     "1bit-console-menu --status 2>&1 | grep -q \"$w\" || echo missing-line-$w; done; "
                     "1bit-console-menu --status 2>&1 | grep -q 'http://' && echo menu-status-lines",
                     "menu-status-lines"),
    ("U2 menu actions", "1bit-console-menu --selftest > /tmp/menu-self.txt 2>&1; echo selftest-rc=$?; "
                     "grep -E '^(--- item|item [0-9] FAILED|api-key-shown|auth-mode|models-listed|install-guard-checked|desktop-available|shell-bash|logs-shown|power-noop)' "
                     "/tmp/menu-self.txt | tr '\\n' ' '", "selftest-rc=0"),
    # Cycle 17 item 4: the menu's probes run with Lemonade's environment, and no line is cut in the middle of a
    # word (the c16 board showed "firmware prese"). Both are read off the shipped script and its own status
    # block, so an edit that drops either is a red row rather than a nicer-looking menu.
    ("U2 menu env", "grep -c 'libhsa.env' /usr/local/bin/1bit-console-menu; "
                     "1bit-console-menu --status | awk 'length($0) > 78 {print \"long-line:\" $0}'; "
                     # the c16 board's line was cut mid-word ("firmware prese"): the whole words must be there
                     "1bit-console-menu --status | grep -q 'firmware present' && echo lines-whole || echo truncated",
                     "lines-whole"),
    # Cycle 17 item 8: the key is not in the page or its URL, and the same-origin proxy still answers WITH it.
    # Cycle 20: the direct probe follows the MODE - blank (the shipped default) answers 200 without a header, keyed
    # answers 401 - so the row proves the mode is coherent rather than pinning one auth behaviour.
    # (the page's own text mentions "?key=" in a comment, so the count is of a KEY-SHAPED value, not of the word)
    ("U3 webapp key", "n=$(curl -s -m 5 http://127.0.0.1:8080/ | grep -Ec 'key=[0-9a-f]{16,}'); "
                     "a=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/v1/models); "
                     "b=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:13305/api/v1/models); "
                     "k=$(sed -n 's/^LEMONADE_API_KEY=//p' /etc/1bit/lemond.conf 2>/dev/null | head -1); "
                     "if [ -n \"$k\" ]; then mode=keyed want=401; else mode=blank want=200; fi; "
                     "echo \"key-in-page=$n proxy=$a direct-no-key=$b mode=$mode\"; "
                     "[ \"$n\" = 0 ] && [ \"$a\" = 200 ] && [ \"$b\" = \"$want\" ] && echo webapp-key-not-in-url",
                     "webapp-key-not-in-url"),
    # Cycle 18 items 1-2, from the c17 board demo: a thinking model (Qwen3) spent the whole token budget in
    # reasoning_content, so the page - which read only delta.content - showed a mid-sentence stop and then
    # "done: 0 chunks"; and the model could only be chosen with ?model= in the URL. The page the proxy actually
    # serves must ask for enable_thinking:false, render reasoning_content, give the answer 1024 tokens, and carry
    # the picker. It is read off the RUNNING server (8080), not off the file in the tree.
    ("U3 webapp page", "p=$(curl -s -m 5 http://127.0.0.1:8080/); "
                     "t=$(printf '%s' \"$p\" | grep -c enable_thinking); "
                     "r=$(printf '%s' \"$p\" | grep -c reasoning_content); "
                     "k=$(printf '%s' \"$p\" | grep -c 'select id=.model.'); "
                     "m=$(printf '%s' \"$p\" | grep -c 'MAXTOK = 1024'); "
                     # engine#924: Lemonade does not retry a 503 device_busy on the request path, so the page is
                     # the only layer that can - and the refusal reaches it as EITHER a 503 + Retry-After or, for a
                     # streaming request (what this page sends), one `data: {"error":...}` frame before [DONE].
                     # Both spellings and the retry text must be in the served page, not only in the source.
                     "b=$(printf '%s' \"$p\" | grep -c 'device_busy'); "
                     "w=$(printf '%s' \"$p\" | grep -c 'NPU busy, retrying'); "
                     "echo \"thinking=$t reasoning=$r picker=$k max_tokens=$m busy=$b retry-text=$w\"; "
                     "[ \"$t\" -ge 1 ] && [ \"$r\" -ge 1 ] && [ \"$k\" = 1 ] && [ \"$m\" = 1 ] && "
                     "[ \"$b\" -ge 1 ] && [ \"$w\" -ge 1 ] && "
                     "echo webapp-thinking-picker-busy",
                     "webapp-thinking-picker-busy"),
    # Cycle 18 item 4 (the owner: "when you first boot the headless there is no data to display on how to
    # connect, just the menu"): the screen a person sees must carry the CONNECT block - the address, the web UI,
    # the API base, the key's location, ssh and the mobile app - not just the menu. Read off tty1's own VGA
    # buffer, the same surface the U2 tty1 row uses, so this is the screen and not the script.
    # tr -s ' ': the screen is drawn by ONE OF TWO renderers now (the dialog menu, which collapses runs of spaces in
    # a --menu prompt, and the ratatui TUI, which does not), so the row must not depend on how many spaces sit
    # between "api" and its URL. Measured on both renders of the same facts.
    ("U2 menu connect", "s=$(1bit-tty1 1 | tr -s ' '); printf '%s\\n' \"$s\" | head -10; "
                     "for w in CONNECT 'web UI http://' 'api http://' 'ssh root@' 'Lemonade Mobile'; do "
                     "printf '%s' \"$s\" | grep -qF \"$w\" || echo missing-\"$w\"; done; "
                     "printf '%s' \"$s\" | grep -qE 'web UI http://[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+:13305' && "
                     "echo menu-connect-block",
                     "menu-connect-block"),
    # The mechanism behind U3-lan and U7 (cycle 15's finding, and the row the c16 refactor folded away): inbound
    # connections from the LAN pass only if the appliance opened the ports AFTER Puppy's rc.firewall (which ends
    # in `-P INPUT DROP`) and let the RFC1918 ranges through tcpwrappers. Printing all three - the rules, the
    # listener, hosts.allow - is what tells "the image never opened them" apart from "something else answered".
    ("U3 inbound", "iptables -S INPUT 2> /dev/null | head -8; "
                     "r=$(iptables -S INPUT 2> /dev/null | grep -c dport); "
                     "l=$(netstat -lnt 2> /dev/null | grep -cE ':22 |:13305 '); "
                     "a=$(grep -c 1bit-lan /etc/hosts.allow 2> /dev/null); "
                     "echo \"dport-rules=$r listeners=$l hosts-allow-lan=${a:-0}\"; "
                     "[ \"$r\" -ge 2 ] && [ \"$l\" -ge 2 ] && [ \"${a:-0}\" -ge 1 ] && echo inbound-open",
                     "inbound-open"),
    # Cycle 17 item 3: the appliance's name is /etc/hostname, not the DHCP server's opinion - connman's
    # AllowHostnameUpdates defaults to true, which is the one path a name nobody typed can take.
    ("U1 hostname pin", "grep -n '^AllowHostnameUpdates' /etc/connman/main.conf; "
                     "grep -q '^AllowHostnameUpdates = false' /etc/connman/main.conf && echo hostname-dhcp-pinned",
                     "hostname-dhcp-pinned"),
    # Cycle 17 item 2: the boot's evidence is on the DATA partition, not in RAM.
    # Cycle 22 (TOS3-NOTE-c21-logs): the log has to be ON the partition AND the daemon has to be alive. The 4 Nov
    # failure was a boot whose /data mounted after 06onboard-logs' one-shot test: no log on the stick, no daemon, and
    # nothing failing. A 1bit-logs that only tested "can I mkdir" would have written the "persisted" log to RAM and
    # reported success, which is why the mount point is part of the marker.
    ("U3 boot logs", "grep -q ' /data ' /proc/mounts && echo data-is-mount; "
                     "pgrep -f '1bit-logs daemon' > /dev/null && echo logs-daemon-running; "
                     "ls -d /data/logs/*/ 2> /dev/null | tail -2; "
                     "grep -q cmdline /data/logs/*/BOOT 2> /dev/null && "
                     "grep -q ' /data ' /proc/mounts && pgrep -f '1bit-logs daemon' > /dev/null && "
                     "echo boot-logs-on-data", "boot-logs-on-data"),
    # The same mechanism, by hand, in the guest: the daemon is pointed at a directory that is NOT a mount point
    # (nothing may be written there - that is the RAM log), then a tmpfs is mounted over it late, which is /data
    # arriving after 06onboard-logs ran. The log must appear without anything restarting the daemon.
    ("U3 logs wait", "d=/tmp/logs-late; umount $d 2> /dev/null; rm -rf $d; mkdir -p $d; "
                     "ONEBIT_LOG_DATA=$d 1bit-logs daemon > /tmp/logs-late.out 2> /tmp/logs-late.err & "
                     "p=$!; sleep 4; pre=$(ls $d 2> /dev/null | wc -l); "
                     "mount -t tmpfs none $d && sleep 10; post=$(find $d -name BOOT 2> /dev/null | wc -l); "
                     "kill $p 2> /dev/null; umount $d 2> /dev/null; "
                     "echo \"logs-wait pre=$pre post=$post\"; tail -2 /tmp/logs-late.err; "
                     "[ \"$pre\" = 0 ] && [ \"$post\" -ge 1 ] && echo logs-wait-for-data", "logs-wait-for-data"),
    # Cycle 17 item 1: menu item 6 can never reset the machine - the guard is built and a reboot THROUGH it is
    # refused, which is the same call xwin's exit path makes (`exec reboot`) and the WM's shutdown menu makes.
    ("U2 desktop safe", "1bit-desktop --check | tail -3; 1bit-desktop --guard-test 2>&1 | tail -2",
                     "reboot-refused"),
    # Cycle 17 item 9: the desktop's first run is preset (zone, no Quick Setup, the browser's default-browser
    # prompt off) and the chat screen has a startup hook Puppy actually runs.
    ("U2 desktop preset", "cat /etc/timezone; readlink /etc/localtime; date; ls /etc/personal_settings_popup_disabled "
                     "/root/Startup/1bit-lemonade-webapp /etc/1bit/palemoon-prefs.js 2>&1; "
                     "grep -c checkDefaultBrowser /etc/1bit/palemoon-prefs.js; "
                     "grep -c 'defaults/pref/1bit-os.js' /usr/local/bin/1bit-lemonade-webapp; "
                     "[ \"$(readlink /etc/localtime)\" = /usr/share/zoneinfo/America/Moncton ] && "
                     "echo desktop-preset-ok", "desktop-preset-ok"),
    ("U1 sudo",      "id -un; sudo -n id -un 2>&1 | tail -2", "root"),
    # the wanted string must be evidence of an INSTALL, not the word appearing in an error: the previous
    # version matched "Unable to locate package neofetch" and reported a pass (measured 2026-11-02)
    # neofetch is not in the 26.04 archive at all (measured on this box: `apt-cache policy neofetch` and
    # `apt-cache search '^neofetch$'` are both empty), so the owner's exact command cannot pass on resolute.
    # The row stays, as a fact rather than a gate, and fastfetch is the packaged equivalent.
    ("U1 neofetch",  "apt install -y neofetch 2>&1 | tail -2; "
                     "command -v neofetch && echo neofetch-installed || echo neofetch-not-in-resolute", None),
    # Both apt rows wait for dpkg to be idle first: U1 fastfetch's install leaves a postinst running (Puppy's
    # menu/desktop hooks print `find: /usr/X11R7/bin`), and the row that follows was measured failing on the lock
    # rather than on the package - cowsay is in the 26.04 archive (`apt-cache policy cowsay`), so a red row here
    # would have been the harness, not the image. The tail is 3 lines so a real apt error is visible.
    ("U1 fastfetch", "for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -x dpkg > /dev/null || break; sleep 3; done; "
                     "apt update 2>&1 | tail -2; apt install -y fastfetch 2>&1 | tail -3; "
                     "command -v fastfetch && echo fastfetch-installed", "fastfetch-installed", 600),
    ("U1 apt-any",   "for i in 1 2 3 4 5 6 7 8 9 10; do pgrep -x dpkg > /dev/null || break; sleep 3; done; "
                     "apt install -y --no-install-recommends cowsay 2>&1 | tail -3; "
                     "command -v cowsay && echo apt-install-works", "apt-install-works", 600),
    ("U3 service",   "ps | grep -c '[l]emond'; netstat -lntp 2>/dev/null | grep -c 13305", None),
    ("U3 key",       "k=$(sed -n 's/^LEMONADE_API_KEY=//p' /etc/1bit/lemond.conf 2>/dev/null | head -1); "
                     "if [ -n \"$k\" ]; then echo \"auth-mode=keyed (${#k} hex chars)\"; "
                     "else echo 'auth-mode=blank (no LEMONADE_API_KEY line)'; fi",
                     "auth-mode=blank"),
    # Cycle 20, the owner's "blank key": blank is the SHIPPED state, so these rows assert the MODE and the mode's own
    # contract rather than a key nobody asked for. `U3 key` names what this image ships; `U3 auth` proves it (blank:
    # 200 with no header at all; keyed: 401 without the key, 200 with it) as one token, whichever mode is in force.
    ("U3 key file",  "readlink -f /etc/1bit/lemond.conf; "
                     "readlink -f /etc/1bit/lemond.conf | grep -q '/data/lemonade/lemond.conf' && echo key-file-on-data",
                     "key-file-on-data"),
    ("U3 auth",      "k=$(sed -n 's/^LEMONADE_API_KEY=//p' /etc/1bit/lemond.conf 2>/dev/null | head -1); "
                     "n=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:13305/api/v1/models); "
                     "if [ -n \"$k\" ]; then "
                     "y=$(curl -s -o /dev/null -w '%{http_code}' -H \"Authorization: Bearer $k\" http://127.0.0.1:13305/api/v1/models); "
                     "echo keyed no-key=$n with-key=$y; [ \"$n\" = 401 ] && [ \"$y\" = 200 ] && echo auth-mode-ok; "
                     "else echo blank keyless=$n; [ \"$n\" = 200 ] && echo auth-mode-ok; fi",
                     "auth-mode-ok"),

    ("U3 catalogue", "k=$(sed -n 's/^LEMONADE_API_KEY=//p' /etc/1bit/lemond.conf 2>/dev/null | head -1); "
                     "curl -s ${k:+-H \"Authorization: Bearer $k\"} http://127.0.0.1:13305/api/v1/models "
                     "| grep -o onebit | wc -l", None),
    ("U3 api local", "k=$(sed -n 's/^LEMONADE_API_KEY=//p' /etc/1bit/lemond.conf 2>/dev/null | head -1); "
                     "curl -s -o /dev/null -w 'http=%{http_code}' ${k:+-H \"Authorization: Bearer $k\"} "
                     "http://127.0.0.1:13305/api/v1/models", "http=200"),
    ("U3 keyvalue",  'sed -n "s/^LEMONADE_API_KEY=//p" /etc/1bit/lemond.conf 2>/dev/null | head -1', None),
    # Cycle 20, pi-prs' note 2 on #934: item 1's mint is the ONE new write path in this image and it was reachable
    # only from a dialog. The menu carries its own selftest for it, driven on a throwaway conf with the restart
    # replaced by `true`, so this row proves the write (one 48-hex line, mode 600, the other lines kept) and the
    # removal (no line, no temp file) INSIDE the image without touching /etc/1bit or restarting Lemonade.
    ("U3 key toggle", "d=$(mktemp -d); 1bit-console-menu --mint-selftest \"$d\" | tail -2; rm -rf \"$d\"",
                     "mint-selftest-ok"),
    # Cycle 20 item 6 and item 5 in one row: the box's own Lemonade UI works THROUGH the key-injecting proxy (health
    # 200), a same-origin page is served, and a request that carries a foreign Origin is refused before the key is
    # attached - i.e. the proxy cannot be borrowed as a key-adding open relay by a page from anywhere else.
    ("U3 proxy",     "h=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/api/v1/health); "
                     "o=$(curl -s -o /dev/null -w '%{http_code}' -H 'Origin: http://localhost:8080' http://127.0.0.1:8080/api/v1/health); "
                     "x=$(curl -s -o /dev/null -w '%{http_code}' -H 'Origin: http://evil.example' http://127.0.0.1:8080/api/v1/health); "
                     "echo proxy-health=$h same-origin=$o cross-origin=$x; "
                     "[ \"$h\" = 200 ] && [ \"$o\" = 200 ] && [ \"$x\" = 403 ] && echo webapp-proxy-keyed-and-guarded",
                     "webapp-proxy-keyed-and-guarded"),
    ("U4 engine",    "command -v 1bit; 1bit --version 2>&1 | head -2", "1bit"),
    ("U4 addlist",   "grep -i '^ADDLIST' /etc/rc.d/MODULESCONFIG; lsmod | grep -c ec_su_axb35", "ec_su_axb35"),
    ("U4 package",   "ls /opt/1bit-package | tr '\\n' ' '; ls -l /usr/local/bin/1bit /usr/local/bin/lemond-onebit | tail -2",
                     "/opt/1bit-package"),
    # U3's CLI (cycle 12 item 2): the owner is told `lemonade` works from bash, so it has to be there and
    # it has to be the release package's binary, not something a package manager left behind.
    ("U3 cli",       "ls -l /usr/local/bin/lemonade; readlink /usr/local/bin/lemonade; "
                     "lemonade --version 2>&1 | head -2", "/opt/1bit-package/bin/lemonade"),
    # U4 (cycle 12 item 4): the release check ships in the appliance. Its GPU/NPU rows SKIP in the VM (no device), and
    # the board run is the one that has to be 0 FAIL. Cycle 21: this row no longer just RECORDS the result - the c20
    # image reached the owner with release-check saying `FAIL models` and every chat row SKIPped while this row printed
    # it as INFO. It now fails on the same line the board reads.
    ("U4 release-check", "/usr/local/bin/1bit-release-check --self-test 2>&1 | tail -1; "
                     "/usr/local/bin/1bit-release-check > /tmp/release-check.log 2>&1; "
                     "sed -n 's/^\\(FAIL\|SKIP\\)  *\\([a-z0-9-]*\\).*/\\1 \\2/p' /tmp/release-check.log "
                     "| tr '\\n' ' '; echo; "
                     "grep '^RESULT' /tmp/release-check.log; "
                     "grep -c '^PASS ' /tmp/release-check.log", "RESULT: PASS", 900),
    # Item 10 (owner 2026-11-04): the appliance announces itself as 1bit-os.local, so http://1bit-os.local:13305 is a
    # link a phone can click. Two halves are asserted HERE: a responder is up, and the name resolves on the box.
    # Whether a PEER on the LAN sees it needs multicast from outside the guest, which a QEMU user-mode network does
    # not carry - that is the board's row, and the hand-off says so instead of implying this one covers it.
    ("U2 mdns", "tail -2 /var/log/mdns.log 2>/dev/null; pidof avahi-daemon >/dev/null 2>&1 && echo avahi-up; "
                "getent hosts 1bit-os.local", "1bit-os.local"),
    # U6 (cycle 12 item 5): the installer the release ships, and the tools it needs on this boot.
    ("U6 installer", "ls -l /usr/local/bin/1bit-install | tail -1; "
                     "ls /bin/1bit-install; sfdisk --version 2>&1 | head -1; "
                     "command -v efibootmgr mkfs.vfat mkfs.ext4; echo installer-present", "installer-present"),
    ("U3 log",       "tail -3 /var/log/lemond-onebit.log 2>&1", None),
    # BEFORE any in-VM fix runs: on an image that carries the fixes these two say so (root, and 1), on an
    # older image they say what is missing. The fix rows below then repair it so the login can still be
    # measured on that image - which is the difference between "the fix works" and "this artifact has it".
    ("image pre",    "ls -l /root/.ssh/authorized_keys | awk '{print \"key-owner=\"$3\"/\"$4}'; "
                     "echo hosts-lan=$(grep -c 1bit-lan /etc/hosts.allow); "
                     "ls /lib/modules/$(uname -r)/modules.dep >/dev/null 2>&1 && echo modules-tree=visible "
                     "|| echo modules-tree=MISSING", None),
    # the fix, measured before it ships: StrictModes wants root-owned keys, and the image's file was not
    ("U7 key-owner", "chown 0:0 /root/.ssh /root/.ssh/authorized_keys 2>&1; "
                     "ls -l /root/.ssh/authorized_keys | awk '{print $3\"/\"$4\"/\"$1}'; "
                     "/usr/sbin/sshd -t 2>&1 | head -2; echo key-owner-done", None),
    # the in-VM half of the fix, so it is measured before a rebuild: tcpwrappers is what closes the kex
    ("U7 hosts",     "grep -v '^#' /etc/hosts.deny | grep . ; grep -v '^#' /etc/hosts.allow | grep . ; "
                     "grep -q '1bit-lan' /etc/hosts.allow || "
                     "printf 'ALL: LOCAL, 10.0.0.0/255.0.0.0, 172.16.0.0/255.240.0.0, 192.168.0.0/255.255.0.0  "
                     "# 1bit-lan\\n' >> /etc/hosts.allow; "
                     "grep -c '1bit-lan' /etc/hosts.allow", None),
    ("U7 listen",    "netstat -lntp 2>/dev/null | grep ':22 ' | head -2; "
                     "grep -iE '^(PermitRootLogin|DenyUsers|AllowUsers|Port|PasswordAuthentication)' "
                     "/etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf 2>/dev/null | head -6", None),
    ("U7 sshd",      "ps | grep -c '[s]shd'; ls -l /root/.ssh/authorized_keys 2>&1 | tail -1; "
                     "ls /etc/ssh/ssh_host_* 2>&1 | head -3; /usr/sbin/sshd -t 2>&1 | head -3; echo sshd-checked", None),
    ("data mount",   "grep -c ' /data ' /proc/mounts; df -h /data 2>/dev/null | tail -1", "/data"),
    # Cycle 20: blank is the shipped mode, so there may be NO key file yet - what this row owes is that when one is
    # turned on it lands on /data (the symlink 05onboard-data makes), not that a key exists.
    ("data key",     "ls -l /etc/1bit/lemond.conf; readlink -f /etc/1bit/lemond.conf; "
                     "readlink -f /etc/1bit/lemond.conf | grep -q '/data/lemonade/lemond.conf' && echo key-path-on-data",
                     "key-path-on-data"),
    ("data ssh",     "ls -l /etc/ssh/ssh_host_ed25519_key; readlink -f /etc/ssh/ssh_host_ed25519_key", "/data/ssh/"),
    ("data state",   "ls /data; grep -c . /data/lemonade/config/user_models.json 2>/dev/null || echo no-registry", None),
    # release identity: the package reports "1bit 0.0.1" whatever was built, so the binary hash is what ties
    # the image to a payload. Compare it with `sha256sum <payload>/1bit/1bit` on the build machine.
    ("U4 engine sha", "sha256sum /opt/1bit-package/1bit/1bit | cut -c1-32", None),
    # the two paths pi-release asked for, asserted on a booted image rather than in a listing
    ("U4 hrx path",   "ls -l /opt/1bit-llama/llama-server 2>&1 | tail -1; "
                      "/opt/1bit-llama/llama-server --version 2>&1 | head -2", None),
    ("U4 npu sets",   "ls /opt/1bit-package/1bit/npu/kernels", None),
    # U3 says "reachable on the LAN", so whether the appliance has an address at all is a row, not a
    # footnote: the cycle-2 image booted with no /data and answered nothing on the forwarded port, and
    # these two lines say whether that was the listener or the interface.
    # the durable assertion is the RULE, not the policy: whether rc.firewall has finished at the moment a
    # row runs is a race, but an ACCEPT for tcp/13305 is what makes U3 reachable either way.
    ("U3 port rule", "iptables -C INPUT -p tcp --dport 13305 -j ACCEPT && echo port-13305-open; "
                     "iptables -C INPUT -p tcp --dport 22 -j ACCEPT && echo port-22-open", None),
    ("net addr",     "ip -4 -o addr show 2>/dev/null | tr -s ' '", None),
    ("net route",    "ip -4 route 2>/dev/null | head -3", None),
]
for idx, row in enumerate(rows):
    name, cmd, want = row[0], row[1], row[2]
    limit = row[3] if len(row) > 3 else 180
    # The sentinel is expanded by the shell, so the terminal's echo of the command line shows
    # "__DONE_$(date +%s%N)__" and only the real answer carries digits. A literal sentinel matched its own
    # echo: every row after the first came back with an empty body and was reported as a failure.
    # A row is sent once, and once again if it got no answer: a LINE THAT NEVER RAN must not move every row after
    # it one exchange out of step, which is what a silently discarded line looks like in a row table (measured on
    # the cycle-15 image before 21serial-console stopped using readline: row 1 waited its whole 180 s, and rows 2..n
    # each reported the PREVIOUS row's output). The resend is the harness being suspicious of its own transport.
    # The retry needs a sentinel that identifies THIS attempt, and the reason is a bug this retry caused on the
    # cycle-16 prepared copy (19 PASS / 19 FAIL, every row holding the previous row's answer): when the first line
    # HAD run but its answer was slow, the resend queued a second identical command, the row matched the first
    # answer, and the second answer became the next row's body. With a per-attempt token a stale answer can never
    # match, and whatever a duplicate leaves behind is dropped instead of being fed to the next row.
    out = None
    for attempt in (1, 2):
        # the token must NOT be a literal in the command line: the terminal echoes the line, so a fixed token
        # matched its own echo and every row came back empty (measured). `$(date +%s%N)` is expanded only when the
        # command RUNS, and the per-attempt number in front keeps a stale attempt from matching this one.
        s.sendall((cmd + "; echo __DONE_" + str(attempt) + "_$(date +%s%N)__\n").encode())
        out = read_until_re(("__DONE_" + str(attempt) + r"_[0-9]{10,}__").encode(), limit)
        if out is not None:
            if attempt == 2:
                time.sleep(0.3)
                buf = b""      # a duplicate's tail must not become the next row's body
            break
        if attempt == 1:
            print("INFO %-13s no answer in %ss, sending it again" % (name, limit), flush=True)
    if out is None:
        print("FAIL %-13s no answer" % name, flush=True)
        continue
    body = re.sub(r"__DONE_[0-9]{10,}__", "", out)
    body = re.sub(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", "", body)   # OSC: bash 5.3 emits ]3008;…
    body = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", body)
    # ANY sentinel text that survived into the body is noise from a resend or a late duplicate: it made
    # `awk '{print $NF}'` pick a sentinel instead of the API key and turned U3 lan into a 401 (measured). Strip every
    # one of them before anything reads the body.
    body = re.sub(r"__DONE_[0-9]+_[0-9]{10,}__", "", body)
    # and keep only what came after the LAST echo of this command: on a resend the earlier attempt's output is
    # still in the stream, and reporting it as this row's answer is how a table starts lying.
    if cmd[:24] in body:
        body = body[body.rindex(cmd[:24]):]
    body = "\n".join(l.strip() for l in body.splitlines() if l.strip() and cmd[:24] not in l)
    flat = body.replace("\n", " | ")
    if want is None:
        print("INFO %-13s %s" % (name, flat), flush=True)
    elif want in body:
        print("PASS %-13s %s" % (name, flat), flush=True)
    else:
        print("FAIL %-13s %s" % (name, flat), flush=True)
PY
driver_rc=$?
cat "$work/rows.txt"

# Cycle 14, item 6: the console says what it is waiting for. The two lines patch-init.sh puts in the initrd
# print the guest's own /proc/uptime, so the transcript answers "how long is the copy to RAM" without a
# stopwatch - and on the board, at USB speed, it is the number the owner was staring at.
prog1=$(grep -ao '1bit: loading the system at [0-9.]*s' "$work/serial.log" 2>/dev/null | tail -1)
prog2=$(grep -ao '1bit: system in RAM after [0-9.]*s' "$work/serial.log" 2>/dev/null | tail -1)
if [ -n "$prog1" ]; then
    echo "PASS U8 boot progress $prog1 | ${prog2:-no second line}"
else
    echo "FAIL U8 boot progress the console never said what it was doing (no progress line in the transcript)"
fi

# U3 over the network, from outside the guest. Cycle 20: blank mode is the shipped default, so an empty key is not a
# reason to SKIP - the probe then carries no Authorization header, which is what a stock-Lemonade client does.
key=$(awk '/^INFO U3 keyvalue/ {print $NF}' "$work/rows.txt" 2>/dev/null | tail -1)
if [ -n "${key:-}" ]; then
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' -H "Authorization: Bearer $key" \
        "http://127.0.0.1:$hostport/api/v1/models" 2>/dev/null)
    mode=" with the image's key"
else
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' \
        "http://127.0.0.1:$hostport/api/v1/models" 2>/dev/null)
    mode=" with no key (blank mode)"
fi
if [ "$code" = "200" ]; then
    echo "PASS U3 lan        http=200 on the forwarded :13305 (host :$hostport)$mode"
elif ! vm_alive; then
    echo "FAIL U3 lan        http=${code:-none} - $(vm_gone)"
else
    echo "FAIL U3 lan        http=${code:-none} on the forwarded :13305 (host :$hostport)$mode"
fi
# U7, over the wire: SSHKEY=<private key> and the key half of which must be in the image's keys.
if [ -n "${SSHKEY:-}" ] && [ -f "$SSHKEY" ]; then
    # the remote command prints a marker, and THAT is what the row matches: filtering the verbose stream for
    # debug lines once threw the command's own output away and the row failed while the login had worked.
    # Cycle 15: the same login also has to land in BASH, not in the console menu. The menu is inittab's tty1
    # surface and nothing else runs it, so this is the row that says so - a menu that also caught ssh would
    # make every remote administrator open a console UI first (measured: `$0` is /bin/bash and BASH_VERSION set).
    said=$(timeout 40 ssh -v -p "$sshport" -i "$SSHKEY" -o BatchMode=yes -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null root@127.0.0.1 \
        'echo ONBOARD-SSH-OK; uname -r; command -v 1bit; echo ssh-shell=$0; \
         [ -n "$BASH_VERSION" ] && echo ssh-bash; 1bit-console-menu --status | head -1' 2>&1 | tr '\n' ' ')
    case $said in
    *ONBOARD-SSH-OK*ssh-bash*) echo "PASS U7 ssh        $(printf '%s' "$said" | tail -c 200)" ;;
    *ONBOARD-SSH-OK*) echo "FAIL U7 ssh        the login works but the shell is not bash: $(printf '%s' "$said" | tail -c 200)" ;;
    *) if vm_alive; then echo "FAIL U7 ssh        $(printf '%s' "$said" | tail -c 240)"
       else echo "FAIL U7 ssh        $(printf '%s' "$said" | tail -c 160) - $(vm_gone)"; fi ;;
    esac
else
    echo "SKIP U7 ssh        set SSHKEY=<private key> whose public half is in the image"
fi
vmqemu
echo "SKIP U4 board      HRX chat on the 8060S, the NPU through Lemonade, amdgpu/amdxdna binding: board rows"
echo "SKIP U6 board      an installer writing a firmware entry for a SECOND disk, and booting from it: needs the disk"
echo "== $iso, transcript $work/serial.log"
[ "$driver_rc" = "0" ] || exit 2
grep -q '^FAIL' "$work/rows.txt" && exit 1
exit 0
