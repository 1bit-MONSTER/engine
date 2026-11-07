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
# update-gate.sh <installed-disk.img> <work-dir> [secs]
#
# engine#712's update path, measured on an ALREADY INSTALLED disk (the `target.img` that `install-probe.sh` leaves
# behind), over three boots of that disk:
#
#   boot 1  the installed machine: write a marker into the save layer, `1bit-update --status`, and
#           `1bit-update /data/update-src` - the next appliance set, which the gate put on the data partition the
#           way a release would arrive (a directory with the six files and a SHA256SUMS)
#   boot 2  the machine must come up on the OTHER slot: psubdir=1bit-alt, PUPMODE=12 (the save layer is found
#           through the slot's relative symlink), the new set's marker file present, and the marker written on
#           boot 1 still there - same machine, new OS. Then `1bit-update --rollback`.
#   boot 3  back on /1bit: psubdir=1bit, the new set's marker GONE (the previous appliance is intact), and the
#           boot-1 marker still there.
#
# The "next release" is built here, not shipped: the installed disk's own puppy SFS with ONE added file
# (etc/1bit/set-alt), so "which set am I running" is a fact read off the running system rather than a date or a size.
#
# Exit: 0 the update and the rollback both measured, 1 a check failed, 2 the harness could not run.
set -uo pipefail
prog=$(basename "$0")
[ $# -ge 2 ] || { sed -n "/^# $prog /,/^# Exit:/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
disk=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
work=$2
secs=${3:-600}
[ -f "$disk" ] || { echo "$prog: no such disk image: $disk" >&2; exit 2; }
for t in qemu-system-x86_64 python3 sudo; do
    command -v "$t" > /dev/null || { echo "$prog: $t is missing" >&2; exit 2; }
done
mkdir -p "$work" || exit 2
work=$(cd "$work" && pwd)
sudo -n true 2> /dev/null || { echo "$prog: needs passwordless sudo for the host-side mounts" >&2; exit 2; }
tree=$(cd "$(dirname "$0")/../../.." && pwd)

out=$(python3 - "$disk" "$work" "$secs" "$tree" <<'PY'
import os, re, socket, subprocess, sys, time
disk, work, secs, tree = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
rows = []
ok = True

def say(kind, name, text):
    rows.append((kind, name, text))
    print("%-4s %-12s %s" % (kind, name, text[:400]), flush=True)

def sdo(*a, **kw):
    r = subprocess.run(["sudo", "-n"] + list(a), capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit("%s: %s failed: %s%s" % (os.path.basename(sys.argv[0]), " ".join(a), r.stdout, r.stderr))
    return r.stdout

# ---- partitions of the installed disk (the same shape 1bit-install writes: ESP 1, 1BIT-DATA 2)
parts = {}
for line in sdo("sfdisk", "-d", disk).splitlines():
    m = re.match(r"^(\S+)\s*:\s*start=\s*(\d+),\s*size=\s*(\d+)", line)
    if m and os.path.basename(m.group(1)).endswith(("1", "2")):
        parts[os.path.basename(m.group(1))[-1]] = (int(m.group(2)), int(m.group(3)))
if set(parts) != {"1", "2"}:
    sys.exit("update-gate: cannot read the disk image's two partitions")
mnt = os.path.join(work, "mnt")
os.makedirs(mnt + "/data", exist_ok=True)
loops = []
def mount_part(n, where, rw=True):
    start, size = parts[n]
    dev = subprocess.run(["sudo", "-n", "losetup", "-f", "--show", "--offset", str(start * 512),
                          "--sizelimit", str(size * 512), disk], capture_output=True, text=True).stdout.strip()
    if not dev:
        sys.exit("update-gate: losetup failed for partition %s" % n)
    loops.append(dev)
    sdo("mount", "-o", "rw" if rw else "ro", dev, where)

try:
    # ---- build the "next release" ON the data partition, the way a release would arrive
    mount_part("2", mnt + "/data")
    data = mnt + "/data"
    src = os.path.join(data, "1bit")
    sfs = [f for f in sorted(os.listdir(src)) if f.endswith(".sfs") and f.startswith("puppy_")]
    if not sfs:
        sys.exit("update-gate: no puppy SFS in the installed /1bit")
    base = os.path.join(work, "base-" + sfs[0])
    if not os.path.exists(base):
        sdo("cp", os.path.join(src, sfs[0]), base)
    marker = os.path.join(work, "set-alt-marker")
    open(marker, "w").write("this file exists only in the updated appliance set\n")
    patched = os.path.join(work, sfs[0])
    if not os.path.exists(patched):
        # the MANIFEST form, with the mode stated: the single-file form refuses an ADDITION (its one-difference check
        # compares the raw `Only in <dir>: <name>` line against the target path, and the manifest branch is the one
        # with `matches_target` for that shape - measured here, and reported to patch-sfs's owner rather than patched
        # in passing). 0644 is also explicit, because a copied host file arrives group-writable.
        man = os.path.join(work, "manifest")
        open(man, "w").write("etc/1bit/set-alt\t%s\t0644\n" % marker)
        sdo("bash", os.path.join(tree, "os/v3/puppy/patch-sfs.sh"),
            "--manifest", man, base, patched)
    up = os.path.join(data, "update-src")
    sdo("mkdir", "-p", up)
    for f in [f for f in sorted(os.listdir(src)) if f.endswith(".sfs") or f in ("vmlinuz", "initrd.gz")]:
        sdo("cp", "-f", os.path.join(src, f), os.path.join(up, f))
    sdo("cp", "-f", patched, os.path.join(up, sfs[0]))
    sdo("bash", "-c", "cd '%s' && sha256sum * > SHA256SUMS" % up)
    say("INFO", "update-src", "%s: %d files + SHA256SUMS, puppy SFS patched with etc/1bit/set-alt"
        % (up, len(os.listdir(up)) - 1))
finally:
    subprocess.run(["sudo", "-n", "umount", mnt + "/data"], capture_output=True)
    for dev in loops:
        subprocess.run(["sudo", "-n", "losetup", "-d", dev], capture_output=True)

# ---- one boot of the installed disk, driven over its serial console
class Vm:
    def __init__(self, tag, downs):
        self.tag, self.s, self.buf, self.text = tag, None, b"", ""
        self.path = os.path.join(work, "ser-" + tag)
        for f in (self.path, self.path + ".log"):
            try: os.unlink(f)
            except OSError: pass
        varsf = os.path.join(work, "VARS.fd")
        src = "/usr/share/OVMF/OVMF_VARS_4M.fd"
        if not os.path.exists(varsf):
            open(varsf, "wb").write(open(src, "rb").read())
        q = ["qemu-system-x86_64", "-m", "4096", "-smp", "4",
             "-drive", "file=" + disk + ",format=raw,if=none,id=d0",
             "-device", "qemu-xhci", "-device", "usb-storage,drive=d0",
             "-drive", "if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd",
             "-drive", "if=pflash,format=raw,file=" + varsf,
             "-display", "none", "-vga", "std", "-serial", "unix:" + self.path + ",server,nowait"]
        self.proc = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=6G", "--"] + q,
                                     stdout=open(os.path.join(work, "qemu-%s.out" % tag), "wb"),
                                     stderr=subprocess.STDOUT)
        t = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        end = time.time() + 120
        while True:
            try:
                t.connect(self.path); break
            except OSError:
                if time.time() > end:
                    say("FAIL", tag, "the VM never opened its serial socket"); return
                time.sleep(0.2)
        t.settimeout(0.5)
        self.s, self.buf = t, b""
        self.log = open(self.path + ".log", "ab", buffering=0)
        if self.until(b"Press enter to boot the selected OS", 240) is None:
            say("FAIL", tag, "no GRUB menu on the installed disk"); self.s = None; return
        time.sleep(1.0)
        self.s.sendall(b"\x1b[B" * downs + b"\r")
        for token in (b"21serial-console: root shell", b"root@1bit-os", b"bash-5.3#"):
            if self.until(token, 240) is not None:
                self.buf = b""
                self.s.sendall(b"\n")
                time.sleep(1)
                return
        say("FAIL", tag, "no root shell (is entry %d the serial one?)" % downs)
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

    def val(self, token, cmdline, timeout=1800):
        # concatenation, NOT "%s" formatting: the sentinel itself contains `%s%N` (`date +%s%N`), which a format
        # string reads as two more placeholders - measured here as "TypeError: not enough arguments for format string"
        self.s.sendall(("echo __S__; echo " + token + "=$(" + cmdline + "); echo __E$?__$(date +%s%N)\n").encode())
        end = time.time() + timeout
        while time.time() < end:
            m = re.search(rb"__E[0-9]+__[0-9]{10,}", self.buf)
            if m:
                raw, self.buf = self.buf[:m.start()], self.buf[m.end():]
                text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07|\x1b\[[0-9;=?]*[A-Za-z]", "",
                              raw.decode("utf-8", "replace"))
                i = text.rfind("__S__")
                if i >= 0:
                    text = text[i + 5:]
                f = re.findall(r"\b%s=(.*)" % token, text)
                return f[-1].strip() if f else ""
            self.s.settimeout(0.5)
            try:
                d = self.s.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return ""
            if not d:
                return ""
            self.log.write(d); self.buf += d; self.text += d.decode("utf-8", "replace")
        return ""

    def reboot(self, downs):
        self.s.sendall(b"sync; reboot -f\n")
        if self.until(b"Press enter to boot the selected OS", 300) is None:
            return False
        time.sleep(1.0)
        self.s.sendall(b"\x1b[B" * downs + b"\r")
        for token in (b"21serial-console: root shell", b"root@1bit-os", b"bash-5.3#"):
            if self.until(token, 300) is not None:
                self.buf = b""
                self.s.sendall(b"\n")
                time.sleep(1)
                return True
        return False

    def close(self):
        try: self.log.close()
        except Exception: pass
        try: self.s.close()
        except Exception: pass
        subprocess.run(["pkill", "-f", "qemu-system-x86_64.*" + disk],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: self.proc.wait(timeout=10)
        except Exception: pass

# ---- boot 1: the installed machine updates itself
v = Vm("update", downs=3)
try:
    if v.s is None:
        ok = False
    else:
        pre = v.val("PRE", "sh -c 'echo 712-pre-update > /root/pre-update-marker; sync; cat /root/pre-update-marker'")
        if pre != "712-pre-update":
            say("FAIL", "pre-marker", "cannot write into the save layer (%s)" % (pre or "<nothing>")); ok = False
        st = v.val("STATUS", "1bit-update --status 2>&1 | tr '\\n' '; '")
        say("INFO", "status", st[:400])
        up = v.val("UPDATE", "1bit-update /data/update-src 2>&1 | tr '\\n' '; '", timeout=2400)
        say("INFO", "update", up[:1200])
        if "UPDATED" not in up:
            # the last part is where a failure's cause is: the script logs and prints its own reason
            say("FAIL", "update", "1bit-update did not report UPDATED: %s" % (up[-300:] or "<no output>")); ok = False
        st2 = v.val("STATUS2", "1bit-update --status 2>&1 | tr '\\n' '; '")
        say("INFO", "status2", st2[:400])
        if "default index 4" not in st2:
            say("FAIL", "status2", "after the update GRUB's default is not the updated slot's headless entry "
                                   "(want index 4): %s" % (st2[:160] or "<no answer>")); ok = False
        else:
            say("PASS", "status2", "GRUB default is index 4, the updated slot's headless entry")
        if not v.reboot(downs=5):
            say("FAIL", "boot-alt", "the machine did not come back on the updated slot"); ok = False
            v.close()
        else:
            # ---- boot 2: it must be the OTHER set, on the SAME save layer
            sub = v.val("SUB", "tr ' ' '\\n' < /proc/cmdline | sed -n 's/^psubdir=//p'")
            pm = v.val("PM", "sed -n 's/^PUPMODE=//p' /etc/rc.d/PUPSTATE | head -1")
            mark = v.val("MARK", "cat /etc/1bit/set-alt 2>&1 | head -1")
            premark = v.val("PRE", "cat /root/pre-update-marker 2>&1 | head -1")
            rw = v.val("RW", "readlink /initrd/pup_rw")
            say("INFO", "boot-alt", "psubdir=%s PUPMODE=%s rw=%s" % (sub, pm, rw))
            say("INFO", "boot-alt", "new set's marker: %s | previous marker: %s" % (mark[:60], premark))
            if sub != "1bit-alt":
                say("FAIL", "boot-alt", "booted psubdir=%s, not 1bit-alt" % (sub or "<none>")); ok = False
            elif pm != "12":
                say("FAIL", "boot-alt", "PUPMODE=%s: the shared save layer was not found through the slot" % (pm or "<unread>")); ok = False
            elif "only in the updated appliance" not in mark:
                say("FAIL", "boot-alt", "the new set's marker file is not there: this is not the updated set"); ok = False
            elif premark != "712-pre-update":
                say("FAIL", "boot-alt", "the pre-update marker is gone: the save layer forked"); ok = False
            else:
                say("PASS", "boot-alt", "psubdir=1bit-alt, PUPMODE=12, the new set's marker present, and the "
                                        "machine's own state carried over from /1bit")
            rb = v.val("ROLLBACK", "1bit-update --rollback 2>&1 | tr '\\n' '; '")
            say("INFO", "rollback", rb[:300])
            if "rolled back" not in rb:
                say("FAIL", "rollback", "1bit-update --rollback did not report a rollback"); ok = False
            if not v.reboot(downs=3):
                say("FAIL", "boot-old", "the machine did not come back on /1bit after the rollback"); ok = False
                v.close()
            else:
                # ---- boot 3: the previous appliance, intact, with the state the update carried
                sub3 = v.val("SUB", "tr ' ' '\\n' < /proc/cmdline | sed -n 's/^psubdir=//p'")
                pm3 = v.val("PM", "sed -n 's/^PUPMODE=//p' /etc/rc.d/PUPSTATE | head -1")
                mark3 = v.val("MARK", "ls /etc/1bit/set-alt 2>&1 | head -1")
                pre3 = v.val("PRE", "cat /root/pre-update-marker 2>&1 | head -1")
                say("INFO", "boot-old", "psubdir=%s PUPMODE=%s | set-alt: %s | marker: %s"
                    % (sub3, pm3, mark3[:40], pre3[:40]))
                if sub3 != "1bit" or pm3 != "12":
                    say("FAIL", "boot-old", "after the rollback the machine is psubdir=%s PUPMODE=%s"
                        % (sub3 or "<none>", pm3 or "<unread>")); ok = False
                elif "No such file" not in mark3 and "cannot access" not in mark3:
                    say("FAIL", "boot-old", "the updated set's marker is visible while running /1bit"); ok = False
                elif pre3 != "712-pre-update":
                    say("FAIL", "boot-old", "the marker written before the update is gone after the rollback"); ok = False
                else:
                    say("PASS", "boot-old", "rolled back to /1bit with PUPMODE=12, the previous appliance intact "
                                            "and the machine's state unchanged")
finally:
    v.close()

print("== %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
)
rc=$?
echo "$out"
echo "== transcript $work/ser-update.log"
[ "$rc" = 0 ] && exit 0
[ "$rc" = 2 ] && exit 2
exit 1
