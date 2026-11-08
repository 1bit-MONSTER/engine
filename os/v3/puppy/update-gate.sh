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
# RESET=1 puts the disk back to the post-install state first (no /1bit-alt, no release directories, the installed
# grub.cfg with default=0) so the gate can be re-run on an already-used target.img instead of re-installing:
#
#   RESET=1 update-gate.sh target.img work     # after a failed or a successful run
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
def detach(i):
    """stop tracking a loop device and free it now (the `finally` block detaches whatever is still in the list)"""
    dev = loops.pop(i)
    subprocess.run(["sudo", "-n", "losetup", "-d", dev], capture_output=True)

def mount_part(n, where, rw=True):
    start, size = parts[n]
    dev = subprocess.run(["sudo", "-n", "losetup", "-f", "--show", "--offset", str(start * 512),
                          "--sizelimit", str(size * 512), disk], capture_output=True, text=True).stdout.strip()
    if not dev:
        sys.exit("update-gate: losetup failed for partition %s" % n)
    loops.append(dev)
    sdo("mount", "-o", "rw" if rw else "ro", dev, where)

def esp_path():
    return mnt + "/esp"

try:
    # ---- RESET=1: back to the post-install state, so a re-run does not need a re-install
    if os.environ.get("RESET") == "1":
        os.makedirs(esp_path(), exist_ok=True)
        mount_part("2", mnt + "/data")
        for d in ("1bit-alt", "update-set-alt", "update-set-alt2"):
            sdo("rm", "-rf", os.path.join(mnt + "/data", d))
        subprocess.run(["sudo", "-n", "umount", mnt + "/data"], capture_output=True)
        while loops:
            detach(len(loops) - 1)
        loops_before = 0
        mount_part("1", esp_path())
        cfg = os.path.join(esp_path(), "EFI/BOOT/grub.cfg")
        text = open(cfg).read().splitlines()
        cut = next((i for i, l in enumerate(text) if l.startswith("menuentry ") and "1bit-alt" in l), None)
        if cut is not None:
            text = text[:cut]
            while text and not text[-1].strip():
                text.pop()
        text = [("set default=0" if l.startswith("set default=") else l) for l in text]
        # the ESP is vfat owned by root: write the computed file as root, not as the invoking user
        tmp = os.path.join(work, "grub.cfg.reset")
        open(tmp, "w").write("\n".join(text) + "\n")
        sdo("cp", "-f", tmp, cfg)
        say("INFO", "reset", "removed /1bit-alt and the release dirs; grub.cfg back to %d entries, default=0"
            % sum(1 for l in text if l.startswith("menuentry ")))
        subprocess.run(["sudo", "-n", "umount", esp_path()], capture_output=True)
        while len(loops) > loops_before:
            detach(len(loops) - 1)

    # ---- build TWO "next releases" ON the data partition, the way a release would arrive. Two, because the claim this
    # gate has to prove is that a machine can keep updating: the first update moves it to the other slot, and the second
    # one - from that other slot - updates the slot the layer lives in, which is where "exactly one update" used to die.
    mount_part("2", mnt + "/data")
    data = mnt + "/data"
    src = os.path.join(data, "1bit")
    global sfsname
    sfs = [f for f in sorted(os.listdir(src)) if f.endswith(".sfs") and f.startswith("puppy_")]
    if not sfs:
        sys.exit("update-gate: no puppy SFS in the installed /1bit")
    sfsname = sfs[0]
    base = os.path.join(work, "base-" + sfs[0])
    if not os.path.exists(base):
        sdo("cp", os.path.join(src, sfs[0]), base)
    setfiles = [f for f in sorted(os.listdir(src)) if f.endswith(".sfs") or f in ("vmlinuz", "initrd.gz")]

    def build_release(name, marker_text):
        """a release directory on /data: the set's own files, the patched puppy SFS, and a real SHA256SUMS. The
        marker file in the SFS is the only difference between the two releases."""
        marker = os.path.join(work, name + ".marker")
        open(marker, "w").write(marker_text)
        patched = os.path.join(work, name + ".sfs")
        if not os.path.exists(patched):
            # the MANIFEST form, with the mode stated: the single-file form refuses an ADDITION (its one-difference
            # check compares the raw `Only in <dir>: <name>` line against the target path, and the manifest branch is
            # the one with `matches_target` for that shape - measured here, reported to patch-sfs's owner).
            man = os.path.join(work, "manifest-" + name)
            open(man, "w").write("etc/1bit/%s\t%s\t0644\n" % (name, marker))
            sdo("bash", os.path.join(tree, "os/v3/puppy/patch-sfs.sh"), "--manifest", man, base, patched)
        up = os.path.join(data, "update-" + name)
        sdo("mkdir", "-p", up)
        for f in setfiles:
            # real copies, NOT hard links: a release directory whose files are the same inodes as the running slot's
            # would make the update's `cp -f src dst` fail ("are the same file") - a hard link is not what a release is
            sdo("cp", "-f", os.path.join(src, f), os.path.join(up, f))
        sdo("cp", "-f", patched, os.path.join(up, sfs[0]))
        sdo("bash", "-c", "cd '%s' && sha256sum * > SHA256SUMS" % up)
        say("INFO", "release", "%s: %d files + SHA256SUMS, puppy SFS carrying etc/1bit/%s" % (up, len(setfiles) + 1, name))
        return up

    src1 = build_release("set-alt", "this file exists only in the first updated appliance set\n")
    src2 = build_release("set-alt2", "this file exists only in the second updated appliance set\n")
    # a THIRD set, as one file on /data: the guest builds an .img around it for the image-form update (the #1385 review:
    # "add one gate row that updates from an .img the way a user would" - and the image form is where the hashes come
    # from the image's own SHA256SUMS, which is the fix that review asked for)
    marker3 = os.path.join(work, "set-alt3.marker")
    open(marker3, "w").write("this file exists only in the image-updated appliance set\n")
    patched3 = os.path.join(work, "set-alt3.sfs")
    if not os.path.exists(patched3):
        man3 = os.path.join(work, "manifest-set-alt3")
        open(man3, "w").write("etc/1bit/set-alt3\t%s\t0644\n" % marker3)
        sdo("bash", os.path.join(tree, "os/v3/puppy/patch-sfs.sh"), "--manifest", man3, base, patched3)
    sdo("cp", "-f", patched3, os.path.join(data, "set-alt3.sfs"))
    say("INFO", "release", "%s/set-alt3.sfs: the set the third (image-form) update will carry" % data)
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
        if not self.select(downs, 240):
            say("FAIL", tag, "no root shell (is entry %d the serial one?)" % downs)
            self.s = None      # INSIDE the failure branch: at function level it cleared a boot that had worked, and
                               # every later row saw "no VM" and the gate failed silently after its prep rows (measured)

    def select(self, downs, timeout):
        """walk `downs` entries at the menu and press Enter. The arrows CANCEL GRUB's countdown, so an Enter lost while
        the menu is still drawing leaves the machine sitting there for ever (measured: one boot of the second-update
        run) - hence a pause and a second Enter, and one more try if nothing boots."""
        for attempt in (1, 2):
            self.s.sendall(b"\x1b[B" * downs)
            time.sleep(0.5)
            self.s.sendall(b"\r")
            time.sleep(0.5)
            self.s.sendall(b"\r")
            for token in (b"21serial-console: root shell", b"root@1bit-os", b"bash-5.3#"):
                got = self.until(token, 60 if attempt == 1 else timeout)
                if got is not None:
                    self.buf = b""
                    self.s.sendall(b"\n")
                    time.sleep(1)
                    return True
            if attempt == 1:
                say("INFO", self.tag, "nothing booted after the first Enter; pressing it again at the menu")
        return False

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
        return self.select(downs, 300)

    def close(self):
        try: self.log.close()
        except Exception: pass
        try: self.s.close()
        except Exception: pass
        subprocess.run(["pkill", "-f", "qemu-system-x86_64.*" + disk],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try: self.proc.wait(timeout=10)
        except Exception: pass

# ---- boot 1 (/1bit): the installed machine updates itself into the other slot
v = Vm("b1-update", downs=3)
try:
    if v.s is None:
        ok = False
    else:
        pre = v.val("PRE", "sh -c 'echo 712-pre-update > /root/pre-update-marker; sync; cat /root/pre-update-marker'")
        if pre != "712-pre-update":
            say("FAIL", "pre-marker", "cannot write into the save layer (%s)" % (pre or "<nothing>")); ok = False
        st = v.val("STATUS", "1bit-update --status 2>&1 | tr '\\n' '; '")
        say("INFO", "status", st[:400])
        up = v.val("UPDATE", "1bit-update /data/update-set-alt 2>&1 | tr '\\n' '; '", timeout=2400)
        say("INFO", "update1", up[:1200])
        if "UPDATED" not in up:
            say("FAIL", "update1", "the first 1bit-update did not report UPDATED: %s" % (up[-300:] or "<no output>")); ok = False
        st2 = v.val("STATUS2", "1bit-update --status 2>&1 | tr '\\n' '; '")
        say("INFO", "status2", st2[:400])
        if "default index 4" not in st2:
            say("FAIL", "status2", "after the first update GRUB's default is not the updated slot's headless entry "
                                   "(want index 4): %s" % (st2[:160] or "<no answer>")); ok = False
        else:
            say("PASS", "status2", "GRUB default is index 4, the updated slot's headless entry")
        # the release just installed is not needed any more, and the disk is 8 GB: free it for the second one
        v.val("RM1", "rm -rf /data/update-set-alt; echo gone")

        # ---- boot 2 (/1bit-alt): the FIRST update's set, on the same layer - and then the SECOND update, which is the
        # one that used to be refused: from here the target slot holds the real save layer directory
        if not v.reboot(downs=5):
            say("FAIL", "boot-alt", "the machine did not come back on the updated slot"); ok = False
            v.close()
        else:
            sub = v.val("SUB", "tr ' ' '\\n' < /proc/cmdline | sed -n 's/^psubdir=//p'")
            pm = v.val("PM", "sed -n 's/^PUPMODE=//p' /etc/rc.d/PUPSTATE | head -1")
            mark = v.val("MARK", "cat /etc/1bit/set-alt 2>&1 | head -1")
            premark = v.val("PRE", "cat /root/pre-update-marker 2>&1 | head -1")
            rw = v.val("RW", "readlink /initrd/pup_rw")
            say("INFO", "boot-alt", "psubdir=%s PUPMODE=%s rw=%s" % (sub, pm, rw))
            say("INFO", "boot-alt", "first set's marker: %s | boot-1 marker: %s" % (mark[:60], premark))
            if sub != "1bit-alt":
                say("FAIL", "boot-alt", "booted psubdir=%s, not 1bit-alt" % (sub or "<none>")); ok = False
            elif pm != "12":
                say("FAIL", "boot-alt", "PUPMODE=%s: the shared save layer was not found through the slot" % (pm or "<unread>")); ok = False
            elif "only in the first updated" not in mark:
                say("FAIL", "boot-alt", "the first set's marker file is not there: this is not the updated set"); ok = False
            elif premark != "712-pre-update":
                say("FAIL", "boot-alt", "the pre-update marker is gone: the save layer forked"); ok = False
            else:
                say("PASS", "boot-alt", "psubdir=1bit-alt, PUPMODE=12, the first set's marker present, and the "
                                        "machine's own state carried over from /1bit")

            up2 = v.val("UPDATE2", "1bit-update /data/update-set-alt2 2>&1 | tr '\\n' '; '", timeout=2400)
            say("INFO", "update2", up2[:1200])
            if "UPDATED" not in up2:
                say("FAIL", "update2", "the SECOND 1bit-update was refused or failed: %s" % (up2[-300:] or "<no output>"))
                ok = False
            elif "left in place, no link needed" not in up2:
                say("FAIL", "update2", "the second update did not recognise the target slot's layer as the machine's own: %s"
                    % up2[-300:]); ok = False
            else:
                say("PASS", "update2", "the second update wrote /1bit around the machine's own layer "
                                       "(left in place, no link made over it)")
            st3 = v.val("STATUS3", "1bit-update --status 2>&1 | tr '\\n' '; '")
            say("INFO", "status3", st3[:400])
            if "default index 0" not in st3:
                say("FAIL", "status3", "after the second update GRUB's default is not the other slot (want index 0): %s"
                    % (st3[:160] or "<no answer>")); ok = False
            else:
                say("PASS", "status3", "GRUB default is index 0, the headless entry of the other slot")
            # ---- the IMAGE form: build a release image in the guest out of the second release's files around the third
            # SFS, so the last boot can update the way a user does - from an .img, with the hashes read out of the
            # image's own /1bit/SHA256SUMS (mkimage writes that now; without it this form needed --no-verify)
            mk = ("rm -rf /data/imgstage /data/update.img; mkdir -p /data/imgstage/1bit; "
                  "cp -f /data/update-set-alt2/vmlinuz /data/update-set-alt2/initrd.gz /data/imgstage/1bit/; "
                  "for f in /data/update-set-alt2/*.sfs; do b=$(basename \"$f\"); "
                  "[ \"$b\" = \"" + sfsname + "\" ] && continue; cp -f \"$f\" /data/imgstage/1bit/; done; "
                  "cp -f /data/set-alt3.sfs /data/imgstage/1bit/" + sfsname + "; "
                  "( cd /data/imgstage/1bit && sha256sum vmlinuz initrd.gz ./*.sfs > SHA256SUMS ); "
                  "truncate -s 1700M /data/update.img; "
                  "printf 'label: gpt\\nstart=2048, type=L, name=\"1BIT-DATA\"\\n' | sfdisk -q /data/update.img; "
                  "mkfs.ext4 -q -F -L 1BIT-DATA -d /data/imgstage /data/update.img 1664M; "
                  "rm -rf /data/imgstage; echo image-ok $(stat -c%s /data/update.img)")
            img = v.val("MKIMG", mk, timeout=1800)
            say("INFO", "mkimg", (img or "<no answer>")[-120:])
            if "image-ok" not in (img or ""):
                say("FAIL", "mkimg", "could not build the test image: %s" % ((img or "")[-200:] or "<no output>")); ok = False
            v.val("RM2", "rm -rf /data/update-set-alt2 /data/set-alt3.sfs; echo gone")

            # ---- boot 3 (/1bit again): the SECOND update's set, the boot-1 marker still there, and no link nested
            # inside the layer (the failure mode the fix exists for)
            if not v.reboot(downs=3):
                say("FAIL", "boot-2nd", "the machine did not come back on /1bit after the second update"); ok = False
                v.close()
            else:
                sub3 = v.val("SUB", "tr ' ' '\\n' < /proc/cmdline | sed -n 's/^psubdir=//p'")
                pm3 = v.val("PM", "sed -n 's/^PUPMODE=//p' /etc/rc.d/PUPSTATE | head -1")
                m2 = v.val("MARK2", "cat /etc/1bit/set-alt2 2>&1 | head -1")
                m1 = v.val("MARK1", "cat /etc/1bit/set-alt 2>&1 | head -1")
                pre3 = v.val("PRE", "cat /root/pre-update-marker 2>&1 | head -1")
                # /data/1bit, NOT /1bit: the frugal set lives on the data partition, which the running system sees at
                # /data (and at /initrd/mnt/dev_save); `/1bit` is grub's spelling of it, not a path in this filesystem
                # (measured: the first version of these two rows asked for /1bit and "passed" only because the negative
                # it looks for is also what a missing directory prints)
                nested = v.val("NESTED", "ls -d /data/1bit/upup64rsave/upup64rsave 2>&1 | head -1")
                lay = v.val("LAY", "sh -c 'test -d /data/1bit/upup64rsave && test ! -L /data/1bit/upup64rsave && echo real-dir || echo not-a-real-dir'")
                link = v.val("LINK", "sh -c '[ -L /data/1bit-alt/upup64rsave ] && [ -d /data/1bit-alt/upup64rsave/upper ] && echo link-ok || echo link-broken'")
                say("INFO", "boot-2nd", "psubdir=%s PUPMODE=%s | set-alt2: %s | set-alt: %s | layer: %s"
                    % (sub3, pm3, m2[:40], m1[:40], lay))
                say("INFO", "boot-2nd", "boot-1 marker: %s | nested link: %s | the other slot's link: %s"
                    % (pre3[:40], nested[:60], link))
                if sub3 != "1bit" or pm3 != "12":
                    say("FAIL", "boot-2nd", "after the second update the machine is psubdir=%s PUPMODE=%s"
                        % (sub3 or "<none>", pm3 or "<unread>")); ok = False
                elif "only in the second updated" not in m2:
                    say("FAIL", "boot-2nd", "the second set's marker is not there: /1bit does not hold the new set"); ok = False
                elif "No such file" not in m1 and "cannot access" not in m1:
                    say("FAIL", "boot-2nd", "/1bit still carries the first set's marker: the set was not replaced"); ok = False
                elif pre3 != "712-pre-update":
                    say("FAIL", "boot-2nd", "the boot-1 marker is gone after the second update"); ok = False
                elif lay != "real-dir":
                    say("FAIL", "boot-2nd", "the save layer in /1bit is not a real directory after the second update"); ok = False
                elif "upup64rsave/upup64rsave" in nested and "No such file" not in nested:
                    say("FAIL", "boot-2nd", "a link was made INSIDE the layer (upup64rsave/upup64rsave): %s" % nested); ok = False
                elif link != "link-ok":
                    say("FAIL", "boot-2nd", "the other slot's save-layer link does not resolve after the second update: %s" % link); ok = False
                else:
                    say("PASS", "boot-2nd", "second update booted: psubdir=1bit, PUPMODE=12, the new set in place, the "
                                            "boot-1 marker intact, the layer a real directory with no nested link")

                rb = v.val("ROLLBACK", "1bit-update --rollback 2>&1 | tr '\\n' '; '")
                say("INFO", "rollback", rb[:300])
                if "rolled back" not in rb:
                    say("FAIL", "rollback", "1bit-update --rollback did not report a rollback"); ok = False

                # ---- boot 4 (/1bit-alt): the FIRST update's set is still the rollback, untouched by the second update
                if not v.reboot(downs=5):
                    say("FAIL", "boot-old", "the machine did not come back on the other slot after the rollback"); ok = False
                    v.close()
                else:
                    sub4 = v.val("SUB", "tr ' ' '\\n' < /proc/cmdline | sed -n 's/^psubdir=//p'")
                    pm4 = v.val("PM", "sed -n 's/^PUPMODE=//p' /etc/rc.d/PUPSTATE | head -1")
                    m1b = v.val("MARK1", "cat /etc/1bit/set-alt 2>&1 | head -1")
                    m2b = v.val("MARK2", "cat /etc/1bit/set-alt2 2>&1 | head -1")
                    pre4 = v.val("PRE", "cat /root/pre-update-marker 2>&1 | head -1")
                    say("INFO", "boot-old", "psubdir=%s PUPMODE=%s | set-alt: %s | set-alt2: %s | marker: %s"
                        % (sub4, pm4, m1b[:40], m2b[:40], pre4[:40]))
                    if sub4 != "1bit-alt" or pm4 != "12":
                        say("FAIL", "boot-old", "after the rollback the machine is psubdir=%s PUPMODE=%s"
                            % (sub4 or "<none>", pm4 or "<unread>")); ok = False
                    elif "only in the first updated" not in m1b:
                        say("FAIL", "boot-old", "the rollback slot does not hold the first updated set"); ok = False
                    elif "No such file" not in m2b and "cannot access" not in m2b:
                        say("FAIL", "boot-old", "the second set's marker is visible in the rollback slot"); ok = False
                    elif pre4 != "712-pre-update":
                        say("FAIL", "boot-old", "the boot-1 marker is gone after the rollback"); ok = False
                    else:
                        say("PASS", "boot-old", "rolled back to the first updated set on psubdir=1bit-alt, PUPMODE=12, "
                                                "the machine's state unchanged - both sets still on the disk")

                    # ---- image form: the same command a user runs, against an .img, with the hashes taken from the
                    # image's own /1bit/SHA256SUMS (mkimage writes it now; before that every image needed --no-verify)
                    upi = v.val("UPDATE3", "1bit-update /data/update.img 2>&1 | tr '\\n' '; '", timeout=2400)
                    say("INFO", "update3", upi[:900])
                    if "UPDATED" not in upi:
                        say("FAIL", "update3", "the image-form update failed: %s" % (upi[-300:] or "<no output>")); ok = False
                    elif "1bit/SHA256SUMS" not in upi:
                        say("FAIL", "update3", "the image-form update did not verify against the image's own "
                              "SHA256SUMS: %s" % upi[-400:]); ok = False
                    else:
                        say("PASS", "update3", "an .img updated the machine, with the set's bytes verified against the "
                                               "image's own SHA256SUMS")
                    if not v.reboot(downs=3):
                        say("FAIL", "boot-img", "the machine did not come back after the image-form update"); ok = False
                        v.close()
                    else:
                        # ---- boot 5: the set the image carried
                        sub5 = v.val("SUB", "tr ' ' '\\n' < /proc/cmdline | sed -n 's/^psubdir=//p'")
                        pm5 = v.val("PM", "sed -n 's/^PUPMODE=//p' /etc/rc.d/PUPSTATE | head -1")
                        m3 = v.val("MARK3", "cat /etc/1bit/set-alt3 2>&1 | head -1")
                        pre5 = v.val("PRE", "cat /root/pre-update-marker 2>&1 | head -1")
                        say("INFO", "boot-img", "psubdir=%s PUPMODE=%s | set-alt3: %s | marker: %s"
                            % (sub5, pm5, m3[:44], pre5[:40]))
                        if sub5 != "1bit" or pm5 != "12":
                            say("FAIL", "boot-img", "after the image-form update the machine is psubdir=%s PUPMODE=%s"
                                % (sub5 or "<none>", pm5 or "<unread>")); ok = False
                        elif "only in the image-updated" not in m3:
                            say("FAIL", "boot-img", "the image's set is not running (set-alt3: %s)" % (m3[:60] or "<none>")); ok = False
                        elif pre5 != "712-pre-update":
                            say("FAIL", "boot-img", "the boot-1 marker is gone after the image-form update"); ok = False
                        else:
                            say("PASS", "boot-img", "the image's set booted: psubdir=1bit, PUPMODE=12, its marker "
                                                    "present, the machine's state intact")
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
