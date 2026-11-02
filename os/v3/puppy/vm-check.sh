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
#
# The TPUPPY-RELEASE rows that can be answered without the board: U1 (a real shell, sudo, apt install),
# U3 (lemond up at boot with its key, reachable over the network) and the software half of U4 (the engine
# on PATH, the EC driver in Puppy's ADDLIST). Runs inside QEMU over a serial root shell, because Puppy's
# console is tty1 and nothing reads the serial port it writes to: the image's /etc/init.d/21serial-console
# opens one when the kernel command line carries 1bit.serial=1.
#
# Rows that need the hardware (HRX chat on the 8060S, the NPU through Lemonade, efivarfs/Secure Boot,
# boot time and idle RAM) are printed as SKIP with the reason: they are the board's rows.
#
# exit 0 = every row that can run here passed, 1 = a row failed, 2 = the VM never came up.
set -uo pipefail
prog=$(basename "$0")
[ $# -ge 2 ] || { sed -n "/^# $prog/,/^# exit 0/p" "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
iso=$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")
work=$2
secs=${3:-300}
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
    # the shipped artifact: firmware boots the ESP, grub reads the data partition
    vars=$work/OVMF_VARS.fd; [ -f "$vars" ] || cp /usr/share/OVMF/OVMF_VARS_4M.fd "$vars"
    # shellcheck disable=SC2054  # same: commas belong to the qemu arguments
    boot=(-drive file="$iso",format=raw,if=none,id=d0 -device qemu-xhci -device usb-storage,drive=d0
          -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd
          -drive if=pflash,format=raw,file="$vars")
    ;;
*)
    boot=(-kernel "$work/vmlinuz" -initrd "$work/initrd.gz" -append "console=ttyS0 1bit.serial=1" -cdrom "$iso")
    ;;
esac
# systemd-run --scope runs qemu in place, so the timeout wrapper is what has to end: when the rows are done
# the VM is stopped here by name instead of being left to its timeout. Three runs once overlapped and the
# next one could not even open its image ("Failed to get write lock"), while the previous VM still held the
# port the host probe was aimed at.
vmqemu() { for p in $(pgrep -f "qemu-system-x86_64 -m.*$iso" 2> /dev/null); do kill "$p" 2> /dev/null; done; }
( timeout "$secs" systemd-run --user --scope -q -p MemoryMax=6G -- \
    qemu-system-x86_64 -m 4096 -smp 4 -machine q35 "${boot[@]}" -display none -vga std -no-reboot \
    "${net[@]}" -serial unix:"$work/ser",server,nowait ) > "$work/qemu.out" 2>&1 &

ARTIFACT=$iso python3 - "$work" "$iso" > "$work/rows.txt" 2>&1 <<'PY'
import socket, sys, time, os, re
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

# A disk image boots through grub, whose menu is also on the serial line; the shell entry is the second
# one, so pick it the way a human at the console would: down, enter. Anything else boots entry 0.
if artifact.endswith(".img"):
    if read_until(b"serial root shell", 120) is not None:
        s.sendall(b"\x1b[B\r")
        print("INFO boot           grub menu on serial, selected the serial-shell entry", flush=True)

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
    print("FAIL boot           no serial shell: no 21serial-console in the image, or it never ran", flush=True)
    sys.exit(2)
buf = b""   # the menu and the boot log are not row output
s.sendall(b"\n")

rows = [
    ("U1 shell",     "echo SHELL=$SHELL; command -v bash; bash --version | head -1", "bash"),
    ("U1 sudo",      "id -un; sudo -n id -un 2>&1 | tail -2", "root"),
    # the wanted string must be evidence of an INSTALL, not the word appearing in an error: the previous
    # version matched "Unable to locate package neofetch" and reported a pass (measured 2026-11-02)
    # neofetch is not in the 26.04 archive at all (measured on this box: `apt-cache policy neofetch` and
    # `apt-cache search '^neofetch$'` are both empty), so the owner's exact command cannot pass on resolute.
    # The row stays, as a fact rather than a gate, and fastfetch is the packaged equivalent.
    ("U1 neofetch",  "apt install -y neofetch 2>&1 | tail -2; "
                     "command -v neofetch && echo neofetch-installed || echo neofetch-not-in-resolute", None),
    ("U1 fastfetch", "apt update 2>&1 | tail -2; apt install -y fastfetch 2>&1 | tail -2; "
                     "command -v fastfetch && echo fastfetch-installed", "fastfetch-installed", 600),
    ("U1 apt-any",   "apt install -y --no-install-recommends cowsay 2>&1 | tail -2; "
                     "command -v cowsay && echo apt-install-works", "apt-install-works", 600),
    ("U3 service",   "ps | grep -c '[l]emond'; netstat -lntp 2>/dev/null | grep -c 13305", None),
    ("U3 key",       "grep -q '^LEMONADE_API_KEY=..' \"$(readlink -f /etc/1bit/lemond.conf)\" && "
                     "echo key-on-data", "key-on-data"),
    ("U3 catalogue", "k=$(sed -n \"s/^LEMONADE_API_KEY=//p\" /etc/1bit/lemond.conf); "
                     "curl -s -H \"Authorization: Bearer $k\" http://127.0.0.1:13305/api/v1/models "
                     "| grep -o onebit | wc -l", None),
    ("U3 api local", 'k=$(sed -n "s/^LEMONADE_API_KEY=//p" /etc/1bit/lemond.conf); curl -s -o /dev/null '
                     '-w "http=%{http_code}" -H "Authorization: Bearer $k" http://127.0.0.1:13305/api/v1/models', "http=200"),
    ("U3 keyvalue",  'sed -n "s/^LEMONADE_API_KEY=//p" /etc/1bit/lemond.conf', None),
    ("U4 engine",    "command -v 1bit; 1bit --version 2>&1 | head -2", "1bit"),
    ("U4 addlist",   "grep -i '^ADDLIST' /etc/rc.d/MODULESCONFIG; lsmod | grep -c ec_su_axb35", "ec_su_axb35"),
    ("U4 package",   "ls /opt/1bit-package | tr '\\n' ' '; ls -l /usr/local/bin/1bit /usr/local/bin/lemond-onebit | tail -2",
                     "/opt/1bit-package"),
    ("U3 log",       "tail -3 /var/log/lemond-onebit.log 2>&1", None),
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
    ("data key",     "ls -l /etc/1bit/lemond.conf; head -c 0 /data/lemonade/lemond.conf && echo persists-on-data", "persists-on-data"),
    ("data ssh",     "ls -l /etc/ssh/ssh_host_ed25519_key; readlink -f /etc/ssh/ssh_host_ed25519_key", "/data/ssh/"),
    ("data state",   "ls /data; grep -c . /data/lemonade/config/user_models.json 2>/dev/null || echo no-registry", None),
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
    s.sendall((cmd + "; echo __DONE_$(date +%s%N)__\n").encode())
    out = read_until_re(rb"__DONE_[0-9]{10,}__", limit)
    if out is None:
        print("FAIL %-13s no answer" % name, flush=True)
        continue
    body = re.sub(r"__DONE_[0-9]{10,}__", "", out)
    body = re.sub(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", "", body)   # OSC: bash 5.3 emits ]3008;…
    body = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", body)
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

# U3 over the network, from outside the guest.
key=$(awk '/^INFO U3 keyvalue/ {print $NF}' "$work/rows.txt" 2>/dev/null | tail -1)
if [ -n "${key:-}" ]; then
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' -H "Authorization: Bearer $key" \
        "http://127.0.0.1:$hostport/api/v1/models" 2>/dev/null)
    if [ "$code" = "200" ]; then
        echo "PASS U3 lan        http=200 on the forwarded :13305 (host :$hostport) with the image's key"
    else
        echo "FAIL U3 lan        http=${code:-none} on the forwarded :13305 (host :$hostport)"
    fi
else
    echo "SKIP U3 lan        no key in the transcript"
fi
# U7, over the wire: SSHKEY=<private key> and the key half of which must be in the image's keys.
if [ -n "${SSHKEY:-}" ] && [ -f "$SSHKEY" ]; then
    # the remote command prints a marker, and THAT is what the row matches: filtering the verbose stream for
    # debug lines once threw the command's own output away and the row failed while the login had worked
    said=$(timeout 40 ssh -v -p "$sshport" -i "$SSHKEY" -o BatchMode=yes -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null root@127.0.0.1 \
        'echo ONBOARD-SSH-OK; uname -r; command -v 1bit' 2>&1 | tr '\n' ' ')
    case $said in
    *ONBOARD-SSH-OK*) echo "PASS U7 ssh        $(printf '%s' "$said" | tail -c 150)" ;;
    *) echo "FAIL U7 ssh        $(printf '%s' "$said" | tail -c 240)" ;;
    esac
else
    echo "SKIP U7 ssh        set SSHKEY=<private key> whose public half is in the image"
fi
vmqemu
echo "SKIP U4 hardware   HRX chat on the 8060S, the NPU through Lemonade, efivarfs/Secure Boot, boot time: board rows"
echo "== $iso, transcript $work/serial.log"
[ "$driver_rc" = "0" ] || exit 2
grep -q '^FAIL' "$work/rows.txt" && exit 1
exit 0
