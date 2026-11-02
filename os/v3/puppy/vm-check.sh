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
# shellcheck disable=SC2054  # qemu arguments carry commas inside a single word
net=(-netdev user,id=n0,hostfwd=tcp:127.0.0.1:"$hostport"-:13305 -device virtio-net-pci,netdev=n0)
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

rows = [
    ("U1 shell",     "echo SHELL=$SHELL; command -v bash; bash --version | head -1", "bash"),
    ("U1 sudo",      "id -un; sudo -n id -un 2>&1 | tail -2", "root"),
    ("U1 apt",       "apt update >/dev/null 2>&1; apt install -y neofetch >/dev/null 2>&1; command -v neofetch", "neofetch"),
    ("U3 service",   "ps | grep -c '[l]emond'; netstat -lntp 2>/dev/null | grep -c 13305", None),
    ("U3 key",       "ls -l /etc/1bit/lemond.conf; grep -c '^LEMONADE_API_KEY=' /etc/1bit/lemond.conf", "LEMONADE_API_KEY"),
    ("U3 api local", 'k=$(sed -n "s/^LEMONADE_API_KEY=//p" /etc/1bit/lemond.conf); curl -s -o /dev/null '
                     '-w "http=%{http_code}" -H "Authorization: Bearer $k" http://127.0.0.1:13305/api/v1/models', "http=200"),
    ("U3 keyvalue",  'sed -n "s/^LEMONADE_API_KEY=//p" /etc/1bit/lemond.conf', None),
    ("U4 engine",    "command -v 1bit; 1bit --version 2>&1 | head -2", "1bit"),
    ("U4 addlist",   "grep -i '^ADDLIST' /etc/rc.d/MODULESCONFIG; lsmod | grep -c ec_su_axb35", "ec_su_axb35"),
    ("U4 package",   "ls /opt/1bit-package | tr '\\n' ' '; ls -l /usr/local/bin/1bit /usr/local/bin/lemond-onebit | tail -2",
                     "/opt/1bit-package"),
    ("U3 log",       "tail -3 /var/log/lemond-onebit.log 2>&1", None),
    ("data mount",   "grep -c ' /data ' /proc/mounts; df -h /data 2>/dev/null | tail -1", "/data"),
    ("data key",     "ls -l /etc/1bit/lemond.conf; head -c 0 /data/lemonade/lemond.conf && echo persists-on-data", "persists-on-data"),
    ("data ssh",     "ls -l /etc/ssh/ssh_host_ed25519_key; readlink -f /etc/ssh/ssh_host_ed25519_key", "/data/ssh/"),
    ("data state",   "ls /data; grep -c . /data/lemonade/config/user_models.json 2>/dev/null || echo no-registry", None),
    # U3 says "reachable on the LAN", so whether the appliance has an address at all is a row, not a
    # footnote: the cycle-2 image booted with no /data and answered nothing on the forwarded port, and
    # these two lines say whether that was the listener or the interface.
    ("net addr",     "ip -4 -o addr show 2>/dev/null | tr -s ' '", None),
    ("net route",    "ip -4 route 2>/dev/null | head -3", None),
]
for idx, (name, cmd, want) in enumerate(rows):
    sentinel = "__DONE%d__" % idx
    s.sendall((cmd + "; echo " + sentinel + "\n").encode())
    out = read_until(sentinel.encode(), 180)
    if out is None:
        print("FAIL %-13s no answer" % name, flush=True)
        continue
    body = out.split(sentinel)[0]
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
echo "SKIP U4 hardware   HRX chat on the 8060S, the NPU through Lemonade, efivarfs/Secure Boot, boot time: board rows"
echo "== $iso, transcript $work/serial.log"
[ "$driver_rc" = "0" ] || exit 2
grep -q '^FAIL' "$work/rows.txt" && exit 1
exit 0
