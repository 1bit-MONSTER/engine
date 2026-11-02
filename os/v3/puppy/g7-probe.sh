#!/usr/bin/env bash
# g7-probe.sh <img> <work> - G7: does a change made in the running appliance survive a reboot?
# Boots the image, writes a marker into /etc (the save layer's job, not /data's), powers off from the serial
# shell, then boots again and looks for the marker. A save FOLDER is pre-created on the data partition first,
# because Puppy asks at shutdown otherwise and a dialog is not something a probe can answer.
set -uo pipefail
img=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); work=$2
mkdir -p "$work"; work=$(cd "$work" && pwd)
rm -f "$work/ser" "$work/serial.log" "$work/qemu.out" "$work/boot2.log"
vars=$work/OVMF_VARS.fd; [ -f "$vars" ] || cp /usr/share/OVMF/OVMF_VARS_4M.fd "$vars"
hostport=21605
while ss -lnt 2>/dev/null | grep -q ":$hostport "; do hostport=$((hostport+2)); done
vmqemu() { for p in $(pgrep -f "qemu-system-x86_64 -m.*$img" 2>/dev/null); do kill "$p" 2>/dev/null; done; }

# 1. a save folder on the data partition, so the first boot finds one instead of asking
L=$(sudo -n losetup -f --show -P "$img")
sudo -n mkdir -p /tmp/g7; sudo -n mount "${L}p2" /tmp/g7
sudo -n mkdir -p /tmp/g7/1bit/upup64rsave-g7
sudo -n umount /tmp/g7; sudo -n losetup -d "$L"
echo "== save folder pre-created: /1bit/upup64rsave-g7"

boot() {   # <tag> <python commands...> - boots, drives the shell, prints what came back
    local tag=$1; shift
    ( timeout 300 systemd-run --user --scope -q -p MemoryMax=6G -- \
        qemu-system-x86_64 -m 4096 -smp 4 -machine q35 \
        -drive file="$img",format=raw,if=none,id=d0 -device qemu-xhci -device usb-storage,drive=d0 \
        -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
        -drive if=pflash,format=raw,file="$vars" \
        -display none -vga std -no-reboot \
        -netdev user,id=n0,hostfwd=tcp:127.0.0.1:"$hostport"-:13305 -device virtio-net-pci,netdev=n0 \
        -serial unix:"$work/ser",server,nowait ) > "$work/qemu-$tag.out" 2>&1 &
    python3 - "$work" "$tag" "$@" <<'PY'
import socket, sys, time, os, re
work, tag = sys.argv[1], sys.argv[2]
cmds = sys.argv[3:]
log = open(os.path.join(work, "serial.log"), "ab", buffering=0)
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
stop = time.time() + 60
while True:
    try:
        s.connect(os.path.join(work, "ser")); break
    except OSError:
        if time.time() > stop: sys.exit(2)
        time.sleep(0.2)
buf = b""
def wait_for(pat, limit, nudge=False):
    global buf
    end = time.time() + limit; last = 0
    while time.time() < end:
        m = re.search(pat, buf)
        if m:
            out, buf = buf[:m.end()], buf[m.end():]
            return out.decode("utf-8", "replace")
        if nudge and time.time() - last > 10:
            s.sendall(b"\n"); last = time.time()
        s.settimeout(0.5)
        try: d = s.recv(65536)
        except socket.timeout: continue
        except OSError: return None
        if not d: return None
        log.write(d); buf += d
    return None
if wait_for(rb"serial root shell", 120) is not None:
    s.sendall(b"\x1b[B\r")
if wait_for(rb"root@[^\r\n]{0,80}[$#] ", 240, nudge=True) is None:
    print("FAIL %s  no shell" % tag, flush=True); sys.exit(2)
buf = b""
for cmd in cmds:
    s.sendall((cmd + "; echo __D__$(date +%s%N)\n").encode())
    out = wait_for(rb"__D__[0-9]{10,}", 180)
    body = re.sub(r"__D__[0-9]{10,}", "", out or "")
    body = re.sub(r"\x1b\][^\x07\x1b]*(\x07|\x1b\\)", "", body)
    body = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", body)
    body = "\n".join(l.strip() for l in body.splitlines() if l.strip() and cmd[:20] not in l)
    print("[%s] %s" % (tag, body.replace("\n", " | ")), flush=True)
PY
    sleep 3; vmqemu
}

echo "== boot 1: write a marker into /etc (the save layer), then power off"
boot b1 "grep -m1 PUPMODE /etc/rc.d/PUPSTATE; echo marker-$(date +%s) > /etc/g7-marker; cat /etc/g7-marker" \
        "sync; poweroff" 
sleep 5

echo "== boot 2: is it still there?"
boot b2 "grep -m1 PUPMODE /etc/rc.d/PUPSTATE; cat /etc/g7-marker 2>&1 || echo marker-GONE"
echo "== transcripts: $work/serial.log"
