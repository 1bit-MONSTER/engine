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
# vmtest-v2.sh <rootfs-dir> <data-seed-dir> [ssh-key]
#
# Boots the v2 root filesystem as a machine - a virtio disk with the rootfs (LABEL 1BIT-ROOT) and a
# second one with the data partition (LABEL 1BIT-DATA) - and asks the appliance itself, rather than the
# host, because the appliance is what has to work:
#
#   * a test-only oneshot unit runs the checks inside the guest (packages, banner, /data, the two
#     engine commands, Lemonade with its key and a chat, first-boot state, apt from the repository,
#     sshd listening), writes the transcript to /data/vmtest.txt and powers the machine off;
#   * the host reads that transcript out of the data image afterwards, plus the console log for the
#     banner. No host-to-guest port forwarding is involved in the verdict: QEMU's user networking
#     forwards ports that answer or hang, and a test that depends on it fails for the wrong reason.
#
# Prints VMTEST-V2 PASS or FAIL.
set -uo pipefail
[ $# -ge 2 ] || { sed -n '21,34p' "$0"; exit 2; }
rootfs=$(cd "$1" && pwd) seed=$(cd "$2" && pwd) key=${3:-}
# an mmdebstrap rootfs is root-owned with root-only files: populating a filesystem image from it and
# handing it to qemu needs root, so the whole test runs as root rather than half of it
if [ "$(id -u)" != 0 ]; then exec sudo -n "$0" "$@"; fi
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
kern=$(ls "$rootfs"/boot/vmlinuz-* 2>/dev/null | head -1)
initrd=$(ls "$rootfs"/boot/initrd.img-* 2>/dev/null | head -1)
[ -n "$kern" ] && [ -n "$initrd" ] || { echo "VMTEST-V2 FAIL (no kernel/initrd in $rootfs/boot)"; exit 1; }
echo "kernel: $(basename "$kern"), initrd $(du -h "$initrd" | cut -f1), rootfs $(du -sh "$rootfs" | cut -f1)"

# --- the checks, run inside the appliance ---------------------------------------------------------
install -d -m 755 "$rootfs/usr/lib/1bit"
cat > "$rootfs/usr/lib/1bit/vmtest.sh" <<'INGUEST'
#!/bin/sh
# Test-only: what the appliance says about itself, one section at a time, into /data/vmtest.txt.
exec > /data/vmtest.txt 2>&1
echo "== packages"
dpkg-query -W -f='${Package} ${Version}\n' '1bit-*' 2>/dev/null
echo "== banner"
cat /etc/issue
echo "== data partition"
findmnt -n -o SOURCE,FSTYPE,LABEL /data 2>/dev/null
echo "== engine"
/usr/bin/1bit --help >/dev/null 2>&1 && echo "1bit --help: ok" || echo "1bit --help: FAILED"
# llama-server's HIP runtime is the machine's (the ROCm userspace), so in a VM it cannot load - but its
# OWN libraries must resolve from the package beside it ($ORIGIN), and if they do not, that is the
# package's fault, not the VM's
miss=$(ldd /opt/1bit/hrx/llama/bin/llama-server 2>/dev/null | awk '/not found/ {print $1}' | tr '\n' ' ')
echo "llama-server missing: ${miss:-none}"
echo "== kernel package"
ki=$(dpkg-query -W -f='${Package}\n' '1bit-kernel-image-*' 2>/dev/null | head -1)
k=${ki#1bit-kernel-image-}
[ -n "$k" ] && [ -e "/boot/vmlinuz-$k" ] && echo "kernel image: /boot/vmlinuz-$k present" || echo "kernel image: MISSING"
[ -n "$k" ] && [ -e "/boot/initrd.img-$k" ] && echo "initramfs: /boot/initrd.img-$k present" || echo "initramfs: MISSING"
[ -n "$k" ] && ls "/lib/modules/$k/updates/" 2>/dev/null | head -3
echo "== selftest"
1bit-selftest 2>&1 | grep -E "mode      :|RESULT" | head -4
echo "== release-check"
1bit-release-check 2>&1 | grep -E "^(PASS|FAIL|SKIP|RESULT)"
echo "== first boot"
[ -e /etc/1bit/installed ] && echo "/etc/1bit/installed: present" || echo "/etc/1bit/installed: MISSING"
[ -s /etc/ssh/ssh_host_ed25519_key ] && echo "ssh host key: present" || echo "ssh host key: MISSING"
echo "== sshd listening"
awk 'NR>1 {print $2}' /proc/net/tcp 2>/dev/null | grep -qi ':0016$' && echo "port 22: listening" || echo "port 22: not listening"
echo "== apt"
apt-get update -q -o Dir::Etc::sourcelist=/etc/apt/sources.list.d/99-1bit-test.list \
    -o Dir::Etc::sourceparts=/dev/null -o APT::Get::List-Cleanup=0 2>&1 | tail -2
apt-cache policy 1bit-engine 2>/dev/null | head -3
grep -h '^deb ' /etc/apt/sources.list.d/*.list 2>/dev/null
echo "== lemonade"
key=$(sed -n 's/^LEMONADE_API_KEY=//p' /etc/1bit/lemonade.env 2>/dev/null)
if [ -z "$key" ]; then echo "API key: MISSING"; else echo "API key: generated (${#key} chars)"; fi
i=0; while [ $i -lt 30 ]; do
    curl -sf -H "Authorization: Bearer $key" http://127.0.0.1:13305/api/v1/health >/dev/null 2>&1 && break
    i=$((i + 1)); sleep 2
done
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:13305/api/v1/models)
echo "without the key: HTTP $code (expected 401)"
echo "models: $(curl -s -H "Authorization: Bearer $key" http://127.0.0.1:13305/api/v1/models 2>/dev/null | tr ',' '\n' | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p' | tr '\n' ' ')"
reply=$(curl -s --max-time 300 -H "Authorization: Bearer $key" -H 'Content-Type: application/json' \
    http://127.0.0.1:13305/api/v1/chat/completions \
    -d '{"model":"Qwen3-0.6B-Q4_K_M","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' 2>/dev/null |
    sed -n 's/.*"content": *"\([^"]*\)".*/\1/p' | head -1)
echo "chat: ${reply:-no answer}"
echo "== end"
touch /data/vmtest.done
poweroff -f
INGUEST
chmod 755 "$rootfs/usr/lib/1bit/vmtest.sh"
cat > "$rootfs/etc/systemd/system/1bit-vmtest.service" <<'UNIT'
[Unit]
Description=test-only: appliance self-check
After=multi-user.target 1bit-lemonade.service 1bit-first-boot.service
Wants=1bit-lemonade.service

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 10
ExecStart=/usr/lib/1bit/vmtest.sh

[Install]
WantedBy=multi-user.target
UNIT
ln -sfn /etc/systemd/system/1bit-vmtest.service \
    "$rootfs/etc/systemd/system/multi-user.target.wants/1bit-vmtest.service"

# --- the two disks --------------------------------------------------------------------------------
rootimg=$work/rootfs.img; dataimg=$work/data.img
# EFI firmware in the VM, not a legacy BIOS boot: then /sys/firmware/efi exists and the secureboot row
# answers something, instead of failing for a QEMU configuration choice. Secure Boot itself stays off
# (OVMF_CODE, not .secboot): enrolling our MOK is the machine's step, not a VM's.
for f in OVMF_CODE_4M.fd OVMF_VARS_4M.fd; do
    [ -f /usr/share/OVMF/$f ] && cp /usr/share/OVMF/$f "$work/$f"
done
code=$work/OVMF_CODE_4M.fd; vars=$work/OVMF_VARS_4M.fd
[ -f "$code" ] && [ -f "$vars" ] || echo "vmtest-v2: no OVMF on this host; booting legacy (the secureboot row will fail)" >&2
truncate -s "$(( $(du -sm "$rootfs" | cut -f1) + 2048 ))M" "$rootimg"
mkfs.ext4 -q -F -L 1BIT-ROOT -d "$rootfs" "$rootimg"
truncate -s 8G "$dataimg"
mkfs.ext4 -q -F -L 1BIT-DATA -d "$seed" "$dataimg"

qemu-system-x86_64 -enable-kvm -cpu host -m 8G -smp 8 -machine q35 \
    ${code:+-drive if=pflash,format=raw,readonly=on,file=$code} \
    ${vars:+-drive if=pflash,format=raw,file=$vars} \
    -kernel "$kern" -initrd "$initrd" \
    -append "root=/dev/vda rw console=ttyS0,115200 loglevel=4" \
    -drive file="$rootimg",if=virtio,format=raw,cache=writeback \
    -drive file="$dataimg",if=virtio,format=raw,cache=writeback \
    -nic user,model=virtio-net-pci \
    -display none -serial file:"$work/serial.log" -no-reboot &
vm=$!
trap 'kill $vm 2>/dev/null; rm -rf "$work"' EXIT
echo "booting; the appliance powers itself off when its checks are done (max 12 min)"
waited=0
while kill -0 $vm 2>/dev/null && [ $waited -lt 720 ]; do sleep 5; waited=$((waited + 5)); done
if kill -0 $vm 2>/dev/null; then echo "VMTEST-V2 FAIL (the guest never finished; serial: $work/serial.log)"; kill $vm; tail -20 "$work/serial.log"; exit 1; fi
echo "guest finished after about ${waited}s"

# --- read the transcript out of the data image ----------------------------------------------------
mkdir -p "$work/data"
if ! mount -o ro,loop,noload "$dataimg" "$work/data" 2>/dev/null; then
    echo "VMTEST-V2 FAIL (could not read the data image)"; exit 1
fi
tr=$(cat "$work/data/vmtest.txt" 2>/dev/null || true)
umount "$work/data" 2>/dev/null || true
printf '%s\n' "$tr" | sed 's/^/  /'
echo "--- console:"
grep -a "1bit OS, Lemonade Server appliance" "$work/serial.log" | head -1

ok=1
have() { printf '%s\n' "$tr" | grep -q -- "$1"; }
if have "== end"; then echo "guest: the self-check ran to the end"; else echo "guest: the self-check did not finish"; ok=0; fi
for p in 1bit-engine 1bit-lemonade 1bit-appliance 1bit-archive-keyring 1bit-kernel-image- 1bit-kernel-modules- 1bit-ryzenadj; do
    if have "$p"; then echo "package installed: $p"; else echo "package MISSING: $p"; ok=0; fi
done
have "1bit OS, Lemonade Server appliance" && echo "banner: edition wording" || { echo "banner: edition wording missing"; ok=0; }
have "Lemonade Server by AMD, running on the 1bit engine" && echo "banner: credit line" || { echo "banner: credit line missing"; ok=0; }
have "ext4" && have "/data" && echo "/data: mounted (ext4, from its own disk)" || { echo "/data: not mounted"; ok=0; }
have "1bit --help: ok" && echo "engine: 1bit runs" || { echo "engine: 1bit does not run"; ok=0; }
# the package's own libraries must resolve beside it: only the machine's ROCm userspace may be missing
lmiss=$(printf '%s\n' "$tr" | sed -n 's/^llama-server missing: //p' | head -1)
if printf '%s' "$lmiss" | grep -q -E 'libllama|libggml'; then
    echo "engine: llama-server cannot load its own libraries ($lmiss)"; ok=0
else
    echo "engine: llama-server's own libraries resolve from the package (missing only: ${lmiss:-none})"
fi
have "kernel image: /boot/vmlinuz-" && echo "kernel package: our vmlinuz is in /boot" || { echo "kernel package: no vmlinuz"; ok=0; }
have "initramfs: /boot/initrd.img-" && echo "kernel package: its initramfs was generated" || { echo "kernel package: no initramfs"; ok=0; }
have "mode      :" && echo "1bit-selftest: reported a mode" || { echo "1bit-selftest: no mode"; ok=0; }
have "/etc/1bit/installed: present" && echo "first boot: installed marker written" || { echo "first boot: no installed marker"; ok=0; }
have "ssh host key: present" && echo "first boot: SSH host key generated on the box" || { echo "first boot: no SSH host key"; ok=0; }
have "port 22: listening" && echo "sshd: listening on 22" || { echo "sshd: not listening"; ok=0; }
have "Installed: " && echo "apt: the repository answers (apt-cache policy resolved 1bit-engine)" || { echo "apt: no answer from the repository"; ok=0; }
have "https://1bit.gg/apt" && have "signed-by=/usr/share/keyrings/1bit-archive-keyring.gpg" \
    && echo "apt: the shipped source is 1bit.gg/apt, trusted through our keyring" \
    || { echo "apt: the shipped source is wrong"; ok=0; }
have "API key: generated" && echo "Lemonade: API key generated at first boot" || { echo "Lemonade: no API key"; ok=0; }
have "without the key: HTTP 401" && echo "Lemonade: refuses without the key" || { echo "Lemonade: does not refuse without the key"; ok=0; }
have "Qwen3-0.6B-Q4_K_M" && echo "Lemonade: lists the data partition's GGUF" || { echo "Lemonade: does not list the GGUF"; ok=0; }
case "$tr" in
    *"chat: Paris"*) echo "Lemonade: answered a chat (vulkan in the VM)";;
    *) echo "Lemonade: chat did not answer ($(printf '%s' "$tr" | grep '^chat:' | head -1))"; ok=0;;
esac
# release-check: every FAIL row must be the sealed-NPU one, which a stage without the sealed sets
# cannot pass (they are built where the NPU is)
rcfail=$(printf '%s\n' "$tr" | sed -n '/== release-check/,/== first boot/p' | grep -c '^FAIL' || true)
rcother=$(printf '%s\n' "$tr" | sed -n '/== release-check/,/== first boot/p' | grep '^FAIL' | grep -vc sealed || true)
if [ "${rcfail:-0}" = 0 ]; then echo "1bit-release-check: no FAIL rows"
elif [ "${rcother:-1}" = 0 ]; then echo "1bit-release-check: only the sealed-NPU row fails (this stage carries no sealed sets)"
else echo "1bit-release-check: ${rcfail} FAIL row(s), not just the sealed one"; ok=0; fi

[ $ok = 1 ] && echo "VMTEST-V2 PASS" || echo "VMTEST-V2 FAIL"
[ $ok = 1 ] || exit 1
