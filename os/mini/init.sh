#!/bin/busybox sh
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
# 1bit OS init (os/mini): everything runs from RAM; the internal disk is never mounted.
export PATH=/bin:/sbin:/usr/bin:/usr/sbin:/opt/lemonade/bin:/opt/1bit-llama HOME=/root
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
mkdir -p /dev/pts && mount -t devpts devpts /dev/pts
mount -t tmpfs tmpfs /tmp
mount -t tmpfs tmpfs /run
hostname -F /etc/hostname

echo; echo "  1bit OS  —  Lemonade with the 1bit engine  —  https://1bit.gg"; echo

# drivers: GPU, NPU, network, USB storage
# sd_mod makes a USB stick a disk (usb-storage only presents it as a SCSI device)
for m in amdgpu amdxdna r8169 mt7925e virtio_net xhci_pci usb_storage uas sd_mod vfat ext4; do modprobe -q "$m"; done
mdev -s 2>/dev/null

# network: DHCP on every wired interface
ip link set lo up
for n in /sys/class/net/e*; do
    [ -e "$n" ] || continue
    i=${n##*/}; ip link set "$i" up
    udhcpc -i "$i" -b -q -t 5 -s /etc/udhcpc.script >/dev/null 2>&1
done

# models: the stick's 1BIT-DATA partition, if present (the internal disk is left alone)
# USB storage can take several seconds to appear: wait for the partition, up to 20 s
data=
for i in $(seq 20); do
    data=$(findfs LABEL=1BIT-DATA 2>/dev/null) && break
    sleep 1
done
if [ -n "$data" ]; then mount "$data" /data && echo "models: /data/models (from $data)"; fi

# HRX: TheRock's HSA runtime (1bit serve also finds it at its compiled-in path), and the Loom
# kernel cache on the stick, so kernels compiled on the first boot are reused after a reboot
[ -s /etc/1bit/libhsa ] && export IREE_HAL_AMDGPU_LIBHSA_PATH="$(cat /etc/1bit/libhsa)"
if grep -q " /data " /proc/mounts && mkdir -p /data/cache/hrx-jit 2>/dev/null; then
    export GGML_HRX_JIT_CACHE_DIR=/data/cache/hrx-jit
else
    export GGML_HRX_JIT_CACHE_DIR=/tmp/hrx-jit
fi

# SSH: key logins only, keys from the stick; the host key is made once and kept there
if [ -s /data/ssh/authorized_keys ]; then
    mkdir -p /root/.ssh && cp /data/ssh/authorized_keys /root/.ssh/ && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
    [ -f /data/ssh/host_ed25519 ] || dropbearkey -t ed25519 -f /data/ssh/host_ed25519 >/dev/null 2>&1
    dropbear -r /data/ssh/host_ed25519 -s -g && echo "SSH: on (key logins; keys from /data/ssh/authorized_keys)"
else
    echo "SSH: off (put a public key in /data/ssh/authorized_keys on the stick)"
fi

# Lemonade at boot: lemond with Lemonade's defaults (localhost:13305), its config, state, cache and
# downloaded backends on the stick (/data/lemonade), so they survive a reboot and need no network.
# Host, port, API keys: Lemonade's own settings, in /data/lemonade/config/config.json and
# /data/lemonade/lemond.conf (os/README.md).
if grep -q " /data " /proc/mounts; then lem=/data/lemonade; else lem=/tmp/lemonade; echo "Lemonade: no 1BIT-DATA partition, its state is in RAM"; fi
mkdir -p "$lem/cache" "$lem/config"
# the models on the stick for Lemonade's onebit recipe (the 1bit engine), in Lemonade's own model
# registry, user_models.json: each GGUF (not projectors, only a split model's first part) and each
# NPU model directory, as <name>-1bit. Lemonade lists the GGUFs for llamacpp itself (extra_models_dir).
onebit_register() {   # <name> <path>: add the entry unless the registry has that name
    f=$lem/config/user_models.json
    case "$1$2" in *'"'*|*'\'*) return ;; esac
    [ -s "$f" ] || echo '{}' > "$f"
    grep -q "\"$1\"[[:space:]]*:" "$f" && return
    e="\"$1\": {\"checkpoint\": \"$2\", \"source\": \"local_path\", \"recipe\": \"onebit\"}"
    tr '\n' ' ' < "$f" | awk -v e="$e" '{ i = length($0); while (i > 0 && substr($0, i, 1) != "}") i--
        head = substr($0, 1, i - 1); inner = head; sub(/^[ \t]*\{/, "", inner); gsub(/[ \t]/, "", inner)
        print head (inner == "" ? "" : ", ") e "}" }' > "$f.new" && mv "$f.new" "$f" && echo "Lemonade: user.$1 (onebit) -> $2"
}
for m in /data/models/*.gguf; do
    [ -f "$m" ] || continue
    n=${m##*/}; n=${n%.gguf}
    case $n in mmproj*|*-0000[2-9]-of-*|*-000[1-9][0-9]-of-*) continue ;; esac
    onebit_register "$n-1bit" "$m"
done
for d in /data/models/*/model.q4nx; do [ -f "$d" ] && d=${d%/model.q4nx} && onebit_register "${d##*/}-1bit" "$d"; done
# Lemonade's environment file (LEMONADE_API_KEY, LEMONADE_ADMIN_API_KEY, HF_TOKEN), for lemond only
( cd "$lem" || exit 1
  if [ -f lemond.conf ]; then set -a; . ./lemond.conf; set +a; fi
  exec lemond "$lem/cache" "$lem/config" ) > /tmp/lemond.log 2>&1 &
cfg() { sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\).*/\1/p" "$lem/config/config.json" 2>/dev/null | head -1; }
key=off; grep -q '^[[:space:]]*\(export[[:space:]]*\)\{0,1\}LEMONADE_\(ADMIN_\)\{0,1\}API_KEY=..*' "$lem/lemond.conf" 2>/dev/null && key=on
lhost=$(cfg host); lport=$(cfg port); lhost=${lhost:-localhost}; lport=${lport:-13305}
echo "Lemonade: http://$lhost:$lport  (API key: $key; log: /tmp/lemond.log)"
[ -f /data/1bit.conf ] && echo "note: /data/1bit.conf is no longer read; Lemonade serves at boot (/data/lemonade, os/README.md)"

echo "GPU: $(ls /dev/dri/renderD* /dev/kfd 2>/dev/null | tr '\n' ' ')  NPU: $(ls /dev/accel/accel* 2>/dev/null | tr '\n' ' ')"
echo "IP:  $(ip -4 -o addr show scope global | awk '{print $2, $4}' | tr '\n' ' ')"
echo; echo "Lemonade's web app and API: http://$lhost:$lport   (from another machine: ssh -L $lport:localhost:$lport root@<IP>)"
echo "The engine alone, for debugging:  1bit serve -m /data/models/<file.gguf>   (auto: HRX0, on 127.0.0.1:8000)"; echo

# 1bit.check on the kernel command line: check the userspace starts, report, power off (for CI
# and for QEMU, which has neither the GPU nor the NPU)
if grep -q "1bit.check" /proc/cmdline; then
    ok=1
    1bit --help >/dev/null 2>&1 && echo "CHECK 1bit: runs" || { echo "CHECK 1bit: FAILED"; ok=0; }
    /opt/1bit-llama/llama-server --version >/dev/null 2>&1 && echo "CHECK llama-server: runs" || { echo "CHECK llama-server: FAILED"; ok=0; }
    lemonade --version >/dev/null 2>&1 && [ -f /opt/lemonade/bin/resources/defaults.json ] && echo "CHECK Lemonade: runs" || { echo "CHECK Lemonade: FAILED"; ok=0; }
    [ -d /opt/lemonade/bin/resources/web-app ] && echo "CHECK Lemonade web app: present" || echo "CHECK Lemonade web app: left out"
    if [ -s /etc/1bit/libhsa ]; then
        [ -e "$(cat /etc/1bit/libhsa)" ] && echo "CHECK HSA runtime (HRX): present" || { echo "CHECK HSA runtime: FAILED"; ok=0; }
    else
        echo "CHECK HSA runtime: none (a CPU-only test image)"
    fi
    if [ -e /opt/xilinx/xrt/lib/libxrt_driver_xdna.so ]; then echo "CHECK XRT XDNA plugin: present"
    elif [ -e /etc/1bit/no_xrt ]; then echo "CHECK XRT XDNA plugin: left out (test image)"
    else echo "CHECK XRT: FAILED"; ok=0; fi
    [ -e /lib/firmware/amdnpu/17f0_11 ] && echo "CHECK NPU firmware: present" || { echo "CHECK NPU firmware: FAILED"; ok=0; }
    dropbear -V 2>&1 | grep -q Dropbear && echo "CHECK dropbear: runs" || { echo "CHECK dropbear: FAILED"; ok=0; }
    echo "CHECK modules: $(find /lib/modules -name '*.ko*' | wc -l) kernel modules, amdgpu $(modinfo -F version amdgpu >/dev/null 2>&1 && echo found || echo MISSING)"
    [ $ok = 1 ] && echo "1BIT-OS CHECK PASS" || echo "1BIT-OS CHECK FAIL"
    poweroff -f
fi

# a shell on the console and the serial port
[ -e /dev/ttyS0 ] && setsid sh -c 'exec sh </dev/ttyS0 >/dev/ttyS0 2>&1' &
exec setsid cttyhack sh
