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
export PATH=/bin:/sbin:/usr/bin:/usr/sbin HOME=/root
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
mkdir -p /dev/pts && mount -t devpts devpts /dev/pts
mount -t tmpfs tmpfs /tmp
mount -t tmpfs tmpfs /run
hostname -F /etc/hostname

echo; echo "  1bit OS  —  one engine, any model  —  https://1bit.gg"; echo

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

# SSH: key logins only, keys from the stick; the host key is made once and kept there
if [ -s /data/ssh/authorized_keys ]; then
    mkdir -p /root/.ssh && cp /data/ssh/authorized_keys /root/.ssh/ && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys
    [ -f /data/ssh/host_ed25519 ] || dropbearkey -t ed25519 -f /data/ssh/host_ed25519 >/dev/null 2>&1
    dropbear -r /data/ssh/host_ed25519 -s -g && echo "SSH: on (key logins; keys from /data/ssh/authorized_keys)"
else
    echo "SSH: off (put a public key in /data/ssh/authorized_keys on the stick)"
fi

# 1bit serve at boot, when /data/1bit.conf names a model
if [ -f /data/1bit.conf ]; then
    . /data/1bit.conf
    if [ -n "$MODEL" ]; then
        1bit serve -m "$MODEL" --device "${DEVICE:-vulkan}" --host 0.0.0.0 --port "${PORT:-8000}" $ARGS > /tmp/serve.log 2>&1 &
        echo "serving $MODEL on port ${PORT:-8000} (log: /tmp/serve.log)"
    fi
fi

echo "GPU: $(ls /dev/dri/renderD* 2>/dev/null | tr '\n' ' ')  NPU: $(ls /dev/accel/accel* 2>/dev/null | tr '\n' ' ')"
echo "IP:  $(ip -4 -o addr show scope global | awk '{print $2, $4}' | tr '\n' ' ')"
echo; echo "Serve a model:  1bit serve -m /data/models/<file.gguf> --device vulkan --host 0.0.0.0"; echo

# 1bit.check on the kernel command line: check the userspace starts, report, power off (for CI
# and for QEMU, which has neither the GPU nor the NPU)
if grep -q "1bit.check" /proc/cmdline; then
    ok=1
    1bit --help >/dev/null 2>&1 && echo "CHECK 1bit: runs" || { echo "CHECK 1bit: FAILED"; ok=0; }
    /opt/1bit-llama/llama-server --version >/dev/null 2>&1 && echo "CHECK llama-server: runs" || { echo "CHECK llama-server: FAILED"; ok=0; }
    [ -e /opt/xilinx/xrt/lib/libxrt_driver_xdna.so ] && echo "CHECK XRT XDNA plugin: present" || { echo "CHECK XRT: FAILED"; ok=0; }
    [ -e /lib/firmware/amdnpu/17f0_11 ] && echo "CHECK NPU firmware: present" || { echo "CHECK NPU firmware: FAILED"; ok=0; }
    dropbear -V 2>&1 | grep -q Dropbear && echo "CHECK dropbear: runs" || { echo "CHECK dropbear: FAILED"; ok=0; }
    echo "CHECK modules: $(find /lib/modules -name '*.ko*' | wc -l) kernel modules, amdgpu $(modinfo -F version amdgpu >/dev/null 2>&1 && echo found || echo MISSING)"
    [ $ok = 1 ] && echo "1BIT-OS CHECK PASS" || echo "1BIT-OS CHECK FAIL"
    poweroff -f
fi

# a shell on the console and the serial port
[ -e /dev/ttyS0 ] && setsid sh -c 'exec sh </dev/ttyS0 >/dev/ttyS0 2>&1' &
exec setsid cttyhack sh
