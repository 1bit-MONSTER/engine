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

# drivers, in the order the boot needs them: network, USB storage and the USB4/Type-C stack first, so
# the machine is reachable and the stick readable even if a GPU probe is slow. sd_mod makes a USB
# stick a disk (usb-storage only presents it as a SCSI device); the TYPEC/Thunderbolt modules are what
# bring the USB4 ports up (they were in the image but nothing loaded them until now). realtek is the
# PHY driver the wired NIC needs and must load first: without it r8169 fails with -49 and the machine
# comes up with no network at all, which looks exactly like a hang.
for m in realtek r8169 mt7925e virtio_net xhci_pci usb_storage uas sd_mod nvme ahci vfat nls_cp437 nls_iso8859_1 nls_ascii ext4 phy-generic typec typec_ucsi ucsi_acpi typec_displayport thunderbolt; do modprobe -q "$m"; done
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
# A USB 2.0 stick can take 45 s or more to appear on this machine (sd 1:0:0:0: [sda] at t=46.4 s in the
# box's own log), and the old 20 s window expired before it did - so /data never mounted, the keys
# never arrived and nothing was logged. Wait much longer, and say so as we go.
i=0
while [ $i -lt 90 ]; do
    data=$(findfs LABEL=1BIT-DATA 2>/dev/null) && break
    [ $i = 30 ] && echo "still waiting for the 1BIT-DATA partition (t=${i}s)..."
    [ $i = 60 ] && echo "still waiting for the 1BIT-DATA partition (t=${i}s)..."
    sleep 1; i=$((i+1))
done
if [ -n "$data" ]; then mount "$data" /data && echo "models: /data/models (from $data)"; fi

# Evidence that outlives a live boot: stage markers on the stick, so a machine that hangs later can be
# diagnosed by reading /data/boot.log from another system instead of squinting at the console.
stage() { echo "$(date -u +%H:%M:%S) $*" >> /data/boot.log 2>/dev/null; echo "$*"; }
# If the data partition did not turn up, log to the EFI partition instead - it is the one pieces of the
# stick the firmware could read, so it is the best chance of keeping evidence across a panic.
if ! grep -q " /data " /proc/mounts; then
    os=$(findfs LABEL=1BIT-OS 2>/dev/null)
    if [ -n "$os" ] && mount -t vfat -o rw "$os" /data 2>/dev/null; then
        echo "log fallback: EFI partition mounted at /data"
    fi
fi
stage "stick mounted at ${data:-nothing}"

# Every message goes to the screen, the serial line and the stick: a live boot must never be a blank
# screen you have to guess about. Defined before its first use: it used to sit below the kernel link
# line, so the console lost that line to "say: not found" (seen in the VM's serial log).
say() { echo "$*"; echo "$*" >> /data/boot.log 2>/dev/null
    for c in /dev/console /dev/tty1; do [ -w "$c" ] && echo "$*" > "$c" 2>/dev/null; done; }

# The engine's own llama-server and the HSA runtime it loads are on the DATA PARTITION, not in the RAM
# image (mkimage.sh: grub will not load an initrd over ~103 MB and this one is at 89.5 MB, so anything
# added to the image after it is packed never reaches the stick - which is why 1bit.check said
# "CHECK llama-server: FAILED" on a released image). Put them where the engine and 1bit.check look:
# /opt/1bit-llama is on PATH, and /etc/1bit/libhsa names the on-stick runtime for the engine.
if [ -d /data/1bit-llama ]; then
    mkdir -p /opt /etc/1bit
    ln -sfn /data/1bit-llama /opt/1bit-llama
    # same on /data, from the Lemonade package that is already there (the build machine's
    # /opt/rocm-therock path does not exist on a stick)
    hsa=$(ls /data/lemonade/cache/bin/llamacpp-hrx/*/lib/libhsa-runtime64.so* 2>/dev/null | head -1)
    [ -n "$hsa" ] && echo "$hsa" > /etc/1bit/libhsa
    say "[engine] llama-server from /data/1bit-llama, HSA runtime ${hsa:-not found}"
else
    say "[engine] no /data/1bit-llama: '1bit serve' has no HRX llama-server on this boot"
fi

# The sealed NPU kernels live on the stick (they used to be inside the RAM image, which made it too big
# to load): link them to the path the engine looks for, beside itself.
if [ -d /data/npu/kernels ] && [ -d /home ]; then
    mkdir -p /home/bcloud/wt/npupkg-build/npu 2>/dev/null
    ln -sfn /data/npu/kernels /home/bcloud/wt/npupkg-build/npu/kernels 2>/dev/null
    say "[kernels] $(ls /data/npu/kernels | wc -l) sealed NPU sets linked beside the engine"
fi

n=0
step() { n=$((n+1)); say "[$n] $*"; }

# The screen gets a real interactive login on tty1, never a shell on /dev/console: /dev/console stops
# being what is painted once fbcon takes the DRM framebuffer, and a non-interactive shell has no
# prompt - so the machine sat there looking dead while it was serving. tty1 is the VT we are on.
screen_login() {
    [ -c /dev/tty1 ] || return 0
    ip=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
    [ -n "$ip" ] || ip="<this address>"
    chvt 1 2>/dev/null || true
    {
      echo
      if [ "$INSTALLED" = 1 ]; then
        echo "  1bit OS  v2026.43  -  INSTALLED on this machine. /data is on its disk; the engine and Lemonade start from here."
      else
        echo "  1bit OS  v2026.43  -  LIVE. You are root, on RAM: nothing is installed and no disk is touched."
      fi
      echo "  ----------------------------------------------------------------------------------------"
      echo "   1bit-selftest                 prove the engine and the NPU (writes /data/selftest.log)"
      echo "   lemonade list                 the models;  lemonade chat Qwen3-0.6B-1bit"
      echo "   cat /data/boot.log            what this boot did"
      echo "   ssh root@<this address>       also works, with the key in /data/ssh/authorized_keys"
      echo
      echo "  Lemonade is on the network:  http://${ip}:13305   (Lemonade Mobile finds it via the UDP beacon)"
      echo "  its API key:                 grep LEMONADE_API_KEY /data/lemonade/lemond.conf"
      echo
    } > /dev/tty1 2>/dev/null
    # cttyhack is what a classic initramfs uses, but here it exits at once (it cannot claim a tty that
    # setsid has already detached from) - measured on the box. setsid -c makes tty1 the controlling
    # terminal instead, which is what gives a prompt AND job control; the loop puts it back if you exit.
    ( while :; do setsid -c /bin/sh -i < /dev/tty1 > /dev/tty1 2>&1; sleep 2; done ) &
}

# Everything init prints goes to the visible VT as well as /dev/console: once fbcon takes the DRM
# framebuffer /dev/console is no longer what is painted, and the screen must never be the part that lies.
mc() { while IFS= read -r l; do for c in /dev/console /dev/tty1; do [ -w "$c" ] && printf '%s\n' "$l" > "$c" 2>/dev/null; done; done; }
# An installed disk carries /data/.installed (1bit-install writes it). The same init then boots the
# appliance from the disk: no stick menu to offer, and INSTALLED instead of LIVE on the screen and in
# /etc/1bit. Nothing else about the boot changes - same initramfs, same signed chain.
INSTALLED=0
grep -q " /data " /proc/mounts && [ -e /data/.installed ] && INSTALLED=1
# this boot's state lives here, and init makes it rather than trusting the image to carry it: mkimage.sh
# creates /etc/1bit after it packs the initramfs, so the directory was never in any image and the marker
# below was silently lost (the installed boot reported INSTALLED on screen and no marker in /etc/1bit)
mkdir -p /etc/1bit 2>/dev/null
MODE=live
if [ -c /dev/console ] && [ "$INSTALLED" = 0 ]; then
    {
      echo
      echo "  1bit OS  v2026.43  -  LIVE: runs from RAM, nothing is installed, disks are never touched"
      echo "  ------------------------------------------------------------------------------"
      echo "   1) Lemonade appliance            (default, starts in 12s)"
      echo "   2) Lemonade, verbose, no GPU driver"
      echo "   3) Root shell (no services)"
      echo "   4) Power off"
      echo
    } | mc
    printf "  choice: " | mc
    i=0
    while [ $i -lt 12 ]; do
        if read -r -t 1 -n 1 c < /dev/console 2>/dev/null; then
            case "$c" in
                2) MODE=nogpu;  break ;;
                3) MODE=shell;  break ;;
                4) MODE=off;    break ;;
                1|"") break ;;
            esac
        fi
        i=$((i+1))
    done
    printf "\n" | mc
fi
case "$MODE" in
    off)  say "powering off (choice 4)"; sync; poweroff -f ;;
    shell) say "root shell - the appliance will not start (choice 3)"; screen_login; while :; do sleep 3600; done ;;
    nogpu) say "verbose mode: GPU/NPU drivers are skipped (choice 2)" ;;
esac

# GPU and NPU last: on real hardware these probes are the slow part of the boot (and the usual suspect
# when a boot appears to stop), and by now the network is up and the stick is mounted.
# GPU/NPU drivers are NOT loaded here. The live image ships the machine's own /lib/modules, and its
# stock amdxdna disagrees with the private 1bit build - loading it at boot panics before the appliance
# is even reachable. The engine loads what it needs when it starts a model, so a live boot does not
# need them: leave them out and the appliance comes up, stays visible and stays reachable.
if [ "$MODE" = nogpu ]; then
    say "[gpu] skipped by menu choice"
else
    say "[gpu] loading the private NPU driver (amdxdna) and amdgpu"
    # amd_pmf first: it owns the platform sensors (hwmon) and the profile interface; without it the
    # appliance has no temperatures or fan readings at all ("no HWMON support due to missing pmf sensor")
    for m in amd_pmf amdgpu amdxdna; do modprobe -q "$m"; done
    mdev -s 2>/dev/null
    say "[gpu] devices: $(ls /dev/dri/renderD* /dev/kfd /dev/accel/accel* 2>/dev/null | tr '\n' ' ')"
    say "[gpu] hwmon: $(for h in /sys/class/hwmon/hwmon*; do cat $h/name 2>/dev/null; done | tr '\n' ' ')"
fi

# HRX: TheRock's HSA runtime (1bit serve also finds it at its compiled-in path), and the Loom
# kernel cache on the stick, so kernels compiled on the first boot are reused after a reboot
[ -s /etc/1bit/libhsa ] && export IREE_HAL_AMDGPU_LIBHSA_PATH="$(cat /etc/1bit/libhsa)"
if grep -q " /data " /proc/mounts && mkdir -p /data/cache/hrx-jit 2>/dev/null; then
    export GGML_HRX_JIT_CACHE_DIR=/data/cache/hrx-jit
else
    export GGML_HRX_JIT_CACHE_DIR=/tmp/hrx-jit
fi

# SSH sessions get the console's environment: Dropbear starts them with its own PATH
# (/usr/bin:/bin), so a login shell reads it from /etc/profile, and the Lemonade CLI and lemond are
# also linked into /usr/bin for `ssh root@<IP> lemonade ...` (no profile is read for a command)
{ echo "export PATH=$PATH HOME=/root GGML_HRX_JIT_CACHE_DIR=$GGML_HRX_JIT_CACHE_DIR"
  [ -n "${IREE_HAL_AMDGPU_LIBHSA_PATH:-}" ] && echo "export IREE_HAL_AMDGPU_LIBHSA_PATH=$IREE_HAL_AMDGPU_LIBHSA_PATH"
} > /etc/profile
for b in lemonade lemond; do [ -x "/opt/lemonade/bin/$b" ] && ln -sf "/opt/lemonade/bin/$b" "/usr/bin/$b"; done

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
    # A recipe option belongs in recipe_options. A top-level "onebit_backend" is not read by the
    # Lemonade server as of the #3771 refactor (measured: lemond passed --device auto and the engine
    # only found the NPU by luck), so an NPU model directory says it the way the catalog does.
    e="\"$1\": {\"checkpoint\": \"$2\", \"source\": \"local_path\", \"recipe\": \"onebit\"${3:+, \"recipe_options\": {\"onebit_backend\": \"$3\"}}}"
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
# A .q4nx directory is the engine's own model format, so it runs on the NPU: say so, or Lemonade
# sends it down the CPU path, which wants a llama-server this image does not carry. Lemonade's own
# catalog also carries the three NPU entries by name (Qwen3-{0.6B,1.7B,4B}-NPU-1bit): they list in
# /api/v1/models from the image, and this registers the copy on the stick so the same model loads
# without the Hub.
for d in /data/models/*/model.q4nx; do
    [ -f "$d" ] || continue
    d=${d%/model.q4nx}
    onebit_register "${d##*/}-1bit" "$d" npu
done
# Lemonade listens on the LAN (config/config.json "host": 0.0.0.0) so Lemonade Mobile and any other
# client can reach it, which means it must not be open: mint an API key once, keep it on the stick, and
# every request presents it as "Authorization: Bearer <key>".
kf=/data/lemonade/lemond.conf
if [ -f "$kf" ] && ! grep -q '^LEMONADE_API_KEY=..*' "$kf" 2>/dev/null; then
    { echo "LEMONADE_API_KEY=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
      echo "# minted at boot: send it as  Authorization: Bearer <key>  (Lemonade Mobile asks for it once)"
    } >> "$kf" && chmod 600 "$kf"
fi
# Lemonade's environment file (LEMONADE_API_KEY, LEMONADE_ADMIN_API_KEY, HF_TOKEN), for lemond only
( cd "$lem" || exit 1
  if [ -f lemond.conf ]; then set -a; . ./lemond.conf; set +a; fi
  exec lemond "$lem/cache" "$lem/config" ) > /tmp/lemond.log 2>&1 &
cfg() { sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\{0,1\}\([^\",}]*\).*/\1/p" "$lem/config/config.json" 2>/dev/null | head -1; }
# Only LEMONADE_API_KEY guards /api and /v1; the admin key alone guards /internal and leaves them open
setkey() { grep -q "^[[:space:]]*\\(export[[:space:]]*\\)\\{0,1\\}$1=..*" "$lem/lemond.conf" 2>/dev/null; }
key=off; setkey LEMONADE_API_KEY && key=on
[ "$key" = off ] && setkey LEMONADE_ADMIN_API_KEY && key="off, admin key only: /api and /v1 are open"
lhost=$(cfg host); lport=$(cfg port); lhost=${lhost:-localhost}; lport=${lport:-13305}
echo "Lemonade: http://$lhost:$lport  (API key: $key; log: /tmp/lemond.log)"
[ -f /data/1bit.conf ] && echo "note: /data/1bit.conf is no longer read; Lemonade serves at boot (/data/lemonade, os/README.md)"

if [ "$INSTALLED" = 1 ]; then
    echo "MODE: INSTALLED - 1bit OS runs from this machine's disk; /data is $(findfs LABEL=1BIT-DATA 2>/dev/null)."
    echo "      Reboot to restart it; the install is permanent (1bit-install, os/README.md)."
    touch /etc/1bit/installed 2>/dev/null
else
    echo "MODE: LIVE - everything runs from RAM; no internal disk is mounted and nothing is installed."
    echo "      Reboot to discard everything; only /data on the stick persists."
    touch /etc/1bit/live 2>/dev/null
fi
echo "GPU: $(ls /dev/dri/renderD* /dev/kfd 2>/dev/null | tr '\n' ' ')  NPU: $(ls /dev/accel/accel* 2>/dev/null | tr '\n' ' ')"
echo "IP:  $(ip -4 -o addr show scope global | awk '{print $2, $4}' | tr '\n' ' ')"
echo; echo "Lemonade's web app and API: http://$lhost:$lport   (from another machine: ssh -L $lport:localhost:$lport root@<IP>)"
echo "Prove the engine answers from this boot:  1bit-selftest   (console or SSH; logs to /data/selftest.log)"
echo "The engine alone, for debugging:  1bit serve -m /data/models/<file.gguf>   (auto: HRX0, on 127.0.0.1:8000)"; echo

# 1bit.check on the kernel command line: check the userspace starts, report, power off (for CI
# and for QEMU, which has neither the GPU nor the NPU)
if grep -q "1bit.check" /proc/cmdline; then
    ok=1
    1bit --help >/dev/null 2>&1 && echo "CHECK 1bit: runs" || { echo "CHECK 1bit: FAILED"; ok=0; }
    # The engine's own llama-server lives on the data partition (mkimage.sh), so a boot assembled from
    # the bare -kernel/-initrd pair has nothing to check: say SKIPPED with the reason instead of FAILED,
    # which is what a release gate that boots without the stick's data partition would otherwise report.
    if [ -d /data/1bit-llama ]; then
        /opt/1bit-llama/llama-server --version >/dev/null 2>&1 && echo "CHECK llama-server: runs" ||
            { echo "CHECK llama-server: FAILED (it is on /data but does not run)"; ok=0; }
    else
        echo "CHECK llama-server: SKIPPED - no data partition on this boot; the engine's llama-server is on /data/1bit-llama"
    fi
    lemonade --version >/dev/null 2>&1 && [ -f /opt/lemonade/bin/resources/defaults.json ] && echo "CHECK Lemonade: runs" || { echo "CHECK Lemonade: FAILED"; ok=0; }
    [ -d /opt/lemonade/bin/resources/web-app ] && echo "CHECK Lemonade web app: present" || echo "CHECK Lemonade web app: left out"
    if [ -s /etc/1bit/libhsa ]; then
        [ -e "$(cat /etc/1bit/libhsa)" ] && echo "CHECK HSA runtime (HRX): present" || { echo "CHECK HSA runtime: FAILED"; ok=0; }
    elif [ -d /data/1bit-llama ]; then
        echo "CHECK HSA runtime: FAILED - /data/1bit-llama is here but no libhsa was found in the data partition's Lemonade package"; ok=0
    else
        echo "CHECK HSA runtime: none (no data partition on this boot; the runtime comes from the Lemonade package on /data)"
    fi
    # the NPU runtime is compiled into 1bit: the image carries no XRT
    if find / -xdev -name 'libxrt*' 2>/dev/null | grep -q .; then echo "CHECK no XRT: FAILED"; ok=0
    else echo "CHECK no XRT: none in the image"; fi
    [ -e /lib/firmware/amdnpu/17f0_11 ] && echo "CHECK NPU firmware: present" || { echo "CHECK NPU firmware: FAILED"; ok=0; }
    dropbear -V 2>&1 | grep -q Dropbear && echo "CHECK dropbear: runs" || { echo "CHECK dropbear: FAILED"; ok=0; }
    echo "CHECK modules: $(find /lib/modules -name '*.ko*' | wc -l) kernel modules, amdgpu $(modinfo -F version amdgpu >/dev/null 2>&1 && echo found || echo MISSING)"
    [ $ok = 1 ] && echo "1BIT-OS CHECK PASS" || echo "1BIT-OS CHECK FAIL"
    poweroff -f
fi

# a shell on the console and the serial port
[ -e /dev/ttyS0 ] && setsid sh -c 'exec sh </dev/ttyS0 >/dev/ttyS0 2>&1' &
# PID 1 must never exit: the kernel panics with "Attempted to kill init" if it does (that is exactly
# what happened on real hardware - the script finished, the final exec returned, and the machine
# rebooted 42 s into the boot). Give the console a shell in the background and keep init alive forever.
screen_login
while :; do sleep 3600; done
