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
# vmtest.sh <out> [engine build dir] [model.gguf]
#
# Boots the 1bit OS USB image the way a PC boots the stick: UEFI firmware (OVMF), the image on
# a USB xHCI controller, a network card, in QEMU/KVM. The test image carries a throwaway SSH key,
# a small model, a 1bit.conf that serves it at boot, and lavapipe (Vulkan on the CPU, as the VM
# has no Radeon GPU). Then over SSH and the OpenAI API: the system is up, the stick's data
# partition is mounted, 1bit serve answers a chat request. Prints VMTEST PASS or FAIL.
set -euo pipefail
out=${1:?usage: vmtest.sh <out> [engine build dir] [model.gguf]}
build=${2:-$HOME/.cache/1bit-os/src/build}
model=${3:-$HOME/models/Qwen3-0.6B-Q4_K_M.gguf}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)

rm -f "$out/key" "$out/key.pub"; ssh-keygen -q -t ed25519 -N "" -f "$out/key"
# lavapipe is the VM's only Vulkan device; llama.cpp skips CPU Vulkan devices unless listed
# (a 4096-token context: lavapipe reports too little memory for the model's full one)
printf 'MODEL=/data/models/%s\nDEVICE=vulkan\nPORT=8000\nARGS=\"--ctx-size 4096\"\nexport GGML_VK_VISIBLE_DEVICES=0\n' "$(basename "$model")" > "$out/1bit.conf"
LAVAPIPE=1 AUTHORIZED_KEYS="$out/key.pub" MODELS="$model" CONF="$out/1bit.conf" \
    "$here/mkimage.sh" "$out/image" "$build" > "$out/mkimage.log" 2>&1

cp /usr/share/OVMF/OVMF_VARS_4M.fd "$out/vars.fd"
qemu-system-x86_64 -enable-kvm -cpu host -m 16G -smp 8 -machine q35 \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
    -drive if=pflash,format=raw,file="$out/vars.fd" \
    -device qemu-xhci -drive id=stick,if=none,format=raw,file="$out/image/1bit-os.img" -device usb-storage,drive=stick \
    -nic user,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2222-:22,hostfwd=tcp:127.0.0.1:18000-:8000 \
    -display none -serial file:"$out/serial.log" -no-reboot &
vm=$!
trap 'kill $vm 2>/dev/null || true' EXIT
ssh_vm() { ssh -q -i "$out/key" -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 root@127.0.0.1 "$@"; }

ok=1
for i in $(seq 60); do ssh_vm true 2>/dev/null && break; sleep 3; done
if ssh_vm true 2>/dev/null; then
    echo "VM: SSH up after about $((i * 3)) s (UEFI boot from the USB image)"
    ssh_vm 'uname -r; mount | grep -q " /data " && echo "data partition: mounted, $(ls /data/models)"; ls /usr/share/vulkan/icd.d'
else
    echo "VM: no SSH"; ok=0
fi
for i in $(seq 90); do curl -sf 127.0.0.1:18000/v1/models >/dev/null 2>&1 && break; sleep 2; done
reply=$(curl -s --max-time 300 127.0.0.1:18000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null || true)
if [ -n "$reply" ]; then echo "1bit serve (Vulkan on the VM's CPU) answered: $reply"; else echo "1bit serve: no answer"; ok=0
    ssh_vm 'grep -i -E "error|fail|abort|assert|out of memory|what\\(\\)" /tmp/serve.log | head -12; echo ...; tail -4 /tmp/serve.log' 2>/dev/null || true; fi
ssh_vm 'poweroff -f' 2>/dev/null || true
wait $vm 2>/dev/null || true
trap - EXIT
[ $ok = 1 ] && echo "VMTEST PASS" || { echo "VMTEST FAIL (serial console: $out/serial.log)"; exit 1; }
