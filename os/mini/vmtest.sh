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
# a USB xHCI controller, a network card, in QEMU/KVM. The network is QEMU's restricted one: the
# guest reaches nothing outside the VM, so the stick has to work offline (Lemonade's backends come
# from the image, not a download). The test image carries a throwaway SSH key, a small model, and
# Lemonade's own settings for a network-facing server: "host": "0.0.0.0" in its config.json (QEMU's
# port forward reaches the guest's network address, not its loopback) and LEMONADE_API_KEY in its
# environment file. Then over SSH and HTTP: the system is up, the stick's data partition is mounted,
# Lemonade's web app loads, Lemonade refuses a request without the key, lists the stick's model in
# /api/v1/models, and answers a chat request with it. The VM has no Radeon GPU, so Lemonade serves
# that chat with its llamacpp CPU backend (its own choice); the onebit entry for the same model
# needs an AMD GPU or NPU for Lemonade to run it (the recipe's backends are hrx, vulkan, npu and
# cuda), so the test only reports what Lemonade does with it. Prints VMTEST PASS or FAIL. On a
# machine without TheRock or XRT, LLAMA_SERVER and NO_XRT (mkimage.sh) make a CPU-only test image.
set -euo pipefail
out=${1:?usage: vmtest.sh <out> [engine build dir] [model.gguf]}
build=${2:-$HOME/.cache/1bit-os/src/build}
model=${3:-$HOME/models/Qwen3-0.6B-Q4_K_M.gguf}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
name=$(basename "$model" .gguf)

rm -f "$out/key" "$out/key.pub"; ssh-keygen -q -t ed25519 -N "" -f "$out/key"
apikey=vmtest-$(od -An -N8 -tx8 /dev/urandom | tr -d ' ')
printf '{\n    "host": "0.0.0.0",\n    "extra_models_dir": "/data/models",\n    "models_dir": "/data/lemonade/huggingface"\n}\n' > "$out/config.json"
printf 'LEMONADE_API_KEY=%s\n' "$apikey" > "$out/lemond.conf"
AUTHORIZED_KEYS="$out/key.pub" MODELS="$model" LEMONADE_CONFIG="$out/config.json" LEMONADE_ENV="$out/lemond.conf" \
    "$here/mkimage.sh" "$out/image" "$build" > "$out/mkimage.log" 2>&1 || { tail -5 "$out/mkimage.log"; echo "VMTEST FAIL (mkimage)"; exit 1; }

cp /usr/share/OVMF/OVMF_VARS_4M.fd "$out/vars.fd"
qemu-system-x86_64 -enable-kvm -cpu host -m 16G -smp 8 -machine q35 \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
    -drive if=pflash,format=raw,file="$out/vars.fd" \
    -device qemu-xhci -drive id=stick,if=none,format=raw,file="$out/image/1bit-os.img" -device usb-storage,drive=stick \
    -nic user,restrict=on,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2222-:22,hostfwd=tcp:127.0.0.1:18000-:13305 \
    -display none -serial file:"$out/serial.log" -no-reboot &
vm=$!
trap 'kill $vm 2>/dev/null || true' EXIT
ssh_vm() { ssh -q -i "$out/key" -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 root@127.0.0.1 "$@"; }
api=127.0.0.1:18000
auth=(-H "Authorization: Bearer $apikey")

ok=1
for i in $(seq 60); do ssh_vm true 2>/dev/null && break; sleep 3; done
if ssh_vm true 2>/dev/null; then
    echo "VM: SSH up after about $((i * 3)) s (UEFI boot from the USB image)"
    ssh_vm 'uname -r; mount | grep -q " /data " && echo "data partition: mounted, $(ls /data/models)"
            echo "HRX JIT cache: $(ls -d /data/cache/hrx-jit 2>/dev/null || echo none)"
            echo "Lemonade backends on the stick: $(cd /data/lemonade/cache/bin 2>/dev/null && ls -d */* | tr "\n" " ")"
            echo "1bit (for the shell): $(1bit version 2>&1 | head -1)"'
else
    echo "VM: no SSH"; ok=0
fi
for i in $(seq 60); do curl -sf "${auth[@]}" "$api/api/v1/health" >/dev/null && break; sleep 2; done
code=$(curl -s -o "$out/ui.html" -w '%{http_code} %{content_type}' "$api/")
if [ "${code%% *}" = 200 ] && grep -qi '<html' "$out/ui.html"; then echo "Lemonade web app: HTTP $code"; else echo "Lemonade web app: HTTP $code"; ok=0; fi
code=$(curl -s -o /dev/null -w '%{http_code}' "$api/api/v1/models")
if [ "$code" = 401 ]; then echo "Lemonade without the API key: HTTP 401"; else echo "Lemonade without the API key: HTTP $code (expected 401)"; ok=0; fi
models=$(curl -s "${auth[@]}" "$api/api/v1/models" | python3 -c 'import json,sys; print(" ".join(m["id"] + "(" + m.get("recipe", "?") + ")" for m in json.load(sys.stdin)["data"]))' 2>/dev/null || true)
echo "Lemonade /api/v1/models: ${models:-nothing}"
case " $models " in *" $name(llamacpp) "*) ;; *) echo "  $name (llamacpp) is missing"; ok=0 ;; esac
chat() {   # <model>: the reply's text, or the error
    curl -s --max-time 300 "${auth[@]}" "$api/api/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$1\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? One word.\"}],\"max_tokens\":16,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" |
        python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"].strip() if "choices" in d else "error: " + json.dumps(d.get("error", d))[:300])' 2>/dev/null || true
}
reply=
for i in $(seq 10); do reply=$(chat "$name"); case $reply in ""|error*) sleep 10 ;; *) break ;; esac; done
case $reply in ""|error*) echo "Lemonade chat with $name: ${reply:-no answer}"; ok=0 ;;
               *) echo "Lemonade chat with $name answered: $reply ($(ssh_vm 'grep -o "llamacpp[^ ]* backend[^,]*\|Using backend: [a-z]*" /tmp/lemond.log | tail -1' 2>/dev/null))" ;; esac
# the onebit entry init.sh registered in Lemonade's registry on the stick. Lemonade hides a model
# whose recipe has no backend for this machine (onebit: hrx, vulkan, npu, cuda), even with show_all
if ssh_vm "grep -q '\"$name-1bit\": {\"checkpoint\": \"/data/models/$name.gguf\", \"source\": \"local_path\", \"recipe\": \"onebit\"}' /data/lemonade/config/user_models.json" 2>/dev/null; then
    echo "Lemonade registry on the stick: user.$name-1bit (onebit) -> /data/models/$name.gguf"
else echo "Lemonade registry on the stick: user.$name-1bit (onebit) is missing"; ok=0; fi
echo "Lemonade chat with $name-1bit (onebit; the VM has no AMD GPU or NPU, reported only): $(chat "$name-1bit")"
echo "Lemonade's llama-server in the VM:"; ssh_vm 'grep -i -E "vulkan|offload|device" /tmp/lemond.log | grep -v -i "nvidia\|download" | head -6' 2>/dev/null || true
dl=$(ssh_vm 'grep -i -c "downloading" /tmp/lemond.log' 2>/dev/null || true)
echo "Lemonade downloads during the test: ${dl:-?} (QEMU's network is restricted: the backends came from the stick)"
[ $ok = 1 ] || ssh_vm 'grep -i -E "error|fail" /tmp/lemond.log | tail -12' 2>/dev/null || true
ssh_vm 'poweroff -f' 2>/dev/null || true
wait $vm 2>/dev/null || true
trap - EXIT
[ $ok = 1 ] && echo "VMTEST PASS" || { echo "VMTEST FAIL (serial console: $out/serial.log)"; exit 1; }
