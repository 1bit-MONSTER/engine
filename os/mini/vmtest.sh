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
# Lemonade's web app loads and serves its KaTeX font as a file, Lemonade refuses a request without
# the key, lists the stick's model in /api/v1/models, and answers a chat request with it. The VM
# has no Radeon GPU, so Lemonade serves that chat with its llamacpp CPU backend (its own choice);
# the onebit entry for the same model needs an AMD GPU or NPU for Lemonade to run it (the recipe's
# backends are hrx and npu), so the test only reports what Lemonade does with it. Prints VMTEST
# PASS or FAIL. On a machine without TheRock, LLAMA_SERVER (mkimage.sh) makes a CPU-only test image.
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
    # the Lemonade CLI must be on an SSH command's PATH, as on the console (os/mini/init.sh)
    if lv=$(ssh_vm 'lemonade --version 2>&1 | head -1' 2>/dev/null) && [ -n "$lv" ] && ! echo "$lv" | grep -q 'not found'; then
        echo "lemonade CLI over SSH: $lv"
    else echo "lemonade CLI over SSH: not on PATH (${lv:-no output})"; ok=0; fi
else
    echo "VM: no SSH"; ok=0
fi
for i in $(seq 60); do curl -sf "${auth[@]}" "$api/api/v1/health" >/dev/null && break; sleep 2; done
code=$(curl -s -o "$out/ui.html" -w '%{http_code} %{content_type}' "$api/")
if [ "${code%% *}" = 200 ] && grep -qi '<html' "$out/ui.html"; then echo "Lemonade web app: HTTP $code"; else echo "Lemonade web app: HTTP $code"; ok=0; fi
# the web app's math fonts are files on the stick, not links to the build machine's fonts-katex
# (os/mini/mkimage.sh, pack_localize_links)
code=$(curl -s -o "$out/katex.woff2" -w '%{http_code}' "$api/KaTeX_Main-Regular.woff2")
if [ "$code" = 200 ] && [ "$(head -c 4 "$out/katex.woff2")" = wOF2 ]; then
    echo "Lemonade web app font KaTeX_Main-Regular.woff2: HTTP 200, $(stat -c %s "$out/katex.woff2") bytes (woff2)"
else echo "Lemonade web app font KaTeX_Main-Regular.woff2: HTTP $code, not a woff2 font"; ok=0; fi
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
# whose recipe has no backend for this machine (onebit: hrx, npu), even with show_all
if ssh_vm "grep -q '\"$name-1bit\": {\"checkpoint\": \"/data/models/$name.gguf\", \"source\": \"local_path\", \"recipe\": \"onebit\"}' /data/lemonade/config/user_models.json" 2>/dev/null; then
    echo "Lemonade registry on the stick: user.$name-1bit (onebit) -> /data/models/$name.gguf"
else echo "Lemonade registry on the stick: user.$name-1bit (onebit) is missing"; ok=0; fi
echo "Lemonade chat with $name-1bit (onebit; the VM has no AMD GPU or NPU, reported only): $(chat "$name-1bit")"
echo "Lemonade's llama-server in the VM:"; ssh_vm 'grep -i -E "vulkan|offload|device" /tmp/lemond.log | grep -v -i "nvidia\|download" | head -6' 2>/dev/null || true
dl=$(ssh_vm 'grep -i -c "downloading" /tmp/lemond.log' 2>/dev/null || true)
echo "Lemonade downloads during the test: ${dl:-?} (QEMU's network is restricted: the backends came from the stick)"
# --- what this image is FOR: 1bit OS = Lemonade with the 1bit engine embedded (os/README.md) ---
# The LIVE banner has to reach the console: the serial line is the VM's screen, and a blank screen is
# what made a working appliance look dead on real hardware (os/mini/init.sh writes to tty1 as well).
# grub's own menu says "1bit OS" too, so match init's live-mode banner, which say() writes to
# /dev/console (= this serial line in the VM) and to tty1.
if grep -q "LIVE. You are root, on RAM" "$out/serial.log" 2>/dev/null; then
    echo "Live banner on the console: $(grep "LIVE. You are root, on RAM" "$out/serial.log" | head -1 | cut -c1-72)"
elif ssh_vm 'test -e /etc/1bit/live' 2>/dev/null; then
    echo "Live banner on the console: not on the serial line, but the live marker exists (tty1 only)"; ok=0
else echo "Live banner on the console: missing from $out/serial.log"; ok=0; fi

# The self-test has to run from the image. It cannot PASS in a VM - there is no Radeon GPU and no XDNA
# NPU - so what is asserted is that it runs, reports the live mode, and says plainly that the NPU is
# what is missing. The NPU PASS belongs to the real boot (owner's step).
st=$(ssh_vm '1bit-selftest 2>&1 | tail -25' 2>/dev/null || true)
case "$st" in
    *"mode      : LIVE"*) echo "1bit-selftest: runs, and reports the LIVE mode";;
    *) echo "1bit-selftest: did not report the LIVE mode"; ok=0;;
esac
case "$st" in
    *"RESULT-NPU"*) echo "1bit-selftest NPU line in the VM: $(printf '%s' "$st" | grep "RESULT-NPU" | head -1 | cut -c1-90)";;
    *) echo "1bit-selftest: no RESULT-NPU line"; ok=0;;
esac

# Lemonade's own catalog carries the three NPU entries by name (catalog: onebit, recipe_options
# onebit_backend npu). They must LIST in the image - loading is impossible here because the NPU cannot
# run in QEMU, and saying so is the point of this check.
catalog=$(ssh_vm 'cat /opt/lemonade/bin/resources/server_models.json 2>/dev/null' 2>/dev/null || true)
if [ -n "$catalog" ]; then
    for want in Qwen3-0.6B-NPU-1bit Qwen3-1.7B-NPU-1bit Qwen3-4B-NPU-1bit; do
        case "$catalog" in
            *"\"$want\""*) echo "The image's catalog carries the NPU entry by name: $want";;
            *) echo "The image's catalog is missing the NPU entry: $want"; ok=0;;
        esac
    done
    case "$catalog" in
        *'\"onebit_backend\": \"npu\"'*) echo "The catalog's NPU entries carry recipe_options.onebit_backend: npu";;
        *) echo "The catalog's NPU entries do not carry recipe_options.onebit_backend"; ok=0;;
    esac
else echo "could not read the image's server_models.json"; ok=0; fi
# and say plainly why they are not in the listing here
echo "Note: /api/v1/models cannot list them under QEMU - this Lemonade hides every onebit model when the" \
     "machine has no AMD GPU and no NPU (the stick's own onebit register answers model_not_found too)."

# Guest tier intact: take the engine binary away and Lemonade must stop offering the onebit entries;
# put it back and they return. The NPU entries above are exactly the guest-tier ones.
engine=$(ssh_vm 'readlink -f /bin/1bit 2>/dev/null' 2>/dev/null || true)
if [ -n "$engine" ]; then
    gt=$(ssh_vm "mv '$engine' /tmp/1bit.hidden && mkdir -p /tmp/gt-cfg /tmp/gt-cache && \
        cp /data/lemonade/config/user_models.json /tmp/gt-cfg/ && \
        (env -u LEMONADE_ONEBIT_BIN /opt/lemonade/bin/lemond --host 127.0.0.1 --port 13399 --no-broadcast \
            /tmp/gt-cache /tmp/gt-cfg >/tmp/gt.log 2>&1 &) && sleep 14 && \
        curl -s http://127.0.0.1:13399/api/v1/models | grep -o onebit | wc -l; \
        pkill -f 'port 13399' >/dev/null 2>&1; mv /tmp/1bit.hidden '$engine'; sleep 1" 2>/dev/null || true)
    case "${gt:-x}" in
        0) echo "Guest tier: with the engine hidden, 0 onebit entries (as upstream v2026.43.3 defines it)";;
        x) echo "Guest tier: could not run the hidden-engine check";;
        *) echo "Guest tier: $gt onebit entries with the engine hidden (expected 0)"; ok=0;;
    esac
else echo "Guest tier: could not resolve /bin/1bit in the VM"; ok=0; fi

[ $ok = 1 ] || ssh_vm 'grep -i -E "error|fail" /tmp/lemond.log | tail -12' 2>/dev/null || true
ssh_vm 'poweroff -f' 2>/dev/null || true
wait $vm 2>/dev/null || true
trap - EXIT
[ $ok = 1 ] && echo "VMTEST PASS" || { echo "VMTEST FAIL (serial console: $out/serial.log)"; exit 1; }
