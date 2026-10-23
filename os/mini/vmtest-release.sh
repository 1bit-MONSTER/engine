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
# vmtest-release.sh <out> [engine build dir] [model.gguf]
# Boots the 1bit OS USB image the way a PC does (UEFI/OVMF, on a USB xHCI controller, restricted
# network) and runs /bin/1bit-release-check over SSH, the same command the owner runs on the stick.
# Prints VMTEST-RELEASE PASS or FAIL. The VM has no Radeon GPU and no NPU, so the rows that need the
# hardware report SKIP with that reason and the script says which rows those are.
set -euo pipefail
out=${1:?usage: vmtest-release.sh <out> [engine build dir] [model.gguf]}
build=${2:-$HOME/.cache/1bit-os/src/build}
model=${3:-$HOME/models/Qwen3-0.6B-Q4_K_M.gguf}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"; out=$(cd "$out" && pwd)
rm -f "$out/key" "$out/key.pub"; ssh-keygen -q -t ed25519 -N "" -f "$out/key"
apikey=vmtest-$(od -An -N8 -tx8 /dev/urandom | tr -d ' ')
printf '{\n    "host": "0.0.0.0",\n    "extra_models_dir": "/data/models",\n    "models_dir": "/data/lemonade/huggingface"\n}\n' > "$out/config.json"
printf 'LEMONADE_API_KEY=%s\n' "$apikey" > "$out/lemond.conf"
AUTHORIZED_KEYS="$out/key.pub" MODELS="$model" LEMONADE_CONFIG="$out/config.json" LEMONADE_ENV="$out/lemond.conf" \
    "$here/mkimage.sh" "$out/image" "$build" > "$out/mkimage.log" 2>&1 || { tail -5 "$out/mkimage.log"; echo "VMTEST-RELEASE FAIL (mkimage)"; exit 1; }
cp /usr/share/OVMF/OVMF_VARS_4M.fd "$out/vars.fd"
qemu-system-x86_64 -enable-kvm -cpu host -m 16G -smp 8 -machine q35 \
    -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
    -drive if=pflash,format=raw,file="$out/vars.fd" \
    -device qemu-xhci -drive id=stick,if=none,format=raw,file="$out/image/1bit-os.img" -device usb-storage,drive=stick \
    -nic user,restrict=on,model=virtio-net-pci,hostfwd=tcp:127.0.0.1:2223-:22 \
    -display none -serial file:"$out/serial.log" -no-reboot &
vm=$!
trap 'kill $vm 2>/dev/null || true' EXIT
ssh_vm() { ssh -q -i "$out/key" -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 root@127.0.0.1 "$@"; }
for i in $(seq 120); do ssh_vm true 2>/dev/null && up=1 && break; sleep 2; done
[ "${up:-0}" = 1 ] || { echo "VMTEST-RELEASE FAIL (no SSH in 240 s, see $out/serial.log)"; exit 1; }
ssh_vm 'test -x /bin/1bit-release-check' || { echo "VMTEST-RELEASE FAIL (/bin/1bit-release-check is not in the image)"; exit 1; }
set +e
ssh_vm '/bin/1bit-release-check' > "$out/release-check.log" 2>&1
rc=$?
set -e
cat "$out/release-check.log"
echo "release-check exit status in the guest: $rc"
if grep -q '^RESULT: PASS' "$out/release-check.log"; then
    echo "VMTEST-RELEASE PASS ($(grep -c '^PASS' "$out/release-check.log") PASS, $(grep -c '^SKIP' "$out/release-check.log") SKIP, $(grep -c '^FAIL' "$out/release-check.log") FAIL)"
    echo "hardware-only rows proved here as SKIP: $(grep '^SKIP' "$out/release-check.log" | cut -d' ' -f2 | tr '\n' ' ')"
    exit 0
fi
echo "VMTEST-RELEASE FAIL"; exit 1
