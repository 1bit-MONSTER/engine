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
# proto.sh <work-dir> <engine-stage> <lemonade-prefix> [backend-cache] [model.gguf]
#
# The whole v2 prototype in one command, on the build host and nowhere near a device:
#   a throwaway archive key -> the four debs -> a signed apt repository -> an mmdebstrap rootfs that
#   apt-installs 1bit-appliance from it -> that rootfs booted in QEMU as a machine -> VMTEST-V2.
#
# <engine-stage> is a release stage (tarball or directory with 1bit and the backend dirs), and
# <lemonade-prefix> a build-lemonade.sh prefix. [backend-cache] is an existing Lemonade cache/bin
# directory to seed the data partition with (so the VM needs no network for Lemonade's backends), and
# [model.gguf] a small model for the chat row.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '21,32p' "$0"; exit 2; }
w=$1 stage=$2 lemon=$3 backends=${4:-} model=${5:-$HOME/models/Qwen3-0.6B-Q4_K_M.gguf}
here=$(cd "$(dirname "$0")" && pwd)
version=v2026.43
mkdir -p "$w"
w=$(cd "$w" && pwd)

echo "== archive key (throwaway: the published repository is signed with the operator's key)"
if [ ! -s "$w/gnupg/pubring.kbx" ]; then
    mkdir -p "$w/gnupg"; chmod 700 "$w/gnupg"
    GNUPGHOME=$w/gnupg gpg --batch --quiet --passphrase '' --quick-gen-key \
        '1bit archive (prototype) <archive@1bit.gg>' rsa2048 sign never 2>/dev/null
fi
keyid=$(GNUPGHOME=$w/gnupg gpg --batch --with-colons --list-secret-keys 2>/dev/null | awk -F: '/^sec:/ {print $5; exit}')
[ -n "$keyid" ] || { echo "proto: no signing key"; exit 1; }
GNUPGHOME=$w/gnupg gpg --batch --yes --export "$keyid" > "$w/1bit-archive-keyring.gpg"
echo "   key $keyid"

echo "== the packages, from the existing stage (no binary is rebuilt)"
"$here/mkdebs.sh" "$stage" "$lemon" "$version" "$w/debs" "$w/1bit-archive-keyring.gpg"

echo "== the repository, signed"
"$here/mkrepo.sh" "$w/debs" "$w/repo" "$w/gnupg" "$keyid"

echo "== the data partition the VM boots with (models, Lemonade's config and backends)"
seed=$w/data
rm -rf "$seed"; mkdir -p "$seed/models" "$seed/lemonade/config" "$seed/lemonade/huggingface" "$seed/cache/hrx-jit" "$seed/npu"
[ -f "$model" ] && cp -L "$model" "$seed/models/"
printf '{\n    "host": "0.0.0.0",\n    "extra_models_dir": "/data/models",\n    "models_dir": "/data/lemonade/huggingface"\n}\n' \
    > "$seed/lemonade/config/config.json"
if [ -n "$backends" ] && [ -d "$backends/llamacpp" ]; then
    mkdir -p "$seed/lemonade/cache/bin"
    cp -a "$backends/llamacpp" "$seed/lemonade/cache/bin/"
    echo "   backends: $(ls "$seed/lemonade/cache/bin/llamacpp" | tr '\n' ' ')"
else
    echo "   WARNING: no backend cache given - Lemonade will have no llama.cpp backend in the VM" >&2
fi

echo "== an SSH key for the test image (the real live image bakes its own)"
[ -s "$w/testkey" ] || ssh-keygen -q -t ed25519 -N '' -f "$w/testkey"

echo "== the root filesystem (mmdebstrap, one Ubuntu snapshot) + apt install 1bit-appliance"
"$here/mkrootfs.sh" "$w/repo" "$w/rootfs" "$w/testkey.pub"

echo "== boot it and check the appliance"
"$here/vmtest-v2.sh" "$w/rootfs" "$seed" "$w/testkey"
