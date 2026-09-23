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
# build-kernel.sh <out> [jobs]
#
# Builds the Linux kernel pinned in third_party/linux (upstream torvalds/linux)
# as Debian packages in <out>: linux-image, linux-headers and linux-libc-dev
# for <release>-1bit. The config is config/kernel/strixhalo.config with
# config/kernel/strixhalo.fragment merged on top, then olddefconfig. It
# installs nothing: installing and rebooting are separate, deliberate steps
# (docs/kernel.md).
#
# The build uses LLVM (clang + lld), like the base config's kernel. The kernel
# source can be given as $LINUX_SRC (a checkout of the pinned commit), else
# third_party/linux, fetched at the pinned commit with depth 1.
set -euo pipefail
out=${1:?usage: build-kernel.sh <out> [jobs]}
jobs=${2:-$(nproc)}
root=$(cd "$(dirname "$0")/.." && pwd)
src=${LINUX_SRC:-$root/third_party/linux}
mkdir -p "$out"
out=$(cd "$out" && pwd)

if [ ! -f "$src/Makefile" ]; then
    [ -z "${LINUX_SRC:-}" ] || { echo "LINUX_SRC=$src is not a kernel tree"; exit 1; }
    git -C "$root" submodule update --init --depth 1 third_party/linux
fi
for t in clang ld.lld flex bison bc pahole zstd; do
    command -v "$t" >/dev/null || { echo "missing build tool: $t"; exit 1; }
done
# bindeb-pkg checks the Debian build dependencies itself; on Debian/Ubuntu:
#   apt-get install clang lld llvm flex bison bc dwarves libelf-dev libdw-dev libssl-dev dpkg-dev debhelper rsync kmod cpio zstd

# Out-of-tree object directory, so the source stays clean.
obj=$out/obj
mkdir -p "$obj"
cp "$root/config/kernel/strixhalo.config" "$obj/.config"
(cd "$src" && KCONFIG_CONFIG="$obj/.config" \
    scripts/kconfig/merge_config.sh -m -O "$obj" "$obj/.config" "$root/config/kernel/strixhalo.fragment")
make -C "$src" O="$obj" LLVM=1 olddefconfig

# The fragment is what the engine depends on: fail if olddefconfig dropped it.
for opt in CONFIG_DRM_ACCEL=y CONFIG_DRM_ACCEL_AMDXDNA=m; do
    grep -qx "$opt" "$obj/.config" || { echo "config lost $opt after olddefconfig"; exit 1; }
done

make -C "$src" O="$obj" LLVM=1 -j "$jobs" bindeb-pkg
# bindeb-pkg writes the packages next to the object directory.
ls -1 "$out"/*.deb
