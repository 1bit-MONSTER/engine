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
# build-xdna.sh <prefix> [jobs]
#
# Builds the XDNA userspace stack pinned in third_party/xdna-driver (upstream
# amd/xdna-driver and its xrt submodule) into <prefix>: XRT, then the XDNA shim
# plugin (libxrt_driver_xdna). It installs nothing system-wide: both are staged
# with DESTDIR=<prefix>/root (XRT hard-codes /etc/OpenCL/vendors), so the XRT
# root is <prefix>/root/opt/xilinx/xrt. Point the engine at it with
# -DONEBIT_XRT_ROOT=<prefix>/root/opt/xilinx/xrt (docs/npu.md, "The XDNA stack").
#
# The kernel driver is not built: amdxdna ships in the kernel (drivers/accel).
# Build dependencies come from upstream's third_party/xdna-driver/tools/amdxdna_deps.sh.
set -euo pipefail
prefix=${1:?usage: build-xdna.sh <prefix> [jobs]}
jobs=${2:-$(nproc)}
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/third_party/xdna-driver
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

[ -f "$src/build/build.sh" ] || { echo "third_party/xdna-driver is empty: git submodule update --init third_party/xdna-driver"; exit 1; }
git -C "$src" submodule update --init --recursive --depth 1

# XRT, NPU package. Its OpenCL layer (xocl) is left out: the NPU does not use
# it, and it fails to compile where the distro's ocl_icd.h is newer than XRT's
# bundled OpenCL 1.2 headers.
(cd "$src/xrt/build" && ./build.sh -npu -opt -noctest -j "$jobs" \
    -cmake-flags "-DXRT_EXCLUDE_SUB_DIRECTORY=src/runtime_src/xocl")
stage=$prefix/root
xrt=$stage/opt/xilinx/xrt
DESTDIR=$stage cmake --install "$src/xrt/build/Release"

# The XDNA shim plugin, against that XRT; no kernel module.
(cd "$src/build" && XILINX_XRT="$xrt" ./build.sh -release -nokmod -j "$jobs")
DESTDIR=$stage cmake --install "$src/build/Release"
# XRT loads device plugins from its own lib directory; the shim installs under
# its build's prefix, so put it next to XRT.
find "$stage" -name 'libxrt_driver_xdna.so*' -not -path "$xrt/*" -exec cp -a {} "$xrt/lib/" \;

echo "XDNA stack (XRT root): $xrt"
ls "$xrt"/lib*/libxrt_coreutil.so* "$xrt"/lib*/libxrt_driver_xdna.so* 2>/dev/null
