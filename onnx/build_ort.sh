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
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Build ONNX Runtime from source and install it to a prefix we own, so onnx/ links against our
# libonnxruntime instead of the distro's. ORT is MIT: patching it and shipping the result closed
# source is allowed, and keeping the source tree is what makes the artefact ours.
#
#   PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" cmake -S onnx -B onnx/build
#
# Version is pinned to the one the pack documents, and so are the builtin EPs: --use_xnnpack and
# --use_dnnl give us the same three the distro build has (Xnnpack, Dnnl, CPU), because a session that
# can partition differently depending on which libonnxruntime it loaded is not a pinned build.
set -euo pipefail

# CMake 4 refuses cmake_minimum_required < 3.5, which oneDNN v3.0.1 (ORT's pinned DNNL_TAG) still
# declares; the environment variable initialises the cache in its external-project build tree.
export CMAKE_POLICY_VERSION_MINIMUM=3.5

VERSION="${ORT_VERSION:-v1.23.2}"
PREFIX="${ORT_PREFIX:-$HOME/vendor/onnxruntime-${VERSION#v}}"
SRC="$(dirname "$PREFIX")/src-${VERSION#v}"

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$VERSION" --recurse-submodules --shallow-submodules \
    https://github.com/microsoft/onnxruntime "$SRC"
fi

# GCC 15: semver.h uses uint32_t without <cstdint>. Guarded, so a re-run does not add it twice.
if ! grep -q '#include <cstdint>' "$SRC/onnxruntime/core/common/semver.h"; then
  sed -i 's/^#include <optional>$/#include <cstdint>\n#include <optional>/' \
    "$SRC/onnxruntime/core/common/semver.h"
fi
grep -q '#include <cstdint>' "$SRC/onnxruntime/core/common/semver.h"

cd "$SRC"
# onnxruntime_BUILD_UNIT_TESTS=OFF: --skip_tests only skips *running* them, not compiling 400+ TUs.
"$SRC/build.sh" --config Release --build_shared_lib --parallel "$(nproc)" --skip_tests \
  --use_xnnpack --use_dnnl --compile_no_warning_as_error \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
  --cmake_extra_defines "CMAKE_INSTALL_PREFIX=$PREFIX"
cmake --install build/Linux/Release --prefix "$PREFIX"

# oneDNN is an external project with its own install prefix, and the DNNL EP links libdnnl.so.3 by
# soname: without it beside libonnxruntime the provider .so loads and the EP is simply not registered.
cp -a build/Linux/Release/dnnl/install/lib/libdnnl.so* "$PREFIX/lib/"

echo
echo "installed: $PREFIX/lib/libonnxruntime.so.1.23.2"
echo "use it:    export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig"
echo "run it:    export LD_LIBRARY_PATH=$PREFIX/lib"
