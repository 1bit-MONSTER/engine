#!/bin/bash
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
# runner/build.sh -- build the arena's host-side on-device scorer.
# Links the engine's own NPU runtime (pinned in TOOLCHAIN.lock); no XRT.
set -u
ARENA=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
. "$ARENA/lib/env.sh"
arena_require "$ENGINE_TREE/npu/rt.h" "$NPU_RT_LIB" || exit 1
out=${1:-$ARENA/runner/runner}
"${HOST_CXX:-/usr/bin/g++-16}" -O2 -std=c++20 -DONEBIT_NPU_RUNTIME_OWN \
    -I "$ENGINE_TREE/npu" -I "$(dirname "$NPU_RT_LIB")" \
    "$ARENA/runner/runner.cpp" "$NPU_RT_LIB" -o "$out" || exit 1
echo "built $out"
