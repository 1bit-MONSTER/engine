#!/bin/bash
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
