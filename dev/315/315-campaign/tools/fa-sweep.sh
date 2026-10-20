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
# Sweep the standalone FA-race harness against the pin's build over the shapes that matter for engine#315.
#
# fa-race <T> <cells> <visible> <reps>: repeat one prefill FLASH_ATTN_EXT on HRX0 with identical inputs and
# count runs whose output differs from run 0, grouped by (query tile, head, 16-channel block). Cells/visible
# cover the arms' regime (cells 2048 = -c 2048, visible ~810 = the faulting prompts) and the shape the Oct 5
# investigation found racing (cells 4864, visible 4798, T 190), which the pin no longer reproduces.
set -u
HSA=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
D=${D:-$HOME/wt/315fp-build}
DEPS=$D/ggml/src/ggml-hrx/hrx/src/ggml-hrx-deps-build
LP="$D/bin:$DEPS/libhrx/src/libhrx:$DEPS/loom/binding/c"
CACHE=${CACHE:-/tmp/fa-sweep-cache}
mkdir -p "$CACHE"
REPS=${REPS:-250}
cd "$HOME/.cache/kvnan"

printf "%-8s %-8s %-8s %-6s %s\n" T cells visible reps result
for shape in "$@"; do
    set -- $shape
    T=$1; CELLS=$2; VIS=$3
    out=$(LD_LIBRARY_PATH="$LP" IREE_HAL_AMDGPU_LIBHSA_PATH="$HSA" GGML_HRX_JIT_CACHE_DIR="$CACHE" \
          timeout 300 ./fa-race "$T" "$CELLS" "$VIS" "$REPS" 2>&1)
    sum=$(echo "$out" | grep -E "bad_runs" | tail -1)
    cs=$(echo "$out" | grep -E "checksum" | tail -1 | awk '{print $2}')
    bad=$(echo "$sum" | grep -oE "bad_runs=[0-9]+" | cut -d= -f2)
    if [ -z "$sum" ]; then
        printf "%-8s %-8s %-8s %-6s ERROR: %s\n" "$T" "$CELLS" "$VIS" "$REPS" "$(echo "$out" | tail -1 | cut -c1-70)"
    else
        flag=""
        [ "${bad:-0}" != "0" ] && flag="   <-- RACE"
        printf "%-8s %-8s %-8s %-6s checksum=%s bad_runs=%s%s\n" "$T" "$CELLS" "$VIS" "$REPS" "$cs" "$bad" "$flag"
        [ "${bad:-0}" != "0" ] && echo "$out" | grep "differ" | head -3 | sed 's/^/         /'
    fi
done
echo SWEEP-DONE
