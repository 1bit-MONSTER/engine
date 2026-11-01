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
# build-ds4.sh <prefix>
#
# Builds DwarfStar pinned in third_party/ds4 (upstream antirez/ds4) on its HIP backend
# (Strix Halo, gfx1151; make strix-halo) into <prefix>/hip/ds4-server, with ds4 and
# ds4-bench beside it. The engine builds DwarfStar on HIP only: no CUDA, Metal or CPU
# builds. Needs HIP, hipBLAS, hipBLASLt, rocBLAS, rocWMMA and hipCUB: ROCM_PATH, else
# TheRock's SDK under /opt/rocm-therock, else /opt/rocm. ROCM_ARCH overrides gfx1151.
# DS4_TEST=1 also runs DwarfStar's model-free routed-MoE test on the GPU (make test-mxfp4-rocm).
set -euo pipefail
prefix=${1:?usage: build-ds4.sh <prefix>}
backend=hip
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/third_party/ds4
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

[ -f "$src/ds4_server.c" ] || { echo "third_party/ds4 is empty: git submodule update --init third_party/ds4"; exit 1; }

out=$prefix/$backend
# Reuse a ds4-server already staged at the prefix, unless DS4_REBUILD=1 asks for a fresh build. A CMake
# custom target always runs its command, so on a box whose ROCm cannot compile these kernels (the
# 2026-10-31 rebuild: no hipCUB/rocPRIM/rocWMMA headers) an already-present binary built from this pin
# turned into a hard build failure for the whole engine - `1bit` depends on this target. Checked before
# the ROCm lookup and before make, so the reuse works on a box with no usable hipcc at all.
# The pin is printed so the record says what the binary is.
# All three binaries, and the pin the staged ones were built from (`ds4.pin`, written below). `$src`'s HEAD is the
# pin of the tree being built *today*, not of the binary that is already there, so the two are compared rather than
# printed as if they were the same fact (the same defect #720 fixed for hrx-system).
if [ "${DS4_REBUILD:-0}" != 1 ] && [ -x "$out/ds4-server" ] && [ -x "$out/ds4" ] && [ -x "$out/ds4-bench" ]; then
    staged=$(cat "$out/ds4.pin" 2>/dev/null || echo unknown)
    now=$(git -C "$src" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
    if [ "$staged" = "$now" ]; then
        echo "DwarfStar ($backend, $staged): reusing $out/ds4-server"
        exit 0
    fi
    echo "DwarfStar: staged $staged differs from pin $now - rebuilding"
fi

work=$prefix/src/$backend
mkdir -p "$work"
# copy the tree, keeping objects from an earlier build of the same backend
(cd "$src" && tar --exclude=.git -cf - .) | (cd "$work" && tar -xf -)
jobs=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}   # the weekly job and thermal guards cap it

rocm=${ROCM_PATH:-}
if [ -z "$rocm" ]; then
    for d in /opt/rocm-therock/lib/python3*/site-packages/_rocm_sdk_devel /opt/rocm; do
        [ -x "$d/bin/hipcc" ] && { rocm=$d; break; }
    done
fi
[ -n "$rocm" ] && [ -x "$rocm/bin/hipcc" ] || { echo "no ROCm with bin/hipcc: set ROCM_PATH"; exit 1; }
# -isystem: clang otherwise appends the ROCm include directory after /usr/include,
# so a distro HIP there (another version) shadows this one and the build fails
cflags="-O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=${ROCM_ARCH:-gfx1151} -isystem $rocm/include"
libs="-L$rocm/lib -Wl,-rpath,$rocm/lib -lm -pthread -lhipblas -lhipblaslt -lrocblas"
(cd "$work" && ROCM_PATH=$rocm HIP_PATH=$rocm make -j"$jobs" strix-halo \
    HIPCC="$rocm/bin/hipcc" ROCM_CFLAGS="$cflags" ROCM_LDLIBS="$libs")
if [ "${DS4_TEST:-0}" = 1 ]; then
    (cd "$work" && ROCM_PATH=$rocm HIP_PATH=$rocm make test-mxfp4-rocm \
        HIPCC="$rocm/bin/hipcc" ROCM_CFLAGS="$cflags" ROCM_LDLIBS="$libs")
fi

mkdir -p "$out"
for b in ds4-server ds4 ds4-bench; do cp "$work/$b" "$out/$b"; done
commit=$(git -C "$src" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
echo "$commit" > "$out/ds4.pin"
echo "DwarfStar ($backend, $commit): $out/ds4-server"
