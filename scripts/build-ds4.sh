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
# build-ds4.sh <prefix> [backend]
#
# Builds DwarfStar pinned in third_party/ds4 (upstream antirez/ds4) into
# <prefix>/<backend>/ds4-server (with ds4 and ds4-bench beside it). backend is
# one of rocm (default on Linux), cuda, metal (macOS), cpu:
#   rocm   Strix Halo / gfx1151 (make strix-halo). Needs HIP, hipBLAS, hipBLASLt,
#          rocBLAS, rocWMMA and hipCUB: ROCM_PATH, else TheRock's SDK under
#          /opt/rocm-therock, else /opt/rocm
#   cuda   CUDA_ARCH (default native: make cuda-generic)
#   metal  Apple Silicon (plain make on macOS)
#   cpu    CPU only
#
# DwarfStar builds in its source tree, so the build runs on a copy under
# <prefix>/src and the submodule stays clean (docs/dwarfstar.md). DS4_TEST=1 then runs
# DwarfStar's model-free routed-MoE test on the GPU (rocm only: make test-mxfp4-rocm).
set -euo pipefail
prefix=${1:?usage: build-ds4.sh <prefix> [rocm|cuda|metal|cpu]}
if [ "$(uname -s)" = Darwin ]; then default=metal; else default=rocm; fi
backend=${2:-$default}
case "$backend" in rocm|cuda|metal|cpu) ;; *) echo "unknown backend: $backend"; exit 1 ;; esac
root=$(cd "$(dirname "$0")/.." && pwd)
src=$root/third_party/ds4
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd)

[ -f "$src/ds4_server.c" ] || { echo "third_party/ds4 is empty: git submodule update --init third_party/ds4"; exit 1; }

work=$prefix/src/$backend
mkdir -p "$work"
# copy the tree, keeping objects from an earlier build of the same backend
(cd "$src" && tar --exclude=.git -cf - .) | (cd "$work" && tar -xf -)
jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu)

case "$backend" in
rocm)
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
    ;;
cuda)  (cd "$work" && make -j"$jobs" cuda CUDA_ARCH="${CUDA_ARCH:-native}") ;;
metal) (cd "$work" && make -j"$jobs") ;;
cpu)   (cd "$work" && make -j"$jobs" cpu) ;;
esac

out=$prefix/$backend
mkdir -p "$out"
for b in ds4-server ds4 ds4-bench; do cp "$work/$b" "$out/$b"; done
commit=$(git -C "$src" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
echo "DwarfStar ($backend, $commit): $out/ds4-server"
