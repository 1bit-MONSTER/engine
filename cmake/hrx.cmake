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

# HRX + Vulkan in one llama.cpp build (docs/hrx.md). Two pinned sources, the pair
# AMD's integration repo (ROCm/ggml-staging-automation) builds and tests together:
#
#   third_party/llama.cpp   AMD-Ecosystem/llama.cpp hrx-graph-develop-v2: ggml-hrx
#                           on ggml-org llama.cpp
#   third_party/hrx-system  ROCm/hrx-system: libhrx, loomc and the Loom tools
#
# -DONEBIT_GPU_PRIVATE swaps both for the private trees (CMakeLists.txt): the paths
# below are ONEBIT_GPU_LLAMA_SOURCE and ONEBIT_GPU_HRX_SYSTEM_SOURCE.
#
# llama.cpp's ggml-hrx builds hrx-system itself when given HRX_SOURCE_DIR, with
# the compilers of this build, so the only inputs are the two trees and TheRock.
# IREE_ROCM_PATH is deliberately not passed: it switches hrx-system to "package"
# mode, which requires TheRock's own aqlprofile-sdk headers (absent from our
# /opt/rocm-therock); unset, hrx-system fetches its pinned HSA/AQL headers.
# .github/workflows/bump-hrx.yml moves both pins when AMD moves its pair.
include(ExternalProject)

set(ONEBIT_HRX_TOOLCHAIN "/opt/rocm-therock" CACHE PATH "ROCm/TheRock root: its amdclang builds llama.cpp and HRX")

foreach(_src ${ONEBIT_GPU_HRX_SYSTEM_SOURCE} ${ONEBIT_GPU_LLAMA_SOURCE})
    if(NOT EXISTS "${_src}/CMakeLists.txt")
        message(FATAL_ERROR "ONEBIT_HRX needs ${_src}: git submodule update --init --depth 1 in its repository")
    endif()
endforeach()

# HRX dlopens the HSA runtime. The distro's libhsa-runtime64 rejects the
# HSA_AMD_AGENT_INFO_PM4_EMULATION probe on gfx1151, so HRX registers no device;
# TheRock's answers it. `1bit lemonade` passes this path to llama-server as
# IREE_HAL_AMDGPU_LIBHSA_PATH (docs/hrx.md).
file(GLOB_RECURSE _hrx_libhsa "${ONEBIT_HRX_TOOLCHAIN}/*/libhsa-runtime64.so.1")
list(SORT _hrx_libhsa)
list(GET _hrx_libhsa 0 _hrx_libhsa_first)
if(NOT _hrx_libhsa_first)
    message(FATAL_ERROR "ONEBIT_HRX: no libhsa-runtime64.so.1 under ONEBIT_HRX_TOOLCHAIN=${ONEBIT_HRX_TOOLCHAIN}")
endif()
set(ONEBIT_HRX_LIBHSA "${_hrx_libhsa_first}" CACHE FILEPATH "TheRock HSA runtime HRX loads (IREE_HAL_AMDGPU_LIBHSA_PATH)")
message(STATUS "ONEBIT_HRX: HSA runtime ${ONEBIT_HRX_LIBHSA}")

set(ONEBIT_HRX_SERVER "${CMAKE_BINARY_DIR}/hrx/llama/bin/llama-server")
ExternalProject_Add(llama_hrx
    SOURCE_DIR ${ONEBIT_GPU_LLAMA_SOURCE}
    BINARY_DIR ${CMAKE_BINARY_DIR}/hrx/llama
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang
        -DCMAKE_CXX_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang++
        -DHRX_SOURCE_DIR=${ONEBIT_GPU_HRX_SYSTEM_SOURCE}
        -DGGML_HRX=ON -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server llama-bench
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    USES_TERMINAL_BUILD ON)
