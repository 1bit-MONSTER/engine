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

# The HRX llama.cpp build (docs/hrx.md). Two pinned sources, the pair AMD's integration repo
# (ROCm/ggml-staging-automation) builds and tests together:
#
#   third_party/llama.cpp   1bit-MONSTER/llama.cpp 1bit/hrx-vulkan-patched: AMD's
#                           hrx-graph-develop-v2 (ggml-hrx) on ggml-org llama.cpp, plus ours
#   third_party/hrx-system  ROCm/hrx-system: libhrx, loomc and the Loom tools
#
# llama.cpp's ggml-hrx builds hrx-system itself when given HRX_SOURCE_DIR, with
# the compilers of this build, so the only inputs are the two trees and TheRock.
# IREE_ROCM_PATH is deliberately not passed: it switches hrx-system to "package"
# mode, which requires TheRock's own aqlprofile-sdk headers (absent from our
# /opt/rocm-therock); unset, hrx-system fetches its pinned HSA/AQL headers.
# .github/workflows/bump-hrx.yml moves both pins when AMD moves its pair.
# Only HRX0 and the CPU are built: no Vulkan, ROCm/HIP or CUDA backend (RFC #213 stage 3). The
# branch name still says "vulkan" (AMD's pair carried both); the engine no longer builds it.
include(ExternalProject)

set(ONEBIT_HRX_TOOLCHAIN "/opt/rocm-therock" CACHE PATH "ROCm/TheRock root: its amdclang builds llama.cpp and HRX")

foreach(_sub hrx-system llama.cpp)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/${_sub}/CMakeLists.txt")
        message(FATAL_ERROR "ONEBIT_HRX needs third_party/${_sub}: git submodule update --init third_party/${_sub}")
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

# Private GPU kernels (docs/hrx.md, "Private HIP kernels"): a checkout of the private gpu-kernels
# repository whose addons/hip-hrx is built into the HRX llama.cpp (its GGML_HRX_HIP_ADDON_DIR hook).
set(_hrx_gpu_private_args)
if(ONEBIT_GPU_PRIVATE)
    if(NOT IS_DIRECTORY "${ONEBIT_GPU_PRIVATE}/addons/hip-hrx/kernels")
        message(FATAL_ERROR "ONEBIT_GPU_PRIVATE=${ONEBIT_GPU_PRIVATE} has no addons/hip-hrx/kernels")
    endif()
    set(_hrx_hip_cmake "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/src/ggml-hrx/hip/ggml-hrx-hip.cmake")
    set(_hrx_addon_hook)
    if(EXISTS "${_hrx_hip_cmake}")
        file(STRINGS "${_hrx_hip_cmake}" _hrx_addon_hook REGEX "GGML_HRX_HIP_ADDON_DIR")
    endif()
    if(NOT _hrx_addon_hook)
        message(FATAL_ERROR "ONEBIT_GPU_PRIVATE: third_party/llama.cpp has no GGML_HRX_HIP_ADDON_DIR hook (needs the HIP plumbing)")
    endif()
    list(APPEND _hrx_gpu_private_args "-DGGML_HRX_HIP_ADDON_DIR=${ONEBIT_GPU_PRIVATE}/addons/hip-hrx")
    message(STATUS "ONEBIT_HRX: private HIP kernels from ${ONEBIT_GPU_PRIVATE}/addons/hip-hrx")
endif()

set(ONEBIT_HRX_SERVER "${CMAKE_BINARY_DIR}/hrx/llama/bin/llama-server")
ExternalProject_Add(llama_hrx
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
    BINARY_DIR ${CMAKE_BINARY_DIR}/hrx/llama
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang
        -DCMAKE_CXX_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang++
        -DHRX_SOURCE_DIR=${CMAKE_SOURCE_DIR}/third_party/hrx-system
        -DGGML_HRX=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
        ${_hrx_gpu_private_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server llama-bench
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    USES_TERMINAL_BUILD ON)
