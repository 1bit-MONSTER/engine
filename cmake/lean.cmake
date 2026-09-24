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

# Vulkan llama-server from upstream llama.cpp (docs/vulkan.md).
#
# The lean option (docs/lean.md): llama-server and llama-quantize from
#
#   third_party/llama.cpp-rocmfpx  charlie12345/ROCmFPX (MIT), a llama.cpp fork with
#                                  AMD-focused weight formats: ROCmFP4, ROCmI4, ...
#
# Upstream llama.cpp cannot read these formats, so the lean route has its own tree.
#   llama_lean       Vulkan build: ROCmFP4 files (`1bit serve --lean`)
#   llama_lean_rocm  ROCm build with the gfx1151 W4A4 path: ROCmI4 files
#                    (`1bit serve --lean --device rocm`), with ONEBIT_LEAN_ROCM
# The web UI is off: its build step downloads assets, and serve does not use it.
# The ROCm build forces llama.cpp's own MMQ kernels (GGML_CUDA_FORCE_MMQ): hipBLAS returns
# wrong GEMMs on gfx1151 (ROCm/rocm-libraries#11530), and MMQ-only measured more accurate at
# the same speed (Qwen3.8-27B UD-Q4_K_XL, KLD 0.0064 against 0.0094; docs/lean.md).
include(ExternalProject)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp-rocmfpx/CMakeLists.txt")
    message(FATAL_ERROR "ONEBIT_LEAN needs third_party/llama.cpp-rocmfpx: git submodule update --init third_party/llama.cpp-rocmfpx")
endif()

set(ONEBIT_LEAN_SERVER "${CMAKE_BINARY_DIR}/lean/llama/bin/llama-server")
set(ONEBIT_LEAN_QUANTIZE "${CMAKE_BINARY_DIR}/lean/llama/bin/llama-quantize")
ExternalProject_Add(llama_lean
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp-rocmfpx
    BINARY_DIR ${CMAKE_BINARY_DIR}/lean/llama
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
        -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_WEBUI=OFF -DLLAMA_CURL=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server llama-quantize llama-bench
    BUILD_BYPRODUCTS ${ONEBIT_LEAN_SERVER} ${ONEBIT_LEAN_QUANTIZE}
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    USES_TERMINAL_BUILD ON)

if(ONEBIT_LEAN_ROCM)
    # TheRock's amdclang++. Some installs pair it with the distribution's HIP headers
    # and fail on __ocml_*; point ONEBIT_LEAN_ROCM_TOOLCHAIN at a TheRock tree that works.
    set(ONEBIT_LEAN_ROCM_TOOLCHAIN "/opt/rocm-therock" CACHE PATH "TheRock root whose bin/amdclang++ builds the lean ROCm llama-server")
    set(ONEBIT_LEAN_ROCM_SERVER "${CMAKE_BINARY_DIR}/lean/llama-rocm/bin/llama-server")
    ExternalProject_Add(llama_lean_rocm
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp-rocmfpx
        BINARY_DIR ${CMAKE_BINARY_DIR}/lean/llama-rocm
        DOWNLOAD_COMMAND ""
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
            -DGGML_HIP=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON
            -DCMAKE_HIP_COMPILER=${ONEBIT_LEAN_ROCM_TOOLCHAIN}/bin/amdclang++
            -DCMAKE_PREFIX_PATH=/opt/rocm-therock
            -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGGML_HIP_ROCMI4_W4A4=ON
            -DGGML_CUDA_FORCE_MMQ=ON
            -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_WEBUI=OFF -DLLAMA_CURL=OFF
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server llama-bench
        BUILD_BYPRODUCTS ${ONEBIT_LEAN_ROCM_SERVER}
        INSTALL_COMMAND ""
        BUILD_ALWAYS ON
        USES_TERMINAL_BUILD ON)
endif()
