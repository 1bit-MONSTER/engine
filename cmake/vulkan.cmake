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
#   third_party/llama.cpp-vulkan  ggml-org/llama.cpp, pinned to upstream's
#                                 latest release
#
# Kept apart from third_party/llama.cpp, which stays on the llama.cpp +
# hrx-system pair AMD tests for HRX: new architectures (Qwen3.8-Flash-Next's
# qwen4exp, say) reach the Vulkan route the day upstream releases them, without
# waiting for AMD's pair to move. .github/workflows/bump-llama-vulkan.yml moves
# the pin when upstream releases. Needs the Vulkan headers/loader and glslc.
include(ExternalProject)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/llama.cpp-vulkan/CMakeLists.txt")
    message(FATAL_ERROR "ONEBIT_VULKAN needs third_party/llama.cpp-vulkan: git submodule update --init third_party/llama.cpp-vulkan")
endif()

set(ONEBIT_VULKAN_SERVER "${CMAKE_BINARY_DIR}/vulkan/llama/bin/llama-server")
ExternalProject_Add(llama_vulkan
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp-vulkan
    BINARY_DIR ${CMAKE_BINARY_DIR}/vulkan/llama
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
        -DGGML_VULKAN=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_NATIVE=ON -DGGML_CPU=ON
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server llama-bench
    BUILD_BYPRODUCTS ${ONEBIT_VULKAN_SERVER}
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    USES_TERMINAL_BUILD ON)
