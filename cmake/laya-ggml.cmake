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

# The router's fast scorer (docs/laya.md): ggmlc's `laya` (third_party/ggmlc, MIT) runs the
# Laya GGUFs on GGML's Vulkan backend. `1bit serve --laya` starts it as a child when this
# binary and the pinned GGUF (scripts/fetch-laya.sh) are both there, else it uses the C++
# scorer on the safetensors.
include(ExternalProject)

if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/ggmlc/CMakeLists.txt")
    message(FATAL_ERROR "ONEBIT_LAYA_GGML needs third_party/ggmlc: git submodule update --init third_party/ggmlc")
endif()

set(ONEBIT_LAYA_GGML_BIN "${CMAKE_BINARY_DIR}/laya-ggml/examples/laya/laya")
ExternalProject_Add(laya_ggml
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/ggmlc
    BINARY_DIR ${CMAKE_BINARY_DIR}/laya-ggml
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DGGMLC_ENABLE_VULKAN=ON -DGGMLC_BUILD_EXAMPLES=ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target laya
    BUILD_BYPRODUCTS ${ONEBIT_LAYA_GGML_BIN}
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    USES_TERMINAL_BUILD ON)
