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

# The router's fast scorer (docs/laya.md): ggmlc's `laya` (third_party/ggmlc, our fork, MIT) runs
# the Laya GGUFs on HRX0, AMD's backend: it builds third_party/llama.cpp's ggml (ggml-hrx, with the
# kernels ModernBERT needs) with TheRock's amdclang, like cmake/hrx.cmake. `1bit serve --laya`
# starts it as a child when this binary and the pinned GGUF (scripts/fetch-laya.sh) are both there,
# else it uses the C++ scorer on the safetensors.
include(ExternalProject)

foreach(_sub ggmlc llama.cpp hrx-system)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/${_sub}/CMakeLists.txt")
        message(FATAL_ERROR "ONEBIT_LAYA_GGML needs third_party/${_sub}: git submodule update --init third_party/${_sub}")
    endif()
endforeach()
set(ONEBIT_HRX_TOOLCHAIN "/opt/rocm-therock" CACHE PATH "ROCm/TheRock root: its amdclang builds llama.cpp and HRX")

set(ONEBIT_LAYA_GGML_BIN "${CMAKE_BINARY_DIR}/laya-ggml/examples/laya/laya")
ExternalProject_Add(laya_ggml
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/ggmlc
    BINARY_DIR ${CMAKE_BINARY_DIR}/laya-ggml
    DOWNLOAD_COMMAND ""
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang
        -DCMAKE_CXX_COMPILER=${ONEBIT_HRX_TOOLCHAIN}/bin/amdclang++
        -DGGMLC_ENABLE_HRX=ON -DGGMLC_GGML_DIR=${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml
        -DHRX_SOURCE_DIR=${CMAKE_SOURCE_DIR}/third_party/hrx-system
        -DGGML_NATIVE=ON -DGGMLC_BUILD_EXAMPLES=ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target laya
    BUILD_BYPRODUCTS ${ONEBIT_LAYA_GGML_BIN}
    INSTALL_COMMAND ""
    BUILD_ALWAYS ON
    USES_TERMINAL_BUILD ON)
