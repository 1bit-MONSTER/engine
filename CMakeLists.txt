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
cmake_minimum_required(VERSION 3.24)
project(onebit_engine VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

# Components are ported in the order listed in docs/PORTING.md.

# ── Step 1: Lemonade's server core, embedded ───────────────────────────────
# Vendored at third_party/lemonade (v11.9.0 plus the local deltas listed in
# its UPSTREAM.md, including the `onebit` backend).
set(BUILD_WEB_APP OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/lemonade EXCLUDE_FROM_ALL)

add_executable(onebit app/main.cpp)
set_target_properties(onebit PROPERTIES OUTPUT_NAME 1bit)
target_link_libraries(onebit PRIVATE lemonade-server-core)

# ── Step 2: HRX + Vulkan in one llama.cpp build (docs/hrx.md) ─────────────
# Off by default: it needs TheRock/ROCm and the three HRX submodules, and it
# only runs on AMD GPUs. `1bit lemonade` then uses this llama-server for both
# the llamacpp-hrx recipe (HRX20) and the llamacpp recipe's Vulkan backend.
option(ONEBIT_HRX "Build HRX2 + Vulkan llama-server and wire it into Lemonade" OFF)
if(ONEBIT_HRX)
    include(cmake/hrx.cmake)
    add_dependencies(onebit llama_hrx)
    target_compile_definitions(onebit PRIVATE ONEBIT_HRX_SERVER="${ONEBIT_HRX_SERVER}")
    add_test(NAME hrx_lemonade_e2e
             COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_lemonade_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_HRX_SERVER})
endif()

# ── Step 3: NPU engine, full ELFs only (docs/npu.md) ──────────────────────
# 3a: full-ELF generation. Pure C++, no XRT, so it builds and tests anywhere.
add_library(onebit_npu STATIC npu/full_elf.cpp)
target_include_directories(onebit_npu PUBLIC npu)

add_executable(npu_full_elf_test tests/npu_full_elf_test.cpp)
target_link_libraries(npu_full_elf_test PRIVATE onebit_npu)
add_test(NAME npu_full_elf_md5 COMMAND npu_full_elf_test)
# Against a model's kernel artifacts (docs/npu.md, "Checking the generator").
set(ONEBIT_NPU_CAPTURED "" CACHE PATH "Directory of captured layer_ctx<N>.elf instruction ELFs")
set(ONEBIT_NPU_PDI "" CACHE FILEPATH "The layer kernel's design PDI")
set(ONEBIT_NPU_GOLDEN "" CACHE PATH "Directory of full ELFs the fast lane ran")
if(ONEBIT_NPU_CAPTURED AND ONEBIT_NPU_PDI AND ONEBIT_NPU_GOLDEN)
    add_test(NAME npu_full_elf_model
             COMMAND npu_full_elf_test ${ONEBIT_NPU_CAPTURED} ${ONEBIT_NPU_PDI} ${ONEBIT_NPU_GOLDEN})
endif()

add_test(NAME smoke_lemonade COMMAND ${CMAKE_SOURCE_DIR}/tests/smoke_lemonade.sh $<TARGET_FILE:onebit>)
