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

add_executable(onebit app/main.cpp app/serve.cpp)
set_target_properties(onebit PROPERTIES OUTPUT_NAME 1bit)
target_link_libraries(onebit PRIVATE lemonade-server-core)

# ── `1bit serve` end to end (docs/serve.md) ───────────────────────────────
# Needs a GGUF (e.g. Qwen3-0.6B Q4_K_M); tests each device this build has.
set(ONEBIT_SERVE_TEST_GGUF "" CACHE FILEPATH "GGUF the serve_e2e tests load")

# ── Step 2: HRX + Vulkan in one llama.cpp build (docs/hrx.md) ─────────────
# Off by default: it needs TheRock/ROCm and the three HRX submodules, and it
# only runs on AMD GPUs. `1bit lemonade` then uses this llama-server for both
# the llamacpp-hrx recipe (HRX0) and the llamacpp recipe's Vulkan backend.
option(ONEBIT_HRX "Build HRX + Vulkan llama-server and wire it into Lemonade" OFF)
if(ONEBIT_HRX)
    include(cmake/hrx.cmake)
    add_dependencies(onebit llama_hrx)
    target_compile_definitions(onebit PRIVATE ONEBIT_HRX_SERVER="${ONEBIT_HRX_SERVER}" ONEBIT_HRX_LIBHSA="${ONEBIT_HRX_LIBHSA}")
    if(ONEBIT_SERVE_TEST_GGUF)
        foreach(_dev vulkan hrx)
            add_test(NAME serve_e2e_${_dev}
                     COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_SERVE_TEST_GGUF} ${_dev})
        endforeach()
    endif()
    add_test(NAME hrx_lemonade_e2e
             COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_lemonade_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_HRX_SERVER})
endif()

# ── ZINC (docs/zinc.md) ────────────────────────────────────────────────────
# Off by default: it builds third_party/zinc with scripts/build-zinc.sh (which
# fetches its own Zig) for one GPU backend. `1bit lemonade` then serves the zinc
# recipe with this binary. ONEBIT_ZINC_BACKEND=cuda is the NVIDIA build.
option(ONEBIT_ZINC "Build ZINC (third_party/zinc) and wire it into Lemonade" OFF)
if(ONEBIT_ZINC)
    set(ONEBIT_ZINC_BACKEND "vulkan" CACHE STRING "ZINC GPU backend: vulkan, rocm or cuda")
    set_property(CACHE ONEBIT_ZINC_BACKEND PROPERTY STRINGS vulkan rocm cuda)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/zinc/build.zig")
        message(FATAL_ERROR "ONEBIT_ZINC needs third_party/zinc: git submodule update --init third_party/zinc")
    endif()
    set(ONEBIT_ZINC_SERVER "${CMAKE_BINARY_DIR}/zinc/${ONEBIT_ZINC_BACKEND}/bin/zinc")
    add_custom_target(zinc
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/build-zinc.sh ${CMAKE_BINARY_DIR}/zinc ${ONEBIT_ZINC_BACKEND}
        BYPRODUCTS ${ONEBIT_ZINC_SERVER}
        USES_TERMINAL)
    add_dependencies(onebit zinc)
    target_compile_definitions(onebit PRIVATE ONEBIT_ZINC_SERVER="${ONEBIT_ZINC_SERVER}")
    if(ONEBIT_SERVE_TEST_GGUF)
        add_test(NAME serve_e2e_zinc
                 COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_SERVE_TEST_GGUF} zinc)
    endif()
    add_test(NAME zinc_lemonade_e2e
             COMMAND ${CMAKE_SOURCE_DIR}/tests/zinc_lemonade_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_ZINC_SERVER})
endif()

# ── Hugging Face tokenizers (docs/tokenizers.md) ───────────────────────────
# Off by default: it needs cargo. Any model's tokenizer.json, through our C ABI
# over third_party/tokenizers (the same release as the tokenizers npm/PyPI
# packages). tests/data/tokenizers/ holds golden cases for 18 models.
option(ONEBIT_HF_TOKENIZERS "Build Hugging Face tokenizers (Rust) behind a C ABI" OFF)
if(ONEBIT_HF_TOKENIZERS)
    find_package(Threads REQUIRED)
    include(cmake/hf_tokenizers.cmake)
    add_executable(hf_tokenizer_test tests/hf_tokenizer_test.cpp)
    target_link_libraries(hf_tokenizer_test PRIVATE onebit_hf_tokenizers nlohmann_json::nlohmann_json)
    # Golden cases whose tokenizer.json is under a model root (ONEBIT_TOKENIZER_ROOT,
    # e.g. ~/.config/flm/models: <root>/<model>/tokenizer.json).
    set(ONEBIT_TOKENIZER_ROOT "" CACHE PATH "Directory of model dirs holding tokenizer.json, for hf_tokenizer tests")
    if(ONEBIT_TOKENIZER_ROOT)
        file(GLOB _tok_golden "${CMAKE_SOURCE_DIR}/tests/data/tokenizers/*.json")
        foreach(_g ${_tok_golden})
            get_filename_component(_m "${_g}" NAME_WLE)
            if(EXISTS "${ONEBIT_TOKENIZER_ROOT}/${_m}/tokenizer.json")
                add_test(NAME hf_tokenizer_${_m}
                         COMMAND hf_tokenizer_test ${ONEBIT_TOKENIZER_ROOT}/${_m}/tokenizer.json ${_g})
            endif()
        endforeach()
    endif()
endif()

# ── Step 3: NPU engine, full ELFs only (docs/npu.md) ──────────────────────
# Host side (full-ELF generation, model reader, buffer packing): pure C++, no
# XRT. It needs PCRE2 (the tokenizer), and macOS has no NPU, so it is off there
# by default; the Lemonade face (including the mlx backend) does not use it.
if(APPLE)
    set(_onebit_npu_host_default OFF)
else()
    set(_onebit_npu_host_default ON)
endif()
option(ONEBIT_NPU_HOST "Build the NPU host library and its tests" ${_onebit_npu_host_default})
if(ONEBIT_NPU_HOST)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
add_library(onebit_npu STATIC npu/full_elf.cpp npu/model.cpp npu/pack.cpp npu/tokenizer.cpp)
target_include_directories(onebit_npu PUBLIC npu)
target_link_libraries(onebit_npu PUBLIC nlohmann_json::nlohmann_json PRIVATE PkgConfig::PCRE2)

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

add_executable(npu_pack_test tests/npu_pack_test.cpp)
target_link_libraries(npu_pack_test PRIVATE onebit_npu)
set(ONEBIT_NPU_MODEL "" CACHE PATH "Q4NX model directory (model.q4nx + config.json)")
set(ONEBIT_NPU_BO_DUMP "" CACHE PATH "The reference lane's buffer dump (RT_DUMP_BOS)")
add_executable(tokenizer_test tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE onebit_npu)
if(ONEBIT_NPU_MODEL)
    add_test(NAME tokenizer_qwen3
             COMMAND tokenizer_test ${ONEBIT_NPU_MODEL}/tokenizer.json ${CMAKE_SOURCE_DIR}/tests/data/qwen3_tokenizer_golden.json)
endif()
if(ONEBIT_NPU_MODEL AND ONEBIT_NPU_BO_DUMP)
    add_test(NAME npu_pack_model COMMAND npu_pack_test ${ONEBIT_NPU_MODEL} ${ONEBIT_NPU_BO_DUMP})
endif()
endif()  # ONEBIT_NPU_HOST

# The lane itself: needs XRT (the NPU runtime) and an XDNA NPU to run.
option(ONEBIT_NPU "Build the NPU fast lane (needs XRT)" OFF)
set(ONEBIT_XRT_ROOT "/opt/xilinx/xrt" CACHE PATH "XRT install prefix")
if(ONEBIT_NPU AND NOT ONEBIT_NPU_HOST)
    message(FATAL_ERROR "ONEBIT_NPU needs ONEBIT_NPU_HOST=ON")
endif()
if(ONEBIT_NPU)
    if(NOT EXISTS "${ONEBIT_XRT_ROOT}/include/xrt/xrt_device.h")
        message(FATAL_ERROR "ONEBIT_NPU needs XRT; none at ONEBIT_XRT_ROOT=${ONEBIT_XRT_ROOT}")
    endif()
    add_library(onebit_npu_lane STATIC npu/lane.cpp npu/generate.cpp)
    target_include_directories(onebit_npu_lane SYSTEM PUBLIC ${ONEBIT_XRT_ROOT}/include)
    target_link_libraries(onebit_npu_lane PUBLIC onebit_npu ${ONEBIT_XRT_ROOT}/lib/libxrt_coreutil.so)
    target_sources(onebit PRIVATE app/unified.cpp)
    target_link_libraries(onebit PRIVATE onebit_npu_lane)
    target_compile_definitions(onebit PRIVATE ONEBIT_NPU)
    set_target_properties(onebit PROPERTIES BUILD_RPATH ${ONEBIT_XRT_ROOT}/lib INSTALL_RPATH ${ONEBIT_XRT_ROOT}/lib)
    set(ONEBIT_NPU_KERNELS "" CACHE PATH "Kernel directory (layer_ctx{1,2,17}.elf, lmhead.elf, layer.pdi)")
    set(ONEBIT_NPU_REFERENCE "" CACHE PATH "Reference lane's logits for the anchor prompt")
    if(ONEBIT_NPU_MODEL AND ONEBIT_NPU_KERNELS AND ONEBIT_NPU_REFERENCE)
        add_test(NAME npu_lane_e2e
                 COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_lane_e2e.sh $<TARGET_FILE:onebit>
                         ${ONEBIT_NPU_MODEL} ${ONEBIT_NPU_KERNELS} ${ONEBIT_NPU_REFERENCE})
    endif()
endif()

add_test(NAME smoke_lemonade COMMAND ${CMAKE_SOURCE_DIR}/tests/smoke_lemonade.sh $<TARGET_FILE:onebit>)

# MLX on Apple Silicon through Lemonade's mlx backend (docs/apple.md).
set(ONEBIT_MLX_SERVER "" CACHE FILEPATH "lemon-mlx-engine's server binary, for tests/mlx_lemonade_e2e.sh")
if(APPLE AND ONEBIT_MLX_SERVER)
    add_test(NAME mlx_lemonade_e2e
             COMMAND ${CMAKE_SOURCE_DIR}/tests/mlx_lemonade_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_MLX_SERVER})
endif()
