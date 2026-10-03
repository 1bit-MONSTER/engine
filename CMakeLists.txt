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

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

enable_testing()

# Components are ported in the order listed in docs/PORTING.md.

# ── The engine binary: `1bit serve`, an OpenAI-compatible API (docs/serve.md)
# The engine is a backend a host such as Lemonade launches; it no longer embeds
# one. JSON and the HTTP server/client are pinned here, at the commits the
# engine used before (nlohmann/json v3.11.3, cpp-httplib v0.47.0).
include(FetchContent)
if(WIN32)
    # Windows 10 or later: cpp-httplib's floor; the headers leave out min/max macros
    add_compile_definitions(_WIN32_WINNT=0x0A00 WINVER=0x0A00 NOMINMAX WIN32_LEAN_AND_MEAN)
    # MinGW's headers lack GetAddrInfoExCancel; the engine only ever connects to 127.0.0.1
    set(HTTPLIB_USE_NON_BLOCKING_GETADDRINFO OFF CACHE BOOL "" FORCE)
    if(MINGW)
        # one self-contained 1bit.exe: the C++ runtime linked in, not beside it as DLLs
        add_link_options(-static)
    endif()
endif()
find_package(Threads REQUIRED)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03)  # v3.11.3
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG fe332fa06bac76a1c6d402c08f414052999347da)  # v0.47.0
FetchContent_MakeAvailable(json httplib)

add_executable(onebit app/main.cpp app/serve.cpp app/recipes.cpp)
# config/recipes.json (tuned llama-server settings per model and route, docs/recipes.md) is
# compiled in as serve's default recipes; --recipes replaces them at run time.
file(READ ${CMAKE_SOURCE_DIR}/config/recipes.json ONEBIT_RECIPES_JSON)
file(WRITE ${CMAKE_BINARY_DIR}/generated/recipes_default.h
     "// generated from config/recipes.json by CMakeLists.txt\n#pragma once\n"
     "static const char kDefaultRecipesJson[] = R\"ONEBITRECIPES(${ONEBIT_RECIPES_JSON})ONEBITRECIPES\";\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/config/recipes.json)
target_include_directories(onebit PRIVATE ${CMAKE_BINARY_DIR}/generated)
# The GGUF architectures the pinned llama.cpp fork builds a model for (registry/architectures.json,
# gguf_architectures.hrx, from tools/registry_build.py): serve refuses any other on hrx and cpu.
file(READ ${CMAKE_SOURCE_DIR}/registry/architectures.json _onebit_registry)
string(JSON _onebit_n_archs LENGTH "${_onebit_registry}" gguf_architectures hrx)
set(_onebit_archs "")
math(EXPR _onebit_last "${_onebit_n_archs} - 1")
foreach(_i RANGE ${_onebit_last})
    string(JSON _a GET "${_onebit_registry}" gguf_architectures hrx ${_i})
    string(APPEND _onebit_archs "    \"${_a}\",\n")
endforeach()
file(WRITE ${CMAKE_BINARY_DIR}/generated/hrx_archs.h
     "// generated from registry/architectures.json by CMakeLists.txt\n#pragma once\n"
     "static const char* const kHrxGgufArchs[] = {\n${_onebit_archs}};\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/registry/architectures.json)
set_target_properties(onebit PROPERTIES OUTPUT_NAME 1bit)
target_link_libraries(onebit PRIVATE nlohmann_json::nlohmann_json httplib::httplib Threads::Threads)
# backends are compiled in by build-tree path; a package keeps that layout (app/built_path.h)
target_compile_definitions(onebit PRIVATE ONEBIT_BUILD_DIR="${CMAKE_BINARY_DIR}")

# Runs everywhere, CI included: serve's proxy against a fake backend.
add_test(NAME smoke_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/smoke_serve.sh $<TARGET_FILE:onebit>)
add_test(NAME parallel_args COMMAND ${CMAKE_SOURCE_DIR}/tests/parallel_args.sh $<TARGET_FILE:onebit>)
add_test(NAME recipes_route COMMAND ${CMAKE_SOURCE_DIR}/tests/recipes_route.sh $<TARGET_FILE:onebit>)
add_test(NAME device_route COMMAND ${CMAKE_SOURCE_DIR}/tests/device_route.sh $<TARGET_FILE:onebit>)
add_test(NAME bench_selftest COMMAND ${CMAKE_SOURCE_DIR}/tests/bench_selftest.sh $<TARGET_FILE:onebit>)
add_test(NAME prism_route COMMAND ${CMAKE_SOURCE_DIR}/tests/prism_route.sh $<TARGET_FILE:onebit>)
add_test(NAME laya_gguf_route COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_gguf_route.sh $<TARGET_FILE:onebit>)
# The NPU route's <think> split into reasoning_content (app/think_split.h); header only, runs in CI.
add_executable(think_split_test tests/think_split_test.cpp)
target_include_directories(think_split_test PRIVATE app)
add_test(NAME think_split COMMAND think_split_test)
# general.architecture from a GGUF header (app/gguf_meta.h), which routes fork-only
# architectures to our llama.cpp; header only, runs in CI.
add_executable(gguf_meta_test tests/gguf_meta_test.cpp)
target_include_directories(gguf_meta_test PRIVATE app)
add_test(NAME gguf_meta COMMAND gguf_meta_test)

# ── `1bit serve` end to end (docs/serve.md) ───────────────────────────────
# Needs a GGUF (e.g. Qwen3-0.6B Q4_K_M); tests each device this build has.
set(ONEBIT_SERVE_TEST_GGUF "" CACHE FILEPATH "GGUF the serve_e2e tests load")

# ── Step 2: HRX, the engine's GPU route (docs/hrx.md) ─────────────────────
# Off by default: it needs TheRock/ROCm and the HRX submodules, and it only runs on AMD GPUs.
# `1bit serve --device hrx` (and auto for a .gguf) then uses this llama-server on HRX0, and
# --device cpu the same build with no GPU layers. The build has no Vulkan or ROCm backend
# (RFC #213 stage 3): what HRX does not run, Lemonade serves with its own llamacpp backends.
option(ONEBIT_HRX "Build the HRX llama-server and wire it into Lemonade" OFF)
if(ONEBIT_HRX)
    include(cmake/hrx.cmake)
    add_dependencies(onebit llama_hrx)
    target_compile_definitions(onebit PRIVATE ONEBIT_HRX_SERVER="${ONEBIT_HRX_SERVER}" ONEBIT_HRX_LIBHSA="${ONEBIT_HRX_LIBHSA}")
    if(ONEBIT_SERVE_TEST_GGUF)
        foreach(_dev hrx cpu)
            add_test(NAME serve_e2e_${_dev}
                     COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_SERVE_TEST_GGUF} ${_dev})
        endforeach()
    endif()
endif()

# ── ZINC (docs/zinc.md) ────────────────────────────────────────────────────
# Off by default: it builds third_party/zinc with scripts/build-zinc.sh (which
# fetches its own Zig) for one GPU backend. `1bit serve --device zinc` then runs the zinc
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
endif()

# ── DwarfStar (docs/dwarfstar.md) ─────────────────────────────────────────
# Off by default: it builds third_party/ds4 (antirez/ds4, MIT) with scripts/build-ds4.sh for
# one backend (rocm: Strix Halo, needs HIP/hipBLAS(Lt)/rocBLAS/rocWMMA/hipCUB; cuda; metal;
# cpu). `1bit serve --device ds4` then runs its ds4-server on DwarfStar's own GGUFs.
option(ONEBIT_DS4 "Build DwarfStar (third_party/ds4) for --device ds4" OFF)
if(ONEBIT_DS4)
    if(APPLE)
        set(_ds4_default metal)
    else()
        set(_ds4_default rocm)
    endif()
    set(ONEBIT_DS4_BACKEND "${_ds4_default}" CACHE STRING "DwarfStar backend: rocm, cuda, metal or cpu")
    set_property(CACHE ONEBIT_DS4_BACKEND PROPERTY STRINGS rocm cuda metal cpu)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/ds4/ds4_server.c")
        message(FATAL_ERROR "ONEBIT_DS4 needs third_party/ds4: git submodule update --init third_party/ds4")
    endif()
    set(ONEBIT_DS4_SERVER "${CMAKE_BINARY_DIR}/ds4/${ONEBIT_DS4_BACKEND}/ds4-server")
    add_custom_target(ds4
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/build-ds4.sh ${CMAKE_BINARY_DIR}/ds4 ${ONEBIT_DS4_BACKEND}
        BYPRODUCTS ${ONEBIT_DS4_SERVER}
        USES_TERMINAL)
    add_dependencies(onebit ds4)
    target_compile_definitions(onebit PRIVATE ONEBIT_DS4_SERVER="${ONEBIT_DS4_SERVER}")
endif()

# ── ComfyUI.cpp (docs/comfyui.md) ──────────────────────────────────────────
# Off by default: it needs LibTorch. Builds third_party/comfyui.cpp (GPL-3.0) with
# scripts/build-comfyui.sh as its own program; `1bit comfy <workflow.json>` runs it as a
# child process. The engine never links it, so the engine itself stays Apache-2.0.
option(ONEBIT_ONNX "Build ryzenai-server (third_party/ryzenai-server) for --device onnx" OFF)
if(ONEBIT_ONNX)
    set(ONEBIT_ONNX_SERVER "${CMAKE_BINARY_DIR}/onnx/ryzenai-server")
    add_custom_target(onnx_server
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/build-onnx.sh ${CMAKE_BINARY_DIR}/onnx
        BYPRODUCTS ${ONEBIT_ONNX_SERVER}
        USES_TERMINAL)
    add_dependencies(onebit onnx_server)
    target_compile_definitions(onebit PRIVATE ONEBIT_ONNX_SERVER="${ONEBIT_ONNX_SERVER}")
endif()

option(ONEBIT_COMFYUI "Build ComfyUI.cpp (third_party/comfyui.cpp) for 1bit comfy" OFF)
if(ONEBIT_COMFYUI)
    # fetch the pinned submodule at configure time when it is missing
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/comfyui.cpp/CMakeLists.txt")
        find_package(Git QUIET)
        if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
            message(STATUS "Fetching third_party/comfyui.cpp")
            execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init third_party/comfyui.cpp
                            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endif()
    endif()
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/comfyui.cpp/CMakeLists.txt")
        message(FATAL_ERROR "ONEBIT_COMFYUI needs third_party/comfyui.cpp: git submodule update --init third_party/comfyui.cpp")
    endif()
    set(ONEBIT_COMFYUI_BIN "${CMAKE_BINARY_DIR}/comfyui/comfyui_cpp")
    add_custom_target(comfyui
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/build-comfyui.sh ${CMAKE_BINARY_DIR}/comfyui
        BYPRODUCTS ${ONEBIT_COMFYUI_BIN}
        USES_TERMINAL)
    add_dependencies(onebit comfyui)
    target_compile_definitions(onebit PRIVATE ONEBIT_COMFYUI_BIN="${ONEBIT_COMFYUI_BIN}")
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
add_library(onebit_npu STATIC npu/file_map.cpp npu/full_elf.cpp npu/model.cpp npu/pack.cpp npu/private_route.cpp npu/q4nx.cpp
                            npu/tokenizer.cpp)
target_include_directories(onebit_npu PUBLIC npu)
target_link_libraries(onebit_npu PUBLIC nlohmann_json::nlohmann_json PRIVATE PkgConfig::PCRE2)

# ── MoE expert streaming (docs/moe-streaming.md) ─────────────────────────
# The routed-expert cache: pinned RAM slots filled with O_DIRECT reads, and
# `1bit moe-cache`, which replays expert traces through it. Linux only.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(onebit_moe STATIC moe/gguf_index.cpp moe/expert_cache.cpp moe/cache_bench.cpp)
    target_include_directories(onebit_moe PUBLIC moe)
    find_package(Threads REQUIRED)
    target_link_libraries(onebit_moe PUBLIC Threads::Threads)
    target_link_libraries(onebit PRIVATE onebit_moe)
    target_compile_definitions(onebit PRIVATE ONEBIT_MOE)
endif()

# ── Step 4: Laya router (docs/laya.md) ───────────────────────────────────
# The non-autoregressive decision scorer that picks where each request runs.
# Depends on onebit_npu only for its tokenizer.json reader; the scorer itself
# is device-independent (weights via model.safetensors, no Python at runtime).
# config/route-policy.json (request class -> device, RFC #186) is compiled in as the default
# policy; --route-policy replaces it at run time.
file(READ ${CMAKE_SOURCE_DIR}/config/route-policy.json ONEBIT_ROUTE_POLICY_JSON)
file(WRITE ${CMAKE_BINARY_DIR}/generated/route_policy_default.h
     "// generated from config/route-policy.json by CMakeLists.txt\n#pragma once\n"
     "static const char kDefaultRoutePolicyJson[] = R\"ONEBITPOLICY(${ONEBIT_ROUTE_POLICY_JSON})ONEBITPOLICY\";\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/config/route-policy.json)
add_library(onebit_laya STATIC laya/scorer.cpp laya/safetensors.cpp laya/route.cpp laya/policy.cpp laya/encoder.cpp)
target_include_directories(onebit_laya PUBLIC laya PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(onebit_laya PUBLIC onebit_npu nlohmann_json::nlohmann_json)
# The engine binary gets `1bit route` and per-request routing through the scorer.
target_link_libraries(onebit PRIVATE onebit_laya)
target_compile_definitions(onebit PRIVATE ONEBIT_LAYA)
# Off by default (it needs TheRock and the HRX submodules): ggmlc's laya runs the pinned Laya GGUF
# on HRX0, 25 ms a routing decision (15 at a steady length) instead of about 540 ms on the CPU scorer, same accuracy (docs/laya.md).
option(ONEBIT_LAYA_GGML "Build ggmlc's laya (HRX) as the router's scorer" OFF)
if(ONEBIT_LAYA_GGML)
    include(cmake/laya-ggml.cmake)
    add_dependencies(onebit laya_ggml)
    target_compile_definitions(onebit PRIVATE ONEBIT_LAYA_GGML_BIN="${ONEBIT_LAYA_GGML_BIN}")
endif()

add_executable(laya_gate tests/laya_gate.cpp)
add_executable(laya_encoder_test tests/laya_encoder_test.cpp)
target_link_libraries(laya_encoder_test PRIVATE onebit_laya)
add_test(NAME laya_encoder COMMAND laya_encoder_test)
add_executable(laya_classify_eval tests/laya_classify_eval.cpp)
target_link_libraries(laya_classify_eval PRIVATE onebit_laya nlohmann_json::nlohmann_json)
target_link_libraries(laya_gate PRIVATE onebit_laya nlohmann_json::nlohmann_json)
# Each pinned checkpoint gates against its own golden.json from the Python reference.
set(ONEBIT_LAYA_ROOT "" CACHE PATH "Root Laya checkpoint dir (model.safetensors + tokenizer/tokenizer.json)")
set(ONEBIT_LAYA_GOLDEN "" CACHE FILEPATH "golden.json for the root checkpoint")
set(ONEBIT_LAYA_TYPED_ROOT "" CACHE PATH "typed-decisions checkpoint dir")
set(ONEBIT_LAYA_TYPED_GOLDEN "" CACHE FILEPATH "golden.json for the typed-decisions checkpoint")
if(ONEBIT_LAYA_ROOT AND ONEBIT_LAYA_GOLDEN)
    add_test(NAME laya_gate_root COMMAND laya_gate ${ONEBIT_LAYA_ROOT} ${ONEBIT_LAYA_GOLDEN})
endif()
if(ONEBIT_LAYA_TYPED_ROOT AND ONEBIT_LAYA_TYPED_GOLDEN)
    add_test(NAME laya_gate_typed_decisions COMMAND laya_gate ${ONEBIT_LAYA_TYPED_ROOT} ${ONEBIT_LAYA_TYPED_GOLDEN})
endif()
set(ONEBIT_LAYA_ROUTE_MODEL "" CACHE PATH "Laya checkpoint dir for the per-request routing e2e test")
if(ONEBIT_LAYA_ROUTE_MODEL)
    add_test(NAME laya_route_e2e COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_route_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_LAYA_ROUTE_MODEL})
    # the request classifier on the 200 labelled requests (RFC #186): 95.5% measured, floor 90%
    add_test(NAME laya_classify COMMAND laya_classify_eval ${ONEBIT_LAYA_ROUTE_MODEL}/typed-decisions
             ${CMAKE_SOURCE_DIR}/tests/laya_route_cases.json --min-accuracy 90)
endif()

# The registry against the reviewed not-an-alias list (docs/arch-gaps.md) and the committed pins:
# no checkouts or network.
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME registry_gaps COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/registry_build.py --check-gaps)
    # pin the guard's firing behavior: the committed registry always passes --check-gaps, so
    # this drives gap_violations against synthetic registries to prove it still fires.
    add_test(NAME registry_gaps_fire COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/registry_gaps_test.py)
    # a pin bump that did not regenerate the registry (only the gitlinks: no checkouts)
    add_test(NAME registry_pins COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/registry_build.py --check-pins)
    # the new-model watcher's alert contract (offline: synthetic batches against the
    # committed registry), so a regression cannot leave the daily alert permanently green.
    add_test(NAME census_watch COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/census_watch_test.py)
endif()

add_executable(npu_full_elf_test tests/npu_full_elf_test.cpp)
target_link_libraries(npu_full_elf_test PRIVATE onebit_npu)
add_test(NAME npu_full_elf_md5 COMMAND npu_full_elf_test)
# Against a model's kernel artifacts (docs/npu.md, "Checking the generator").
set(ONEBIT_NPU_CAPTURED "" CACHE PATH "Directory of captured layer_ctx<N>.elf instruction ELFs")
set(ONEBIT_NPU_PDI "" CACHE FILEPATH "The layer kernel's design PDI")
set(ONEBIT_NPU_GOLDEN "" CACHE PATH "Directory of full ELFs the fast lane ran")
set(ONEBIT_NPU_KERNELS_DIR "" CACHE PATH "Full ELF / xclbin directory the GGUF-on-NPU forward loads (docs/npu.md)")
target_compile_definitions(onebit PRIVATE ONEBIT_NPU_REPACK_SCRIPT="${CMAKE_SOURCE_DIR}/scripts/repack_gguf.py")
# The GGUF -> fused-lane Q4NX layout packer (docs/npu.md, "From a GGUF").  A
# different layout from repack_gguf.py; used only when ONEBIT_NPU_LANE_DIR routes
# a GGUF to the fast lane, and never on the per-op forward's path.
target_compile_definitions(onebit PRIVATE ONEBIT_NPU_LANE_PACK_SCRIPT="${CMAKE_SOURCE_DIR}/tools/gguf_to_q4nx.py")
if(ONEBIT_NPU_KERNELS_DIR)
    target_compile_definitions(onebit PRIVATE ONEBIT_NPU_KERNELS_DIR="${ONEBIT_NPU_KERNELS_DIR}")
endif()
if(ONEBIT_NPU_CAPTURED AND ONEBIT_NPU_PDI AND ONEBIT_NPU_GOLDEN)
    add_test(NAME npu_full_elf_model
             COMMAND npu_full_elf_test ${ONEBIT_NPU_CAPTURED} ${ONEBIT_NPU_PDI} ${ONEBIT_NPU_GOLDEN})
endif()

# Q4NX chunk decoders against 1bit-MONSTER's verified decoders, on committed real tiles
add_executable(npu_q4nx_test tests/npu_q4nx_test.cpp)
target_link_libraries(npu_q4nx_test PRIVATE onebit_npu)
add_test(NAME npu_q4nx COMMAND npu_q4nx_test ${CMAKE_SOURCE_DIR}/tests/golden/q4nx)
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
# The private NPU route registry (npu/private_route.h), without an add-on.
add_executable(npu_private_route_test tests/npu_private_route_test.cpp)
target_link_libraries(npu_private_route_test PRIVATE onebit_npu)
add_test(NAME npu_private_route COMMAND npu_private_route_test)
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

    # The model-generic forward (npu/forward): the GGUF-on-NPU serve route
    # (docs/npu.md, "Open"). q4nx_forward.cpp + its Q4NX dequant/pack/semantics
    # + model loader, against the engine tree's I8Ctx header.
    add_library(onebit_npu_forward STATIC
        npu/forward/src/model.c
        npu/forward/src/q4nx_dequant.c
        npu/forward/src/q4nx_pack.c
        npu/forward/src/q4nx_semantics.c
        npu/forward/src/q4nx_forward.cpp)
    # The Q4NX pack and dequant loops carry `#pragma omp parallel for`; without the
    # OpenMP flags those compile to serial code. Wire it in (both C and C++ sources)
    # so they actually use the cores.
    find_package(OpenMP)
    if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
        target_link_libraries(onebit_npu_forward PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    endif()
    # NPU_INFER_DUMP_ATTN / _LN / _HEAD write debug dumps to a path taken from the environment;
    # only debug builds carry them
    option(ONEBIT_NPU_DEBUG_DUMPS "NPU forward: env-named debug dumps (developer builds)" OFF)
    if(ONEBIT_NPU_DEBUG_DUMPS)
        target_compile_definitions(onebit_npu_forward PRIVATE NPU_INFER_DEBUG_DUMPS)
    endif()
    target_include_directories(onebit_npu_forward SYSTEM PUBLIC
        npu/forward/include npu/forward/engine-src ${ONEBIT_XRT_ROOT}/include)
    target_link_libraries(onebit_npu_forward PUBLIC
        ${ONEBIT_XRT_ROOT}/lib/libxrt_coreutil.so aiebu uuid dl pthread m)
    target_sources(onebit PRIVATE app/forward_serve.cpp)
    target_link_libraries(onebit PRIVATE onebit_npu_forward)
    set(ONEBIT_NPU_KERNELS "" CACHE PATH "Kernel directory (layer_ctx{1,2,17}.elf, lmhead.elf, layer.pdi)")
    # The fast-lane end-to-end test (npu_lane_e2e) is retired: its kernels were FastFlowLM-derived
    # and its reference logits are lost. The NPU check is the private dx add-on's from-source
    # check, outside this repository (docs/npu.md, "Checking the NPU").
endif()

# Private NPU routes (docs/npu.md, "Private routes"): closed-source add-ons in a checkout of
# the private npu-kernels repository. Its addons/ directory defines the target
# onebit_npu_private, which holds register_private_addon() (npu/private_route.h) and brings
# its own options and tests.
set(ONEBIT_NPU_PRIVATE "" CACHE PATH "An npu-kernels checkout whose addons/ are built into 1bit (needs ONEBIT_NPU)")
if(ONEBIT_NPU_PRIVATE)
    if(NOT ONEBIT_NPU)
        message(FATAL_ERROR "ONEBIT_NPU_PRIVATE needs ONEBIT_NPU=ON")
    endif()
    if(NOT EXISTS "${ONEBIT_NPU_PRIVATE}/addons/CMakeLists.txt")
        message(FATAL_ERROR "ONEBIT_NPU_PRIVATE=${ONEBIT_NPU_PRIVATE} has no addons/CMakeLists.txt")
    endif()
    add_subdirectory(${ONEBIT_NPU_PRIVATE}/addons ${CMAKE_BINARY_DIR}/npu_private)
    if(NOT TARGET onebit_npu_private)
        message(FATAL_ERROR "${ONEBIT_NPU_PRIVATE}/addons defines no onebit_npu_private target")
    endif()
    target_link_libraries(onebit PRIVATE onebit_npu_private)
    target_compile_definitions(onebit PRIVATE ONEBIT_NPU_PRIVATE)
    # Make the dx kernels' build default visible to app/serve.cpp's GGUF -> fused dx
    # routing branch too (the add-on compiles it only onto onebit_npu_dx, PRIVATE).
    if(ONEBIT_NPU_DX_KERNELS)
        target_compile_definitions(onebit PRIVATE ONEBIT_NPU_DX_KERNELS_DEFAULT="${ONEBIT_NPU_DX_KERNELS}")
    endif()
endif()


# MLX on Apple Silicon: `1bit serve --device mlx` (docs/apple.md).
set(ONEBIT_MLX_SERVER "" CACHE FILEPATH "lemon-mlx-engine's server binary, for serve_e2e_mlx")
if(APPLE AND ONEBIT_MLX_SERVER)
    add_test(NAME serve_e2e_mlx
             COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_e2e.sh $<TARGET_FILE:onebit> mlx-community/Qwen3-0.6B-4bit mlx
                     --mlx-server ${ONEBIT_MLX_SERVER})
endif()
