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
# The engine needs GCC 16: C++26 std::simd (<simd>) is in libstdc++ from GCC 16.
# Pick gcc-16 / g++-16 when no compiler was chosen and they are installed.
if(NOT CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  find_program(ONEBIT_GXX16 g++-16)
  find_program(ONEBIT_GCC16 gcc-16)
  if(ONEBIT_GXX16 AND ONEBIT_GCC16)
    set(CMAKE_C_COMPILER "${ONEBIT_GCC16}")
    set(CMAKE_CXX_COMPILER "${ONEBIT_GXX16}")
  endif()
endif()
project(onebit_engine VERSION 0.0.1 LANGUAGES C CXX)
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
  message(FATAL_ERROR "The engine needs GCC 16 or newer for C++26 std::simd; found "
                      "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}. Install g++-16 or pass "
                      "-DCMAKE_CXX_COMPILER=g++-16 -DCMAKE_C_COMPILER=gcc-16.")
endif()

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Every string the build bakes in (asserts, log prefixes, __FILE__) has to name a path that exists in
# the package, not on the machine that built it: the shipped binaries are gated on it
# (scripts/pack-elf.sh, pack_artifact_hygiene). -ffile-prefix-map covers -fmacro-prefix-map and
# -fdebug-prefix-map on GCC and clang alike; ONEBIT_PATH_MAPS is also handed to the externals below.
set(ONEBIT_PATH_MAP_FLAGS
    "-ffile-prefix-map=${CMAKE_SOURCE_DIR}=."
    "-ffile-prefix-map=${CMAKE_BINARY_DIR}=build"
    CACHE STRING "Compiler flags that rewrite build-machine paths out of embedded strings")
add_compile_options(${ONEBIT_PATH_MAP_FLAGS})

# The public binary drop (docs/releases.md): prebuilt proprietary parts for a source build outside
# the org. Sets ONEBIT_BINARY_DROP_* and verifies the drop; a no-op when ONEBIT_BINARY_DROP is unset.
include(cmake/binary_drop.cmake)
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

add_executable(onebit app/main.cpp app/serve.cpp app/recipes.cpp app/pm.cpp)
# config/recipes.json (tuned llama-server settings per model and route, docs/recipes.md) is
# compiled in as serve's default recipes; --recipes replaces them at run time.
file(READ ${CMAKE_SOURCE_DIR}/config/recipes.json ONEBIT_RECIPES_JSON)
file(WRITE ${CMAKE_BINARY_DIR}/generated/recipes_default.h
     "// generated from config/recipes.json by CMakeLists.txt\n#pragma once\n"
     "static const char kDefaultRecipesJson[] = R\"ONEBITRECIPES(${ONEBIT_RECIPES_JSON})ONEBITRECIPES\";\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/config/recipes.json)
# config/pm-experts.json (the Project Manager's experts, docs/pm.md) is compiled in the same way;
# `serve --pm-experts FILE` replaces it at run time.
file(READ ${CMAKE_SOURCE_DIR}/config/pm-experts.json ONEBIT_PM_EXPERTS_JSON)
file(WRITE ${CMAKE_BINARY_DIR}/generated/pm_experts_default.h
     "// generated from config/pm-experts.json by CMakeLists.txt\n#pragma once\n"
     "static const char kDefaultPmExpertsJson[] = R\"ONEBITPM(${ONEBIT_PM_EXPERTS_JSON})ONEBITPM\";\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/config/pm-experts.json)
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
# Backend paths are embedded RELATIVE to the build dir (see app/built_path.h): a shipped binary
# must not name a build-machine path. Each define below is turned into its relative form first.
function(onebit_build_relative var out)
    file(RELATIVE_PATH _rel "${CMAKE_BINARY_DIR}" "${${var}}")
    set(${out} "${_rel}" PARENT_SCOPE)
endfunction()

# Runs everywhere, CI included: serve's proxy against a fake backend.
add_test(NAME smoke_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/smoke_serve.sh $<TARGET_FILE:onebit>)
# serve takes a hung backend down (docs/serve.md, "When a backend hangs"): a fake backend, no GPU.
add_test(NAME watchdog_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/watchdog_serve.sh $<TARGET_FILE:onebit>)
# serve's GET /metrics (docs/serve.md, "Metrics"): a fake backend, no GPU.
add_test(NAME metrics_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/metrics_serve.sh $<TARGET_FILE:onebit>)
# serve --print-backend: the llama-server Lemonade starts itself, one hop (docs/serve.md): no GPU.
add_test(NAME print_backend COMMAND ${CMAKE_SOURCE_DIR}/tests/print_backend.sh $<TARGET_FILE:onebit>)
# scripts/lemonade-hrx: the PPA lemond's HRX recipe runs our llama-server (docs/lemonade.md): no GPU.
add_test(NAME lemonade_hrx COMMAND ${CMAKE_SOURCE_DIR}/tests/lemonade_hrx.sh $<TARGET_FILE:onebit>)
add_test(NAME parallel_args COMMAND ${CMAKE_SOURCE_DIR}/tests/parallel_args.sh $<TARGET_FILE:onebit>)
add_test(NAME recipes_route COMMAND ${CMAKE_SOURCE_DIR}/tests/recipes_route.sh $<TARGET_FILE:onebit>)
add_test(NAME greedy_shortcut COMMAND ${CMAKE_SOURCE_DIR}/tests/greedy_shortcut.sh $<TARGET_FILE:onebit>)
add_test(NAME device_route COMMAND ${CMAKE_SOURCE_DIR}/tests/device_route.sh $<TARGET_FILE:onebit>)
# where --device npu finds the GGUF repack script: beside 1bit in a package (scripts/package-linux.sh),
# else ONEBIT_Q4NX_REPACK; no NPU needed (the repack runs before the device)
add_test(NAME npu_repack_path COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_repack_path.sh $<TARGET_FILE:onebit>)
# a build without the NPU lane has no repack path: the test says skip (77) there, not fail
set_tests_properties(npu_repack_path PROPERTIES SKIP_RETURN_CODE 77)
# the OS image's own release check: its lemonade row must prove the API key is enforced, not infer it
# from an empty body (os/mini/release-check.sh --self-test; no device and no build needed)
add_test(NAME release_check_row COMMAND ${CMAKE_SOURCE_DIR}/os/mini/release-check.sh --self-test)
# engine #574: the signed manifest of a sealed NPU kernel set. tools/seal_manifest.py writes it (sha256
# of the ciphertext, Ed25519 over the manifest) and checks it; the test is a throwaway set in a temp
# dir, so no key, no device and no build: the four refusals are a set with no manifest, a file the
# manifest does not list, a ciphertext byte flipped after signing, and a manifest signed by another key.
add_test(NAME seal_manifest COMMAND ${CMAKE_SOURCE_DIR}/tests/seal_manifest_test.sh)
add_test(NAME pm_route COMMAND ${CMAKE_SOURCE_DIR}/tests/pm_route.sh $<TARGET_FILE:onebit>)
add_test(NAME bench_selftest COMMAND ${CMAKE_SOURCE_DIR}/tests/bench_selftest.sh $<TARGET_FILE:onebit>)
add_test(NAME prism_route COMMAND ${CMAKE_SOURCE_DIR}/tests/prism_route.sh $<TARGET_FILE:onebit>)
add_test(NAME laya_gguf_route COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_gguf_route.sh $<TARGET_FILE:onebit>)
add_test(NAME mem_admit_route COMMAND ${CMAKE_SOURCE_DIR}/tests/mem_admit_route.sh $<TARGET_FILE:onebit>)
# engine#315: what --hrx-cwsr-placement puts in an HRX backend's environment (app/hrx_cwsr.h); fake backend, no GPU.
add_test(NAME hrx_cwsr_route COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_cwsr_route.sh $<TARGET_FILE:onebit>)
# --device hybrid (docs/hybrid.md): the handoff with a fake backend and a fake provider; and, with a model,
# end to end on the CPU llama-server (ONEBIT_SERVE_TEST_LLAMA_SERVER, else the build's own; skips without one)
add_test(NAME hybrid_route COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_route.sh $<TARGET_FILE:onebit>)
set(ONEBIT_SERVE_TEST_LLAMA_SERVER "" CACHE FILEPATH "llama-server for hybrid_e2e (default: the build's own)")
if(ONEBIT_SERVE_TEST_GGUF)
    add_test(NAME hybrid_e2e
             COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_SERVE_TEST_GGUF} ${ONEBIT_SERVE_TEST_LLAMA_SERVER})
    set_tests_properties(hybrid_e2e PROPERTIES SKIP_RETURN_CODE 77)
endif()
# The NPU route's <think> split into reasoning_content (app/think_split.h); header only, runs in CI.
add_executable(think_split_test tests/think_split_test.cpp)
target_include_directories(think_split_test PRIVATE app)
add_test(NAME think_split COMMAND think_split_test)
# Host / Origin checks every engine HTTP server applies (app/http_guard.h); header only, runs in CI.
add_executable(http_guard_test tests/http_guard_test.cpp)
target_include_directories(http_guard_test PRIVATE app)
add_test(NAME http_guard COMMAND http_guard_test)
# general.architecture from a GGUF header (app/gguf_meta.h), which routes fork-only
# architectures to our llama.cpp; header only, runs in CI.
add_executable(gguf_meta_test tests/gguf_meta_test.cpp)
target_include_directories(gguf_meta_test PRIVATE app)
add_test(NAME gguf_meta COMMAND gguf_meta_test)
# serve's memory admission (app/mem_admit.h, docs/serve.md "Will it fit?"): KV/state sizes against
# HRX0 logs, the box readings on a fake /proc and /sys; header only, runs in CI.
add_executable(mem_admit_test tests/mem_admit_test.cpp)
target_include_directories(mem_admit_test PRIVATE app)
add_test(NAME mem_admit COMMAND mem_admit_test)
# engine #390's load-time placement report and --strict-device (app/serve_metrics.h, docs/serve.md
# "Off-HRX placement"): the backend's placement line in, serve's line and the refusal out; header
# only, no GPU, no backend.
add_executable(hrx_placement_test tests/hrx_placement_test.cpp)
target_include_directories(hrx_placement_test PRIVATE app)
target_link_libraries(hrx_placement_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME hrx_placement COMMAND hrx_placement_test)
# engine#315 candidate 1's --hrx-cwsr-placement decision (app/hrx_cwsr.h, docs/hrx.md "CWSR
# save-area placement"): a fake cwsr_enable file and environment in, the variables out; header
# only, no GPU.
add_executable(hrx_cwsr_test tests/hrx_cwsr_test.cpp)
target_include_directories(hrx_cwsr_test PRIVATE app)
add_test(NAME hrx_cwsr COMMAND hrx_cwsr_test)
# --pm-router's class -> expert table and its flag values (app/pm.cpp, docs/pm.md "Routing with
# Laya"); no GPU, no checkpoint, no Lemonade: config in, mapping out.
add_executable(pm_routes_test tests/pm_routes_test.cpp app/pm.cpp)
target_include_directories(pm_routes_test PRIVATE app ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(pm_routes_test PRIVATE nlohmann_json::nlohmann_json httplib::httplib Threads::Threads)
add_test(NAME pm_routes COMMAND pm_routes_test)

# ── `1bit serve` end to end (docs/serve.md) ───────────────────────────────
# Needs a GGUF (e.g. Qwen3-0.6B Q4_K_M); tests each device this build has.
set(ONEBIT_SERVE_TEST_GGUF "" CACHE FILEPATH "GGUF the serve_e2e tests load")

# ── Step 2: HRX, the engine's GPU route (docs/hrx.md) ─────────────────────
# Off by default: it needs TheRock/ROCm and the HRX submodules, and it only runs on AMD GPUs.
# `1bit serve --device hrx` (and auto for a .gguf) then uses this llama-server on HRX0, and
# --device cpu the same build with no GPU layers. The build has no Vulkan or ROCm backend
# (RFC #213 stage 3): what HRX does not run, Lemonade serves with its own llamacpp backends.
option(ONEBIT_HRX "Build the HRX llama-server and wire it into Lemonade" OFF)
set(ONEBIT_GPU_PRIVATE "" CACHE PATH "A gpu-kernels checkout whose addons/hip-hrx HIP kernels are built into the HRX llama.cpp (needs ONEBIT_HRX)")
if(ONEBIT_GPU_PRIVATE AND NOT ONEBIT_HRX)
    message(FATAL_ERROR "ONEBIT_GPU_PRIVATE needs ONEBIT_HRX=ON")
endif()
if(ONEBIT_HRX)
    include(cmake/hrx.cmake)
    # A drop build has no llama_hrx target: the prebuilt server and its libraries come from the drop.
    if(TARGET llama_hrx)
        add_dependencies(onebit llama_hrx)
    endif()
    onebit_build_relative(ONEBIT_HRX_SERVER _onebit_hrx_server)
    target_compile_definitions(onebit PRIVATE ONEBIT_HRX_SERVER="${_onebit_hrx_server}" ONEBIT_HRX_LIBHSA="${ONEBIT_HRX_LIBHSA}"
                               ONEBIT_HRX_ALLOW_NO_SVM=${ONEBIT_HRX_ALLOW_NO_SVM})
    if(ONEBIT_SERVE_TEST_GGUF)
        foreach(_dev hrx cpu)
            add_test(NAME serve_e2e_${_dev}
                     COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_SERVE_TEST_GGUF} ${_dev})
            # hrx skips without a gfx1151 (ONEBIT_HRX_E2E_TARGETS)
            set_tests_properties(serve_e2e_${_dev} PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
    endif()
endif()

# ── DwarfStar (docs/dwarfstar.md) ─────────────────────────────────────────
# Off by default: it builds third_party/ds4 (antirez/ds4, MIT) with scripts/build-ds4.sh on
# its HIP backend only (Strix Halo, gfx1151: HIP, hipBLAS(Lt), rocBLAS, rocWMMA, hipCUB).
# `1bit serve --device ds4` then runs its ds4-server on DwarfStar's own GGUFs.
option(ONEBIT_DS4 "Build DwarfStar (third_party/ds4) for --device ds4" OFF)
if(ONEBIT_DS4)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/ds4/ds4_server.c")
        message(FATAL_ERROR "ONEBIT_DS4 needs third_party/ds4: git submodule update --init third_party/ds4")
    endif()
    set(ONEBIT_DS4_SERVER "${CMAKE_BINARY_DIR}/ds4/hip/ds4-server")
    add_custom_target(ds4
        COMMAND ${CMAKE_SOURCE_DIR}/scripts/build-ds4.sh ${CMAKE_BINARY_DIR}/ds4
        BYPRODUCTS ${ONEBIT_DS4_SERVER}
        USES_TERMINAL)
    add_dependencies(onebit ds4)
    onebit_build_relative(ONEBIT_DS4_SERVER _onebit_ds4_server)
    target_compile_definitions(onebit PRIVATE ONEBIT_DS4_SERVER="${_onebit_ds4_server}")
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
# NPU runtime. It needs PCRE2 (the tokenizer), and macOS has no NPU, so it is off there
# by default.
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
    # Crafted GGUF headers the --moe-slots index must refuse (written at run time); runs in CI.
    add_executable(moe_gguf_index_test tests/moe_gguf_index_test.cpp)
    target_link_libraries(moe_gguf_index_test PRIVATE onebit_moe)
    add_test(NAME moe_gguf_index COMMAND moe_gguf_index_test)
    set_tests_properties(moe_gguf_index PROPERTIES TIMEOUT 60)
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
    onebit_build_relative(ONEBIT_LAYA_GGML_BIN _onebit_laya_ggml)
    target_compile_definitions(onebit PRIVATE ONEBIT_LAYA_GGML_BIN="${_onebit_laya_ggml}")
endif()

add_executable(laya_gate tests/laya_gate.cpp)
add_executable(laya_encoder_test tests/laya_encoder_test.cpp)
target_link_libraries(laya_encoder_test PRIVATE onebit_laya)
add_test(NAME laya_encoder COMMAND laya_encoder_test)
# Crafted Laya checkpoints the reader and the scorer must refuse (written at run time); runs in CI.
add_executable(laya_bounds_test tests/laya_bounds_test.cpp)
target_link_libraries(laya_bounds_test PRIVATE onebit_laya nlohmann_json::nlohmann_json)
add_test(NAME laya_bounds COMMAND laya_bounds_test)
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
    # `--pm-router laya` end to end without a GPU: the same classifier decides the class of two
    # labelled requests and the engine's table decides whether the PM delegates (docs/pm.md).
    add_test(NAME pm_laya_route COMMAND ${CMAKE_SOURCE_DIR}/tests/pm_laya_route.sh $<TARGET_FILE:onebit> ${ONEBIT_LAYA_ROUTE_MODEL})
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
    # committed registry, plus FastFlowLM's org listing and releases from the recorded
    # payloads under tests/fixtures/fflm/ — engine #588), so a regression cannot leave the
    # daily alert permanently green.
    add_test(NAME census_watch COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/census_watch_test.py)

    # a Q4NX directory's chunk layout (engine #587): synthetic containers, so the check a
    # published FastFlowLM model goes through is proved to fire without the model at hand.
    add_test(NAME q4nx_dir_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/check_q4nx_dir.py --self-test)
    # the HRX hardware gate's judges (docs/hw-gate.md) against a fake server and synthetic logs;
    # the gate itself needs the GPU and runs on Strix Halo (.github/workflows/hw-gate.yml)
    add_test(NAME hw_gate_judges COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/hw_gate_test.py)
endif()

add_executable(npu_full_elf_test tests/npu_full_elf_test.cpp)
target_link_libraries(npu_full_elf_test PRIVATE onebit_npu)
add_test(NAME npu_full_elf_md5 COMMAND npu_full_elf_test)
# Against a model's kernel artifacts (docs/npu.md, "Checking the generator").
set(ONEBIT_NPU_CAPTURED "" CACHE PATH "Directory of captured layer_ctx<N>.elf instruction ELFs")
set(ONEBIT_NPU_PDI "" CACHE FILEPATH "The layer kernel's design PDI")
set(ONEBIT_NPU_GOLDEN "" CACHE PATH "Directory of full ELFs the fast lane ran")
set(ONEBIT_NPU_KERNELS_DIR "" CACHE PATH "Full-ELF directory the GGUF-on-NPU forward loads (docs/npu.md)")
set(_onebit_npu_repack "${CMAKE_SOURCE_DIR}/scripts/repack_gguf.py")
onebit_build_relative(_onebit_npu_repack _onebit_npu_repack_rel)
target_compile_definitions(onebit PRIVATE ONEBIT_NPU_REPACK_SCRIPT="${_onebit_npu_repack_rel}")
# The GGUF -> fused-lane Q4NX layout packer (docs/npu.md, "From a GGUF").  A
# different layout from repack_gguf.py; used only when ONEBIT_NPU_LANE_DIR routes
# a GGUF to the fast lane, and never on the per-op forward's path.
set(_onebit_npu_lane_pack "${CMAKE_SOURCE_DIR}/tools/gguf_to_q4nx.py")
onebit_build_relative(_onebit_npu_lane_pack _onebit_npu_lane_pack_rel)
target_compile_definitions(onebit PRIVATE ONEBIT_NPU_LANE_PACK_SCRIPT="${_onebit_npu_lane_pack_rel}")
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
# Crafted model.q4nx files the loader and the packer must refuse (written at run time); runs in CI.
add_executable(npu_model_bounds_test tests/npu_model_bounds_test.cpp)
target_link_libraries(npu_model_bounds_test PRIVATE onebit_npu)
add_test(NAME npu_model_bounds COMMAND npu_model_bounds_test)
# The forward's Q4NX loader is plain C (no NPU runtime), so its crafted-file checks run in CI too.
add_executable(npu_forward_loader_test tests/npu_forward_loader_test.c npu/forward/src/model.c)
target_include_directories(npu_forward_loader_test PRIVATE npu/forward/include)
target_link_libraries(npu_forward_loader_test PRIVATE m)
add_test(NAME npu_forward_loader COMMAND npu_forward_loader_test)
set(ONEBIT_NPU_MODEL "" CACHE PATH "Q4NX model directory (model.q4nx + config.json)")
set(ONEBIT_NPU_BO_DUMP "" CACHE PATH "The reference lane's buffer dump (RT_DUMP_BOS)")
add_executable(tokenizer_test tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE onebit_npu)
# Large inputs tokenize in linear time (a synthetic vocabulary, no model needed); runs in CI.
add_executable(tokenizer_linear_test tests/tokenizer_linear_test.cpp)
target_link_libraries(tokenizer_linear_test PRIVATE onebit_npu)
add_test(NAME tokenizer_linear COMMAND tokenizer_linear_test ${CMAKE_SOURCE_DIR}/tests/data/tokenizer_linear.json)
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

# The lane itself: needs the NPU runtime and an XDNA NPU to run.
option(ONEBIT_NPU "Build the NPU lane (needs the private NPU runtime: ONEBIT_NPU_RT_LIB or ONEBIT_NPU_RT_DIR)" OFF)
# The NPU code compiles against one runtime (npu/rt.h, docs/npu.md "The runtime"): our own runtime,
# which is not part of this repository, behind our public header npu/npu_rt_api.h. It comes one of
# two ways, exactly one of which must be set:
#   ONEBIT_NPU_RT_LIB  a prebuilt static library (libnpu_rt_api.a) that implements npu_rt_api.h;
#                      no runtime source or private header is needed.
#   ONEBIT_NPU_RT_DIR  the runtime's private sources (npu_rt.h, npu_rt.cpp, npu_rt_api.cpp), built
#                      here into the static library onebit_npu_rt. npu_rt.h stays on that library's
#                      own include path only; the engine's sources see npu_rt_api.h.
# The release build takes the runtime from the pinned private checkout in config/npu-runtime.config
# (scripts/fetch-private.sh); the engine has that one NPU runtime and no other.
set(ONEBIT_NPU_RUNTIME "own" CACHE STRING "NPU runtime: own (the only value)")
set_property(CACHE ONEBIT_NPU_RUNTIME PROPERTY STRINGS own)
set(ONEBIT_NPU_RT_DIR "" CACHE PATH "Directory with the NPU runtime's private sources: npu_rt.h, npu_rt.cpp, npu_rt_api.cpp")
set(ONEBIT_NPU_RT_LIB "" CACHE FILEPATH "Prebuilt NPU runtime static library implementing npu/npu_rt_api.h (libnpu_rt_api.a)")
if(NOT ONEBIT_NPU_RUNTIME STREQUAL "own")
    message(FATAL_ERROR "ONEBIT_NPU_RUNTIME must be own, not '${ONEBIT_NPU_RUNTIME}': the engine has one "
                        "NPU runtime, its own (docs/npu.md, \"The runtime\")")
endif()
if(ONEBIT_NPU AND NOT ONEBIT_NPU_HOST)
    message(FATAL_ERROR "ONEBIT_NPU needs ONEBIT_NPU_HOST=ON")
endif()
# The option's refusals (bad value, no runtime directory); needs no NPU runtime, runs in CI.
if(ONEBIT_NPU_HOST AND NOT WIN32)
    add_test(NAME npu_runtime_option
             COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_runtime_option.sh ${CMAKE_SOURCE_DIR} ${FETCHCONTENT_BASE_DIR})
    set_tests_properties(npu_runtime_option PROPERTIES TIMEOUT 300)
endif()
if(ONEBIT_NPU)
    # The public binary drop carries the runtime already compiled (libnpu_rt_api.a behind
    # npu/npu_rt_api.h), for a source build that has no private runtime sources
    # (docs/releases.md, "The public binary drop").
    if(NOT ONEBIT_NPU_RT_LIB AND NOT ONEBIT_NPU_RT_DIR AND ONEBIT_BINARY_DROP_NPU_LIB)
        set(ONEBIT_NPU_RT_LIB "${ONEBIT_BINARY_DROP_NPU_LIB}")
        message(STATUS "ONEBIT_NPU: runtime from the binary drop: ${ONEBIT_NPU_RT_LIB}")
    endif()
    if(ONEBIT_NPU_RT_LIB AND ONEBIT_NPU_RT_DIR)
        message(FATAL_ERROR "ONEBIT_NPU takes one NPU runtime: set ONEBIT_NPU_RT_LIB or ONEBIT_NPU_RT_DIR, "
                            "not both (docs/npu.md, \"The NPU runtime\")")
    endif()
    find_package(Threads REQUIRED)
    if(ONEBIT_NPU_RT_LIB)
        if(NOT EXISTS "${ONEBIT_NPU_RT_LIB}" OR IS_DIRECTORY "${ONEBIT_NPU_RT_LIB}")
            message(FATAL_ERROR "ONEBIT_NPU_RT_LIB='${ONEBIT_NPU_RT_LIB}' is not a library file "
                                "(docs/npu.md, \"The NPU runtime\")")
        endif()
        add_library(onebit_npu_rt_prebuilt STATIC IMPORTED)
        set_target_properties(onebit_npu_rt_prebuilt PROPERTIES IMPORTED_LOCATION "${ONEBIT_NPU_RT_LIB}")
        add_library(onebit_npu_rt INTERFACE)
        target_link_libraries(onebit_npu_rt INTERFACE onebit_npu_rt_prebuilt Threads::Threads)
        target_include_directories(onebit_npu_rt INTERFACE ${CMAKE_SOURCE_DIR}/npu)
        target_compile_definitions(onebit_npu_rt INTERFACE ONEBIT_NPU_RUNTIME_OWN)
    else()
        if(NOT EXISTS "${ONEBIT_NPU_RT_DIR}/npu_rt.h" OR NOT EXISTS "${ONEBIT_NPU_RT_DIR}/npu_rt.cpp"
           OR NOT EXISTS "${ONEBIT_NPU_RT_DIR}/npu_rt_api.cpp")
            message(FATAL_ERROR "ONEBIT_NPU needs the NPU runtime: ONEBIT_NPU_RT_LIB must name its prebuilt "
                                "library, or ONEBIT_NPU_RT_DIR must hold npu_rt.h, npu_rt.cpp and npu_rt_api.cpp "
                                "(ONEBIT_NPU_RT_DIR='${ONEBIT_NPU_RT_DIR}'). The runtime is private; "
                                "without it, build with ONEBIT_NPU=OFF (docs/npu.md, \"The NPU runtime\")")
        endif()
        add_library(onebit_npu_rt STATIC ${ONEBIT_NPU_RT_DIR}/npu_rt.cpp ${ONEBIT_NPU_RT_DIR}/npu_rt_api.cpp)
        # npu_rt.h: this library's own include path only (PRIVATE), never the engine's.
        target_include_directories(onebit_npu_rt SYSTEM PRIVATE ${ONEBIT_NPU_RT_DIR})
        target_include_directories(onebit_npu_rt PUBLIC ${CMAKE_SOURCE_DIR}/npu)
        target_compile_definitions(onebit_npu_rt PUBLIC ONEBIT_NPU_RUNTIME_OWN)
        target_link_libraries(onebit_npu_rt PUBLIC Threads::Threads)
    endif()
    add_library(onebit_npu_lane STATIC npu/lane.cpp npu/generate.cpp)
    target_link_libraries(onebit_npu_lane PUBLIC onebit_npu onebit_npu_rt)
    target_sources(onebit PRIVATE app/unified.cpp)
    target_link_libraries(onebit PRIVATE onebit_npu_lane)
    target_compile_definitions(onebit PRIVATE ONEBIT_NPU)
    # The shim compiles against the runtime (and runs without an NPU: it only builds objects and
    # checks the API surface the engine uses).
    add_executable(npu_rt_shim_test tests/npu_rt_shim_test.cpp)
    target_link_libraries(npu_rt_shim_test PRIVATE onebit_npu_lane)
    add_test(NAME npu_rt_shim COMMAND npu_rt_shim_test)

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
        npu/forward/include npu/forward/engine-src npu)
    target_link_libraries(onebit_npu_forward PUBLIC onebit_npu_rt dl pthread m)
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
