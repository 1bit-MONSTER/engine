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

# config/laya-first.json (the routing table, docs/laya-first.md §4) is compiled in like
# config/recipes.json; `1bit laya-first --laya-first FILE` replaces it at run time.
file(READ ${CMAKE_SOURCE_DIR}/config/laya-first.json ONEBIT_LAYA_FIRST_JSON)
file(WRITE ${CMAKE_BINARY_DIR}/generated/laya_first_default.h
     "// generated from config/laya-first.json by CMakeLists.txt\n#pragma once\n"
     "static const char kDefaultLayaFirstJson[] = R\"ONEBITLAYAFIRST(${ONEBIT_LAYA_FIRST_JSON})ONEBITLAYAFIRST\";\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/config/laya-first.json)
add_executable(onebit app/main.cpp app/serve.cpp app/recipes.cpp app/pm.cpp app/route_serve.cpp app/route_eval.cpp app/laya_checkpoint.cpp app/laya_first.cpp
                     app/embed_serve.cpp app/embed_model.cpp app/q4nx_cache.cpp)
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
# config/template-pins.json (the chat-template sha256 expected per model, docs/serve.md "Chat template") is
# compiled in the same way; `serve --template-pin FILE` replaces it, `--no-template-pin` drops it.
file(READ ${CMAKE_SOURCE_DIR}/config/template-pins.json ONEBIT_TEMPLATE_PINS_JSON)
file(WRITE ${CMAKE_BINARY_DIR}/generated/template_pins_default.h
     "// generated from config/template-pins.json by CMakeLists.txt\n#pragma once\n"
     "static const char kDefaultTemplatePinsJson[] = R\"ONEBITTEMPLATE(${ONEBIT_TEMPLATE_PINS_JSON})ONEBITTEMPLATE\";\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/config/template-pins.json)
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

# ── The chat-template renderer (engine #963, docs/npu-chat-template.md) ──────
# The NPU chat route renders the container's own `chat_template` (app/npu_template.h) with the pinned
# llama.cpp fork's Jinja engine (`third_party/llama.cpp/common/jinja`, upstream's, the one
# `llama-server --jinja` uses) rather than with a second, weaker interpreter of the same language.
#
# The sources are the fork's, so they exist exactly when the submodule is checked out: the option
# defaults to that, and a tree that never initialised the submodules still builds -- without the
# renderer, which the load line and the route say out loud rather than rendering something else
# (app/npu_template.h). Nothing is downloaded at build time.
if(EXISTS ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common/jinja/lexer.cpp)
    set(_onebit_jinja_default ON)
else()
    set(_onebit_jinja_default OFF)
endif()
option(ONEBIT_JINJA "Build the Jinja chat-template renderer from the llama.cpp submodule" ${_onebit_jinja_default})
if(ONEBIT_JINJA)
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common/jinja/lexer.cpp)
        message(FATAL_ERROR "ONEBIT_JINJA=ON but third_party/llama.cpp/common/jinja is missing: "
                            "git submodule update --init third_party/llama.cpp")
    endif()
    set(_onebit_jinja_dir ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/common)
    add_library(onebit_jinja STATIC
        ${_onebit_jinja_dir}/jinja/lexer.cpp
        ${_onebit_jinja_dir}/jinja/parser.cpp
        ${_onebit_jinja_dir}/jinja/runtime.cpp
        ${_onebit_jinja_dir}/jinja/value.cpp
        ${_onebit_jinja_dir}/jinja/string.cpp
        ${_onebit_jinja_dir}/jinja/caps.cpp
        # jinja/value.cpp includes the fork's unicode.h, and these are its implementation
        ${_onebit_jinja_dir}/unicode.cpp)
    target_include_directories(onebit_jinja PUBLIC ${_onebit_jinja_dir})
    target_link_libraries(onebit_jinja PUBLIC nlohmann_json::nlohmann_json)
    target_compile_definitions(onebit_jinja PUBLIC ONEBIT_JINJA=1)
    target_compile_definitions(onebit PRIVATE ONEBIT_JINJA=1)
    target_link_libraries(onebit PRIVATE onebit_jinja)
    message(STATUS "ONEBIT_JINJA=ON — the NPU chat route renders the container's own chat template")
else()
    message(STATUS "ONEBIT_JINJA=OFF — no chat-template renderer: the NPU chat route keeps ChatML (#963 option 1)")
endif()
if(ONEBIT_ORT)
    add_subdirectory(onnx)
    target_link_libraries(onebit PRIVATE onebit_ort_info onebit_ort_run)
    target_compile_definitions(onebit PRIVATE ONEBIT_ORT=1)
    message(STATUS "ONEBIT_ORT=ON — ONNX Runtime ${ONEBIT_ORT_PC_VERSION}: '1bit ort-info' + '1bit ort-run' + libonebit_ep.so")
endif()
# backends are compiled in by build-tree path; a package keeps that layout (app/built_path.h)
# Backend paths are embedded RELATIVE to the build dir (see app/built_path.h): a shipped binary
# must not name a build-machine path. Each define below is turned into its relative form first.
function(onebit_build_relative var out)
    file(RELATIVE_PATH _rel "${CMAKE_BINARY_DIR}" "${${var}}")
    set(${out} "${_rel}" PARENT_SCOPE)
endfunction()

# Runs everywhere, CI included: serve's proxy against a fake backend.
add_test(NAME smoke_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/smoke_serve.sh $<TARGET_FILE:onebit>)
# The embedding role served in-process from a 1-bit container (engine#1311): /v1/embeddings for a string and an
# array, the OpenAI shape, and the refusals named. Device-free on the 32-token fixture.
add_test(NAME embed_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/embed_serve.sh $<TARGET_FILE:onebit>)
# serve takes a hung backend down (docs/serve.md, "When a backend hangs"): a fake backend, no GPU.
add_test(NAME watchdog_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/watchdog_serve.sh $<TARGET_FILE:onebit>)
# serve's GET /metrics (docs/serve.md, "Metrics"): a fake backend, no GPU.
add_test(NAME serve_hangup COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_hangup_test.sh $<TARGET_FILE:onebit>)
# engine#1945: scripts/bump-hrx-sync.sh tells AMD's own commit from a line AMD rebased apart from a commit that
# is not AMD's. Local git repositories only: no network, no GPU.
add_test(NAME bump_hrx_sync COMMAND bash ${CMAKE_SOURCE_DIR}/tests/bump_hrx_sync_test.sh ${CMAKE_SOURCE_DIR}/scripts/bump-hrx-sync.sh)
# engine#2102: the sentence a serve owes an operator when no 16-row decode kernel was found (app/npu_decode_note.h). Pure:
# the silent cases are the ones that matter, and they are arms.
add_executable(npu_decode_note_test tests/npu_decode_note_test.cpp)
target_include_directories(npu_decode_note_test PRIVATE app)
add_test(NAME npu_decode_note COMMAND npu_decode_note_test)
add_dependencies(onebit npu_decode_note_test)
add_test(NAME bump_hrx_report COMMAND bash ${CMAKE_SOURCE_DIR}/tests/bump_hrx_report_test.sh ${CMAKE_SOURCE_DIR}/scripts/bump-hrx-report.sh)
add_test(NAME metrics_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/metrics_serve.sh $<TARGET_FILE:onebit>)
# serve --print-backend: the llama-server Lemonade starts itself, one hop (docs/serve.md): no GPU.
add_test(NAME print_backend COMMAND ${CMAKE_SOURCE_DIR}/tests/print_backend.sh $<TARGET_FILE:onebit>)
# scripts/lemonade-hrx: the PPA lemond's HRX recipe runs our llama-server (docs/lemonade.md): no GPU.
add_test(NAME lemonade_hrx COMMAND ${CMAKE_SOURCE_DIR}/tests/lemonade_hrx.sh $<TARGET_FILE:onebit>)
add_test(NAME parallel_args COMMAND ${CMAKE_SOURCE_DIR}/tests/parallel_args.sh $<TARGET_FILE:onebit>)
add_test(NAME recipes_route COMMAND ${CMAKE_SOURCE_DIR}/tests/recipes_route.sh $<TARGET_FILE:onebit>)
# os/mini/release-check.sh --self-test: the shipped release gate's own verdict rows, asserted (engine#1596). It needs no
# build and no device - the subject is the row functions - so it is the one test here that takes no $<TARGET_FILE>, and
# it is here because nothing else runs it: the rows had 128 assertions and no runner. It names no add_executable, so it
# adds nothing to the set of test targets the CI --target list would have to build (the direction-B measurement, #1595).
add_test(NAME release_check_selftest COMMAND bash ${CMAKE_SOURCE_DIR}/os/mini/release-check.sh --self-test)
# engine#1800: the 1bit OS base layer's mode normaliser and reporter (scripts/pack-elf.sh, engine#1474), run over a
# crafted tree. Same shape as the row above: no build, no device, no target, so it adds nothing to the CI --target list.
# It was in tests/ with nothing referencing it, so the property it pins (the normaliser only NARROWS modes: a 0600
# secret stays 0600 and /tmp keeps 1777) had no runner at all.
add_test(NAME base_modes COMMAND bash ${CMAKE_SOURCE_DIR}/tests/base_modes_test.sh ${CMAKE_SOURCE_DIR})
add_test(NAME template_pin COMMAND ${CMAKE_SOURCE_DIR}/tests/template_pin.sh $<TARGET_FILE:onebit>)
add_test(NAME greedy_shortcut COMMAND ${CMAKE_SOURCE_DIR}/tests/greedy_shortcut.sh $<TARGET_FILE:onebit>)
add_test(NAME device_route COMMAND ${CMAKE_SOURCE_DIR}/tests/device_route.sh $<TARGET_FILE:onebit>)
# where --device npu finds the GGUF repack script: beside 1bit in a package (scripts/package-linux.sh),
# else ONEBIT_Q4NX_REPACK; no NPU needed (the repack runs before the device)
add_test(NAME npu_repack_path COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_repack_path.sh $<TARGET_FILE:onebit>)
# a build without the NPU lane has no repack path: the test says skip (77) there, not fail
set_tests_properties(npu_repack_path PROPERTIES SKIP_RETURN_CODE 77)
# engine#812: the XDNA import class advertises device execution on an imported dma-buf. This pairs that
# advertisement (the pinned libamdf's import profile) with what the loaded driver actually does (libamdf's
# own CTS: GPU export -> NPU import -> NPU executes on the imported range), keyed on the module's
# fingerprint against third_party/patches/amdxdna/README.md. It skips (77) without the array or without a
# CTS build (ONEBIT_LIBAMDF_CTS_DIR), so public CI neither runs nor fails it.
# 1bit-update's slot decisions (engine#1334 review): the second update targets the slot holding the machine's own
# save layer and must keep it, a foreign layer is still refused. The script sourced as a library over fake /data
# layouts: no root, no mounts, no device, no VM.
add_test(NAME one_bit_update_slots
         COMMAND sh ${CMAKE_SOURCE_DIR}/tests/one_bit_update_slots_test.sh ${CMAKE_SOURCE_DIR})
add_test(NAME one_bit_network_conf
         COMMAND sh ${CMAKE_SOURCE_DIR}/tests/one_bit_network_conf_test.sh ${CMAKE_SOURCE_DIR})
add_test(NAME npu_import_contract
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_import_contract.sh ${CMAKE_SOURCE_DIR})
set_tests_properties(npu_import_contract PROPERTIES SKIP_RETURN_CODE 77)
# The advertisement half of that test is a parse of the pinned libamdf, so it is tested device-free here: the
# two fixtures are an advertising import profile and the same profile with the flag dropped from its dma-buf
# entry. Runs everywhere, needs no array, no GPU and no submodule (the pinned file is read only if present).
add_test(NAME npu_import_advert
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_import_contract.sh --self-test)
# engine#876: the gate row for a folded (pb16i/t2s) container.  Its device half needs the array, but its
# assertion does not: these are the accept/refuse cases of `scripts/npu-window11.sh --compare` on hand-written
# logs -- folded against folded passes, one differing rotated activation fails, a folded log against a plain one
# fails, and the plain row keeps its old ms tolerance.  No device, no model, no build products.
add_test(NAME npu_window11_compare COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_window11_compare.sh)
# the OS image's own release check: its lemonade row must prove the API key is enforced, not infer it
# from an empty body (os/mini/release-check.sh --self-test; no device and no build needed)
add_test(NAME release_check_row COMMAND ${CMAKE_SOURCE_DIR}/os/mini/release-check.sh --self-test)
# engine #574: the signed manifest of a sealed NPU kernel set. tools/seal_manifest.py writes it (sha256
# of the ciphertext, Ed25519 over the manifest) and checks it; the test is a throwaway set in a temp
# dir, so no key, no device and no build: the four refusals are a set with no manifest, a file the
# manifest does not list, a ciphertext byte flipped after signing, and a manifest signed by another key.
# the handoff driver must not orphan its server on SIGKILL (PR_SET_PDEATHSIG); no GPU, no model
add_test(NAME kvshare_orphan COMMAND ${CMAKE_SOURCE_DIR}/tests/kvshare_orphan_test.sh)
add_test(NAME seal_manifest COMMAND ${CMAKE_SOURCE_DIR}/tests/seal_manifest_test.sh)
# engine#1564: which files the copyright gate looks at -- a staged file is seen, an untracked one is seen (that is why
# running it before `git add` is useful), an ignored one is not. A throwaway repo in a temp dir: no build, no device.
add_test(NAME copyright_gate COMMAND ${CMAKE_SOURCE_DIR}/tests/copyright_gate_test.sh)
# engine#1701: the NPU hold's queue/token follow ONEBIT_HOLD_DIR, and its queue settings never reach the payload.
# Shell only (a stub gpu-hold plus the tree's own), no device, ~1 s.
add_test(NAME npu_hold COMMAND bash ${CMAKE_SOURCE_DIR}/tests/npu_hold_test.sh)
# engine#1696: the gpu_hold_* registrations live in cmake/tests-gpu-hold.cmake. It is no longer named here: every
# cmake/tests-*.cmake is included by the glob near the end of this file (engine#1979).
# engine#1485: the two port read-backs the route tests use (tests/serve_port.sh). Script-based, no device, no model --
# it feeds the parsers the exact lines `1bit serve` and `llama-server` print, because a read-back that silently returns
# empty turns into "the server did not come up" and is blamed on whatever the row under test checks.
add_test(NAME serve_port COMMAND bash ${CMAKE_SOURCE_DIR}/tests/serve_port_test.sh)
# engine#1844: the serve_context_limit row's classifier used to accept anything that was not 2xx, so a server that DIED
# mid-request (curl 000) or answered 5xx passed as "a refusal". The classifier and its pass set are pure functions of
# (http code, body), so their cases -- especially the ones that must reject -- run here with no model, no server and no
# device. Deliberately OUTSIDE if(ONEBIT_SERVE_TEST_GGUF): that row is device-gated, and the whole point is that the
# classification is checked in every configure, including the ones that never load a model.
add_test(NAME serve_context_limit_classify COMMAND bash ${CMAKE_SOURCE_DIR}/tests/serve_context_limit_test.sh --self-test)
# engine#1785: `serve --flash` names two settings (--ctx-size 1024, --prefix-pin on --device npu) and refuses what
# contradicts them. Every case ends in the option check, so the row needs no model, no device and no port.
add_test(NAME serve_flash_refusals COMMAND bash ${CMAKE_SOURCE_DIR}/tests/serve_flash_test.sh $<TARGET_FILE:onebit>)
# engine#1594: every target ci.yml's `cmake --build --target …` list names must exist in the configuration ci.yml
# creates. #1583 added one name for a target defined inside if(ONEBIT_NPU), which CI does not configure -- and ninja
# rejects the whole command line before building anything, so `onebit` stopped being compiled for every PR. Reads the
# two files, no build and no submodules; --self-test drives it.
# engine#1864: `find_package(Python3)` used to be at :761, long after the first `${Python3_EXECUTABLE}` users above, so those
# registrations carried an EMPTY interpreter and ctest executed each script directly -- which passes only while the file keeps
# its exec bit (a mode-only rewrite of tools/ci/hrx_build_scope.py turned that row into BAD_COMMAND). Find the interpreter
# first and register every Python test through one place: it runs the script THROUGH the interpreter, and on a machine
# without one the test is dropped with a status line instead of a .py being handed to the kernel as a program.
find_package(Python3 COMPONENTS Interpreter)
function(add_python_test name)
    if(Python3_Interpreter_FOUND)
        add_test(NAME ${name} COMMAND ${Python3_EXECUTABLE} ${ARGN})
    else()
        message(STATUS "ctest: not registering ${name} (no Python3 interpreter)")
    endif()
endfunction()
# engine#1864: the ordering rule itself, checked on the file (no configure, no build): every ${Python3_EXECUTABLE} use has
# to follow the find_package above. The nine that did not ran through their exec bit, and a mode-only rewrite of one
# script turned the row into BAD_COMMAND.
add_python_test(python_test_order_check ${CMAKE_SOURCE_DIR}/tests/python_test_order_check.py --check --root ${CMAKE_SOURCE_DIR})
add_python_test(python_test_order_check_self ${CMAKE_SOURCE_DIR}/tests/python_test_order_check.py --self-test)
add_python_test(ci_targets ${CMAKE_SOURCE_DIR}/tools/ci/ci_targets_check.py --check --root ${CMAKE_SOURCE_DIR})
add_python_test(ci_targets_self ${CMAKE_SOURCE_DIR}/tools/ci/ci_targets_check.py --self-test)
# engine#803 gate, second half: the comparator that turns two --dump-logits directories into the KLD verdict the
# P4 gate asks for. No device and no model, so the gate is one command when a ticket is free; the self-test pins
# the four cases a KLD implementation gets wrong (identical -> exactly 0, a shift is positive and directional, a
# uniform shift stays 0, and an asymmetric file set is refused by name).
add_python_test(logits_kld_self ${CMAKE_SOURCE_DIR}/tools/logits_kld.py --self-test)
# engine#2018: this one was in tools/ci/unreferenced_py_tests.txt as a baseline entry, and it should not have been: the
# file is portable and passes (worst-case |sum| 11.18e6 inside int32 at K=11008, 4 arms), it just never had a runner. A
# baseline records a test as never-run, which for a passing test is the wrong outcome -- so it is registered and the line
# is removed, which is the shrink the baseline's own comment asks for.
add_python_test(moe_expert_chunks_accum_test ${CMAKE_SOURCE_DIR}/tests/moe_expert_chunks_accum_test.py)
# engine#2018: the last baseline entry. The file recorded it as "needs the benchmark harness", but the test stubs
# llama-bench (stdlib only: hashlib/importlib.util/json/pathlib/platform/sys) and passes on a plain interpreter, so it is a
# test with no runner rather than a test that cannot run here. Registered, and the baseline is now empty: from here the
# sweep is zero-tolerance instead of a record of known gaps.
add_python_test(hyperloom_benchmark_test ${CMAKE_SOURCE_DIR}/tests/hyperloom_benchmark_test.py)
# engine#1820: the other half of ci_targets_check -- a standalone shell test that nothing references cannot fail.
# `need` is every tests/*.sh, `have` is every file that mentions its basename somewhere that can run it, and
# tools/ci/unreferenced_tests.txt carries the deliberate exclusions WITH their reasons, so the list can shrink.
add_python_test(tests_referenced ${CMAKE_SOURCE_DIR}/tools/ci/tests_referenced_check.py --check --root ${CMAKE_SOURCE_DIR})
add_python_test(tests_referenced_self ${CMAKE_SOURCE_DIR}/tools/ci/tests_referenced_check.py --self-test)
# engine#1901: five tools ship a --self-test that no runner named, so the arms existed, were maintained beside the logic they
# check, and never executed. All five run device-free and fast (58-277 ms measured), so they are rows rather than allow-list
# excuses, and tools/ci/tests_referenced_check.py now counts tools/**/*.py that declare one and are named by no RUNNER
# (CMakeLists/cmake/workflows -- a citation in another tool's source is not a runner: that is the 5-vs-2 difference).
# Registered through add_python_test (engine#1864), so they run THROUGH the interpreter: my first version used the raw
# `${Python3_EXECUTABLE}` form above the find_package and the repository's own ordering guard caught it by name --
# tools/mtp_head_ref.py is the one of the five without an exec bit, and it became BAD_COMMAND under ctest.
# engine#1962: the narrowing-body enumeration, as a ratchet. The corpus is a submodule, so the check
# skips (rc 77) where third_party/llama.cpp is not checked out; the baseline file is in the engine tree,
# so the recorded set is reviewable everywhere.
add_python_test(tool_selftest_hrx_f16_narrowing ${CMAKE_SOURCE_DIR}/tools/ci/hrx_f16_narrowing.py --self-test)
add_test(NAME hrx_f16_narrowing_check COMMAND ${Python3_EXECUTABLE}
         ${CMAKE_SOURCE_DIR}/tools/ci/hrx_f16_narrowing.py --check --root ${CMAKE_SOURCE_DIR})
set_tests_properties(hrx_f16_narrowing_check PROPERTIES SKIP_RETURN_CODE 77)
add_python_test(tool_selftest_batched_repeat ${CMAKE_SOURCE_DIR}/tools/batched_repeat.py --selftest)
add_python_test(tool_selftest_logprob_agree ${CMAKE_SOURCE_DIR}/tools/logprob_agree.py --selftest)
add_python_test(tool_selftest_mtp_head_ref ${CMAKE_SOURCE_DIR}/tools/mtp_head_ref.py --selftest)
add_python_test(tool_selftest_spec_decode_acceptance ${CMAKE_SOURCE_DIR}/tools/spec_decode_acceptance.py --selftest)
add_python_test(tool_selftest_whisper_encoder_baseline ${CMAKE_SOURCE_DIR}/tools/whisper_encoder_baseline.py --self-test)
# engine#1780: the DFlash draft package's gate before a loader -- tile-grid sizes from the config, tile-header sanity and
# the container's scalars against config.json, on a synthetic package with eight negative controls. Stdlib only.
add_python_test(tool_selftest_dflash_draft_info ${CMAKE_SOURCE_DIR}/tools/dflash_draft_info.py --selftest)
# engine#1750: what a TDR recovery did, counted from the kernel log (failed context restarts among them). The self-test
# parses lines in the driver's own formats for three module generations. Stdlib only, no device.
add_python_test(tool_selftest_amdxdna_tdr_report ${CMAKE_SOURCE_DIR}/tools/amdxdna_tdr_report.py --self-test)
# engine#2013: the qwen35 (gated linear attention) GGUF -> Q4NX converter. Its own selftest (tile round-trips, the
# value-head permutation), and a pack + verify of a small qwen35 GGUF the test writes itself, with the three inputs it
# has to refuse. numpy only: no gguf package, no model, no device.
add_python_test(tool_selftest_qwen35_to_q4nx ${CMAKE_SOURCE_DIR}/tools/qwen35_to_q4nx.py selftest)
add_python_test(qwen35_to_q4nx ${CMAKE_SOURCE_DIR}/tests/qwen35_to_q4nx_test.py)
# engine#1777: the release row's own arm, which merged unregistered -- this sweep is what found it (engine#1820).
add_test(NAME release_check_hrx_init_reason COMMAND bash ${CMAKE_SOURCE_DIR}/tests/release_check_hrx_init_reason_test.sh ${CMAKE_SOURCE_DIR}/os/mini/release-check.sh)
# engine#1615: every process holding the GPU/NPU device must belong to a live gpu-hold/npu-hold ticket. Three
# orphans of that shape landed in two days (51 GiB of GTT for 14 h; an NPU server 70 min past a 6-min ticket;
# then two lanes each starting an HRX server outside the token within one half hour). The ancestry test is the
# check; the two permanent services are exempt by name. --self-test drives the decision, no device needed.
add_python_test(device_holders ${CMAKE_SOURCE_DIR}/tools/device_holders.py --self-test)
# engine #1668: the bench preflight must pin the load mode, because `-lm none` is +17.5 % prefill at -1.6 % decode for +604 MiB of
# GTT. The script existed with a test that nothing ran; a test nobody runs is a comment, so the loader/flag assertions are registered
# here (the test fakes the box, so it needs no device and no ticket).
add_test(NAME bench_preflight COMMAND bash ${CMAKE_SOURCE_DIR}/tools/tests/test-bench-preflight.sh)
# engine#1702: builds a per-layer MoE expert placement table from a moe_trace, and refuses a prefill-only trace because
# placement is a decode lever (5.4-6.9 % on batched prefill against dedup's 90-94 %). --self-test drives it on synthetic
# traces with a known answer; no model, no device.
# engine#1990: the padded-head row arithmetic, driven on the shape that broke it (CodeLlama-7b's I8 [16016, 6144] for a
# 32,032-row head at kpad 4,096). It is header-only (`npu/forward/include/pad_head_rows.h`) precisely so it can live HERE:
# an earlier version linked `onebit_npu_forward`, which exists only under `if(ONEBIT_NPU)`, so CI -- which configures
# without the NPU runtime -- never built or ran the arm. `add_dependencies(onebit ...)` keeps it in the build CI runs.
add_executable(npu_pad_head_rows_test tests/npu_pad_head_rows_test.cpp)
target_include_directories(npu_pad_head_rows_test PRIVATE npu/forward/include)
add_dependencies(onebit npu_pad_head_rows_test)
add_test(NAME npu_pad_head_rows COMMAND npu_pad_head_rows_test)

# engine#1934: reading ONE tensor out of a safetensors shard needs `data_offsets` measured from 8 + header_len, not
# from byte 8. Getting that wrong returns a *different* tensor of the same size that still looks finite, and it cost
# two false conclusions about a pack in one night. `--self-test` drives the rule on a synthetic shard (no network):
# the same offset from the data section returns the tensor, from byte 8 it does not.
add_python_test(hf_tensor_self ${CMAKE_SOURCE_DIR}/tools/hf_tensor.py --self-test)
# engine#1946: a test that returns 77 is green, and green-because-it-did-not-run is the failure this fleet keeps paying
# for. `tools/ci/ctest_skips.py` reads a ctest log and reports every test that did not execute -- `***Skipped` (77) against
# `tools/ci/ctest_skips.txt`, and `(Not Run)` (a missing binary) as a build-list mistake that can never be baselined.
# Self-test only here: wiring it into ci.yml's Test step needs the log plus a `workflow`-scoped token (engine#1817).
add_python_test(ctest_skips_self ${CMAKE_SOURCE_DIR}/tools/ci/ctest_skips.py --self-test)
# engine#2106 / #2102: the release gate's NPU serve check asserts the fast route, a decode floor, the first token and a
# non-empty answer, because a serve on the slow per-op route still answers. --self-test drives the judging on the shape
# of each failure (the silent fallback at 1.9 tok/s, thinking left on, a degenerate reply); no device, no binary.
add_python_test(npu_serve_smoke_self ${CMAKE_SOURCE_DIR}/tools/npu_serve_smoke.py --self-test)
add_python_test(moe_placement ${CMAKE_SOURCE_DIR}/tools/moe_placement.py --self-test)
# The three ways this corpus loses a value to half precision (engine #1962/#2127/#2186): an input
# truncated without a prescale, a computed value published to f16, and an f16 accumulator updated by a
# multiply -- the last has no conversion instruction at all, which is why a truncation scan cannot see it.
add_python_test(hrx_f16_narrowing ${CMAKE_SOURCE_DIR}/tools/hrx_f16_narrowing_check.py --self-test)
# engine#1313 item 3: the "after" of "before/after routing accuracy" -- a ridge head over --dump-encoder output, with the
# table's own logits as the before arm. Self-test only, so it needs no dump and no model.
add_python_test(laya_head_finetune ${CMAKE_SOURCE_DIR}/tools/laya_head_finetune.py --self-test)
# engine#856: one HRX process's GTT split into the loader's copy and HRX's packed second copy, from its own -v log.
add_python_test(hrx_gtt_split_self ${CMAKE_SOURCE_DIR}/tools/hrx_gtt_split.py --self-test)
# engine#2143: GGML_HRX_LOG_DISPATCH stays a logging switch - read only in the two static void dispatch log functions
# of the pinned fork (skips the tree check, not its self-test, when the submodule is not checked out).
add_python_test(hrx_log_dispatch_neutral ${CMAKE_SOURCE_DIR}/tests/hrx_log_dispatch_neutral_test.py)
# engine#679: time per HRX dispatch from the backend's serial trace (the id hash and the per-kernel sums).
add_python_test(hrx_dispatch_profile_self ${CMAKE_SOURCE_DIR}/tools/hrx_dispatch_profile.py --self-test)
# engine #1279: the version a package is built with (scripts/pack-version.sh) -- a release tag as
# it stands, a candidate mapped into the ISO week of its own date with a suffix that sorts below that
# release, and the ordering a candidate needs: previous release < candidate < its release. The
# ordering is the row dpkg, rpm and pacman had to agree on; no container, no build, dpkg only.
add_test(NAME pack_version COMMAND ${CMAKE_SOURCE_DIR}/tests/pack_version_test.sh)
# engine #1461/#1411: the two release rows that read a PACKED tree (scripts/pack-check.sh). Synthetic trees
# built with the C compiler: a build-machine path in an ELF, an absolute RUNPATH, a relative rpath that
# resolves outside the payload, and a runtime set shipped without the aqlprofile agent libhsa opens. Each
# arm asserts the row's message, and a clean tree is the control. No package build, no GPU, runs in CI.
add_test(NAME pack_check COMMAND ${CMAKE_SOURCE_DIR}/tests/pack_check_test.sh)
# engine#2106: the hardware gate's `npuserve` phase, end to end against the stand-in serve tools/npu_serve_smoke.py
# writes: a fast-route serve passes the gate, a slow-route one fails it while its answers are fine, a dead one fails,
# and an unconfigured or NPU-less host skips. No device, no 1bit binary, and the fleet's queues are not touched.
add_test(NAME hw_gate_npuserve COMMAND ${CMAKE_SOURCE_DIR}/tests/hw_gate_npuserve_test.sh)
# Both start a loopback serve on a port picked by binding port 0 and releasing it, so they take the lock every
# loopback-server test takes (tests/route_port_lock_test.py): two of them racing for one freed port is a flaky failure.
set_tests_properties(hw_gate_npuserve npu_serve_smoke_self PROPERTIES RESOURCE_LOCK route_ports)
add_test(NAME check_ci_runners COMMAND ${CMAKE_SOURCE_DIR}/tests/check_ci_runners_test.sh)
# engine #1504: the release preflight blocks on a runner unit that is down and only warns on a short restart policy.
# The real scripts/ci-runner-health.sh and weekly.sh's preflight_runner_units() against a stub systemctl; no systemd.
add_test(NAME weekly_runner_preflight COMMAND ${CMAKE_SOURCE_DIR}/tests/weekly_runner_preflight_test.sh)
add_test(NAME ci_runner_health_selftest COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/ci-runner-health.sh --self-test)
# engine #1750: the carried amdxdna series applies in order to the pinned driver, and the context-restart code it
# produces fails a context's jobs (and logs it) when the restart fails, instead of leaving them parked or running
# them at close on a half-built firmware context. The driver's own functions in a user-space harness: no NPU, no
# kernel headers. Skips (77) where the driver source is not checked out (third_party/xdna-driver or ONEBIT_XDNA_SRC).
add_test(NAME amdxdna_restart COMMAND bash ${CMAKE_SOURCE_DIR}/tests/amdxdna_restart_test.sh ${CMAKE_SOURCE_DIR})
set_tests_properties(amdxdna_restart PROPERTIES SKIP_RETURN_CODE 77)
# engine #713: the pinned HRX toolchain (config/hrx-toolchain.config, scripts/fetch-rocm-sdk.sh) -- the
# pin file's refusals, the completeness check against a fabricated tree, and that cmake and
# weekly.sh both use the pin instead of the box's /opt/rocm-therock. No network, no SDK, no GPU.
add_test(NAME hrx_toolchain_pin COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_toolchain_pin.sh)
add_test(NAME pm_route COMMAND ${CMAKE_SOURCE_DIR}/tests/pm_route.sh $<TARGET_FILE:onebit>)
add_test(NAME bench_selftest COMMAND ${CMAKE_SOURCE_DIR}/tests/bench_selftest.sh $<TARGET_FILE:onebit>)
add_test(NAME prism_route COMMAND ${CMAKE_SOURCE_DIR}/tests/prism_route.sh $<TARGET_FILE:onebit>)
add_test(NAME laya_gguf_route COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_gguf_route.sh $<TARGET_FILE:onebit>)
# The routing table that ships and the collection.router policy it desugars into (docs/laya-first.md §4).
add_test(NAME laya_first COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_first.sh $<TARGET_FILE:onebit>)
add_test(NAME mem_admit_route COMMAND ${CMAKE_SOURCE_DIR}/tests/mem_admit_route.sh $<TARGET_FILE:onebit>)
# engine#315: what --hrx-cwsr-placement puts in an HRX backend's environment (app/hrx_cwsr.h); fake backend, no GPU.
add_test(NAME hrx_cwsr_route COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_cwsr_route.sh $<TARGET_FILE:onebit>)
# engine#713 criterion 4 end to end without a GPU: a fake runtime file and a fake backend, the line
# serve prints and the IREE_HAL_AMDGPU_LIBHSA_PATH it hands over, sha256 included.
add_test(NAME hrx_runtime_route COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_runtime_route.sh $<TARGET_FILE:onebit>)
# --device hybrid (docs/hybrid.md): the handoff with a fake backend and a fake provider; and, with a model,
# end to end on the CPU llama-server (ONEBIT_SERVE_TEST_LLAMA_SERVER, else the build's own; skips without one)
add_test(NAME hybrid_route COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_route.sh $<TARGET_FILE:onebit>)
# target 1's shared route (ONEBIT_KV_SHARE=1): the fake backend speaks ?action=share/adopt and the fake
# provider fails unless it was handed one readable fd per region -- so the socket, the inherited fds, the
# share_id and the fallback are all pinned with no model, no GPU and no dma-buf. Needs no exporter, so CI can
# run it (the GPU gate in ~/wt/zc-probes cannot).
add_test(NAME hybrid_kvshare_route COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_kvshare_route.sh $<TARGET_FILE:onebit>)
set(ONEBIT_SERVE_TEST_LLAMA_SERVER "" CACHE FILEPATH "llama-server for hybrid_e2e (default: the build's own)")
set(ONEBIT_KVSHARE_PREFILL "" CACHE FILEPATH "reference provider for hybrid_kvshare_e2e (the fork's llama-kvshare-prefill; empty skips)")
if(ONEBIT_SERVE_TEST_GGUF)
    add_test(NAME hybrid_e2e
             COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_SERVE_TEST_GGUF} ${ONEBIT_SERVE_TEST_LLAMA_SERVER})
    set_tests_properties(hybrid_e2e PROPERTIES SKIP_RETURN_CODE 77)
endif()
# The NPU route's <think> split into reasoning_content (app/think_split.h); header only, runs in CI.
add_executable(think_split_test tests/think_split_test.cpp)
target_include_directories(think_split_test PRIVATE app)
add_test(NAME think_split COMMAND think_split_test)
# engine#2135: what enable_thinking can mean for a pack, read off its own chat template (app/think_template.h).
add_executable(think_template_test tests/think_template_test.cpp)
target_include_directories(think_template_test PRIVATE app)
add_test(NAME think_template COMMAND think_template_test)
# ci.yml's build list is not ours to edit without `workflow` scope, so the test reaches CI as a build dependency of a
# target it already builds (tools/ci/ci_targets_check.py follows this edge).
add_dependencies(onebit think_template_test)
# Host / Origin checks every engine HTTP server applies (app/http_guard.h); header only, runs in CI.
# engine#1779 (F10 fault 2): the degenerate-repetition predicate at the context limit. Header only, no device.
add_executable(serve_degenerate_test tests/serve_degenerate_test.cpp)
target_include_directories(serve_degenerate_test PRIVATE app)
add_test(NAME serve_degenerate COMMAND serve_degenerate_test)
# Built with `onebit`, which CI's --target list already names: a test target of its own would need a line in
# .github/workflows/ci.yml, and a PR that edits a workflow cannot be merged by a token without the `workflow` scope.
add_dependencies(onebit serve_degenerate_test)

add_executable(http_guard_test tests/http_guard_test.cpp)
target_include_directories(http_guard_test PRIVATE app)
add_test(NAME http_guard COMMAND http_guard_test)
# The "the device is gone" classifiers of app/serve_watchdog.h: a backend's device-lost stderr line, and the
# NPU runtime's failed-runlist message that makes `1bit unified` answer the request and exit non-zero. Header
# only, no device, no model.
add_executable(serve_watchdog_test tests/serve_watchdog_test.cpp)
target_include_directories(serve_watchdog_test PRIVATE app)
add_test(NAME serve_watchdog COMMAND serve_watchdog_test)
# The NPU route's sampling controls and GBNF output constraint (app/npu_sampling.h,
# app/npu_grammar.h, docs/npu.md "Sampling and structured output"); header only, no device.
# engine#1781: the comparator behind NPU_BATCH_CHECK -- pure arithmetic over two float arrays, so it runs with no model,
# no device and no runtime. It is a header, so the test includes it directly and needs no link libraries.
add_executable(npu_sampling_test tests/npu_sampling_test.cpp)
# The shape rule for a 1.8 external-data initializer (npu/pack_initializer_shape.h, engine#1503's graph-initializer
# reader). Those packs zero every `dims`, so a byte length has to become a shape -- and a shape inferred wrongly imports
# cleanly and serves garbage, so the refusals are the arms worth pinning. Header-only, no parser, no pack, no device.
add_executable(pack_initializer_shape_test tests/pack_initializer_shape_test.cpp)
target_include_directories(pack_initializer_shape_test PRIVATE npu)
add_test(NAME pack_initializer_shape COMMAND pack_initializer_shape_test)
add_dependencies(onebit pack_initializer_shape_test)
target_include_directories(npu_sampling_test PRIVATE app)
target_link_libraries(npu_sampling_test PRIVATE nlohmann_json::nlohmann_json)
# The speculative round's bookkeeping (app/spec_decode.h, engine#1781): a drafter's proposals against the verifier's
# per-row logits, with the acceptance rule from npu/mtp/spec_accept.h (#1958, merged). Header-only and pure -- the
# uniforms come from the caller, so every arm is a hand-computed value and there is no model, device or RNG here.
# `add_dependencies(onebit ...)` because a target ci.yml never builds is the failure that check exists to catch.
add_executable(spec_decode_test tests/spec_decode_test.cpp)
# ${CMAKE_SOURCE_DIR} as well as app/, because app/spec_decode.h includes "npu/mtp/spec_accept.h" by its repo-relative
# path -- which is how every other consumer of that header spells it. Without the root on this target's path the build
# fails with "npu/mtp/spec_accept.h: No such file or directory", measured on CI run 42935896650 (the risk the PR body
# named, so it cost one run rather than a review).
target_include_directories(spec_decode_test PRIVATE app ${CMAKE_SOURCE_DIR})
add_test(NAME spec_decode COMMAND spec_decode_test)
add_dependencies(onebit spec_decode_test)
# The reader for a tools/moe_placement.py table (app/moe_placement.h, engine#1702): a consumer that splits experts across
# devices adopts a table only if it was written for this model -- schema, group count, every routed layer, every expert
# id exactly once -- and only on the held-out share, never the in-sample one. Header only, no device, no model.
add_executable(moe_placement_test tests/moe_placement_test.cpp)
target_include_directories(moe_placement_test PRIVATE app)
target_link_libraries(moe_placement_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME moe_placement_reader COMMAND moe_placement_test)
add_dependencies(onebit moe_placement_test)
# The NPU route's bounded load retry (app/npu_load_retry.h, engine #1010, docs/npu.md "When a load follows an evict"):
# a load started right after an evict can hit the previous backend's hardware context still being torn down, which used
# to be fatal on the first attempt. Header only, no device, no model.
add_executable(npu_load_retry_test tests/npu_load_retry_test.cpp)
target_include_directories(npu_load_retry_test PRIVATE app)
add_test(NAME npu_load_retry COMMAND npu_load_retry_test)

# The optional-op fallback seam (npu/model/op_dispatch.h, engine#803 A3): every op but `gemv` is optional, and the
# model runs its own implementation when the target does not take it -- including when the target claims it and
# fails. Header only, stub targets, no device and no runtime library, so it runs in CI.
add_executable(npu_op_dispatch_test tests/npu_op_dispatch_test.cpp)
target_include_directories(npu_op_dispatch_test PRIVATE npu)
add_test(NAME npu_op_dispatch COMMAND npu_op_dispatch_test)
# ci.yml builds a named list of targets, so a test target that is not in it reports "Not Run" (#635); this lane's
# token cannot edit the workflow. Hooking onto one that IS in the list gets it built there too.
add_dependencies(npu_sampling_test npu_op_dispatch_test)

add_test(NAME npu_sampling COMMAND npu_sampling_test)
# Same reason as the other device-free tests: ci.yml builds a named list of targets, so a test whose target is
# not built reports "Not Run" (#635). This lane's token cannot edit the workflow.
add_dependencies(npu_sampling_test serve_watchdog_test)
add_dependencies(npu_sampling_test npu_load_retry_test)
# The MTP draft loop (npu/mtp/draft.h): draft k tokens, verify them as one batch, keep the longest prefix
# the base model agrees with, roll its state back.  Policy only -- a toy base model and toy draft heads stand
# in for the forward and the MTP head, so the invariants (identical text whatever the head proposes, exact
# rollback, k clamped to the batch width) are checked on every build without a device.
add_executable(mtp_draft_test tests/mtp_draft_test.cpp npu/mtp/mtp_head.cpp)
target_include_directories(mtp_draft_test PRIVATE npu)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(mtp_draft_test PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME mtp_draft COMMAND mtp_draft_test)
# engine#1781 gate 2: the distribution-following acceptance rule (accept with min(1, p/q), resample the residual at the
# first rejection), with the temperature-0 identity and a deliberately wrong drafter whose output must still follow p.
# Header-only and device-free; pulled into CI through npu_sampling_test for the same reason as mtp_draft_test below.
add_executable(spec_accept_test tests/spec_accept_test.cpp)
target_include_directories(spec_accept_test PRIVATE npu)
add_test(NAME spec_accept COMMAND spec_accept_test)
add_dependencies(npu_sampling_test spec_accept_test)
# ci.yml's build step lists every ctest target explicitly, and this branch cannot edit that file (the push
# token has no `workflow` scope), so the binary is pulled in as a dependency of a listed test target. When a
# workflow-scoped push is available, add `mtp_draft_test` to ci.yml's build list and delete this line.
add_dependencies(npu_sampling_test mtp_draft_test)
# Who may have the NPU (app/npu_device_lock.h, engine#612, docs/npu.md "Two streams on one
# NPU"): the exclusive default and the shared --npu-streams >= 2 opt-in, on a lock file in a
# temp dir; header only, no device, runs in CI. Since engine#1241 the same test forks a live
# holder and proves the wait names it after ~2 s and ONEBIT_NPU_LOCK_WAIT_MAX fails fast with it.
add_executable(npu_device_lock_test tests/npu_device_lock_test.cpp)
target_include_directories(npu_device_lock_test PRIVATE app)
add_test(NAME npu_device_lock COMMAND npu_device_lock_test)
# The snapshot's two kinds of plane (npu/forward/include/q4nx_state_planes.h): the KV cut to the positions being
# snapshotted, and the GDN conv window and recurrent state carried whole, since those are one fixed block per layer
# that every position updates rather than appends to. Header only, no device, runs in CI. Regression it pins: the
# whole planes used to be cut with the per-position count, so restore_state refused every snapshot the prefix cache
# offered (`slot has no usable state`) and every shared-prefix request re-prefilled the prompt from 0.
add_executable(q4nx_state_planes_test tests/q4nx_state_planes_test.cpp)
target_include_directories(q4nx_state_planes_test PRIVATE npu/forward/include app)
add_test(NAME q4nx_state_planes COMMAND q4nx_state_planes_test)
# ci.yml's build step lists every ctest target explicitly and this branch cannot edit that file (the push token has
# no `workflow` scope), so as with mtp_draft_test the binary is pulled in as a dependency of a listed target.
add_dependencies(npu_sampling_test q4nx_state_planes_test)

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
# engine#713 criterion 4's runtime identity (app/hrx_runtime.h, docs/hrx.md "The HSA runtime"):
# sha256 of a buffer and of a file, and the line that names the runtime HRX will dlopen; FIPS 180-4
# vectors, no GPU, no runtime.
add_executable(hrx_runtime_test tests/hrx_runtime_test.cpp)
target_include_directories(hrx_runtime_test PRIVATE app)
add_test(NAME hrx_runtime COMMAND hrx_runtime_test)
# --pm-router's class -> expert table and its flag values (app/pm.cpp, docs/pm.md "Routing with
# Laya"); no GPU, no checkpoint, no Lemonade: config in, mapping out.
add_executable(pm_routes_test tests/pm_routes_test.cpp app/pm.cpp)
target_include_directories(pm_routes_test PRIVATE app ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(pm_routes_test PRIVATE nlohmann_json::nlohmann_json httplib::httplib Threads::Threads)
add_test(NAME pm_routes COMMAND pm_routes_test)

# ── `1bit serve` end to end (docs/serve.md) ───────────────────────────────
# Needs a GGUF (e.g. Qwen3-0.6B Q4_K_M); tests each device this build has.
set(ONEBIT_SERVE_TEST_GGUF "" CACHE FILEPATH "GGUF the serve_e2e tests load")

# ── ONNX Runtime in-process (docs/onnx.md) ───────────────────────────────
# Off by default: it needs the libonnxruntime 1.23.2 dev package (pkg-config libonnxruntime) and
# nothing in CI or a normal engine build should depend on it. ON adds `1bit ort-info` (the pin and
# the EP roster this binary actually has) and links onnx/, which also builds libonebit_ep.so.
option(ONEBIT_ORT "Link ONNX Runtime 1.23.2 in-process (1bit ort-info + the plugin EP)" OFF)

# ── the ONNX EP's NPU leg compiles and links without ORT, the device or the runtime (#874) ───────
# npu/ort/onebit_npu_gemm.cpp is built by onebit_npu_forward (the embedding route's device path, engine#1311)
# and by onnx/ep when ONEBIT_EP_NPU=ON, which needs ONNX
# Runtime *and* the private NPU runtime -- so no configuration CI has ever built it, and #874 (a call
# to q4nx_elf_arg_sizes with no declaration, then the file defining it missing from the link) was
# green for a week. This target compiles onnx/ep's own source list (onnx/ep/npu_sources.cmake, the
# list the EP builds) and links it against onnx/ep/npu_runtime_ci_stub.cpp instead of the private
# runtime, so a broken leg turns the `build` job red. `onebit` depends on it because that is the
# target the CI build step names; nothing here is linked into `onebit` or into libonebit_ep.so.
if(NOT WIN32 AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/onnx/ep/npu_sources.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/onnx/ep/npu_sources.cmake)
    add_executable(onebit_ep_npu_linkcheck ${ONEBIT_EP_NPU_COMMON_SRC}
        onnx/ep/npu_runtime_ci_stub.cpp onnx/ep/npu_linkcheck.cpp)
    target_include_directories(onebit_ep_npu_linkcheck PRIVATE ${ONEBIT_EP_NPU_COMMON_INCS})
    target_compile_definitions(onebit_ep_npu_linkcheck PRIVATE ${ONEBIT_EP_NPU_COMMON_DEFS})
    add_dependencies(onebit onebit_ep_npu_linkcheck)
endif()

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
    # ONEBIT_HRX_LIBHSA (the auto-detected path in the cache) is NOT compiled in. It is a *build-machine* path by
    # construction - cmake/hrx.cmake globs it out of ONEBIT_HRX_TOOLCHAIN - so baking it (a) makes the bytes depend
    # on where the build ran, (b) writes the builder's tree into a public tarball (the c33 roll of 2026-11-08 failed
    # its own package gate on the one string /home/<builder>/wt/.../libhsa-runtime64.so.1), and (c) names a location
    # the target does not have: the appliance installs the runtime at /opt/rocm-therock/lib and the distro packages
    # at /opt/1bit/engine/hrx/llama/bin, so the baked path never resolves there and serve falls through to its scan
    # anyway (app/serve.cpp:212-224; docs/hrx.md). The cache value stays build metadata - os/mini/mkimage.sh reads
    # it to decide which runtime to SHIP - and ONEBIT_HRX_LIBHSA_BAKE is the explicit way to put a path into the
    # binary, for a builder whose layout is the target's.
    if(ONEBIT_HRX_LIBHSA_BAKE)
        target_compile_definitions(onebit PRIVATE ONEBIT_HRX_LIBHSA="${ONEBIT_HRX_LIBHSA_BAKE}")
    endif()
    target_compile_definitions(onebit PRIVATE ONEBIT_HRX_SERVER="${_onebit_hrx_server}"
                               ONEBIT_HRX_ALLOW_NO_SVM=${ONEBIT_HRX_ALLOW_NO_SVM})
    if(ONEBIT_SERVE_TEST_GGUF)
        foreach(_dev hrx cpu)
            add_test(NAME serve_e2e_${_dev}
                     COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_SERVE_TEST_GGUF} ${_dev})
            # hrx skips without a gfx1151 (ONEBIT_HRX_E2E_TARGETS)
            set_tests_properties(serve_e2e_${_dev} PROPERTIES SKIP_RETURN_CODE 77)
        endforeach()
        # Placement must not change the answer (engine #674): the same greedy prompt through the CPU, through the device
        # with every layer assigned to the CPU, and fully offloaded, compared byte for byte.
        add_test(NAME hrx_cpu_weight_parity
                 COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_cpu_weight_parity.sh ${ONEBIT_HRX_SERVER} ${ONEBIT_SERVE_TEST_GGUF})
        set_tests_properties(hrx_cpu_weight_parity PROPERTIES SKIP_RETURN_CODE 77)
        # Target 1 with real dma-bufs (docs/zero-copy.md, "Target 1"): the decoder exports its KV region, a
        # separate provider process receives the fds over SCM_RIGHTS and prefills into them, and the decoder
        # adopts the cells with no data movement. Registered here, after include(cmake/hrx.cmake), because
        # ONEBIT_HRX_SERVER is only defined there -- reading it earlier drops the argument on a first configure
        # and shifts the rest. Needs the fork's reference provider, which this build does not produce: pass
        # ONEBIT_KVSHARE_PREFILL or the test skips (77), as it does without a gfx1151.
        add_test(NAME hybrid_kvshare_e2e
                 COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_kvshare_e2e.sh "${ONEBIT_HRX_SERVER}" "${ONEBIT_SERVE_TEST_GGUF}" "${ONEBIT_KVSHARE_PREFILL}")
        set_tests_properties(hybrid_kvshare_e2e PROPERTIES SKIP_RETURN_CODE 77)
    endif()

endif()

    # engine#1779: these two are registered OUTSIDE both `if(ONEBIT_SERVE_TEST_GGUF)` and `if(ONEBIT_HRX)` on purpose.
    # Inside the GGUF guard they did not exist in CI at all (a configure-time cache path CI never sets: the build job
    # runs 164 tests and none of the serve_* family is among them, so a gap in the family looked like coverage), and
    # inside the HRX guard they would only exist for an HRX build while ci.yml configures without ONEBIT_HRX. They skip
    # (77) when the model path is empty, which is a visible skip in the CI table, not an absent test. `onebit` is an
    # unconditional target (CMakeLists.txt:97), so $<TARGET_FILE:onebit> resolves in any configuration.
    # engine#1779 fault 1 (FLM 1.0.12: "a server that stopped accepting connections after clients hung up"): a
    # client that disconnects mid-response, repeated, then a fresh client must still be served. The assertion is
    # about the LISTENER, so an interrupted request is allowed to get nothing. `cpu` on purpose: the fault is
    # server logic, and the issue's rule is that these four are tested without a device where possible.
    add_test(NAME serve_client_hangup_cpu
             COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_client_hangup_test.sh $<TARGET_FILE:onebit>
                     "${ONEBIT_SERVE_TEST_GGUF}" cpu)
    set_tests_properties(serve_client_hangup_cpu PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK route_ports)
    # engine#1779 fault 2 (FLM 1.0.12: "Qwen3.5 answering `//////` after hitting the context limit"): a prompt that
    # fills the context is served, the next turn cannot fit, and the answer must be a refusal or a truncated-but-clean
    # reply - never one character repeated. Also `cpu`: the fault is in the request path's clamp, not in a kernel.
    add_test(NAME serve_context_limit_cpu
             COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_context_limit_test.sh $<TARGET_FILE:onebit>
                     "${ONEBIT_SERVE_TEST_GGUF}" cpu)
    set_tests_properties(serve_context_limit_cpu PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK route_ports)
# engine#1779 fault 3 ("wrong vectors from embed-gemma"): tested on the route where OUR code computes the vector -
# `serve.cpp:2264 if (embed_serve::container(model))` serves a container directory in-process - not on the
# `rag_launch()` proxy, where the numbers come from a delegated llama-server. Same registration rule as the two
# above (outside both guards, skip 77 when unset); ONEBIT_EMBED_TEST_CONTAINER names a directory holding model.embed.
set(ONEBIT_EMBED_TEST_CONTAINER "" CACHE PATH "1-bit encoder container (.embed dir) the embed-vector test loads")
add_test(NAME serve_embed_vectors_cpu
         COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_embed_vectors_test.sh $<TARGET_FILE:onebit>
                 "${ONEBIT_EMBED_TEST_CONTAINER}" cpu)
set_tests_properties(serve_embed_vectors_cpu PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK route_ports)

# ── DwarfStar is gone (owner 2026-11-01, removed 2026-11-02) ───────────────
# `--device ds4` ran DwarfStar (third_party/ds4: antirez's CUDA kernels behind a HIP shim plus
# hipBLAS), and its ds4-server was the engine's only 1BP reader. It is out of the shipped builds
# and out of the tree: serve lists it as a removed device, and nothing here builds it any more.
# A stale -DONEBIT_DS4=ON is refused by name instead of being silently ignored.
if(ONEBIT_DS4)
    message(FATAL_ERROR "ONEBIT_DS4 was removed with DwarfStar (owner 2026-11-01): drop the option and use --device hrx")
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
add_library(onebit_npu STATIC npu/file_map.cpp npu/flm_head_import.cpp npu/full_elf.cpp npu/model.cpp npu/vision_pack.cpp npu/vision_tower.cpp npu/vision_image.cpp
                            npu/moe.cpp npu/pack.cpp npu/pack_import.cpp npu/private_route.cpp
                            npu/q4nx.cpp npu/q4nx_device.cpp npu/tokenizer.cpp)
target_include_directories(onebit_npu PUBLIC npu)
target_link_libraries(onebit_npu PUBLIC nlohmann_json::nlohmann_json PRIVATE PkgConfig::PCRE2)

# ── npu/forward's forward C++ compiles without the NPU runtime (engine #898) ───────────────
# CI configures without ONEBIT_NPU, so npu/forward/src/q4nx_forward.cpp - where every A1 defect so
# far has lived (engine #808, #874) - reached a green build without being compiled. This OBJECT
# library compiles it and links nothing: the runtime's symbols are only needed at link time, and
# the in-tree npu/rt.h needs no XRT. `onebit` depends on it, so the CI build job (which builds
# `onebit`) compiles it without a workflow change.
add_library(onebit_npu_forward_syntax OBJECT npu/forward/src/q4nx_forward.cpp)
target_compile_definitions(onebit_npu_forward_syntax PRIVATE ONEBIT_NPU_RUNTIME_OWN)
target_include_directories(onebit_npu_forward_syntax PRIVATE
    npu/forward/include npu/forward/engine-src npu)
target_link_libraries(onebit_npu_forward_syntax PRIVATE nlohmann_json::nlohmann_json)
# Same OpenMP flags as onebit_npu_forward, so `#pragma omp` clauses are parsed here too.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(onebit_npu_forward_syntax PRIVATE OpenMP::OpenMP_CXX)
endif()
add_dependencies(onebit onebit_npu_forward_syntax)

# What one MTP head step costs on the HOST (tools/mtp_head_bench.cpp).  The draft route's per-iteration cost is two
# base passes plus the drafter's logits calls (both on the array) and the head's own arithmetic (this host), so this
# is the only term measurable without the device -- and the term a maths change has to be stated on.  It also prints
# a hash of every output hidden, so "no number moves" is checked bit for bit instead of argued.
add_executable(mtp_head_bench tools/mtp_head_bench.cpp npu/mtp/mtp_head.cpp)
target_include_directories(mtp_head_bench PRIVATE npu)
if(OpenMP_CXX_FOUND)
    target_link_libraries(mtp_head_bench PRIVATE OpenMP::OpenMP_CXX)
endif()

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
add_library(onebit_laya STATIC laya/scorer.cpp laya/safetensors.cpp laya/route.cpp laya/policy.cpp laya/encoder.cpp
            laya/models.cpp laya/eval.cpp)
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
add_executable(laya_models_test tests/laya_models_test.cpp)
target_link_libraries(laya_models_test PRIVATE onebit_laya)
add_test(NAME laya_models COMMAND laya_models_test ${CMAKE_SOURCE_DIR}/config/laya-models-game.json)
# The decision record itself (app/route_log.h): the classified state a decision keeps, so a
# /v1/route/decisions log can become `--eval` cases and fine-tune data (engine#1313, docs/laya-models.md
# "Measuring and fine-tuning the router").  Header only: no checkpoint, no device.
add_executable(route_log_test tests/route_log_test.cpp)
target_include_directories(route_log_test PRIVATE app)
target_link_libraries(route_log_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME route_log COMMAND route_log_test)
# `tools/copyright.py --check` must see a file that has not been staged yet (the check used `git
# ls-files`, so a brand-new file was invisible to it -- PR #1406's test script passed --check locally
# and failed CI's copyright step on the same head).  Runs the real tool in a scratch git repo.
# The HRX build's scope decision (engine#1499, .github/workflows/hrx-build.yml): a PR whose only
# CMakeLists.txt change is a ctest registration skips the compile but still configures.  Each self-test
# case is an artefact someone would actually push, and the conservative direction -- "full" -- is the
# one every uncertain case must land on, because the build being skipped is the one that catches a
# broken HRX pin.  Stdlib only: no git needed for the decision itself.
add_python_test(hrx_build_scope ${CMAKE_SOURCE_DIR}/tools/ci/hrx_build_scope.py --self-test)
add_python_test(copyright_check_untracked ${CMAKE_SOURCE_DIR}/tests/copyright_check_untracked_test.py)
# engine#1517: the prefill flash-attention ratio is measured in adjacent pairs and only quoted when the pairs agree, so a
# loaded box yields VOID instead of a number.  The self-test is the estimator's own gate and needs no GPU: it accepts
# agreement, refuses disagreement, and proves a dropped run cannot shift an arm.
add_python_test(fa_ratio_estimator ${CMAKE_SOURCE_DIR}/tools/fa_ratio.py --self-test)
add_dependencies(onebit route_log_test)
# CI builds `--target onebit` plus a fixed list in .github/workflows/ci.yml; building this check with onebit
# keeps the laya_models ctest from reporting "Not Run" without a workflow change.
add_dependencies(onebit laya_models_test)
# The routing eval's arithmetic (laya/eval.cpp): accuracy, recall, precision, confusion, ECE, log loss.
add_executable(laya_eval_test tests/laya_eval_test.cpp)
target_link_libraries(laya_eval_test PRIVATE onebit_laya nlohmann_json::nlohmann_json)
add_test(NAME laya_eval COMMAND laya_eval_test)
add_dependencies(onebit laya_eval_test)
# engine#1313 item 3: the head-only fine-tune over --dump-encoder dumps, on a synthetic checkpoint and dump (no model,
# no device). It needs torch + numpy, which the engine build does not, so a box without them skips (77).
add_python_test(laya_finetune ${CMAKE_SOURCE_DIR}/tests/laya_finetune_test.py)
if(Python3_Interpreter_FOUND)
    set_tests_properties(laya_finetune PROPERTIES SKIP_RETURN_CODE 77)
endif()
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
    # --laya-models: model auto picked per request among the table's rows (docs/laya-models.md)
    add_test(NAME laya_models_e2e COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_models_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_LAYA_ROUTE_MODEL})
    # `1bit route --laya-models --eval / --dump-encoder`: the table's routing measured, the head's input written
    add_test(NAME laya_eval_e2e COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_eval_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_LAYA_ROUTE_MODEL})
    # `1bit route --serve`, the resident form the fork calls per request (docs/laya-first.md §8).
    add_test(NAME laya_route_serve COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_route_serve.sh $<TARGET_FILE:onebit> ${ONEBIT_LAYA_ROUTE_MODEL})
    # One checkpoint resolution for serve --laya and 1bit route (app/laya_checkpoint.h).
    add_test(NAME laya_checkpoint COMMAND ${CMAKE_SOURCE_DIR}/tests/laya_checkpoint.sh $<TARGET_FILE:onebit> ${ONEBIT_LAYA_ROUTE_MODEL})
    # the request classifier on the 200 labelled requests (RFC #186): 95.5% measured, floor 90%
    add_test(NAME laya_classify COMMAND laya_classify_eval ${ONEBIT_LAYA_ROUTE_MODEL}/typed-decisions
             ${CMAKE_SOURCE_DIR}/tests/laya_route_cases.json --min-accuracy 90)
    # `--pm-router laya` end to end without a GPU: the same classifier decides the class of two
    # labelled requests and the engine's table decides whether the PM delegates (docs/pm.md).
    add_test(NAME pm_laya_route COMMAND ${CMAKE_SOURCE_DIR}/tests/pm_laya_route.sh $<TARGET_FILE:onebit> ${ONEBIT_LAYA_ROUTE_MODEL})
endif()

# The registry against the reviewed not-an-alias list (docs/arch-gaps.md) and the committed pins:
# no checkouts or network. (The interpreter is found once, at :287: engine#1864.)
if(Python3_Interpreter_FOUND)
    add_test(NAME registry_gaps COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/registry_build.py --check-gaps)
    # pin the guard's firing behavior: the committed registry always passes --check-gaps, so
    # this drives gap_violations against synthetic registries to prove it still fires.
    add_test(NAME registry_gaps_fire COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/registry_gaps_test.py)
    # a pin bump that did not regenerate the registry (only the gitlinks: no checkouts)
    add_test(NAME registry_pins COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/registry_build.py --check-pins)
    # engine#1745 review: pin() reads the INDEX (a staged bump) and refuses an UNMERGED index instead of returning the
    # merge base. Hermetic: a scratch git repo and a copy of the tool, git only, ~0.1 s.
    add_test(NAME registry_pin_index COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/registry_pin_index_test.py)
    # the same check where a lane actually runs it: with the bump staged and HEAD still behind, where a HEAD-only
    # read of the gitlink passed a registry recording the pin being replaced (engine#1738's post-commit failure)
    add_test(NAME registry_pins_staged COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/registry_pin_staged_test.py)
    # the 1.9 measurement tool's own paths, on synthetic packs: the "trying next" line it shipped broken (its
    # only real pair verified on the first candidate, so the line never ran) and the residual tolerance it
    # accepts and refuses at. Fixtures are ~200 KB each and built from the rule, so they can be wrong on purpose.
    add_test(NAME fconst19_measure COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/fconst19_measure_test.py)
    # The 1.9 tail-head tool (engine#1693/#1740): its geometry, its decoder against the vendor golden tile, and the
    # three refusals a real pack needed -- a block-32 head refused by name, an untied head given no false oracle, the
    # trailing zeros counted. It was never registered, so nothing ran it after #1710.
    add_test(NAME q4nx_tail_head COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/q4nx_tail_head_test.py)
    # Without onnx its two refusal arms cannot run: it exits 77, so ctest shows SKIPPED rather than a pass (#1747 review).
    set_tests_properties(q4nx_tail_head PROPERTIES SKIP_RETURN_CODE 77)
    # A member declared in npu/forward/include/q4nx_forward.h and never defined keeps `main` green while a full
    # `cmake --build` fails at link time (engine#1573: the 3-argument step_batch_embeds).  CI cannot catch it --
    # .github/workflows/ci.yml builds `--target onebit` only, because a full build wants the private NPU runtime --
    # and `-fsyntax-only` cannot see a link-time fact.  This pairs the two lists by (name, arity): device-free, no
    # runtime, no build, milliseconds.
    add_test(NAME npu_decl_check
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ci/npu_decl_check.py --check --root ${CMAKE_SOURCE_DIR})
    add_test(NAME npu_decl_check_self
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ci/npu_decl_check.py --self-test)
    # engine#2057: no fopen(..., "w"/"a") in the tree without a stated reason -- CodeQL's world-writable class, kept fixed
    # by a check rather than by the next scan. Source text only: no build, no device.
    add_test(NAME file_create_check
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ci/file_create_check.py --check --root ${CMAKE_SOURCE_DIR})
    add_test(NAME file_create_check_self
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ci/file_create_check.py --self-test)
    # the new-model watcher's alert contract (offline: synthetic batches against the
    # committed registry, plus FastFlowLM's org listing and releases from the recorded
    # payloads under tests/fixtures/fflm/ — engine #588), so a regression cannot leave the
    # daily alert permanently green.
    add_test(NAME census_watch COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/census_watch_test.py)
    # an installed package must find its libraries beside itself: the 2026-11-08 incident's rule, from
    # the installed side that no release gate can reach (engine #1411). Skips without patchelf/readelf.
    add_test(NAME installed_rpath COMMAND bash ${CMAKE_SOURCE_DIR}/tests/installed_rpath_test.sh)
    # every scheduled bump workflow must report its own failure (engine #665): bump-hrx froze its
    # pin while going red every day unnoticed. The step is driven against a stubbed gh, with and
    # without an open issue.
    add_test(NAME bump_workflow_report COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/bump_workflow_report_test.py)
    # every workflow's `run:` must parse, and the pin bump step must open one PR per moved repo while a
    # repo that went upstream stays a human decision: the step shipped with an unterminated quote that
    # nothing executed until a pin moved, i.e. until its first scheduled run (engine #1148)
    add_test(NAME workflow_shell COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/workflow_shell_test.py)
    # engine#1070: every configured test that starts a server holds RESOURCE_LOCK route_ports, so
    # the list at the end of this file cannot drift away from the tests it protects. Reads the
    # configured test list (ctest -N --show-only=json-v1); no device, no model, no build products.
add_test(NAME route_port_lock COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/route_port_lock_test.py ${CMAKE_CTEST_COMMAND} ${CMAKE_BINARY_DIR})

    # a Q4NX directory's chunk layout (engine #587): synthetic containers, so the check a
    # published FastFlowLM model goes through is proved to fire without the model at hand.
    add_test(NAME q4nx_dir_check COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/check_q4nx_dir.py --self-test)
    # engine#1443: "carries HIP code objects" is an ELF e_machine, not a bundle-header string
    add_test(NAME code_objects COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/code_objects_test.py)
    add_test(NAME trit_embed_pack COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/trit_embed_pack_test.py)
    # the FFN row/block sparsity measurements (docs/npu.md, "FFN row sparsity"): a synthetic GGUF and
    # a synthetic capture, so the byte budget and the capture arithmetic are gated without a model
    add_test(NAME ffn_sparsity COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ffn_sparsity.py --self-test)
    # the HRX hardware gate's judges (docs/hw-gate.md) against a fake server and synthetic logs;
    # the gate itself needs the GPU and runs on Strix Halo (.github/workflows/hw-gate.yml)
    add_test(NAME hw_gate_judges COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/hw_gate_test.py)
    # engine#1781: the comparator behind NPU_BATCH_CHECK, host-only (no model, no device, no runtime):
    # the batched verify pass is compared row for row against the sequential one, bitwise.
    add_test(NAME q4nx_pb16i COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/q4nx_pb16i_test.py)
    # The padded-grid read (engine#1097/#1110): tools/q4nx_diff.py decode reads a container whose weight sits on a
    # WIDER tile grid than the model's dimensions (the FastFlowLM repack pads intermediate_size to 512 columns, IM
    # 8960 -> 9216) instead of reporting it as an unreadable reshape.  Pure python + numpy, no model, no device.
    add_test(NAME q4nx_diff_grid COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/q4nx_diff_grid_test.py)
    # The pooled per-depth live-vs-offline verdict (tools/mtp_depth_pool.py): the brief's "within 2 pp at every
    # depth" is judged on 8 prompts x 64 tokens POOLED, which is two or three npu-hold tickets of <= 3 min, and
    # pooling those logs by eye produced two wrong numbers in this lane.  Fixture tickets only -- no model, no
    # device; the recorded 8x64 window is the tool's known-answer check and is run by hand with --selftest.
    add_test(NAME mtp_depth_pool COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/mtp_depth_pool_test.py)

    # The speculative-decoding harness for AMD's draft models (docs/amd-drafters.md): the acceptance measurement is against
    # a live llama-server, which CI cannot start, so the test covers the part that is wrong silently -- the accepted-prefix
    # rule, the two units printed side by side, and the refusal when no server answers.
    add_test(NAME spec_decode_acceptance COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tests/spec_decode_acceptance_test.py)

    # Declared-vs-defined guard for the forward classes (engine#1575): CI links only --target onebit, so a declaration with no
    # definition is invisible until a target that odr-uses it is linked -- and the NPU targets cannot be linked in CI at all.
    # Arity-aware on purpose: the defect that motivated it was a three-argument overload declared while only the four-argument
    # one had a body (#1573, fixed by #1574).
    add_test(NAME declared_vs_defined COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tests/declared_vs_defined_test.py)
    # The AXB35-02 xHCI keepawake check (docs/axb35-usb-ports.md, engine#823): board-specific, so the test builds fake sysfs
    # trees and asserts each verdict -- including "NOT THIS BOARD", which is what CI will get from the real /sys run and must
    # never be reported as a pass for the board.
    add_test(NAME xhci_keepawake_check COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tests/xhci_keepawake_check_test.py)

    # The OGA probe's ticket guard (engine#1371): a fake token file naming an ancestor's pid is the positive case, and the two
    # argv-based versions this replaced could never match a bash-script helper at all.
    add_test(NAME oga_pack_probe COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tests/oga_pack_probe_test.py)
    # docs/npu-leaderboard.md is generated by tools/npu_leaderboard.py from bench/npu-leaderboard/*.json and says
    # "edit the records, not this file", but nothing regenerated it: a record could change and the page would keep
    # the old table, still reading as measured.  This regenerates into a temp file and fails on any difference, and
    # it self-checks the comparison with a one-value change so a comparator that always passed cannot look the same.
    # Pure python, no model, no device, no network.
    add_test(NAME npu_leaderboard COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/npu_leaderboard_test.py)
    # An accuracy figure is the easiest number here to produce wrongly and the hardest to notice (tools/gsm8k_eval.py):
    # a scorer that reads the wrong number out of a completion, or that counts a completion cut off by the token
    # budget as a clean miss, still prints a plausible percentage.  This drives the extractor, the gold normaliser and
    # the whole run loop against canned completions, and re-derives the summary from the written transcript.  Stdlib
    # only, with no route, no socket, no dataset fetch and no device.
    add_test(NAME gsm8k_eval COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/gsm8k_eval_test.py)
    # engine#1808's integer host arm is pinned inside tests/npu_pack_test.cpp (ctest npu_pack): that binary is
    # already built by ci.yml, and a new ctest target would need a `.github/workflows/ci.yml` edit, which this
    # lane's token cannot push (the OAuth app has no `workflow` scope) -- the ci_targets check caught exactly
    # that, with both failures ("ci_targets" and "npu_host_gemm_i8 Not Run").
    # The repack's tile-order claim (engine#1110): `npu_plan.py` writes container.tile_order=engine-1 for every
    # plan, so the load-time guard (#1130) is green on a container whose TILES are another order.  This pins the
    # mapping from q4nx_diff's verdict to (stamp value, fatal), stdlib only: no model, no numpy, no device.
    add_test(NAME repack_tile_order COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/repack_tile_order_test.py)
    # The M-split ceiling (engine#754): tools/split_ceiling.py turns two measured rates into the optimum NPU
    # share, the per-GEMM ceiling and the end-to-end factor, and this pins it to the three iGPU states
    # docs/hetero-prefill.md records (40.7 TFLOP/s, the engine#782 regression, post-#782) so the arithmetic
    # cannot drift from the page.  Stdlib only: no model, no device.
    add_test(NAME split_ceiling COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/split_ceiling_test.py)
    # The same gate end to end (engine#1110 review): a synthetic one-tensor GGUF and a synthetic container whose
    # only tensor is named for another layer, so nothing is comparable -- the pair that used to read as a match --
    # plus the same bytes under the mapped name, which must be compared and refused.  numpy + gguf-py, no model.
    add_test(NAME tile_check_no_shared_tensor
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/tile_check_no_shared_tensor_test.py)
    # The offline half (engine#1110): a container whose q4_1 TILE ORDER is foreign has no metadata signature at
    # all -- same chunk kind, same bytes, same names -- so the fixture encodes one known weight in the engine's
    # order and in the foreign one and requires check_q4nx_dir.py --against to accept the first and refuse the
    # second by name.  numpy + gguf-py, no model, no device.
    add_test(NAME check_q4nx_tile_order
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/check_q4nx_tile_order_test.py)
    # The other half of that gate: the plan's container.tile_order is a measurement read from the repack's
    # stamp, not a constant -- a plan written for a directory the repack refused would hand the loader the
    # one value it accepts, undoing that refusal.  Stdlib only: no model, no numpy, no device.
    add_test(NAME npu_plan_tile_order COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/npu_plan_tile_order_test.py)
    # tools/onebp_import.py (a legacy .1bp package -> this engine's container): the v1 package's index
    # claims Q4NX for every entry while its norms are raw f32, so both the import and --verify have to
    # classify by rank and size -- verify reading the index quant reported 141 of 337 "differing" tensors
    # on a container it had just written.  Synthetic: no gguf, no model.
    add_test(NAME onebp_import
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/onebp_import_test.py)
    # the t2 chunk kind (docs/npu.md "The t2 chunk kind"): synthetic only, because the real-model
    # leg needs 14 GB of Bonsai-2 images. Run tests/q4nx_t2_test.py without --no-model for the gate.
    add_test(NAME q4nx_t2 COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/q4nx_t2_test.py --no-model)
    # the 2-bit ternary stream the XDNA decode kernel reads (tools/q4nx_t2s.py): the expansion it derives
    # must reproduce the pb16i image's own int4 code bytes, block for block. Synthetic weights only -- the
    # real-image leg is the same check on 14 GB of Bonsai-2 images and is run by hand.
    add_test(NAME q4nx_t2s COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/q4nx_t2s_test.py)
    # the acceptance gate for a t2 consumer (docs/npu.md, "Gating a t2 target"): the pinned
    # accumulation order, bit for bit -- it accepts the oracle and refuses four near-misses
    # (float64 accumulation, a reversed group order, a one-bit change, a short dump).  numpy only;
    # the real-tensor leg needs --tensor/--dump and runs by hand.  engine #804.
    add_test(NAME npu_t2_target_gate
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/npu_t2_target_gate.py --quiet)
    # the safetensors -> Q4NX converter and the d3 readout head it packs (engine #653): every
    # fixture is synthesised, so this needs numpy and nothing else — no gguf-py, no model, no
    # device, no download.
    add_test(NAME safetensors_to_q4nx
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/safetensors_to_q4nx_test.py)
endif()

# The d3 readout head loader (npu/decision.h): two files and the arithmetic on a hidden vector, so
# it runs with no model, no backbone and no device.
add_executable(decision_test tests/decision_test.cpp)
target_include_directories(decision_test PRIVATE ${CMAKE_SOURCE_DIR}/npu)
target_link_libraries(decision_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME decision_head COMMAND decision_test)
# .github/workflows/ci.yml builds a named list of targets and ctest runs whatever binaries exist, so a
# test whose target is not in that list reports "Not Run" and never executes (the #635 trap). Building
# decision_test as a dependency of a listed target keeps its row live without a workflow edit, which
# this lane's token cannot make.
add_dependencies(npu_sampling_test decision_test)

# The d3 question/answer contract, byte-compared against fixtures generated by the PACK'S own
# d3_format.py (tests/golden/systemone/prompts.json): no device, no model, no network.
add_executable(systemone_test tests/systemone_test.cpp)
target_include_directories(systemone_test PRIVATE ${CMAKE_SOURCE_DIR}/npu)
target_link_libraries(systemone_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME systemone_contract
         COMMAND systemone_test ${CMAKE_SOURCE_DIR}/tests/golden/systemone/prompts.json)
# Same reason as decision_test: ci.yml builds a named list of targets, and a test whose target is not
# built reports "Not Run" and silently never executes (#635). This lane's token cannot edit the
# workflow, so the binary is built from a listed target until someone with workflow scope adds it.
add_dependencies(npu_sampling_test systemone_test)

# The HTTP half of the same contract (app/npu_systemone.h): the request envelope, the chat-template
# wrapper around the user turn and the reply shape. Still no device, no model, no network.
add_executable(systemone_route_test tests/systemone_route_test.cpp)
target_include_directories(systemone_route_test PRIVATE ${CMAKE_SOURCE_DIR}/app ${CMAKE_SOURCE_DIR}/npu)
target_link_libraries(systemone_route_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME systemone_route
         COMMAND systemone_route_test ${CMAKE_SOURCE_DIR}/tests/golden/systemone/prompts.json)
# Same reason as decision_test and systemone_test: ci.yml builds a named list of targets, so a test
# whose target is not built reports "Not Run" and silently never executes (#635). This lane's token
# cannot edit the workflow.
add_dependencies(npu_sampling_test systemone_route_test)

# The Qwen3-lane chat prompt of `1bit forward-serve` (app/npu_chatml.h): `chat_template_kwargs.enable_thinking`
# and the bytes it produces, which is the half of issue #832 a test can pin without a model.
add_executable(npu_chatml_test tests/npu_chatml_test.cpp)
target_include_directories(npu_chatml_test PRIVATE ${CMAKE_SOURCE_DIR}/app ${CMAKE_SOURCE_DIR}/npu)
target_link_libraries(npu_chatml_test PRIVATE onebit_npu nlohmann_json::nlohmann_json)
# The #963 rows read the two fixtures: a tokenizer with ChatML anchors (served) and one without (refused).
add_test(NAME npu_chatml COMMAND npu_chatml_test ${CMAKE_SOURCE_DIR}/tests/data/tokenizer_chatml.json
                                          ${CMAKE_SOURCE_DIR}/tests/data/tokenizer_linear.json)
# Same reason as the other device-free tests: ci.yml builds a named list of targets, so a test whose
# target is not built reports "Not Run" (#635). This lane's token cannot edit the workflow.
add_dependencies(npu_sampling_test npu_chatml_test)

# The container's own chat template rendered (app/npu_template.h, engine #963 option 2): the golden rows are
# Jinja2 3.1.6's own renders of the fixtures' templates, so the assertion is a comparison against an oracle
# that is not this code. Device-free.  Without the renderer (ONEBIT_JINJA=OFF) the two fixture rows cannot
# pass, so the test says so in its first line rather than skipping quietly.
add_executable(npu_template_test tests/npu_template_test.cpp)
# app/template_pin.h includes the pins compiled in from config/template-pins.json (the same read the
# load line does), which CMake writes into the build's generated/ at configure time.
target_include_directories(npu_template_test PRIVATE ${CMAKE_SOURCE_DIR}/app ${CMAKE_SOURCE_DIR}/npu
                            ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(npu_template_test PRIVATE onebit_npu nlohmann_json::nlohmann_json)
if(ONEBIT_JINJA)
    target_link_libraries(npu_template_test PRIVATE onebit_jinja)
    target_compile_definitions(npu_template_test PRIVATE ONEBIT_JINJA=1)
    add_test(NAME npu_template
             COMMAND npu_template_test ${CMAKE_SOURCE_DIR}/tests/data/templates/qwen3_flm.json
                                       ${CMAKE_SOURCE_DIR}/tests/data/templates/zr1.json
                                       --tokenizer ${CMAKE_SOURCE_DIR}/tests/data/tokenizer_chatml.json)
else()
    add_test(NAME npu_template COMMAND npu_template_test --no-renderer)
    set_tests_properties(npu_template PROPERTIES SKIP_RETURN_CODE 77)
endif()

# `1bit unified`'s wait for the single NPU (app/npu_queue.h, engine#1024): bounded, and a queued request says so
# instead of leaving a stream silent past the proxy's no-data timer. Device-free.
add_executable(npu_queue_test tests/npu_queue_test.cpp)
target_include_directories(npu_queue_test PRIVATE ${CMAKE_SOURCE_DIR}/app)
add_test(NAME npu_queue COMMAND npu_queue_test)
# Same reason as the other device-free tests: ci.yml builds a named list of targets, so a test whose
# target is not built reports "Not Run" (#635). This lane's token cannot edit the workflow.
add_dependencies(npu_sampling_test npu_queue_test)

# 1-bit sentence embeddings (engine#1311): the container the importer writes and the encoder that reads it.
# Device-free on a tiny fixture -- the real model's gate (min cosine 0.9997 vs the reference on 1,000
# sentences) needs the 17.9 MB container the importer writes (the source safetensors is 13.9 MB) and stays
# a measured row in the PR.
add_executable(embed_model_test tests/embed_model_test.cpp app/embed_model.cpp)
# engine#1956: the Q4NX container cache's sweep, driven on a synthetic cache (bound, LRU order, the entry just written, the
# age window, and the disabled case). Listed explicitly like the other app/ units so the test needs no `onebit`.
add_executable(q4nx_cache_test tests/q4nx_cache_test.cpp app/q4nx_cache.cpp)
target_include_directories(q4nx_cache_test PRIVATE app)
add_test(NAME q4nx_cache COMMAND q4nx_cache_test)
# engine#1956 review: an `add_test` whose target ci.yml never builds never runs in CI -- the documented route, no workflow edit.
add_dependencies(onebit q4nx_cache_test)
target_include_directories(embed_model_test PRIVATE ${CMAKE_SOURCE_DIR}/app)
target_link_libraries(embed_model_test PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(embed_model_test PRIVATE SRC_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME embed_model COMMAND embed_model_test)
# The WordPiece pipeline an embedding model's ids depend on (cleaning, lowercase, accents, punctuation, CJK,
# "##" continuations, [CLS]/[SEP], truncation). Device-free.
add_executable(embed_wordpiece_test tests/embed_wordpiece_test.cpp)
target_include_directories(embed_wordpiece_test PRIVATE ${CMAKE_SOURCE_DIR}/app)
add_test(NAME embed_wordpiece COMMAND embed_wordpiece_test)
add_dependencies(npu_sampling_test embed_model_test)
add_dependencies(npu_sampling_test embed_wordpiece_test)

# `forward-serve`'s /v1/completions prompt as text or as the caller's exact token ids (app/npu_prompt_ids.h):
# the acceptance rules and the refusals that keep an out-of-vocabulary id from reaching the embedding
# lookup.  Device-free, model-free.
add_executable(npu_prompt_ids_test tests/npu_prompt_ids_test.cpp)
target_include_directories(npu_prompt_ids_test PRIVATE ${CMAKE_SOURCE_DIR}/app ${CMAKE_SOURCE_DIR}/npu)
target_link_libraries(npu_prompt_ids_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME npu_prompt_ids COMMAND npu_prompt_ids_test)
# Same reason as the other device-free tests: ci.yml builds a named list of targets, so a test whose
# target is not built reports "Not Run" (#635). This lane's token cannot edit the workflow.
add_dependencies(npu_sampling_test npu_prompt_ids_test)

# The ids that end a generation are a list, not an integer (app/npu_stop_tokens.h, engine#1044): a
# Llama-3.2 container declares eos_token_id [128001, 128008, 128009] and forward-serve read that key as
# one integer, so 128009 was served as text with finish_reason "length".
add_executable(npu_stop_tokens_test tests/npu_stop_tokens_test.cpp)
target_include_directories(npu_stop_tokens_test PRIVATE ${CMAKE_SOURCE_DIR}/app ${CMAKE_SOURCE_DIR}/npu)
target_link_libraries(npu_stop_tokens_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME npu_stop_tokens COMMAND npu_stop_tokens_test)
add_dependencies(npu_sampling_test npu_stop_tokens_test)

# The hand-off between the thread that generates a streamed reply and the one that writes it (app/stream_bridge.h,
# engine#1785: the NPU route sent every frame after the generation ended). Two threads and a clock; no device.
add_executable(stream_bridge_test tests/stream_bridge_test.cpp)
target_include_directories(stream_bridge_test PRIVATE ${CMAKE_SOURCE_DIR}/app)
target_link_libraries(stream_bridge_test PRIVATE Threads::Threads)
add_test(NAME stream_bridge COMMAND stream_bridge_test)
add_dependencies(npu_sampling_test stream_bridge_test)

# The plan.json loader (npu/plan/plan.h, engine #799 P1): schema, refusals and the decision fields that
# replaced the filename probes.  No model, no container, no device.
add_executable(npu_elf_args_test tests/npu_elf_args_test.c npu/forward/src/q4nx_pack.c npu/forward/src/q4nx_dequant.c)
target_include_directories(npu_elf_args_test PRIVATE npu/forward/include)
target_link_libraries(npu_elf_args_test PRIVATE m)
add_test(NAME npu_elf_args COMMAND npu_elf_args_test)
# Every op a plan may name must ship a host reference with known answers, and the test walks the table
# (engine#799): an op without one is named by op_coverage() and asserted here, so the gap is a decision in
# the test rather than an omission.  Host-only, no model, no device.
add_executable(npu_plan_op_reference_test tests/npu_plan_op_reference.cpp npu/plan/op_registry.cpp)
target_include_directories(npu_plan_op_reference_test PRIVATE ${CMAKE_SOURCE_DIR}/npu)
add_test(NAME npu_plan_op_reference COMMAND npu_plan_op_reference_test)
add_executable(npu_plan_test tests/npu_plan_test.cpp)
target_include_directories(npu_plan_test PRIVATE ${CMAKE_SOURCE_DIR}/npu)
target_link_libraries(npu_plan_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME npu_plan COMMAND npu_plan_test)
# Same reason as the other device-free tests: ci.yml builds a named list of targets, so a test whose
# target is not built reports "Not Run" (#635).  This lane's token cannot edit the workflow.
add_dependencies(npu_sampling_test npu_plan_test)
# The plan-vs-target rule (npu/target/target.h, check_kv_caps): a plan that asks for a cache the target
# cannot store is a refusal by name, never a silent f16 store.  Device-free: the refusal happens before any
# kernel is touched, and it is the same answer for a target that does not exist yet (engine#806 P7a).
add_executable(npu_target_caps_test tests/npu_target_caps_test.cpp)
target_include_directories(npu_target_caps_test PRIVATE ${CMAKE_SOURCE_DIR}/npu)
target_link_libraries(npu_target_caps_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME npu_target_caps COMMAND npu_target_caps_test)
add_dependencies(npu_sampling_test npu_target_caps_test)

# The generator behind that loader: a synthetic model dir, so the fused/split decision and --check are
# tested without a model or a kernel build.
add_test(NAME npu_plan_generate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/npu_plan_generate_test.py
                 ${CMAKE_SOURCE_DIR}/tools/npu_plan.py)
# engine #709: the installer's firmware boot-entry rules, exercised with a fake efibootmgr and a fake
# /sys/class/block, so CI covers selection/rollback without NVRAM. The OVMF install gate
# (os/mini/vmtest.sh) is the end-to-end row and is run separately.
add_test(NAME os_mini_efi_boot COMMAND ${CMAKE_SOURCE_DIR}/tests/os_mini_efi_boot_test.sh)
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
# engine#1788: the vision component reader (vision_weight.q4nx). Crafted containers for every refusal; the published
# pack's own numbers when ONEBIT_MINICPM_PACK names a MiniCPM-V 4.7 pack directory.
add_executable(npu_vision_pack_test tests/npu_vision_pack_test.cpp)
target_link_libraries(npu_vision_pack_test PRIVATE onebit_npu)
add_test(NAME npu_vision_pack COMMAND npu_vision_pack_test)
add_dependencies(onebit npu_vision_pack_test)
# The host float32 tower against goldens from the transformers implementation (see the test's header).
add_executable(npu_vision_tower_test tests/npu_vision_tower_test.cpp)
target_link_libraries(npu_vision_tower_test PRIVATE onebit_npu)
add_test(NAME npu_vision_tower COMMAND npu_vision_tower_test)
add_dependencies(onebit npu_vision_tower_test)
# The image plan (sizes, slice grid, text rows) against the reference image processor's answers for 3,481 sizes.
add_executable(npu_vision_plan_test tests/npu_vision_plan_test.cpp)
target_include_directories(npu_vision_plan_test PRIVATE npu)
add_test(NAME npu_vision_plan COMMAND npu_vision_plan_test)
add_dependencies(onebit npu_vision_plan_test)
# The resize (bit for bit against torch's uint8 antialiased bicubic) and the pieces handed to the tower.
add_executable(npu_vision_image_test tests/npu_vision_image_test.cpp)
target_link_libraries(npu_vision_image_test PRIVATE onebit_npu)
add_test(NAME npu_vision_image COMMAND npu_vision_image_test)
add_dependencies(onebit npu_vision_image_test)
# The text side of an image: its tokens and the canvas positions, against the reference model code's own answers.
add_executable(npu_vision_prompt_test tests/npu_vision_prompt_test.cpp)
target_include_directories(npu_vision_prompt_test PRIVATE npu)
add_test(NAME npu_vision_prompt COMMAND npu_vision_prompt_test)
add_dependencies(onebit npu_vision_prompt_test)
add_executable(npu_q4nx_test tests/npu_q4nx_test.cpp)
target_include_directories(npu_q4nx_test PRIVATE app)
target_link_libraries(npu_q4nx_test PRIVATE onebit_npu)
add_test(NAME npu_q4nx COMMAND npu_q4nx_test ${CMAKE_SOURCE_DIR}/tests/golden/q4nx)
add_executable(npu_pack_import_test tests/npu_pack_import_test.cpp)
target_link_libraries(npu_pack_import_test PRIVATE onebit_npu)
add_test(NAME npu_pack_import COMMAND npu_pack_import_test ${CMAKE_SOURCE_DIR}/tests/golden)
# engine#1726: NPU_INFER_DUMP_DIR's residual write must SAY when it cannot write -- a missing or unwritable directory, or a
# full disk, used to leave a run that printed it was dumping with no files and no error. Pure stdio, no device, no NPU
# headers: the arm a version that only checks fopen would pass is the short write, which /dev/full produces for real.
add_executable(npu_residual_dump_test tests/npu_residual_dump_test.cpp)
target_include_directories(npu_residual_dump_test PRIVATE npu/forward/include)
add_test(NAME npu_residual_dump COMMAND npu_residual_dump_test)
# The OGA pack importer against a real pack (1bit npu-pack), when one is on the box
set(ONEBIT_OGA_PACK "" CACHE PATH "A Ryzen AI OGA NPU model pack directory")
if(ONEBIT_OGA_PACK)
    add_test(NAME npu_pack_import_oga
             COMMAND npu_pack_import_test ${CMAKE_SOURCE_DIR}/tests/golden ${ONEBIT_OGA_PACK})
endif()
# q4nx_pack.c is listed here rather than reached through onebit_npu: the integer host arm's checks
# (engine#1808) call q4nx_host_gemm_i8*() directly, and those live in that file.
add_executable(npu_pack_test tests/npu_pack_test.cpp npu/forward/src/q4nx_pack.c npu/forward/src/q4nx_dequant.c)
# A DFlash draft package as a model source (engine#1780): config + container -> a checked model whose tensors decode
# one at a time. The loader is compiled here and not into onebit_npu because it needs the q4_1 tile decoder, which
# lives with the forward (npu/forward/src/q4nx_dequant.c); it moves there with the draft's forward. Synthetic package,
# so no model, no device; a real package is an optional argument (-DONEBIT_DFLASH_DRAFT=<dir>).
add_executable(dflash_draft_test tests/dflash_draft_test.cpp npu/dflash_draft.cpp npu/forward/src/q4nx_dequant.c)
target_include_directories(dflash_draft_test PRIVATE npu npu/forward/include)
target_link_libraries(dflash_draft_test PRIVATE onebit_npu)
add_dependencies(onebit dflash_draft_test)
set(ONEBIT_DFLASH_DRAFT "" CACHE PATH "A DFlash draft package directory for dflash_draft_test's real-package arm")
add_test(NAME dflash_draft COMMAND dflash_draft_test ${ONEBIT_DFLASH_DRAFT})
# The draft's forward as a host reference (engine#1780): hand-computed cases and structure properties; the real
# package arm runs when ONEBIT_DFLASH_DRAFT names one.
add_executable(dflash_forward_test tests/dflash_forward_test.cpp npu/dflash_forward.cpp npu/dflash_draft.cpp npu/forward/src/q4nx_dequant.c)
target_include_directories(dflash_forward_test PRIVATE npu npu/forward/include)
target_link_libraries(dflash_forward_test PRIVATE onebit_npu)
add_dependencies(onebit dflash_forward_test)
add_test(NAME dflash_forward COMMAND dflash_forward_test ${ONEBIT_DFLASH_DRAFT})
add_executable(flm_head_import tools/flm_head_import.cpp)
target_link_libraries(flm_head_import PRIVATE onebit_npu)
add_executable(npu_flm_head_import_test tests/npu_flm_head_import_test.cpp)
target_link_libraries(npu_flm_head_import_test PRIVATE onebit_npu)
add_test(NAME npu_flm_head_import COMMAND npu_flm_head_import_test ${CMAKE_SOURCE_DIR}/tests/golden/flm-i8)
target_link_libraries(npu_pack_test PRIVATE onebit_npu)
# engine#1808: the integer host arm's checks live in this binary (see above), so it needs the header and the
# golden fixture that pins them.
target_include_directories(npu_pack_test PRIVATE npu/forward/include ${CMAKE_SOURCE_DIR}/tests)

# NPU_INFER_DUMP_OPS (npu/forward/include/ops-dump.h, engine#1503): the on/off rule, the mean|x|/absmax arithmetic and
# the line's text. Header-only, no device. `add_dependencies(onebit ...)` is what gets it built in CI, whose target
# list is explicit (tools/ci/ci_targets_check.py).
# npu/forward/include/file_create.h (engine#2057): the openers every dump and the head-image cache write through -- 0600 /
# at most 0644 whatever the umask, an existing 0666 file narrowed, a symlink refused with ELOOP. Plain C, no model, no
# device; the call sites only swapped fopen() for an opener, so without this arm a revert to fopen() passes every test.
if(NOT WIN32)
    add_executable(file_create_test tests/file_create_test.c)
    target_include_directories(file_create_test PRIVATE npu/forward/include)
    add_test(NAME file_create COMMAND file_create_test)
    add_dependencies(onebit file_create_test)
endif()
add_executable(ops_dump_test tests/ops_dump_test.cpp)
target_include_directories(ops_dump_test PRIVATE npu/forward/include)
add_test(NAME ops_dump COMMAND ops_dump_test)
add_dependencies(onebit ops_dump_test)
# When the dense-tail cache is on by default (npu/forward/include/ffn_cache_policy.h): the size rule and the two
# models it was measured on, one on each side of the old floor. Header-only, no device.
add_executable(ffn_cache_policy_test tests/ffn_cache_policy_test.cpp)
target_include_directories(ffn_cache_policy_test PRIVATE npu/forward/include)
add_test(NAME ffn_cache_policy COMMAND ffn_cache_policy_test)
add_dependencies(onebit ffn_cache_policy_test)
# LFM2's gated short convolution as host arithmetic (npu/forward/include/lfm2_shortconv.h, engine#1786): the batched
# form must equal the token-by-token form bit for bit. Header-only, no model, no device.
add_executable(lfm2_shortconv_test tests/lfm2_shortconv_test.cpp)
target_include_directories(lfm2_shortconv_test PRIVATE npu/forward/include)
add_test(NAME lfm2_shortconv COMMAND lfm2_shortconv_test)
add_dependencies(onebit lfm2_shortconv_test)

# The device weight layout contract (q4nx_weight_layout): pure index math, no device
# and no runtime.  CI builds an explicit target list, so the check compiles itself with
# the build's compiler at test time (tests/npu_blocked_layout_test.sh) rather than being
# one missing target away from "Not Run".
add_test(NAME npu_blocked_layout
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_blocked_layout_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}/npu/forward/include
                 ${CMAKE_SOURCE_DIR}/tests/npu_blocked_layout_test.cpp)
# The pb16i weight image (q4nx_weight_layout "pb16i"): the same kind of contract, plus the
# packer that builds the images (tools/q4nx_pb16i.py).  No device, no model: the real-GGUF
# leg is skipped when the model is not on the machine.
add_test(NAME npu_pb16i_layout
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_pb16i_layout_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}/npu/forward/include
                 ${CMAKE_SOURCE_DIR}/tests/npu_pb16i_layout_test.cpp)
# The t2s weight stream (q4nx_weight_layout "t2s"): the same integer-exact weights at 2.125 b/w,
# expanded on the core.  Its host-side contract is the file size, so the test pins the tile against
# the numbers tools/q4nx_t2s.py measured on a real image (50,135,040 -> 23,674,880 B for gate/up).
add_test(NAME npu_t2s_layout
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_t2s_layout_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}/npu/forward/include
                 ${CMAKE_SOURCE_DIR}/tests/npu_t2s_layout_test.cpp)
# The t2 chunk (npu/q4nx.h, "The t2 chunk kind"): the same kind of contract, but the thing under
# test is the shipped decoder, so it compiles npu/q4nx.cpp as well.  No device, no model.
add_test(NAME npu_t2_layout
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_t2_layout_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}
                 ${CMAKE_SOURCE_DIR}/tests/npu_t2_layout_test.cpp)
# The padded-grid contract (engine#1097): a container may store a weight on a WIDER tile grid than the
# model's own dimensions -- the FastFlowLM repack pads intermediate_size up to its 512-column grid, so
# an 8960-wide MLP arrives as 9216 and holds 1728 tiles where the model's dims imply 1680.  The
# decoder reads the container's grid and writes the model's [rows, cols], so a caller's buffer, its
# row stride and the kernel's B layout never move.  Self-compiling like the other layout contracts
# (tests/npu_padded_grid_test.sh) so ctest needs no build target and CI cannot report "Not Run".
add_test(NAME npu_padded_grid
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_padded_grid_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}
                 ${CMAKE_SOURCE_DIR}/tests/npu_padded_grid_test.cpp)
# The MoE container-convention contract (npu/forward/include/moe_container.h, engine#1035): the
# engine converter's `mlp.experts.*` names + [N][K] per-expert tile grid and HF-shaped config keys,
# against the FLM-style `*_exps_proj` names + [K][N] grid and llama.cpp-shaped keys. A wrong
# choice is silent (a missing tensor, or a permuted expert), so the test is a known-answer decode.
# Self-compiling like the layout contracts, with the shipped chunk decoder, so CI needs no target.
add_test(NAME npu_moe_container
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_moe_container_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}
                 ${CMAKE_SOURCE_DIR}/tests/npu_moe_container_test.cpp)
# The host side of a MoE container: __metadata__.moe, per-expert chunk pointers and the router.
# Crafted directories at run time, so CI needs no 18 GB model; with two arguments it checks the
# same code against a real container and a gate.json from tools/moe_expert_chunks.py gate.
add_executable(npu_moe_test tests/npu_moe_test.cpp)
target_link_libraries(npu_moe_test PRIVATE onebit_npu)
add_test(NAME npu_moe COMMAND npu_moe_test)
if(ONEBIT_NPU_MODEL AND ONEBIT_NPU_MOE_GATE)
    add_test(NAME npu_moe_model COMMAND npu_moe_test ${ONEBIT_NPU_MODEL} ${ONEBIT_NPU_MOE_GATE})
endif()
# Crafted model.q4nx files the loader and the packer must refuse (written at run time); runs in CI.
add_executable(npu_model_bounds_test tests/npu_model_bounds_test.cpp)
target_link_libraries(npu_model_bounds_test PRIVATE onebit_npu)
add_test(NAME npu_model_bounds COMMAND npu_model_bounds_test)
# CI builds an explicit target list (.github/workflows/ci.yml, "Keep this list in step with
# add_test()"), and extending that list needs a token with the `workflow` scope, which a lane does
# not carry. Binding npu_moe_test to a target the list already names gets it built either way (a
# target-level dependency needs that target to exist, hence this position). Delete this line and add
# npu_moe_test to the workflow list the next time the workflow is touched with the right scope.
add_dependencies(npu_model_bounds_test npu_moe_test)
# The forward's Q4NX loader is plain C (no NPU runtime), so its crafted-file checks run in CI too.
add_executable(npu_forward_loader_test tests/npu_forward_loader_test.c npu/forward/src/model.c)
target_include_directories(npu_forward_loader_test PRIVATE npu/forward/include)
target_link_libraries(npu_forward_loader_test PRIVATE m)
add_test(NAME npu_forward_loader COMMAND npu_forward_loader_test)

# rope_type "llama3" (npu/forward/include/q4nx_rope.h, engine#1048): the frequency band is fixed data.
# Old code built theta^(-i/half) for every dim pair, and 17 of Llama-3.2-1B's 32 differ from that.
add_executable(npu_rope_llama3_test tests/npu_rope_llama3_test.c)
target_include_directories(npu_rope_llama3_test PRIVATE npu/forward/include)
target_link_libraries(npu_rope_llama3_test PRIVATE m)
add_test(NAME npu_rope_llama3 COMMAND npu_rope_llama3_test)
add_dependencies(npu_sampling_test npu_rope_llama3_test)
# The activation rotation a folded container needs (npu/forward/include/hadamard.h): its SIMD path must be
# bit-identical to the shipped scalar loop on every shape a container carries, and the scalar loop must still
# be the Hadamard transform.  Header only, no device, no model: runs in CI.
add_executable(hadamard_rotate_test tests/hadamard_rotate_test.cpp)
target_include_directories(hadamard_rotate_test PRIVATE npu/forward/include)
add_test(NAME hadamard_rotate COMMAND hadamard_rotate_test)
# The arithmetic P7 (#806) part A rests on: (QH)(KH)^T = QK^T to f32 rounding, and what the rotation is
# worth against an f16 KV cache (attention-weight KLD) with and without outlier channels.  Header only, no
# device, no model: runs in CI.
add_executable(npu_kv_int8_test tests/npu_kv_int8_test.cpp)
target_include_directories(npu_kv_int8_test PRIVATE npu/forward/include)
add_test(NAME npu_kv_int8 COMMAND npu_kv_int8_test)
# The A-row quantiser every NPU GEMM runs on the host before its launch (npu/forward/include/quant_i8.h): its
# AVX2 path must be byte-identical to the shipped scalar loop, including the .5 boundaries, the NaN/inf inputs
# and the values past 2^31 that the (int) cast saturates.  Header only, no device, no model: runs in CI.
add_executable(quant_i8_row_test tests/quant_i8_row_test.cpp)
target_include_directories(quant_i8_row_test PRIVATE npu/forward/include)
add_test(NAME quant_i8_row COMMAND quant_i8_row_test)
# The forward's two attention kernels (npu/forward/include/attn_math.h): the AVX2+FMA body and the portable
# four-accumulator body must both be a float rounding of the same sum, checked against a double-precision
# direct sum over the head dims the forward serves plus the widths where a vector body's tail runs.  This is
# also the only thing that exercises the portable body on an AVX2 CI host, where production never takes it.
# Header only, no device, no model: runs in CI.
add_executable(attn_math_test tests/attn_math_test.cpp)
target_include_directories(attn_math_test PRIVATE npu/forward/include)
add_test(NAME attn_math COMMAND attn_math_test)
# engine#1990: the test also holds rows_context() (M rows in one thread team) to the per-row form bit for bit,
# and that comparison is only about threads when the test has them.
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(attn_math_test PRIVATE OpenMP::OpenMP_CXX)
endif()
# ci.yml's build step names its targets explicitly and needs the `workflow` scope to extend, so bind each new
# test binary to a target the list already builds (same reason as npu_moe_test above); drop this line when the
# workflow is next touched with that scope and add these to the list itself. `host_gemm_pad_test` is here for the
# same reason: without it ctest reports the row as "Not Run" and the job fails after every other test passes.
add_dependencies(npu_forward_loader_test hadamard_rotate_test quant_i8_row_test attn_math_test npu_kv_int8_test
                 host_gemm_pad_test npu_derive_designs_dump npu_plan_op_reference_test)
# The window11 gate's comparison policy (engine#834): no tolerance without a measurement behind it, the
# statistic named, and a control pair setting the tolerance from its own spread.  Shell + the fixtures it
# writes itself, so there is no target for the build list to miss.
add_test(NAME npu_window11_policy COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/npu-window11.sh --self-test)
# The package's file modes (engine#1352): a release artifact's modes are the packer's decision, so every staged file is
# 0755 (was executable) or 0644 (was not), no file or directory is group/other-writable, and the reporter that catches a
# staging line added below the normalisation still fires on a tree nobody normalised.  Fixture tree: no build, no tar.
add_test(NAME pack_modes COMMAND bash ${CMAKE_SOURCE_DIR}/tests/pack_modes_test.sh)
# patch-sfs must not destroy the artifact it was asked to replace (#1491: two pre-pack refusals did `rm -f "$out"` before the
# output had been reassigned to its .partial name, so a failed cut deleted the previous good SFS). Skips without root, since
# patch-sfs needs uid/gid and setuid bits - this is the only test here that does, hence the property.
add_test(NAME patch_sfs_atomic COMMAND ${CMAKE_SOURCE_DIR}/tests/patch_sfs_atomic_test.sh)
set_tests_properties(patch_sfs_atomic PROPERTIES SKIP_RETURN_CODE 77)
# build-static.sh must never print a quotable artifact hash for a tree that is not a commit (engine#1326, from
# engine#1323's mis-quoted hash). The toolchain is stubbed inside the test - it is the provenance line under test,
# not the compile - so this costs milliseconds and needs no rustup, no musl target and no network.
add_test(NAME build_static_provenance COMMAND ${CMAKE_SOURCE_DIR}/tests/build_static_provenance_test.sh)
# embed/usr/local/bin/1bit-console owns tty1 and its whole job is that there is no path from this screen to a bare
# prompt; nothing tested it. Same shape as the test above - the script is copied and its absolute paths rewritten into
# a temp tree, so every branch (installer, desktop, TUI-then-menu, no-TUI) is reachable without root or a build.
add_test(NAME console_chain COMMAND ${CMAKE_SOURCE_DIR}/tests/console_chain_test.sh)
# build_path_strings must see compiler output, not only ELFs: a shipped devicetable-offsets.s with 449 build-tree paths
# passed the row that exists to stop that (engine#1487). No root, no fixture beyond two text files.
add_test(NAME pack_check_generated_text COMMAND bash ${CMAKE_SOURCE_DIR}/tests/pack_check_generated_text_test.sh)
# The GEMM designs are derived on the container's K grid and the kernel's N grid (engine#845); a
# model already on both is unchanged.  Pure arithmetic + one assemble_B, so it runs without a model.
# The engine's derived design set for a set of dims, as `ROLE M K N elf` lines.  The instrument the
# parity test above needs: the plan generator must state the same kernels q4nx_derive_designs() does, and
# comparing the two is the only way that can be checked rather than assumed (engine#803 A2a).
add_executable(npu_derive_designs_dump tests/npu_derive_designs_dump.c npu/forward/src/q4nx_pack.c npu/forward/src/q4nx_dequant.c)
target_include_directories(npu_derive_designs_dump PRIVATE npu/forward/include)
target_link_libraries(npu_derive_designs_dump PRIVATE m)
# q4nx_dequant.c joins this target: the fused chunks->int8 path (engine #1416) dequantises one tile at a time and
# the test compares it against the two-step dequant+pack, so both live here.
# q4nx_pack_blocked.cpp is here because the head-tile identity arm (engine#1716) compares the blocked packer the forward
# actually calls against the reference; the test links both. It is C++, so CMake switches this target's linker to CXX.
# q4nx_head_image.c is the image cache's file format (engine#1716's cache slice, this branch): the identity arm above pins
# what the image must contain, this one pins that what is written and read back is byte-identical to it.
# q4nx_head_tile_cache.cpp is what a serve calls per head tile (load the image a previous serve wrote, or pack and write
# it): the arms here are the cache's behaviour -- a second pass does not pack, a changed key does, a damaged or stale image
# is replaced rather than used -- with a counter on the gather, so "rebuilt on every serve" cannot come back unnoticed.
add_executable(npu_design_grid_test tests/npu_design_grid_test.c npu/forward/src/q4nx_pack.c npu/forward/src/q4nx_dequant.c
                                  npu/forward/src/q4nx_pack_blocked.cpp npu/forward/src/q4nx_head_image.c
                                  npu/forward/src/q4nx_head_tile_cache.cpp)
target_include_directories(npu_design_grid_test PRIVATE npu/forward/include)
target_link_libraries(npu_design_grid_test PRIVATE m)
# engine#1779 fault 4 ("Gemma 4 tool-call parsing"): the answer parser is header-only (app/npu_answer_parser.h), so
# this is a PURE unit test - no server, no model, no device - of the contract that header states: a template's own
# <tool_call>{"name":...,"arguments":{...}}</tool_call> becomes a structured call with the arguments re-dumped as JSON
# text; a call-only reply reports content_null; WITHOUT tools declared the marker stays text in content; and a
# malformed call body is never dropped. The issue asked for name, arguments and a non-tool reply that must not be
# parsed as one; the last two arms are the loss-proofing the header promises at :193 and :216-221.
add_executable(npu_answer_parser_test tests/npu_answer_parser_test.cpp)
target_include_directories(npu_answer_parser_test PRIVATE app)
target_link_libraries(npu_answer_parser_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME npu_answer_parser COMMAND npu_answer_parser_test)
# direction B (tools/ci/ci_targets_check.py, engine#1595): a target no ci.yml line names is "never built", and its test
# then reports "Not Run" instead of running. `ci.yml` is a workflow file no lane can patch, so the documented hook is the
# one this tree already uses for onebit_npu_forward_syntax: add_dependencies(onebit <target>), which the checker follows.
add_dependencies(onebit npu_answer_parser_test)
# engine#1781's oracle (merged in #1813) arrived with its test file but no registration, so the comparator behind
# NPU_BATCH_CHECK -- "every drafted position's logits, bitwise" instead of one argmax at the last row -- was neither
# built nor run while every job stayed green. Header-only, so the test needs no link libraries; host-only, so it needs no
# model and no device.
add_executable(batch_check_compare_test tests/batch_check_compare_test.cpp)
target_include_directories(batch_check_compare_test PRIVATE npu/forward/include)
add_test(NAME batch_check_compare COMMAND batch_check_compare_test)

# engine#1780: the DFlash draft block's acceptance arithmetic -- tokens = accepted + bonus, the break-even rule, and the
# published OGA-route counts as known answers. Header-only and host-only: no container, no device, no NPU runtime, so it
# runs wherever this file is built. Registered at top level with the same hook as the row above, for the same reason.
# The per-layer weight-cache keys are distinct (npu/forward/include/weight_keys.h): two projections of one layer
# under one key launch with the same weights on the device. Header-only, no model.
add_executable(weight_keys_test tests/weight_keys_test.cpp)
target_include_directories(weight_keys_test PRIVATE npu/forward/include)
add_test(NAME weight_keys COMMAND weight_keys_test)
add_dependencies(onebit weight_keys_test)
# engine#1780: where a tapped layer's rows go (npu/forward/include/layer_taps.h), header-only, no model.
add_executable(layer_taps_test tests/layer_taps_test.cpp)
target_include_directories(layer_taps_test PRIVATE npu/forward/include)
add_test(NAME layer_taps COMMAND layer_taps_test)
add_dependencies(onebit layer_taps_test)
add_executable(dflash_accept_test tests/dflash_accept_test.cpp)
target_include_directories(dflash_accept_test PRIVATE npu/forward/include)
add_test(NAME dflash_accept COMMAND dflash_accept_test)
add_dependencies(onebit dflash_accept_test)
# direction B (tools/ci/ci_targets_check.py): a target ci.yml's --target list does not name is "never built", and ci.yml
# is a workflow file no lane can patch. The documented hook is the one onebit_npu_forward_syntax uses.
add_dependencies(onebit batch_check_compare_test)
add_test(NAME npu_design_grid COMMAND npu_design_grid_test)
# The plan's `designs` section against the engine's own derivation (engine#803 A2a): the generator and
# q4nx_derive_designs() must state the same kernels, because a section that disagreed would be refused at
# load for every model that carried one.  Runs the dump binary, so it is a build dependency above.
add_test(NAME npu_plan_designs_parity
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/npu_plan_designs_parity_test.py
                 ${CMAKE_SOURCE_DIR}/tools/npu_plan.py $<TARGET_FILE:npu_derive_designs_dump>)
# The host fallback GEMM reads a padded design's activations at stride a_k, inside the caller's buffer
# (engine#808 review: the split moved the engine#845 pad into the target and left the host branch on the
# raw A). The test's activation buffer carries a NaN guard tail, so an implementation that reads past the
# buffer fails even when its numbers look plausible. Host-only, no model, no device.
add_executable(host_gemm_pad_test tests/host_gemm_pad_test.c npu/forward/src/q4nx_pack.c npu/forward/src/q4nx_dequant.c)
target_include_directories(host_gemm_pad_test PRIVATE npu/forward/include)
target_link_libraries(host_gemm_pad_test PRIVATE m)
add_test(NAME host_gemm_pad COMMAND host_gemm_pad_test)
# A PLAIN bf16 tensor (rows*cols*2 bytes, no tile header) must decode as bf16, not as 5120-byte
# q4_1 tiles, and a buffer of any other size must be refused (engine#875).  Plain C, no device.
add_executable(npu_dequant_bf16_test tests/npu_dequant_bf16_test.c npu/forward/src/q4nx_dequant.c)
target_include_directories(npu_dequant_bf16_test PRIVATE npu/forward/include)
add_test(NAME npu_dequant_bf16 COMMAND npu_dequant_bf16_test)
# ci.yml's target list needs the `workflow` scope to extend, so bind this binary to a target the list
# already builds (same reason as npu_forward_loader_test above); drop this line when the workflow is next
# touched with that scope and add npu_dequant_bf16_test to the list itself.
add_dependencies(npu_forward_loader_test npu_dequant_bf16_test)
# qwen3_5 ships BOTH RMSNorm conventions under one model_type, so the forward resolves (1 + w) vs w from
# the layer-0 input_layernorm weight (engine#796).  The rule is exported and checked here with the two
# real shapes on this box; no model, no device.  Plain C.
add_executable(npu_rmsnorm_offset_test tests/npu_rmsnorm_offset_test.c npu/forward/src/q4nx_semantics.c)
target_include_directories(npu_rmsnorm_offset_test PRIVATE npu/forward/include)
target_link_libraries(npu_rmsnorm_offset_test PRIVATE m)
add_test(NAME npu_rmsnorm_offset COMMAND npu_rmsnorm_offset_test)
# Same ci.yml workaround as above: npu_sampling_test is in the workflow's target list, this binary is not.
add_dependencies(npu_sampling_test npu_rmsnorm_offset_test)
# layer_types -> per-layer kind, and the name the forward refuses an unimplemented kind by (q4nx_forward.cpp,
# the engine#1087 shape).  Plain C, links the semantics TU directly, no model and no device.
# Interleaved MRoPE is declarable in two spellings (rope_parameters for OGA/Qwen3.5 packs, rope_scaling for a
# Hugging Face qwen3_vl config); the flag is what a route or a refusal can act on, so both are pinned here, with
# the controls that keep a yarn/linear rope_scaling block from being read as mrope.  Plain C, no model, no device.
add_executable(q4nx_mrope_test tests/q4nx_mrope_test.c npu/forward/src/q4nx_semantics.c)
target_include_directories(q4nx_mrope_test PRIVATE npu/forward/include)
target_link_libraries(q4nx_mrope_test PRIVATE m)
add_test(NAME q4nx_mrope COMMAND q4nx_mrope_test)
add_dependencies(npu_sampling_test q4nx_mrope_test)
# Which models get the per-head q/k RMSNorm: config.json's use_qk_norm when the config states it, else the
# model_type table (engine#1889: Hunyuan declares it under a model_type the table has no row for, so its gains were
# loaded and never applied).  Plain C, links the semantics TU directly, no model and no device.
add_executable(q4nx_qk_norm_test tests/q4nx_qk_norm_test.c npu/forward/src/q4nx_semantics.c)
target_include_directories(q4nx_qk_norm_test PRIVATE npu/forward/include)
target_link_libraries(q4nx_qk_norm_test PRIVATE m)
add_test(NAME q4nx_qk_norm COMMAND q4nx_qk_norm_test)
add_dependencies(npu_sampling_test q4nx_qk_norm_test)
add_executable(q4nx_layer_kinds_test tests/q4nx_layer_kinds_test.c npu/forward/src/q4nx_semantics.c)
target_include_directories(q4nx_layer_kinds_test PRIVATE npu/forward/include)
target_link_libraries(q4nx_layer_kinds_test PRIVATE m)
add_test(NAME q4nx_layer_kinds COMMAND q4nx_layer_kinds_test)
add_dependencies(npu_sampling_test q4nx_layer_kinds_test)
set(ONEBIT_NPU_MODEL "" CACHE PATH "Q4NX model directory (model.q4nx + config.json)")
set(ONEBIT_NPU_MOE_GATE "" CACHE FILEPATH "gate.json from tools/moe_expert_chunks.py gate")
set(ONEBIT_NPU_BO_DUMP "" CACHE PATH "The reference lane's buffer dump (RT_DUMP_BOS)")
add_executable(tokenizer_test tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE onebit_npu)
# Large inputs tokenize in linear time (a synthetic vocabulary, no model needed); runs in CI.
add_executable(tokenizer_linear_test tests/tokenizer_linear_test.cpp)
target_link_libraries(tokenizer_linear_test PRIVATE onebit_npu)
add_test(NAME tokenizer_linear COMMAND tokenizer_linear_test ${CMAKE_SOURCE_DIR}/tests/data/tokenizer_linear.json)
# A vocabulary whose normalizer rewrites spaces to the metaspace symbol without prepending one (Gemma4's shape):
# ids held to the reference library's on a synthetic nine-token file; no model needed, runs in CI.
add_executable(tokenizer_replace_meta_test tests/tokenizer_replace_meta_test.cpp)
target_link_libraries(tokenizer_replace_meta_test PRIVATE onebit_npu)
add_test(NAME tokenizer_replace_meta COMMAND tokenizer_replace_meta_test ${CMAKE_SOURCE_DIR}/tests/data/tokenizer_replace_meta.json)
add_dependencies(onebit tokenizer_replace_meta_test)
# A SentencePiece-shaped vocabulary that KEEPS its merge list (engine#1189): the shape Phi-3.5-mini's
# pack has, whose ids come from Hugging Face `tokenizers` on the same fixture. No model needed.
add_test(NAME tokenizer_spm_bpe
         COMMAND tokenizer_test ${CMAKE_SOURCE_DIR}/tests/data/spm_bpe_tokenizer.json ${CMAKE_SOURCE_DIR}/tests/data/spm_bpe_tokenizer_golden.json)
if(ONEBIT_NPU_MODEL)
    add_test(NAME tokenizer_qwen3
             COMMAND tokenizer_test ${ONEBIT_NPU_MODEL}/tokenizer.json ${CMAKE_SOURCE_DIR}/tests/data/qwen3_tokenizer_golden.json)
endif()
# The Gemma family's own tokenizer.json through the NPU reader (engine#2215), against the reference-library cases
# already kept for it under tests/data/tokenizers/. The file is 32 MB and is not in the tree, so this needs
# -DONEBIT_GEMMA_TOKENIZER=<a Gemma4 tokenizer.json> (sha256 cc8d3a0ce364..., the one the cases were made from).
set(ONEBIT_GEMMA_TOKENIZER "" CACHE FILEPATH "A Gemma4 tokenizer.json, for the NPU reader's reference cases")
if(ONEBIT_GEMMA_TOKENIZER)
    add_test(NAME tokenizer_gemma4
             COMMAND tokenizer_test ${ONEBIT_GEMMA_TOKENIZER} ${CMAKE_SOURCE_DIR}/tests/data/tokenizers/Gemma4-E4B-IT-NPU2.json)
endif()
if(ONEBIT_NPU_MODEL AND ONEBIT_NPU_BO_DUMP)
    add_test(NAME npu_pack_model COMMAND npu_pack_test ${ONEBIT_NPU_MODEL} ${ONEBIT_NPU_BO_DUMP})
endif()
# The private NPU route registry (npu/private_route.h), without an add-on.
add_executable(npu_private_route_test tests/npu_private_route_test.cpp)
target_link_libraries(npu_private_route_test PRIVATE onebit_npu)
add_test(NAME npu_private_route COMMAND npu_private_route_test)
# `1bit npu-pack` is host-only: it reads a vendor pack's weights and writes our container,
# so it belongs to a build without the device lane too (docs/npu.md, "OGA packs").
target_link_libraries(onebit PRIVATE onebit_npu)
target_compile_definitions(onebit PRIVATE ONEBIT_NPU_HOST)
endif()  # ONEBIT_NPU_HOST

# Device-free: `tests/serve_kill_tree_test.sh` drives the helper against a synthetic wrapper, so this needs neither
# the NPU host lane nor Python -- registered unconditionally. It sat inside ONEBIT_NPU_HOST /
# Python3_Interpreter_FOUND by accident (appended next to route_port_lock in engine #1653), which silently
# unregistered it in any build with -DONEBIT_NPU_HOST=OFF.
# Ending a serve must end the process that holds the memory: `1bit serve` is a wrapper and the model lives in the
# `llama-server` child (engine #1653 review). Device-free: it drives the helper against a synthetic wrapper.
add_test(NAME serve_kill_tree COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_kill_tree_test.sh)

# The NPU forward library: one source list and one link interface, built against a runtime target named by the
# caller. if(ONEBIT_NPU) builds onebit_npu_forward against the private runtime; a configure without it (CI)
# builds onebit_npu_forward_ci against onnx/ep/npu_runtime_ci_stub.cpp, whose device calls throw -- enough for
# the host-GEMM forward (init_host), which never opens the device (engine#1946).
option(ONEBIT_NPU_DEBUG_DUMPS "NPU forward: env-named debug dumps (developer builds)" OFF)
function(onebit_npu_forward_library name runtime)
    add_library(${name} STATIC
        # gemm_q8_raw / gemm_f32: the int8 GEMM the ONNX EP's QDQ leg uses, and now the embedding route's
        # ternary linears on the array (engine#1311). Its own probe checks the accumulators are exact.
        npu/ort/onebit_npu_gemm.cpp
        npu/forward/src/model.c
        npu/forward/src/q4nx_dequant.c
        npu/forward/src/q4nx_pack.c
        # The production weight packer, extracted from I8Ctx so it can be tested without the device runtime (#1416).
        npu/forward/src/q4nx_pack_blocked.cpp
        # The lm-head image cache (engine#1716): the file format, and the load-or-pack a serve calls per head tile.
        npu/forward/src/q4nx_head_image.c
        npu/forward/src/q4nx_head_tile_cache.cpp
        npu/forward/src/q4nx_semantics.c
        npu/forward/src/q4nx_forward.cpp
        # The NPU target interface and this box's target (engine#799 P4): the model keeps the maths, the
        # target keeps the runtime, the kernels and the weight residency.
        npu/target/xdna/xdna_target.cpp)
    # The Q4NX pack and dequant loops carry `#pragma omp parallel for`; without the
    # OpenMP flags those compile to serial code. Wire it in (both C and C++ sources)
    # so they actually use the cores.
    find_package(OpenMP)
    if(OpenMP_C_FOUND AND OpenMP_CXX_FOUND)
        target_link_libraries(${name} PUBLIC OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    endif()
    # NPU_INFER_DUMP_ATTN / _LN / _HEAD write debug dumps to a path taken from the environment;
    # only debug builds carry them (ONEBIT_NPU_DEBUG_DUMPS)
    if(ONEBIT_NPU_DEBUG_DUMPS)
        target_compile_definitions(${name} PRIVATE NPU_INFER_DEBUG_DUMPS)
    endif()
    target_include_directories(${name} SYSTEM PUBLIC
        npu/forward/include npu/forward/engine-src npu ${CMAKE_CURRENT_SOURCE_DIR})
    # plan.json (npu/plan/plan.h, engine #799 P1) is read with nlohmann's parser, and the header is
    # part of this library's interface, so the dependency is PUBLIC.
    # engine#1185: the forward converts an imported q4nx-2 container to the device's 5,120-byte
    # q4_1 form before model_load (npu/q4nx_device.cpp, onebit_npu), so the host layer is part of
    # this library's link interface.
    target_link_libraries(${name} PUBLIC onebit_npu ${runtime} nlohmann_json::nlohmann_json dl
                                                  pthread m)
endfunction()

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
    # engine#753: does the engine EXECUTE on a dma-buf it did not allocate? This one does open the device (it imports a udmabuf and
    # hands the exporter's mapping to the runtime), so it takes the kernel ELF as an argument and SKIPS (77) without one -- a box
    # with no ELF registers a skip rather than a red row, the same rule as the other device-dependent rows.
    add_executable(npu_dmabuf_import_test tests/npu_dmabuf_import_test.cpp)
    target_link_libraries(npu_dmabuf_import_test PRIVATE onebit_npu_lane)
    add_test(NAME npu_dmabuf_import COMMAND npu_dmabuf_import_test "")
    set_tests_properties(npu_dmabuf_import PROPERTIES SKIP_RETURN_CODE 77)

    # The model-generic forward (npu/forward): the GGUF-on-NPU serve route
    # (docs/npu.md, "Open"). q4nx_forward.cpp + its Q4NX dequant/pack/semantics
    # + model loader, against the engine tree's I8Ctx header.
# ── 1-bit sentence embeddings on the NPU (engine#1311) ────────────────────────────────────────
# The encoder's ternary linears run as int8 GEMMs through the engine's own NPU GEMM (npu/ort/
# onebit_npu_gemm.cpp, the same call the ONNX EP's QDQ leg uses), so a device row can be produced from the
# tree: encode a fixed sentence list and compare with the reference vectors.
if(ONEBIT_NPU)
    # The decoder's precomputed-embedding entry point (npu/forward/embeds_equiv_probe.cpp): rows built from the
    # model's own table must give byte-identical logits to the token-id path, on the host and on the array.
    # The three-plane (MRoPE) position input: equal planes must be byte-identical to plain RoPE, different planes
    # must move the logits (npu/forward/mrope_planes_probe.cpp).
    # The wiring a vision-conditioned prompt needs on the engine side: vision rows + 3-plane positions + deepstack
    # sets through step_batch_embeds(), then a greedy decode (npu/forward/hybrid_prompt_probe.cpp).
    add_executable(hybrid_prompt_probe npu/forward/hybrid_prompt_probe.cpp)
    target_include_directories(hybrid_prompt_probe PRIVATE ${CMAKE_SOURCE_DIR}/npu
        ${CMAKE_SOURCE_DIR}/npu/forward/include ${CMAKE_SOURCE_DIR}/npu/forward/engine-src)
    target_link_libraries(hybrid_prompt_probe PRIVATE onebit_npu_forward onebit_npu_rt)
    # engine#1788: one picture and one question through a MiniCPM-V 4.7 pack with every step the engine's own
    # (plan, resize, host tower, prompt tokens, canvas positions, then step_batch_embeds on the text model).
    add_executable(minicpm_image_probe npu/forward/minicpm_image_probe.cpp)
    target_include_directories(minicpm_image_probe PRIVATE ${CMAKE_SOURCE_DIR}/npu
        ${CMAKE_SOURCE_DIR}/npu/forward/include ${CMAKE_SOURCE_DIR}/npu/forward/engine-src)
    target_link_libraries(minicpm_image_probe PRIVATE onebit_npu onebit_npu_forward onebit_npu_rt)
    add_executable(mrope_planes_probe npu/forward/mrope_planes_probe.cpp)
    target_include_directories(mrope_planes_probe PRIVATE ${CMAKE_SOURCE_DIR}/npu
        ${CMAKE_SOURCE_DIR}/npu/forward/include ${CMAKE_SOURCE_DIR}/npu/forward/engine-src)
    target_link_libraries(mrope_planes_probe PRIVATE onebit_npu_forward onebit_npu_rt)
    add_executable(embeds_equiv_probe npu/forward/embeds_equiv_probe.cpp)
    target_include_directories(embeds_equiv_probe PRIVATE ${CMAKE_SOURCE_DIR}/npu
        ${CMAKE_SOURCE_DIR}/npu/forward/include ${CMAKE_SOURCE_DIR}/npu/forward/engine-src)
    target_link_libraries(embeds_equiv_probe PRIVATE onebit_npu_forward onebit_npu_rt)
    add_executable(embed_npu_gate_probe npu/embed/embed_gate_probe.cpp app/embed_model.cpp
                   npu/ort/onebit_npu_gemm.cpp)
    target_include_directories(embed_npu_gate_probe PRIVATE ${CMAKE_SOURCE_DIR}/app
        ${CMAKE_SOURCE_DIR}/npu ${CMAKE_SOURCE_DIR}/npu/forward/include
        ${CMAKE_SOURCE_DIR}/npu/forward/engine-src ${CMAKE_SOURCE_DIR}/npu/ort ${CMAKE_SOURCE_DIR}/npu/embed)
    target_compile_definitions(embed_npu_gate_probe PRIVATE ONEBIT_NPU_RUNTIME_OWN)
    target_link_libraries(embed_npu_gate_probe PRIVATE onebit_npu_forward onebit_npu_rt
        nlohmann_json::nlohmann_json)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(embed_npu_gate_probe PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

    # The library body is onebit_npu_forward_library() (defined above the runtime options), so the CI build of the
    # same sources against the runtime stub (engine#1946) cannot drift from this one.
    onebit_npu_forward_library(onebit_npu_forward onebit_npu_rt)
    target_sources(onebit PRIVATE app/forward_serve.cpp)
    # engine#2111: the usage object's pure function, its own TU so tests/serve_usage_test.cpp can link it without
    # dragging in npu/rt.h (see cmake/tests-serve-usage.cmake).
    target_sources(onebit PRIVATE app/serve_usage.cpp)
    # app/embed_serve.cpp's device path includes npu/embed/embed_npu.h (engine#1311): the embedding route's
    # ternary linears on the array, through the same int8 GEMM the forward links.
    target_include_directories(onebit PRIVATE ${CMAKE_SOURCE_DIR}/npu/embed ${CMAKE_SOURCE_DIR}/app
        ${CMAKE_SOURCE_DIR}/npu/ort)
    target_link_libraries(onebit PRIVATE onebit_npu_forward)
    set(ONEBIT_NPU_KERNELS "" CACHE PATH "Kernel directory (layer_ctx{1,2,17}.elf, lmhead.elf, layer.pdi)")
    # step_batch_all(): per-row logits from ONE batched pass -- the contract speculative verification
    # rests on (a draft token is accepted only when the model's own choice at that position matches it).
    # Host-GEMM mode, no device; needs a Q4NX container, so it skips (77) unless ONEBIT_TEST_Q4NX names one.
    add_executable(mtp_batch_all_test tests/mtp_batch_all_test.cpp)
    target_include_directories(mtp_batch_all_test PRIVATE npu/forward/include)
    target_link_libraries(mtp_batch_all_test PRIVATE onebit_npu_forward)
    add_dependencies(npu_sampling_test mtp_batch_all_test)
    add_test(NAME mtp_batch_all COMMAND mtp_batch_all_test)
    # The batched GDN recurrence is head-major (one parallel region per layer, not per layer AND row); the engine
    # asserts the two nestings agree bit for bit under NPU_GDN_CHECK=1, so this test enables it and makes the
    # batched path run.  Host-GEMM, no device; skips (77) unless ONEBIT_TEST_Q4NX names a linear-attention
    # container.
    add_executable(gdn_batch_order_test tests/gdn_batch_order_test.cpp)
    target_include_directories(gdn_batch_order_test PRIVATE npu/forward/include)
    target_link_libraries(gdn_batch_order_test PRIVATE onebit_npu_forward)
    add_dependencies(npu_sampling_test gdn_batch_order_test)
    add_test(NAME gdn_batch_order COMMAND gdn_batch_order_test)
    # engine#1780: set_layer_taps() on a real container -- the last layer's tap is the forward's own residual, bit for
    # bit, in the batched pass and in step(). Host-GEMM, no device; skips (77) unless ONEBIT_TEST_Q4NX names a model.
    add_executable(layer_taps_model_test tests/layer_taps_model_test.cpp)
    target_include_directories(layer_taps_model_test PRIVATE npu/forward/include)
    target_link_libraries(layer_taps_model_test PRIVATE onebit_npu_forward)
    add_dependencies(npu_sampling_test layer_taps_model_test)
    add_test(NAME layer_taps_model COMMAND layer_taps_model_test)
    # engine#1905: the NPU_BATCH_CHECK oracle restores the prefix it replays from instead of resetting it, so a chunk
    # at start_pos > 0 is compared (and then run) from the state the batched pass continues. Host-GEMM, no device;
    # skips (77) unless ONEBIT_TEST_Q4NX names a container. This is the dev-box row (the private runtime, a real
    # container); a configure without ONEBIT_NPU -- CI's -- registers the same test on a fabricated container below.
    add_executable(batch_check_state_test tests/batch_check_state_test.cpp)
    target_include_directories(batch_check_state_test PRIVATE npu/forward/include)
    target_link_libraries(batch_check_state_test PRIVATE onebit_npu_forward)
    add_dependencies(npu_sampling_test batch_check_state_test)
    add_test(NAME batch_check_state COMMAND batch_check_state_test)
    # The MTP drafter measured against the real base model (tools/mtp_acceptance.cpp): draft, verify through
    # step_batch_all(), commit the accepted prefix, repeat.  The head runs on the host in the pb16i-exact
    # arithmetic, the base on the exact route.  Needs the device, so it is a tool, not a ctest.
    # Device-free harness for the batched GDN recurrence: why gdn_layer_batch holds one parallel region per layer
    # instead of one per layer AND row.  Times today's nesting against the exchanged one (both through one body,
    # bit-identity required before any timing is printed) so the region-count claim can be reproduced without the
    # device: 2.06 ms/row idle, 252-301 with spinners, 16.8 vs 4.1 between the two nestings at equal contention.
    add_executable(gdn_row_bench tools/gdn_row_bench.cpp)
    find_package(OpenMP COMPONENTS CXX QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(gdn_row_bench PRIVATE OpenMP::OpenMP_CXX)
    endif()
    # A DFlash draft scored against its target, both on the host (engine#1780): the target's tapped hidden states
    # into the draft's reference forward, proposals checked against the target's greedy tokens. A tool, not a ctest:
    # it needs two packages and minutes of host time.
    add_executable(dflash_acceptance tools/dflash_acceptance.cpp npu/dflash_forward.cpp npu/dflash_draft.cpp)
    target_include_directories(dflash_acceptance PRIVATE npu npu/forward/include)
    target_link_libraries(dflash_acceptance PRIVATE onebit_npu_forward onebit_npu)
    add_executable(mtp_acceptance tools/mtp_acceptance.cpp npu/mtp/mtp_head.cpp)
    target_include_directories(mtp_acceptance PRIVATE npu npu/forward/include)
    target_link_libraries(mtp_acceptance PRIVATE onebit_npu_forward)
    find_package(OpenMP COMPONENTS CXX QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(mtp_acceptance PRIVATE OpenMP::OpenMP_CXX)
    endif()
    set_tests_properties(mtp_batch_all PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(gdn_batch_order PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(layer_taps_model PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(batch_check_state PROPERTIES SKIP_RETURN_CODE 77)
    # The fast-lane end-to-end test (npu_lane_e2e) is retired: its kernels were FastFlowLM-derived
    # and its reference logits are lost. The NPU check is the private dx add-on's from-source
    # check, outside this repository (docs/npu.md, "Checking the NPU").
endif()

# engine#1946: the batch-check state pin in CI. Under if(ONEBIT_NPU) above, batch_check_state_test is a dev-box row (the
# private runtime, a container named by ONEBIT_TEST_Q4NX), so a regression of #1909 -- the oracle resetting the prefix it
# replays from -- would land green. Here, in a configure WITHOUT the NPU lane (CI's), the same test is built from the same
# forward sources linked against the public runtime stub (its device calls throw; the host-GEMM forward never makes one),
# and tests/batch_check_state_ci.py runs it on a two-layer qwen3 container it writes and packs at test time
# (tools/safetensors_to_q4nx.py): no device, no model on disk, under a second. Skips (77) without numpy.
if(ONEBIT_NPU_HOST AND NOT ONEBIT_NPU AND NOT WIN32 AND Python3_Interpreter_FOUND)
    add_library(onebit_npu_rt_ci_stub STATIC onnx/ep/npu_runtime_ci_stub.cpp)
    target_include_directories(onebit_npu_rt_ci_stub PUBLIC ${CMAKE_SOURCE_DIR}/npu)
    target_compile_definitions(onebit_npu_rt_ci_stub PUBLIC ONEBIT_NPU_RUNTIME_OWN)
    target_link_libraries(onebit_npu_rt_ci_stub PUBLIC Threads::Threads)
    onebit_npu_forward_library(onebit_npu_forward_ci onebit_npu_rt_ci_stub)
    add_executable(batch_check_state_test tests/batch_check_state_test.cpp)
    target_link_libraries(batch_check_state_test PRIVATE onebit_npu_forward_ci)
    # direction B (tools/ci/ci_targets_check.py): ci.yml's --target list is a workflow file no lane can patch; the
    # documented hook is the one onebit_npu_forward_syntax uses.
    add_dependencies(onebit batch_check_state_test)
    add_test(NAME batch_check_state
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/batch_check_state_ci.py
                     $<TARGET_FILE:batch_check_state_test>)
    set_tests_properties(batch_check_state PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    # The same test on an LFM2-shaped hybrid the script fabricates (conv, attention, conv): the only container with
    # short-convolution layers CI loads through the forward (engine#1786).
    add_test(NAME batch_check_state_lfm2
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/batch_check_state_ci.py
                     $<TARGET_FILE:batch_check_state_test> lfm2)
    set_tests_properties(batch_check_state_lfm2 PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    # A fabricated qwen3_5 hybrid (a linear-attention layer, then attention): load only, see the script's header.
    add_test(NAME batch_check_state_qwen35_load
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/batch_check_state_ci.py
                     $<TARGET_FILE:batch_check_state_test> qwen3_5)
    set_tests_properties(batch_check_state_qwen35_load PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
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

# ── Test registrations that live in their own file (engine#1696, engine#1979) ─
# Every cmake/tests-*.cmake is included here, in name order. A test added as a NEW file there touches no line any
# other pull request touches, so it cannot conflict and its registration cannot go stale while the PR waits: 37 commits
# changed this file in the 24 h before engine#1979 was filed, and three open PRs had a registration hunk that no longer
# applied. The rows above stay where they are; nothing has to move.
# Here, after every target and after find_package(Python3), so a file may use if(TARGET ...), add_python_test() and
# ${Python3_EXECUTABLE}; and before the RESOURCE_LOCK loop below, which asks if(TEST ...).
# CONFIGURE_DEPENDS re-runs the glob at build time, so a file added after the first configure is picked up.
# Readers that follow this shape: tools/ci/ci_targets_check.py (the glob), tools/ci/hrx_build_scope.py and
# .github/workflows/hrx-build.yml (the cmake/tests-*.cmake path), tools/ci/tests_referenced_check.py (all of cmake/).
# The lines between the two markers are run as they stand by tests/registration_glob_test.sh, in a scratch project.
# >>> registration glob
file(GLOB ONEBIT_TEST_REGISTRATION_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/cmake/tests-*.cmake)
list(SORT ONEBIT_TEST_REGISTRATION_FILES)
# A file this CMakeLists.txt already include()s by name is skipped: that include fixes the file's position (some rows
# need one), and including it twice stops the configure with `add_test given test NAME ... which already exists`. So a
# family file moved out with its own include() line and this glob can land in either order.
# "By name" means a line CMake would execute -- include( at the start of a line, the same rule
# tools/ci/ci_targets_check.py reads by -- and not the same text in a comment: a substring search took
# `# was include(cmake/tests-gpu-hold.cmake)` for an include, and the seven gpu_hold_* tests left the configuration with
# every gate green (review of engine#1983). An include this pattern does not recognise is the loud miss: both includes
# run and the configure stops.
file(READ ${CMAKE_CURRENT_LIST_FILE} _onebit_top_cmake_text)
set(_onebit_include_prefix "(^|\n)[ \t]*include\\([ \t]*\"?(\\$\\{CMAKE_(CURRENT_)?SOURCE_DIR\\}/)?cmake/")
foreach(_onebit_test_registration_file IN LISTS ONEBIT_TEST_REGISTRATION_FILES)
    get_filename_component(_onebit_test_registration_name ${_onebit_test_registration_file} NAME)
    string(REGEX REPLACE "([.+])" "\\\\\\1" _onebit_test_registration_re "${_onebit_test_registration_name}")
    string(REGEX MATCH "${_onebit_include_prefix}${_onebit_test_registration_re}\"?[ \t]*\\)" _onebit_named
           "${_onebit_top_cmake_text}")
    if(_onebit_named STREQUAL "")
        include(${_onebit_test_registration_file})
    endif()
endforeach()
unset(_onebit_top_cmake_text)
# <<< registration glob

# ── Loopback ports: route tests run one at a time (engine#1070) ──────────────
# Every test named below starts a server on 127.0.0.1. Most pick a port by binding :0, closing the
# socket, then letting the child bind that number; three use fixed 43xxx literals. Under `ctest -j`
# two such tests can be handed the same port between the close and the child's bind, which surfaces
# as EADDRINUSE in the child and a JSON decode error on an empty /health body in the parent (pm_route
# did exactly that on PR #1065, and passed on the same commit's re-run). RESOURCE_LOCK makes the
# collision impossible rather than unlikely, and serialises nothing else: the rest of the suite stays
# parallel. A test that starts a server belongs in this list.
# The scripts' own failure path says which port could not be bound, so a socket leaked by an aborted
# run is still named when it bites.
# engine #1485: a script that runs `serve --port 0` no longer picks a port and releases it, so there is no window to
# collide in and nothing for the lock to prevent - those are gone from this list (watchdog_serve, metrics_serve,
# parallel_args, recipes_route, greedy_shortcut, prism_route, hrx_runtime_route, laya_gguf_route, mem_admit_route,
# npu_repack_path, serve_e2e_<dev>, laya_route_e2e, laya_models_e2e, device_route, hrx_cwsr_route, hybrid_route,
# hybrid_kvshare_route), and the list now names exactly the
# tests that still choose their port up front. bench_selftest stays: tools/bench.py picks its own free_port under it.
set(_onebit_route_tests
    smoke_serve print_backend lemonade_hrx template_pin pm_route bench_selftest)
# Tests that only exist for some configurations (a model was passed, a device is built in).
# serve_e2e_<dev>, laya_route_e2e and laya_models_e2e left this list with engine #1485: they run `serve --port 0`
# now, so they pick nothing and there is no window for the lock to close. hrx_cpu_weight_parity stays - it starts `llama-server` itself, whose port is not ours to
# choose (that is also why it is not in the `1bit serve` migration).
# engine #1485: hybrid_kvshare_e2e chooses its port in its own driver (tests/kvshare_fd_handoff.py calls free_port(),
# binds :0, closes, and hands the number to the server it starts), so the census - which reads the .sh text - sees no
# bind to require the lock for. Same shape as bench_selftest, so it is named here for the same reason.
# hybrid_e2e is no longer here: it takes `serve --port 0` and reads the port back (this PR).
list(APPEND _onebit_route_tests pm_laya_route hrx_cpu_weight_parity hybrid_kvshare_e2e)
foreach(_onebit_route_test IN LISTS _onebit_route_tests)
    if(TEST ${_onebit_route_test})
        set_tests_properties(${_onebit_route_test} PROPERTIES RESOURCE_LOCK route_ports)
    endif()
endforeach()

