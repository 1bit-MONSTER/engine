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

# onnx/ — the ONNX Runtime in-process integration. Only built with -DONEBIT_ORT=ON
# (off by default, so CI and normal engine builds never need libonnxruntime).
#
# Works two ways:
#   in-tree   : add_subdirectory(onnx) from the root, which links onebit_ort_info
#               into the single binary and registers the ctest
#   standalone: cmake -S onnx -B onnx/build && cmake --build onnx/build \
#                       && ctest --test-dir onnx/build
#               (no HIP, no submodules, seconds to build — the cheap proof)

cmake_minimum_required(VERSION 3.21)

# The pin: libonnxruntime 1.23.2, sha256 of libonnxruntime.so.1.23.2 recorded in
# onnx/README.md. pkg-config is how the Debian dev package describes itself;
# a vendored build that ships libonnxruntime.pc is found the same way.
find_package(PkgConfig REQUIRED)
pkg_check_modules(ONEBIT_ORT_PC REQUIRED IMPORTED_TARGET libonnxruntime>=1.23)

add_library(onebit_ort_info STATIC ort_info.cpp)
target_link_libraries(onebit_ort_info PUBLIC PkgConfig::ONEBIT_ORT_PC)
set_target_properties(onebit_ort_info PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(onebit_ort_info PRIVATE -O2 -Wall -Wno-unused-result -std=c++17)

add_executable(ort_info ort_info_main.cpp)
target_link_libraries(ort_info PRIVATE onebit_ort_info)
target_compile_options(ort_info PRIVATE -O2 -Wall -std=c++17)

# ort-run: a plain ONNX model through ORT with our plugin EP registered first (docs/onnx.md,
# "Qwen3-0.6B"). In-tree the root links this object into `1bit` as `1bit ort-run`.
add_library(onebit_ort_run STATIC ort_run.cpp)
target_link_libraries(onebit_ort_run PUBLIC PkgConfig::ONEBIT_ORT_PC)
set_target_properties(onebit_ort_run PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(onebit_ort_run PRIVATE -O2 -Wall -std=c++17)

add_executable(ort_run ort_run_main.cpp)
target_link_libraries(ort_run PRIVATE onebit_ort_run)
target_compile_options(ort_run PRIVATE -O2 -Wall -std=c++17)

# Standalone mode owns the test registry; in-tree, the root CMakeLists already
# called enable_testing() before this subdirectory is added.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
endif()
add_test(NAME ort_info COMMAND ort_info)
# The assertion, not just the exit code: the loaded library must be API v23
# (1.23.x) or the plugin-EP ABI we build against does not exist.
set_tests_properties(ort_info PROPERTIES PASS_REGULAR_EXPRESSION "api     : v23")

# The plugin EP lives in its own directory: it is a .so with its own lifetime and its own proof.
add_subdirectory(ep)
