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
project(onebit_engine VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENGINE_NPU "Build the XDNA 2 NPU backend (needs XRT)" OFF)
option(ENGINE_GPU "Build the llama.cpp GPU backend (Vulkan + HRX)" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

enable_testing()

add_library(onebit_core STATIC
    engine/core/gguf.cpp
    engine/core/dequant.cpp
    engine/core/model_config.cpp
    engine/core/parallel.cpp
    engine/core/unicode.cpp
    engine/core/tokenizer.cpp
    engine/core/sampler.cpp)
target_include_directories(onebit_core PUBLIC engine/core)
find_package(Threads REQUIRED)
target_link_libraries(onebit_core PUBLIC Threads::Threads)

add_library(onebit_cpu STATIC engine/backends/cpu/cpu_model.cpp)
target_include_directories(onebit_cpu PUBLIC engine/backends/cpu)
target_link_libraries(onebit_cpu PUBLIC onebit_core)

include(cmake/deps.cmake)

add_library(onebit_server_lib STATIC
    engine/server/text_stream.cpp
    engine/server/chat_template.cpp
    engine/server/generator.cpp
    engine/server/openai.cpp)
target_include_directories(onebit_server_lib PUBLIC engine/server)
target_link_libraries(onebit_server_lib PUBLIC onebit_cpu onebit_minja)

add_executable(onebit_server engine/server/main.cpp)
set_target_properties(onebit_server PROPERTIES OUTPUT_NAME 1bit-server)
target_link_libraries(onebit_server PRIVATE onebit_server_lib onebit_httplib)

# Unit tests: synthetic data only, so they run anywhere (including CI).
foreach(t test_gguf test_dequant test_unicode test_sampler test_text_stream test_openai)
    add_executable(${t} tests/${t}.cpp)
    target_include_directories(${t} PRIVATE tests)
    target_link_libraries(${t} PRIVATE onebit_server_lib)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# Golden test: the CPU reference against HF transformers fp32 logits. Needs a
# model and a golden dir (tools/golden/make_golden.py), so it is registered
# only when both are given, e.g.
#   -DENGINE_GOLDEN_MODEL=/path/Qwen3-0.6B-BF16.gguf -DENGINE_GOLDEN_DIR=/path/golden
add_executable(golden_cpu tests/golden_cpu.cpp)
target_link_libraries(golden_cpu PRIVATE onebit_cpu)
set(ENGINE_GOLDEN_MODEL "" CACHE FILEPATH "GGUF model for the golden test")
set(ENGINE_GOLDEN_DIR "" CACHE PATH "Golden logits directory for the golden test")
if(ENGINE_GOLDEN_MODEL AND ENGINE_GOLDEN_DIR)
    add_test(NAME golden_cpu COMMAND golden_cpu ${ENGINE_GOLDEN_MODEL} ${ENGINE_GOLDEN_DIR})
    # The server end to end over HTTP, with Lemonade's launch flags.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME server_e2e
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/server_e2e.py --server $<TARGET_FILE:onebit_server>
                     --model ${ENGINE_GOLDEN_MODEL} --golden ${ENGINE_GOLDEN_DIR})
endif()

# Tokenizer golden test: encode/decode/pre-token split against HF tokenizers.
# Runs everywhere: the vocab-only GGUF and the cases are committed
# (tools/golden/make_tokenizer_golden.py, docs/tokenizer.md).
add_executable(golden_tokenizer tests/golden_tokenizer.cpp)
target_link_libraries(golden_tokenizer PRIVATE onebit_core)
add_executable(golden_template tests/golden_template.cpp)
target_link_libraries(golden_template PRIVATE onebit_server_lib)
add_test(NAME golden_template_qwen3
         COMMAND golden_template ${CMAKE_SOURCE_DIR}/tests/golden/qwen3-tokenizer/vocab.gguf
                 ${CMAKE_SOURCE_DIR}/tests/golden/qwen3-template/cases.jsonl)
add_test(NAME golden_tokenizer_qwen3
         COMMAND golden_tokenizer ${CMAKE_SOURCE_DIR}/tests/golden/qwen3-tokenizer/vocab.gguf
                 ${CMAKE_SOURCE_DIR}/tests/golden/qwen3-tokenizer/cases.tsv)
