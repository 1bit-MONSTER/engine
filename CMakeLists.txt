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
    engine/core/parallel.cpp)
target_include_directories(onebit_core PUBLIC engine/core)
find_package(Threads REQUIRED)
target_link_libraries(onebit_core PUBLIC Threads::Threads)

add_library(onebit_cpu STATIC engine/backends/cpu/cpu_model.cpp)
target_include_directories(onebit_cpu PUBLIC engine/backends/cpu)
target_link_libraries(onebit_cpu PUBLIC onebit_core)

# Unit tests: synthetic data only, so they run anywhere (including CI).
foreach(t test_gguf test_dequant)
    add_executable(${t} tests/${t}.cpp)
    target_include_directories(${t} PRIVATE tests)
    target_link_libraries(${t} PRIVATE onebit_core)
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
endif()
