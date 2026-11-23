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
#
# llama-npu-server (tools/npu_accel, docs/npu-accel.md): llama.cpp's server with the NPU ACCEL backend registered, so a
# model llama.cpp runs on the CPU gets its weight matmuls on the NPU. Its own CPU-only llama.cpp build, from the same
# submodule: the HRX build registers the GPU backend at startup, and this server must never touch the GPU.
# Needs ONEBIT_NPU (the NPU runtime). With ONEBIT_NPU_PRIVATE, the add-on's extension (onebit_npu_accel_ext) is linked
# in and claims the matmuls it has kernels for.
option(ONEBIT_NPU_ACCEL "Build llama-npu-server: llama.cpp (CPU build) with the NPU ACCEL backend" OFF)
if(ONEBIT_NPU_ACCEL)
    if(NOT ONEBIT_NPU)
        message(FATAL_ERROR "ONEBIT_NPU_ACCEL needs ONEBIT_NPU=ON (the NPU runtime)")
    endif()
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    include(ExternalProject)
    set(_acc_llama ${CMAKE_BINARY_DIR}/npu-accel/llama)
    set(_acc_libs llama-server-impl llama-common llama mtmd ggml ggml-cpu ggml-base)
    set(_acc_byproducts "")
    foreach(_l ${_acc_libs})
        list(APPEND _acc_byproducts ${_acc_llama}/bin/lib${_l}.so)
    endforeach()
    ExternalProject_Add(llama_npu_accel_cpu
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp
        BINARY_DIR ${_acc_llama}
        DOWNLOAD_COMMAND ""
        CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON -DGGML_OPENMP=ON
            -DGGML_HRX=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_BACKEND_DL=OFF
            -DLLAMA_BUILD_SERVER=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
            -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${_acc_llama}/bin -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${_acc_llama}/bin
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target llama-server
        BUILD_BYPRODUCTS ${_acc_byproducts}
        INSTALL_COMMAND "")
    set(ONEBIT_NPU_ACCEL_GGML_INCLUDE ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/src)
    # The backend, shared by the server and its test.
    add_library(onebit_npu_accel STATIC tools/npu_accel/ggml_npu_accel.cpp npu/forward/src/q4nx_pack_blocked.cpp)
    add_dependencies(onebit_npu_accel llama_npu_accel_cpu)
    target_include_directories(onebit_npu_accel PUBLIC ${ONEBIT_NPU_ACCEL_GGML_INCLUDE} ${CMAKE_SOURCE_DIR}/tools/npu_accel
        PRIVATE ${CMAKE_SOURCE_DIR}/npu/forward/include)
    target_link_libraries(onebit_npu_accel PUBLIC onebit_npu_rt OpenMP::OpenMP_CXX)
    target_link_directories(onebit_npu_accel PUBLIC ${_acc_llama}/bin)
    target_compile_features(onebit_npu_accel PUBLIC cxx_std_20)
    add_executable(llama-npu-server tools/npu_accel/llama_npu_server.cpp)
    target_link_libraries(llama-npu-server PRIVATE onebit_npu_accel ${_acc_libs})
    set_target_properties(llama-npu-server PROPERTIES BUILD_RPATH "${_acc_llama}/bin" INSTALL_RPATH "$ORIGIN")
    # No device, no model: the scheduler hands MUL_MAT / MUL_MAT_ID to the backend (host emulation) and the results
    # match the CPU-only graph within W8A8 error.
    add_executable(npu_accel_host_test tools/npu_accel/tests/npu_accel_host_test.cpp)
    target_link_libraries(npu_accel_host_test PRIVATE onebit_npu_accel ggml ggml-cpu ggml-base)
    set_target_properties(npu_accel_host_test PROPERTIES BUILD_RPATH "${_acc_llama}/bin")
    add_test(NAME npu_accel_host COMMAND npu_accel_host_test)
endif()
