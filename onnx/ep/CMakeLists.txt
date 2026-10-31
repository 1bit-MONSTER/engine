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
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# Unless required by applicable law or agreed to in writing, software
# limitations under the License.

# libonebit_ep.so -- our ORT plugin EP. ONNX Runtime is linked here only by the proof binary: the
# plugin itself resolves every ORT entry point from the OrtApi struct it is handed at load time, so
# the .so is portable across ORT builds that serve OrtApi v23.

# ONEBIT_EP_NPU=ON swaps the EP's MatMul onto the NPU runtime in npu/ort/ (needs the private NPU
# runtime, ONEBIT_NPU_RT_DIR). OFF keeps the host kernel, which is what CI and every non-NPU machine
# build.
option(ONEBIT_EP_NPU "Run the plugin EP's MatMul on the NPU (needs the private NPU runtime)" OFF)

add_library(onebit_ep SHARED
    onebit_ep.cpp
    onebit_ep_kernel.cpp
)
target_include_directories(onebit_ep PRIVATE ${ONEBIT_ORT_PC_INCLUDE_DIRS})

if(ONEBIT_EP_NPU)
    find_package(Threads REQUIRED)
    # The NPU tree this repo builds from: npu/forward/engine-src holds I8Ctx (it includes "rt.h",
    # which refuses to build without the private runtime), npu/ holds rt.h and npu_rt_api.h.
    # The runtime comes from ONEBIT_NPU_RT_DIR, the same knob the engine's ONEBIT_NPU uses.
    if(NOT ONEBIT_NPU_RT_DIR)
        message(FATAL_ERROR
            "ONEBIT_EP_NPU=ON needs -DONEBIT_NPU_RT_DIR=<dir with npu_rt.cpp and npu_rt_api.cpp> "
            "(the private NPU runtime: docs/npu.md)")
    endif()
    set(ONEBIT_NPU_ENGINE_INCS "${CMAKE_CURRENT_SOURCE_DIR}/../../npu/forward/engine-src;${CMAKE_CURRENT_SOURCE_DIR}/../../npu"
        CACHE STRING "Include dirs of the NPU tree holding npu_engine_i8ctx_inc.h (and its runtime)")

    set(_npu_src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../npu/ort/onebit_npu_gemm.cpp
        "${ONEBIT_NPU_RT_DIR}/npu_rt.cpp"
        "${ONEBIT_NPU_RT_DIR}/npu_rt_api.cpp")
    set(_npu_incs ${ONEBIT_NPU_ENGINE_INCS} ${CMAKE_CURRENT_SOURCE_DIR}/../../npu/ort ${ONEBIT_NPU_RT_DIR})
    set(_npu_defs ONEBIT_NPU_RUNTIME_OWN=1)
    set(_npu_libs Threads::Threads)

    target_sources(onebit_ep PRIVATE ${_npu_src})
    target_include_directories(onebit_ep PRIVATE ${_npu_incs})
    target_compile_definitions(onebit_ep PRIVATE ONEBIT_EP_NPU=1 ${_npu_defs})
    target_link_libraries(onebit_ep PRIVATE ${_npu_libs})

    # The module's self-check and the deliverable-3 device proof: one MatMul on the NPU against a
    # host reference. Needs ONEBIT_EP_NPU_ELF/M/K/N in the environment; skipped without them.
    add_executable(onebit_npu_gemm_probe ${CMAKE_CURRENT_SOURCE_DIR}/../../npu/ort/onebit_npu_gemm_probe.cpp
        ${_npu_src})
    target_include_directories(onebit_npu_gemm_probe PRIVATE ${_npu_incs})
    target_compile_definitions(onebit_npu_gemm_probe PRIVATE ${_npu_defs})
    target_link_libraries(onebit_npu_gemm_probe PRIVATE ${_npu_libs})
    add_test(NAME onebit_npu_gemm_probe COMMAND onebit_npu_gemm_probe 1024 1024 4 3)
    set_tests_properties(onebit_npu_gemm_probe PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS NPU gemm matches"
        SKIP_REGULAR_EXPRESSION "NPU unavailable")
endif()
set_target_properties(onebit_ep PROPERTIES
    OUTPUT_NAME onebit_ep
    PREFIX "lib"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
# Only the two entry points ORT looks up by name are exported (ONEBIT_EP_API in onebit_ep.cpp).

add_executable(onebit_ep_proof onebit_ep_proof.cpp)
target_link_libraries(onebit_ep_proof PRIVATE PkgConfig::ONEBIT_ORT_PC)
target_compile_definitions(onebit_ep_proof PRIVATE
    ONEBIT_EP_DEFAULT_PATH="$<TARGET_FILE:onebit_ep>")
add_dependencies(onebit_ep_proof onebit_ep)

enable_testing()
add_test(NAME onebit_ep_proof COMMAND onebit_ep_proof $<TARGET_FILE:onebit_ep>)
set_tests_properties(onebit_ep_proof PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS OneBitEp ran the MatMul|SKIP OneBitEp declined")

# Negative control: with nothing claimed the CPU EP must take the MatMul and our library must stay
# idle, so the numerics above can only come from our kernel when the first test passes.
add_test(NAME onebit_ep_proof_cpu_fallback COMMAND onebit_ep_proof $<TARGET_FILE:onebit_ep>)
set_tests_properties(onebit_ep_proof_cpu_fallback PROPERTIES
    ENVIRONMENT "ONEBIT_EP_OPS=none"
    PASS_REGULAR_EXPRESSION "PASS OneBitEp left alone the MatMul")

# The A8W8 QDQ shape Quark's ONNX-to-ONNX flow emits: int8 in, int8 out, the four nodes claimed as
# ONE fused node so the dequantised float tensors never exist (docs/onnx-pack.md). The proof gates the
# int8 outputs at worst |d| = 0 LSB against the CPU EP and measures the peak RSS of the run.
add_executable(onebit_ep_qdq_proof onebit_ep_qdq_proof.cpp)
target_link_libraries(onebit_ep_qdq_proof PRIVATE PkgConfig::ONEBIT_ORT_PC)
add_dependencies(onebit_ep_qdq_proof onebit_ep)
add_test(NAME onebit_ep_qdq_proof COMMAND onebit_ep_qdq_proof $<TARGET_FILE:onebit_ep>)
set_tests_properties(onebit_ep_qdq_proof PROPERTIES
    PASS_REGULAR_EXPRESSION "PASS OneBitEp fused the A8W8 QDQ group")

# Negative control for the same graph: with fusion off the EP claims the MatMul alone and runs the
# float kernel, so the fused numbers above can only come from the fused path.
add_test(NAME onebit_ep_qdq_proof_unfused COMMAND onebit_ep_qdq_proof $<TARGET_FILE:onebit_ep>)
set_tests_properties(onebit_ep_qdq_proof_unfused PROPERTIES
    ENVIRONMENT "ONEBIT_EP_QDQ=0"
    PASS_REGULAR_EXPRESSION "PASS OneBitEp ran the A8W8 QDQ MatMul unfused")
