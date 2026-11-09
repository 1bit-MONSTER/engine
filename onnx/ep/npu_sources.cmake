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

# The ONNX EP's NPU half, listed once because two configurations build it:
#
#   * onnx/ep/CMakeLists.txt with ONEBIT_EP_NPU=ON -- the real one, which also needs ONNX Runtime, the
#     device and the private runtime (ONEBIT_NPU_RT_DIR);
#   * the root CMakeLists' `onebit_ep_npu_linkcheck` -- no ORT, no device, no runtime sources: it
#     compiles this list against npu_runtime_ci_stub.cpp and links, which is the only way CI can see
#     this leg at all (engine #874: a missing declaration *and* a source missing from this list).
#
# Keep every source that the EP's NPU leg needs in ONEBIT_EP_NPU_COMMON_SRC. The runtime itself
# (npu_rt.cpp, npu_rt_api.cpp) is the one part that cannot be listed here: it is private, so the EP
# adds it and the CI check substitutes the stub.

set(ONEBIT_EP_NPU_COMMON_SRC
    ${CMAKE_CURRENT_LIST_DIR}/../../npu/ort/onebit_npu_gemm.cpp
    # I8Ctx::args_fit_elf() (#869) calls q4nx_elf_arg_sizes(), implemented here; the declaration is in
    # npu/forward/include, the definition is not. Without this line the leg links with an undefined
    # symbol, which is how #874's second half stayed green in CI.
    ${CMAKE_CURRENT_LIST_DIR}/../../npu/forward/src/q4nx_pack.c
    # I8Ctx::packB_blocked() calls q4nx_pack_B_int8_blocked() (#1416 moved its body out of npu_engine_i8ctx_inc.h
    # so it can be tested); the definition lives here, so the leg must link it -- the same shape as the line above.
    ${CMAKE_CURRENT_LIST_DIR}/../../npu/forward/src/q4nx_pack_blocked.cpp)

# Include dirs and defines that both configurations share. ONEBIT_NPU_RUNTIME_OWN is what npu/rt.h
# requires before it will alias npu_rt_api as xrt::.
#
# These are FALLBACKS, and consumers must list them after their own configurable tree: a build that
# sets ONEBIT_NPU_ENGINE_INCS is selecting another NPU tree (onnx/ep/README.md) and it has to win for
# the headers it carries, or the EP compiles an in-tree I8Ctx against someone else's runtime.
set(ONEBIT_EP_NPU_COMMON_INCS
    ${CMAKE_CURRENT_LIST_DIR}/../../npu/forward/engine-src
    ${CMAKE_CURRENT_LIST_DIR}/../../npu/forward/include
    ${CMAKE_CURRENT_LIST_DIR}/../../npu
    ${CMAKE_CURRENT_LIST_DIR}/../../npu/ort)
set(ONEBIT_EP_NPU_COMMON_DEFS ONEBIT_EP_NPU=1 ONEBIT_NPU_RUNTIME_OWN=1)
