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
if(ONEBIT_NPU_HOST)
# The MoE container-convention contract (npu/forward/include/moe_container.h, engine#1035): the
# engine converter's `mlp.experts.*` names + [N][K] per-expert tile grid and HF-shaped config keys,
# against the FLM-style `*_exps_proj` names + [K][N] grid and llama.cpp-shaped keys. A wrong
# choice is silent (a missing tensor, or a permuted expert), so the test is a known-answer decode.
# Self-compiling like the layout contracts, with the shipped chunk decoder, so CI needs no target.
add_test(NAME npu_moe_container
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_moe_container_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}
                 ${CMAKE_SOURCE_DIR}/tests/npu_moe_container_test.cpp)
endif()
