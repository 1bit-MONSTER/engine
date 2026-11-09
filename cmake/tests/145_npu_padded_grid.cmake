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
endif()
