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
# The t2 chunk (npu/q4nx.h, "The t2 chunk kind"): the same kind of contract, but the thing under
# test is the shipped decoder, so it compiles npu/q4nx.cpp as well.  No device, no model.
add_test(NAME npu_t2_layout
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_t2_layout_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}
                 ${CMAKE_SOURCE_DIR}/tests/npu_t2_layout_test.cpp)
endif()
