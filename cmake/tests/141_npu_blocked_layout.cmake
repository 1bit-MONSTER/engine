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
# The device weight layout contract (q4nx_weight_layout): pure index math, no device
# and no runtime.  CI builds an explicit target list, so the check compiles itself with
# the build's compiler at test time (tests/npu_blocked_layout_test.sh) rather than being
# one missing target away from "Not Run".
add_test(NAME npu_blocked_layout
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_blocked_layout_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}/npu/forward/include
                 ${CMAKE_SOURCE_DIR}/tests/npu_blocked_layout_test.cpp)
endif()
