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
# The t2s weight stream (q4nx_weight_layout "t2s"): the same integer-exact weights at 2.125 b/w,
# expanded on the core.  Its host-side contract is the file size, so the test pins the tile against
# the numbers tools/q4nx_t2s.py measured on a real image (50,135,040 -> 23,674,880 B for gate/up).
add_test(NAME npu_t2s_layout
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_t2s_layout_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}/npu/forward/include
                 ${CMAKE_SOURCE_DIR}/tests/npu_t2s_layout_test.cpp)
endif()
