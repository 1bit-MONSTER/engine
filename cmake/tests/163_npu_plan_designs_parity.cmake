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
# The plan's `designs` section against the engine's own derivation (engine#803 A2a): the generator and
# q4nx_derive_designs() must state the same kernels, because a section that disagreed would be refused at
# load for every model that carried one.  Runs the dump binary, so it is a build dependency above.
add_test(NAME npu_plan_designs_parity
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/npu_plan_designs_parity_test.py
                 ${CMAKE_SOURCE_DIR}/tools/npu_plan.py $<TARGET_FILE:npu_derive_designs_dump>)
endif()
