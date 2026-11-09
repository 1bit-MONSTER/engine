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
if(ONEBIT_NPU_HOST AND NOT WIN32)
    add_test(NAME npu_runtime_option
             COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_runtime_option.sh ${CMAKE_SOURCE_DIR} ${FETCHCONTENT_BASE_DIR})
    set_tests_properties(npu_runtime_option PROPERTIES TIMEOUT 300)
endif()
