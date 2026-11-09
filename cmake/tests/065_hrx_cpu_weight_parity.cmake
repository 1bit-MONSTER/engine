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
if(ONEBIT_HRX)
if(ONEBIT_SERVE_TEST_GGUF)
        # Placement must not change the answer (engine #674): the same greedy prompt through the CPU, through the device
        # with every layer assigned to the CPU, and fully offloaded, compared byte for byte.
        add_test(NAME hrx_cpu_weight_parity
                 COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_cpu_weight_parity.sh ${ONEBIT_HRX_SERVER} ${ONEBIT_SERVE_TEST_GGUF})
        set_tests_properties(hrx_cpu_weight_parity PROPERTIES SKIP_RETURN_CODE 77)
endif()
endif()
