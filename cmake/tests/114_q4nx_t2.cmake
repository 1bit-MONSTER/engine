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
if(Python3_Interpreter_FOUND)
    # the t2 chunk kind (docs/npu.md "The t2 chunk kind"): synthetic only, because the real-model
    # leg needs 14 GB of Bonsai-2 images. Run tests/q4nx_t2_test.py without --no-model for the gate.
    add_test(NAME q4nx_t2 COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/q4nx_t2_test.py --no-model)
endif()
endif()
