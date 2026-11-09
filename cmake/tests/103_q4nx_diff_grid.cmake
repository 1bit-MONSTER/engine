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
    # The padded-grid read (engine#1097/#1110): tools/q4nx_diff.py decode reads a container whose weight sits on a
    # WIDER tile grid than the model's dimensions (the FastFlowLM repack pads intermediate_size to 512 columns, IM
    # 8960 -> 9216) instead of reporting it as an unreadable reshape.  Pure python + numpy, no model, no device.
    add_test(NAME q4nx_diff_grid COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/q4nx_diff_grid_test.py)
endif()
endif()
