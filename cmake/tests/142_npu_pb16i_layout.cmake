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
# The pb16i weight image (q4nx_weight_layout "pb16i"): the same kind of contract, plus the
# packer that builds the images (tools/q4nx_pb16i.py).  No device, no model: the real-GGUF
# leg is skipped when the model is not on the machine.
add_test(NAME npu_pb16i_layout
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_pb16i_layout_test.sh
                 ${CMAKE_CXX_COMPILER} ${CMAKE_SOURCE_DIR}/npu/forward/include
                 ${CMAKE_SOURCE_DIR}/tests/npu_pb16i_layout_test.cpp)
endif()
