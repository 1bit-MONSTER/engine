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
    # the 2-bit ternary stream the XDNA decode kernel reads (tools/q4nx_t2s.py): the expansion it derives
    # must reproduce the pb16i image's own int4 code bytes, block for block. Synthetic weights only -- the
    # real-image leg is the same check on 14 GB of Bonsai-2 images and is run by hand.
    add_test(NAME q4nx_t2s COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/q4nx_t2s_test.py)
endif()
endif()
