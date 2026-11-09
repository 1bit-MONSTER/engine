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
    # The offline half (engine#1110): a container whose q4_1 TILE ORDER is foreign has no metadata signature at
    # all -- same chunk kind, same bytes, same names -- so the fixture encodes one known weight in the engine's
    # order and in the foreign one and requires check_q4nx_dir.py --against to accept the first and refuse the
    # second by name.  numpy + gguf-py, no model, no device.
    add_test(NAME check_q4nx_tile_order
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/check_q4nx_tile_order_test.py)
endif()
endif()
