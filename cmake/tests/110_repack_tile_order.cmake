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
    # The repack's tile-order claim (engine#1110): `npu_plan.py` writes container.tile_order=engine-1 for every
    # plan, so the load-time guard (#1130) is green on a container whose TILES are another order.  This pins the
    # mapping from q4nx_diff's verdict to (stamp value, fatal), stdlib only: no model, no numpy, no device.
    add_test(NAME repack_tile_order COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/repack_tile_order_test.py)
endif()
endif()
