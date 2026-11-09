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
    # The other half of that gate: the plan's container.tile_order is a measurement read from the repack's
    # stamp, not a constant -- a plan written for a directory the repack refused would hand the loader the
    # one value it accepts, undoing that refusal.  Stdlib only: no model, no numpy, no device.
    add_test(NAME npu_plan_tile_order COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/npu_plan_tile_order_test.py)
endif()
endif()
