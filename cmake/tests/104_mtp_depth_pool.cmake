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
    # The pooled per-depth live-vs-offline verdict (tools/mtp_depth_pool.py): the brief's "within 2 pp at every
    # depth" is judged on 8 prompts x 64 tokens POOLED, which is two or three npu-hold tickets of <= 3 min, and
    # pooling those logs by eye produced two wrong numbers in this lane.  Fixture tickets only -- no model, no
    # device; the recorded 8x64 window is the tool's known-answer check and is run by hand with --selftest.
    add_test(NAME mtp_depth_pool COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/mtp_depth_pool_test.py)
endif()
endif()
