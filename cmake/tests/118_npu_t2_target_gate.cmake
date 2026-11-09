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
    # the acceptance gate for a t2 consumer (docs/npu.md, "Gating a t2 target"): the pinned
    # accumulation order, bit for bit -- it accepts the oracle and refuses four near-misses
    # (float64 accumulation, a reversed group order, a one-bit change, a short dump).  numpy only;
    # the real-tensor leg needs --tensor/--dump and runs by hand.  engine #804.
    add_test(NAME npu_t2_target_gate
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/npu_t2_target_gate.py --quiet)
endif()
endif()
