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
    # The M-split ceiling (engine#754): tools/split_ceiling.py turns two measured rates into the optimum NPU
    # share, the per-GEMM ceiling and the end-to-end factor, and this pins it to the three iGPU states
    # docs/hetero-prefill.md records (40.7 TFLOP/s, the engine#782 regression, post-#782) so the arithmetic
    # cannot drift from the page.  Stdlib only: no model, no device.
    add_test(NAME split_ceiling COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/split_ceiling_test.py)
endif()
endif()
