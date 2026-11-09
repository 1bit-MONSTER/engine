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
# engine#1517: the prefill flash-attention ratio is measured in adjacent pairs and only quoted when the pairs agree, so a
# loaded box yields VOID instead of a number.  The self-test is the estimator's own gate and needs no GPU: it accepts
# agreement, refuses disagreement, and proves a dropped run cannot shift an arm.
add_test(NAME fa_ratio_estimator
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fa_ratio.py --self-test)
endif()
