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
    # the 1.9 measurement tool's own paths, on synthetic packs: the "trying next" line it shipped broken (its
    # only real pair verified on the first candidate, so the line never ran) and the residual tolerance it
    # accepts and refuses at. Fixtures are ~200 KB each and built from the rule, so they can be wrong on purpose.
    add_test(NAME fconst19_measure COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/fconst19_measure_test.py)
endif()
endif()
