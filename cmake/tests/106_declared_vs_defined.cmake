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
    # Declared-vs-defined guard for the forward classes (engine#1575): CI links only --target onebit, so a declaration with no
    # definition is invisible until a target that odr-uses it is linked -- and the NPU targets cannot be linked in CI at all.
    # Arity-aware on purpose: the defect that motivated it was a three-argument overload declared while only the four-argument
    # one had a body (#1573, fixed by #1574).
    add_test(NAME declared_vs_defined COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tests/declared_vs_defined_test.py)
endif()
endif()
