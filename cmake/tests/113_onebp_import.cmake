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
    # tools/onebp_import.py (a legacy .1bp package -> this engine's container): the v1 package's index
    # claims Q4NX for every entry while its norms are raw f32, so both the import and --verify have to
    # classify by rank and size -- verify reading the index quant reported 141 of 337 "differing" tensors
    # on a container it had just written.  Synthetic: no gguf, no model.
    add_test(NAME onebp_import
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/onebp_import_test.py)
endif()
endif()
