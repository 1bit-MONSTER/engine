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
    # the safetensors -> Q4NX converter and the d3 readout head it packs (engine #653): every
    # fixture is synthesised, so this needs numpy and nothing else — no gguf-py, no model, no
    # device, no download.
    add_test(NAME safetensors_to_q4nx
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/safetensors_to_q4nx_test.py)
endif()
endif()
