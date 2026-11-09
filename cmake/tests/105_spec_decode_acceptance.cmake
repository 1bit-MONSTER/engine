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
    # The speculative-decoding harness for AMD's draft models (docs/amd-drafters.md): the acceptance measurement is against
    # a live llama-server, which CI cannot start, so the test covers the part that is wrong silently -- the accepted-prefix
    # rule, the two units printed side by side, and the refusal when no server answers.
    add_test(NAME spec_decode_acceptance COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tests/spec_decode_acceptance_test.py)
endif()
endif()
