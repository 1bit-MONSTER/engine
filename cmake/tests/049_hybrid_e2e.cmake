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
if(ONEBIT_SERVE_TEST_GGUF)
    add_test(NAME hybrid_e2e
             COMMAND ${CMAKE_SOURCE_DIR}/tests/hybrid_e2e.sh $<TARGET_FILE:onebit> ${ONEBIT_SERVE_TEST_GGUF} ${ONEBIT_SERVE_TEST_LLAMA_SERVER})
    set_tests_properties(hybrid_e2e PROPERTIES SKIP_RETURN_CODE 77)
endif()
