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

# engine#3193: the two NPU chat routes must send the model the same prompt. Device-free: the fixture is the FLM template
# the repo pins and the renderer is the one both routes use. 77 (not a pass) when the build has no Jinja renderer.
add_executable(unified_prompt_test tests/unified_prompt_test.cpp)
target_include_directories(unified_prompt_test PRIVATE ${CMAKE_SOURCE_DIR}/app ${CMAKE_SOURCE_DIR}/npu
                            ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(unified_prompt_test PRIVATE onebit_npu nlohmann_json::nlohmann_json)
if(ONEBIT_JINJA)
    target_link_libraries(unified_prompt_test PRIVATE onebit_jinja)
endif()
add_test(NAME unified_prompt COMMAND unified_prompt_test ${CMAKE_SOURCE_DIR}/tests/data/templates/qwen3_flm.json)
add_dependencies(onebit unified_prompt_test)
