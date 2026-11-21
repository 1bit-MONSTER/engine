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

# engine#3193: the private route cannot carry a tool grammar, so a request with `tools` is refused instead of being
# answered as prose. The refusal is a pure function of the request body (app/npu_tool_grammar.h), so this row needs
# no model, no NPU and no HTTP: header + nlohmann_json only.
add_executable(npu_private_route_tools_test tests/npu_private_route_tools_test.cpp)
target_include_directories(npu_private_route_tools_test PRIVATE app)
target_link_libraries(npu_private_route_tools_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME npu_private_route_tools COMMAND npu_private_route_tools_test)
