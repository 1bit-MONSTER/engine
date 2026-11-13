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
#
# engine#2111: the chat reply's `usage` object as a decision table. It links app/serve_usage.cpp and NOT
# app/forward_serve.cpp, because the latter pulls `npu/rt.h`, which refuses a build with no NPU runtime - and that is
# exactly why this field could only ever be observed on the device before.
add_executable(serve_usage_test tests/serve_usage_test.cpp app/serve_usage.cpp)
target_include_directories(serve_usage_test PRIVATE app)
target_link_libraries(serve_usage_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME serve_usage COMMAND serve_usage_test)
add_dependencies(onebit serve_usage_test)
