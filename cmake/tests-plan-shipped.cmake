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
# engine#799: the plans checked in under configs/plans/ were loaded by nothing. docs/npu-plan.md says
# `tools/npu_plan.py <dir> --check` proves a directory's copy still matches - true, and it needs the model
# directory, which CI has not. So the shipped plans had no gate: a hand edit or a generator change could leave one
# refused by Plan::load() and the first evidence would be a lane loading it on the box.
#
# This loads each one through the same loader the forward uses. No model, no device, no pack.
add_executable(npu_plan_shipped_test tests/npu_plan_shipped_test.cpp)
target_include_directories(npu_plan_shipped_test PRIVATE npu)
target_link_libraries(npu_plan_shipped_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME npu_plan_shipped
         COMMAND npu_plan_shipped_test ${CMAKE_SOURCE_DIR}/configs/plans)
add_dependencies(onebit npu_plan_shipped_test)
