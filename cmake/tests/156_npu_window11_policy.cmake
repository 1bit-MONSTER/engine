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
# The window11 gate's comparison policy (engine#834): no tolerance without a measurement behind it, the
# statistic named, and a control pair setting the tolerance from its own spread.  Shell + the fixtures it
# writes itself, so there is no target for the build list to miss.
add_test(NAME npu_window11_policy COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/npu-window11.sh --self-test)
endif()
