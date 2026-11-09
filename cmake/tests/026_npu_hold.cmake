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
# engine#1701: the NPU hold's queue/token follow ONEBIT_HOLD_DIR, and its queue settings never reach the payload.
# Shell only (a stub gpu-hold plus the tree's own), no device, ~1 s.
add_test(NAME npu_hold COMMAND bash ${CMAKE_SOURCE_DIR}/tests/npu_hold_test.sh)
