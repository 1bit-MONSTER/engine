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
# engine#1702: builds a per-layer MoE expert placement table from a moe_trace, and refuses a prefill-only trace because
# placement is a decode lever (5.4-6.9 % on batched prefill against dedup's 90-94 %). --self-test drives it on synthetic
# traces with a known answer; no model, no device.
add_test(NAME moe_placement COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/moe_placement.py --self-test)
