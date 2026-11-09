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
# engine#1650: a hold ends itself when the box runs out of memory, before the kernel picks a victim — the 11-08/09
# failure where an unbudgeted GPU run took the box to 0.7 GiB and the OOM killer chose the PR-agent LLM.
add_test(NAME gpu_hold_mem_abort COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_mem_abort_test.sh ${CMAKE_SOURCE_DIR}/scripts/gpu-hold)
