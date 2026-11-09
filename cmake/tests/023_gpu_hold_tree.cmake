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
# engine#1621: a hold that ends must end everything it started. Three orphans in two days were all GRANDCHILDREN
# of a ticket (ticket → shell → runner → server): `pkill -P` reaches one level, so the process holding the memory
# outlived the ticket. Drives scripts/gpu-hold with a wrapper → mid → work chain and asserts the deepest one dies.
add_test(NAME gpu_hold_tree COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_tree_test.sh)
