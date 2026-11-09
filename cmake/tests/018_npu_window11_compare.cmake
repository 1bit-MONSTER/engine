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
# engine#876: the gate row for a folded (pb16i/t2s) container.  Its device half needs the array, but its
# assertion does not: these are the accept/refuse cases of `scripts/npu-window11.sh --compare` on hand-written
# logs -- folded against folded passes, one differing rotated activation fails, a folded log against a plain one
# fails, and the plain row keeps its old ms tolerance.  No device, no model, no build products.
add_test(NAME npu_window11_compare COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_window11_compare.sh)
