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
# engine#2021: a hardcoded `/tmp` path is RAM on this box, not disk.
#
# `/tmp` is tmpfs here, so a dump, a staging copy or a benchmark log written there is anonymous memory: swap fills
# and every lane on the box stalls (measured 2026-11-10 21:15Z: /tmp 52 G of 62 G, swap 71/71 GiB). engine#1907/#1952
# bounded the fconst scratch; the literals themselves are still in the tree, 14 of them across 8 files.
#
# Both rows are cheap and device-free: the check is a grep over npu/ app/ tools/ scripts/ against
# tools/ci/tmp_paths.txt, and the ratchet may only shrink.
add_python_test(tmp_path_check ${CMAKE_SOURCE_DIR}/tools/ci/tmp_path_check.py --check --root ${CMAKE_SOURCE_DIR})
add_python_test(tmp_path_check_self ${CMAKE_SOURCE_DIR}/tools/ci/tmp_path_check.py --self-test)
