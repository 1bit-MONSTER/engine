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
# The advertisement half of that test is a parse of the pinned libamdf, so it is tested device-free here: the
# two fixtures are an advertising import profile and the same profile with the flag dropped from its dma-buf
# entry. Runs everywhere, needs no array, no GPU and no submodule (the pinned file is read only if present).
add_test(NAME npu_import_advert
         COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_import_contract.sh --self-test)
