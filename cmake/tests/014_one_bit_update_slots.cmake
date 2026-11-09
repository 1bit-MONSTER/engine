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
# engine#812: the XDNA import class advertises device execution on an imported dma-buf. This pairs that
# advertisement (the pinned libamdf's import profile) with what the loaded driver actually does (libamdf's
# own CTS: GPU export -> NPU import -> NPU executes on the imported range), keyed on the module's
# fingerprint against third_party/patches/amdxdna/README.md. It skips (77) without the array or without a
# CTS build (ONEBIT_LIBAMDF_CTS_DIR), so public CI neither runs nor fails it.
# 1bit-update's slot decisions (engine#1334 review): the second update targets the slot holding the machine's own
# save layer and must keep it, a foreign layer is still refused. The script sourced as a library over fake /data
# layouts: no root, no mounts, no device, no VM.
add_test(NAME one_bit_update_slots
         COMMAND sh ${CMAKE_SOURCE_DIR}/tests/one_bit_update_slots_test.sh ${CMAKE_SOURCE_DIR})
