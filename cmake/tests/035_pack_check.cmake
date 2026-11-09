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
# engine #1461/#1411: the two release rows that read a PACKED tree (scripts/pack-check.sh). Synthetic trees
# built with the C compiler: a build-machine path in an ELF, an absolute RUNPATH, a relative rpath that
# resolves outside the payload, and a runtime set shipped without the aqlprofile agent libhsa opens. Each
# arm asserts the row's message, and a clean tree is the control. No package build, no GPU, runs in CI.
add_test(NAME pack_check COMMAND ${CMAKE_SOURCE_DIR}/tests/pack_check_test.sh)
