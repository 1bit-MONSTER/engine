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
# engine#1594: every target ci.yml's `cmake --build --target …` list names must exist in the configuration ci.yml
# creates. #1583 added one name for a target defined inside if(ONEBIT_NPU), which CI does not configure -- and ninja
# rejects the whole command line before building anything, so `onebit` stopped being compiled for every PR. Reads the
# two files, no build and no submodules; --self-test drives it.
add_test(NAME ci_targets COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ci/ci_targets_check.py --check --root ${CMAKE_SOURCE_DIR})
