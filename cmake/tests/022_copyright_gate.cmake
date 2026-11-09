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
# engine#1564: which files the copyright gate looks at -- a staged file is seen, an untracked one is seen (that is why
# running it before `git add` is useful), an ignored one is not. A throwaway repo in a temp dir: no build, no device.
add_test(NAME copyright_gate COMMAND ${CMAKE_SOURCE_DIR}/tests/copyright_gate_test.sh)
