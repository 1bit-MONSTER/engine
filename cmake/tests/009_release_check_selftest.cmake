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
# os/mini/release-check.sh --self-test: the shipped release gate's own verdict rows, asserted (engine#1596). It needs no
# build and no device - the subject is the row functions - so it is the one test here that takes no $<TARGET_FILE>, and
# it is here because nothing else runs it: the rows had 128 assertions and no runner. It names no add_executable, so it
# adds nothing to the set of test targets the CI --target list would have to build (the direction-B measurement, #1595).
add_test(NAME release_check_selftest COMMAND bash ${CMAKE_SOURCE_DIR}/os/mini/release-check.sh --self-test)
