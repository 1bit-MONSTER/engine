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
#
# The registration glob's own test (engine#1979), registered the way it asks new tests to be: this file is named by
# nothing and is picked up by the cmake/tests-*.cmake glob near the end of CMakeLists.txt.
#
# It configures the glob block of the real CMakeLists.txt in scratch projects: a new file is registered, a file named
# only in a comment is still registered, and a file CMakeLists.txt include()s by name is registered once. Configure
# only (project(... NONE)): no compiler, no device, about a second.
add_test(NAME registration_glob
         COMMAND bash ${CMAKE_SOURCE_DIR}/tests/registration_glob_test.sh ${CMAKE_COMMAND} ${CMAKE_SOURCE_DIR}/CMakeLists.txt)
