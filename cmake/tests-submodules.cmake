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
# Submodule checkout integrity (engine#2129). Its own family file for the reason engine#1979 gives: a registration in
# CMakeLists.txt's trailing block collides with every other test-adding PR, and this one is a new family -- nothing else
# registers here.
#
# The `--self-test` row is hermetic (two throwaway local repos, no network, no submodule of this tree), so it runs in CI.
# The `--root` mode is NOT a ctest row on purpose: it inspects a working tree's checkouts, and CI checks out at the pin by
# construction, so a row there would test the runner rather than the tree. It is for the box, before a build:
#
#     scripts/check-submodules.sh --root .
add_test(NAME submodules_self COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/check-submodules.sh --self-test)

# The release preflight's use of it (engine#2129): a build tree that is not the pinned tree BLOCKS, and "no checkout to
# compare against" only WARNs, so a fresh clone can still build. The function is cut out of scripts/weekly.sh and run
# against stub checkers -- no submodules, no network, no build.
add_test(NAME weekly_submodule_preflight COMMAND ${CMAKE_SOURCE_DIR}/tests/weekly_submodule_preflight_test.sh)
