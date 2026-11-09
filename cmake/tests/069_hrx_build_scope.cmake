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
if(ONEBIT_NPU_HOST)
# `tools/copyright.py --check` must see a file that has not been staged yet (the check used `git
# ls-files`, so a brand-new file was invisible to it -- PR #1406's test script passed --check locally
# and failed CI's copyright step on the same head).  Runs the real tool in a scratch git repo.
# The HRX build's scope decision (engine#1499, .github/workflows/hrx-build.yml): a PR whose only
# CMakeLists.txt change is a ctest registration skips the compile but still configures.  Each self-test
# case is an artefact someone would actually push, and the conservative direction -- "full" -- is the
# one every uncertain case must land on, because the build being skipped is the one that catches a
# broken HRX pin.  Stdlib only: no git needed for the decision itself.
add_test(NAME hrx_build_scope
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ci/hrx_build_scope.py --self-test)
endif()
