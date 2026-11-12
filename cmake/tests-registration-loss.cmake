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
# tools/ci/registration_loss_check.py asks whether a branch still carries the registrations `main` has, because a PR's diff is
# computed against its MERGE BASE: a stale branch shows every row another lane landed as a removal, and a merge that resolves
# the conflict by taking the branch side deletes them. Its --self-test is synthetic repositories, so it needs no network, no
# build and no base ref; the check itself is a pre-PR tool (`--check --base origin/main`), not a ctest row, because it compares
# two revisions and a build only ever has one.
add_python_test(registration_loss_check_self ${CMAKE_SOURCE_DIR}/tools/ci/registration_loss_check.py --self-test)
