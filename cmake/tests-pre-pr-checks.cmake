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

# engine#2384 added a guard to `scripts/pre-pr-checks.sh`: it refuses to run when `docs/pre-pr-checks.md` and its own `checks`
# array disagree about WHICH checks run, or their order. That guard was author-run only -- measured 2026-11-13, `git grep
# pre-pr-checks.sh` over the tree found nothing but the script itself: no CI step and no ctest row ever invoked it, so the
# page/runner drift that happened TWICE in one night (a count sentence six->seven, and three positions out of order) would still
# have needed a human to type the command. This row makes the agreement a build-time failure.
#
# `--list` runs the comparison first and then prints the list without executing anything, so this costs one script start: no
# network, no submodules, no compiler. The script cds to the git toplevel before resolving the page, so the row is independent of
# the ctest working directory.
add_test(NAME pre_pr_checks_agreement COMMAND ${CMAKE_SOURCE_DIR}/scripts/pre-pr-checks.sh --list)
