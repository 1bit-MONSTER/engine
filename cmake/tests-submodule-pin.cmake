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
# engine#2129: every submodule must be checked out at the commit the tree pins.
#
# A build compiles whatever is in the working tree, and every other check in tools/ci reads the tree it is handed -
# so none of them can see that a submodule sits on a different commit than the superproject pins. Measured on
# 2026-11-11: the shared ~/1bit-engine had four submodules off their pins (llama.cpp, hrx-system, tokenizers,
# xdna-driver), and the engine compiles common/jinja from llama.cpp - which was behind the pin on `\xNN` escapes.
# Jinja decides prompt rendering, so that is a difference in what the model sees, not a build failure.
#
# The default run compares only the submodules that ARE checked out (a configure that never inits them is not this
# check's business); `--require-initialised` is the release-gate form, where a missing checkout is a failure too.
add_python_test(submodule_pin_check ${CMAKE_SOURCE_DIR}/tools/ci/submodule_pin_check.py --check --root ${CMAKE_SOURCE_DIR})
add_python_test(submodule_pin_check_self ${CMAKE_SOURCE_DIR}/tools/ci/submodule_pin_check.py --self-test)

# engine#2129: the release preflight's use of the check -- a build tree whose submodules are not the pinned commits
# BLOCKS the run, and "no checkout to compare" only WARNs so a fresh clone can still build. The function is cut out of
# scripts/weekly.sh and run against stub checkers: no submodules, no network, no build.
add_test(NAME weekly_submodule_preflight COMMAND ${CMAKE_SOURCE_DIR}/tests/weekly_submodule_preflight_test.sh)
