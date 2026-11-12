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
# engine#2129: scripts/pkg-proof.sh carries a --self-test that no row ran. It exercises the row/summary accounting on stubs
# (the counter a packaging run prints is what tells a reader whether the run proved anything), and it needed no container,
# no device and no build -- there was simply no registration. tools/ci/tests_referenced_check.py now covers scripts/*.sh so
# the next one cannot be invisible either; this is the row for the one that was.
#
# A family file rather than a line in CMakeLists.txt: the glob at its end picks this up, so adding it touches no line that any
# other test-adding PR is editing.
add_test(NAME pkg_proof_self COMMAND ${CMAKE_SOURCE_DIR}/scripts/pkg-proof.sh --self-test)
