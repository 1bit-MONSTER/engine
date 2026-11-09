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
# build-static.sh must never print a quotable artifact hash for a tree that is not a commit (engine#1326, from
# engine#1323's mis-quoted hash). The toolchain is stubbed inside the test - it is the provenance line under test,
# not the compile - so this costs milliseconds and needs no rustup, no musl target and no network.
add_test(NAME build_static_provenance COMMAND ${CMAKE_SOURCE_DIR}/tests/build_static_provenance_test.sh)
endif()
