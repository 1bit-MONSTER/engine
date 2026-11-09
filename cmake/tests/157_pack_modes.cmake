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
# The package's file modes (engine#1352): a release artifact's modes are the packer's decision, so every staged file is
# 0755 (was executable) or 0644 (was not), no file or directory is group/other-writable, and the reporter that catches a
# staging line added below the normalisation still fires on a tree nobody normalised.  Fixture tree: no build, no tar.
add_test(NAME pack_modes COMMAND bash ${CMAKE_SOURCE_DIR}/tests/pack_modes_test.sh)
endif()
