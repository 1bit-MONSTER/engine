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
# where --device npu finds the GGUF repack script: beside 1bit in a package (scripts/package-linux.sh),
# else ONEBIT_Q4NX_REPACK; no NPU needed (the repack runs before the device)
add_test(NAME npu_repack_path COMMAND ${CMAKE_SOURCE_DIR}/tests/npu_repack_path.sh $<TARGET_FILE:onebit>)
# a build without the NPU lane has no repack path: the test says skip (77) there, not fail
set_tests_properties(npu_repack_path PROPERTIES SKIP_RETURN_CODE 77)
