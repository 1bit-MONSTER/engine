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
# patch-sfs must not destroy the artifact it was asked to replace (#1491: two pre-pack refusals did `rm -f "$out"` before the
# output had been reassigned to its .partial name, so a failed cut deleted the previous good SFS). Skips without root, since
# patch-sfs needs uid/gid and setuid bits - this is the only test here that does, hence the property.
add_test(NAME patch_sfs_atomic COMMAND ${CMAKE_SOURCE_DIR}/tests/patch_sfs_atomic_test.sh)
set_tests_properties(patch_sfs_atomic PROPERTIES SKIP_RETURN_CODE 77)
endif()
