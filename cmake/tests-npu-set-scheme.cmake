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

# A package can carry two kernel-set schemes: the older sealed one (1BPKCRYP ciphertext + format-1 manifest.json) and
# the signed-manifest one (`manifest` first line `npumanifest 2` + `manifest.sig`, plain ELFs at <id>/layer1.elf).
# tools/npu_set_scheme.py owns the difference for the packager and for the weekly checks; this row drives it against
# fixtures it builds itself, so it needs no device, no private tree and no kernel set on disk.
add_test(NAME npu_set_scheme COMMAND bash ${CMAKE_SOURCE_DIR}/tests/npu_set_scheme_test.sh)
