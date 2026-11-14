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
# engine#2422: `release-check.sh`'s payload-modes row printed the binaries' modes and judged only group/other-writability, so a
# payload whose `1bit/1bit` was 0644 read `PASS payload-modes (modes: 644 1bit/1bit …)` while hrx-placement read `SKIP` -- green
# over a payload that cannot run. The fixture builds three real payload trees (0644, 0755, no binaries) and runs the row against
# each; it fails on the pre-fix script with exactly the message from the issue, which is how it is known to test the claim and not
# the plumbing. ~34 s, no device, no build.
add_test(NAME release_check_payload_modes COMMAND bash ${CMAKE_SOURCE_DIR}/tests/release_check_payload_modes_test.sh ${CMAKE_SOURCE_DIR}/os/mini/release-check.sh)
