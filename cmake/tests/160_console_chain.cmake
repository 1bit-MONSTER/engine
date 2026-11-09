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
# embed/usr/local/bin/1bit-console owns tty1 and its whole job is that there is no path from this screen to a bare
# prompt; nothing tested it. Same shape as the test above - the script is copied and its absolute paths rewritten into
# a temp tree, so every branch (installer, desktop, TUI-then-menu, no-TUI) is reachable without root or a build.
add_test(NAME console_chain COMMAND ${CMAKE_SOURCE_DIR}/tests/console_chain_test.sh)
endif()
