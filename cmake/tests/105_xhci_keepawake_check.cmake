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
if(Python3_Interpreter_FOUND)
    # The AXB35-02 xHCI keepawake check (docs/axb35-usb-ports.md, engine#823): board-specific, so the test builds fake sysfs
    # trees and asserts each verdict -- including "NOT THIS BOARD", which is what CI will get from the real /sys run and must
    # never be reported as a pass for the board.
    add_test(NAME xhci_keepawake_check COMMAND ${Python3_EXECUTABLE}
             ${CMAKE_SOURCE_DIR}/tests/xhci_keepawake_check_test.py)
endif()
endif()
