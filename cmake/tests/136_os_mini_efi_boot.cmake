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
# engine #709: the installer's firmware boot-entry rules, exercised with a fake efibootmgr and a fake
# /sys/class/block, so CI covers selection/rollback without NVRAM. The OVMF install gate
# (os/mini/vmtest.sh) is the end-to-end row and is run separately.
add_test(NAME os_mini_efi_boot COMMAND ${CMAKE_SOURCE_DIR}/tests/os_mini_efi_boot_test.sh)
endif()
