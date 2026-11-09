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
# engine#1639: the half of the survivor bound that a CPU-only box can drive. The bound itself fires only for a payload root
# wedged in uninterruptible sleep, which user space cannot create, so this covers the other half: a holder that released with
# a survivor must leave a record, and the NEXT holder must warn about it and clear it.
add_test(NAME gpu_hold_survivors COMMAND bash ${CMAKE_SOURCE_DIR}/tests/gpu_hold_survivors_test.sh)
