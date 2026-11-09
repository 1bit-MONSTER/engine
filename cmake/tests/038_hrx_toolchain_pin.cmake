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
# engine #713: the pinned HRX toolchain (config/hrx-toolchain.config, scripts/fetch-rocm-sdk.sh) -- the
# pin file's refusals, the completeness check against a fabricated tree, and that cmake and
# weekly.sh both use the pin instead of the box's /opt/rocm-therock. No network, no SDK, no GPU.
add_test(NAME hrx_toolchain_pin COMMAND ${CMAKE_SOURCE_DIR}/tests/hrx_toolchain_pin.sh)
