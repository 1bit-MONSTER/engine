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
# scripts/lemonade-hrx: the PPA lemond's HRX recipe runs our llama-server (docs/lemonade.md): no GPU.
add_test(NAME lemonade_hrx COMMAND ${CMAKE_SOURCE_DIR}/tests/lemonade_hrx.sh $<TARGET_FILE:onebit>)
