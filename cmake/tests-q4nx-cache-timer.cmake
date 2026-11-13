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
# engine#2263: scripts/install-q4nx-cache-timer.sh generates a systemd --user sweep for a box that never serves. Its
# --self-test checks the generated units, the refusals (--max-gib 0 means OFF, an empty --interval never fires) and the
# capability probe that stops a timer being installed against a binary older than engine#2251 -- all with stubs, so no
# systemd, no cache and no binary are touched.
add_test(NAME q4nx_cache_timer_self COMMAND ${CMAKE_SOURCE_DIR}/scripts/install-q4nx-cache-timer.sh --self-test)
