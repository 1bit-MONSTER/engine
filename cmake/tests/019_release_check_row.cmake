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
# the OS image's own release check: its lemonade row must prove the API key is enforced, not infer it
# from an empty body (os/mini/release-check.sh --self-test; no device and no build needed)
add_test(NAME release_check_row COMMAND ${CMAKE_SOURCE_DIR}/os/mini/release-check.sh --self-test)
