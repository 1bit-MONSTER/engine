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
# engine #1668: the bench preflight must pin the load mode, because `-lm none` is +17.5 % prefill at -1.6 % decode for +604 MiB of
# GTT. The script existed with a test that nothing ran; a test nobody runs is a comment, so the loader/flag assertions are registered
# here (the test fakes the box, so it needs no device and no ticket).
add_test(NAME bench_preflight COMMAND bash ${CMAKE_SOURCE_DIR}/tools/tests/test-bench-preflight.sh)
