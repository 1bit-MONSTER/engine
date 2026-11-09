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
# engine#1485: the two port read-backs the route tests use (tests/serve_port.sh). Script-based, no device, no model --
# it feeds the parsers the exact lines `1bit serve` and `llama-server` print, because a read-back that silently returns
# empty turns into "the server did not come up" and is blamed on whatever the row under test checks.
add_test(NAME serve_port COMMAND bash ${CMAKE_SOURCE_DIR}/tests/serve_port_test.sh)
