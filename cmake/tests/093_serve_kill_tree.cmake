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
    # engine#1070: every configured test that starts a server holds RESOURCE_LOCK route_ports, so
    # the list at the end of this file cannot drift away from the tests it protects. Reads the
    # configured test list (ctest -N --show-only=json-v1); no device, no model, no build products.
    # Ending a serve must end the process that holds the memory: `1bit serve` is a wrapper and the model lives in the
# `llama-server` child (engine #1653 review). Device-free: it drives the helper against a synthetic wrapper.
add_test(NAME serve_kill_tree COMMAND ${CMAKE_SOURCE_DIR}/tests/serve_kill_tree_test.sh)
endif()
endif()
