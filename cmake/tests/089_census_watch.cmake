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
    # the new-model watcher's alert contract (offline: synthetic batches against the
    # committed registry, plus FastFlowLM's org listing and releases from the recorded
    # payloads under tests/fixtures/fflm/ — engine #588), so a regression cannot leave the
    # daily alert permanently green.
    add_test(NAME census_watch COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/census_watch_test.py)
endif()
endif()
