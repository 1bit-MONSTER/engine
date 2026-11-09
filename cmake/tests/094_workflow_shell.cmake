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
    # every workflow's `run:` must parse, and the pin bump step must open one PR per moved repo while a
    # repo that went upstream stays a human decision: the step shipped with an unterminated quote that
    # nothing executed until a pin moved, i.e. until its first scheduled run (engine #1148)
    add_test(NAME workflow_shell COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/workflow_shell_test.py)
endif()
endif()
