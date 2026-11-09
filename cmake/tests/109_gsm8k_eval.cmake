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
    # An accuracy figure is the easiest number here to produce wrongly and the hardest to notice (tools/gsm8k_eval.py):
    # a scorer that reads the wrong number out of a completion, or that counts a completion cut off by the token
    # budget as a clean miss, still prints a plausible percentage.  This drives the extractor, the gold normaliser and
    # the whole run loop against canned completions, and re-derives the summary from the written transcript.  Stdlib
    # only, with no route, no socket, no dataset fetch and no device.
    add_test(NAME gsm8k_eval COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/gsm8k_eval_test.py)
endif()
endif()
