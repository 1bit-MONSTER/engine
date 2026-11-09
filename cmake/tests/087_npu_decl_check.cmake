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
    # A member declared in npu/forward/include/q4nx_forward.h and never defined keeps `main` green while a full
    # `cmake --build` fails at link time (engine#1573: the 3-argument step_batch_embeds).  CI cannot catch it --
    # .github/workflows/ci.yml builds `--target onebit` only, because a full build wants the private NPU runtime --
    # and `-fsyntax-only` cannot see a link-time fact.  This pairs the two lists by (name, arity): device-free, no
    # runtime, no build, milliseconds.
    add_test(NAME npu_decl_check
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ci/npu_decl_check.py --check --root ${CMAKE_SOURCE_DIR})
endif()
endif()
