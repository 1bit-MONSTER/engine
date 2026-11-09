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
if(ONEBIT_JINJA)
    add_test(NAME npu_template
             COMMAND npu_template_test ${CMAKE_SOURCE_DIR}/tests/data/templates/qwen3_flm.json
                                       ${CMAKE_SOURCE_DIR}/tests/data/templates/zr1.json
                                       --tokenizer ${CMAKE_SOURCE_DIR}/tests/data/tokenizer_chatml.json)
endif()
endif()
