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
    # the FFN row/block sparsity measurements (docs/npu.md, "FFN row sparsity"): a synthetic GGUF and
    # a synthetic capture, so the byte budget and the capture arithmetic are gated without a model
    add_test(NAME ffn_sparsity COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/ffn_sparsity.py --self-test)
endif()
endif()
