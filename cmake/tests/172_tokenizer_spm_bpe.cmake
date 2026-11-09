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
# A SentencePiece-shaped vocabulary that KEEPS its merge list (engine#1189): the shape Phi-3.5-mini's
# pack has, whose ids come from Hugging Face `tokenizers` on the same fixture. No model needed.
add_test(NAME tokenizer_spm_bpe
         COMMAND tokenizer_test ${CMAKE_SOURCE_DIR}/tests/data/spm_bpe_tokenizer.json ${CMAKE_SOURCE_DIR}/tests/data/spm_bpe_tokenizer_golden.json)
endif()
