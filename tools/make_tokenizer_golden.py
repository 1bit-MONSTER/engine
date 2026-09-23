#!/usr/bin/env python3
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
#
# make_tokenizer_golden.py <tokenizer.json> <out.json> [source label]
#
# Golden cases for tests/hf_tokenizer_test.cpp, made with the Python
# `tokenizers` of the same release as third_party/tokenizers (pip install
# tokenizers==<that version>). The corpus mixes the Qwen3 golden texts with
# multilingual text, code, numbers, whitespace runs and emoji.
import json, sys
import tokenizers

CORPUS = [
    "The capital of France is",
    "Hello, world! How's it going?",
    "  leading and   inner   spaces\tand\ttabs\n\nnewlines",
    "Numbers: 3.14159, 1,000,000 and 2026-09-23T17:30:00Z",
    "def f(x):\n    return x**2  # square\n",
    "if (a != b && c <= d) { return std::max(a, b); }",
    "Ünïcödé naïve café résumé",
    "日本語のテキストと漢字",
    "Привет, как дела?",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "🤖🚀 emoji 👍🏽 and ZWJ 👩‍💻",
    "I'm we're they've he'd she'll it's",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>",
    "",
    " ",
    "a" * 300,
]

tok_path, out = sys.argv[1], sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else tok_path
tok = tokenizers.Tokenizer.from_file(tok_path)
cases = []
for text in CORPUS:
    ids = tok.encode(text, add_special_tokens=False).ids
    cases.append({"text": text, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)})
json.dump({"source": f"tokenizers {tokenizers.__version__}, {label}", "add_special": False, "cases": cases},
          open(out, "w"), ensure_ascii=False, indent=1)
print(f"{out}: {len(cases)} cases, tokenizers {tokenizers.__version__}")
