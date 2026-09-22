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
"""Write tokenizer golden cases from HF tokenizers.

usage: make_tokenizer_golden.py --model <hf_dir_or_id> --out cases.tsv [--random 400]

Each line: hex(utf8 text) <TAB> space-separated ids <TAB> hex(utf8 decode(ids))
           <TAB> hex of the pre-token pieces joined by NUL (normalizer + pre-tokenizer only).
Ids come from encode(text, add_special_tokens=False); special tokens written
in the text are recognised, as in the HF default.
"""
import argparse
import random

from transformers import AutoTokenizer

FIXED = [
    "", " ", "  ", "\n", "\r\n", "\t", "a", "Hello, world!", "Hello world", " Hello",
    "The quick brown fox jumps over the lazy dog.",
    "I'm you're we've they'll he'd it's I'M YOU'RE WE'VE THEY'LL HE'D IT'S",
    "don't won't can't shouldn't",
    "'s 't 're 've 'm 'll 'd", "a'b'c", "'", "''", "' s",
    # Contractions followed by more letters. Each must sit right after a letter
    # (or start the text): after a space, " '" is taken by the punctuation branch.
    "a'sgood b'tx c're d'vex e'mama f'llama g'dx", "I'mhere THEY'LLBE it'Sx", "'dx", "'ſx", "x'ſx",
    "a'Dx b'Mx c'Tx d'Rex e'Vex f'Llx", "'r 'v 'l", "x'r x'v x'l x'lx", "'reX'VEy",
    "1234567890", "3.14159 2,718 1e-9 -42 +7", "version 2.0.1-rc3",
    "trailing spaces   ", "   leading", "a  b   c    d", "a\n\nb\n\n\nc", "x \n y", "x\t\ty",
    "  \n  \n  ", "line1\r\nline2\r\n", "\r\r\n\n",
    "def f(x):\n    return x * 2\n\nif __name__ == '__main__':\n\tprint(f(3))\n",
    "int main() { return 0; }  // comment\n#include <vector>",
    "SELECT * FROM users WHERE id = 42;",
    "https://example.com/path?query=1&x=y#frag",
    "email@example.com, @handle, #hashtag, $100, 50%, a+b=c",
    "!!!??? ... --- *** ((())) [[[]]] {{{}}}",
    "你好，世界！这是一个测试。", "日本語のテキストです。カタカナ、ひらがな。", "한국어 텍스트입니다",
    "Привет, мир!", "Γειά σου Κόσμε", "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "สวัสดีชาวโลก",
    "café naïve résumé Zürich São Paulo", "é ä ñ",  # decomposed: NFC must compose
    "각", "ÅΩ", "ﬁle", "½ Ⅷ ٣٤",
    "😀 😃 🎉 👍🏽 👩‍👩‍👧 🇺🇸 🏳️‍🌈", "emoji😀inside", "∑∫√∞ ≠ ≤ ≥ → ←",
    " nbsp 　ideographic　space", "zero​width", "tab\tsep\tvalues",
    "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nreasoning\n</think>\n\nanswer", "<tool_call>{\"a\": 1}</tool_call>",
    "<|endoftext|><|endoftext|>", "<|im_start", "im_end|>", "<|im_start|><|im_end|>",
    "text<|fim_prefix|>code<|fim_suffix|>more<|fim_middle|>",
    "ſ 'ſ 'ſx", "ǅ ǈ", "'K 'k 'K", "ⅷ", "a" * 300, "ab " * 100, "\U0001d400\U0001d401",
]

POOLS = [
    (0x20, 0x7E), (0x20, 0x7E), (0x20, 0x7E), (0xA0, 0x17F), (0x300, 0x36F), (0x370, 0x3FF),
    (0x400, 0x4FF), (0x600, 0x6FF), (0x900, 0x97F), (0x1100, 0x11FF), (0x2000, 0x206F),
    (0x2070, 0x209F), (0x2100, 0x218F), (0x3000, 0x30FF), (0x4E00, 0x4FFF), (0xAC00, 0xAD00),
    (0xFF00, 0xFFEF), (0x1F300, 0x1F64F),
]
EXTRA = ["\n", "\r\n", "\t", " ", "  ", "'s", "'LL", "<|im_start|>", "<|im_end|>", "<think>"]


def random_text(rng):
    parts = []
    for _ in range(rng.randint(1, 24)):
        if rng.random() < 0.15:
            parts.append(rng.choice(EXTRA))
            continue
        lo, hi = rng.choice(POOLS)
        run = "".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 6)))
        parts.append(run)
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    rng = random.Random(args.seed)
    texts = FIXED + [random_text(rng) for _ in range(args.random)]
    backend = tok.backend_tokenizer
    with open(args.out, "w") as f:
        for text in texts:
            ids = tok.encode(text, add_special_tokens=False)
            dec = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            pieces = [p for p, _ in backend.pre_tokenizer.pre_tokenize_str(backend.normalizer.normalize_str(text))]
            f.write("%s\t%s\t%s\t%s\n" % (text.encode().hex(), " ".join(map(str, ids)), dec.encode().hex(),
                                          "\0".join(pieces).encode().hex()))
    print("wrote %d cases to %s" % (len(texts), args.out))


if __name__ == "__main__":
    main()
