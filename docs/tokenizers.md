<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Hugging Face tokenizers

The engine reads any model's `tokenizer.json` through Hugging Face
`tokenizers`, the Rust library behind the `tokenizers` npm and PyPI packages,
at the same release as those packages.

- `third_party/tokenizers` = `huggingface/tokenizers` **v0.23.2** (`88a4498a`).
- `hf_tokenizers/` is our C ABI over it: `onebit_tok_from_json`, `_encode`,
  `_decode`, `_vocab_size` and `_last_error`. It is built as a static library
  with `default-features = false, features = ["onig"]`; the progress bar and
  the unigram-training accelerator are for training only.
  `hf_tokenizers/include/hf_tokenizer.h` adds a small C++ owner,
  `onebit::HfTokenizer`.
- The Rust release is pinned in `hf_tokenizers/rust-toolchain.toml` (1.98.1;
  rustup honours it) and the crate graph in `hf_tokenizers/Cargo.lock`
  (`cargo build --locked`).
- `bump-tokenizers.yml` moves the pin to each new release and refreshes the lock.

There is no official C binding upstream. mlc-ai/tokenizers-cpp wraps the same
crate, but it pins 0.21.2.

## Build and test

```sh
cmake -B build -DONEBIT_HF_TOKENIZERS=ON -DONEBIT_TOKENIZER_ROOT=~/.config/flm/models
cmake --build build --target hf_tokenizer_test && (cd build && ctest -R hf_tokenizer)
```

`tests/data/tokenizers/<model>.json` holds golden cases made with Python
`tokenizers==0.23.2` by `tools/make_tokenizer_golden.py`. The corpus mixes
English, code, numbers, whitespace runs, eight scripts, emoji with ZWJ and
chat markup. Each file names the sha256 of the tokenizer.json it came from.

## Verified 2026-09-23

18 of 18 models, 17 of 17 cases each, ids and decoded text identical:
Qwen3 0.6B/1.7B/4B/8B, Qwen3-VL-4B, Qwen3.5-4B, Qwen3.6-35B-A3B, Llama 3.1-8B
and 3.2-1B/3B, Gemma3 1B/4B, Gemma4 E2B/E4B, Phi4-mini, LFM2 1.2B/2.6B and
Nanbeige4.1-3B. That covers vocabularies from 64k to 262k. The engine's own
Qwen golden (`tests/data/qwen3_tokenizer_golden.json`) also matches, 14 of 14.
