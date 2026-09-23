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
# Apple Silicon: MLX through Lemonade

On macOS, `1bit lemonade` serves MLX models through a local Lemonade backend,
`mlx` (delta 6 in `third_party/lemonade/UPSTREAM.md`). The executor is the
`server` binary of lemon-mlx-engine (fork
`bong-water-water-bong/lemon-mlx-engine`). It builds MLX's Metal backend on
macOS and speaks the OpenAI chat and completions API.

## How it works

- **Models.** Each `*-MLX` model in `server_models.json` has `recipe: mlx`. Its
  checkpoint is an `mlx-community` Hugging Face id, and the MLX server downloads
  it on first load.
- **Loading.** `load()` spawns `<server> <checkpoint> --port <p>`, which preloads
  the model, and waits on `/health`.
- **Requests.** Chat and completion requests are forwarded with the checkpoint in
  `model`, because the MLX server selects its model by Hugging Face id. The
  response carries the Lemonade model name back.
- **The server binary** is found in this order: the `mlx_bin` option (config key
  `mlx.mlx_bin`), then `$LEMONADE_MLX_SERVER`, then `lemon-mlx-server` on PATH.

## Build (macOS)

The engine builds without the NPU host library on macOS (`ONEBIT_NPU_HOST`
defaults to OFF there, since it needs PCRE2 and there is no NPU):

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target onebit
```

lemon-mlx-engine needs the Metal Toolchain
(`xcodebuild -downloadComponent MetalToolchain`), Rust (its tokenizer is built
with cargo) and cmake:

```
git clone https://github.com/bong-water-water-bong/lemon-mlx-engine
cmake -S lemon-mlx-engine -B lemon-mlx-engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLX_LM_BUILD_TESTS=OFF
cmake --build lemon-mlx-engine/build
```

## Verified

Measured on 2026-09-23 on an Apple M4 (10-core, 16 GB) with macOS 26.6.
`tests/mlx_lemonade_e2e.sh build/1bit lemon-mlx-engine/build/server`:

- all four `*-MLX` models are listed;
- `Qwen3-0.6B-MLX` answers "The capital of France is Paris." under its Lemonade
  name;
- streaming returns token chunks.

Decode speed of the MLX server on its own, Qwen3-0.6B-4bit, 256 tokens: 150.9
tok/s. Apple's `mlx-lm` 0.29.1 does 241 tok/s on the same model. The fork pins
MLX from `NripeshN/mlx@rocm-support`.
