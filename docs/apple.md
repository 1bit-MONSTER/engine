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
# Apple Silicon: MLX through `1bit serve`

On macOS, `1bit serve --device mlx` serves MLX models ([serve.md](serve.md)).
The executor is the `server` binary of lemon-mlx-engine (fork
`bong-water-water-bong/lemon-mlx-engine`). It builds MLX's Metal backend on
macOS and speaks the OpenAI chat and completions API.

```
1bit serve -m mlx-community/Qwen3-0.6B-4bit --device mlx --port 8000 [--mlx-server PATH]
```

## How it works

- **Models.** `-m` is an `mlx-community` Hugging Face id. The MLX server
  downloads it on first load.
- **Loading.** `serve` starts `<server> <hf id> --port <p>` as a private child,
  which preloads the model, and waits on `/health`.
- **Requests.** Chat and completion requests go out with the Hugging Face id in
  `model`, because the MLX server selects its model by that id. Replies carry
  the served name back (`--alias`, else the id's last part).
- **The server binary** is found in this order: `--mlx-server`, then
  `$LEMONADE_MLX_SERVER`, then `lemon-mlx-server` on PATH.

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

Measured on 2026-09-23 on an Apple M4 (10-core, 16 GB) with macOS 26.6. This
was the same launch contract, but through the engine's former Lemonade `mlx`
recipe (tests/mlx_lemonade_e2e.sh):

- all four `*-MLX` models were listed;
- `Qwen3-0.6B-MLX` answered "The capital of France is Paris.";
- streaming returned token chunks.

`1bit serve --device mlx` reuses that launch contract and request handling, but
it has not been rerun on the Mac yet. The check is ctest `serve_e2e_mlx`, with
`-DONEBIT_MLX_SERVER=<lemon-mlx-engine server>`.

Decode speed of the MLX server on its own, Qwen3-0.6B-4bit, 256 tokens: 150.9
tok/s. Apple's `mlx-lm` 0.29.1 does 241 tok/s on the same model. The fork pins
MLX from `NripeshN/mlx@rocm-support`.
