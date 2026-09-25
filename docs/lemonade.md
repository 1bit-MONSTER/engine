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
# Lemonade

**The 1bit engine runs inside [Lemonade](https://github.com/lemonade-sdk/lemonade).** Lemonade is
the server users talk to, with its own catalog, downloads, router and UI. For the models it
serves with 1bit, it runs the engine as a backend, the same way it runs `llama-server`: it starts
`1bit serve` ([serve.md](serve.md)), waits for `/health`, and forwards OpenAI requests to it. The
engine exposes nothing but that OpenAI-compatible API.

## How it got here

Step 1 first went the other way. `1bit lemonade` ran Lemonade v11.9.0's server
core inside the `1bit` process, and this repository carried Lemonade local
recipes for the engine (`onebit`, `mlx`, `zinc`, the `hrx_device` option).
geramyL (AMD) pointed out that the engine should be embedded into Lemonade, not
the reverse. So on 2026-09-23:

- `third_party/lemonade` and every local delta were removed. The engine no
  longer builds Lemonade's code or its fetched libraries.
- `1bit serve` took over what the local recipes did, for every device: NPU,
  Vulkan, HRX, ZINC and MLX.

## The recipe that runs it

One recipe in Lemonade runs the engine, `onebit`, in the same shape as its
`llama-server` recipe:

```
1bit serve -m <model> --port <p> [--device ...] [--ctx-size N] [--alias <name>]
```

Lemonade downloads and resolves GGUF checkpoints as for llamacpp, and its
backend selector picks the device (`vulkan`, `hrx`, `npu`, or `cuda` through
ZINC). Replies already carry Lemonade's model name (`--alias`), so requests pass
through unchanged.

The recipe lives in our Lemonade fork,
[1bit-MONSTER/lemonade](https://github.com/1bit-MONSTER/lemonade): upstream
Lemonade plus the `onebit` backend (fork PR #1). Build `lemond` from the fork and
put `1bit` on PATH (or set `$LEMONADE_ONEBIT_BIN`).

The engine pins the Lemonade it is tested with: `third_party/lemonade` is the fork at a
fixed commit. It is never linked into the engine. `scripts/build-lemonade.sh` builds
`lemond` and the `lemonade` CLI from it:

```sh
scripts/build-lemonade.sh ~/.cache/lemonade-pin
LEMONADE_ONEBIT_BIN=$PWD/build/1bit ~/.cache/lemonade-pin/bin/lemond
```

`.github/workflows/bump-lemonade.yml` opens a PR here when the fork's `main` moves (after
the daily upstream sync below). Before merging, rebuild and rerun the LLM suite on Strix Halo.

On Strix Halo the recipe passes Lemonade's own LLM test suite
(`test/server_llm.py --wrapped-server onebit`) on Vulkan and HRX: 31 tests, 9 run,
22 skipped as unsupported. It is not proposed upstream yet: the plan is to embed the engine fully
in the fork first, then propose it upstream. The fork follows upstream daily:
`.github/workflows/sync-lemonade-fork.yml` merges upstream `main` into it. It
pushes only a clean merge that still registers `onebit`; otherwise it opens a PR
in the fork that lists the conflicts. It also keeps the fork's own copies of
upstream's workflows disabled. Lemonade asks for an RFC
before a new backend.

## Embedding checklist

What "fully embedded" still needs, measured on Strix Halo. The runs use the pinned fork
(`third_party/lemonade`), the engine from `main`, and Lemonade's LLM suite
(`test/server_llm.py --wrapped-server onebit --backend <device>`).

| device | 2026-09-25, start | now |
|---|---|---|
| Vulkan | 19 / 31 | **29 / 29 run, all pass** (2 skipped, as for llamacpp) |
| HRX | 19 / 31 | 26 / 29 run: reranking fails in HRX's JIT |
| NPU | 3 / 31 | Qwen3.6-35B-A3B loads and answers through Lemonade; the suite's small test models are GGUF |

1. ~~**Declare what already works.**~~ Done (fork #6). The recipe now declares tool calls,
   `stop`, async, the Responses API, slots, tokenize, embeddings and reranking.
2. ~~**Test on the device asked for.**~~ Done (fork #6). The harness passes `--backend` to
   `onebit`.
3. ~~**Forward the rest of llama-server's API.**~~ Done (#69). `1bit serve` forwards
   `/v1/responses`, `/tokenize`, `/slots`, `/props` and `/metrics` on the llama-server devices.
4. ~~**Embedding and reranking models.**~~ Done (#70, fork #6). `1bit serve --embedding` /
   `--reranking`, and the recipe serves those modes.
5. **Reranking on HRX.** HRX's JIT can't link the fp32 matmul-with-bias kernel that
   jina-reranker-v1-tiny needs. (engine, HRX fork)
6. **The NPU from Lemonade.** Qwen3.6-35B-A3B now runs: Lemonade downloads the Q4NX checkpoint
   repo, the recipe hands `1bit serve` the directory, and the engine runs it on its own lax
   kernels as full ELFs (16.0 tok/s). Still open:
   - the smaller NPU models: their lane kernels are captured, not built from source;
   - ~~`reasoning_content`~~: done (#73), the NPU route splits the thinking out as llama-server does;
   - models in our own Hugging Face repos. (engine + fork)
7. **Beyond the LLM suite:**
   - image generation: `1bit comfy` behind Lemonade's image endpoint;
   - the Laya router;
   - MLX;
   - CUDA through ZINC, which needs an NVIDIA box.

`echo` and generation parameters are skipped, because Lemonade's own llamacpp recipe
declares neither.

## On its own

`1bit serve` also works without Lemonade, for any OpenAI client:

```sh
1bit serve -m ~/models/Qwen3-0.6B-Q4_K_M.gguf --device vulkan --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```
