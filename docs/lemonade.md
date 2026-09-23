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
through unchanged. On Strix Halo the recipe passes Lemonade's own LLM test suite
(`test/server_llm.py --wrapped-server onebit`) on Vulkan and HRX. It is being
prepared as a pull request to `lemonade-sdk/lemonade`.

## On its own

`1bit serve` also works without Lemonade, for any OpenAI client:

```sh
1bit serve -m ~/models/Qwen3-0.6B-Q4_K_M.gguf --device vulkan --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```
