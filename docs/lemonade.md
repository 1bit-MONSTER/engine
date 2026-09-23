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

The engine is a backend that [Lemonade](https://github.com/lemonade-sdk/lemonade)
launches, the way it launches `llama-server`. The engine exposes only an
OpenAI-compatible API (`1bit serve`, [serve.md](serve.md)), and Lemonade stays
what it is: the server users talk to, with its own catalog, downloads, router
and UI.

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

## What Lemonade needs

One recipe in upstream Lemonade that launches the engine, in the same shape as
its `llama-server` recipe:

```
1bit serve -m <model> --port <p> [--device ...] [--ctx-size N] [--alias <name>]
```

It then waits for `/health` to answer 200 and forwards OpenAI requests to
`/v1/chat/completions` and `/v1/completions`. That recipe is being prepared as
a pull request to `lemonade-sdk/lemonade`.

## Until then

Point any OpenAI client at `1bit serve` directly:

```sh
1bit serve -m ~/models/Qwen3-0.6B-Q4_K_M.gguf --device vulkan --port 8000
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```
