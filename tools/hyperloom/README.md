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
# Hyperloom on Strix Halo (gfx1151)

[Hyperloom](https://rocm.docs.amd.com/projects/hyperloom/en/latest/what-is-hyperloom.html) 1.0.0
is AMD's agentic inference optimizer (MIT). It ships tables for Instinct boards only (gfx942/gfx950)
and serving frameworks such as vLLM and SGLang. These files let it tune the engine's llama.cpp fork
on HRX (Strix Halo, gfx1151) through its `--framework custom` + bypass backend route. The first
workload is ZAYA1-8B Q4_K_M decode.

| File | What it does |
| --- | --- |
| `hl_patch_strixhalo.py [pkg]` | Adds a `strixhalo` GPU type (gfx1151, 40 CU, ~256 GB/s, 59.4 TFLOPS f16, 128 GB) next to each `mi355x` entry of an installed Hyperloom package. |
| `hl_cli_login.py [pkg]` | With `ONEBIT_HYPERLOOM_CLI_LOGIN=1`, agent turns use the logged-in `claude` CLI instead of an API key; also lets `RAY_VERSION` override the pinned Ray (2.44.1 has no Python 3.14 wheels). Unsupported upstream. |
| `custom_strixhalo.sh` | The benchmark entrypoint (`custom_<gpu-type>.sh`). Rebuilds the checkout, takes `~/.cache/lax-decode/box.lock`, waits for a cool idle GPU, gates on wikitext perplexity, then reports `llama-bench` tg128 as `output_throughput`. |
| `lemonade_bench.sh` | Lemonade's own benchmark (`lemonade bench`, the settings of Lemonade's nightly benchmark-regression CI: chat-short + code-short, 1 warmup, 5 runs, ctx 4096) on the fork's `llama-server`, routed through `llamacpp.vulkan_bin` of a private lemond. Lemonade reserves `-dev`, so the device goes in as `LLAMA_ARG_DEVICE` (default HRX0). |
| `launch.sh` | Starts `hyperloom ... optimize` for ZAYA1-8B, detached. |
| `zaya1-8b-q4_k_m.reference.json` | Gate reference from the unmodified tree (fork 3b954f0). |

## Two benchmark standards

`output_throughput`, the number Hyperloom optimises, is ours: `llama-bench` tg128. The result also
carries Lemonade's standard under `bench_summary.lemonade_bench` (TPS and TTFT per scenario), for
numbers compared with Lemonade. On ZAYA1-8B at fork 3b954f0: llama-bench tg128 93.5 tok/s; lemonade
bench chat-short 65.4 TPS, code-short 75.0 TPS, TTFT about 361 ms (Lemonade reloads the model for
each run and generates short answers).

## Quality gate

Hyperloom's own quality check targets served models; here the script gates instead. Each candidate
must stay within 0.5% of the reference prefill perplexity (c512, 4 chunks) and within 2% of the
decode-shaped perplexity (`-ub 1`, c128, 4 chunks). Eight prefill chunks trip the 93 C thermal stop
on this box, so the gate uses four. Every step cools below 60 C first and stops at 93 C.

## Setup (strixhalo)

```sh
python3 -m venv ~/hyperloom-ws/.venv && . ~/hyperloom-ws/.venv/bin/activate
pip install "hyperloom-inference-optimizer[runtime]==1.0.0" "ray[default]==2.55.1"
PKG=~/hyperloom-ws/.venv/lib/python3.14/site-packages/hyperloom
python tools/hyperloom/hl_patch_strixhalo.py $PKG
python tools/hyperloom/hl_cli_login.py $PKG
# Hyperloom needs a Ray head with a GPU and a serving_slot resource
ray start --head --num-gpus=1 --resources='{"serving_slot":1}' --disable-usage-stats
mkdir -p ~/hyperloom-zaya/scripts && cp tools/hyperloom/custom_strixhalo.sh ~/hyperloom-zaya/scripts/
cp tools/hyperloom/zaya1-8b-q4_k_m.reference.json ~/hyperloom-zaya/reference.json
# ~/hyperloom-zaya/llama.cpp: a worktree of the fork built with GGML_HRX=ON
# ~/hyperloom-zaya/model -> the ZAYA1-8B Q4_K_M GGUF
cd ~/hyperloom-zaya && (setsid nohup bash /path/to/tools/hyperloom/launch.sh > hyperloom.log 2>&1 < /dev/null &)
```

Setup notes:

- TraceLens pins `xprof==2.20.1`, which does not install on Python 3.14. Install TraceLens with
  `--no-deps` plus `xprof>=2.20.2,<2.21`.
- Hyperloom's `install.sh` also installs the scriptable-quality dependencies (lpips pulls in torch);
  the custom workload does not need them.

Never put credentials in these files.
