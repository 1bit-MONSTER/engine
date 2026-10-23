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

**The 1bit engine runs inside [Lemonade](https://github.com/lemonade-sdk/lemonade)**, and its
`onebit` recipe passes Lemonade's own LLM test suite on HRX: 29 of 29 tests run, all pass. Lemonade
is the server users talk to, with its own catalog, downloads, router and UI. For the models it
serves with 1bit, it runs the engine as a backend, the same way it runs `llama-server`. For a GGUF
it starts the engine's HRX `llama-server` directly, one HTTP hop
([#475](https://github.com/1bit-MONSTER/engine/pull/475), [below](#the-recipe-that-runs-it)); for
the other routes it starts `1bit serve` ([serve.md](serve.md)), waits for `/health`, and forwards
OpenAI requests to it. The engine exposes nothing but that OpenAI-compatible API.

Recovery stays Lemonade's: when a backend hangs, `1bit serve` answers `/health` with 503 and, once
a request has had no progress for `--watchdog` seconds, kills its backend and exits non-zero
([serve.md, "When a backend hangs"](serve.md#when-a-backend-hangs)). Lemonade's BackendWatchdog
then loads the model again.

## How it got here

Step 1 first went the other way. `1bit lemonade` ran Lemonade v11.9.0's server
core inside the `1bit` process, and this repository carried Lemonade local
recipes for the engine (`onebit`, `mlx`, `zinc`, the `hrx_device` option).
A review[^geramyl] pointed out that the engine should be embedded into Lemonade, not
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
backend selector picks the device (`hrx` or `npu`; unset,
the recipe sends `--device auto`). Replies already carry Lemonade's model name (`--alias`),
so requests pass through unchanged.

**One hop for GGUF models** (fork branch `1bit/one-hop`). For a `.gguf`, the recipe first runs that
command with `--print-backend`: serve makes its load-time checks (architecture, memory, recipes)
and prints the llama-server it would start, and Lemonade starts that llama-server itself and
talks to it directly, as it does for `llamacpp`. A request then crosses one HTTP hop instead of
two. NPU model directories, DwarfStar, `--pm` and `--laya` still run through `1bit serve`
(serve exits 5 for them), and `LEMONADE_ONEBIT_ONE_HOP=0` keeps serve in the path for every model
([serve.md, "One hop from Lemonade"](serve.md#one-hop-from-lemonade---print-backend)).

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
the upstream release sync below). Before merging, rebuild and rerun the LLM suite on Strix Halo.

On Strix Halo the recipe passes Lemonade's own LLM test suite
(`test/server_llm.py --wrapped-server onebit`) on HRX (the checklist below has the counts;
Vulkan passed too before RFC #213 stage 3 removed it). It is not proposed upstream yet: the plan is to embed the engine fully
in the fork first, then propose it upstream. The fork follows upstream's releases:
`.github/workflows/sync-lemonade-fork.yml` checks daily and merges upstream's latest
release tag (never a pre-release or `main`), so each engine release carries the Lemonade
release current when it was built. It
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
| HRX | 19 / 31 | **29 / 29 run, all pass** (2 skipped; release pre-check 2026-10-02, from the packaged tarballs) |
| NPU | 3 / 31 | in progress ([npu.md](npu.md)); the suite's small test models are GGUF |
| Vulkan (removed in RFC #213 stage 3) | 19 / 31 | 29 / 29 run, all pass, before its removal |

1. ~~**Declare what already works.**~~ Done (fork #6). The recipe now declares tool calls,
   `stop`, async, the Responses API, slots, tokenize, embeddings and reranking.
2. ~~**Test on the device asked for.**~~ Done (fork #6). The harness passes `--backend` to
   `onebit`.
3. ~~**Forward the rest of llama-server's API.**~~ Done (#69). `1bit serve` forwards
   `/v1/responses`, `/tokenize`, `/slots`, `/props` and `/metrics` on the llama-server devices.
4. ~~**Embedding and reranking models.**~~ Done (#70, fork #6). `1bit serve --embedding` /
   `--reranking`, and the recipe serves those modes.
5. ~~**Reranking on HRX.**~~ Done. jina-reranker-v1-tiny reranks on HRX (the suite's three
   rerank tests return the same top 3 as on Vulkan), and `1bit serve --rerank` runs on HRX (#281).
6. **The NPU from Lemonade.** The path works end to end: Lemonade downloads the Q4NX checkpoint
   repo, the recipe hands `1bit serve` the directory, and the engine runs it on the NPU (the
   experimental Qwen3.6-35B-A3B route builds with a closed add-on only,
   [npu.md](npu.md#add-on-routes)). Still open:
   - the NPU engine's kernels built from source, verified but not merged ([npu.md](npu.md));
   - ~~`reasoning_content`~~: done (#73), the NPU route splits the thinking out as llama-server does;
   - ~~models in our own Hugging Face repos~~: the 35B is `1bit-MONSTER/Qwen3.6-35B-A3B-NPU` (fork #8). (engine + fork)
7. **Beyond the LLM suite:**
   - the Laya router.

`echo` and generation parameters are skipped, because Lemonade's own llamacpp recipe
declares neither.

## Ubuntu 26.04: the PPA's lemond

On Ubuntu 26.04 Lemonade installs from its PPA, and that `lemond` (2026.42.1~26.04) is upstream's,
with no `onebit` recipe. The fork's Debian 13 `.deb` does not install there either: it depends
on `libcpp-httplib0.41`, which 26.04 does not have. So on 26.04 the engine plugs into the stock
`lemond` through the HRX recipe it already has, `llamacpp-hrx`, with no patch to Lemonade:

- `lemond` looks for that recipe's `llama-server` in `LEMONADE_HRX_HRX_BIN`, then in `hrx.hrx_bin`
  in its `config.json`. A path there skips its download of its own HRX build.
- The engine package's systemd drop-in, `lemonade/1bit-hrx.conf`
  ([config/lemonade/1bit-hrx.conf](../config/lemonade/1bit-hrx.conf)), sets
  `LEMONADE_HRX_HRX_BIN` to `lemonade-hrx` in the package.
- `lemonade-hrx` ([scripts/lemonade-hrx](../scripts/lemonade-hrx)) runs the package's HRX
  `llama-server`, with our HIP kernels, on Lemonade's arguments. First it runs
  `1bit serve --print-backend-env`, which makes serve's load-time checks (architecture, memory)
  and prints the environment serve gives an HRX backend: strict placement (`GGML_HRX_STRICT=1`),
  TheRock's HSA runtime, CWSR placement (engine#315) and the ternary decode. It also fills in
  what Lemonade's arguments leave to llama-server's defaults with serve's values: `-ngl 99`,
  `--cache-reuse 256`, and `-kvu` above `--parallel 1`. A variable already set in `lemond`'s
  environment wins, as it does under serve.

To install:

```sh
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt install lemonade-server
tar -I zstd -xf 1bit-<version>-linux-x86_64.tar.zst
sudo 1bit-<version>-linux-x86_64/install-lemonade-ppa.sh      # to /opt/1bit; restarts lemond
```

`install-lemonade-ppa.sh` ([scripts/install-lemonade-ppa.sh](../scripts/install-lemonade-ppa.sh))
copies the package to `/opt/1bit` (or the directory it is given, outside `/home`, since the
service runs with `ProtectHome=yes`). It writes the drop-in to
`/etc/systemd/system/lemond.service.d/1bit-hrx.conf` and reloads systemd. If the PPA's postinst
did not already add the `lemonade` user to `render`, the script does. HRX also needs TheRock in
`/opt/rocm-therock` ([hrx.md](hrx.md)).

To run `lemond` as yourself instead of as the service, set the same variable for it
(`LEMONADE_HRX_HRX_BIN=/opt/1bit/lemonade-hrx lemond`) or put the path in `hrx.hrx_bin` in its
`config.json`.

Lemonade's HRX catalog (the `*-HRX` models) then runs on the engine, and so does a GGUF you
register with `recipe: "llamacpp-hrx"`. Per-model arguments go in Lemonade's `hrx_args`, except
the ones Lemonade sets itself (`-m`, `--ctx-size`, `--device`, `--port`, `--jinja`,
`--metrics`). NPU model directories, DwarfStar, `--pm` and `--laya` are not
llama-server, so they stay with `/opt/1bit/1bit serve` ([serve.md](serve.md)), outside the stock
`lemond`.

[scripts/check-ubuntu-2604-lemond.sh](../scripts/check-ubuntu-2604-lemond.sh) runs this whole path
in a clean `ubuntu:26.04` container: PPA, package, a model registered with `llamacpp-hrx`, a load
and a chat completion. It passes when `lemond` started `lemonade-hrx` rather than installing its
own HRX build, and when the `llama-server` it ran has `GGML_HRX_STRICT=1`. Without `--gpu` it
needs no GPU: a stand-in records the `llama-server` command and environment, then runs Ubuntu's
CPU llama.cpp. On Strix Halo, `--gpu` runs the real one on HRX0:

```sh
scripts/check-ubuntu-2604-lemond.sh out/1bit-<version>-linux-x86_64.tar.zst ~/models/Qwen3-0.6B-Q4_K_M.gguf --gpu
```

## Vulkan, ROCm, Metal and CUDA are Lemonade's

On the GPU the engine is pure HRX with HIP kernels; no Vulkan, ROCm backend, Metal, ZINC or CUDA (RFC #213 stage 3, 2026-10-01: HRX + NPU only; the
core strip, 2026-10-04: no ZINC, so no CUDA; 2026-10-12: no MLX/Metal route and no Windows
package). Lemonade ships its own backends for Vulkan, ROCm, CUDA, the CPU, MLX on Apple Silicon and
Windows, so a model, feature or platform the engine does not run on HRX or the NPU is Lemonade's to
serve:

- `1bit serve --device vulkan`, `--device rocm`, `--device zinc` and `--device mlx` exit at once
  with that reason, as do `--lean`, `--adaptive`, `--long-model`, `--prefill-device`, `--zinc` and
  `--mlx-server`
  ([serve.md](serve.md#removed-devices-and-flags)).
- An architecture only the removed Vulkan build ran (Zyphra Zamba, Zamba2 and BlackMamba, and a
  few upstream-only ones) is refused with a pointer to Lemonade's `llamacpp` recipe.
  Qwen3.8-Flash-Next (`qwen4exp`) runs on HRX since llama.cpp pin `e57beb97`.

In the pinned fork ([1bit-MONSTER/lemonade](https://github.com/1bit-MONSTER/lemonade),
`third_party/lemonade` at `80064ba72a98`, fork #15 on `main`), the `onebit` backend has two entries, `hrx` and `npu`
(`backend_versions.json`); its `vulkan` and `cuda` entries are gone. Lemonade offers `onebit`
models only on machines with `hrx` or `npu`. Lemonade's other recipes (its `llamacpp` Vulkan, ROCm
and CUDA backends, MLX and the rest) are Lemonade's own and unchanged.

## On its own

`1bit serve` also works without Lemonade, for any OpenAI client:

```sh
1bit serve -m ~/models/Qwen3-0.6B-Q4_K_M.gguf --port 8000      # --device auto: HRX0, else the CPU
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

[^geramyl]: The review came from [geramyL](https://github.com/Geramy), a moderator on Lemonade's Discord. Thank you.
