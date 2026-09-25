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
# `1bit serve`: the engine behind one OpenAI-compatible API

The engine runs inside Lemonade ([lemonade.md](lemonade.md)): Lemonade runs
`1bit serve` as a backend, the way it runs `llama-server`. `1bit serve` exposes
nothing but an OpenAI-compatible API, so it also works on its own with any
OpenAI client.

```sh
1bit serve -m <model> [--port 8000] [--host 127.0.0.1]
           [--device auto|npu|vulkan|hrx|rocm|zinc|mlx] [--ctx-size N] [--alias NAME]
           [--llama-server PATH] [--zinc PATH] [--hrx-libhsa PATH] [--mlx-server PATH]
           [--prefill-device hrx] [--prefill-min-tokens N] [--lean]
           [--mtp HEAD.gguf] [--mtp-max N] [--mtp-p-min P]
           [--parallel N] [--adaptive] [--adaptive-at N]
           [--embed MODEL.gguf] [--rerank MODEL.gguf]
           [--npu-opt KEY=VALUE ...]
```

One model per process:

| Endpoint | |
|---|---|
| `GET /health`, `GET /v1/health` | 503 while the model loads, then 200 |
| `GET /v1/models` | the one model (`--alias`, else the file or directory name) |
| `POST /v1/chat/completions` | streamed (SSE) or not; a thinking model's `<think>` block comes back as `reasoning_content` (`"reasoning_format": "none"` keeps it in `content`) |
| `POST /v1/completions` | |
| `POST /v1/embeddings` | with `--embed` (RAG) |
| `POST /v1/rerank` | with `--rerank` (RAG) |
| `POST /v1/embeddings` or `/v1/rerank` | with `--embedding` or `--reranking`: the model itself is an embedding or reranking model |
| `POST /v1/responses` | OpenAI Responses API, streamed or not (llama-server devices) |
| `POST /tokenize`, `/detokenize`, `/apply-template` | llama-server devices |
| `GET /slots`, `POST /slots/<id>?action=…`, `GET /props`, `GET /metrics` | llama-server devices; slot save and restore answer 501 (llama-server runs without `--slot-save-path`) |

"llama-server devices" means `vulkan`, `hrx` and `rocm`. These routes go to the llama-server
behind the model, which is how Lemonade's llamacpp backend reaches them (the `onebit` recipe
inherits it). On `npu`, `zinc` and `mlx` they answer 501.

## Where the model runs

| Model | `--device` | Runs on |
|---|---|---|
| NPU model directory (`model.q4nx` + `npu/`, docs/npu.md) | `auto`, `npu` | the NPU fast lane, in process |
| Qwen3.6-35B-A3B Q4NX directory (`model_type` `qwen3_5_moe`) | `auto`, `npu` | the private NPU route, in process, in builds with `-DONEBIT_NPU_PRIVATE` (docs/npu.md, "Private routes") |
| `.gguf` | `auto`, `vulkan` | the upstream llama.cpp build's llama-server on `Vulkan0` (docs/vulkan.md); without `ONEBIT_VULKAN`, the HRX build's |
| `.gguf` | `hrx` | the HRX build's llama-server on `HRX0` (docs/hrx.md; MoE models above 128 experts need `-DONEBIT_GPU_PRIVATE`, "Private GPU build") |
| `.gguf` | `vulkan --prefill-device hrx` | the HRX build's llama-server decoding on `Vulkan0`, long prompt prefixes prefilled on `HRX0` over one shared KV cache, in builds with `-DONEBIT_GPU_PRIVATE` (docs/hrx.md, "Prefill on HRX, decode on Vulkan") |
| ROCmFP4 `.gguf` | `auto`, `vulkan` with `--lean` | the lean (ROCmFPX) build's llama-server on `Vulkan0` (docs/lean.md) |
| `.gguf` | `rocm` | the ROCm build's llama-server on `ROCm0` (ROCmFPX's tree, `ONEBIT_LEAN_ROCM`); ROCmI4 files take its W4A4 path (docs/lean.md) |
| `.gguf` | `zinc` | this build's ZINC (Vulkan, ROCm or CUDA, whichever it was built for; docs/zinc.md) |
| Hugging Face id | `mlx` | lemon-mlx-engine's server, on Apple Silicon (docs/apple.md) |

A build without the private add-on answers a Qwen3.6-35B-A3B directory with "the
Qwen3.6-35B-A3B NPU route is not part of this build; build with
-DONEBIT_NPU_PRIVATE=<npu-kernels checkout>". `--npu-opt KEY=VALUE` passes an option
through to a private route. On a route that keeps its cache between requests,
`usage.prompt_tokens_details.cached_tokens` counts the reused prompt tokens and
`timings.cache` says where they came from. On the NPU a streamed request with
`stream_options.include_usage` gets a last chunk with `usage` and `timings`, as OpenAI's
API sends it.

`chat_template_kwargs.enable_thinking: false` works on every device. The GPU
backends apply the model's own chat template. The NPU route emits what Qwen3's
template does: an empty think block after the assistant prefix. For Qwen3.6-35B-A3B it
also opens `<think>\n` when thinking is on, as that model's template does.

For a `.gguf` the engine starts that server as a private child on a loopback
port and forwards the OpenAI routes to it, streaming included. Replies carry
the served model name. For ZINC, which rejects foreign model ids, requests go
out without `model`. `auto` means Vulkan for GGUF, the fastest measured device
for standard quants (docs/hrx.md), until the Laya router (docs/laya.md) makes
that choice per request.

HRX needs TheRock's HSA runtime: the distro `libhsa` rejects gfx1151's
PM4-emulation probe, and then HRX registers no device. `serve` sets
`IREE_HAL_AMDGPU_LIBHSA_PATH` itself unless you did. It uses `--hrx-libhsa`,
else the build's copy, else the first one under `/opt/rocm-therock`.

The child binaries default to this build's (`-DONEBIT_HRX`, `-DONEBIT_ZINC`),
then `$ONEBIT_LLAMA_SERVER` / `$ONEBIT_ZINC`, then `llama-server` / `zinc` on PATH.

## Multi-token prediction (`--mtp`)

`--mtp <head.gguf>` turns on llama-server's `draft-mtp` speculative decoding: the model's
own MTP head (Unsloth ships it as `MTP/mtp-<model>-*.gguf`) drafts tokens on the same
device, and the model checks them in one batch. `--mtp-max N` caps the draft length;
`--mtp-p-min P` drafts a token only when the head is at least that sure of it.
It works on the llama.cpp devices (`vulkan`, `hrx`, `rocm`).

Measured through `1bit serve` on Strix Halo, Qwen3.8-27B UD-Q4_K_XL with its Q4_0 MTP
head, decode tok/s on three chat prompts (code / prose / short), 2026-09-24:

| Route | Without `--mtp` | With `--mtp` |
|---|---|---|
| `--device vulkan` (upstream pin) | 12.2 / 12.0 / 12.0 | **35.0 / 28.3 / 28.9** |
| `--device rocm` | 11.8 / 11.8 / 12.2 | 38.8 / 20.6 / 16.9 |
| `--device rocm --mtp-max 3` | | 27.1 / 19.2 / 19.9 |

Vulkan with MTP is the pick: 2.4-2.9x on every prompt, same file, same accuracy (a
drafted token is kept only when the model agrees). ROCm edges it on code only.

The full sweep on Vulkan (draft length x `--mtp-p-min`, code / prose / short; 12.2 / 12.3 / 12.3
without MTP):

| Setting | tok/s | Drafts accepted |
|---|---|---|
| `--mtp` (draft length 3, the default) | 35.9 / 28.2 / 28.8 | 94% / 68% / 75% |
| `--mtp-max 4 --mtp-p-min 0.5` | 38.1 / 28.0 / 26.9 | 93% / 71% / 76% |
| **`--mtp-max 6 --mtp-p-min 0.5`** | **42.0** / 27.3 / 25.2 | 84% / 60% / 72% |
| `--mtp-max 8` | 21-25 / 13-20 / 10-20 | 76-88% / 39-65% / 33-78% |

Use the default for chat and `--mtp-max 6 --mtp-p-min 0.5` for code (3.4x): code is
predictable enough to keep long drafts, prose is not. Draft length 8 collapses on every
prompt. Adding n-gram drafting to MTP gains nothing. Small-active MoE models are the
opposite case: on Qwen3-Coder-30B-A3B (3B active, 88 tok/s on Vulkan) every draft model
tried (Qwen3 0.6B / 1.7B / 4B) was slower than no drafting, even at 82-87% acceptance.

## Many requests at once (`--parallel`)

`--parallel N` gives llama-server N slots (`-np N`): requests that arrive together decode
together, one read of the weights per step for all of them (continuous batching, what
vLLM is built on). `--ctx-size` is split across the slots, so size it for all of them.

Total decode tok/s with 1-16 simultaneous 256-token requests, 16 slots, 2026-09-24:

| Requests | Qwen3.8-27B UD-Q4_K_XL, Vulkan | ... ROCm | Qwen3-Coder-30B-A3B Q4_K_M, Vulkan | ... ROCm |
|---|---|---|---|---|
| 1 | 11.8 | 11.6 | 85.7 | 67.6 |
| 2 | 21.7 | 20.1 | 125.8 | 100.2 |
| 4 | 36.9 | 30.2 | 182.4 | 140.3 |
| 8 | **50.9** | 35.6 | **228.2** | 200.4 |
| 16 | 42.9 | **68.0** | 202.9 | **318.9** |

Vulkan peaks at 8 requests and falls back at 16; ROCm keeps scaling to 16, where it
serves 5.8x (27B) and 3.7x (Coder) the best single stream. Use `--device vulkan` for up
to about 8 concurrent users, `--device rocm --parallel 16` beyond that. Two backends
decoding at once, by comparison, add 18% (docs/lean.md).

## Growing with the load (`--adaptive`)

No single setting wins at every load. One request is fastest on Vulkan with `--mtp`; up to
8 are fastest batched on Vulkan; past that ROCm keeps scaling (the tables above). And MTP
and batching do not mix: a server with MTP loaded batches at about two thirds of the
throughput (Qwen3.8-27B, 4 requests: 24.9 tok/s with MTP loaded, 36.9 without), even with
each request's draft length set to 0.

`--adaptive` runs two backends, the model loaded on each: Vulkan with `--adaptive-at`
slots (and `--mtp` if given) and ROCm with 16 slots (never MTP). A request goes to Vulkan
while Vulkan holds fewer than `--adaptive-at` requests, and to ROCm otherwise; choosing
and reserving the slot is one locked step.

Qwen3.8-27B UD-Q4_K_XL, total tok/s by simultaneous requests (2026-09-24):

| Requests | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---|---|---|---|---|---|---|
| **`--adaptive --adaptive-at 8`** | 12.0 | 21.9 | 36.3 | **49.9** | 43.5 | 44.7 | **63.5** |
| `--adaptive --mtp` (at 1) | 20.1 | 15.3 | 24.7 | 32.8 | 50.1 | 59.0 | 40.7 |
| `--device vulkan --parallel 8` | 11.8 | 21.7 | 36.9 | 50.9 | | 42.9 (16) | |
| `--device rocm --parallel 16` | 11.6 | 20.1 | 30.2 | 35.6 | | 68.0 | |

- **Serving many users: `--adaptive --adaptive-at 8`.** It matches Vulkan batching up to
  8 requests and keeps climbing past it (63.5 at 24, where Vulkan alone falls back). At
  12-16 the two backends contend for the GPU and it dips below ROCm alone.
- **One user at a time: `--mtp` without `--adaptive`** (35-42 tok/s). `--adaptive --mtp`
  gives the lone request MTP speed but trails at every larger load.

```sh
1bit serve -m Qwen3.8-27B-UD-Q4_K_XL.gguf --adaptive --adaptive-at 8 --ctx-size 65536
```

It needs both builds (`ONEBIT_VULKAN`, and `ONEBIT_LEAN` + `ONEBIT_LEAN_ROCM`) and memory for
two copies of the model; `--ctx-size` applies to each backend and is split across its
slots. On exit, serve logs how many requests each backend took.

## RAG (`--embed`, `--rerank`)

`--embed MODEL.gguf` serves `/v1/embeddings` and `--rerank MODEL.gguf` serves `/v1/rerank`,
each from its own llama-server on Vulkan beside the chat model (`--embedding`,
`--reranking`, 4 slots, the whole input in one batch). `/v1/models` lists them with their
role. A RAG client embeds its documents, retrieves by cosine similarity, reranks the best
few and asks the chat model with them as context, all against the one server:

`--embedding` and `--reranking` serve the model itself in that role, as Lemonade loads
embedding and reranking models: `1bit serve -m nomic-embed-text-v2-moe.Q8_0.gguf --embedding`.
They run on `vulkan`, `hrx` or `rocm`. On Strix Halo, embeddings work on Vulkan and HRX
(768-dimension vectors). Reranking works on Vulkan, but on HRX jina-reranker-v1-tiny fails:
HRX's JIT can't link the fp32 matmul-with-bias kernel it needs. For now, rerank on Vulkan.

```sh
1bit serve -m Qwen3.8-27B-UD-Q4_K_XL.gguf --mtp mtp-Qwen3.8-27B-Q4_0.gguf \
  --embed Qwen3-Embedding-0.6B-Q8_0.gguf --rerank qwen3-reranker-0.6b-q8_0.gguf
```

Measured end to end on Strix Halo (2026-09-24): the engine's own docs and README as the
corpus (93 passages of about 120 words), four questions with known answers, Qwen3.8-27B
UD-Q4_K_XL with `--mtp` answering, Qwen3-Embedding-0.6B Q8_0 embedding (queries prefixed
with its `Instruct: ... Query: ` line):

| Stage | Time |
|---|---|
| Embed the 93 passages | 3.4-5.6 s |
| Embed a question | 17-32 ms |
| Cosine search | 4-11 ms |
| Rerank 8 passages | 0.4-0.6 s |
| Answer (700-1,560-token prompt) | 4-5.6 s at 27-33 tok/s |

| Retrieval | Right |
|---|---|
| **Top 8 by embedding, all 8 as context** | **3/4** (the fourth answered 11.0 ms/token, the same fact as 91 tok/s) |
| Top 3 by embedding | 1/4 |
| Top 8, reranked to 3 by Qwen3-Reranker-0.6B | 1/4: it scores every passage 0.96-1.0 through llama.cpp (two conversions tried) |
| Top 8, reranked to 3 by bge-reranker-v2-m3 | 0/4: it discriminates, but keeps the wrong passages |

So use embeddings with a wide context and no reranker: retrieval is about 25 ms, and a
few thousand extra prompt tokens cost little at 270-320 tok/s prefill. `--rerank` stays
for larger corpora, where cutting 50 candidates to 8 matters more.

Long retrieved contexts are where `--prefill-device hrx` pays (26% on an 8192-token
request, above).

## Verified (Strix Halo, 2026-09-23)

`tests/serve_e2e.sh` with Qwen3-0.6B: the Q4_K_M GGUF for the GPU devices and the Q4NX model directory for the NPU. The test checks `/health` 200,
`/v1/models`, a chat that answers "Paris." under the served name, and
streaming:

| Device | Result |
|---|---|
| `npu` | PASS (33 SSE chunks): Qwen3-0.6B NPU model directory on the fast lane |
| `vulkan` | PASS (32 SSE chunks) |
| `hrx` | PASS (32 SSE chunks), with no environment set up |
| `zinc` | PASS (6 SSE chunks) |

Qwen3.6-35B-A3B on the NPU is tested by the private add-on (docs/npu.md, "Private routes").
Without it, `serve` on that directory exits with the message above.

ctest runs these as `serve_e2e_<device>` when configured with
`-DONEBIT_SERVE_TEST_GGUF=<gguf>` (and `serve_e2e_mlx` on macOS with
`-DONEBIT_MLX_SERVER`). `smoke_serve` runs everywhere, CI included:
`tests/fake_backend.py` stands in for llama-server. It checks the proxy (`/health`,
`/v1/models`, the reply rename, SSE relay) and that no backend outlives a
SIGKILLed `serve`.
