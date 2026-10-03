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
           [--device auto|npu|hrx|cpu|zinc|ds4|mlx|onnx] [--ctx-size N] [--alias NAME]
           [--llama-server PATH] [--zinc PATH] [--ds4 PATH] [--ssd-streaming] [--hrx-libhsa PATH] [--mlx-server PATH]
           [--mtp HEAD.gguf | --dflash DRAFT.gguf] [--mtp-max N] [--mtp-p-min P] [--mmproj MMPROJ.gguf]
           [--laya | --laya-model DIR] [--route-policy FILE]
           [--moe-slots N|auto] [--moe-subst R] [--moe-prefetch N]
           [--parallel N]
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

"llama-server devices" means `hrx` and `cpu`. These routes go to the llama-server
behind the model, which is how Lemonade's llamacpp backend reaches them (the `onebit` recipe
inherits it). On `npu`, `zinc`, `ds4` and `mlx` they answer 501.

`1bit serve` has no authentication (SECURITY.md), so it refuses two kinds of request a web
page could make with 403: on a loopback `--host`, any whose `Host` header is not a loopback
name (`localhost`, `127.x.x.x`, `[::1]`), and on any host, a POST whose `Origin` is neither a
loopback origin nor the server's own. API clients send no `Origin`. A reverse proxy in front
of a loopback serve must pass the upstream host (nginx's default `proxy_set_header Host
$proxy_host`), not the public one.

## Where the model runs

| Model | `--device` | Runs on |
|---|---|---|
| NPU model directory (`model.q4nx` + `npu/`, docs/npu.md) | `auto`, `npu` | the NPU fast lane, in process |
| Qwen3.6-35B-A3B Q4NX directory (`model_type` `qwen3_5_moe`) | `auto`, `npu` | the private NPU route, in process, in builds with `-DONEBIT_NPU_PRIVATE` (docs/npu.md, "Private routes") |
| `.gguf` | `auto`, `hrx` | the HRX build's llama-server on `HRX0` (docs/hrx.md), `--mtp` included. Since the HRX prompt-matmul routing (llama.cpp fork #55), prompts on HRX run at pp512 335 tok/s on Qwen3.8-27B UD-Q4_K_XL (was 97-99), and a 14K-token prompt through `serve` at 265 tok/s (was 91). `auto` exceptions: below |
| `.gguf` | `cpu` | the same llama-server with no GPU layers; `auto` on Windows (no GPU route yet, docs/windows.md) and in a build without HRX |
| `.gguf` | `zinc` | this build's ZINC (Vulkan, ROCm or CUDA, whichever it was built for; docs/zinc.md) |
| DwarfStar `.gguf` (DeepSeek V4 Flash, GLM 5.x, Qwen3.8-Flash-Next in its own layouts) | `ds4` | this build's DwarfStar `ds4-server` (ROCm, CUDA or Metal; `--ssd-streaming` streams routed experts; docs/dwarfstar.md) |
| Hugging Face id | `mlx` | lemon-mlx-engine's server, on Apple Silicon (docs/apple.md) |

What `--device auto` does with a `.gguf` that HRX has no route for. Until RFC #213 stage 3 these
fell back to Vulkan (engine #271); with no Vulkan or ROCm build left:

| Case | `auto` |
|---|---|
| an architecture our llama.cpp does not build: anything outside `gguf_architectures.hrx` in registry/architectures.json, which `tools/registry_build.py` reads from the pinned fork and the build compiles in (among them Qwen3.8-Flash-Next's `qwen4exp`; Zyphra Zamba, Zamba2, BlackMamba; Spark2.5, BailingMoeV3, HRM text, MuseGlimmer, Kimi K3, Maple, GraniteSwitch, Granite SWA, HY v4, MiniMax, Dots3 Note, PocketTTS, Qwen3-TTS) | refused on hrx and cpu, pointing at Lemonade's llamacpp backend; Zyphra and Flash-Next are to be ported to HRX |
| `--moe-slots` | refused: not in this build ([below](#moe-models-larger-than-memory---moe-slots)) |
| `--parallel N` > 1 on a gated delta-net model (Qwen3.5, Qwen3.8, Qwen3-Next) | HRX0 with one slot, and a warning on stderr: HRX runs one such sequence at a time (no multi-sequence delta-net yet). `--device cpu` keeps `--parallel N` |
| `--mmproj` | HRX0. **Unverified:** vision on HRX is an open RFC #213 gate, to be checked on ZAYA1-VL; if it fails there, `auto` goes back to the CPU for `--mmproj` |
| `--mtp` | HRX0 (it drafts slower there than Vulkan did: its verify batches of 2-4 tokens are the open item, docs/hrx.md) |
| an H32 file (a Hadamard-rotated Q4_0 stamped `onebit.hadamard_q4_0`) | refused on every device: the format is dropped; use the model's UD-Q4_K_XL or another standard GGUF. PrismML's own rotation (`prism.hadamard`, Ternary Bonsai) is a different thing and still runs on HRX ([hrx.md](hrx.md#ternary-bonsai-prismmls-hadamard-folded-ggufs)) |

### Removed devices and flags

The engine builds no Vulkan or ROCm llama.cpp (RFC #213 stage 3; owner direction
2026-10-01: HRX + NPU, zero Vulkan, zero ROCm). Lemonade ships its own llamacpp backends for those
([lemonade.md](lemonade.md#vulkan-and-rocm-are-lemonades)). `serve` exits at once, with the reason,
when asked for any of these:

| Asked for | What it was |
|---|---|
| `--device vulkan` | upstream llama.cpp's release on `Vulkan0` |
| `--device rocm` | the ROCmFPX tree's ROCm build on `ROCm0` |
| `--lean` | ROCmFPX's ROCmFP4 / ROCmI4 formats |
| `--prefill-device hrx`, `--prefill-min-tokens` | Vulkan decode with the prompt prefilled on HRX0 over one shared KV cache |
| `--adaptive`, `--adaptive-at` | Vulkan for the first requests, ROCm batching the overflow |
| `--long-model`, `--long-from` | long conversations to a Hadamard-rotated Q4_0 on the ROCm W4A4 route |

The DFlash2 one-server route (a Hadamard-rotated file with W4A4 prompts and DFlash2 decode on ROCm)
went with them, and so did its recipes (`rotated-moe-ub1024`, `rocm-dflash-p-min-0.4`), and so did
the H32 format it served (`tools/hadamard_q4_0.py` is removed; serve refuses those files). `--dflash`
itself stays, on HRX and the CPU.

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
out without `model`. `auto` means HRX for GGUF: it met RFC #213's decode gates (docs/hrx.md), and
it is the engine's only GPU route. With `--laya` (or
`--laya-model DIR`), Laya classifies each conversation and the route policy picks the device:
the built-in policy sends every class to HRX, long documents included, since HRX prefills
Qwen3.8-27B at 335 tok/s pp512 (docs/laya.md). The first turn of
a conversation pays about 0.5 s for the decision.

HRX needs TheRock's HSA runtime: the distro `libhsa` rejects gfx1151's
PM4-emulation probe, and then HRX registers no device. `serve` sets
`IREE_HAL_AMDGPU_LIBHSA_PATH` itself unless you did. It uses `--hrx-libhsa`,
else the build's copy, else the first one under `/opt/rocm-therock`.

The child binaries default to this build's (`-DONEBIT_HRX`, `-DONEBIT_ZINC`, `-DONEBIT_DS4`),
then `$ONEBIT_LLAMA_SERVER` / `$ONEBIT_ZINC` / `$ONEBIT_DS4`, then `llama-server` / `zinc` / `ds4-server` on PATH.

## Images (`--mmproj`)

`--mmproj <mmproj.gguf>` hands llama-server a vision projector on the llama.cpp routes
(hrx, cpu; `auto` picks the CPU until HRX vision is checked). Chat messages may then carry `image_url` parts (a `data:` URL or a file
URL), which llama.cpp's mtmd encodes and places in the prompt. The mmproj comes from
`convert_hf_to_gguf.py --mmproj` on the same checkpoint as the model. With it, llama-server runs
with `-b 4096 -ub 4096`: an image is decoded as one ubatch, which models that attend to an image
bidirectionally (ZAYA1-VL, Gemma 3) need. Zyphra's Zamba2-VL and ZAYA1-VL-8B were checked on the
removed Vulkan build ([vulkan.md at 0baf286](https://github.com/1bit-MONSTER/engine/blob/0baf286/docs/vulkan.md)). The NPU, ZINC, DwarfStar, MLX and ONNX routes take no `--mmproj`.

## Multi-token prediction (`--mtp`)

`--mtp <head.gguf>` turns on llama-server's `draft-mtp` speculative decoding: the model's
own MTP head (Unsloth ships it as `MTP/mtp-<model>-*.gguf`) drafts tokens on the same
device, and the model checks them in one batch. `--mtp-max N` caps the draft length;
`--mtp-p-min P` drafts a token only when the head is at least that sure of it.
It works on the llama.cpp devices (`hrx`, `cpu`), and `--device auto` keeps it on HRX0.
A drafted token is kept only when the model agrees, so the output is the model's own. HRX's MTP
numbers are in [hrx.md](hrx.md) ("Not yet": Qwen3.8-27B with its Q4_0 MTP head).

The measurements below were taken on the removed Vulkan build (2026-09-24) and are kept for the
settings they found; re-measure on HRX before relying on the speeds. The sweep (draft length x
`--mtp-p-min`, code / prose / short; 12.2 / 12.3 / 12.3 without MTP):

| Setting | tok/s | Drafts accepted |
|---|---|---|
| `--mtp` (draft length 3, the default) | 35.9 / 28.2 / 28.8 | 94% / 68% / 75% |
| `--mtp-max 4 --mtp-p-min 0.5` | 38.1 / 28.0 / 26.9 | 93% / 71% / 76% |
| **`--mtp-max 6 --mtp-p-min 0.5`** | **42.0** / 27.3 / 25.2 | 84% / 60% / 72% |
| `--mtp-max 8` | 21-25 / 13-20 / 10-20 | 76-88% / 39-65% / 33-78% |

Use the default for chat and `--mtp-max 6 --mtp-p-min 0.5` for code (3.4x): code is
predictable enough to keep long drafts, prose is not. Draft length 8 collapses on every
prompt. Adding n-gram drafting to MTP gains nothing. Small-active MoE models are the
opposite case: on Qwen3-Coder-30B-A3B (3B active, 88 tok/s then) every draft model
tried (Qwen3 0.6B / 1.7B / 4B) was slower than no drafting, even at 82-87% acceptance.

### Qwen3.8-Flash-Next: drafting from part of the vocabulary

Flash-Next's MTP head computes logits over all 248,320 tokens for every drafted token. That
is the model's 644 MiB Q8_0 output matrix, read three times per decode step at draft length 3.
A draft only proposes; the model checks every token, so the head can be cut down to the
tokens it is likely to propose. `tools/mtp_draft_vocab.py` writes such a head from Unsloth's
*shared* head file and the model's own output rows. Our llama.cpp fork (`qwen4exp`,
`src/models/qwen4exp-draft-vocab.cpp` on the removed Vulkan branch) ran it; it comes back with the
HRX port of Flash-Next:

```sh
python3 tools/mtp_draft_vocab.py \
  --mtp MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --target UD-Q4_K_XL/ \
  --ids config/draft-vocab/qwen38-en-code-65536.txt --out mtp-Qwen3.8-Flash-Next-dv65k.gguf
```

`config/draft-vocab/qwen38-en-code-65536.txt` is dime-online's 65,536-id list, picked from
English prose and source code
([qwen3.8-Flash-DGX-UltraFast](https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast),
Apache-2.0, see NOTICE). It makes the head 170 MiB.

Measured with the fork's Vulkan `llama-server` on Strix Halo, 2026-10-01. Settings:
- UD-Q4_K_XL, greedy, draft length 3, `-lm mmap -lzm on`
- decode tok/s, median of rounds 2-3 of 3

| Prompt | Full head | 65,536-id head |
|---|---|---|
| code, 256 tokens | 47.8 | **52.1** |
| prose, 256 tokens | 38.6 | **42.2** |
| code, 768 tokens | 45.8 | **48.3** |
| French, 26 tokens | **46.4** | 39.6 |

- **English and code:** 8-9% faster, the same acceptance, and about 6 ms less per decode step (76 -> 70 ms).
- **Other languages:** text outside the list loses, as the French row shows. Use the full head for such text.
- **Output text:** decoding with a draft is not token-identical to decoding without one, even with the full head. Batched verification flips near-ties (one flip each in prose and long code here). The cut head gives the full head's text in rounds 2-3, with two other flips in round 1.
- **Q4_K_M:** Unsloth's shared Q4_K_M head was no faster and flipped more often.
- **`-lm mmap`:** Flash-Next needs it on this llama.cpp. Without it, the 27.5 GiB per-layer embedding table is copied into RAM next to the ~82 GB the GPU holds, which does not fit in 128 GB.

## DFlash draft models (`--dflash`)

`--dflash <draft.gguf>` uses a DFlash block-diffusion draft model instead of the MTP head: the
draft emits a whole block of tokens in one pass, reading the model's hidden states, and the
model checks the block in one batch. It turns on llama-server's `draft-dflash` (DFlash2's
candidate selector included). A drafter is trained for one target model; z-lab publishes them
on Hugging Face (for example
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2), 2B, Apache-2.0).
`--dflash` and `--mtp` are exclusive. `--mtp-max` and `--mtp-p-min` set the draft length and
threshold for either; without `--mtp-max`, `--dflash` drafts full blocks (llama-server's own
default of 3 throws most of the gain away).

Convert the drafter with upstream llama.cpp's converter. It needs only the target's
`config.json` and tokenizer files, not its weights:

```sh
hf download Qwen/Qwen3.8-27B --include "*.json" "*.jinja" "merges.txt" "vocab.json" --local-dir Qwen3.8-27B-hf
hf download z-lab/Qwen3.8-27B-DFlash2 --local-dir DFlash2
python convert_hf_to_gguf.py DFlash2 --target-model-dir Qwen3.8-27B-hf --outtype bf16 --outfile dflash2-bf16.gguf
llama-quantize dflash2-bf16.gguf dflash2-q8_0.gguf Q8_0
1bit serve -m Qwen3.8-27B-Q4_0.gguf --dflash dflash2-q8_0.gguf
```

Measured through `1bit serve` on the removed Vulkan build on Strix Halo (not yet re-measured on
HRX), Qwen3.8-27B Q4_0 (Unsloth), the
same three chat prompts as above (code / prose / short, decode tok/s), runs back to back on a
shared box, 2026-09-28:

| Drafter | tok/s | Prompt t/s (1,838 tokens) |
|---|---|---|
| none | 11.1 / 11.8 / 12.4 | 330 |
| `--mtp` (Q4_0 MTP head, draft length 3) | 31.7 / 25.8 / **28.2** | 267 |
| `--dflash` (DFlash2, Q8_0, full blocks) | **45.7** / **28.3** / 17.8 | 255 |

DFlash2 is the pick for code (4.1x, +44% over MTP) and ties or beats MTP on prose. MTP wins
short answers (the translation prompt stops after ~16 tokens, too few to fill blocks). The Q8_0
drafter is faster than BF16 (42.5 vs 38.9 on code, direct llama-server). A drafted token is
kept only when the model agrees, so the output is the model's own.

Without `--mtp-max`, `--dflash` drafts the drafter's block minus one (its `dflash.block_size`; 16
when the file does not say). Our HRX llama.cpp's default draft p-min is already 0, which keeps
DFlash blocks whole; `--mtp-p-min` sets another. (The DFlash2 one-server route on the lean ROCm
build is gone, above.)

## Recipes (`--recipes`, `--no-recipes`)

Tuned llama-server settings that depend on the model and route, such as a DFlash drafter's
p-min, are recipes in `config/recipes.json`. Each one
carries the measurement behind it. Serve prints each recipe it applies. Anything serve sets
itself wins over a recipe. `--recipes FILE` replaces the built-in set and `--no-recipes` turns
recipes off ([recipes.md](recipes.md)). To measure a setting before making it a recipe, use
[`tools/bench.py`](bench.md).

## MoE models larger than memory (`--moe-slots`)

**Not in this build.** `--moe-slots` is kept (the option, `--moe-subst`, `--moe-prefetch` and
serve's code for them), but the expert streamer lives in the removed Vulkan llama.cpp pin. Until it
moves to HRX together with Qwen3.8-Flash-Next, `serve` refuses `--moe-slots` with "not available in
this build". What follows describes it as it ran on Vulkan0 (docs/moe-streaming.md).

`--moe-slots N` keeps a MoE model's routed experts in the file and
holds N of them, across all layers, in GPU memory. Decode reads the missing ones from the drive
as the router picks them. The rest of the model loads on the GPU as usual.

```sh
1bit serve -m Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf --moe-slots 4608 --ctx-size 8192
```

Qwen3-Coder-30B has 6,144 experts; at 4,608 slots it decodes 39 tok/s warm, at 1,536 about
6-7 (docs/moe-streaming.md, "Streaming in the inference path"). A model that fits in memory is
faster without it (79-91 tok/s resident).

`--moe-subst R` trades a little accuracy for fewer reads: when the router picks an expert that
is not in memory, a resident one among its next choices takes its place if it scores at least
R times as high. `0.5` cut Coder-30B's drive reads by 20-27% and sped decode up 10-35% at 1,536
and 3,072 slots, for a KL divergence of 0.008-0.012 against the exact model (docs/moe-streaming.md,
"Routing that prefers resident experts"). Without it, routing is exact.

Qwen3.8-Flash-Next UD-Q4_K_XL (111 GB) runs this way from 18-32 GiB of GPU memory: 3.5-6 tok/s
exact at 4,608-9,216 slots. On that model use `--moe-subst 0.9` or none: 0.5 costs a KL
divergence of 0.07 there.

`--moe-slots auto` sizes the cache from free memory: `MemAvailable`, less the other tensors the
GPU holds and a reserve (8 GiB or 15%), divided by the model's mean expert size. On a box with
40 GiB free it gave Flash-Next 9,331 of 24,576 experts (27.2 GiB) beside 5.1 GiB of other
tensors, and `serve` held 145 MiB of anonymous memory. Other processes can take that memory
back later; on a shared box, give a number instead. The gate-ahead prefetch is off unless
`--moe-prefetch N` asks for it (it doubled Flash-Next's decode time).

Prompts are slow when streaming: a batch larger than 8 tokens runs the experts on the CPU
from the mapped file (Flash-Next: 0.4-0.8 tok/s of prompt). Keep prompts short, or see
docs/moe-streaming.md, Next.

## Many requests at once (`--parallel`)

`--parallel N` gives llama-server N slots (`-np N`): requests that arrive together decode
together, one read of the weights per step for all of them (continuous batching, what
vLLM is built on). `--ctx-size` is split across the slots, so size it for all of them.

On HRX, `--parallel N` shares one KV cache across the slots (`-kvu`) so attention stays on the
GPU, and turns off AMD's Qwen attention path, which ignores the other sequences. HRX does not yet
run a gated delta-net model (Qwen3.5, Qwen3.8, Qwen3-Next) with more than one sequence: `serve`
refuses `--parallel` > 1 there, and without `--parallel` gives those models one slot.

The earlier measurements here (Vulkan and ROCm batching, and `--adaptive`, which ran both) went
with those builds (RFC #213 stage 3); see this page at
[0baf286](https://github.com/1bit-MONSTER/engine/blob/0baf286/docs/serve.md#many-requests-at-once---parallel).
`--adaptive` and `--long-model` are refused now ([above](#removed-devices-and-flags)).

## RAG (`--embed`, `--rerank`)

`--embed MODEL.gguf` serves `/v1/embeddings` and `--rerank MODEL.gguf` serves `/v1/rerank`,
each from its own llama-server beside the chat model (`--embedding`, `--reranking`, the whole input
in one batch). In a build with HRX that server runs on `HRX0` with one slot and inputs of up to 2048
tokens (HRX runs one sequence per batch); a build without HRX runs it on the CPU, 4 slots and 8192 tokens. `/v1/models` lists them with their
role. A RAG client embeds its documents, retrieves by cosine similarity, reranks the best
few and asks the chat model with them as context, all against the one server:

`--embedding` and `--reranking` serve the model itself in that role, as Lemonade loads
embedding and reranking models: `1bit serve -m nomic-embed-text-v2-moe.Q8_0.gguf --embedding`.
They run on `hrx` or `cpu` (`auto`: HRX where the build has it). On Strix Halo (2026-10-02, two
repeats each): Qwen3-Embedding-0.6B Q8_0 embeddings repeat bit for bit on HRX; bge-reranker-v2-m3
Q8_0 scores repeat exactly (8.609 / -6.756 / -0.401 / -11.020). jina-reranker-v1-tiny still
fails on HRX: its JIT can't link the fp32 matmul-with-bias kernel it needs.

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

## Verified (Strix Halo, 2026-09-23)

`tests/serve_e2e.sh` with Qwen3-0.6B: the Q4_K_M GGUF for the GPU devices and the Q4NX model directory for the NPU. The test checks `/health` 200,
`/v1/models`, a chat that answers "Paris." under the served name, and
streaming:

| Device | Result |
|---|---|
| `npu` | PASS (33 SSE chunks): Qwen3-0.6B NPU model directory on the fast lane |
| `hrx` | PASS (32 SSE chunks), with no environment set up |
| `zinc` | PASS (6 SSE chunks) |

Qwen3.6-35B-A3B on the NPU is tested by the private add-on (docs/npu.md, "Private routes").
Without it, `serve` on that directory exits with the message above.

(The `vulkan` row, PASS with 32 SSE chunks, went with the Vulkan build.) With `-DONEBIT_HRX=ON`,
ctest runs `serve_e2e_hrx` and `serve_e2e_cpu`, and in general `serve_e2e_<device>`, when configured with
`-DONEBIT_SERVE_TEST_GGUF=<gguf>` (and `serve_e2e_mlx` on macOS with
`-DONEBIT_MLX_SERVER`). `smoke_serve` runs everywhere, CI included:
`tests/fake_backend.py` stands in for llama-server. It checks the proxy (`/health`,
`/v1/models`, the reply rename, SSE relay) and that no backend outlives a
SIGKILLed `serve`.
