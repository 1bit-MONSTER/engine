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
# Laya

Laya picks where each request runs: it sorts each conversation into code, prose, short answer
or long document (95.5% right), and a measured policy picks the device. The test set is 200
labelled requests. This is step 4 of the port (docs/PORTING.md).
[Laya](https://github.com/NandhaKishorM/laya) (Apache-2.0) is a
non-autoregressive decision model: a ModernBERT-style encoder plus an RLCD
decision head. It answers typed questions (`choice`, `score`, `noul`) about a
request in one forward pass, with calibrated confidence. It has three
checkpoints and a router that picks between them.

## The pin

| | Pin | Kept current by |
|---|---|---|
| Source | `third_party/laya` = `NandhaKishorM/laya` main **`1e28ac20`** | `bump-laya.yml` |
| Checkpoints | `config/laya.json` = Hugging Face `convaiinnovations/laya` at revision **`aa8c91ca`**: root (842 MB), `multilingual/` (644 MB, 34 MB tokenizer), `typed-decisions/` (842 MB) | `bump-laya.yml` (moves the source and the revision together) |

```sh
scripts/fetch-laya.sh            # into ~/.local/share/1bit/laya, where serve --laya looks
scripts/fetch-laya.sh ~/models/laya-pinned
```

`fetch-laya.sh` downloads exactly the pinned revision and checks every file
against the Hub's list: sha256 for the LFS weights, the git blob id for the rest.
Files already present and correct are not fetched again. Images and eval plots
are skipped.

Verified 2026-09-23 on Strix Halo: 20 files, 2.3 GB, all hashes match.

## The C++ scorer

Ported from 1bit-MONSTER `src/laya_scorer.cpp` on
`backup/laya-and-results-2026-09-22`, without its history. Zero Python at
runtime: weights from `model.safetensors` (`laya/safetensors.{h,cpp}`, F16/F32
-> f32), tokens from the Hugging Face `tokenizer.json` through the engine's
`npu::Tokenizer` (`laya/scorer.{h,cpp}`). One forward pass answers typed
questions with the same calibrated confidence as the Python reference.

| | Where |
|---|---|
| Scorer | `laya/scorer.{h,cpp}` |
| Safetensors reader | `laya/safetensors.{h,cpp}` |
| Router | `laya/route.{h,cpp}` |
| Gate harness | `tests/laya_gate.cpp` |
| Routing e2e | `tests/laya_route_e2e.sh` + `tests/fake_backend_route.py` |

`npu::Tokenizer` gained `token_id()` and the Rust-tokenizers ByteLevel regex
(the `use_regex` pattern for `{"type":"ByteLevel"}` with no explicit Split, as
ModernBERT serializes it); digits stay together so multi-digit tokens merge.

## Gate against the Python reference

The scorer is gated against the Python `laya` package on the two
ModernBERT-large checkpoints (root and `typed-decisions/`; `multilingual/` is
mmBERT-base and stays a separate follow-up). `tests/laya_gate.cpp` runs the
questions + state recorded in a `golden.json` and compares raw logits and
act logits, the argmax decision, and temperature-1 softmax probabilities.

| Checkpoint | max \|logit\| diff | act_logit rel | argmax | max \|prob\| diff |
|---|---|---|---|---|
| root (`laya`) | 8.6e-6 | 1.1e-6 | 0 | 1.1e-6 |
| `typed-decisions/` | 4.8e-6 | 6.1e-7 | 0 | 5.7e-7 |

Tolerances: logits <= 1e-3, act logits <= 1e-3 relative, identical argmax,
probabilities <= 1e-4. Verified 2026-09-25 on Strix Halo.

## Routing

Asked directly which device should run a request, Laya picks the same one whatever the
request: ZINC for 8 of 8 varied requests among `vulkan,hrx,zinc` (measured 2026-09-28, pinned
root checkpoint). It knows nothing about this hardware. So since RFC
[#186](https://github.com/orgs/1bit-MONSTER/discussions/186) it answers a question about the
request instead, and a measured policy picks the device.

1. **Classify** (`laya/route.h`, `classify_request`). A request of 1,024 characters or more is
   `long_doc` (a pasted document, log, table or file) without a model call. Anything shorter
   gets one Laya choice among `code`, `prose` and `short`, with Laya's calibrated confidence.
2. **Policy** (`config/route-policy.json`, compiled in; `--route-policy FILE` replaces it). Each
   class names a device and why. A class whose confidence is below `min_confidence` (0.16) takes
   the policy's `default`. A device the model cannot run on falls back to `default`.
3. **Once per conversation.** `serve` caches the class by the conversation's opening (the messages
   up to the first user message), so later turns skip the decision.

```sh
scripts/fetch-laya.sh
1bit serve -m model.gguf --device auto --laya            # or --laya-model DIR
1bit route --laya-model DIR --classify --state "Write a Rust function that sums a slice."
# code 0.44 vulkan
```

Every routed reply carries `X-1bit-Route: <class> <confidence> <device>`, and `serve` logs each
new decision, so real traffic shows how requests split. A directory holding the pinned repo's
`typed-decisions/` checkpoint uses that one: it classifies best.

**Classifier accuracy** (`tests/laya_classify_eval` on `tests/laya_route_cases.json`, 200
hand-labelled requests, 50 per class; ctest `laya_classify` fails below 90%):

| Checkpoint, question | Accuracy | code | prose | short | long_doc | ms per decision |
|---|---|---|---|---|---|---|
| **`typed-decisions/`, size gate + three-way question** | **95.5%** | 100% | 82% | 100% | 100% | 588 |
| root, size gate + three-way question | 92.0% | 94% | 74% | 100% | 100% | 401 |
| `typed-decisions/`, four-way question, no size gate | 80.5% | 100% | 62% | 90% | 70% | 1,739 |
| root, four-way question, no size gate | 64.0% | 98% | 30% | 96% | 32% | 1,379 |

Prose is the class Laya misses most, reading it as short. Confidence tracks accuracy: the least
confident quarter is right 84% of the time, the rest 98-100%.

**Why every class goes to Vulkan today.** The policy may only change a row for a measured net
gain, and on Strix Halo none exists yet. Vulkan decodes fastest on every measured `.gguf`
([hrx.md](hrx.md), [serve.md](serve.md)). The only per-class difference measured is the drafter:
on Qwen3.8-27B, short replies decode faster with MTP (28.2 tok/s) than with DFlash2 (17.8), code
the other way round (45.7 against 31.7). But llama-server fixes the drafter per server:
`speculative.type`, `speculative.n_max` and `speculative.p_min` in a request are ignored
(`draft_n` unchanged, 2026-09-28, on the engine's pin and upstream master), so the short-reply
gain would need a second server holding a second copy of the model (about 16 GB for 27B) to save
about 0.3 s on a 16-token reply.

**Cost** (`1bit serve --device auto --laya`, Qwen3.8-27B Q4_0 + DFlash2, 2026-09-28):

| | 1-token reply, first turn | later turn | decode code / prose / short, tok/s |
|---|---|---|---|
| without `--laya` | 498-518 ms | 713-849 ms | 50.5-50.7 / 26.7-26.9 / 19.0-19.2 |
| `--laya` | 1,008-1,048 ms | 703-767 ms | 50.6 / 26.7 / 18.9-19.1 |

The first turn of a conversation pays about 0.5 s; later turns and decode are unchanged. That is
why routing stays opt-in: `--device auto` without `--laya` still means Vulkan, as before.

`tests/laya_route_e2e.sh` proves each class reaches the device the policy names
(`tests/route-policy-e2e.json` sends code, prose and short to three fake devices), the header,
the reuse on a later turn, and the error when no checkpoint is installed. A Q4NX model directory
runs on the NPU only. `multilingual/` (mmBERT-base) and Apple/MLX routing are out of scope.

## Cost per decision

`1bit route` prints the scorer's load and decision times on stderr. On Strix Halo
(`laya-pinned` root checkpoint, candidates `vulkan,hrx,zinc`), loading takes 2.7 s
once, and each decision takes **0.38 s** (it was 8.75 s). Only the encoder's layout
changed. Its projections and GeGLU go through the threaded `linear()`, the attention runs
one thread per (sequence, head), and `linear()` reads each weight row once per four input
rows. Every sum keeps its order, so the gate output is byte-identical to the
single-threaded scorer. Those figures are for the old device question on a short request.
The request-class decision takes 0.4-0.6 s (the table above), and `serve` pays it once per
conversation.

## The GGUF scorer: ggmlc on HRX

[ggmlc](https://github.com/monatis/ggmlc) compiles Laya's checkpoints to GGUFs that its own
`laya` runtime runs on GGML (they are not llama.cpp GGUFs): English
([mys/laya-GGUF](https://huggingface.co/mys/laya-GGUF), ModernBERT-large, 421M parameters,
512-token context), multilingual ([mys/laya-multilingual-GGUF](https://huggingface.co/mys/laya-multilingual-GGUF),
100+ languages) and typed-decisions
([mys/laya-typed-decisions-GGUF](https://huggingface.co/mys/laya-typed-decisions-GGUF), for
`choice` / `score` / `noul`). The engine pins our fork of ggmlc as `third_party/ggmlc`
(`1bit/main`: upstream plus an HRX build) and the typed-decisions Q8_0 in `config/laya.json`;
`scripts/fetch-laya.sh` fetches it into `<laya dir>/gguf/` and checks its sha256.

The scorer runs on HRX0, AMD's backend: ggmlc's `laya` is built against `third_party/llama.cpp`'s
ggml, whose ggml-hrx carries the LOOM kernels a ModernBERT encoder needs and llama models did not
(LayerNorm, strided and broadcast arithmetic, clamp, F32 to F16 copies, attention over
one-block-per-head layouts, small and odd-K matmuls; 1bit-MONSTER/llama.cpp#31).

```sh
cmake -B build -DONEBIT_LAYA_GGML=ON ...   # builds ggmlc's laya on HRX with TheRock's amdclang
scripts/fetch-laya.sh
1bit serve -m MODEL.gguf --laya            # uses the GGUF scorer when both are there
```

With the binary and the GGUF present, `serve --laya` starts `laya daemon <gguf> --family
typed-decisions --device hrx` (with the HSA runtime HRX needs, as for `--device hrx`) and asks it
serve's question over its stdin and stdout (not its `serve`, which listens on every interface
with no key). Without them it uses the C++ scorer on the safetensors. `--laya-model FILE.gguf`
picks another GGUF, `ONEBIT_LAYA_GGML` another binary, `ONEBIT_LAYA_DEVICE` another device (`cpu`).

**Measured** (Strix Halo, the 200 labelled requests of `tests/laya_route_cases.json`, 150 of
them asked of the model, the rest long_doc by size):

| Scorer | Class accuracy | Decision, p50 |
|---|---|---|
| C++ scorer, typed-decisions safetensors, CPU (through `serve`) | 95.5% | 538 ms |
| **ggmlc `laya` on HRX0, typed-decisions Q8_0 GGUF** | **95.5%** | **25 ms** (15 ms at a steady 64 tokens, 25 at 128) |
| ggmlc `laya` on Vulkan, same GGUF (upstream ggmlc, for comparison; not built by the engine) | 95.5% | 22 ms (12 ms at a steady length) |

The p50 includes the first request of each length bucket, which compiles and records that
shape once (about 150 ms). At a steady length HRX is within 3 ms of Vulkan at 64 tokens and level
with it at 128. Three things got it there from 44 ms: ggmlc gives its graphs a uid, so HRX's
graph-program cache and graph replay hit instead of re-importing and re-recording every call
(25 -> 15 ms); a fused kernel for the rotate-half RoPE that ggmlc lowers to eight nodes, and one
for GEGLU (1,170 launches a decision down to about 780); and attention that computes each score
once. Next is the NPU.
Answers match Vulkan's probabilities within about 1e-3. Asked directly on Vulkan: typed-decisions
F16 95.5% at 36 ms; the English GGUF 93.0% (Q8_0) and 91.5% (F16), so the typed-decisions
checkpoint stays the router's. `tests/laya_gguf_route.sh` covers the daemon wiring without a GPU.

**Other languages.** The 150 model-decided cases machine-translated into Spanish, German,
Japanese, Chinese, Hindi and Russian (Qwen3-Coder-30B-A3B, greedy; 682 kept after dropping replies
that answered instead of translating), on HRX0:

| Scorer | es | de | ja | zh | hi | ru | All |
|---|---|---|---|---|---|---|---|
| **typed-decisions for every language** | **92.5%** | **90.7%** | **89.0%** | **86.4%** | 60.0% | **83.7%** | **83.7%** |
| typed-decisions for English, multilingual GGUF otherwise (ggmlc's language routing) | 73.3% | 71.3% | 65.3% | 83.1% | 60.0% | 68.1% | 69.1% |

The multilingual checkpoint answers this question worse in every language, so the router keeps
typed-decisions for all of them; Hindi is the weak spot either way.

ggmlc declares the MIT licence (`pyproject.toml`, README) but its repository carries no LICENSE
file yet; the GGUFs are Apache-2.0 like the checkpoints.
