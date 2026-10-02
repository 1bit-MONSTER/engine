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
| Source | `third_party/laya` = `NandhaKishorM/laya` main **`6d942c92`** | `bump-laya.yml` |
| Checkpoints | `config/laya.json` = Hugging Face `convaiinnovations/laya` at revision **`55cf4c4e`**: root (842 MB), `multilingual/` (644 MB, 34 MB tokenizer), `typed-decisions/` (842 MB) | `bump-laya.yml` (moves the source and the revision together) |

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
why routing stays opt-in: `--device auto` without `--laya` means one device, HRX.

**The policy since the engine went HRX only (2026-10-01).** Every class goes to HRX. Long
documents went to the lean ROCm build while HRX prefilled Qwen3.8-27B UD-Q4_K_XL at 97-99 tok/s.
Since the HRX prompt-matmul routing (llama.cpp fork #55) it prefills 335 tok/s at 512 tokens and
310 at 2,048 (llama-bench), and a 14,435-token prompt through `1bit serve --device hrx` runs at
265 tok/s. The table above was measured on the earlier, all-Vulkan policy.

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

## The encoder on the NPU (private add-on)

`laya/encoder.h` lets an add-on run the C++ scorer's 28 encoder layers somewhere else: it
registers an encoder by name, and `Scorer::set_encoder` hands it each sequence's embedded rows
(the tokenizer, embeddings and classifier head stay on the CPU). The engine registers none. The
private NPU add-on (docs/npu.md, "Private routes"; its kernels live in the closed-source
`npu-kernels` repository) registers `npu`, and

```sh
cmake -B build -DONEBIT_NPU=ON -DONEBIT_NPU_PRIVATE=<npu-kernels checkout> ...
ONEBIT_LAYA_DEVICE=npu 1bit serve -m MODEL.gguf --laya
```

runs the encoder on the NPU instead of the GGUF daemon. It shares the NPU with other users
through the engine's device lock: while another process holds it (an NPU `1bit serve`), the
encoder declines and that decision's layers run on the CPU. Without the add-on,
`ONEBIT_LAYA_DEVICE=npu` exits saying how to build it (checked by `tests/laya_gguf_route.sh`).

**Measured** (Strix Halo, `tests/laya_route_cases.json`): 194/200 = 97.0% class accuracy (the
CPU scorer: 95.5%), about 105 ms a warm decision through `serve` (encoder ~50 ms, the CPU head
~41 ms), about 500 ms when the NPU is busy and the CPU runs the layers. Slower than the GGUF
scorer on HRX today; it keeps the GPU free for the model being served.

## Proof on the HRX engine (2026-10-02)

The question: does Laya work in the engine as it ships now, with HRX as its only GPU route? The
answer is yes, with the measurements below. Setup: engine `main` at `6f324e4` (llama.cpp fork pin `d60cc4f`), built
with `-DONEBIT_HRX=ON -DONEBIT_LAYA_GGML=ON`, on Strix Halo in balanced power mode (85 W).
The checkpoints are the pinned ones from `scripts/fetch-laya.sh`. The CPU runs went through
`~/lb/thermal-run.sh` (4 cores, nice 19, paused at 78 C), so the CPU decision times are slower
than on an idle box.

**Gates**

| Check | Result |
|---|---|
| `laya_gate`, root checkpoint, against the Python reference | PASS: logits 2.6e-6, act logits 2.1e-7 rel, argmax 0, probabilities 3.0e-7 |
| `laya_gate`, `typed-decisions/` | PASS: logits 3.1e-6, act logits 2.6e-7 rel, argmax 0, probabilities 6.2e-7 |
| `laya_encoder_test` | PASS (7/7) |
| `laya_classify_eval`, C++ scorer, CPU | **95.5%** (191/200), 687 ms a decision |
| The same 200 cases through the GGUF scorer on HRX0 (serve's daemon and question) | **95.5%** (191/200), p50 23.7 ms, 17.6 ms warm; daemon ready in 627 ms |
| `tests/laya_route_e2e.sh` | PASS (11/11) after this change; before it, 2 checks failed (below) |
| `tests/laya_gguf_route.sh` | PASS (9/9, 1 skipped) after this change; before it, 1 check failed (below) |
| Determinism: 12 requests, each asked 5 times | identical class and confidence 12/12 on the CPU scorer, 12/12 on HRX |

Both scorers give the same confusion matrix:

| want \ got | code | prose | short | long_doc | right |
|---|---|---|---|---|---|
| code | 50 | 0 | 0 | 0 | 100% |
| prose | 0 | 41 | 9 | 0 | 82% |
| short | 0 | 0 | 50 | 0 | 100% |
| long_doc | 0 | 0 | 0 | 50 | 100% |

Two test expectations were older than the HRX-only build, and neither was a routing bug.
`laya_route_e2e.sh` worked out the expected device with `--devices vulkan,hrx,zinc`, so it
expected `short -> vulkan`. In a build with HRX, serve offers Laya `hrx` and `zinc` (plus `rocm`
with `ONEBIT_LEAN_ROCM`), and it fell back correctly to `hrx` (`short 0.24 hrx`). The test now
reads the candidates from the build's `CMakeCache.txt`, as `gguf_devices()` picks them.
`laya_gguf_route.sh` checked the error for a GGUF without ggmlc's `laya`. A
`-DONEBIT_LAYA_GGML=ON` build always finds its own `laya`, so it now prints that the check was
skipped.

**End to end** (`1bit serve -m Qwen3-4B-Q4_K_M.gguf --device auto --laya` beside the same server
without `--laya`, both on HRX). 48 hand-written requests, 12 per class, each a new conversation,
`temperature 0`, 64 tokens, streamed. Requests alternate between the two servers, and every
request was sent in three rounds, each with fresh servers. Time to first token is the median;
"added" is the median of the per-request difference. Rounds 2 and 3 are pooled (n = 24 a class).
In round 1, time to first token on both servers flipped between about 80 ms and about 300 ms;
its median added time over all 48 was +32 ms.

| Class | right | confidence, median (range) | device | TTFT `--laya` | TTFT without | added |
|---|---|---|---|---|---|---|
| code | 12/12 | 0.42 (0.14-0.53) | hrx | 137 ms | 99 ms | +38 ms |
| prose | 12/12 | 0.17 (0.11-0.35) | hrx | 114 ms | 80 ms | +33 ms |
| short | 12/12 | 0.23 (0.11-0.33) | hrx | 116 ms | 85 ms | +33 ms |
| long_doc | 12/12 | 1.00 (size gate) | hrx | 526 ms | 522 ms | 0 ms |
| all | **48/48** | 0.33 | hrx | 132 ms | 98 ms | **+31 ms** |

- Every reply carried `X-1bit-Route`, and every request went to `hrx`. 8 of the 48 were below
  `min_confidence` (0.16), and the policy's default is `hrx` too.
- Each round gave the same 48 headers. The answers were sensible, and 47-48 of 48 matched the
  server without `--laya` word for word.
- Route cache: one conversation per class, three later turns each, gave the same header on all
  16 turns. Serve logged 49 decisions a round (48 plus the warm-up) and none for the later turns.
- The 2026-09-28 cost table above (about 0.5 s on a first turn) was the CPU scorer. With the
  GGUF scorer on HRX, the first turn of a conversation costs about 33 ms, and a long document
  costs nothing.

**What still names Vulkan or ROCm.** In a build with HRX, serve's Laya candidates for a `.gguf`
are `hrx` and `zinc` (ZINC runs on Vulkan or ROCm on this GPU), plus `rocm` with
`ONEBIT_LEAN_ROCM`. Every row of `config/route-policy.json`, and its default, is `hrx`, and a
candidate's backend starts only when the policy picks it. So no request reached Vulkan, ROCm or
zinc, and serve started only the HRX backend. Some names are left over and route nothing:
`1bit route`'s default `--devices npu,hrx,vulkan,zinc`, `vulkan` in `laya/route.cpp`'s device
descriptions (the old direct device question), and the `vulkan` output in this page's usage
example and its "Why every class goes to Vulkan today" paragraph, which come from the
all-Vulkan policy.

**Agent dispatch, out of domain.** We also tried `1bit route --classify` as a dispatcher for
agent sessions: code to engine work, prose to docs, long_doc to review, short to quick answers,
and anything under 0.16 confidence to a human. The test set was 40 hand-labelled orchestration
tasks, 10 per class. Laya's test set is chat requests, not agent tasks, and it shows:

| want \ got | code | prose | short | long_doc | right |
|---|---|---|---|---|---|
| code | 10 | 0 | 0 | 0 | 10/10 |
| prose | 4 | 6 | 0 | 0 | 6/10 |
| short | 7 | 0 | 3 | 0 | 3/10 |
| long_doc | 5 | 0 | 0 | 5 | 5/10 |

- **60% (24/40).** 23 tasks cleared 0.16 and 17 of those were right. The other 17 went to a human.
- Engine jargon reads as code. "Is box.lock free?" came out as code 0.13, and "Write the
  acknowledgements section" as code 0.07.
- A long-document task described in one line ("review a 30-file PR diff") is under the size gate,
  so it can only come out code, prose or short. With the diff, log or document attached, all 5
  were `long_doc`.
- The other two question wordings in `laya/route.cpp` did worse: 47.5% and 40.0%.
- Laya is good at telling code tasks apart from the rest, but agent dispatch would need its own
  labelled set, or keyword rules ahead of the model.
