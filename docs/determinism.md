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
# Determinism and cross-route log-prob agreement

Two routes serving the same weights are not the same computation. This page states exactly what
`1bit serve` can promise about repetition and about agreement between devices today, with the
measured numbers behind each claim and the command to reproduce them. `tools/logprob_agree.py`
produces every number below.

## The contract we can promise today

1. **A repeated identical request is bit-identical on one route — while that request is served
   alone.** Send the same prompt, greedy, with `cache_prompt` false, and every generated token's
   log-probability has the same float32 bits. Holds on `cpu` and `hrx`.
2. **The same request served concurrently with others is not bit-identical, and this is the
   default configuration** (`--parallel 4`). The log-probabilities a client reads depend on how many
   sequences happened to decode with it. `--parallel 1` restores reproducibility. This is engine
   issue [#751](https://github.com/1bit-MONSTER/engine/issues/751) and is the decode-batch sibling
   of the prefill case already documented in the `--precise` section of [serve.md](serve.md);
   `--precise` does not address it, because it only forces a re-prefill.
3. **Two routes are not bit-identical, and today are not close.** Nothing pins a reduction order
   across a CPU ggml matmul and an HRX kernel, so log-probabilities from different devices differ
   in the third significant figure at the median and by up to ~0.4 nats at the tail. Greedy
   continuations diverge on a substantial fraction of prompts.
4. **All three routes are bitwise repeatable run-to-run**, and the NPU can be *compared* to the other
   two — but only with thinking on and tokens matched by text, and its distance from both carries a
   weight requantisation on top of the arithmetic. Its numbers are therefore not the arithmetic-only
   distance that `cpu` vs `hrx` is. Method and measurements below.

Promises 3 and 4 are measurements, not bug reports: the routes genuinely compute different
floating-point sums over different quantised weights. That is the same class of problem that
motivated a fixed reduction order in dedicated deterministic-operator stacks, and it is why a
meaningful bitwise gate can only be asserted **within** one route, at a fixed batch composition.

## Measured

Qwen3-0.6B Q4_K_M, 8 prompts, 64 tokens each, greedy (`temperature 0`, `top_k 1`), fixed seed,
`logprobs=true`, `top_logprobs=5`, `cache_prompt=false`, 5 repeats per prompt, thinking off except
where a table says otherwise. Binary `lemonade-onebit-rc-2026-10-31c` (c18) for the thinking-off
measurements, and `main@692cfa2` + #864 for the thinking-off three-route row (it reproduces the c18
`cpu`/`hrx` numbers digit for digit, as that section says); the thinking-on ones ran on a build of
`main` at `1b3def2`, which still emits NPU log-probabilities for think tokens.

### Repetition, per route (requests served one at a time)

| Route | run-to-run | max \|Δlogp\| | tokens | text stable |
|---|---|---|---|---|
| `cpu` | bitwise-identical (5 repeats) | 0 | 128 | yes |
| `hrx` | bitwise-identical (5 repeats) | 0 | 122 | yes |
| `npu` (forward path) | bitwise-identical (3 repeats × 16 tokens) | 0 | 128 | yes |
| `npu`, thinking on | bitwise-identical (2 repeats × 24 tokens) | 0 | 192 | yes |
| `npu`, thinking on, 64 tokens | bitwise-identical (2 repeats × 64 tokens) | 0 | 512 | yes |

All three routes are bitwise repeatable, at both thinking settings. That is the only property all three
share.

### Agreement, `cpu` vs `hrx`

Over 122 aligned positions, 88 of which carry the same token id:

| metric | value |
|---|---|
| token-id match rate | 0.721 |
| top-1 agreement rate | 0.721 |
| exact-bit match rate | 0.080 |
| max \|Δlogp\| | 3.98e-01 |
| mean \|Δlogp\| | 4.03e-02 |
| median \|Δlogp\| | 1.05e-03 |
| p90 \|Δlogp\| | 1.37e-01 |

The disagreement is in the distribution, not in how a tie was broken: at the largest deltas the
*same* token is top-1 on both routes while its log-probability differs — token 220 in prompt 1
position 7 at `-0.2097` (cpu) vs `-0.6077` (hrx), with the whole top-5 reordered. The
first-divergence top-2 margins were 0.33/0.11, 0.014/0.32, 0.041/0.26 and 0.29/0.008 (cpu/hrx) —
mostly not near-ties.

For scale against a published gate: a dedicated deterministic-operator stack (RL-Kernel) uses
`max_abs_dlogp <= 0.06` and a proposed `mean_abs_dlogp <= 0.005`, measured for one kernel across
tensor-parallel layouts on one GPU. Our cross-device max is 6.6× their max gate and our mean is 8×
their proposed mean gate.

### Concurrency changes the numbers (`--parallel 4`, the default)

Counts are token-id mismatches and log-prob bit mismatches on positions where the ids agree, out of
48 generated tokens. Probe: *"Explain in one paragraph why the sky appears blue."*

`cpu`, default `n_slots = 4`:

| comparison | ids | logprob bits |
|---|---|---|
| alone vs alone (sequential) | 0/48 | 0/48 |
| alone vs after 3 other prompts ran sequentially | 0/48 | 0/48 |
| alone vs concurrent with 3 *different* prompts | 21/48 | 27/48 |
| 4 identical prompts concurrent, copy0 vs copy1 | 43/48 | 5/48 |

First divergence with 3 other prompts in flight is at **token 0** on the *same* token id:
`-0.000222` alone vs `-0.000252` batched.

`cpu`, `--parallel 1`, the same concurrent test: **0/48 and 0/48** for every copy, against the
alone run and against each other. The two servers agree with each other on the serial prompt, so
batch width, not the build, is the variable.

`hrx`, default `n_slots = 4`: copy0 vs alone `4/48` ids but **44/48** log-prob bits, i.e. the
reported numbers move even where the tokens do not.

### Can a *batched* cell be repeated? (`--parallel N`, the same N requests twice)

Repeating one request answers the solo case. A `--parallel N` cell is measured by issuing **N concurrent
requests**, so its reproducibility question is different: replay the same N requests and you do not get
the same N answers. Measured with `tools/batched_repeat.py`, which releases N byte-identical requests
through a barrier for R rounds and fingerprints each copy over its token ids and float32 log-probability
bits. Qwen3-0.6B Q4_K_M, 32 tokens, greedy, 4 rounds (3 where a width only needed a control), shipped
`lemonade-onebit-rc-2026-11-04-c24` payload:

| request width | serve `--parallel` | `cpu` round-to-round identical | `cpu` states | `hrx` round-to-round identical | `hrx` states |
|---:|---:|---|---:|---|---:|
| 1 | 2 and 8 | **2/2** | **1** | — | — |
| 2 | 2 | 2/4 | 4 | — | — |
| 4 | 4 | 6/12 | 4 | **2/12** | 4 (all four copies distinct) |
| 8 | 8 | 11/24 | 8 | **0/24** | 16 (all eight copies distinct) |

Worst \|Δlogp\| seen inside those mixtures: `cpu` 9.81e-01 at width 4 and 9.81e-01 at width 8; `hrx`
7.43e-02 at width 4 and 2.05e-01 at width 8.

**Not a small-model artifact.** The same run on **Qwen3-4B-Q4_K_M**, same binary, same budget:

| request width | `cpu` round-to-round identical | `cpu` states | `hrx` round-to-round identical | `hrx` states |
|---:|---|---:|---|---:|
| 4 | **4/12**, worst \|Δlogp\| 1.86e-01 | 7 of 16 | **4/12**, worst \|Δlogp\| 6.83e-02 | 4 of 16 (all four copies distinct, every round) |
| 8 | **7/24**, worst \|Δlogp\| 1.80e-01 | 8 of 32 | **3/24**, worst \|Δlogp\| 1.02e-01 | 8 of 32 (all eight copies distinct, every round) |

At 4B the verdict is the same on both routes and the spread is wider on `cpu` (7 states over 16 copies,
against a modal rung at 0.6B), so the effect is a property of batching, not of the model size where these
measurements were cheap enough to take.

**Generated length decides how visible it is.** Width 4 on `cpu`, 0.6B, the same prompt decoded to four
lengths, every concurrent copy scored against the solo answer. **No copy was identical to the solo answer
at any length** (0 of 12 at each of 16/32/128 tokens, 0 of 16 at 64):

| tokens generated | \|Δlogp\| vs solo, median | same token as solo, median |
|---:|---:|---:|
| 16 | 5.49e-02 | **1.000** |
| 32 | 1.07e-01 | 0.844 |
| 64 | 2.52e-01 | 0.438 |
| 128 | 2.60e-01 | 0.394 |

At 16 tokens the batched continuations usually still read the same as the solo one while their
log-probabilities already differ; by 64 tokens the batch has usually left the solo sequence entirely. A
short batched cell therefore hides this and a 180-token one cannot — which is the length a scoreboard cell
actually decodes.

A **state** here is one complete 32-token answer: two copies are in the same state only if every token id
and every compared float32 log-probability bit of all 32 tokens matches. A row's "states" column counts the
distinct states over all copies of all rounds, so it is a lower bound (two different schedules can coincide)
and it widens with the width rather than settling. Read a row like this: at width 8 on `cpu`, 32 copies
produced 8 distinct answers and only 11 of the 24 same-index pairs between consecutive rounds matched.
Each row is **one run**: the exact counts move between runs because *which* sequences share a decode step
moves (a repeat of the width-4 `cpu` row gave 3/12 and 7 states instead of 6/12 and 4, same verdict).

**The state is the per-step batch width, not the serve configuration.** A single request against a
`--parallel 2` serve and against a `--parallel 8` serve produce the *same* fingerprint bit for bit, 3/3
rounds each; the ladder is the width of the decode step that ran, which is the same op #751 localised
(`ggml-cpu/repack.cpp`'s 4-row block vs 1-row gemv). The ladder is visible directly: at width 2 one
round had one copy on the solo state and its partner on a displaced one, i.e. a step that carried one
row for one sequence and two for the other. On `cpu` the full-batch state is modal (8 of 16 copies at
width 4, 20 of 32 at width 8), so a batched run usually *does* look like its neighbours; on `hrx` every
copy in a round is a distinct state (4/4 at width 4, 8/8 at width 8, 0/24 identical round-to-round) — no
two copies shared a per-step row count across the whole run, so there is no modal rung to land on.

Consequence for a correctness gate: a batched cell cannot be gated on one number. Its log-probabilities
are a mixture over that ladder, the displaced rungs are 1-2 nats away on `cpu`, and a single run can be
any rung — so the cell has to report the dispersion over its own rounds, or pin the schedule.

### What the `hrx` width gate is, by name

`GGML_HRX_LOG_DISPATCH=1` prints each dispatch registration the first time it matches
(`dispatch_registry/dispatch-registry.cpp`), so the set of registrations a serve logs is the set of
kernels its graphs actually ran. Keying that log on the root node's shape as well — a throwaway local
patch, not proposed for merge — separates the widths, because a *solo* request also prefills at width
>1. One `hrx` serve, Qwen3-0.6B, 48-token probe at `--parallel 4`, 26 registrations, and the decode
step splits into two disjoint sets:

| decode width | registrations seen only at that width |
|---|---|
| 1 (solo, and any sequential history) | `hip.qkv_rope`, `kquant.mul_mat.decode_f32`, `hip.dense_kquant_swiglu`, `hip.gemv_add_norm`, `common.mul_mat.f32_f32_decode` |
| 2 (concurrent) | `hip.kquant_tokens.mm`, `hip.kquant_tokens.mm_add`, `hip.kquant_tokens.swiglu`, `common.rmsnorm_binary_f32` |

Seventeen more registrations match at both widths, including the attention kernel
(`common.flash_attention_decode_split_next_q8`): on `hrx`, unlike on `cpu`, attention is *not* where
the width dependence comes from. What changes is the **fusion boundaries**. The residual add and
RMSNorm are one kernel at width 1 (`hip.gemv_add_norm`) and two dispatches at width 2
(`hip.kquant_tokens.mm_add` + `common.rmsnorm_binary_f32`); gate/up/SwiGLU is
`hip.dense_kquant_swiglu` versus `hip.kquant_tokens.swiglu`; the K-quant matmul is the per-token
decode family versus the `decode_tokens` family. So the width-2 graph is not the width-1 graph
widened, it is a differently fused graph.

**No single switch restores width invariance on `hrx`.** Six `GGML_HRX_DISABLE_DISPATCH=` arms were
run, each verified to have taken effect by the changed registration set: disabling the width-2
token-matmul family, the width-2 separate norm, the width-1 fused add+norm, and all four width-1
fusions together each still left the width-2 numbers displaced from the solo ones — max |Δlogp|
0.045-0.132 nats in every arm, against 0.092 for the unmodified server. Forcing both widths onto a
single kernel set is not expressible at all: `hip.dense_kquant_swiglu` has no fallback, and with it
disabled `GGML_HRX_STRICT=1` refuses the graph (`ffn_gate-27` `MUL_MAT f32 ne=[3072,1,1,1]` not placed
on HRX, 2 of 986 nodes off HRX). Disabling the width-1 fused add+norm alone moves the *solo* numbers
by 0.121 nats — as much as the width switch itself — so these fused kernels each reassociate a sum
rather than being interchangeable extras. Contrast `cpu`, where one flag (`-DGGML_CPU_REPACK=OFF`)
removes the width dependence entirely: on `hrx` the reproducible configuration is `--parallel 1`, and
there is no equivalent env switch.

### NPU: repeatable, and comparable with two adjustments

The NPU route is **bitwise repeatable**: 8 prompts × 24 tokens × 2 repeats (192 tokens), `max |Δlogp| = 0`,
0 non-float32 values, text stable — as well as the earlier 8 × 16 × 3 (128 tokens) run on the same build.
It stays bitwise repeatable at the other routes' budget too: 8 prompts × **64** tokens × 2 repeats
(512 tokens), `max |Δlogp| = 0`, 0 non-float32 values, text stable.

The extra cost is time, not correctness. The per-op forward path measures ~0.5 s/token here, so the same
8 × 64 × 2 matrix took **586 s of device time in one NPU claim** (against 34 s for `cpu` + `hrx` under one
GPU claim). Five repeats — what the `cpu` and `hrx` columns use — would be ~25 min of the array, so the NPU
column is 2 repeats at every budget.

It **can** join the agreement comparison — the earlier version of this page said it could not — with two
adjustments, both of which are flags on the tool rather than changes to the engine:

1. **Thinking has to be the same on both sides of a pair.** It used to be a *requirement* that every
   route run with thinking **on** (`--thinking on`), because the NPU forward path accepted
   `chat_template_kwargs.enable_thinking=false` and dropped it
   ([#832](https://github.com/1bit-MONSTER/engine/issues/832)), so its stream always opened `<think>` while
   the llama-server routes honoured the kwarg and the pair diverged at position 0 for template reasons —
   which looks exactly like a numerical disagreement. With [#864](https://github.com/1bit-MONSTER/engine/pull/864)
   the NPU path honours it (`app/npu_chatml.h`), so `--thinking off` works on all three routes and that is
   the configuration measured in [All three routes, thinking off](#all-three-routes-thinking-off) below.
   The tables *in this section* are the thinking-on ones, taken on a pre-#748 build.
2. **Match tokens by text, not by id** (`--match-by text`). The NPU reports `token` and `logprob` but no
   `id`, in `logprobs.content` and in `top_logprobs` alike, so an id-keyed comparison between it and
   `cpu`/`hrx` matches nothing at all.

`message.content` being empty on that path does not matter: the comparison reads `logprobs.content`, which
is populated. And the log-probabilities have to come from a build whose NPU path emits them (see
[Routes and logprobs](#routes-and-logprobs)). One build detail matters here: `44d849d` onward emits
entries for **content** tokens only, so a thinking reply's `logprobs.content` is empty there: measure
with a build before `44d849d`, or with `--thinking off` on a build carrying #864 (the #832 fix).

Qwen3-0.6B, 8 prompts, greedy, thinking **on**, 2 repeats (NPU) / 5 repeats (cpu, hrx).
One control first: the `cpu` route was measured in **two separate server runs**, and the two agree
**exactly, over both budgets** — 192/192 positions at 24 tokens and **512/512 at 64 tokens**, exact bits,
`max |Δlogp| = 0` — so comparing routes across runs (necessary here: the GPU and NPU claims are held
separately) does not manufacture or hide a difference.

**At 64 tokens, the same budget the `cpu`/`hrx` table above uses:**

| pair | token match | top-1 | exact-bit match | max \|Δlogp\| | mean \|Δlogp\| | prompts diverging |
|---|---|---|---|---|---|---|
| `cpu` vs `hrx` | 0.766 | 0.766 | 0.031 | 6.31e-01 | 4.62e-02 | 7 of 8 |
| `cpu` vs `npu` | 0.264 | 0.264 | 0.000 | 6.71e-01 | 5.51e-02 | 8 of 8 |
| `hrx` vs `npu` | 0.266 | 0.266 | 0.000 | 6.55e-01 | 5.19e-02 | 8 of 8 |

(392 of the 512 positions carry the same token on `cpu`/`hrx`, 135 and 136 on the NPU pairs; deltas are
over the common positions only.)

**At 24 tokens**, for the shape of the same three pairs over a shorter stream:

| pair | token match | top-1 | exact-bit match | max \|Δlogp\| | mean \|Δlogp\| | prompts diverging |
|---|---|---|---|---|---|---|
| `cpu` vs `hrx` | 0.958 | 0.958 | 0.060 | 6.25e-01 | 2.83e-02 | 1 of 8 |
| `cpu` vs `npu` | 0.641 | 0.641 | 0.000 | 6.71e-01 | 4.66e-02 | 7 of 8 |
| `hrx` vs `npu` | 0.641 | 0.641 | 0.000 | 6.55e-01 | 4.21e-02 | 7 of 8 |

(123 of the 192 positions carry the same token text on the NPU pairs, 184 on `cpu`/`hrx`.)

**The NPU is roughly equidistant from both — and that distance is not arithmetic-only.** It matches `cpu`
and `hrx` at the same rate, with the same deltas (0.641 and 4.2-4.7e-02 mean at 24 tokens; 0.26 and
5.2-5.5e-02 at 64), and it shares **no exact bits with either** (0.000, against 0.031-0.060 for
`cpu`/`hrx`). Its *distance* from `cpu` relative to `hrx`'s narrows as the stream lengthens — **1.6× at 24
tokens, 1.2× at 64** — because a longer stream diverges more, which raises the `cpu`/`hrx` delta as well as
the NPU's; the ordering (NPU furthest, `cpu`/`hrx` closest) is stable at both budgets and the exact-bit
split is not budget-dependent. A Q4NX weight
is a q4_1 tile (bf16 scale and zero per 32 columns) requantised from Q4_K, so the NPU number contains a
**weight change plus** kernel arithmetic and is *not* the same quantity as `cpu` vs `hrx`. Quote it as
"NPU pack vs GGUF" or not at all.

### All three routes, thinking off

Thinking off is the configuration a log-probability comparison actually wants (a thinking stream opens with
stereotyped tokens that make every pair look closer than it is), and it is only measurable on a build
carrying [#864](https://github.com/1bit-MONSTER/engine/pull/864). This row was taken on
**`main@692cfa2` + #864 (`ce1af72`)** — #864 was still open when it ran, so the commit is named here
rather than a merge sha. Qwen3-0.6B Q4_K_M, 8 prompts × 64 tokens, greedy, `--thinking off`,
`--match-by text`, NPU at `--prefix-slots 0`.

| Route | repetitions | run-to-run | max \|Δlogp\| | tokens | text stable |
|---|---|---|---|---|---|
| `cpu` | 5 | **bitwise-identical** | 0 | 128 | yes |
| `hrx` | 5 | **bitwise-identical** | 0 | 122 | yes |
| `npu` (forward path) | 2 | **bitwise-identical** | 0 | 131 | yes |

| pair | token match | top-1 | exact-bit match | max \|Δlogp\| | mean \|Δlogp\| | common tokens | prompts diverging |
|---|---|---|---|---|---|---|---|
| `cpu` vs `hrx` | 0.721 | 0.721 | 0.080 | 3.98e-01 | 4.03e-02 | 88/122 | 4 of 8 |
| `cpu` vs `npu` | 0.331 | 0.331 | 0.000 | 6.23e-01 | 8.62e-02 | 39/118 | 6 of 8 |
| `hrx` vs `npu` | 0.410 | 0.410 | 0.000 | 5.87e-01 | 8.89e-02 | 48/117 | 6 of 8 |

The two runs were separate device claims, so the `cpu` route was served twice and is the control: the two
`cpu` reports agree **128/128 positions, exact bits, `max |Δlogp| = 0`**.

Two things this row adds to the thinking-on tables above:

- **The `cpu`/`hrx` row reproduces the c18 binary digit for digit.** Same 122 aligned positions, 88 common,
token match 0.721, exact-bit 0.080, max 3.98e-01, mean 4.03e-02 as the first measurement of this page
(`lemonade-onebit-rc-2026-10-31c`). Nothing in the CPU or HRX decode path moved between that binary and
`main@692cfa2`, which is the llama.cpp pin's own claim and now also a measurement.
- **Thinking off moves the NPU pairs further away, not closer.** Mean \|Δlogp\| is 8.6-8.9e-02 here against
5.2-5.5e-02 with thinking on, and the token-match rate is higher (0.33-0.41 against 0.26) because thinking
streams are long runs of near-deterministic tokens that both routes pick identically. Neither row is a
better row: they answer different questions ("the same 64 tokens" against "the same 64 tokens of a
reasoning stream"), and the thinking-off one is the one a numeric gate would be written against.

What can be promised today, then: repetition on all three routes, and cross-route agreement numbers — not
an agreement threshold. No pair is bitwise, and no pair is within RL-Kernel's `max_abs_dlogp <= 0.06` gate.

## Routes and logprobs

`logprobs` / `top_logprobs` on `/v1/chat/completions` are available on the `hrx` and `cpu` routes
(llama-server) and, on a build that has them, on the NPU forward path:

- A Q4NX **model directory** served with `--device npu` takes the in-process `unified` path
  (`app/serve.cpp` `run_unified`), which does not build log-probabilities. The reply is correct but
  carries no `logprobs` key even when asked. Issue
  [#651](https://github.com/1bit-MONSTER/engine/issues/651) is the NPU logprobs item; the
  implementation is on the `forward-serve` path only, so `--device npu` with a directory is the
  wrong route to measure on.
- The NPU log-probabilities live in `app/forward_serve.cpp` + `app/npu_sampling.h`. Reach that path
  with `1bit forward-serve -m <dir>/model.q4nx --kernels <dir>`, which needs a directory of
  `full_i8_*.elf` designs — a shipped dx kernel directory (`npu/kernels/<H>x<IM>x<NH>x<NKV>/` of
  `layer*.elf`) does not contain them. A release binary older than that change has no NPU logprobs
  anywhere (`strings <bin> | grep top_logprobs` is the check).
- `chat_template_kwargs.enable_thinking=false` is honoured by every route, the NPU forward path
  included since #864 (`app/npu_chatml.h`); before that it was accepted and dropped there, so an NPU
  reply could open a `<think>` block where the others did not. Compare token streams only with the same
  template behaviour, or the divergence you measure is the template.

The weight caveat above is why the NPU's agreement numbers are reported with the pack named and not
folded into the `cpu` vs `hrx` column: the measured tables are above.

## How to re-run

Start the routes (one process each; the GPU and NPU ones take the box's fair claims):

```sh
# cpu — no device claim needed
1bit serve -m ~/models/Qwen3-0.6B-Q4_K_M.gguf --device cpu --port 18110 --ctx-size 2048

# hrx — under the GPU claim, with the no-SVM variables set
HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
  gpu-hold -l rlk -t 12 -- 1bit serve -m ~/models/Qwen3-0.6B-Q4_K_M.gguf --device hrx \
    --port 18111 --ctx-size 2048

# npu — under the NPU claim; the forward path is the only one with logprobs
npu-hold -l rlk -t 18 -- 1bit forward-serve -m ~/models/npu/Qwen3-0.6B/model.q4nx \
  --kernels <full_i8 ELF dir> -p 18112 --host 127.0.0.1
```

Then:

```sh
python3 tools/logprob_agree.py \
    --route cpu=http://127.0.0.1:18110 \
    --route hrx=http://127.0.0.1:18111 \
    --repeat 5 --max-tokens 64 --json agree.json
```

The batched question (does replaying the same N concurrent requests give the same N answers) is a
different run against the same serves:

```sh
python3 tools/batched_repeat.py --url http://127.0.0.1:18110 --width 4 --rounds 4 --max-tokens 32 \
    --json batched-w4.json
```

`--width 1` against a serve started with any `--parallel` is the control: it must be repeatable and it
must produce the same fingerprint whatever the slot count, which is what shows the state is the batch
content and not the configuration. Point it at an `hrx` serve under the GPU claim for the right-hand
columns above. `--selftest` checks the fingerprint and comparator logic without a server.

With the NPU in the comparison, the same command with `--thinking off --match-by text`, on a build carrying
#864: the forward path reports no token ids, so tokens are matched by text, and thinking off is what the
last three-route row was taken with (`--thinking on` also works there and is what the earlier rows used).
The NPU tries 2 repeats at the same 64-token budget — ~2.5 min in one claim with thinking off, ~10 min with
thinking on, which is what its ~0.5 s/token allows. If the GPU and NPU claims cannot be held at once, run the pairs separately and
compare the two reports with the tool's own comparator:

```sh
python3 tools/logprob_agree.py --route cpu=… --route npu=… \
    --thinking on --match-by text --repeat 2 --max-tokens 64 --json npu.json
python3 tools/logprob_agree.py --route cpu=… --route hrx=… \
    --thinking on --repeat 5 --max-tokens 64 --json hrx.json
python3 - <<'PY'   # agreement across the two reports, by text
import importlib.util, json
s = importlib.util.spec_from_file_location("t", "tools/logprob_agree.py")
t = importlib.util.module_from_spec(s); s.loader.exec_module(t)
load = lambda p, r: json.load(open(p))["routes"][r]
npu, hrx = load("npu.json", "npu"), load("hrx.json", "hrx")
print(t.agreement(npu[0]["runs"][0], hrx[0]["runs"][0], "text"))
PY
```

Exit code 0 means every route was bitwise repeatable *and* every pair matched exactly; 1 means a
route diverged on repeat, a route returned no log-probabilities, or a pair differs; 2 means a route
was unreachable. `--selftest` checks the comparison logic without a server.

To re-derive the `hrx` width gate rather than guess it, run the same serve with `GGML_HRX_LOG_DISPATCH=1`
and issue a solo request in one process and a concurrent one in another; the registration sets differ by
exactly the table above. `GGML_HRX_DISABLE_DISPATCH=<substring>` turns any one registration off, which is
how each arm above was run.

## Two things that will waste your afternoon

- **`/v1/health` is a readiness probe, not a liveness probe.** While the model loads it answers
  503; once loaded it answers 200; but an *idle* llama-server reports
  `{"status":"stalled","seconds_since_progress":N}` with 503 and clears the moment a request
  arrives. Waiting on `/v1/health` == 200 before the first request is correct. Using it to decide
  "is this endpoint up" is not: a perfectly usable idle server fails that test.
- **`/v1/models` answers before the model finishes loading**, so it cannot be used for readiness
  either. It is only a liveness probe. Use each for what it actually reports.

## Known limits

- The wire carries log-probabilities as JSON decimals, i.e. a float32 promoted to a double. The
  tool compares the float32 bit patterns and counts any value that does not round-trip; in every
  run recorded here that count was zero, so the cast is exact for these routes. A route that ever
  emitted a float64 log-probability would not be compared bit-exactly by this tool.
- `cache_prompt` false everywhere means no route is measured through the prompt-cache path. The
  cache path can round differently; that is a separate axis and `tools/kat.py` covers it.
- One model, 0.6B, 8 prompts: enough to show the routes disagree and that repetition is exact, not
  enough to estimate a tail. Treat the max as a lower bound on the worst case.
- The numbers are for one binary and one model quantisation. A change to a quantisation kernel
  moves them; re-run rather than extrapolate.
- The concurrency numbers were measured on a 0.6B model on one host; they show the effect exists
  and is not small, not how large it gets for other shapes. The batched-repeat rows are one prompt,
  32 tokens, 0.6B and 4B: they establish that a batched cell is not repeatable and that the outcome
  set is small and per-engine, not how the dispersion scales with model size or prompt length.
- The NPU agreement rows are 2 repeats (the other routes' 5 would be ~25 min of the array) and, over a
  *requantised* pack, they are not an arithmetic-only distance. The 64-token row removes the earlier
  budget difference, so all three columns can be read at one token count.
- The NPU rows depend on the build. A binary at `44d849d` or later masks the think tokens, so a *thinking*
  reply reports no entries on it at all: the thinking-on rows come from a `main@1b3def2` build, which still
  emits log-probabilities for the think tokens. The thinking-off row needs neither that build nor an
  unfixed engine — it ran on `main@692cfa2` + #864, i.e. exactly what a client gets once #832's fix lands.
