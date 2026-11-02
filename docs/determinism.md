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
measurements; the thinking-on ones below ran on a build of `main` at `1b3def2`, which still emits NPU
log-probabilities for think tokens.

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

1. **Every route must run with thinking on** (`--thinking on`). The NPU forward path ignores
   `chat_template_kwargs.enable_thinking=false` (issue
   [#832](https://github.com/1bit-MONSTER/engine/issues/832)), so its stream always opens `<think>`; the
   llama-server routes honour the kwarg. Run all three with thinking on and the templates agree. This was
   the real blocker: run two routes with `enable_thinking=false` against this one and they diverge at
   position 0 for template reasons, which looks exactly like a numerical disagreement.
2. **Match tokens by text, not by id** (`--match-by text`). The NPU reports `token` and `logprob` but no
   `id`, in `logprobs.content` and in `top_logprobs` alike, so an id-keyed comparison between it and
   `cpu`/`hrx` matches nothing at all.

`message.content` being empty on that path does not matter: the comparison reads `logprobs.content`, which
is populated. And the log-probabilities have to come from a build whose NPU path emits them (see
[Routes and logprobs](#routes-and-logprobs)). One build detail matters here: `44d849d` onward emits
entries for **content** tokens only, so a thinking reply's `logprobs.content` is empty there — measure
with a build before it, or with `#832` fixed.

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

What can be promised today, then: repetition on all three routes, and a cross-route agreement number — not
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
- `chat_template_kwargs.enable_thinking=false` is honoured by the llama-server routes and not by
  the NPU forward path, which can return a `<think>` block where the others do not. Compare token
  streams only with the same template behaviour, or the divergence you measure is the template.

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

With the NPU in the comparison, the same command with `--thinking on --match-by text`: the forward path
needs no `enable_thinking` and reports no token ids, so all routes run thinking on and tokens are matched
by text. The NPU tries 2 repeats at the same 64-token budget — ~10 min in one claim, which is what its
~0.5 s/token allows. If the GPU and NPU claims cannot be held at once, run the pairs separately and
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
  and is not small, not how large it gets for other shapes.
- The NPU agreement rows are 2 repeats (the other routes' 5 would be ~25 min of the array) and, over a
  *requantised* pack, they are not an arithmetic-only distance. The 64-token row removes the earlier
  budget difference, so all three columns can be read at one token count.
- The NPU rows depend on the build. A binary at `44d849d` or later masks the think tokens, so on that
  build the NPU row cannot be reproduced until
  [#832](https://github.com/1bit-MONSTER/engine/issues/832) is fixed; the numbers above come from a
  `main@1b3def2` build, which still emits log-probabilities for the think tokens.
