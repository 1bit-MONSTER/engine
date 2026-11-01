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
   of the prefill case documented in [serve.md](serve.md#-precise-a-follow-up-turn-is-bitwise-identical-to-a-cold-run);
   `--precise` does not address it, because it only forces a re-prefill.
3. **Two routes are not bit-identical, and today are not close.** Nothing pins a reduction order
   across a CPU ggml matmul and an HRX kernel, so log-probabilities from different devices differ
   in the third significant figure at the median and by up to ~0.4 nats at the tail. Greedy
   continuations diverge on a substantial fraction of prompts.
4. **The NPU route cannot currently take part in a log-prob agreement comparison at all** — two
   independent reasons, both below: the in-process NPU path returns no log-probabilities, and the
   NPU serves a *requantised* model, not the same weights.

Promises 3 and 4 are measurements, not bug reports: the routes genuinely compute different
floating-point sums over different quantised weights. That is the same class of problem that
motivated a fixed reduction order in dedicated deterministic-operator stacks, and it is why a
meaningful bitwise gate can only be asserted **within** one route, at a fixed batch composition.

## Measured

Qwen3-0.6B Q4_K_M, 8 prompts, 64 tokens each, greedy (`temperature 0`, `top_k 1`), fixed seed,
`logprobs=true`, `top_logprobs=5`, `cache_prompt=false`, thinking off, 5 repeats per prompt.
Binary `lemonade-onebit-rc-2026-10-31c` (c18).

### Repetition, per route (requests served one at a time)

| Route | run-to-run | max \|Δlogp\| | tokens | text stable |
|---|---|---|---|---|
| `cpu` | bitwise-identical | 0 | 128 | yes |
| `hrx` | bitwise-identical | 0 | 122 | yes |
| `npu` | not answerable — no logprobs; text repeatable | — | — | yes |

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

### NPU: repeatable text, no log-probabilities

`1bit serve -m <q4nx dir> --device npu`, 8 prompts × 64 tokens × 5 repeats: token stream
**repeatable 5/5** (text stable), log-probabilities **absent** — see below. Text identical to the
CPU route on **2/8** prompts, where `cpu` vs `hrx` is 4/8.

## Routes and logprobs

`logprobs` / `top_logprobs` on `/v1/chat/completions` are available on the `hrx` and `cpu` routes
(llama-server) and, on a build that has them, on the NPU forward path:

- A Q4NX **model directory** served with `--device npu` takes the in-process `unified` path
  (`app/serve.cpp` `run_unified`), which does not build log-probabilities. The reply is correct but
  carries no `logprobs` key even when asked. Issue
  [#651](https://github.com/1bit-MONSTER/engine/issues/651) is the NPU logprobs item; the
  implementation landed on the `forward-serve` path only.
- The NPU log-probabilities live in `app/forward_serve.cpp` + `app/npu_sampling.h`. Reach that path
  with `1bit forward-serve -m <dir>/model.q4nx --kernels <dir>`, which needs a directory of
  `full_i8_*.elf` designs — a shipped dx kernel directory (`npu/kernels/<H>x<IM>x<NH>x<NKV>/` of
  `layer*.elf`) does not contain them. A release binary older than that change has no NPU logprobs
  anywhere (`strings <bin> | grep top_logprobs` is the check).
- `chat_template_kwargs.enable_thinking=false` is honoured by the llama-server routes and not by
  the NPU forward path, which can return a `<think>` block where the others do not. Compare token
  streams only with the same template behaviour, or the divergence you measure is the template.

**The NPU does not serve the same weights.** A Q4NX weight is a q4_1 tile (bf16 scale and zero per
32 columns) and the pack is produced by requantising Q4_K, so NPU-vs-GGUF log-prob differences
contain a quantisation change, not only kernel arithmetic. A log-prob agreement number across that
boundary is not comparable to `cpu` vs `hrx` and should not be quoted as if it were.

## How to re-run

Start the routes (one process each; the GPU and NPU ones take the box's fair claims):

```sh
# cpu — no device claim needed
1bit serve -m ~/models/Qwen3-0.6B-Q4_K_M.gguf --device cpu --port 18110 --ctx-size 2048

# hrx — under the GPU claim, with the no-SVM variables set
HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1 \
  gpu-hold -l rlk -t 12 -- 1bit serve -m ~/models/Qwen3-0.6B-Q4_K_M.gguf --device hrx \
    --port 18111 --ctx-size 2048

# npu — under the NPU claim, and only on a build whose forward path has logprobs
npu-hold -l rlk -t 12 -- 1bit forward-serve -m ~/models/npu/Qwen3-0.6B/model.q4nx \
  --kernels <full_i8 ELF dir> -p 18112 --host 127.0.0.1
```

Then:

```sh
python3 tools/logprob_agree.py \
    --route cpu=http://127.0.0.1:18110 \
    --route hrx=http://127.0.0.1:18111 \
    --repeat 5 --max-tokens 64 --json agree.json
```

Exit code 0 means every route was bitwise repeatable *and* every pair matched exactly; 1 means a
route diverged on repeat, a route returned no log-probabilities, or a pair differs; 2 means a route
was unreachable. `--selftest` checks the comparison logic without a server.

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
