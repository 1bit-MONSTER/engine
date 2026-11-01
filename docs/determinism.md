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
4. **The NPU route can be held to repetition but not to agreement with the others** — it is bitwise
   repeatable, but it reports no token ids, cannot disable thinking, returns its tokens only in
   logprobs, and serves *requantised* weights rather than the same ones. All four, below.

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
| `cpu` | bitwise-identical (5 repeats) | 0 | 128 | yes |
| `hrx` | bitwise-identical (5 repeats) | 0 | 122 | yes |
| `npu` (forward path) | bitwise-identical (3 repeats × 16 tokens) | 0 | 128 | yes |

All three routes are bitwise repeatable. That is the only property all three share.

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

### NPU: repeatable, but not comparable to the other two

On a build that carries the NPU log-probabilities, the NPU route is **bitwise repeatable**: 8 prompts ×
16 tokens × 3 repeats, 128 tokens, `max |Δlogp| = 0`, text stable, 0 non-float32 values. Three repeats
rather than five because the per-op forward path measured ~0.48 s/token here (18.8 s for a 32-token
reply), so the 8 × 5 × 64 matrix the other routes ran is not practical on it.

It cannot join the *agreement* comparison, for four independent reasons — any one of them is enough:

1. **It reports no token ids.** Each `logprobs.content` entry carries `token` (text) and `logprob` but
   no `id`, and the same holds inside `top_logprobs`. `tools/logprob_agree.py` therefore falls back to
   comparing token text when a route omits the id; comparing ids alone would "match" on `None == None`
   at every position and report a perfect agreement between two unrelated routes.
2. **Thinking cannot be turned off on that path.** `chat_template_kwargs.enable_thinking=false` is
   honoured by the llama-server routes and ignored here: the stream opens `<think>`, `\n`, `Okay`, …,
   so it diverges from the other two routes at token 0 for template reasons, not numerical ones.
3. **The reply is not in `content`.** `message.content` comes back empty on that path; the tokens are
   only in `logprobs`. A text-level comparison against `cpu`/`hrx` is therefore not meaningful either.
4. **It does not serve the same weights.** A Q4NX weight is a q4_1 tile (bf16 scale and zero per 32
   columns) and the pack is produced by requantising Q4_K, so NPU-vs-GGUF deltas contain a quantisation
   change on top of any kernel arithmetic. Such a number is not comparable to `cpu` vs `hrx` and should
   not be quoted as if it were.

What the NPU route can be held to today is repetition — which it passes — and not agreement.

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
