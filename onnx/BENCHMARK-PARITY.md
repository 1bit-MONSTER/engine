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
# Benchmark parity with AMD's Ryzen AI numbers

AMD's own numbers are the ones to beat, and they are Windows numbers taken with
`model_benchmark.exe` from the Ryzen AI SDK. We cannot run that tool: it needs Windows 11, and the
Linux equivalent needs AMD's `libonnxruntime_providers_ryzenai.so`, which ships only inside the
licence-gated install (our own box has `flm` and XRT but not that EP). So we reproduce the
**protocol** exactly and publish rows in the same units, each labelled with the device, precision
and engine it came from.

## The protocol we are matching

Source: <https://ryzenai.docs.amd.com/en/1.3/npu_oga.html> ("Run Benchmark"), Linux equivalent at
<https://ryzenai.docs.amd.com/en/latest/llm_linux.html>.

```
.\libs\model_benchmark.exe -i <model_path> -p <model_path>\prompts.txt -l "2048,1024,512,256,128"
```

| their flag | meaning | their default | ours |
|---|---|---|---|
| `-i` | model directory | — | `--model <file.onnx>` |
| `-p` | prompts file | — | not used; see "prompt" below |
| `-l` | comma list of prompt lengths | `2048,1024,512,256,128` | `--lens` (same default) |
| `-g` | tokens to generate | `128` | `--gen` (same default) |
| `-w` | warmup iterations | `1` | `--warmup` (same default) |
| `-r` | repeats | `5` | `--repeat` (same default) |
| `-t` | interval for peak-CPU-utilisation sampling, ms | `250` | not implemented (we report peak RSS only) |
| `--verbose` | more output | off | not needed |

Stages, and their own definitions, which we copy verbatim:

- **Prompt processing (time to first token)** — prompt tokens divided by the TTFT of one run.
- **Token generation** — per decoded token, excluding the first (the prefill's own token).
- **Token sampling** — their sampler's own time; ours is the argmax, so it reports the argmax.
- **E2E generation** — the whole generation loop, ms.
- **Peak working set size** — bytes; ours is `VmHWM` from `/proc/self/status`.

Their reference output (Phi-3.5-mini, NPU, int4, 128 prompt / 128 generated) is the only published
timing they print:

```
Prompt processing (time to first token): avg (us): 169860  avg (tokens/s): 753.562
Token generation: avg (us): 20354.1  avg (tokens/s): 49.1301
E2E generation (entire generation loop): avg (ms): 2755.09
Peak working set size (bytes): 3543330816
```

Their model set (11 repos, `-awq-g128-int4-asym-bf16-onnx-ryzen-strix`: Phi-3-mini-4k,
Phi-3.5-mini, Mistral-7B-v0.3, Qwen1.5-7B-Chat, chatglm3-6b, Llama-2-7b, Llama2-7b-chat,
Llama-3-8B, Llama-3.1-8B, Llama-3.2-1B, Llama-3.2-3B) publishes no timings on its cards, only
perplexity — so a per-model comparison table cannot be built from their side. What can be built is
one protocol, run on hardware we have.

## How we run it

```sh
ort_run --model <model>-cached.onnx --cached --bench [--lens "2048,1024,512,256,128"] \
        [--gen 128] [--warmup 1] [--repeat 5] --threads N
```

- One session, one graph; each repeat restarts from a zero-length cache, which is their repeat loop
  resetting the generator.
- Greedy decode (`argmax`), so the token stream is deterministic and reproducibility is checkable.
- **Prompt**: they read a text file; for timing only the token count matters, so we generate
  `100 + (i * 7919) % 30000` ids, valid in every model of their set (the smallest vocab, Phi-3, has
  32064 entries). Prompt text never enters the measurement.
- Statistics are population stddev over the `-r` repeats (these are all the samples there are).

`--bench` needs a `--cached` graph; on an uncached graph "token generation" would include re-running
the whole prompt, which is their prefill number wearing a different name.

## Rows we have measured

Qwen3-0.6B, the same weights as the ONNX work in this directory, 128 generated tokens:

| box | device | dtype | engine | prompt 128 | prompt 256 | prompt 512 | prompt 1024 | prompt 2048 |
|---|---|---|---|---|---|---|---|---|
| `ryzen` | CPU, 8 threads | fp32 | our ORT 1.23.2 + `ort_run --cached` | 16.73 tok/s | 15.21 | 13.12 | 9.87 | 6.73 |
| `strixhalo` | CPU, 32 threads | fp32 | same graph, same runner | **23.31** tok/s | 21.21 | 15.04 | 11.93 | 7.90 |

Prompt processing on the same rows (tokens/s):

| box | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| `ryzen` | 1102.1 | 1220.0 | 1175.2 | 777.8 | 639.1 |
| `strixhalo` | 1034.2 | 1295.4 | 1143.5 | 1041.2 | 901.0 |

Token sampling is flat on both (~2510 tok/s on `ryzen`, ~2280 on `strixhalo`) — that is one argmax
over 151936 logits, and it is not what either box is waiting on. Peak RSS is 8.2 GB on both.
Raw blocks: `/tmp/bench_qwen3_amd.txt` (ryzen), `/tmp/bench_strixhalo_amd.txt` (strixhalo).
`ctest` on the strixhalo build: 4/4 passed.

## Round 2: our own cache buffers, and what it did not fix

`run_cached` used to let ONNX Runtime allocate `present` on every step. It now ping-pongs two
preallocated stacks bound through `OrtIoBinding` (`BindInput "past"`, `BindOutput "present"`,
`BindOutputToDevice "logits"`), so the only cache traffic is the graph's own `Concat`. Numerics are
unchanged (ids `13 16 13 16`, mean KL 1.238e-07 against transformers, max|dlogit| 2.12e-05).

`ryzen`, 8 threads, fp32, 128 generated — before → after:

| prompt | decode before | decode after | prefill before | prefill after | peak RSS before | peak RSS after |
|---|---|---|---|---|---|---|
| 128 | 16.73 tok/s | 16.70 | 1102.1 | 1051.0 | 8.20 GB | 7.98 GB |
| 256 | 15.21 | 15.29 | 1220.0 | 1197.6 | | |
| 512 | 13.12 | 12.66 | 1175.2 | 1072.0 | | |
| 1024 | 9.87 | 9.89 | 777.8 | 866.3 | | |
| 2048 | 6.73 | 6.60 | 639.1 | 620.5 | | |

**The allocation-churn hypothesis was wrong.** Everything is inside noise, and the footprint barely
moved, so the O(n) cost is not `malloc` and not the copy: it is the graph. At a 2048-token prompt the
prefill alone materialises `logits [1,2048,151936]` (1.2 GB) and the attention intermediates, which is
where the 8 GB peak comes from. The change is kept because the cache is now a known, bounded quantity
two stacks wide instead of whatever the arena grew to, but it is not a speed fix and nothing here
claims it is. The next levers, in order, are graph-side: keep only the last logits row (a
`logits[:, -1:, :]` slice would delete most of the prefill's lm_head matmul) and profile the 56
per-layer `Concat`s before touching them.

## Round 3: the last-logits slice, and the prefill lever that was not there

The cached graph used to return `logits [1,seq,151936]` for the whole prompt when the runner reads one
row. Slicing it to the final position is one line, drops peak RSS by 1.4 GB, and **changed no timing**.
That invited a second try — run the backbone, then apply the head to the last hidden row, so the head
matmul is 1 row instead of `seq` — and that is no faster either. `ryzen`, 8 threads, 128 generated,
four graphs in order of when they were built:

| prompt | prefill base → bound cache → slice → head-last | decode base → bound cache → slice → head-last | peak RSS |
|---|---|---|---|
| 128 | 1102 → 1051 → 848 → 792 | 16.73 → 16.70 → 16.73 → 16.33 | |
| 256 | 1220 → 1198 → 1149 → 1028 | 15.21 → 15.29 → 15.24 → 14.64 | |
| 512 | 1175 → 1072 → 1141 → 909 | 13.12 → 12.66 → 12.98 → 12.34 | |
| 1024 | 778 → 866 → 777 → 615 | 9.87 → 9.89 → 10.09 → 9.57 | |
| 2048 | 639 → 621 → 567 → 640 | 6.73 → 6.60 → 6.67 → 6.43 | 8.20 → 7.98 → **6.60** → 6.60 GB |

Read the spread against the noise: per-length stddev on these prefill numbers is 30–460 ms, so those
columns are one number, not four. The only reproducible change across the whole round is memory.

Why the head was never the lever: it is 155M of the model's 596M parameters. Prefill does
2 x 0.596e9 x 2048 = 2.4 TFLOP for a 2048-token prompt and takes ~3.2 s, i.e. **~760 GFLOP/s on eight
cores** — that is the machine's fp32 throughput, so the stage is compute-bound at the hardware, not
slow because of graph shape. Deleting 155M of 596M params from one stage cannot show up.

What that leaves, honestly:

- **Prefill on CPU has no headroom left in the graph.** The fix is arithmetic density (int8/int4
  weights instead of fp32) or a different device — which is exactly why the NPU row exists.
- **Decode is bandwidth-bound** (2.4 GB of fp32 weights read per token), so it wants the same medicine:
  narrower weights.
- The slice is kept for the 1.4 GB it saves and nothing else. The one-line version beats the
  backbone-plus-head version at equal memory, because it is one line and does not depend on HF
  attribute names (`self.inner.model`, `self.inner.lm_head`) that transformers moves between releases.

## NPU row: our engine, the dx route

Qwen3-0.6B through `1bit serve -m <dir> --device npu --npu-opt dx=<kernel dir>`, 64 generated tokens,
3 repeats after 1 warmup, driven by `onnx/bench_npu.sh`. `strixhalo` (Ryzen AI Max+ 395), int4-weight
AIE designs, one design per layer:

| prompt tokens | prompt processing | token generation | ms/token | E2E |
|---|---|---|---|---|
| 162 | 77.69 tok/s | **72.64 tok/s** | 13.77 | 2.95 s |
| 332 | 76.02 | 62.56 | 15.98 | 5.24 s |
| 672 | 71.71 | 56.09 | 17.83 | 9.40 s |
| 1335 | 63.30 | 47.70 | 20.97 | 22.41 s |

Read against the CPU rows for the same model:

- **Decode: the NPU wins 4–7×** (72.6 tok/s at 162 tokens against 16.7 fp32-CPU; 47.7 at 1335
  against 6.6) and it degrades far more slowly — 1.5× from 162 to 1335 tokens, where the CPU loses
  2.5×. That is the KV cache holding, plus kernels that re-read 224 KB/token instead of 2.4 GB.
- **Prefill: the CPU wins 8–16×** (620–1200 tok/s against 63–78). NPU prompt processing is the weak
  stage and it is the same gap the A/B harness records against FLM (native ~67 tok/s vs 1269). No row
  here is a product claim until that closes.
- The prompt token counts are measured, not requested: the chat template adds its own prefix, so
  `--lens 1024,512,256,128` lands on 1335/672/332/162. AMD's block prints measured counts too, which
  is why the column exists.
- The route reports `prompt_ms` and `predicted_ms`, so `bench_npu_serve.py --mode json` reads those.
  Its streamed chunks all arrive after the generation finishes, which makes chunk-arrival timing read
  as 3 µs/token — measured once, discarded, and now the reason `--mode` exists (`stream` is for
  servers that really stream, like `flm`).

**Does a 2048-token prompt complete on dx?** Yes. One bounded request (`--lens 2048 --gen 8 -w 0 -r 1`,
`timeout 600`, single `flock`, port 52703) returned cleanly: the chat template turned the request into
**2,678** prompt tokens, prefill ran at **54.76 tok/s = 48.9 s**, then 8 tokens at **41.11 tok/s**
(24.3 ms/token), E2E 49.07 s, exit 0, device released.

The earlier "hang" was the harness, not the route: that run asked for 128 tokens over 6 repetitions,
i.e. ~52 s per repetition and over five minutes of silence with the client blocked in a read. Looks
like a hang, is not one — recorded because the first write-up called it a hang.

## The NPU prefill gap, in numbers

Same model, same box, the two stages are not close:

| prompt tokens | our prefill | our decode | FLM prefill (1k) | FLM decode |
|---|---|---|---|---|
| 162 | 77.69 tok/s | 72.64 tok/s | — | — |
| 332 | 76.02 | 62.56 | — | — |
| 672 | 71.71 | 56.09 | — | — |
| 1335 | 63.30 | 47.70 | 1269 (at 1024) | ~73 |
| 2678 | 54.76 | 41.11 | — | — |

Three measurements that pin it down:

1. **Our prefill runs at our decode rate** (63.3 vs 47.7 tok/s at 1335 tokens; 77.7 vs 72.6 at 162).
   Prefill is therefore making one token-sized pass over the 28 layers per prompt token, exactly like
   decode does, instead of batching the prompt into wider matmuls.
2. **The budget per layer is the whole story.** At 1335 tokens prefill is 15.8 ms/token across 28
   layers = ~564 us per layer dispatch. FLM at 1024 tokens is 1269 tok/s = 0.79 ms/token = ~28 us per
   layer. Same AIE designs class, same weights, 20x apart.
3. **Arithmetic cannot explain it.** Decode performs the same 28 layer matmuls per token, plus
   attention over the whole cache plus sampling, and lands at 21 ms/token — the extra prefill-only cost
   is not math.

So the concrete next lever, in order:

- **Collapse the per-layer host round trips.** One XRT submission per layer, 28 times per token, is
  what the two numbers above say we pay. Submitting the layer chain as one runlist is the cheap
  experiment, and the harness already knows how to pick that path (`npu_ab.sh --native-path
  runlist|split` exists for exactly this comparison). Target: the ~20x that FLM's 28 us/layer shows is
  reachable on this silicon.
- **If the round trips are already collapsed, the lever is rows, not dispatch:** a multi-row design
  (more activation rows per kernel invocation) makes prefill a batched matmul, which is what the stage
  wants. That is a kernel-generation change in `addons/npu-dx/scripts/build-kernels.sh`, i.e. the next
  generation of designs, not a tuning flag.

What is measured and what is inferred, since the difference matters: the rates above are measured; the
split of those 564 us between dispatch and compute is **inferred** — the dx route exposes no per-layer
profile hook, so nobody has seen where the time goes. A profile is the first thing to add before
choosing between the two levers, and it is cheaper than either.

## Where a 2048-token prefill actually goes

`ort_run --profile PREFIX` turns on ORT's own per-node trace (it writes `PREFIX_<timestamp>.json`) and
`onnx/prof_summary.py` folds it by op. One 2048-token prefill on `ryzen`, 8 threads, fp32:

| op | share of 4182 ms |
|---|---|
| `MatMul` | 40.1% |
| `Add` | 16.0% |
| `FusedMatMul` (attention) | 12.9% |
| `Softmax` | 6.7% |
| `Concat` (cache stacking) | 6.3% |
| `Mul` / `Where` / `Expand` | 10.0% combined |

So 53% is matmul — the part int8 speeds up and nothing else can — and **~22% is mask and broadcast
elementwise work** (`Where` + `Expand` + most of `Add`), which is ours to delete: the causal mask is
rebuilt as a `[1,1,2048,2048]` tensor inside the graph on every step. The single largest node in the
whole run is one `Add` at 161.8 ms and one `Where` at 141.6 ms, both mask construction. The next graph
change is therefore to hand the mask in as a dtype-correct input (the reason it was removed was a
zero-length-`past` `Expand` failure, not cost), not to look for more matmul tricks.

## What this table says

- The numbers are **not** comparable to AMD's published one as a speed claim: theirs is an int4 NPU
  on Windows, ours is fp32 CPU or our own int4 NPU route on Linux. What is comparable is the
  definition, so any of our rows drop into their table without translation.
- Four times the threads buys 1.4× at 128 tokens and 1.2× at 2048 of **decode**, which is
  bandwidth-bound: it reads the 2.4 GB fp32 weight file per token. 32 threads on `strixhalo` and 8 on
  `ryzen` land within 20% of each other at long context, which is the tell. Prefill does not behave
  that way at all — see the next bullet — which is why the two stages need different fixes.
- The CPU graph's decode degrades with context (16.7 → 6.7 tok/s from 128 to 2048) and round 2 shows
  it is not the cache buffers. The NPU route shows what the same model looks like when the cache is
  actually held on device: 72.6 → 47.7 over a comparable range.
- **Prefill on CPU is compute-bound at the hardware, decode is bandwidth-bound at it**, and round 3 is
  what settled that: 2048 prompt tokens cost 2.4 TFLOP and take ~3.2 s, i.e. ~760 GFLOP/s on eight
  cores, which is what those cores do. Deleting the head's matmul over the prompt changed nothing for
  that reason. So neither stage has graph-level headroom left on CPU: both want narrower weights
  (int8/int4) or another device, in that order of effort.
- On the NPU the shape of the problem is different again: prefill 54–78 tok/s (compute-bound, and
  ~20× off FLM's per-layer dispatch cost) with decode 41–73 tok/s on top of it. The lever there is
  described in the prefill section, and it is a dispatch or design change, not a tuning flag.

## Reproducing

```sh
# on any box with libonnxruntime and cmake
cmake -S onnx -B onnx/build -DCMAKE_BUILD_TYPE=Release && cmake --build onnx/build -j
./onnx/build/ort_run --model ~/wt/onnx-artifacts/qwen3-0.6b-cached.onnx --cached --bench --threads 8
```

Build the runtime yourself if you want the pin under your control: `onnx/build_ort.sh` (same tag,
one `#include <cstdint>` patch for GCC 15). Parity against the distro package is checked in
`onnx/README.md` ("Owning the runtime").

```sh
# the NPU row, on the box that has the NPU (kernel dir from addons/npu-dx/scripts/build-kernels.sh)
ONEBIT_DX_KERNELS=~/wt/ross-npu-product/kern/1024x3072x16x8 onnx/bench_npu.sh \
    --lens 1024,512,256,128 --gen 64 -w 1 -r 3 --mode json
```

`bench_npu.sh` owns the whole round trip — stage the model dir, start the serve, wait for health,
drive it, tear it down — and refuses to start if something already answers on its port, because a
stale server on that port once answered the health check and the bench then timed someone else's
process. `--mode stream` measures chunk arrival for servers that stream incrementally (`flm`);
`--mode json` reads the timings our routes report.
