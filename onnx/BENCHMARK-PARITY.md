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

## What this table says

- The numbers are **not** comparable to AMD's published one as a speed claim: theirs is an int4 NPU
  on Windows, ours is fp32 CPU on Linux. What is comparable is the definition, so a future NPU row
  from our own engine drops into the same table without translation.
- Four times the threads buys 1.4× at 128 tokens and 1.2× at 2048: this is a bandwidth-bound fp32
  graph streaming a 2.4 GB weight file, not a compute-bound one. 32 threads on `strixhalo` and 8 on
  `ryzen` land within 20% of each other at long context, which is the tell.
- Our decode **degrades with context** (16.7 → 6.7 tok/s from 128 to 2048 tokens) although the graph
  has a KV cache. The cache is an *output*: `present` is a fresh `[56,1,8,n,128]` stack, so each step
  copies 224 KB/token (459 MB at n=2048). That copy is the O(n) cost, and it is the next thing to
  fix — the attention itself is already cached.
- Prefill is memory-bound: 639 tok/s at 2048 tokens against 1220 at 256, i.e. nothing scales
  superlinearly, and TTFT at 2048 is 3.2 s.

## Reproducing

```sh
# on any box with libonnxruntime and cmake
cmake -S onnx -B onnx/build -DCMAKE_BUILD_TYPE=Release && cmake --build onnx/build -j
./onnx/build/ort_run --model ~/wt/onnx-artifacts/qwen3-0.6b-cached.onnx --cached --bench --threads 8
```

Build the runtime yourself if you want the pin under your control: `onnx/build_ort.sh` (same tag,
one `#include <cstdint>` patch for GCC 15). Parity against the distro package is checked in
`onnx/README.md` ("Owning the runtime").
