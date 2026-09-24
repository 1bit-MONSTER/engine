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
# NPU: Qwen3.6-35B-A3B, one runlist submit per token (lax)

The whole 35B MoE token runs on the NPU as **one XRT runlist submit**:

- 40 layers: 30 DeltaNet linear-attention layers and 10 full-attention layers.
- Each layer's routed experts are chosen and fetched on the device.
- The final norm and the lm head are two more runs.

It matches an fp64 reference, and it chats at 16.1 tok/s.

**Status: experimental. It is not part of `1bit serve`.** The kernels are the merged
whole-layer design `lax` from the open kernels in `third_party/OpenFlowLM-Next`. A
Python driver in that tree runs them through XRT's classic xclbin path. The fast lane
([npu.md](npu.md)) is full-ELF and pure C++, and this path does not meet that bar yet
(see "What is left").

## The pin

`third_party/OpenFlowLM-Next` pins branch `1bit/lax-35b` of
[bong-water-water-bong/OpenFlowLM-Next](https://github.com/bong-water-water-bong/OpenFlowLM-Next)
(MIT). That branch holds the fixes and optimisations below. Upstream's
`open_kernels/designs/layer_x/35b-whole-layer-runlist-status.md` records every
measurement behind them.

## Build and check

```sh
git submodule update --init third_party/OpenFlowLM-Next
IRON_VENV=<mlir-aie 1.4.2 + Peano venv> AIETOOLS=<Vitis>/aietools \
    scripts/build-lax.sh ~/.cache/lax          # ~3.5 min: 4 kernel builds + the harness

# the fp64 reference, once (CPU, ~15 min, writes 40 x 512 MB pools into --pool-dir)
cd ~/.cache/lax/src/open_kernels
python3 model/make_decode.py --requant --model-dir <model dir> --layers 40 --tokens 3 \
    --out ~/.cache/lax-ref --pool-dir ~/.cache/lax-ref/pools

tests/npu_lax_parity.sh ~/.cache/lax <model dir> ~/.cache/lax-ref    # from the engine checkout
```

`<model dir>` is a Q4NX model directory (`model.q4nx`, `config.json`, `tokenizer.json`).
The weights are packed from `model.q4nx` at load time and streamed into the XRT
buffers, so no pre-packed files are needed. `--requant` is required, because the `lax`
builds are all-q4_1.

Chat:

```sh
cd ~/.cache/lax/src/open_kernels
python3 model/lax_chat.py --model-dir <model dir> --lax-l ~/.cache/lax/kernels/lax_l \
    --lax-a ~/.cache/lax/kernels/lax_a "What is the capital of France? Answer in one sentence."
```

Without prompts on the command line, it reads prompts from stdin. The chat is one
device session, so the KV cache and the DeltaNet state carry across turns.

## Results

All results are on Strix Halo, using `model.q4nx` sha256 `688f1e153d10…3cf8de` (the
`Qwen3.6-35B-A3B-NPU2` Q4NX container, 23,235,412,536 B). The pin is `2490fa6e6ddb`,
built by `scripts/build-lax.sh`. It produces `insts.bin` md5 `136cc0c9…` (lax_l) and
`4f3c749d…` (lax_a).

**Parity** (`tests/npu_lax_parity.sh`), three positions from `<|im_start|>`, each
position's greedy token fed back:

| position | logits corr | argmax (ours = reference) | top-5 |
|---|---|---|---|
| 0 | 0.999998 | 846 | identical |
| 1 | 0.999993 | 198 | identical |
| 2 | 0.999998 | 3710 | identical |

The chat turn answers "The capital of France is Paris."

**Speed** (`lax_chat.py`, 128 generated tokens, host load average 8-10):

- 16.1 tok/s end to end. That covers the 40 layers as one submit, the norm, the lm
  head, and the argmax fed back to the host.
- Prompt tokens run at about 19 tok/s.
- Load, from process start to ready (packing 21.8 GB into the buffers): 12.1 s.

### How it got there

| Step (commit on `1bit/lax-35b`) | 40 layers | tok/s |
|---|---|---|
| First correct decode (469c43b) | 75.8 ms | 11.1 |
| Full-attention layers skip the dummy DeltaNet and pad bands (3632f7d) | 67.5 ms | 12.3 |
| DeltaNet fills and drains issued head-major across the 8 cores (0866c3f) | 53.5 ms | 14.7 |
| Router weights on two streams, vector top-8, MoE header overlapped (2490fa6) | 47.7-48.1 ms | 16.1 |

### What made it correct

Before 469c43b, the design had only ever completed on unfilled buffers. On real
weights it returned NaN. Three faults were behind that:

1. **Two `acquire(1)` calls return the same element.** The emitter core took the
   router output and its config as two `acquire(1)` calls on the same object fifo.
   Acquire counts are cumulative, so the second call returned the first element again.
   The pool base address was therefore read from the router's probabilities. Fix:
   `acquire(2)`.
2. **The host and the emitters raced on one queue.** The host queued the shared
   expert's fills on the same MM2S queue the emitters push the routed experts to. The
   host did not wait for the last routed push, and the fills are the same size, so the
   core silently used the wrong weights. Fix: the emitter pushes the shared expert too
   (`ONDV_EMIT_SHARED`).
3. **A lock was released but never acquired.** It counted up by one per layer, and
   AIE2 locks are 6-bit, so the device hung after 8 layers per context. This was the
   "8 layers per context" cap. Fix: the emitter acquires the lock back
   (`ONDV_PKTDONE_ACQ`).

### Where the time goes

A per-stage profile of one layer at 469c43b, taken with on-device timestamps, found
two things:

- The weight streams already run at 45-53 GB/s, close to a measured DDR read peak of
  about 57 GB/s.
- The losses were a staggered DeltaNet schedule, dead work in the full-attention
  layers, and the router stage.

The last three steps in the table remove those losses. The DDR bandwidth limit for a
token is about 22 tok/s (estimate).

### Compared with the GPU

On the same box, the Q8_0 GGUF of the model does:

| Backend | Decode (tg128) | Prompt (pp512) | Binary |
|---|---|---|---|
| Vulkan | 53.4 tok/s | 1269 tok/s | `llama-bench`, llama.cpp 7fe450e |
| HRX0 | 40.9 tok/s | fails: memory fault on batches of 2 or more | `llama-bench`, llama.cpp f1a0aca / hrx-system 51b1739 |

Both used `-ngl 99 -fa 1 -r 3`. The GPU is the faster path for this model. What the
NPU adds is a second, independent decode stream. At 469c43b the NPU decoded 11.0 tok/s
next to a running Vulkan benchmark without slowing it (as noise measures). It also
draws about 75-80 W less package power than GPU decode.

## What is left

- **Full ELFs and C++ (rule 4).** Move the kernels from xclbin + `insts.bin` to full
  ELFs, and the driver from Python to `npu/`, so that `1bit serve` can route the 35B
  here. Upstream found that the ELF path translated the buffer arguments twice, which
  is why this path uses the classic one.
- **Prompt processing.** Prompt tokens go through one position at a time. There is no
  batched prefill on the NPU.
- **Memory.** A session pins about 22 GB of host memory in XRT buffers.
