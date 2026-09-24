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
whole-layer design `lax` from the open kernels in `third_party/OpenFlowLM-Next`. Two
drivers run them through XRT's classic xclbin path:

- `1bit npu-lax`, the engine's C++ driver (`npu/lax*.{h,cpp}`). It needs no Python. See
  "The C++ driver".
- The Python driver in the pinned tree, which the C++ one is checked against.

The fast lane ([npu.md](npu.md)) is full-ELF, and this path does not meet that bar yet
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

## The C++ driver

`1bit npu-lax` does what the pinned tree's `model/lax_chat.py` and `model/lax_decode_cfg.py`
do, but in the engine:

- It packs every buffer from `model.q4nx` in parallel threads, straight into the XRT
  buffers. There is no pipe, no helper process and nothing on disk.
- It builds the 40-run runlist on one context. Each layer's arguments are `pool xres
  consts kv|dkv act ptab state|dstate cfg`.
- It patches the full-attention stream for each position, then runs the norm and the lm
  head and takes the greedy argmax.

The packer is `npu/lax_pack.{h,cpp}`. It is re-implemented from the recipe's
specification (`recipes/qwen36moe.py` `pack_plan` and `recipes/pack.py`, with every
projection at q4_1 as `--requant` sets it), not copied. The position patches and the
layer config words are `npu/lax_stream.{h,cpp}`. The kernel loading sits behind
`npu/lax_kernels.h`, so a full-ELF kernel set can replace the classic one without
changing the decoder.

```sh
cmake -S . -B build -DONEBIT_NPU=ON \
    -DONEBIT_NPU_LAX_MODEL=<model dir> -DONEBIT_NPU_LAX_KERNELS=~/.cache/lax/kernels \
    -DONEBIT_NPU_LAX_REF=~/.cache/lax-ref
cmake --build build
build/1bit npu-lax --model <model dir> --kernels ~/.cache/lax/kernels \
    "What is the capital of France? Answer in one sentence."
build/1bit npu-lax --model <model dir> --kernels ~/.cache/lax/kernels --parity ~/.cache/lax-ref
flock <lockfile> ctest --test-dir build -R npu_lax --output-on-failure
```

Without prompts on the command line, it reads one prompt per line from stdin, and the
conversation stays one device session. `--parity` runs the reference's `xres<t>.bin` at
position t and scores the logits with `compare_decode.py`'s metric.

The tests:

| CTest | What it checks | Needs |
|---|---|---|
| `npu_lax_host` | The q8 -> q4_1 re-quantization and the signed-nibble transcode on synthetic chunks, every chunk permutation, the 4096-row position table, the position patches on a synthetic stream and the cfg words. Each is checked against hashes of the Python packer's output for the same input. | nothing (CI) |
| `npu_lax_pack_model` | All 83 buffers (ptab, final norm, lm head pool, 40 pools and 40 consts) against the SHA-256 of the Python packer's bytes, in `tests/golden/npu_lax/sha256.tsv` | the model |
| `npu_lax_patches` | The four position patches found in `lax_a/insts.bin` against the reference harness's table (`tests/golden/npu_lax/lax_a_patches.tsv`) | the kernels |
| `npu_lax_e2e` | `tests/npu_lax_cpp.sh`: parity on three positions, then one chat turn that must answer with Paris | model, kernels, reference, the NPU |

The reference hashes come from `model/lax_pack.py`'s `Model.build` at the pin. That
path produces the bytes `lax_chat.py` streams into the device, and they equal
`make_decode.py --requant`'s consts and ptab files. The q8 re-quantization has to round
the way NumPy does, step for step:

- It is built without fused multiply-adds.
- A tied min or max goes to the later element. In a block whose scale is exactly zero,
  every value is +0 or -0, and the min's sign is stored.

**Results** on Strix Halo, with the same model and kernels as the Python numbers below.
The binary is `build/1bit` from branch `npu/lax-cpp`:

- **Packer:** all 83 buffers are byte-identical to the Python packer. That covers the 40
  layers, the lm head and the ptab (`npu_lax_pack_model`, 17 s).
- **Parity** (`1bit npu-lax --parity`): corr 0.999998 / 0.999993 / 0.999998 and argmax
  846 / 198 / 3710 at positions 0 / 1 / 2. The top-5 equals the reference's. These are
  the same numbers as the Python driver's.
- **Chat:** "The capital of France is Paris."

Speed, measured back to back on the same prompt ("Write a detailed essay about the history
of Paris, from the Romans to today.", 28 prompt tokens, 128 generated). The generated text
is the same from both drivers:

| Driver | Ready (process start to first prompt) | Prompt | Decode |
|---|---|---|---|
| `1bit npu-lax` (C++) | 3.8 s | 20.7 tok/s | 16.4 tok/s |
| `lax_chat.py` (Python) | 11.7 s | 18.7 tok/s | 16.2 tok/s |

Decode is bound by the device: 40 layers take about 48 ms as one submit, and the lm head
streams 542 MB. The C++ driver gains its time at load. It fills 22.4 GB in 0.9 s over 16
threads, after 2.8 s allocating the buffers. Both drivers were measured with
`model.q4nx` already in the page cache.

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

- **Full ELFs (rule 4).** The driver is now C++ (`1bit npu-lax`). The kernels still load
  as xclbin + `insts.bin`. A full-ELF `KernelSet` (`npu/lax_kernels.h`) would replace
  the classic one, and then `1bit serve` can route the 35B here. Upstream found that the
  ELF path translated the buffer arguments twice, which is why this path uses the
  classic one.
- **Prompt processing.** Prompt tokens go through one position at a time. There is no
  batched prefill on the NPU.
- **Memory.** A session pins about 22 GB of host memory in XRT buffers.
