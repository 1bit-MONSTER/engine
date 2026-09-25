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

It matches an fp64 reference, and it decodes at 16.3-16.5 tok/s.

**Status: experimental, and served by `1bit serve --device npu` on full ELFs**
([serve.md](serve.md)). The kernels are the merged whole-layer design `lax` from the open
kernels in `third_party/OpenFlowLM-Next`. Two drivers run them:

- The engine's C++ driver (`npu/lax*.{h,cpp}`): `1bit serve` and `1bit npu-lax`. It needs
  no Python. It loads the kernels as full ELFs (PDI + control code, no xclbin, rule 4) by
  default, or through XRT's classic xclbin path for A/B. See "The C++ driver" and "Full
  ELFs".
- The Python driver in the pinned tree, which the C++ one is checked against.

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

The kernels land in `~/.cache/lax/kernels` in the layout under "Kernel directory".
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
`npu/lax_kernels.h`: `elf_kernels` (the default, `npu/lax_elf*.{h,cpp}`) and
`classic_kernels` (`--transport classic`) run the same decoder.

```sh
cmake -S . -B build -DONEBIT_NPU=ON \
    -DONEBIT_NPU_LAX_MODEL=<model dir> -DONEBIT_NPU_LAX_KERNELS=~/.cache/lax/kernels \
    -DONEBIT_NPU_LAX_REF=~/.cache/lax-ref
cmake --build build
build/1bit npu-lax --model <model dir> --kernels ~/.cache/lax/kernels \
    "What is the capital of France? Answer in one sentence."
build/1bit npu-lax --model <model dir> --kernels ~/.cache/lax/kernels --parity ~/.cache/lax-ref
build/1bit npu-lax --model <model dir> --kernels ~/.cache/lax/kernels --bench 256 [--transport classic]
flock <lockfile> ctest --test-dir build -R npu_lax --output-on-failure
```

Without prompts on the command line, it reads one prompt per line from stdin, and the
conversation stays one device session. `--parity` runs the reference's `xres<t>.bin` at
position t and scores the logits with `compare_decode.py`'s metric, then resets the decoder
(`Decoder::reset`: the DeltaNet state back to zero) and runs the positions again, which
must give bit-identical logits; `1bit serve` resets that way between conversations.
`--bench N` runs N tokens at positions 0..N-1, twice (new, then cached position configs),
and prints the host preparation, 40-layer and head times per token.

The tests:

| CTest | What it checks | Needs |
|---|---|---|
| `npu_lax_host` | The q8 -> q4_1 re-quantization and the signed-nibble transcode on synthetic chunks, every chunk permutation, the 4096-row position table, the position patches on a synthetic stream and the cfg words. Each is checked against hashes of the Python packer's output for the same input. | nothing (CI) |
| `npu_lax_pack_model` | All 83 buffers (ptab, final norm, lm head pool, 40 pools and 40 consts) against the SHA-256 of the Python packer's bytes, in `tests/golden/npu_lax/sha256.tsv` | the model |
| `npu_lax_host` (also) | The full-ELF position sites on a synthetic instruction ELF: which words and addends move, their values at a position, the fold kept until assembly clears it, and three malformed inputs refused | nothing (CI) |
| `npu_lax_patches` | The four position patches found in `lax_a/insts.bin` against the reference harness's table (`tests/golden/npu_lax/lax_a_patches.tsv`) | the kernels |
| `npu_lax_elf` | The seven full-ELF position sites of `lax_a/insts.elf` against the investigation's (`tests/golden/npu_lax/lax_a_elf_sites.tsv`); the PDI-less position configs equal the PDI ones in every other section; with `-DONEBIT_NPU_LAX_ELF_GOLDEN=<dir>`, all seven full ELFs byte for byte against the ones the reference harness ran | the kernels |
| `npu_lax_e2e` | `tests/npu_lax_cpp.sh`: on full ELFs, parity on three positions and again after a reset, then one chat turn that must answer with Paris | model, kernels, reference, the NPU |
| `npu_lax_e2e_classic` | the same on the classic path | model, kernels, reference, the NPU |
| `npu_lax_serve` | `tests/npu_lax_serve.sh`: `1bit serve --device npu` under strace; the chat answers "The capital of France is Paris.", twice, streams, and opens no `.xclbin` | model, kernels, the NPU |

The reference hashes come from `model/lax_pack.py`'s `Model.build` at the pin. That
path produces the bytes `lax_chat.py` streams into the device, and they equal
`make_decode.py --requant`'s consts and ptab files. The q8 re-quantization has to round
the way NumPy does, step for step:

- It is built without fused multiply-adds.
- A tied min or max goes to the later element. In a block whose scale is exactly zero,
  every value is +0 or -0, and the min's sign is stored.

**Results** on Strix Halo, with the same model and kernels as the Python numbers below.
The binary is `build/1bit` from branch `npu/lax-cpp` (the classic path; the full-ELF
results are under "Full ELFs"):

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

## Kernel directory

`scripts/build-lax.sh <prefix>` writes `<prefix>/kernels`, which both transports read:

```
kernels/
  lax_l/       insts.elf  main.pdi  final.xclbin  insts.bin  final.prj/   linear-attention text
  lax_a/       insts.elf  main.pdi  final.xclbin  insts.bin  final.prj/   full-attention text
  ln/          insts.elf  main.pdi  final.xclbin  insts.bin               final RMSNorm
  lm_head_q8/  insts.elf  main.pdi  final.xclbin  insts.bin               q8 lm head
  manifest.json  spec.json  toolchain.json
```

Full ELFs read `insts.elf` and `main.pdi` (a kind without `main.pdi` falls back to
`final.prj/main.pdi`); the classic path reads `final.xclbin` and `insts.bin`. lax_l and
lax_a come from one design build, so lax_l's PDI configures the array for both. The export
of the pinned recipe copies only the classic files of ln and lm_head_q8, so the script
copies their `insts.elf` and PDI from the builds under `src/open_kernels/designs`.
`1bit serve` finds this directory through `--npu-kernels`, `<model dir>/npu/lax` or
`$ONEBIT_NPU_LAX_KERNELS`.

## Full ELFs

Each kernel is assembled in memory at load time by `npu/full_elf` (`assemble_full_elf`, the
fast lane's assembler) from the build's `insts.elf` and PDI, and opened with
`xrt::hw_context(dev, xrt::elf)` / `add_config`. `strace` shows no `.xclbin` opened.
`npu/lax_elf.h` builds them, and `npu/lax_elf_kernels.cpp` loads them:

| Kernel | From | Context |
|---|---|---|
| `laxinit` | lax_l's PDI, only `load_pdi`, one unreferenced buffer argument | the layer context is created from it |
| `lxf` | lax_l's text, no `load_pdi` | a config of the layer context |
| `axf<pos>` | lax_a's text written for position pos, no `load_pdi`, no PDI | a config of the layer context, added the first time the decode reaches pos |
| `ln`, `lm` | `load_pdi` + text, their own PDIs | one stand-alone context each |

Buffers go from argument 0: `lxf`/`axf` `pool xres consts kv act ptab state cfg` (lxf gets
the unused 8 MB kv, axf the unused 2.3 MB state), `ln` `xres zero normw xresf hn`, `lm`
`lmpool hn logits`. `cfg` words 0..1 stay the pool's address + 0x80000000.

**Why the ELF path used to fault.** mlir-aie folds 0x80000000 (`kDDRAIEAddrOffset`) into
the DDR_PATCH offset of buffer arguments >= 5, for the classic firmware, which translates
only the first five. aiebu-asm moves that offset into the `.rela.dyn` addend, and XRT's
ELF patcher adds `bo.address() + 0x80000000` for every argument. So `ptab`, `state` and
`cfg` were translated twice. `unfold()` (`npu/lax_elf.h`) clears the fold, as
`assemble_full_elf`'s bit-31 mask already did, and refreshes the UID note to the md5 of
the control text XRT runs. This was found on OpenFlowLM-Next branch `1bit/lax-elf`
(65ad44d, `35b-whole-layer-runlist-status.md`). The pin does not need that branch: the
engine unfolds the default build.

**Every token's runlist starts with the `laxinit` run.** ln and lm_head run in their own
contexts, and after them the layer context's array configuration is gone: with init once,
token 1 timed out. Heading each runlist with it costs nothing measurable.

**Positions.** On the ELF path the full-attention position is not a word the host can
rewrite: XRT patches the BD from the relocation addend. `elf_position_sites`
(`npu/lax_stream.h`) derives the seven places from `insts.elf`. It finds the same four
DDR_PATCH patches as the classic `position_patches`. For the window fill, the row drain and
the RoPE record, it takes the DDR_PATCH word and the addend of the one relocation on the
BD that patch rewrites (its symbol must name the same argument, and its addend must equal
the word). The window length is the fourth site. For the pinned build these are ctrl words
3222 (window length, `max(pos, 1) * 512`), 2762 + relocation 68 (drain, `pos * 2048`),
2984 + relocation 74 (record, `pos * 1024`) and 3240 + relocation 78 (window offset, 0).
Each position is its own config:

- Adding one takes 0.2-0.5 ms.
- The decoder prepares the next position's runlist (its config, the ten full-attention
  runs, the runlist) while the device runs the current one, so none of it is on the
  token's critical path.
- Position configs carry no PDI. They have no `load_pdi`, so XRT never uploads one, but it
  keeps every added ELF. With lax_l's 257 KiB PDI in each, 128 positions raised the
  maximum RSS by 34 MB over the classic path (about 1 GB at 4096 positions). Without it,
  512 positions stayed within 13 MB of the classic path (21.98 vs 21.96-21.97 GB).

**Results** on Strix Halo, 2026-09-24:

- Model: `model.q4nx` sha256 `688f1e153d10…3cf8de`.
- Kernels: `scripts/build-lax.sh` at the pin, with `insts.bin` md5 `136cc0c9…` / `4f3c749d…`
  and lax_a `insts.elf` md5 `426b8dd1…`.
- Binary: `build/1bit` from branch `npu/lax-elf`.

The results:

- **Assembly** (`npu_lax_elf` with the harness's ELFs): lax_init, lxf, axf_p0/1/2, ln
  and lm are byte-identical to the ones `harness/full_elf.py lax` assembled and ran.
- **Parity** (`1bit npu-lax --parity`, full ELFs): corr 0.999998 / 0.999993 / 0.999998 and
  argmax 846 / 198 / 3710 at positions 0 / 1 / 2, the same top-5 as the reference, and
  bit-identical again after a reset. The logits are byte-identical to the classic path's
  (`--dump`, `cmp`).
- **Serve** (`tests/npu_lax_serve.sh`): "The capital of France is Paris.", no `.xclbin`
  opened.

Speed, A/B in alternating runs of the same binary, host load average 1-5:

| | Full ELF | Classic |
|---|---|---|
| `--bench 256`, per token (40 layers + norm + lm head) | 59.7-60.8 ms | 61.2-62.1 ms |
| of which the 40-layer runlist | 49.2-50.0 ms | 50.6-51.3 ms |
| host preparation before the submit | 0.001-0.003 ms (plus 0.8-1.7 ms inside the runlist's time, preparing the next position) | 0.004 ms |
| Chat, "Write a detailed essay about the history of Paris, from the Romans to today.", 128 tokens: decode | **16.3 / 16.5 tok/s** | 16.0 / 16.1 tok/s |
| the same: prompt (28 tokens) | 20.8 / 20.9 tok/s | 20.4 / 20.3 tok/s |
| Ready (process start to first prompt) | 3.6-3.8 s | 3.6 s |

Both transports generate the same text. Before the lookahead, building the next
runlist after the submit cost 1.1-1.4 ms per token, as much as the ELF path saves.

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

- **Prompt processing.** Prompt tokens go through one position at a time. There is no
  batched prefill on the NPU.
- **Memory.** A session pins about 22 GB of host memory in XRT buffers.
- **Cache reuse across requests.** `1bit serve` continues the device's conversation only
  when a request's tokens extend it exactly, because the DeltaNet state cannot rewind. A
  raw prompt that does reuses the whole previous turn (31 of 49 tokens in
  `tests/npu_lax_serve.sh`). A chat client's follow-up does not: Qwen3.6's template
  renders earlier answers without the think block they were generated after, so it starts
  over (0 of 45) and re-runs its prompt at about 21 tok/s.
