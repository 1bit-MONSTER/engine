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
# HRX + Vulkan

One llama.cpp build with two GPU backends, ggml-hrx (AMD's HRX runtime) and
ggml-vulkan. Its single `llama-server` exposes both devices on Strix Halo:

```
HRX0:    AMD Radeon 8060S Graphics (Node 1) (gfx1151)
Vulkan0: AMD Radeon 8060S Graphics (RADV STRIX_HALO)
```

`1bit serve` runs GGUF models on either device with it ([serve.md](serve.md)):

| `--device` | Device | Serves |
|---|---|---|
| `hrx` | `HRX0` | AMD's ggml-hrx |
| `vulkan` (and `auto` for GGUF) | `Vulkan0` | standard GGUF quants, the fastest measured device for them |

## Pinned sources, kept current

| Submodule | Source | Pinned to |
|---|---|---|
| `third_party/llama.cpp` | [1bit-MONSTER/llama.cpp](https://github.com/1bit-MONSTER/llama.cpp), branch `1bit/hrx-vulkan-patched` | AMD's `hrx-graph-develop-v2` (ggml-hrx on ggml-org llama.cpp) plus our commits ("Our patches") |
| `third_party/hrx-system` | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) | libhrx, loomc and the Loom tools |

- **Where the pair comes from.** The two pins are the pair AMD's integration repo,
  [ROCm/ggml-staging-automation](https://github.com/ROCm/ggml-staging-automation),
  builds and tests together.
- **How it stays current.** `.github/workflows/bump-hrx.yml` runs daily. When AMD
  moves its pair, it:
  - syncs the fork (`master` to ggml-org, `1bit/hrx-vulkan` to AMD's pin);
  - rebases our commits onto AMD's pin and moves `1bit/hrx-vulkan-patched` there,
    after tagging the previous tip `patched-<sha>` so every commit the engine has
    pinned stays reachable (a rebase conflict fails the run: rebase by hand);
  - opens a PR here moving both submodules, listing the rebased commits.
- **The token it needs.** The secret `HRX_BUMP_TOKEN` is a fine-grained token with
  Contents read/write on `1bit-MONSTER/llama.cpp` and `1bit-MONSTER/engine`, and
  Pull requests read/write on `1bit-MONSTER/engine`.
- **Validation.** CI builds that PR without HRX, so run the checks below on Strix
  Halo before merging.

### Fixed: `--device hrx` answered wrongly on mid-length prompts

Measured 2026-09-24, Qwen3-0.6B Q4_K_M, greedy chat through llama-server on `HRX0`:
before the fix, prompts of 404 to 1,733 tokens came back as garbled text, while 146
tokens and 2,167 tokens or more read correctly. Prompt processing was right at every
length; decoding one token at a time went wrong once the KV cache held more than 256
tokens.

**Cause:** above 256 KV tokens the single-query flash attention kernel
(`flash_attention_decode_split_f32_f16_wmma.loom`) combines its per-block partial
results with a cooperative reducer. That reducer runs four subgroups over eight query
rows per KV head, and it checked only that a row's query head exists, not that the row
belongs to this KV head. With fewer than eight query heads per KV head (Qwen3-0.6B has
2), the spare subgroups reduced stale rows and wrote them over the next KV head's output.
The reducer used at 256 tokens and below already stays inside its own rows. The fix
(fork commit `79788e9`, on `1bit/hrx-vulkan-patched`) adds that bound to both copies of
the kernel.

After the fix, fed the same tokens as Vulkan (8 decode steps each), the relative logit
difference is 0.01 to 0.03 at 248, 250, 256, 300, 400, 1,200 and 2,040 tokens, the same
as below 256 before; and the chat answers match Vulkan's at every length from 404 to
2,844 tokens. `test-backend-ops -o FLASH_ATTN_EXT` cannot check it on this pin: 12 of
5,141 cases pass with and without the fix, because HRX0 refuses the test's empty `NONE`
graph node.

### Fixed: MoE models with more than 128 experts (Qwen3.6-35B-A3B)

Before this pin, Qwen3.6-35B-A3B (256 experts) failed on `HRX0` in two ways:
- **Prompt batches faulted.** Every batch of 2 or more tokens faulted the GPU
  (`HSA_STATUS_ERROR_MEMORY_FAULT`).
- **Decode was wrong.** Single-token decode ran at about 41 tok/s, but its output was
  wrong: KLD 15.3 against Vulkan, top-1 0%.

There were two causes:

1. **The router assumed 128 experts.** `dispatch-moe-router.cpp` wrote the expert
   partition table in the 128-expert layout (7-bit expert ids) for every expert count.
   The `mul_mat_id` kernels that read the table decode a 9-bit layout. So experts
   128-255 got no partitions, and an expert with 2 or more tokens read past its row.
   Fork commit `96049a2` picks the layout by expert count
   (`dispatch_registration/common/dispatch-moe-routing-layout.h`). Nothing changes up
   to 128 experts.
2. **Fused-only ops went to the CPU.** Our earlier claim rule accepted only nodes that
   run standalone. So the router chain, L2_NORM, SOFTPLUS, GATED_DELTA_NET and the
   per-head RMS_NORM/ROPE, which HRX runs only inside fused patterns, went to the CPU.
   The 35B then split per layer (KLD 2.63), and Qwen3-Coder-30B-A3B could not decode.
   Fork commit `9a7aad6` (`fused-context-claim.h`) claims those nodes when their fused
   pattern's producers are present and their shapes fit.

Measured on Strix Halo (llama-bench, fa on, `-r 3`; KLD against Vulkan):

| model | metric | before | this pin |
|---|---|---|---|
| Qwen3.6-35B-A3B Q8_0 | pp512 | 141 | 909 |
| Qwen3.6-35B-A3B Q8_0 | pp2048 | 156 | 1028 |
| Qwen3.6-35B-A3B Q8_0 | tg128 | 14.1 | 35.3 |
| Qwen3.6-35B-A3B Q8_0 | KLD vs Vulkan | 2.63 | 0.0046 |
| Qwen3-Coder-30B-A3B Q4_K_M | pp512 | 1114 | 2040 |
| Qwen3-Coder-30B-A3B Q4_K_M | tg128 | fails | 89.4 |
| Qwen3-0.6B Q4_K_M | pp512 / tg128 | 21938 / 322.5 | 21789 / 323.8 |

`test-backend-ops -b HRX0`: 790/790.

### Fixed: decode-split multipass gap above 2048 ([engine#124](https://github.com/1bit-MONSTER/engine/issues/124))

The multipass decode-split reducer (`reduce_completed.multipass`, used above
`key_value_token_capacity` 2048) finished with a scalar output pass: each workitem owned
one output channel and walked every KV block serially, recomputing
`expf(partial_max - maximum)` once per (block, element). Only half the workgroup is live
at `value_head_size = 128`, so decode above 2048 stayed about 16% below the <=2048
corridor. Fork PR
[1bit-MONSTER/llama.cpp#30](https://github.com/1bit-MONSTER/llama.cpp/pull/30) computes
the per-block scale once into a per-row workgroup (LDS) stage and runs one vectorised,
all-rows output pass (4 channels per workitem). The per-channel block order is
unchanged, so the reduce is bit-identical.

Measured on Strix Halo (Qwen3-Coder-30B-A3B-Instruct Q4_K_M, `-dev HRX0`,
`llama-bench -p 0 -n 8 -r 5`, median of 6 interleaved runs):

| depth | blocks | before | this pin | delta |
|---|---:|---:|---:|---:|
| 2000 (<=2048 control) | 32 | 66.06 | 68.82 | +4.2% |
| 2100 | 33 | 58.19 | 67.15 | **+15.4%** |
| 3000 | 47 | 50.84 | 60.89 | **+19.8%** |
| 4800 | 76 | 41.83 | 51.02 | **+22.0%** |

`d2100/d2000` moves 0.881 -> 0.976, so the boundary cliff is gone. 1.24 GB of
decode-path logits (`llama-perplexity -c 2049 -b 1 --save-all-logits`) are byte-identical
to the previous pin, the buried code word is exact at 4700 tokens, and no GPU faults
were observed.

### MoE decode and the kernel cache (llama.cpp #32, #33)

- **MoE decode.** With one to four tokens, a `MUL_MAT_ID` (a MoE expert matmul) runs as a GEMV over
  the routed experts' rows (`common.mul_mat_id.decode_f32_wave64`), not the batched WMMA kernel,
  which read a single token's expert at ~45 GB/s. HRX0 decode: ZAYA1-8B 46.3 -> 62.0 tok/s,
  GLM-4.7-Flash 22.1 -> 25.3; ZAYA's teacher-forced check is unchanged (69/96) and batch-1
  perplexity stays within 0.1-0.35% of the CPU backend. `GGML_HRX_DISABLE_DISPATCH=mul_mat_id.decode`
  turns it off for comparison. Found with HRX's dispatch profiler: `HRX_PROFILE_MODE=dispatch
  HRX_PROFILE_FILE=x.prof`, read with `iree-profile executable x.prof` (built from hrx-system with
  Bazel).
- **Kernel cache.** HRX compiles each kernel specialization on first use, and the specialization
  includes the batch's token count, so each new prompt-length remainder compiles a new set. The
  compiled kernels are kept in `~/.cache/1bit/hrx-jit` (`GGML_HRX_JIT_CACHE_DIR`), keyed on every
  compile input and the compiler library, so later processes skip the compiles: ZAYA1-8B's first
  5.4K-token prompt in a new process 4.46 -> 3.43 s, first short prompt 0.47 -> 0.22 s, with
  identical output. `GGML_HRX_JIT_CACHE=0` turns it off; it is off while a Loom sanitizer is set.
  Deleting the directory is always safe.

### Known issues on `HRX0`

- **Several sequences per batch fail.** `llama-perplexity` with `n_seq` > 1 stops on an
  unsupported 3-D MUL_MAT; use `-b 512`.
- **`-fa off` fails.** A SET_ROWS into the non-flash-attention V cache is rejected.
- **Decode-split flash attention is on.** `flash_attention_decode_split_next_q8` used to give
  nondeterministic attention on HRX0 ([#140](https://github.com/1bit-MONSTER/engine/issues/140)) and
  to fault Qwen3-Coder-30B-A3B, so `1bit serve --device hrx` turned it off (#148). Since llama.cpp
  `00adc2b` (see the #123 section below) it is correct, and it is as fast or faster:

  | model (Q4_K_M), tg tok/s | ctx 0 | ctx 512 | ctx 2100 |
  |---|---|---|---|
  | Qwen3-0.6B, decode-split on | 286 | 254 | 140–158 |
  | Qwen3-0.6B, off | 296–306 | 220–223 | 118–122 |
  | Qwen3-Coder-30B-A3B, on | 78–85 | 81 | 42–62 |
  | Qwen3-Coder-30B-A3B, off | 78–81 | 65–69 | 47–49 |
  | ZAYA1-8B, on | 39–41 | 40–43 | 38–42 |
  | ZAYA1-8B, off | 39–42 | 39–40 | 34–36 |

  (llama-bench `-p 0 -n 32/64`, runs interleaved on a shared strixhalo, 2026-09-27.)
  `ONEBIT_HRX_DECODE_SPLIT=0` turns it off; a `GGML_HRX_DISABLE_DISPATCH` you set yourself wins.

The prefill split below is not affected by either: HRX0 only prefills whole ubatches
there, and decoding is Vulkan's.

### Prefill on HRX, decode on Vulkan

HRX prefills faster than Vulkan and Vulkan decodes faster, on the same GPU.
`1bit serve --device vulkan --prefill-device hrx` uses both: the HRX build's
llama-server decodes on `Vulkan0`, and a second copy of the model on `HRX0` runs
the prompt prefix. The two contexts share one KV cache with no copy: HRX owns it
in its device memory and exports it as a dma-buf (HSA), Vulkan maps the same
memory, and only the KV cell metadata (positions, sequences) moves between them.

- **What runs on HRX:** the prompt prefix in whole ubatches (HRX prefill halves on
  a partial one), from `--prefill-min-tokens` tokens (default 1024; below that the
  split does not pay). The rest, and all decoding, run on Vulkan.
- **Needs:** flash attention on both sides (`serve` passes `-fa on`), no LoRA, no
  multimodal. The weights are loaded twice (once per device).
- **Layout:** the shared region is cut into chunks of at most 1 GiB, each its own
  dma-buf: Vulkan reads garbage from one buffer past about 4 GiB (a 40960-token
  Qwen3-0.6B cache), and both sides must compute the same layout.

Measured on Strix Halo, llama-server on `Vulkan0`, prefill / whole request:

| Model | Prompt | Vulkan alone | HRX prefill + Vulkan |
|---|---|---|---|
| Qwen2.5-7B Q4_K_M | 2048 | 1541 / 4314 ms | **1078** / 3890 ms (-10%) |
| Qwen2.5-7B Q4_K_M | 8192 | 7347 / 10376 ms | **4646** / 7668 ms (**-26%**) |
| Qwen3-0.6B Q4_K_M | 8192 | 1289 / 2267 ms | **1089** / 2032 ms (-10%) |

Decode on the shared KV is within 3% of Vulkan with its own KV. Against Vulkan
alone, teacher-forced over 64 tokens, the split's mean KL is 0.0002-0.0006 nats
(max 0.007) and the top token agrees on 64-65 of 65 positions; Q4_K_M against
BF16 is 0.063.

### Our patches

`1bit/hrx-vulkan` is AMD's commit unchanged; `1bit/hrx-vulkan-patched` adds:

- **Zero-copy KV sharing** (the section above): dma-buf export in ggml-hrx
  (`ggml-hrx-dmabuf.cpp`), dma-buf import in ggml-vulkan (`ggml-vulkan-dmabuf.inc`),
  `llama_kv_share_next` / `llama_kv_share_from` / `llama_kv_cells_copy` in
  llama (`llama-kv-share.cpp`), and `ONEBIT_PREFILL_DEVICE` in llama-server
  (`server-prefill.cpp`).
- **IQ3_XXS matmul on HRX.** A matrix-vector kernel (`kernels/hrx/mul_mat_vec_iq3xxs_f32.loom`)
  and its matcher (`common/dispatch-mul-mat-iq3-xxs.cpp`), ported from the
  `feat/hrx-port-ae91949` work.
- **Honest op claims.** ggml-hrx used to claim every op of a declared type, so
  ggml's scheduler handed it nodes it could not dispatch and the graph failed
  (`unsupported HRX node`) with no CPU fallback. It now claims a node only when
  its dispatcher can execute it, except nodes already in HRX memory (KV cache
  views), which cannot move. Leaf (`NONE`) nodes count as covered. AMD's IQ4_NL
  and IQ4_XS matmul kernels, and GET_ROWS for IQ4_XS and batched IQ3_S, give wrong
  values, so those nodes are left to the CPU.
- **Kernels for ops HRX sent to the CPU** ([llama.cpp #24](https://github.com/1bit-MONSTER/llama.cpp/pull/24)):
  a batched F16 matmul (`ggml_grouped_mul_mat_f16_f32`, e.g. ZAYA's grouped convolution, whose
  weights the loader can now place on HRX), and short-row kernels in `small_rows_f32.loom`:
  SOFT_MAX without a mask, SUM_ROWS, ARGSORT, GET_ROWS for narrow rows (strided ids, as top-k views
  are), CONT of strided views and broadcast-only REPEAT. Each matcher claims only what the existing
  kernels do not. `test-backend-ops -b HRX0` passes every case of these ops. ZAYA1-8B's decode graph
  goes from 641 graph splits to 1: 25.5 to 47.9 tok/s (tg128).

Measured on Strix Halo (Qwen3-0.6B, perplexity over 8 x 512 wikitext tokens):

| File | Vulkan0 | HRX0 on AMD's commit | HRX0 with our patches |
|---|---|---|---|
| Q4_K_M | 22.53 | 22.52 | 22.52, same speed (21691 / 324 tok/s) |
| UD-Q4_K_XL | 22.43 | fails | **22.41** |
| UD-Q2_K_XL | 36.26 | fails | **36.46** |
| UD-IQ2_M | 55.44 | fails | **55.58** |

`test-backend-ops -b HRX0`: 791 OK, 0 failed (on AMD's commit, about 1150 fail).
Correct is not fast: the UD files run their IQ4_XS and sub-4-bit layers on the CPU
(UD-Q4_K_XL 6303 / 246 tok/s, UD-Q2_K_XL 974 / 76), so they belong on `Vulkan0`.

`1bit-MONSTER/llama.cpp` also holds
`1bit/hrx2-archive`, the previous build (AMD's abandoned ggml-hrx2 plus our Q4NX
kernels). It is kept for a later port of that work to ggml-hrx. Its zaya
architecture has been ported to this branch (ZAYA1, [vulkan.md](vulkan.md#zaya1-zyphra-from-our-llamacpp)).

The Vulkan route has its own upstream llama.cpp pin (latest release), so new
architectures do not wait for AMD's pair: see [vulkan.md](vulkan.md). This build
still has a Vulkan backend, which `--device vulkan` uses when `ONEBIT_VULKAN` is
off.

### The GPU's matrix units (WMMA)

The Radeon 8060S (gfx1151, RDNA 3.5) has RDNA3's six matrix instructions, all on 16×16×16 tiles.
The rates come from AMD's
[matrix instruction calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) (commit
`2ef9189`), for example
`uv run --with tabulate python3 matrix_calculator.py -a rdna3 -i v_wmma_i32_16x16x16_iu4 -d -w 32`:

| Instruction | Inputs | Per WGP per clock | Cycles |
|---|---|---|---|
| `v_wmma_f32_16x16x16_f16` / `_bf16` | f16 / bf16, f32 accumulate | 1024 FLOPs | 32 |
| `v_wmma_f16_16x16x16_f16`, `v_wmma_bf16_16x16x16_bf16` | same, 16-bit accumulate | 1024 FLOPs | 32 |
| `v_wmma_i32_16x16x16_iu8` | int8 × int8 | 1024 ops | 32 |
| `v_wmma_i32_16x16x16_iu4` | int4 × int4 | 2048 ops | 16 |

- int8 runs at the f16 rate. Converting activations to int8 saves memory traffic, not matrix time.
  Only int4 × int4 doubles the rate.
- WMMA does not co-execute with other vector instructions.
- In wave32, A and B are held twice, in lanes 0–15 and again in 16–31. `-A -R` (or `-B`, `-C`,
  `-D`) prints the register and lane of every element.
- This box reports 40 compute units (20 WGPs) at up to 2,900 MHz (`rocminfo`). That gives a
  theoretical peak of about 59 TFLOPS for f16 and 119 TOPS for int4. These are computed from the
  table, not measured. They bound prefill; decode is limited by memory bandwidth.

HRX's WMMA kernels (e.g. `flash_attention_decode_split_f32_f16_wmma`) load and store fragments
through the kernel compiler, which places them in these layouts. The fragment sizes match the
table: 16 f16 inputs and 4 f32 results per lane in wave64.

## Build

```bash
git submodule update --init --depth 1 third_party/hrx-system third_party/llama.cpp
cmake -B build -G Ninja -DONEBIT_HRX=ON          # needs TheRock at /opt/rocm-therock (ONEBIT_HRX_TOOLCHAIN)
cmake --build build --target onebit
```

- **One external project.** llama.cpp's ggml-hrx builds hrx-system itself
  (`HRX_SOURCE_DIR`) with TheRock's `amdclang`.
- **`IREE_ROCM_PATH` is not passed.** It would switch hrx-system to "package"
  mode, which needs TheRock's aqlprofile-sdk headers, and our `/opt/rocm-therock`
  does not ship them. Left unset, hrx-system fetches its pinned HSA/AQL headers.
- **The HSA runtime.** HRX dlopens it at run time, and the distro's
  `libhsa-runtime64` rejects the `HSA_AMD_AGENT_INFO_PM4_EMULATION` probe on
  gfx1151, so HRX then registers no device. CMake finds TheRock's
  `libhsa-runtime64.so.1` (`ONEBIT_HRX_LIBHSA`), and `1bit serve` passes it to
  llama-server as `IREE_HAL_AMDGPU_LIBHSA_PATH`, unless it is already set. Without a
  build path it searches `/opt/rocm-therock`. To run
  llama-server or llama-bench by hand, export that variable yourself.

## Verified (2026-09-23, llama.cpp `f1a0aca`, hrx-system `51b1739`)

`ctest` passed, re-run after the first daily bump (#13). At the time the check
was `tests/hrx_lemonade_e2e.sh`, through the engine's former embedded Lemonade:
`unsloth/Qwen3-0.6B-GGUF:Q4_0` answered "Paris" on `Vulkan0` and on `HRX0`. The
same build now passes `tests/serve_e2e.sh` on both devices through `1bit serve`.

llama-bench, Qwen3-0.6B, pp512 / tg128 tok/s, against the previous build (April
llama.cpp with ggml-hrx2 on `HRX20`):

| Device, file | previous build | this build |
|---|---|---|
| Vulkan0, Q4_K_M | 10627 / 306 | **14097 / 349** |
| Vulkan0, UD-Q4_K_XL | 10668 / ~300 | 13943 / 324 |
| HRX, Q4_K_M | 1046 / 68 | **21990 / 314** |
| HRX, UD-Q4_K_XL | 837 / 83 | 17906 / 293 |
| Vulkan0, BF16 | n/a / 115 | 4504 / 144 |
| HRX, BF16 | n/a / 40 | 7646 / 101 |

HRX prefill is now faster than Vulkan's.

## MoE router expert-id fault fix ([engine#123](https://github.com/1bit-MONSTER/engine/issues/123))

`llama_decode` could die with `HSA_STATUS_ERROR_MEMORY_FAULT` (`ret = -3`) on MoE
models (`Qwen3-Coder-30B-A3B`). It was reported against the decode-split
multipass path, but it is not in flash attention: the fault is a wild weight
read in `qwen3_moe_routed_gate_up_swiglu_q4k_q8`.

`router_top8_f32.loom` seeded each lane's argmax with `best_id = 0x7FFFFFFF` and
only replaced it on an ordered `ogt` or the equal-value tie-break. A lane whose
candidate logits are unordered (NaN) or below `-FLT_MAX` therefore published the
sentinel verbatim into `route_ids`; consumers treat the id as bounded via
`index.assume` (a hint, not a check) and compute `expert * expert_stride` in 32
bits, so `0x7FFFFFFF * 884736 mod 2^32 = 0xFFF28000` (~4.29 GB) walked off the
end of the weights buffer. The fix seeds the argmax with the lane's own first
expert id. Upstream PR: [1bit-MONSTER/llama.cpp#25](https://github.com/1bit-MONSTER/llama.cpp/pull/25),
merged as `895d63f`, which the engine pins.

Measured on `HRX0` (`Qwen3-Coder-30B-A3B-Instruct-Q4_K_M`, `llama-bench -p 0 -n 8`,
quiet box) with the fault made observable by aligning the three decode-split
*partial* transients to 256 B instead of 4096 B: pre-fix `-d 2100` 5/5 faults,
post-fix `-d 2100` 0/5, `-d 3000`/`-d 4800` 0/3, `<=2048` path 0/3, and greedy
output identical to the CPU backend. The alignment change is a repro stress knob
and is **not** part of the fix.

What the fix does not cover: the fault needs router logits that are NaN, and
the fix routes such a lane to a valid expert instead of faulting. Since
[llama.cpp #26](https://github.com/1bit-MONSTER/llama.cpp/pull/26) that case is loud: a router row
with no ordered logit publishes NaN, and llama-server refuses to sample NaN logits (it logs
"HRX returned NaN logits" and aborts) instead of decoding a wrong but plausible token. On the
author's repro 6 of 13 samples used to decode `' Paris???…'` silently; they now abort. Healthy
runs are unchanged (Qwen3-Coder-30B and ZAYA1-8B perplexity on HRX0 identical to before).
`GGML_HRX_FA_PARTIAL_ALIGN` (default 4096) sets the decode-split partials' alignment, a test knob
that reproduces the fault's layout; the investigation's artifacts are archived privately.
**Root cause of the NaNs ([engine#123](https://github.com/1bit-MONSTER/engine/issues/123),
[#140](https://github.com/1bit-MONSTER/engine/issues/140)), fixed in llama.cpp `00adc2b`**
([#28](https://github.com/1bit-MONSTER/llama.cpp/pull/28),
[#29](https://github.com/1bit-MONSTER/llama.cpp/pull/29)). In every decode-split `reduce_fused`
variant the last workgroup writes the attention output to global memory, and `pack_completed_q8`
then reads it back with other waves to build `next_q8`. The barrier between the two was
`kernel.barrier<workgroup>`, which fences LDS only (no store wait, no `buffer_gl0_inv`), so a pack
wave could quantize the output slot's previous contents. The f32 output was right and `next_q8`
was wrong, and NaN when the stale bytes decoded as NaN. The window opens when a dispatch is
preempted, and any process's KFD queue eviction (page compaction, KSM, memory pressure) does that,
which is why the rate followed those settings. It was never the cause. The fix fences both memory
spaces (`barrier<global>` then `barrier<workgroup>`). With a separate process forcing queue
evictions: Qwen3-Coder-30B (2113-token prompt, partial alignment 256) went from 1 of 8 correct to
8 of 8 with no NaN, and greedy Qwen3-0.6B from 13-34 divergent per ~177 requests to none.

## Not yet

- **Fast sub-4-bit and IQ4 kernels on HRX.** With our patches the UD files are
  correct on `HRX0` but slow, since IQ4_XS, IQ2 and IQ3_S matmuls fall back to the
  CPU. AMD's IQ4_XS / IQ4_NL kernels need fixing upstream.
- **Q4NX on HRX.** Our Q4NX kernels lived in ggml-hrx2 and are not in ggml-hrx.
  They are kept on `1bit/hrx2-archive` until they are ported.
- **First-request cost.** Lemonade's telemetry for the first short HRX chat shows
  18 tok/s against llama-bench's 303. It is probably first-use kernel
  compilation, not steady state, but this is not yet separated.
- **CI.** CI has no AMD GPU and no TheRock, so `ONEBIT_HRX` is exercised on Strix
  Halo only.
