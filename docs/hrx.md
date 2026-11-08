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
# HRX on the Radeon iGPU

HRX runs every graph node of every model and mode we test on the Strix Halo GPU: 43 of 43
configurations, with 0 nodes on the CPU. That is the 2026-10-20 census (#476, #480, #481). HRX is
AMD's ggml-hrx with its Loom kernels, plus our HIP kernels. Qwen3.8-27B with MTP decodes at 36.0 / 26.8 / 29.9 tok/s on code / prose /
short chats (#487), and ZAYA1-8B at about 90 tok/s. HRX is the engine's one GPU route.

It is one llama.cpp build with one GPU backend, ggml-hrx, and the CPU: the engine is pure HRX with HIP kernels; no Vulkan, ROCm backend, Metal, ZINC or CUDA (RFC #213 stage 3, the core strip). What HRX does not run, Lemonade serves with its own
backends ([lemonade.md](lemonade.md)). Its `llama-server` exposes on Strix Halo:

```
HRX0:    AMD Radeon 8060S Graphics (Node 1) (gfx1151)
```

`1bit serve` runs GGUF models with it ([serve.md](serve.md)):

| `--device` | Device | Serves |
|---|---|---|
| `hrx` (and `auto` for GGUF) | `HRX0` | AMD's ggml-hrx: the engine's GPU route |
| `cpu` | the CPU | the same llama-server with no GPU layers |

On `hrx` the whole graph runs on HRX0 or the model does not load. Since the HRX census closed on
2026-10-20 (43 of 43 runnable cells, 0 nodes off HRX in every graph), serve runs every HRX backend
with `GGML_HRX_STRICT=1`: a node HRX would leave on the CPU stops the load (or the request whose
graph has it) with the node's name, op, type and shape, instead of running slower in silence.
`--no-strict-device` (or `ONEBIT_STRICT_DEVICE=0`) restores the CPU fallback
([serve.md](serve.md#off-hrx-placement-strict-by-default)), and the [hardware gate](hw-gate.md)
fails a pin that leaves a gate model's node off HRX.

## Pinned sources, kept current

| Submodule | Source | Pinned to |
|---|---|---|
| `third_party/llama.cpp` | [1bit-MONSTER/llama.cpp](https://github.com/1bit-MONSTER/llama.cpp), branch `1bit/hrx-vulkan-patched` | AMD's `hrx-graph-develop-v2` (ggml-hrx on ggml-org llama.cpp) plus our commits ("Our patches") |
| `third_party/hrx-system` | [1bit-MONSTER/hrx-system](https://github.com/1bit-MONSTER/hrx-system), pinned on branch `1bit/e3e6a976-pin-6` (the commit `config/hrx-system-hold.config` names; `1bit/main` is where the unheld bump merges AMD) | [ROCm/hrx-system](https://github.com/ROCm/hrx-system) (libhrx, loomc and the Loom tools) at AMD's pin plus our commits ("Our hrx-system commits") |
| private HIP kernels | a private kernel add-on (not a submodule: a build points at a checkout with `-DONEBIT_GPU_PRIVATE`) | `4f586cac` — the commit the release builds and the hardware gate were run at |

- **Our hrx-system commits** (since 2026-10-18 on branch `1bit/e3e6a976-pin-6` of our fork, on
  AMD's `e3e6a9766ccb`): two libhrx device knobs for diagnosis, `HRX_AQL_BLOCK_SIZE` and
  `HRX_COMMAND_BUFFER_MODE` (defaults unchanged); engine#409 (libhrx bounds-checks a length-less ELF
  extent instead of skipping the check); engine#410 (`libhip_intercept` covers every HIP symbol the
  backend exports, init is idempotent); engine#401 (libhsa no longer opens `/dev/accel` in GPU-only
  processes); #6 (`generate_stubs.py` takes signatures from `api.h`, so `--check` is trustworthy;
  regenerated `stubs_generated.c`); #8 (engine#315: `HRX_ALLOW_NO_SVM=1` lets the amdgpu HAL open a ROCr
  that reports no SVM, which `1bit serve` uses to put the CWSR save area in GTT on a CWSR-on box;
  unset, nothing changes); and our fix for the long-context attention miscompile
  ([ROCm/hrx-system#1535](https://github.com/ROCm/hrx-system/issues/1535)): AMD's newer Loom no longer
  reaches the defect on our prompts, but its multi-unit live-out guard is still in
  `unit_liveness_builder.c`, so we keep removing it. The GFX11 wave64 lane-mask backport (#313) is
  no longer ours: AMD carries its own (`planning/mask_write_wait`). The pin itself is a `-s ours`
  merge of the previous pin `3319e536` (so every commit the engine shipped stays reachable, with the
  tree of the tested commit `fc2213a1c`) followed by #6. Rebase these by hand when AMD's pair moves.
- **The pin is off `1bit/main`.** `.gitmodules` names `1bit/main` for hrx-system, but the pin
  (`82804197`) is on `1bit/e3e6a976-pin-6` (and `1bit/315-allow-no-svm`, the same commit), not on
  `1bit/main`. The two have diverged (checked 2026-10-20 with GitHub's compare API): `1bit/main`
  (`c9656af5`) has 6 commits the pin does not, the merges of AMD's `40a1a36c` and `c0b135a7`
  (the newer hrx-system the hold keeps out) and the engine#409, #410, #401 and #6 fixes, which the
  pin line carries as its own replayed commits. So `1bit/main` cannot be fast-forwarded to the pin;
  moving it there means resetting the branch (a force push), which drops those two AMD merges from
  it. Until that is decided, the pin stays off-branch on purpose. What it costs: a bump PR that moves
  hrx-system fails the release's on-branch check (`pins_on_branch` in `scripts/weekly.sh`), and a
  shallow submodule clone that follows `1bit/main` cannot fetch the pin, so the release clones it
  in full (`weekly.sh` retries the submodule update without `--depth 1`). The pin stays reachable
  through the `1bit/e3e6a976-pin-6` branch, which must not be deleted.
- **Where the pair comes from.** The two pins are the pair AMD's integration repo,
  [ROCm/ggml-staging-automation](https://github.com/ROCm/ggml-staging-automation),
  builds and tests together.
- **How it stays current.** `.github/workflows/bump-hrx.yml` runs daily. When AMD
  moves its pair, it (through `scripts/bump-hrx-sync.sh`, which also runs locally with
  `DRY_RUN=1` and pushes nothing):
  - syncs the fork (`master` to ggml-org, `1bit/hrx-vulkan` to AMD's pin). When AMD
    rewrites its history, the old `1bit/hrx-vulkan` tip is tagged `hrx-vulkan-<sha>` first,
    and the run fails if that branch held a commit that is on no AMD branch;
  - merges AMD's pin into `1bit/hrx-vulkan-patched`, after tagging the previous tip
    `patched-<sha>` so every commit the engine has pinned stays reachable. Conflicts under
    `ggml/src/ggml-hrx` keep our side; a conflict anywhere else fails the run and names the files;
  - merges AMD's hrx-system pin into our `1bit/main` the same way;
  - opens a PR here moving the submodules, listing the llama.cpp commits new to the engine.
- **The hold.** While [`config/hrx-system-hold.config`](../config/hrx-system-hold.config)
  names a commit, the run does not merge AMD's hrx-system, keeps `third_party/hrx-system`
  where it is, and takes only AMD's llama.cpp changes outside the HRX backend paths
  (`ggml/src/ggml-hrx`, `ggml/include/ggml-hrx.h`, `tests/*hrx*`), because AMD's ggml-hrx is
  written against AMD's hrx-system. Those changes land as an ordinary commit, not a merge, so
  AMD's HRX changes still arrive in full once the hold is lifted. It reports the hold in the run
  summary and succeeds. From 2026-10-10 hrx-system was held at `98d05d94` while AMD's newer
  hrx-system miscompiled long-context attention
  ([ROCm/hrx-system#1535](https://github.com/ROCm/hrx-system/issues/1535)). On 2026-10-18 the
  engine moved by hand to AMD's `e3e6a9766ccb` with our commits and our #1535 fix (above), after
  test-backend-ops 1141/1141, the long-prompt needles, KLD 0 against the old pin on two models and
  the hardware gate; the hold now names that pin, so later AMD hrx-system moves stay manual too.
- **The backport line (2026-10-16, superseded 2026-10-18).** The hold named `1bit/held-98d05d94-3fixes`, a branch off
  that same held commit carrying three fixes that have nothing to do with the AMD-core move:
  engine#409 (libhrx bounds-checks a length-less ELF extent instead of skipping the check),
  engine#410 (`libhip_intercept` covers every HIP symbol the backend exports; the documented
  interceptor variable works and init is idempotent) and engine#401 (libhsa no longer opens
  `/dev/accel` in GPU-only processes), plus the +61/−0 `api.h` forward-port the regenerated
  passthrough stubs need. Nothing under the held AMD core changes, so the #1535 question is
  untouched. Those fixes are now replayed on the e3e6a976 line above, and this branch is history.
- **What guards a pair that no longer compiles.** AMD removes public LoomC symbols without notice
  (2026-10-07: Loom `32d0c76a` dropped `loomc_amdgpu_runtime_global_flags_t` and
  `loomc_amdgpu_emit_options_t`, which our `ggml/src/ggml-hrx/loom-jit.cpp` used), and CI builds
  without HRX, so #353, #357 and #361 once merged a pair that did not compile with `ONEBIT_HRX=ON`
  and #359 / #363 had to pin hrx-system back by hand. `bump-hrx.yml` now holds hrx-system itself when
  any `loomc_*` / `LOOMC_*` identifier our backend uses is missing from the new revision's
  `loom/binding/c/include`, and moves llama.cpp only. The hand move of 2026-10-18 (#441) was validated
  the same way at the pinned pair: `onebit`, `llama_hrx`, `test-backend-ops`, `llama-server`,
  `llama-bench`, `llama-perplexity`, `libamdhip64` and `libhip_intercept` all build and
  `generate_stubs.py --check` passes against the built `libamdhip64.so`. `hrx-backend-test` builds at
  the pinned pair again since f2099e9b deleted the stale `dispatch-symmetric-i4.h` include that broke it
  (#422, closed 2026-10-18), but **nothing runs it yet**: it carries no ctest label (the fork's
  `build-*.yml` workflows run `ctest -L main`) and `hw-gate.sh`'s target list is `llama-server test-backend-ops
  llama-perplexity`, so its cases only execute when someone builds and runs the binary by hand. Until one
  of those two lists carries it, a case added there is correct and dark.
- **The token it needs.** The secret `HRX_BUMP_TOKEN` is a fine-grained token with
  Contents read/write on `1bit-MONSTER/llama.cpp` and `1bit-MONSTER/engine`, and
  Pull requests read/write on `1bit-MONSTER/engine`.
- **Validation.** CI builds that PR without HRX. The HRX hardware gate
  ([hw-gate.md](hw-gate.md)) runs on every bump-hrx PR on Strix Halo: test-backend-ops on
  HRX0, known-answer and drift checks on three models, KLD against the release pin and the
  4.7K-token needle. Its check (`hw_gate`) must pass before merging. Before merging, also run the
  checks below on Strix Halo. The long-prompt result is pin-specific: run the probe against
  the proposed `hrx-system` pin, and update
  [`config/hrx-long-prompt-validated-pin.config`](../config/hrx-long-prompt-validated-pin.config)
  to that full commit only after it passes. CI rejects a changed HRX pin that does not
  match this record.

### Long-prompt HRX pin check

Before recording a new `hrx-system` pin, build and run `llama-server` for the
proposed llama.cpp/hrx-system pair on Strix Halo (with context size 8,192 and HRX0
selected), then run:

```sh
python3 tests/hrx_long_prompt.py
```

The probe tokenizes a 320-record prompt and requires at least 4,700 tokens before
asking for the code buried in record 203. It passes only when Qwen3-0.6B returns
`7341` exactly. The test endpoint defaults to `http://127.0.0.1:8080`; use
`--url` or `--model` if the server uses different settings.

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
- **Fused residual scale (llama.cpp #35).** ZAYA scales both residual branches twice per layer,
  `(x + bx) * sx + (r + br) * sr`, which ran as up to nine ADD / MUL dispatches per layer. Each side
  now runs as one dispatch (`common.res_scale_pair.*`, `ops/res_scale_pair_f32.loom`): 3,366 -> 3,165
  dispatches per token, bitwise the same output, ZAYA1-8B HRX0 decode 67.0 -> ~69.6 tok/s and
  pp512 2,109 -> 2,183. `GGML_HRX_DISABLE_DISPATCH=res_scale_pair` turns it off. At decode most of
  a ZAYA token is small dispatches and the ~1.8 us gaps between them, not matmuls; the profiler
  (`HRX_PROFILE_MODE=dispatch`) shows which ones. Profile through `llama-server`: `llama-bench`
  under the profiler fails.
- **ZAYA decode fusions (llama.cpp #36-#40).** ZAYA1-8B HRX0 decode 67 -> ~90 tok/s (Vulkan 93),
  each step measured in interleaved A/B runs:
  - PR #36 skips CONTs of fresh contiguous results in `src/models/zaya.cpp` (views keep theirs: on HRX
    their consumers take slower strided paths), bitwise the same output: +2.4%.
  - PR #39 runs the CCA convolution (Q/K concat, conv-state update, depthwise conv, grouped 2-tap conv
    and biases, ~35 graph nodes) as one dispatch (`zaya.cca_conv.decode_f32`): 72 -> 81 tok/s.
  - PR #40 runs the CCA query/key mixing and per-head norms (~30 nodes) as one dispatch
    (`zaya.cca_qk_norm.decode_f32`): 81 -> 90 tok/s.
  PRs #39 and #40 accumulate in their own order, so output is not bitwise the same; teacher-forced top-1
  against the FP32 model stays 69/96 and batch-1 perplexity moves 33.31 -> 33.49 -> 33.45 (CPU backend
  34.30). Both match decode only (one token, one sequence). #37 and #38 remove runtime integer
  division from #35's and #32's kernels, which newer Loom rejects. Each new `.loom` file carries a
  reference kernel and `check.case` differentials that pass under `iree-test-loom --sanitizer=access`.
  `GGML_HRX_DISABLE_DISPATCH=zaya.cca` turns the two ZAYA kernels off for comparison.

### Ternary Bonsai (PrismML's Hadamard-folded GGUFs)

PrismML's [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) is Qwen3.8-27B
trained to weights of -1, 0 and +1, with one fp16 scale per 128 weights. It runs on `HRX0`,
straight from PrismML's file:

```sh
1bit serve -m Ternary-Bonsai-2-27B-PTQ1_0.gguf        # --device auto picks hrx for this file
```

- **The weights** ([llama.cpp #62](https://github.com/1bit-MONSTER/llama.cpp/pull/62)). PrismML's
  PQ2_0 (ggml type 142) and PTQ1_0 (143) are native types in our HRX build, with the layouts of
  PrismML's llama.cpp fork. Our CPU decode of every tensor of both 27B files matches PrismML's own
  build bit for bit. On `HRX0` the K-quant decode kernels read them directly, so one copy of the
  weights is resident:

  | Ternary-Bonsai-2-27B | weights | GPU peak over idle | tg128 (tok/s) | pp512 (tok/s) |
  |---|---|---|---|---|
  | Q4_0 + packed ternary (below) | 14.13 GiB | 21.8 GiB | 15.3 | 13.1 |
  | PQ2_0 | 6.70 GiB | 10.1 GiB | 15.8 | 14.2 |
  | PTQ1_0 | 5.53 GiB | 8.9 GiB | 14.3 | 14.2 |

  Balanced power mode, llama-bench `-fa 1`, medians of 3. Against the CPU logits of the
  bit-identical Q4_0 copy (3 x 512) all three give the same KLD, 0.000129 at decode (`-ub 8`) and
  0.002051 at `-ub 512`. Files in these types run on `--device hrx` only, with or without
  `prism.hadamard` keys (Ternary-Bonsai-1.7B has none). For another route,
  `tools/ternary_to_q4_0.py` writes the same weights as Q4_0 (q = trit + 8 with the group's own fp16
  scale in each of its four blocks); every group is decoded back and compared before the file is
  kept, and `tests/ternary_to_q4_0_test.py` checks it against PrismML's own encoder.
- **Packed ternary decode** ([llama.cpp #54](https://github.com/1bit-MONSTER/llama.cpp/pull/54)).
  For a file stamped `onebit.ternary_q4_0`, `1bit serve` sets `GGML_HRX_TERNARY_Q4_0=1`. HRX0's
  K-quant decode kernels then read those Q4_0 weights from a 2-bit copy made at load (68 bytes per
  256 values: the two group scales and the trits), 2.125 bits per weight instead of 4.5. The load
  checks that every block is ternary and refuses the model otherwise. Prompt batches still use the
  Q4_0 weights, so both copies stay resident: about +4 GiB of GPU memory for Bonsai-2-27B (21.3 GiB
  peak over idle instead of 17.3), roughly 30% of the file. Bonsai-2-27B decode goes from 9.3 to
  15.4 tok/s (balanced power mode); prompt speed is unchanged. Against CPU logits the packed path is
  closer than the Q4_0 one (KLD 0.00013 against 0.0019). `GGML_HRX_TERNARY_Q4_0=0` turns it off.
- **The rotation.** Bonsai stores almost every matmul weight rotated by a 1024-point
  Walsh-Hadamard transform with fixed signs (`prism.hadamard.*` keys), and the activations have to
  be rotated to match. Our llama.cpp does that (`src/llama-hadamard.{h,cpp}`, llama.cpp #34,
  following PrismML's implementation): a plain F32 matmul against the rotation matrix before each
  folded weight, the inverse after the token-embedding lookup, and a check on the first graph that
  no folded weight is used without it. Upstream llama.cpp would load the file and answer garbage,
  so `1bit serve` sends a `prism.hadamard` file, or one in PrismML's own types, to `hrx` only
  (`tests/prism_route.sh`).
- **The rotation on the GPU** ([llama.cpp #64](https://github.com/1bit-MONSTER/llama.cpp/pull/64)).
  The rotation is a MUL_MAT that ggml hints as `GGML_HINT_SRC0_IS_HADAMARD`. The dense HRX matmul
  takes at most 2048 rows, so at a 512-token prompt the 17408-wide FFN input (8704 rows of 1024)
  ran on the CPU in every layer. `hadamard_f32.loom` now computes the normalized transform per row
  (log2(n) butterfly stages, the matrix never read), as the CPU, Vulkan, CUDA and Metal backends do
  for this hint. Ternary-Bonsai-2-27B (balanced mode): pp512 13.5 -> 90.9 tok/s, and decode
  15.8 -> 19.0 tok/s, since the decode rotations no longer go through a dense WMMA kernel.
  `tests/test-hrx-hadamard` checks six shapes against the CPU and an exact product, four times each.
- **Checked.** Against the F16 model on the CPU in PrismML's own fork, wikitext-2 running
  perplexity on `HRX0` is 7.962 / 10.408 / 10.171 at chunks 5 / 10 / 15 (CPU: 7.958 / 10.382 /
  10.142), the difference Q4_0's 8-bit activations make. Files without the keys build the same graph
  as before.

### Prompt matmuls on the q8_1 x4 kernel (llama.cpp #55)

On Qwen3.8-27B UD-Q4_K_XL, 94% of HRX prompt time went to the generic F32 WMMA matmuls,
which work in 32-token tiles. An iree-profile dispatch trace of a 1.8K-token prompt split it
as `mul_mat_swiglu` 51%, `mul_mat_add` 25% and `mul_mat` 18%. HRX already has a q8_1 x4
prefill kernel, but two routing rules kept this model's projections, mostly Q5_K and IQ4_XS,
away from it:

1. Its matcher took Q5_K and IQ4_XS weights only when q8_1 activations already existed, and
   never for a matmul feeding a GLU. Q4_K was allowed both. Q5_K and IQ4_XS now get the Q4_K
   policy.
2. The generic fused `mul_mat_swiglu` (priority 290) outranks the q8_1 x4 matcher (285) and
   claimed every FFN gate/up projection. For prompt chunks of 256-2048 tokens in multiples of
   256, whose gate and up are Q4_K, Q5_K or IQ4_XS, it now declines. The two projections
   then take the q8_1 x4 kernel and the GLU runs as its own op.

`GGML_HRX_Q8_PREFILL_RELAX=0` restores the old routing.

Measured on Strix Halo, llama.cpp `6e42b51`, Qwen3.8-27B UD-Q4_K_XL, 2026-10-01:

| metric | old routing | this pin |
|---|---|---|
| pp512 (llama-bench, `-b 512 -ub 512`) | 98.5 | **334.6** |
| pp2048 | 98.1 | **309.5** |
| `1bit serve --device hrx`, 14,435-token prompt | 90.7 tok/s | **264.7 tok/s** (same text) |
| KLD vs BF16 logits (wikitext-2, 20 x 512) | 0.00717 | 0.00712 |
| same top token | 96.13% | 96.27% |

`test-backend-ops -o MUL_MAT -b HRX0`: 287/287. Decode does not change, since both rules apply
only to chunks of at least 256 tokens. Since
[llama.cpp #61](https://github.com/1bit-MONSTER/llama.cpp/pull/61), a Q5_K or IQ4_XS chunk that is not a
multiple of 256 sends its 256-aligned head to the q8_1 x4 kernel too, and only the tail goes to the
generic kernel. Q4_K remainders still go to the generic kernel: its packed Row64 weight layout cannot
also be resident raw. Measured against `6e42b51` (3 interleaved runs, `-b 512 -ub 512`):

| metric | `6e42b51` | `6e42b51` + #61 (`a36d547`) |
|---|---|---|
| pp400 | 67.4 | **103.4** |
| pp1862 | 172.1 | **234.6** |
| KLD vs `6e42b51` (20 chunks of 400 = 256 + 144 tokens) | | 0.00215, same top token 97.19% |

`test-backend-ops -o MUL_MAT -b HRX0`: 287/287. `GGML_HRX_Q8_PREFILL_RELAX=0` turns this off too.

### Flash-Next prefill: where the 287 tok/s goes ([engine#572](https://github.com/1bit-MONSTER/engine/issues/572))

Qwen3.8-Flash-Next (`qwen4exp A3B`, 512 experts, 176.9 B parameters, UD-IQ4_XS 93.7 GB) prefills at
287 tok/s on Strix Halo where another stack reports 1,679 for the same class of model. The ubatch cap
was one of the limiters and is now lifted (what it bought is measured below); the expert GEMM's
tile width is not the lever.

The ubatch cap is gone. An ubatch above 512 tokens used to fail the whole graph with
`GGML_STATUS_FAILED` (engine#389); the token bounds in
`ggml/src/ggml-hrx/ubatch-token-limits.h` are what removed it — `kSsmConvRollbackMaxTokens` 512
chained by `dispatch-ssm-conv-chunked.cpp`, `kSetRowsKernelMaxTokens` 2048 by
`dispatch-row-chunks.cpp`, `kUbatchMaxTokens` 65536. Lifting the cap did pay: `-ub 2048` is
**2.03x** `-ub 512` at pp4096 on the product path (205.05 vs 101.05 tok/s, same library, same
quant, ctx 16384, `-fa on --cache-reuse 0`, 2026-11-05 16:06 and 17:11, `~/lb/supreme/raw/
llamasrv-q31-prep/` and `...-ub512/`), so the cap was one of the limiters and not a detail. It is
still not the whole gap.

Measured on Strix Halo, engine `0d0c846` + llama.cpp `8ee20a0`, `-ngl 99 -fa 1`, 2026-10-27T02:38Z,
3 runs (`samples_ts` 287.384 / 286.933 / 285.578):

| metric | tok/s |
|---|---|
| pp512 | 286.63 ± 0.94 |
| pp4096 | 260.59 ± 4.51 |
| p1024, `-ub 576` | 269.34 |

`HRX_PROFILE_MODE=dispatch`, bundles parsed with `iree-profile dispatch` (32 MB at pp512, 46 MB at
pp4096). Device time accounts for 1,355 ms of the 1,786 ms pp512 pass (76%) and 12.84 s of 15.72 s
at pp4096 (82%), so 18–24% is host-side and the table below is the rest. Top rows, share of device
time:

| kernel | pp512 | pp4096 | dispatches | workgroups per dispatch |
|---|---|---|---|---|
| `hrx_prefill_mmid_swiglu_iq3_s_m64_w4` | 27.1% | 25.1% | 92 / 736 | m64 tiles |
| `hrx_dp_mm_q8_0_m128_w8_f2` | 15.9% | 14.7% | 552 / 4416 | 4x10x1 and 4x40x1 |
| `hrx_prefill_mmid_iq4_nl_m64_w4` | 11.6% | 10.6% | 86 / 688 | m64 tiles |
| `ggml_mul_mat_f32_f32_wmma` | 8.6% | 8.0% | 522 / 4272 | f32 fallback |
| `ggml_flash_attention_f32_f16_wmma` | | 8.0% | 192 | |
| `hrx_dp_mm_q8_0_m64_w8_f2_sb` | 3.5% | 3.2% | 190 / 1520 | |
| `llm_gated_delta_net_f32_wmma_head128` | 3.0% | 2.7% | 72 / 576 | |

The expert GEMMs are a third of the pass and already the fastest thing per token: the m64 tile is
in play, so a wider MMQ tile is not the lever, and those kernels belong to the private add-on
(the removed DwarfStar tree's `ROCM_MMQ_Y` was a different route and never appeared in this bundle). The two
levers that do appear are both decode-shaped code running on a prefill. `hrx_dp_mm_q8_0_*` is ~16%
of it while its grid stays 4x10x1 and 4x40x1 workgroups — 40 and 160 workgroups on a 40-CU part —
however many tokens the chunk carries. `ggml_mul_mat_f32_f32_wmma` is 8% of a 4.25 bpw model's
prefill, which means some projection is falling back to an unquantized matmul.

Not claimed here: the 1,679 tok/s is another stack's number on another harness, and no MTP head
exists for this model in this tree, so there is no speculative-decode row. The profiled pass also
carries two multi-second gaps after `ggml_get_rows_f32` that the timed pass does not (1,786 ms ≈
1,355 ms device + ~0.4 s host), so those are bundle-write or cold-start cost, not the steady-state
host gap.

#### The two flags and the chunk ladder that got the product path off 85 tok/s (2026-11-05)

Everything above is the 2026-10-27 record, measured on the pre-loom kernel set. The same file goes
much further through `llama-server`, and the difference is **flags a user has to set**, not code in
the backend. Ladder run 2026-11-05T02:36-02:47 (`~/lb/supreme/raw/llamasrv-fork177*`, one binary
`add45105`, q31 = UD 3.1 `d9620119`, ctx 16384, 5 reps, `--cache-reuse` default is 256):

| config | load | pp512 | pp4096 (3,857 tok) | tg128@8k |
|---|---:|---:|---:|---:|
| product defaults (`-fa` auto, `--cache-reuse 256`) | 687 s | 300.83 | 85.05 | 10.00 |
| `+ -fa on` | 976 s | 390.52 | 153.71 | 8.75 |
| `+ -fa on --cache-reuse 0` | **439 s** | **419.43** | **444.89** | **12.99** |

- **`--cache-reuse` is the dominant cost on this model**: 2.89x at pp4096, 1.48x at 8k decode, and it
  is also why the load misses the front-end's 600 s readiness budget (687 s with the default, 439 s
  without it). The likely mechanism — GDN recurrent state plus checkpoints every 256 tokens, paid on
  every unique prompt and never earned back — is written down here but **was not traced in the
  scheduler**.
- **`-fa auto` resolves against this stack, not for it**: `-fa on` is a separate, smaller lever
  (1.30x at pp4096, 1.81x at pp512) that also collapses the pp4096 spread from 68.8 % to 9.1 %.
- The unpatched `llama-bench` cell (84.1 tok/s) equals the patched build **with product defaults**
  (85.05), so the 6.5x that threaded `GET_ROWS` buys is invisible through `1bit serve` until these
  two flags are set.

Chunk size, measured on one library (`27e7d641…`) in two windows so the pair is not confounded by a
rebuild, ctx 16384, `-fa on --cache-reuse 0`:

| arm | pp512 | pp4096 | load→ready | JIT compiles |
|---|---:|---:|---:|---:|
| `-ub 2048` | 183.76 | **205.05** | 999 s | 540 + 287 |
| `-ub 512` | 133.38 | 101.05 | 1025 s | 632 |

Do not multiply the two tables together: they are different windows on different policies, and this
file's own history has a 568.75-vs-272.61 pair for *identical* argv. Only the within-table rows are
comparable.

With the loom kernel set (2026-11-05T23:15, q31, `-ub 2048`, `HRX_PROFILE_MODE=dispatch`, 135 MB
bundle, 63,176 dispatches, 33.27 s of device time) the pp4096 pass attributes like this — a bundle
covers load, warmup and one cell, so the shares are indicative and its tok/s is not quotable:

| kernel | share of device time |
|---|---:|
| `ggml_mul_mat_id_swiglu_f32_f32_wmma` (routed gate/up) | 39.6 % |
| `hrx_dp_mm_q6_k_m128_w8_f2_sb` | 11.2 % |
| `ggml_flash_attention_f32_f16_wmma` | 11.2 % |
| `ggml_mul_mat_id_f32_f32_wmma` | 6.5 % |
| `ggml_mul_mat_f32_f32_wmma` | 4.2 % |
| `hrx_dp_mm_q8_0_m64_w8_f2_sb` / `_m128` | 3.3 % / 2.8 % |

Two things this replaces in the 10-27 table above: the pre-loom `hrx_prefill_mmid_swiglu_iq3_s_m64_w4`
row (27.1 % / 25.1 %) is now one kernel family at 39.6 %, and attention has grown to 11.2 % at this
context where the 10-27 pp512 bundle had it at 3.6 %.

#### `GGML_HRX_IQ3S_TABLE_DECODE` and the IQ3_XXS expert blocks ([engine#572](https://github.com/1bit-MONSTER/engine/issues/572))

Flash-Next's 48 MoE blocks are a mix of quant formats — for the UD-IQ4_XS build, `ffn_gate_exps`/`ffn_up_exps` are
IQ3_XXS on 28 blocks and IQ4_XS on 17 — and on identical geometry the IQ3_XXS blocks cost about **2.4x** what the
IQ4_XS blocks cost (79.94 ms vs 33.34 ms per dispatch at a 10x639 grid, from the dispatch profiler). The IQ3_XXS grid
decode was the reason: it selected a 12-bit grid code out of 256 inline constants, once per 4-value packet.

`GGML_HRX_IQ3S_TABLE_DECODE=1` selects a second router-expert kernel that reads that grid from a small per-workgroup
table instead. It is **off by default**; unset or `0` is the shipping path. Measured with it on, in one hold against a
control arm on the same binary:

| cell | control | `=1` | |
|---|---:|---:|---|
| pp4096 (product path) | 270.62 tok/s | **318.52** | 1.18x |
| pp4096, second window | 140.84 | **182.10** | 1.29x |
| pp512 | 177.62 | **221.38** | 1.25x |
| PPL 8x512 | 1.7198 | **1.7198** | identical |

The PPL row is the point: the value is unchanged to four decimals, measured in three separate windows, so the faster
decode is the same arithmetic. The absolute tok/s above are one box's state on one day — the same arm has read 182-318
tok/s across windows for identical flags, which is why only the paired ratios are quoted.

It stays opt-in because the two windows above ran on a fork commit that predates the SET_ROWS fix (#227), and a
3,858-token prompt on that commit intermittently died in the CPU fallback (engine #1251). Before this becomes a default
it needs the long-prompt cell plus `test-backend-ops -o MUL_MAT -b HRX0` on the current pin, neither of which this lane
has run.

Still open on #572, stated so nobody reads this as closed: the 1,679 tok/s gap is not explained; the
ubatch lever was measured up to 2048 and 4096 was tried and does not fit this 87 GiB file; no MTP
head exists in this tree for the decode side; and nothing here is a claim about released-flag
defaults — the ladder above is what an operator sets today.

#### Where a prefill forward's time goes, and the ~1.0 s that is not HRX ([engine#572](https://github.com/1bit-MONSTER/engine/issues/572))

Issue #572 reads the 1,786 ms pp512 pass as 755.7 ms of device time (41.9 %) and ~1.0 s of host time between
dispatches. Measured on the file itself, the host figure is right and the cause is not the HRX host path:

    Qwen3.8-Flash-Next UD-IQ4_XS (87.3 GiB), -ngl 99 -fa 1 -ub 512 -p 512 -dev HRX0
      llama-bench:                              1,795.4 ms/forward  = 285.2 tok/s
      libhrx dispatch bundle, 2 forwards:       1,353.5 ms of device kernels = 676.8 ms/forward (37.7 %)
      HRX graph computes, per forward:          618.6 ms in `replay.wait` (the kernels), 0.13 ms of host
      => outside the HRX backend:               ~1,177 ms/forward, 66 % of the forward

**The section above's "76 % device" is a two-forward number compared against a one-forward pass.**
1,353.5 ms is exactly two forwards of 676.8 ms, which is the same two-forwards reading that
[#572's comment (b)](https://github.com/1bit-MONSTER/engine/issues/572) retracted; it survived here in the
table. Per forward the device leg is 37.7 %, and #572's 41.9 % was the closer figure. The bundle's own
counts carry the same two-pass structure: 5,134 dispatches, and 334 dispatches of
`ggml_mul_mat_f32_f32_wmma` where the phase profiler sees one pass of 2,566 dispatches per forward.

The 676.8 ms above and the 618.6 ms below are the same device leg from two runs, and that 9 % is the
profiler: kernel times come from a run with `HRX_PROFILE_FILE` set, and writing that bundle on the dispatch
path inflates it relative to the `replay.wait` the unprofiled run times. The same sign and a similar size
show up on the two control models below (2 % and 7 %).

The host leg is real, and it is **not** the HRX backend: a graph compute on this model is 618.6 ms of
`replay.wait` with the graph-program lookup, binding, cache key, transient lease, host-staging rebind,
upload, launch, flush and download together under 0.13 ms. What costs the other ~1.18 s is llama.cpp-side
work - the CPU computing graph nodes that the HRX backend did not take (this model's graph runs as two
graph computes per forward, a 2,566-dispatch HRX one and a 1-dispatch one, so the rest is CPU-resident
work in llama.cpp's own graph, not a gap between HRX dispatches).

`GGML_HRX_HOST_GAP=1` (llama.cpp #154) is the profiler that separates those legs, and `HRX_PROFILE_FILE`
+ `HRX_PROFILE_MODE=dispatch`, read with `iree-profile statistics`, is the only device-time source: that
bundle reports 5,134 dispatches and 1,353.522 ms of kernels for the two forwards above, and per-kernel
shares - `hrx_dp_mm_q8_0_m128_w8_f2` 52.2 + 91.6 ms, `ggml_mul_mat_f32_f32_wmma` 334 dispatches for
94.8 ms, `hrx_hc_pre_gated` 42.9 ms.

**What that CPU work is: one GET_ROWS, declined by a size cap.** `llama-server -lv 4` on the same file
prints the placement of the prefill graph — `placement: 6060 nodes, 1 off HRX (GET_ROWS x 1), 3 graph
splits`. One node, on the CPU, and it is the whole ~1.18 s. It reads
`per_layer_token_embd.weight`, which this file carries as `IQ4_NL [160, 320001536]` - 18 bytes per 32
values, so 28.80 GB of table (26.8 GiB), against `token_embd.weight`'s ordinary `Q8_0 [2560, 248320]` -
and `dispatch_registration/common/dispatch-get-rows.cpp` declines it at
`is_supported_row_count()`: `kMaxGetRowsRowCount` is 262,208, the table has 320,001,536 rows. A gather of
512 rows out of a 28.8 GB table costs the page faults of touching it - and it grew with the token count
(35 ms at `-p 128`, ~1.18 s at `-p 512`) because more rows means more of the table.

That is the "GET_ROWS fallback" line in #572's own list, and it is the item to fix: either raise
`kMaxGetRowsRowCount` for a table this shape or give the gather a dispatch that does not walk 28.8 GB on the
host. The cap mirrors a promise in the kernel rather than being arbitrary:
`kernel-corpus/kernels/loom-libs/ops/get_rows_f32.loom` assumes `range(%token, 0, 262207)` on the row
index, so raising the matcher's cap alone would read out of range until that bound becomes a compile-time
config.

**The forward on a roofline.** Weights and shapes come from the GGUF tensor directory itself, FLOPs from
TraceLens's `PerfModel` (`GEMM`/`flash_attention`), peaks from the measured Radeon 8060S spec TraceLens
keeps for gfx1151 (256 GB/s, 59.39 TFLOP/s fp16, 118.78 TFLOP/s int4), and measured device time per class
from the dispatch bundle of the same model (`~/lb/i572/roofline_fn.py`). At 512 tokens every one of the 512
experts is touched (512 x 10 picks, E[untouched] = 0.02), so the expert weights are compulsory traffic,
while the 28.8 GB PLE table is only gathered (one 90-byte row per token per layer) and is not streamed:

| class | weights GB | touched GB | FLOPs T | mem roof ms | compute roof ms | measured ms |
|---|---:|---:|---:|---:|---:|---:|
| MoE experts | 59.52 | 59.52 | 2.416 | **232.5** | 20.3 | 306.1 |
| gated-delta-net layers (36) | 2.25 | 2.25 | 2.126 | 8.8 | 35.8 | 33.7 |
| token embedding + LM head | 1.20 | 1.20 | 0.651 | 4.7 | 11.0 | (in dense) |
| full-attention layers (12) | 1.02 | 1.02 | 0.671 | 4.0 | 11.3 | (in dense) |
| MoE shared expert + router | 0.50 | 0.50 | 0.306 | 2.0 | 5.2 | (in dense) |
| hyper-connections | 0.34 | 0.34 | 0.644 | 1.3 | 10.8 | 35.1 |
| PLE table | 28.80 | 0.002 | 0 | 0.0 | 0.0 | 0.5 |
| **total** | **93.67** | **64.87** | **6.81** | **253** | 115 | **676.8** |

The forward is memory-bound with 2.2x the compute roof, and the device leg runs at **37 % of its own
compulsory-traffic roof — 2.7x the floor**. The MoE expert GEMMs are the class that is close to it (306 ms
against a 232 ms floor, 76 %); the slack is elsewhere: 1,189 elementwise/copy/norm dispatches cost 106.6 ms
for no weight traffic at all (~90 us each, dispatch-bound), the f32 small matmuls cost 58.4 ms over 273
dispatches, and the dense projection GEMMs 136.3 ms over 395.

Against that, the host leg is not a rounding error but the largest single line in the forward: the one CPU
`GET_ROWS` costs ~1,177 ms, **4.6x the memory roof of the whole forward**, to move 2.2 MB of gathered rows
— it is page faults on a cold 28.8 GB mmap'd table, not arithmetic, which is why it does not improve from
one forward to the next (1,788.6 then 1,802.1 ms) while the device-side `get_rows` dispatches cost 0.5 ms.
Give that gather a dispatch and the forward is ~620 ms of device, i.e. ~2.9x pp512 on this file.

**The controls.** A model whose graph has no off-HRX node shows no host leg at all, which is how the two
legs are told apart:

| model | forward | `graph.lookup` | `graph.bind` | `replay.upload` | `replay.flush` | `replay.wait` (device) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-Coder-30B-A3B Q4_K_M, 1,161 dispatches | 263.5 ms | 0.004 ms | 0.002 ms | 0.005 ms | 0.001 ms | 263.46 ms |
| Qwen3.8-27B UD-Q4_K_XL, 1,519 dispatches | 870.2 ms | 0.003 ms | 0.003 ms | 0.004 ms | 0.001 ms | 870.14 ms |

and libhrx's dispatch profile agrees on those two: 2,322 dispatches / 538.14 ms of device kernels over two
forwards (103 % of the 523.8 ms they took) and 3,038 / 1,871.15 ms (113 % of 1,657.2 ms), the surplus being
the profiler's own record writing. The host path does grow with the ubatch, but it stays small: `-p 4096
-ub 2048` on the 30B-A3B is 1,143.7 ms of `replay.wait` in a 1,158.6 ms graph (98.7 %), with every other
phase under 0.25 ms, plus a one-time 4.5 s on the first graph of a new ubatch for prepare and kernel JIT.

What a host leg looks like when the *whole* graph leaves HRX: the same 27B at `-ngl 20` runs 3,746 graph
computes per forward instead of 2, and its host-staging upload alone becomes 145.7 ms per forward.

`rocprofv3` cannot produce any of this: it exits 0 on an HRX run and writes no traceback files, because
HRX dlopens TheRock's `libhsa-runtime64.so.1` itself and libhrx writes its own AQL packets, so the
profiler's HSA interposition never sees a dispatch.

### Hadamard-rotated Q4_0 files ("H32", dropped)

The engine dropped its own Hadamard-rotated Q4_0 format (owner, 2026-10-03): on HRX it was slower
and less accurate than the UD-Q4_K_XL file of the same model (Qwen3.8-27B pp512 63.5 against 335
tok/s; KLD 0.029 against 0.007 vs BF16), and the lean ROCm W4A4 route it was made for left with the
ROCm build. `tools/hadamard_q4_0.py` is removed, and `1bit serve` refuses a file stamped
`onebit.hadamard_q4_0` by name. The loader support in our llama.cpp (`llama-hadamard`,
llama.cpp #58) stays in the fork; it is the same code that rotates PrismML's files
([Ternary Bonsai](#ternary-bonsai-prismmls-hadamard-folded-ggufs)), which keep running.

### Server memory: the graph program cache cap (llama.cpp #63)

HRX builds a graph program for every new graph shape (each new prompt or batch length) and kept every
one: a long-running `llama-server` with prompts of varying length grew its GPU memory without bound
(ZAYA1-8B: about 1 GiB per request, 24.6 GiB after 28 requests). Since
[llama.cpp #63](https://github.com/1bit-MONSTER/llama.cpp/pull/63) the cache holds at most
`GGML_HRX_GRAPH_PROGRAM_CACHE` programs (default 64, `0` = unbounded): when a new program takes it over
the limit, HRX waits for the stream and drops every other program. On ZAYA1-8B, 40 requests of varying
length peak at 8.7 GiB instead of growing; greedy answers are identical, and llama-bench pp512/tg128
is unchanged. A repeated prompt length right after a flush costs 1.75 s instead of 1.12 s, since its
program is rebuilt.

### Server memory: HRX reads the weights instead of mapping them ([engine#856](https://github.com/1bit-MONSTER/engine/issues/856), [engine#1339](https://github.com/1bit-MONSTER/engine/issues/1339))

With a model on HRX (`-ngl` > 0), the loader reads the weights into their device buffers with `read()` and drops each weight's
page-cache pages as soon as it is read ([llama.cpp #234](https://github.com/1bit-MONSTER/llama.cpp/pull/234)). A mapped load
left the whole file in the page cache next to the device copy. On Qwen3.8-27B UD-Q4_K_XL (16.4 GiB), from a cold cache:
peak RSS 2.9 GiB instead of 16.4 GiB, nothing of the file left cached instead of up to 16.5 GiB, and a load of 18.6 s instead
of 14.3 s. Output and wikitext perplexity are unchanged. `--mlock` with mmap keeps the mapping.

The earlier weight-mapping import ([llama.cpp #226](https://github.com/1bit-MONSTER/llama.cpp/pull/226), opt-in since
[#231](https://github.com/1bit-MONSTER/llama.cpp/pull/231)) is gone, and `GGML_HRX_WEIGHT_HOST_PLACEMENT` does nothing.
It registered the file mapping as a KFD userptr, which the kernel always write-faults, so it dirtied the file, changed
its mtime, and never finished on large models while writeback kept invalidating it (engine#1339). HRX's repacked copy of the
dense weights still sits next to the device buffer in GTT.

### Measuring on `HRX0`: what makes a number citable

Numbers measured on this device swing far more between *runs* than between *files*, so the harness matters as much as the arm. Everything below is
either a guardrail the engine prints or an effect measured on this box on 2026-11-07/08; where another lane measured it first, they are named.

**1. The GTT pool is finite and it is shared.** Every launch prints the guardrail line:

```
ggml_hrx: HRX0 guardrail: GTT total 120.0 GiB, in use 40.7 GiB, ceiling 104.0 GiB (reserve 16.0 GiB),
          pool budget 63.3 GiB - a launch that needs more is refused here rather than faulting the device (engine #680)
```

`pool budget = ceiling - in use`, and a launch needing more than that is **refused**, not faulted. So the first thing a measurement needs is the budget,
not the file: two lanes independently arrived at the same rule from opposite sides — the Zyphra lane refuses below a 12 GiB budget for a 5.8 GiB model, and
the architecture lane refuses below `3.0 x weights + 2 GiB` (an 8B row was refused at budgets of 13.6 / 13.5 / 10.8 GiB). Over eight hours of this box's
evening the budget sat at **14-17 GiB** while long-lived services held the rest, which is why runs "that worked yesterday" crawl today. Two `llama-server`
processes from a resident embed install have been measured holding **94.3 GiB of 120** while arms died with `unable to allocate HRX0 buffer` at a
`pool budget 42.6 GiB` — so the co-tenant is a service that is **supposed to be resident**, and "take a quiet box" is a scheduling problem, not a courtesy.

**2. Budget the ticket for the *load*, not for the arms.** Loading is single-core-bound when other jobs share the four pinned cores: a 24 GiB GGUF spent
**35 minutes at 88 % of one core with GTT unmoved** — it was still loading, having measured nothing. A `gpu-hold -t 40` therefore expires before the first
arm, which happened four times in one night. Either take a long ticket on a quiet box, or reduce the **number of loads** (one process per arm means one
load per arm).

**3. A fallback is invisible in the dispatch histogram.** `GGML_HRX_LOG_DISPATCH=1` lists the registrations that **ran**; an op that never dispatched on
the device leaves **no line at all**. Measured: two arms differing only in one tensor type's format had a **near-identical** registration set (one extra
line, none missing) while `pp512` differed **5.4x** and `pp8` **8.1x**. So the histogram answers *"which kernel won"*, never *"did something fall back"*.

Use the right instrument for each question — three are needed and the histogram is only the first:

| question | instrument |
|---|---|
| which kernel **won** | `GGML_HRX_LOG_DISPATCH=1` (one line per registration that ran) |
| which node **fell back** | `GGML_HRX_REPORT=2` — `placement: 994 nodes, 1 off HRX (GET_ROWS x 1), 2 graph splits`, and `GGML_HRX_REPORT=1` for the once-at-load fallback line. Add `-lv 4` and it prints the matcher's own rejection reason and the node that broke the graph |
| was the result **right** | `test-backend-ops -o <OP> -b HRX0`, known-answer against the CPU backend |

And when a fallback is suspected but all three say nothing, the cost still shows up in the **timing** — against a control arm in the same window.

**4. Two windows are not a comparison.** The same unchanged file measured `211.16 / 47.14 / 4.26 tok/s` (pp512 / pp4096 / tg128) in one hour and
`108.07 / 92.28 / 12.20` in another — **2-2.9x apart, with sd 71 % on decode**. Rows may only be compared inside **one ticket**, with the guardrail's GTT
figure recorded at **each arm** and an unchanged-file control in the same window; otherwise the report is of the window, not of the file.

**5. A hot die starves the queue, and the thresholds do not line up.** `~/lb/thermal-run.sh` freezes a CPU job at **78 C** and thaws at 68 C, while
`gpu-hold` only starts below **75 C**. A CPU job under thermal-run can therefore park the die in the 75-78 C band indefinitely — too hot for the GPU
queue to start, too cool to be paused — and a queue of 22 tickets waited an hour behind exactly that. CPU-heavy measurement belongs under thermal-run
*and* out of a window with a deep device queue.

**6. State the binary, the pool budget and the tensor types with every row.** The build a row came from may be deleted before the row is read (it
happened: a 12-row table outlived its build by a day), and a file's *name* is not its format — a GGUF called `UD-Q4_K_XL` measured here was Q8_0 x279 /
F32 x246 / Q5_K x129 / IQ4_XS x69 / BF16 x47 / Q4_K x6, i.e. six tensors of the type its name advertises. Read the types out of the produced file, and
quote them, before quoting a number. **And digest the binary you actually *invoke*, not the one you built**: a gate run against a snapshot `bin/`
taken hours before a rebuild printed a clean `484/484` from the *pre-change* library, which reads exactly like a no-op. The citable form is the arm's own
`md5sum` of the invoked binary plus `libggml-hrx.so`, printed by the run that produced the row.

### Known issues on `HRX0`

- **Several sequences per batch fail.** `llama-perplexity` with `n_seq` > 1 stops on an
  unsupported 3-D MUL_MAT; use `-b 512`. On Qwen3.5 / Qwen3.8 (Bonsai too) the gated delta net runs
  one sequence per batch, so `serve` gives these models one slot on HRX and refuses `--parallel`
  (`--parallel N` under "Not yet" below).
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
- **The ZAYA1-8B NaN abort does not reproduce (engine#433).** Three aborts were reported on 2026-10-16
  on the then-current pin (`06cc0ef` with llama.cpp `dd74f6bb0`) with a second HRX process on the GPU:
  the sampler guard logged "HRX returned NaN logits" and the server stopped. Re-running that condition
  on the current line gave **no abort in 60 fresh loads** — 20 loads alone, 20 with a resident
  Qwen3-0.6B process on the GPU, and 20 with that second process on the older pin `1a5dcba2` — with
  four slots, unified KV, `--ctx-size 16384` and the game's tool-templated request asserted on every
  load, and 0 NaN lines. The report's 458-token prompt also cannot have carried a tool block: this
  model's own chat template gives 257 tokens with no tools, 629 with one, 726 with two and 851 with
  the four tools the game sends, so a 458-token request either had no tools or came from a different
  template. The guard itself is unchanged: NaN still aborts unless `GGML_HRX_NAN_CONTINUE` is set, and
  the current line logs `count`, `index range` and `contiguous` with it.
- **GLM-4.7-Flash MLA decode accuracy (engine#310).** The reported 11.79-vs-10.60
  perplexity gap is not reproducible on the current pin (`f2099e9b7`), which includes the MLA
  strided-V fix ([llama.cpp #70](https://github.com/1bit-MONSTER/llama.cpp/pull/70)) and the
  masked-V flash-attention fix ([#72](https://github.com/1bit-MONSTER/llama.cpp/pull/72)).
  On the same model and wikitext corpus, HRX decode (`-ub 1`) agrees with the CPU reference at
  both tested lengths:

  | Context | HRX decode PPL | CPU PPL | Absolute difference |
  |---|---:|---:|---:|
  | 4096 tokens (2 chunks) | 10.7208 ± 0.49991 | 10.7414 ± 0.50110 | 0.021 |
  | 8192 tokens (4 chunks) | 7.4988 ± 0.23237 | 7.4936 ± 0.23212 | 0.005 |

  The CPU reference uses its default ubatch with `-fa` off; CPU `-ub 1` and `-fa 1` are
  invalid for this model ([engine#337](https://github.com/1bit-MONSTER/engine/issues/337)).
  CPU `llama-perplexity` on GLM-4.7-Flash-Q4_K_M, `-c 4096 --chunks 2`, flags changed independently:

  | CPU flags | PPL |
  |---|---:|
  | default ubatch, no `-fa` | 10.7414 |
  | default ubatch, `-fa 1` | 1014.1151 (30x slower per pass) |
  | `-ub 1`, no `-fa` | 387.5591 |
  | `-ub 1`, `-fa 1` | 264.9719 |

  HRX with `-fa 1` stays correct (10.72, 6.82, 23.15 across configurations), and CPU generation
  at `-ub 1` produces the same text as the default ubatch. So the CPU perplexity path is
  configuration-sensitive for this deepseek2/MLA model; only the default ubatch with flash
  attention off is a trustworthy CPU reference. The historical 11.79 and 10.64 results do not have a runnable
  old-path A/B build, so the op-level drift bisect remains unverified. The masked-V kernel
  change includes a regression test that varies stale masked rows and requires bit-identical
  outputs.

### CWSR save-area placement (engine#315)

### The trio belongs to every HRX process, not just to serve's backends (engine#715)

`serve` puts the save-area variables in its backends' environment *and in its own* (`app/hrx_cwsr.h`,
`apply_cwsr_env_to_self`), so a child that execs with the inherited environment — a helper started by
`run_program`, the Laya daemon — inherits them too, and a spawn site added later cannot forget them.

Everything else that initialises HRX on a box whose libhsa is the TheRock nightly needs them as well, because the
fault they prevent happens inside ROCr/the HAL before any of our code runs: `test-backend-ops`, `llama-bench`,
`llama-cli`, and any harness that starts `llama-server` itself. For scripts there is one definition to source:

```sh
. scripts/hrx-env.sh --check     # sets the trio and the runtime path, refuses to continue without the trio
. scripts/hrx-env.sh             # sets it, no check
```

The script also fills `IREE_HAL_AMDGPU_LIBHSA_PATH` when the caller has not set it, because the trio is not enough on a
box whose device only opens through TheRock's runtime: without it the HAL resolves the system ROCr, registers no `HRX0`,
and `--list-devices` prints `(none)` — a harness then measures nothing and looks like a null result. `scripts/hw-gate.sh`
has always had to do this itself; now the recipe does it once, and a caller's value still wins.

and the recipe itself, for anything that cannot source a file:

```sh
HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1
IREE_HAL_AMDGPU_LIBHSA_PATH=<therock>/lib/libhsa-runtime64.so.1
```

`HRX_SVM=1` and `--hrx-cwsr-placement=off` remain the opt-outs, but they are no longer silent: on an `hrx`/`auto` route
`serve` refuses to start when *you* opted out and the trio is absent, and prints the recipe above — the mode those flags
select is the one that page-faults on the first graph replay on this box. The refusal keys on the opt-out, deliberately
not on "the trio happens to be empty": a build whose hrx-system has no `HRX_ALLOW_NO_SVM` gate also has an empty trio,
and it cannot start HRX0 at all. `ONEBIT_HRX_NO_TRIO_OK=1` is the acknowledgement for a box whose
ROCr does not need the placement. `--device cpu` and `--device npu` are never affected.

The shipped configuration for engine#315 is amdgpu with CWSR **off** (`amdgpu.cwsr_enable=0` on the
kernel command line and in `/etc/modprobe.d`). Nothing below changes that. The replacement below was
accepted by a soak on a CWSR-on boot (2026-10-18/19: 37 in-band arms, 9 h 03 min, 0 faults, every graded
arm's save area two GTT BOs and no SVM lookup; KNOWN-ISSUES.md, engine#315), and since **2026-10-31 it is
the default for every HRX backend, CWSR on or off** - see the A/B below, where the pageable placement
faulted on the first graph replay of a CWSR-**off** box.

On such a box, `1bit serve` moves each HRX queue's CWSR context save area out of pageable memory. By
default libhsakmt allocates the save area as an SVM range. With `HSA_USE_SVM=0` and
`HSA_USERPTR_FOR_PAGED_MEM=0` it allocates a KFD GTT buffer object instead. ROCr then reports no SVM,
which HRX refuses unless `HRX_ALLOW_NO_SVM=1` is set; that gate is our hrx-system commit
(`util/no_svm_gate.c`). serve adds the three variables to the environment of every HRX backend it starts
(the chat server, the RAG servers, the Laya scorer), so they are in place before ROCr initialises in that
process, and prints one line:

```
1bit serve: HRX: CWSR save area placed in GTT by default (engine#315; amdgpu cwsr_enable=0; --hrx-cwsr-placement=off or HRX_SVM=1 opts out)
```

`--hrx-cwsr-placement` picks the mode:

| value | what serve does |
|---|---|
| `auto` (default) | adds the three variables, whatever `cwsr_enable` says. A value you set yourself still wins. `HRX_SVM=1` in the environment turns it off again (`HRX: HRX_SVM=1; CWSR save-area placement left to ROCr (engine#315)`) |
| `on` | the same, but refuses to start in a build without the gate |
| `off` | never adds them |

A build whose hrx-system has no `HRX_ALLOW_NO_SVM` gate (CMake prints
`HRX_ALLOW_NO_SVM gate in hrx-system: 0`) adds nothing in `auto` and refuses `on`, because
`HSA_USE_SVM=0` would leave HRX with no device. `llama-server`, `llama-bench`, `test-backend-ops` or other
HRX tools you start yourself get the variables only if you set them - `scripts/hw-gate.sh` exports them for
its own direct runs for that reason.

#### Why it is the default: the 2026-10-31 A/B (kernel `7.3.0-rc4-1bit-perfopt+`)

After that kernel was rebuilt, **every** HRX process on strixhalo page-faulted on its first graph replay,
the release tarball included: `wait for HRX graph replay commands failed: … ABORTED; AMDGPU memory access
fault at device address 0x…`, and in `dmesg` `[gfxhub] page fault … client CPF`, `PERMISSION_FAULTS 0x8`,
`WALKER_ERROR 0`, `MAPPING_ERROR 0`. Three HSA runtimes exist on the box; same binary (`llama-server` from
`1bit/hrx-vulkan-patched` + #161), same Qwen3-0.6B file, same argv, one `gpu-hold` claim, 32 tokens per arm:

| HSA runtime (sha256) | SVM mode | result | new gfxhub faults |
|---|---|---|---:|
| release-bundled `f6ef689d…` | `HSA_USE_SVM` unset, `HSA_USERPTR_FOR_PAGED_MEM=0` | **no HRX0 at all**: `--device: invalid device: HRX0` | 0 (nothing ran) |
| release-bundled `f6ef689d…` | `HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1` | **no HRX0 at all**, same error | 0 (nothing ran) |
| TheRock nightly `21a66824…` (10.3.0a20261031) | `HSA_USE_SVM` unset, `HSA_USERPTR_FOR_PAGED_MEM=0` | server up, first completion **HTTP 500**, 31 fault lines | **+1** |
| TheRock nightly `21a66824…` | `HSA_USE_SVM=0 HSA_USERPTR_FOR_PAGED_MEM=0 HRX_ALLOW_NO_SVM=1` | **HTTP 200, 260.1 tok/s** | **0** |

So on a dev build the runtime the release *image* installs could not register HRX0 in **that** arm, and the nightly
libhsa the engine picks page-faults in the mode above.

Two corrections to how those first two rows should be read (2026-11-04, measured with the image's own runtime set):

* **The arm did not use the set the image installs.** That A/B arm pointed
  `IREE_HAL_AMDGPU_LIBHSA_PATH` at a **single-file** extract of the same bytes, and ROCr 1.21.0's build needs its eight
  `librocm_sysdeps_*` siblings plus `librocprofiler-register.so.0` in the same directory. Without them `dlopen` fails
  (`librocm_sysdeps_elf.so.1: cannot open shared object file`) and no device can appear, whatever the SVM mode.
* **With the closure, the image's runtime does register a device.** The 10-file set at `/opt/rocm-therock/lib`
  (sha256 `f6ef689d163b42ded68cb99ffa4f9103251f05ba91fc3ca70f4a1d1909d9c34a`) exports all 121 HSA entry points the
  driver resolves by name, and `hrx_cts_device` passes 7/7 device cases on it. So the digest is not the problem and the
  two TheRock builds are interchangeable by symbol set — the *directory* was. Diagnose with `ldd`, not with `sha256sum`
  alone.

One diagnostic caveat that follows: a runtime path that cannot be `dlopen`ed does not fail loudly. The loader appends
that error and keeps searching, so the run falls through to the bare `libhsa-runtime64.so.1` on the system path — the
_distro_ ROCr, which lacks `hsa_amd_queue_create` — and the error you see names the distro copy, not the path you
passed. "Not found" about the system library usually means the runtime you named could not be loaded.

**Which variable is responsible is narrower than "SVM":** a four-corner table on the same 0.6B file (same binary,
cache limit 128) shows the fault needs the two knobs *together* — libhsakmt takes the SVM save-area path only when
pageable userptr is also disabled:

| `HSA_USE_SVM` | `HSA_USERPTR_FOR_PAGED_MEM` | result | new faults |
|---|---|---|---:|
| unset (1) | **0** | first completion fails | **+1** |
| unset (1) | unset (1) — libhsakmt's real default | **ok** | 0 |
| 0 | unset (1) | **ok** | 0 |
| 0 | 0 (this placement) | **ok** | 0 |

So the faulting mode is *SVM with pageable userptr disabled* — what the fleet's shell harness had been setting
(`HSA_USERPTR_FOR_PAGED_MEM=0`, inherited from `612-hrx.sh`) — not libhsakmt's default. That makes this placement one
of three working modes rather than the only one. It is still the mode `serve` can choose safely on a box whose
defaults are not known to be clean, because it is the one the #315 soak covered; the true-defaults mode was clean on
the 0.6B file and its 76 GiB measurement was still pending when this landed.

Corroborating evidence from the same session: plain HIP through the same runtime is clean
(`hipMalloc -> H2D -> memset -> kernel -> D2H -> pinned H2D`, `HIP_PATH_OK`, 0 faults), `rocminfo` reports
`XNACK enabled: NO` and `Coherent Host Access: FALSE` for `gfx1151`, and `test-backend-ops -o TOP_K -b HRX0` passes
484/484 with the placement in place and 0 fault lines. The box's `cwsr_enable` was `0` throughout, which is why
`auto` no longer keys off it.

Measured on strixhalo (performance mode, 120 W cap), engine `fce3a68` build with the gate:

- CWSR-on boot, GLM-4.7-Flash Q4_K_M at the #315 composition (4 neighbour servers, 16 GiB ballast,
  `gtt_start` 104 GiB), one binary: 46 runs with the variables, 0 faults (42 clean in band);
  16 runs without them, 4 faults (Fisher exact two-sided p = 0.0043). Every save area of every
  run with the variables was a 19226624-byte GTT BO (kprobes on `kfd_queue_buffer_get`).
  0 of 42 bounds the fault rate at 6.9 % or less (one-sided 95 %), not at zero.
- CWSR-off boot, Qwen3-0.6B Q4_K_M, the variables against none: needle at 4732 tokens 5/5 both,
  same output; KLD top-20 0.000000 nats; `llama-bench -p 512 -n 128`, medians of 2 rounds:
  pp512 21373 vs 21047, tg128 336.6 vs 337.6 tok/s.

### Prefill on HRX, decode on Vulkan (removed)

`--prefill-device hrx` left with the Vulkan build (RFC #213 stage 3): `1bit serve` refuses it.
The fork still carries the zero-copy KV sharing code (below); what it did and how it measured is
in [hrx.md at 0baf286](https://github.com/1bit-MONSTER/engine/blob/0baf286/docs/hrx.md).

### Our patches

`1bit/hrx-vulkan` is AMD's commit unchanged; `1bit/hrx-vulkan-patched` adds:

- **Zero-copy KV sharing** (unused by the engine since it builds no ggml-vulkan): dma-buf export in ggml-hrx
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
  views), which cannot move. Leaf (`NONE`) nodes count as covered. GET_ROWS for batched IQ3_S
  gives wrong values, so those nodes are left to the CPU. IQ4_NL and IQ4_XS were left to the CPU too
  until [llama.cpp #41](https://github.com/1bit-MONSTER/llama.cpp/pull/41) (below).
- **Kernels for ops HRX sent to the CPU** ([llama.cpp #24](https://github.com/1bit-MONSTER/llama.cpp/pull/24)):
  a batched F16 matmul (`ggml_grouped_mul_mat_f16_f32`, e.g. ZAYA's grouped convolution, whose
  weights the loader can now place on HRX), and short-row kernels in `small_rows_f32.loom`:
  SOFT_MAX without a mask, SUM_ROWS, ARGSORT, GET_ROWS for narrow rows (strided ids, as top-k views
  are), CONT of strided views and broadcast-only REPEAT. Each matcher claims only what the existing
  kernels do not. `test-backend-ops -b HRX0` passes every case of these ops. ZAYA1-8B's decode graph
  goes from 641 graph splits to 1: 25.5 to 47.9 tok/s (tg128).
- **Prompt matmuls on the q8_1 x4 kernel** ([llama.cpp #55](https://github.com/1bit-MONSTER/llama.cpp/pull/55)):
  Q5_K and IQ4_XS prefill matmuls get Q4_K's q8_1 x4 policy, and the generic fused SwiGLU leaves those
  prompt chunks to it. On Qwen3.8-27B UD-Q4_K_XL, pp512 goes 98.5 to 334.6 tok/s
  ([section above](#prompt-matmuls-on-the-q8_1-x4-kernel-llamacpp-55)).
- **Remainder chunks on the q8_1 x4 kernel** ([llama.cpp #61](https://github.com/1bit-MONSTER/llama.cpp/pull/61)):
  Q5_K / IQ4_XS chunks not a multiple of 256 send their aligned head to it; pp400 67.4 to 103.4 tok/s
  (same section).
- **Prompts past 32K context** ([llama.cpp #59](https://github.com/1bit-MONSTER/llama.cpp/pull/59)):
  V stays row-major past `copy_transpose_f16`'s 32768-row range. pp512 at depth 32768 failed before
  (JIT `row_count` 33280 violates `range`) and runs at 227.4 tok/s now (depth 16384 unchanged at 269);
  one 34,816-token wikitext chunk gives PPL 6.2150.
- **Mixed-format SwiGLU pairs** ([llama.cpp #60](https://github.com/1bit-MONSTER/llama.cpp/pull/60)):
  the K-quant SwiGLU decode kernels stage a separate codebook for gate and for up, so Unsloth's sub-4-bit
  UD files no longer fall back to the generic path for pairs that mix formats. Qwen3.8-27B UD-IQ2_S
  tg128 3.12 to 8.32 tok/s; UD-Q4_K_XL unchanged (11.88 / 11.81); KLD 0.00013 vs the previous build,
  `test-backend-ops -b HRX0` 996/996.
- **PrismML's PQ2_0 / PTQ1_0 types** ([llama.cpp #62](https://github.com/1bit-MONSTER/llama.cpp/pull/62)):
  native ggml types 142 / 143 with a CPU reference and HRX decode on the K-quant kernels; Q1_0 decode
  moves onto them too ([Ternary Bonsai](#ternary-bonsai-prismmls-hadamard-folded-ggufs)).
- **The Hadamard rotation on HRX** ([llama.cpp #64](https://github.com/1bit-MONSTER/llama.cpp/pull/64)):
  a kernel for the `GGML_HINT_SRC0_IS_HADAMARD` MUL_MAT ([Ternary Bonsai](#ternary-bonsai-prismmls-hadamard-folded-ggufs)).
- **MXFP4 weights** ([llama.cpp #65](https://github.com/1bit-MONSTER/llama.cpp/pull/65)): MXFP4 in the
  shared dequantizer (17 bytes per 32 values, the E8M0 scale built exactly as ggml does).
  `tests/test-hrx-mxfp4` checks GET_ROWS on HRX0 bit for bit against ggml's own decoder;
  `test-backend-ops -b HRX0` 1051/1051. MUL_MAT_ID now takes only input sizes that are a multiple of
  256, which its kernels declare (until #66 below).
- **gpt-oss on HRX** ([llama.cpp #66](https://github.com/1bit-MONSTER/llama.cpp/pull/66),
  [#67](https://github.com/1bit-MONSTER/llama.cpp/pull/67), [#73](https://github.com/1bit-MONSTER/llama.cpp/pull/73)):
  MUL_MAT_ID takes input sizes that are a multiple of 32 (gpt-oss-20b's experts are 2880 wide), with a
  decode-loader stride fix that this exposed (rows were read 3072 apart); ADD_ID and the clamped SWIGLU_OAI
  run on HRX in our own kernels; and a placement guard (#73) keeps those two ops with their MUL_MAT_ID when
  the experts run on the CPU, where the mix gave wrong values (root cause open, engine #286).
  gpt-oss-20b MXFP4, balanced mode: pp512 25.8 -> about 1000 tok/s, tg128 12.6 -> about 35, text correct,
  KLD vs CPU 0.029.
- **Attention sinks** ([llama.cpp #68](https://github.com/1bit-MONSTER/llama.cpp/pull/68)): gpt-oss's
  per-head sink logits run on HRX as an exact post-correction of AMD's unchanged flash-attention output,
  O x sigmoid(LSE - sink), in our own dispatch; an output that overlaps an input's storage is refused.
  gpt-oss-20b greedy text is identical to the CPU-sink path; pp512 990 +/- 22, tg128 35.0 +/- 5.5 tok/s.
- **PrismML tile bytes** ([llama.cpp #74](https://github.com/1bit-MONSTER/llama.cpp/pull/74)): the low-token
  SwiGLU sized PQ2_0 / PTQ1_0 weight rows from a tile size of 0, so every row read row 0. Ternary-Bonsai-1.7B
  PQ2_0 decode perplexity 1,225,624 -> 20.65 (CPU 20.70); the 27B files took another kernel and were unaffected.
  **PQ2_0 on the dense prefill kernels and ternary weights on the 2..16-token verify kernels** (private
  add-on, 2026-10-25): Ternary-Bonsai-2-27B PQ2_0 pp512 188 -> 553 tok/s, and its ternary verify batch
  decodes at 43.3 / 37.3 / 27.1 tok/s on code / prose / short with the Qwen3.8-27B MTP head (39/40 GSM8K).
- **NVFP4 (ggml type 40) and Q2_0 (type 42)**: prompt and decode on HRX through our HIP kernels
  (private add-on, 2026-10-24), no extra flag or environment: a Qwen3.8-27B file of either type loads
  and runs with **one graph split**. Measured on the 27B, performance mode: NVFP4 pp512 613 / tg128
  12.7 tok/s, KLD vs the CPU 0.00144; Q2_0 pp512 632 / tg128 23.1 tok/s, KLD vs the CPU 0.00055.
- **TQ1_0 / TQ2_0** ([llama.cpp #69](https://github.com/1bit-MONSTER/llama.cpp/pull/69)): ggml's ternary types
  in the shared dequantizer and on the K-quant decode kernels. Ternary-Bonsai-1.7B, HRX0 vs CPU: KLD 0.000523,
  same top token 98.67%; pp512 / tg128 TQ1_0 3542 / 113, TQ2_0 4100 / 156 tok/s.
- **MLA attention past 512 tokens** ([llama.cpp #70](https://github.com/1bit-MONSTER/llama.cpp/pull/70)): the
  V transpose assumed contiguous rows, but MLA (deepseek2, GLM) stores V with the QK head stride, so prompts of
  512+ tokens read misaligned rows. GLM-4.7-Flash Q4_K_M: PPL 315,664 -> 5.916 (CPU 5.959).
- **Hadamard files: qwen3next** ([llama.cpp #71](https://github.com/1bit-MONSTER/llama.cpp/pull/71)): `ssm_ba`
  is folded too, and a stamped file with any other unfoldable Q4_0 weight is refused by name.
- **Masked keys in flash attention** ([llama.cpp #72](https://github.com/1bit-MONSTER/llama.cpp/pull/72)):
  KV rows past a sequence's end hold whatever an earlier request or rejected draft wrote, and the F16 WMMA made
  P = 0 x V depend on that V's sign, so identical requests gave different logits (up to 0.13 on Qwen3-0.6B,
  13.5 on GLM-4.7-Flash). Masked rows now reach P x V as exactly +0; Qwen3-0.6B and Qwen3.8-27B repeat
  bit-identically. Cost: pp512 -4.7% on Qwen3-0.6B, the rest within noise.
- **Weights read, page cache dropped** ([llama.cpp #234](https://github.com/1bit-MONSTER/llama.cpp/pull/234)): with a model
  on HRX the loader reads the weights instead of mapping the file and drops their page cache; replaces the weight-mapping
  import of #226/#231 ([section above](#server-memory-hrx-reads-the-weights-instead-of-mapping-them-engine856-engine1339)).
- **A cap on the graph program cache** ([llama.cpp #63](https://github.com/1bit-MONSTER/llama.cpp/pull/63)):
  `GGML_HRX_GRAPH_PROGRAM_CACHE`, default 64 ([section above](#server-memory-the-graph-program-cache-cap-llamacpp-63)).
- **Hadamard-rotated Q4_0 files** ([llama.cpp #58](https://github.com/1bit-MONSTER/llama.cpp/pull/58)):
  llama loaded the engine's `onebit.hadamard_q4_0` ("H32") files through `llama-hadamard`; Qwen3.8-27B-Q4_0-H32
  on HRX vs BF16 (wikitext 40 x 512): PPL 6.021, KLD 0.0292, same top token 92.5%. The engine has since
  dropped the format ([above](#hadamard-rotated-q4_0-files-h32-dropped)).

Measured on Strix Halo (Qwen3-0.6B, perplexity over 8 x 512 wikitext tokens):

| File | Vulkan0 | HRX0 on AMD's commit | HRX0 with our patches |
|---|---|---|---|
| Q4_K_M | 22.53 | 22.52 | 22.52, same speed (21691 / 324 tok/s) |
| UD-Q4_K_XL | 22.43 | fails | **22.41** |
| UD-Q2_K_XL | 36.26 | fails | **36.46** |
| UD-IQ2_M | 55.44 | fails | **55.58** |

`test-backend-ops -b HRX0`: 791 OK, 0 failed (on AMD's commit, about 1150 fail).
Correct is not fast: the UD files run their IQ4_XS and sub-4-bit layers on the CPU
(UD-Q4_K_XL 6303 / 246 tok/s, UD-Q2_K_XL 974 / 76). (That was on AMD's commit; our later patches
run IQ4_XS on HRX0, "Not yet" below.)

`1bit-MONSTER/llama.cpp` also holds
`1bit/hrx2-archive`, the previous build (AMD's abandoned ggml-hrx2 plus our Q4NX
kernels). It is kept for a later port of that work to ggml-hrx. Its zaya
architecture has been ported to this branch (ZAYA1).

The engine's separate upstream llama.cpp pin for Vulkan (`third_party/llama.cpp-vulkan`) is gone
(RFC #213 stage 3). Architectures only that pin had (Zyphra Zamba, Zamba2 and BlackMamba, and a
few upstream-only ones) are refused by `1bit serve` until they are ported here; Lemonade's own
llamacpp backend serves them meanwhile. Qwen3.8-Flash-Next (`qwen4exp`, with its MTP draft head)
was ported at pin `e57beb97`. This build has no Vulkan
backend (`GGML_VULKAN=OFF`).

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
scripts/fetch-rocm-sdk.sh                      # the pinned toolchain, into third_party/toolchain/
git submodule update --init --depth 1 third_party/hrx-system third_party/llama.cpp
cmake -B build -G Ninja -DONEBIT_HRX=ON          # -DONEBIT_HRX_TOOLCHAIN=<dir> overrides the toolchain
cmake --build build --target onebit
```

- **The toolchain.** The build compiles with TheRock's ROCm SDK at one pinned version and one GPU
  target, `config/hrx-toolchain.config` (engine#713): `scripts/fetch-rocm-sdk.sh` installs
  `rocm[libraries,devel,device-gfx1151]` at that version from the nightly wheel index into
  `third_party/toolchain/rocm-sdk-<version>-<gfx>`, and `cmake/hrx.cmake` defaults to that copy, so
  a fresh box needs only this repository. `-DONEBIT_HRX_TOOLCHAIN` / `ROCM_PATH` still override it,
  and an existing `/opt/rocm-therock` is used when the tree copy is absent. The device library
  (the `amdgcn` bitcode) is read from the toolchain itself, so no `HIP_DEVICE_LIB_PATH`,
  `ROCM_PATH` or `HIP_PATH` is needed in the environment.

- **One external project.** llama.cpp's ggml-hrx builds hrx-system itself
  (`HRX_SOURCE_DIR`) with TheRock's `amdclang`.
- **HRX0 and the CPU only.** `GGML_VULKAN`, `GGML_HIP` and `GGML_CUDA` are off. The targets are
  `llama-server` and `llama-bench`. With
  `-DONEBIT_SERVE_TEST_GGUF=<gguf>`, ctest runs `serve_e2e_hrx` and `serve_e2e_cpu`.
  `serve_e2e_hrx` skips on a machine without a gfx1151 (`ONEBIT_HRX_E2E_TARGETS` names others),
  found in sysfs by `scripts/gpu-targets.sh` without opening the GPU.
- **`IREE_ROCM_PATH` is not passed.** It would switch hrx-system to "package"
  mode, which needs TheRock's aqlprofile-sdk headers, and our `/opt/rocm-therock`
  does not ship them. Left unset, hrx-system fetches its pinned HSA/AQL headers.
- **The HSA runtime.** HRX dlopens it at run time, and the distro's
  `libhsa-runtime64` rejects the `HSA_AMD_AGENT_INFO_PM4_EMULATION` probe on
  gfx1151, so HRX then registers no device. CMake finds TheRock's
  `libhsa-runtime64.so.1` (`ONEBIT_HRX_LIBHSA`), and `1bit serve` passes it to
  llama-server as `IREE_HAL_AMDGPU_LIBHSA_PATH`, unless it is already set. Without a
  build path it searches the pinned toolchain, then `/opt/rocm-therock`. To run
  llama-server or llama-bench by hand, export that variable yourself.
- **The found path is build metadata, not a literal (engine#1453).** CMake records the
  toolchain's `libhsa-runtime64.so.1` in the cache so `os/mini/mkimage.sh` knows which
  runtime to *ship*, but nothing is compiled into the binary: baked in, it is a
  build-machine path by construction - it makes the tarball's bytes depend on where the
  build ran, writes the builder's tree into a public artifact (the c33 roll of 2026-11-08
  failed its own package gate on `/home/<builder>/wt/.../libhsa-runtime64.so.1`), and names
  a location the target does not have, since the appliance installs the runtime at
  `/opt/rocm-therock/lib` and the distro packages at `/opt/1bit/engine/hrx/llama/bin`. So
  `serve` resolves `--hrx-libhsa`, then a path baked on purpose with
  `-DONEBIT_HRX_LIBHSA_BAKE=<path>`, then the scan.
- **The runtime is named at init (engine#713).** Three `libhsa-runtime64.so.1` exist on a
  box (the release-shipped one, the pinned toolchain's, the distro's) and they are not
  interchangeable: the distro's ROCr is old enough to be missing an entry point the driver
  resolves by name, and a copy without its `librocm_sysdeps_*` siblings cannot be loaded at
  all. So the first HRX backend `serve` starts logs the file HRX will dlopen and its digest, once:

  ```
  1bit serve: HRX: HSA runtime /home/you/engine/third_party/toolchain/rocm-sdk-10.3.0a20261031-gfx1151/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1 sha256 21a6682452e159cfcdc6bdc84676f6b7f9c1236f3a2957a97a2e7170889cdfa3
  ```

  Check it against `sha256sum <path>`, and against the digest the release table records
  ([releases.md](releases.md), "Pins") — a run that fell back to another copy is visible
  there instead of passing silently. The line comes from `app/hrx_runtime.h`; `--hrx-libhsa
  <file>` overrides the path, and a file that cannot be read is still named, without a
  digest.

### Private HIP kernels

Our own HIP kernels for HRX are closed source, like the NPU kernels: they live in a private
repository, as an add-on (`addons/hip-hrx`). The llama.cpp fork keeps only the
public HIP-through-HRX plumbing (`ggml/src/ggml-hrx/hip/`). `-DONEBIT_GPU_PRIVATE=<add-on checkout>`
(with `-DONEBIT_HRX=ON`) passes `GGML_HRX_HIP_ADDON_DIR=<checkout>/addons/hip-hrx` to the HRX
llama.cpp build, which compiles those kernels and matchers in; it needs a llama.cpp pin with
that hook. Without the option the build is unchanged. The release builds with it, from the
commit pinned in the release config, so the release binaries carry the kernels as
compiled code objects while their source stays private ([releases.md](releases.md), "Private
components"). To drop the add-on from an existing
build directory, delete `build/hrx/llama` (its CMake cache keeps the setting).

## Verified (2026-09-23, llama.cpp `f1a0aca`, hrx-system `51b1739`)

`ctest` passed, re-run after the first daily bump (#13). At the time the check
was `tests/hrx_lemonade_e2e.sh`, through the engine's former embedded Lemonade:
`unsloth/Qwen3-0.6B-GGUF:Q4_0` answered "Paris" on `Vulkan0` and on `HRX0`. The
same build now passes `tests/serve_e2e.sh` on both devices through `1bit serve`.

`tests/hrx_cpu_weight_parity.sh` (engine #674) is the row that compares placements instead of just checking that each one
answers: the same greedy prompt through the CPU (`-dev none`), through the device with **every layer assigned to the CPU**
(`-dev HRX0 -ngl 0`, the partial-offload path where the device still claims nodes and materializes their weights) and fully
offloaded (`-ngl 99`). It compares the **distributions**, not the text: the CPU path and the device paths are different
arithmetic (f32 CPU kernels vs HRX's kernels over an f16 KV cache), their logits differ by ~0.1 nats, and at temperature 0 a
coin flip between two candidates flips one token — measured on `ca1d42f`, where the *shipped* `-ngl 99` placement flips its
own coin 24 tokens in. So each device arm must keep the CPU's top-5 distribution (overlap >= 0.30, median |dlogprob| <=
0.50 nats, calibrated: 0.63-0.78 overlap and 0.038-0.094 nats on a correct stack, 0.00 overlap when placement actually
corrupts the answer), compared over **at least 12 steps** — an arm that stops early (EOS, or a placement bug that ends the
graph) is reported as a failure rather than compared over the one or two steps it managed. It skips without a gfx1151, and it fails rather than skips when the
machine has a GPU but the HRX runtime cannot open it — the `IREE_HAL_AMDGPU_LIBHSA_PATH` case `scripts/hrx-env.sh` now
resolves for hand-run tools.

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

- **IQ4_NL / IQ4_XS on HRX ([llama.cpp #41](https://github.com/1bit-MONSTER/llama.cpp/pull/41), done).**
  AMD's IQ4 nibble packers remapped every table index (XOR 12) for a lowering that does not apply on
  gfx1151, in `motifs/dequant.loom` and again in the Q5_K/IQ4_XS prefill kernel, so IQ4 was declined
  and ran on the CPU. #41 fixes both copies and the q8-plane SwiGLU path they exposed: the shared
  dequant now reads the real `kvalues_iq4nl` table through a table lookup, and the corpus contains no
  XOR-12 on that path at all. Qwen3.8-27B UD-Q4_K_XL: 91 graph splits -> 1, wikitext PPL 5.8294 (CPU
  5.8288), decode 7.25 -> 9.0 tok/s (Vulkan 12.5).

  Re-verified on the **Loom-only** path on 2026-10-25 (a build with no HIP add-on at all, so every case
  below runs through the Loom corpus). `test-backend-ops -b HRX0`: `MUL_MAT` iq4_nl **OK** for n=1..9 at
  k=256, iq4_xs the same, `GET_ROWS` iq4_nl **OK** at `be1=1`; **0 FAIL, 2/2 backends passed**. Two
  groups stay *declined* rather than wrong, and both are shape coverage, not the codebook: `k=32`
  projections and `GET_ROWS` on a view or with a batched gather (`v=1`, `be1=7` — the same gap
  [390](https://github.com/1bit-MONSTER/engine/issues/390) found, where `q8_0` fails identically, so the
  quant is not the variable); and **`MUL_MAT_ID` for iq4_nl / iq4_xs**, which the Loom corpus does not
  claim at all, so routed experts of those types take the CPU on a Loom-only build instead of decoding
  garbage. The build the engine ships carries the HIP add-on, whose routed-expert kernels handle those
  types on HRX.
- **K-quant decode kernels ([llama.cpp #42](https://github.com/1bit-MONSTER/llama.cpp/pull/42)).**
  One-token projections on Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS and Q8_0 now read the GGUF blocks
  directly with f32 activations: the FFN gate/up pair fused with SwiGLU, and plain projections
  with the residual add folded in. Mixed-quant files (Unsloth UD) used to take generic kernels at
  ~105-176 GB/s; these run at ~200-224. Decode, tg128 against the Vulkan build on the same box:

  | Model | HRX | Vulkan | |
  |---|---|---|---|
  | Qwen3.8-27B UD-Q4_K_XL | 12.15 | 12.48 | 97.3% (was 9.0) |
  | ZAYA1-8B Q4_K_M | 94.0 | 93 | ~100% (was ~90) |
  | Qwen3-0.6B Q4_K_M | 326.6 | 357 | 91.4% (was 320.8) |
  | Qwen3-Coder-30B-A3B Q4_K_M | 90.6 | 93.7 | 96.6% (unchanged) |

  [llama.cpp #43](https://github.com/1bit-MONSTER/llama.cpp/pull/43) added Q3_K and IQ3_S (IQ3_S keeps
  its 512-entry grid in workgroup memory), which covers every weight type of the UD-Q4_K_XL 27B.
  Decode-shaped wikitext PPL (`-ub 1`) on the 27B: 7.1595, CPU 7.1475.
- **HRX decode no longer spins a CPU core
  ([llama.cpp #44](https://github.com/1bit-MONSTER/llama.cpp/pull/44)).** The per-token stream wait
  sat in ROCr's signal wait, which returned from the KFD event ioctl on every completed command
  (~900 per 27B token) without sleeping: a full core for the whole token, ~10 W of package power,
  and a 90-93 C APU during 27B decode. Long expected waits (over 20 ms) now sleep through 80% of
  the wait; shorter ones are unchanged. 27B decode: 71-76 C and ~125 W instead of 83-93 C and
  133-137 W, 0.4% slower; Qwen3-0.6B and ZAYA1-8B unchanged. `ONEBIT_HRX_BLOCKING_WAIT=1` turns it
  off. Model load and the first request still burst to ~88 C for a couple of seconds (host weight
  conversions and first-use GPU work share the package power limit).
- **Lossless SwiGLU for every FFN layer and MTP verify batches
  ([llama.cpp #45](https://github.com/1bit-MONSTER/llama.cpp/pull/45),
  [#46](https://github.com/1bit-MONSTER/llama.cpp/pull/46)).** The K-quant SwiGLU kernel now wins
  over AMD's int4 lowrow path (weights repacked to int4 at first use, activations quantized to
  int4), which also ran slowly under llama-server: 27B `llama-server` decode 10.7 -> 12.0 tok/s,
  KLD against CPU logits 0.00167 (AMD kernels 0.00318). #46 adds 2-8 token variants for speculative
  verify batches: Qwen3.8-27B with its MTP head (`1bit serve --mtp`, draft 3, p-min 0.75) decodes
  22.5 / 17.7 / 19.0 tok/s on code / prose / translation prompts (was 20.1 / 16.4 / 17.6; Vulkan
  34.6 / 24.3 / 24.5) with the same draft acceptance. A 4-token step still cost ~1.7x a 1-token
  step (f32 multiply-adds per token). Since #487 our HIP kernels run 2 to 8 drafted tokens on q8_1
  activations, reading each weight block once for every token: the 4-token verify step went from
  121.5 to 88.0 ms, and the 27B decodes at 36.0 / 26.8 / 29.9 tok/s on code / prose / short
  (from 25.9 / 20.5 / 21.9).
- **NaN after an MTP rollback on Qwen3.5/3.8
  ([llama.cpp #52](https://github.com/1bit-MONSTER/llama.cpp/pull/52)).** With `--mtp`, the gated
  delta-net kernel publishes one recurrent-state snapshot per verify token so a rejected draft can
  roll back. It formed each snapshot's decay ratio c_t/c_s by dividing two exponentials. When a
  strongly forgetting head drove both to 0, that gave 0/0 = NaN, and the next verify resuming from
  that snapshot returned NaN logits ("NaN logits at vocab index 0"). It depends on the prompt, and
  showed up on Qwen3.8-27B Q8_0, UD-Q5_K_XL and UD-Q6_K files. The same bug caused an AMDGPU memory
  fault on a repeated request without MTP. The ratios are now formed in log space, as the main
  recurrence already did. Q8_0 `--mtp` now runs clean on prose and code (15.7 / 20.2 tok/s, balanced power
  mode, 85 W); UD-Q4_K_XL `--mtp` speed and output are unchanged (also measured in balanced mode).
- **`1bit serve --device hrx --parallel N`.** With one KV stream per slot, the K/V tensors are
  4-D and HRX's flash attention fell back to the CPU in every layer (Qwen3-4B, 4 slots: 39 tok/s
  aggregate, below one stream). serve now passes `-kvu` (one shared KV cache) and turns off AMD's
  Qwen attention path (`qwen.attention`), which derives its mask from positions and let the slots'
  answers bleed into each other under `-kvu`. Qwen3-4B 4 slots: 153 tok/s aggregate (single stream
  76; Vulkan 216); Qwen3-Coder-30B-A3B 34 -> 74; ZAYA1-8B 47 -> 71. Hybrid gated delta-net models
  (qwen35, qwen35moe, qwen3next: Qwen3.5/3.8) take `--parallel` too since 2026-10-20: the gated delta-net
  matchers hold up to 8 sequences ([llama.cpp #117](https://github.com/1bit-MONSTER/llama.cpp/pull/117);
  [#48](https://github.com/1bit-MONSTER/llama.cpp/pull/48) fixed the two failures before it).
- **Fast sub-4-bit kernels on HRX.** IQ3_S has a decode kernel (#43) but not a fast prefill one. Q2_K
  ([llama.cpp #49](https://github.com/1bit-MONSTER/llama.cpp/pull/49)), IQ2_XXS/IQ2_XS
  ([llama.cpp #50](https://github.com/1bit-MONSTER/llama.cpp/pull/50)) and IQ3_XXS/IQ2_S
  ([llama.cpp #51](https://github.com/1bit-MONSTER/llama.cpp/pull/51)) and IQ1_S/IQ1_M
  ([llama.cpp #53](https://github.com/1bit-MONSTER/llama.cpp/pull/53)) run on HRX0: in the shared
  dequantizer (prefill WMMA, generic decode, GET_ROWS; IQ2_S was already there) and the K-quant
  decode kernels. Before them, llama.cpp's load-time buffer check (a 512-token matmul) failed, and
  every such weight went to the CPU. On Qwen3-4B, with KLD against the Q8_0 model equal to the CPU's:
  - Q2_K: 709 MiB CPU_REPACK and 217 graph splits -> all on HRX0 and 73 splits; pp512
    686 -> 1300 tok/s, tg128 47 -> 75.
  - IQ2_XXS: pp512 59 -> 715 tok/s, tg128 23.0 -> 39.5.
  - IQ2_XS: pp512 87 -> 268 tok/s, tg128 28.0 -> 30.2.
  - IQ3_XXS: pp512 2.8 -> 250 tok/s (mixed quant; 510-560 with every tensor IQ3_XXS), tg 1.5 -> 8.9
    (mixed) and 33.4 (pure), balanced power mode.
  - IQ2_S: tg 1.5 -> 11.3 tok/s (pure); its prefill is unchanged.
  - IQ1_S: pp512 70.8 -> 81.7 tok/s, tg128 14.1 -> 18.6 (balanced; the first decode run in a process
    includes about 18 s of kernel JIT for the IQ1 grid). IQ1_M: pp512 70.8 -> 80.4, tg128 14.1 -> 18.1.

  Some batched IQ2_XXS matmul shapes are still declined (not in the 2026-10-20 census); under strict
  placement, the default, such a shape refuses the load, and `--no-strict-device` runs it on the CPU. Mixed files decode
  slower than pure ones: a SwiGLU gate/up pair whose two formats need different grids shares one
  grid buffer, so it is declined and takes the generic path.
- **Q4NX on HRX.** Our Q4NX kernels lived in ggml-hrx2 and are not in ggml-hrx.
  They are kept on `1bit/hrx2-archive` until they are ported.
- **First-request cost.** Lemonade's telemetry for the first short HRX chat shows
  18 tok/s against llama-bench's 303. It is probably first-use kernel
  compilation, not steady state, but this is not yet separated.
- **CI.** The hosted CI has no AMD GPU and no TheRock. `ONEBIT_HRX=ON` is built at PR time by
  `hrx-build.yml` on the Strix Halo runner (docs/hw-gate.md, "The HRX build check"), and run on
  the hardware by the gate.
