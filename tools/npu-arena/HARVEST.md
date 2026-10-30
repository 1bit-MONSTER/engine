# Harvest — accepted wins as techniques

Every accepted leaderboard entry gets a row here: the technique in one sentence, the number it
bought, and **which source file should absorb it**. The files that can absorb an arena win are:

| file | what it holds |
|---|---|
| the ELF tree's `npu/generators/n1_core_i8_v27.py` | the per-role ELF generator (tile geometry, column/core placement, fifo depth, DMA issue/await discipline) — the emitter the engine's `_m16` twins come from |
| the ELF tree's `npu/generators/mm_kernel_reference.cc` | the microkernel (`aie::mmul` wrapper, DIM_M/K/N specialisations) |
| the ELF tree's `npu/generators/emit_m16_elfs.sh` | the recipe that emits the decode family (flags, microkernel build, ELF naming) |
| the working-notes `kern-exp2/gen/n1_pipe.py` (outside this repo) | the preserved windowed issue/await experiment (+7% measured) |
| `npu-build/gguf_to_q4nx_qwen35.py` | the converter — the only file that may change the *weight layout* (a DMA-side layout transform belongs here, per arXiv 2504.03083) |
| `npu/dx/designs/layer1.py` (npu-kernels) | the fused per-layer design the sealed 24 GB/s set was built from — the model for a phase-fused ELF |
| `npu/forward/src/q4nx_forward.cpp` | how the engine picks and launches a kernel (the consumer of any new artifact) |

Nothing is harvested yet: entry #1 on the GU leaderboard is the shipping design, recorded as the
starting point rather than as a win. The first harvested technique will be the one that beats it.

## Findings from setting the arena up (each one is a work item, not a win)

**F0 (answered by harvest #2): the 3.2 GB/s ceiling was the descriptor burst length, not the core
count or the await discipline.** Pre-blocking B took GU to 8.13 GB/s with the same 8 cores, same
microkernel, same bytes. So the earlier "descriptor shape is the wall" diagnosis was right and the
"B-repack is worth ≤1.1x" non-goal in PLAN v2 was an artefact of only sweeping run length *inside*
the row-major layout.

**F1. The M=16 twin can only ever use 8 of the 32 cores.** `n1_core_i8_v27.py` derives the core-row
count from the M-tile count (`assert (M//m) % n_aie_rows == 0`), so at M=16/m=16 there is exactly
one M-tile and `-r 4` cannot be placed. At M=128 the same relation is why `-r 1` was 0.78 GB/s and
`-r 4` was 2.85 (linear in cores). To use all 32 cores in decode the design has to partition **N**
across core rows instead of M. This is the first concrete lever the arena can score, and it is a
generator change (`n1_core_i8_v27.py` + `emit_m16_elfs.sh`), not a flag.

**F2. Per-core rate is not the limit at M=16:** 8 cores reach 3.2 GB/s = 400 MB/s/core, 4.4× what
the M=128 design gets per core (89 MB/s). So M=128 was compute-bound *per core* and M=16 is
DMA-bound — which means F1's 4× core count will only pay if the shared DDR/DMA side has headroom
left, the open question the next submission answers.

**F3. The remaining structural lever is the weight bytes, and it is now scoreable.** The device
streams 1 byte per weight (the int8 pack), 25.60 GB per decode token, for a model whose weights are
*ternary*. A lossless 2-bit device format (sign/magnitude per 32-row group, which is exactly the
structure the harness now freezes) would cut that to ~6.4 GB/token — 4× — and the arena's
`layout.sh` + `W_BYTES` hook scores it as bytes moved, with byte-exactness still pinning the maths.
This needs a microkernel that unpacks in-core (there is ~100× compute headroom) and a converter
change in `npu-build/gguf_to_q4nx_qwen35.py`. Bigger than any descriptor tuning.

**F4. The runtime intermittently returns a short result BO** (zeros for a few percent of the tile,
identical inputs, unchanged ELF; 2 of 7 seed runs at 12:50, none at 12:56). Same finding as
`npu/forward/engine-src/npu_engine_i8ctx_inc.h` sections 68/82. The runner now retries the launch
(up to 3) and needs two consecutive clean reads, and the oracle requires every seed *and* the timed
run to pass. Flagged to the orchestrator: production decode reads C the same way.

## Harvested #2 (GU, 3.32 -> 8.13 GB/s, +145%)

**Technique: pre-block the weights into the microkernel's own tile order, then move each tile as one
contiguous burst.** In the engine's row-major packed B every 8,192-byte tile is delivered by a
4-level tap `sizes=[8,16,8,8] strides=[8N,8,N,1]`, i.e. 64 runs of 128 contiguous bytes with a
17,408-byte row stride between them. Rewriting B at pack time as `[n_tile][k_tile][kb][nb][8][8]` —
exactly the `[kb][nb][8][8]` image the microkernel already indexes (`mm_kernel_reference.cc`,
"B LAYOUT (silicon-verified)") — turns the same tile into `sizes=[8,16,8,8]
strides=[1024,64,8,1]`: one contiguous 8 KiB burst, no DDR row-hopping. Byte-identical maths, 2.45x.

Absorb it in this order:
1. `npu-build/gguf_to_q4nx_qwen35.py` (converter): emit the blocked order, or add a pack-time
   transform. This is the file that owns the weight layout, and the arena's `layout.sh` is the
   reference implementation (14 lines of numpy).
2. `npu/forward/engine-src/npu_engine_i8ctx_inc.h` `packB_blocked()`: it already writes a padded
   `[KD][ND]` row-major image — it must write the blocked image instead, and `make_weight_bo` sizes
   stay the same (the bytes are a permutation).
3. `n1_core_i8_v27.py` + `emit_m16_elfs.sh`: the B tap becomes the contiguous one
   (`--blocked-b` in the arena's `submissions/GU/cand-blocked-b/n1_blocked.py` is the diff).
4. Then re-run the engine's own `NPU_M16_CHECK` gate: the twin vs M=128 comparison stays valid
   (both read the new layout), and the greedy text gate must not move.

Note for whoever does (1): the offsets `b_off = (n_tile*n_k + ki) * k*n` assume tile-n-major order,
which is what makes a core's inner k-loop read forward through memory; keep it.

## Harvested #2 note (D): the win reproduces on the transposed geometry

D (K=17408, N=5120) went 3.19 -> 8.25 GB/s with the identical layout transform, so the harvest order
above does not depend on which axis is long: the converter should emit blocked tiles for every
weight tensor, not just the K=H ones.

**It pays on every phase**, including the short ones — GU 2.45x, D 2.59x, QKV 2.43x, O 2.55x,
Z 2.43x, all at 8.1-8.4 GB/s, all byte-exact on 3 seeds plus the timed run. (An earlier "it does not
pay on the small phases" reading was a stale `W_BYTES` in those bundles making the runner refuse the
layout buffer; the oracle then reported the leader's own line. Both are fixed: `W_BYTES` is per
submission and the RESULT file is truncated before the submission runs.)

So the whole-token arithmetic is now: 25.60 GB / 8.2 GB/s = **3.12 s/token** from the layout change
alone, against 8.87 s today and 1.07 s for the fused 24 GB/s route. Absorb it everywhere.

## Open questions the harvester is waiting on

1. **Does the 3.2 GB/s ceiling come from the descriptor burst length (128 B runs) or from the
   column-loop barrier?** `kern-exp2`'s contiguity sweep says runs past 256 B buy nothing and
   depth ≥7 hangs, so if neither moves, the remaining difference to the sealed design has to be
   the fused per-layer structure (one launch, five phases, 5,120-byte DX4 tiles, objectfifo depth
   2) and not a parameter.
2. **Can one ELF hold the shape as runtime parameters** (arXiv 2504.03083: one m=64/k=64/n=32 tile
   for twelve GEMM shapes, RTP for the rest, 3.5x cheaper than reconfiguring the array)? If yes, the
   five phase tasks collapse into one artifact and the engine's per-role dispatch disappears.
3. **What does a real double buffer of the B stream cost in DM?** The m16 core has 8 KiB of tile
   budget per core; the sealed design moves 5,120-byte tiles, which is what makes ping/pong fit.
