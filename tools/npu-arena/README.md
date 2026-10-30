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
# NPU kernel arena — Bonsai-2-27B single-launch phases on the XDNA2 NPU

A competition bench for ONE kernel at a time, in the shape Quadric's Kernel Arena and GEAK's
task-dir contract use: a frozen task bundle anybody can build, a single oracle that rebuilds a
submission from source, proves it byte-exact on the device and scores it in GB/s, a leaderboard
you must beat from its own code, and a harvester that turns each win into a source change.

Why here: a phase's score only means something as a fraction of what this box can actually stream
for that phase's bytes.  So the arena measures that ceiling first (`ceiling.sh`), and every entry is
judged against it rather than against another design class.  The layout that the engine ships today
(`q4nx_weight_layout: "b8"`, npu/forward/include/q4nx_weight_layout.h) came out of this harness, and
the remaining work is the stream structure (all 32 cores, N partitioned across rows) and a packed
2-bit ternary weight stream; the measured records and the probe kernel live in the private kernel
tree (npu-kernels) with the rest of the ELF material.

`ceiling.sh` (phase 1) runs the per-phase DMA ceiling probe: the phase's own objectfifos, taps,
blocked layout, tile geometry and column-group loop, with the matrix unit replaced by a
pass-through microkernel (`probe/mm_dmabw.cc`).  Two variants: `nm16` (8 cores, the shipped decode
twin's shape) and `r4` (M=64 / 4 core rows = 32 cores, the same B taps with B broadcast down the
rows).  Its output is `CEILING.md`; every later leaderboard row reports "% of ceiling".

## Layout

    TOOLCHAIN.lock        the pinned toolchain (Loom mlir-aie + Peano + generators + runtime lib)
    lib/env.sh            sources TOOLCHAIN.lock; no network, no fallbacks
    runner/runner.cpp     the on-device scorer: loads one ELF, byte-exact check, median of 5
    harness/gen.py        freezes a task's int8 inputs / expected int32 (3 seeds)
    make_task.sh          creates a task bundle
    tasks/<PHASE>/        kernel/ (the leader's source), harness/ (frozen), README.md, SKILLS.md
    oracle.sh             rebuild from source -> byte-exact -> GB/s
    submissions/<PHASE>/  leader.elf + leader-src (the code a submission must start from)
    (records)             the leaderboard and harvest tables live in the private kernel tree

Phases for the 27B geometry (H=5120, IM=17408), decode tile M=16:

| task | K | N | B bytes | share of the token stream |
|---|---|---|---|---|
| GU | 5120 | 17408 | 89.13 MB | 67% (G+U+D; run once here for G and once for U) |
| D  | 17408 | 5120 | 89.13 MB | (same 67%; separate phase because its K is the transposed one) |
| QKV | 5120 | 10240 | 52.43 MB | 10% |
| O  | 6144 | 5120 | 31.46 MB | 2% |
| Z  | 5120 | 6144 | 31.46 MB | 6% |

## Using it

    ./make_task.sh GU                 # (re)create the bundle from the pinned generators
    python3 harness/gen.py GU         # freeze the harness once (--force re-freezes; don't)
    ./oracle.sh GU tasks/GU/kernel    # build + prove + score the current leader
    cp -a tasks/GU/kernel /tmp/my-gu && $EDITOR /tmp/my-gu/...   # a submission
    ./oracle.sh GU /tmp/my-gu        # score it

The oracle refuses to run while a serve or another process holds `/dev/accel/accel0`, wraps every
device run in `gpu-hold -l "npu arena"`, and prints the leaderboard row on ACCEPT.

## Rules

1. **A submission starts from the leader's code** (`submissions/<PHASE>/leader-src`) and must be a
   change to it; `oracle.sh` builds `submissions/<PHASE>/leader.elf` first every time, so a score is
   always from the same box state as the thing it beats.
2. **Byte-exact or nothing.** int8 x int8 -> int32 is exact, so the whole C tile must match the
   frozen expected tensor on 3 seeds. No tolerance, no rel_rms, no "close enough": this is a GEMM.
3. **Source-built.** The oracle copies the submission to a scratch dir and rebuilds it with the
   pinned toolchain. A hand-copied ELF is not a submission.
4. **Same bytes.** A submission may not shrink the weight stream by changing the data format
   (the harness's B is the engine's packed int8 B). Changing *how* those bytes are moved is the
   whole point; changing *what* is computed is out of scope for this arena.
5. Accepted entries and the technique behind each one are recorded in the private kernel tree
   (npu-kernels: `LEADERBOARD.md`, `HARVEST.md`), and the winner becomes the new `leader.elf` so the
   next entry has to beat it.

## Known risk the oracle carries

The runtime's readback of a result BO is occasionally short (zeros for a few percent of the tile)
with the kernel unchanged and the inputs identical — the engine records the same finding in
`npu/forward/engine-src/npu_engine_i8ctx_inc.h` ("bC is identical immediately before each launch,
and yet what the host gets back from it is not", RESULTS-coverage-multifamily 68/82). The runner
therefore requires **two consecutive clean reads** before it reports PASS, and the oracle runs
3 seeds plus a timing run. This window was observed 11:40–12:15 on 2026-10-30 and not reproducible
after 12:19; if a submission ever fails with `got=0` on a *stable* set of elements, that is the
runtime, not the kernel — re-run before believing it.
