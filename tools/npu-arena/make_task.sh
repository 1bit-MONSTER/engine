#!/bin/bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# make_task.sh -- create or refresh a task bundle for one phase.
#   make_task.sh GU|D|QKV|O|Z
# The bundle is self-contained: kernel/ holds the source of the current leader
# (the shipping per-role design: generators/n1_core_i8_v27.py + its microkernel),
# so a competitor starts from working code and only has to beat it.
set -u
ARENA=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$ARENA/lib/env.sh"
T=${1:?usage: make_task.sh GU|D|QKV|O|Z}
arena_require "$GENERATORS/n1_core_i8_v27.py" "$GENERATORS/mm_kernel_reference.cc" \
              "$GENERATORS/elf_emit.sh" || exit 1
case "$T" in
    GU)  K=5120;  N=17408 ;;
    D)   K=17408; N=5120  ;;
    QKV) K=5120;  N=10240 ;;
    O)   K=6144;  N=5120  ;;
    Z)   K=5120;  N=6144  ;;
    *) if [ -n "${K_OVERRIDE:-}" ] && [ -n "${N_OVERRIDE:-}" ]; then K=$K_OVERRIDE; N=$N_OVERRIDE
       else echo "unknown task $T (set K_OVERRIDE/N_OVERRIDE for a toy)"; exit 2; fi ;;
esac
D=$ARENA/tasks/$T
mkdir -p "$D/kernel" "$D/out"
cp -n "$GENERATORS/n1_core_i8_v27.py" "$GENERATORS/mm_kernel_reference.cc" \
      "$GENERATORS/elf_emit.sh" "$GENERATORS/silu_quant.h" "$GENERATORS/zero.cc" \
      "$D/kernel/" 2>/dev/null
cat > "$D/kernel/GEOMETRY" <<EOF
# frozen by make_task.sh; the oracle reads K and N from the harness, this is for humans
TASK=$T
K=$K
N=$N
M=16
MICROKERNEL=m=16 k=64 n=128
COLUMNS=8
CORE_ROWS=1
FIFO_DEPTH=6
EOF
cat > "$D/kernel/build.sh" <<'EOF'
#!/bin/bash
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# build.sh -- emit out/<TASK>.elf from the sources in this directory.
# Contract: run with cwd = this directory, no network, no arguments; write the
# full ELF to out/<TASK>.elf.  The oracle copies the whole submission into a
# scratch dir first, so anything you generate outside out/ is ignored.
#
# This starting point is the shipping decode design: the M=16 twin of the
# per-role geometry (microkernel m=16 k=64 n=128, 8 columns, 1 core row, fifo
# depth 6), one launch covering the whole N via the in-ELF column-group loop.
set -u
ARENA="${ARENA:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
. "$ARENA/lib/env.sh"
TASK=$(sed -n 's/^TASK=//p' GEOMETRY)
K=$(sed -n 's/^K=//p' GEOMETRY)
N=$(sed -n 's/^N=//p' GEOMETRY)
mkdir -p out
work=$(mktemp -d /tmp/arena-build.$$.XXXXXX); trap 'rm -rf "$work"' EXIT
arena_aie_env
set -e
"$PEANO/bin/clang++" --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=16 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \
    -isystem "$PEANO/include/c++/v1" -I "$AIE_API_HEADERS" -I "$AIE_KERNELS_INC" \
    -c mm_kernel_reference.cc -o "$work/mm_32x64x128.o"
python3 n1_core_i8_v27.py -M 16 -K "$K" -N "$N" -m 16 -k 64 -n 128 -c 8 -r 1 -b 5 \
    > "$work/d.mlir"
# aiecc resolves link_with objects next to the design, and the design is in $work
. ./elf_emit.sh
emit_full_elf "$MLIR_AIE_INSTALL/bin/aiecc" "$PEANO" "$MLIR_AIE_INSTALL" \
    "$work/d.mlir" "$PWD/out/$TASK.elf"
[ -s "out/$TASK.elf" ] || { echo "build failed"; exit 1; }
echo "built out/$TASK.elf $(stat -c%s "out/$TASK.elf") B"
EOF
chmod +x "$D/kernel/build.sh"
# ---- task documentation (regenerated every time; edit make_task.sh to change) ----
cat > "$D/README.md" <<EOF
# Task $T — one launch of the $T projection, Bonsai-2-27B decode on XDNA2

| | |
|---|---|
| geometry | K=$K, N=$N, activation tile M=16 (the decode twin; production decode fills 1 row, a short-prompt prefill chunk up to 16) |
| B streamed | $(( K * N )) B ($(python3 -c "print(f'{$K*$N/1e6:.2f}')") MB) of packed int8 weights, one launch |
| A + C | $(( 16 * K )) B in, $(( 16 * N * 4 )) B out |
| current leader | the shipping M=16 twin: microkernel m=16 k=64 n=128, \`-c 8 -r 1\`, fifo depth 6, whole N in one in-ELF column-group loop |
| measured | see LEADERBOARD.md (3.16–3.25 GB/s in the engine's own profile) |

## Roofline (vai-peak-bandwidth-analysis method, applied to the numbers we have)

Achieved bandwidth = bytes / time. There is no AI Analyzer capture for a full-ELF design, so the
side attribution comes from the design itself and from \`kern-exp2/VERIFIED-DESCRIPTORS.md\`:
every 8 KiB B tile is emitted as \`[<8,139264>,<16,8>,<8,17408>,<8,1>]\` = **64 runs of 128
contiguous bytes, row stride 17,408** — i.e. a short burst per descriptor, not a stream.

| point | bytes/s | ms for this phase | verdict |
|---|---|---|---|
| this design (M=16 twin) | 3.2 GB/s | 27.9 | **memory-bound, DDR side, latency/burst-limited** |
| the fused per-layer gate design (\`kern-sealed/1024x3072x16x8\`) | ~24 GB/s | 3.7 | what "good" looks like on this array |
| practical LPDDR5X ceiling for the array | ~150 GB/s | 0.6 | not reachable from a single 8-column pass |
| int8 compute peak of the array | ~196 TMAC/s | <0.1 | irrelevant: this phase needs 2·M = 32 int8 ops per weight byte |

Arithmetic intensity is $(( 2 * 16 )) int8 ops per weight byte, so the phase is memory-bound by
two orders of magnitude: **only bytes/second can move the score.** Nothing here is compute-bound,
and \`kern-exp2\` already showed the issue/await discipline is worth +7% at most (2.86 → 3.07 GB/s)
and that fifo depth 7+ hangs the array — so the levers are the descriptor shape, the number of
cores actually streaming B, and how many of the 8 columns each launch keeps busy.

## What a submission may change

Anything that keeps the int8 x int8 -> int32 result byte-exact: tile geometry (m/k/n), column and
core-row placement (\`-c\`, \`-r\`), fifo depth (\`-b\`), the generator's DMA issue/await discipline
(\`n1_pipe.py\`'s windowed ring is the preserved experiment), RTP-driven shape parameters instead of
per-shape ELFs (arXiv 2504.03083), and DMA-side layout transforms instead of host repacks. What it
may not change: the packed int8 B the harness feeds, the microkernel's numerics, or the harness.

## How to run

    python3 ../../harness/gen.py $T   # once: freezes harness/ (already frozen)
    ../../oracle.sh $T .              # build + byte-exact + GB/s
EOF
cat > "$D/SKILLS.md" <<'EOF'
# Skills that apply to this task

Read these before editing the kernel (the first two are required by the lane rules; they are
linked in `~/.pi/agent/skills/`, AMD Ross AI assistant 2026.10.1):

| skill | what to take from it here |
|---|---|
| `vai-aie-kernel-development` | VLIW slot awareness, DM bank annotation (A/B/C/D), zero-overhead loops, `aie::mmul`/AIE API primitive choice from `aie_api_references.md`, register-pressure loop splitting, in-place updates. Its rule "prefer the AIE API over hand-rolled intrinsics" applies directly to the microkernel. |
| `vai-aie-compiler-oriented-optimizations` | `__restrict` on the loop's pointers (the single biggest schedule unlock), `noinline` core + inline wrapper, `AIE_LOOP_RANGE` trip counts, pre- vs post-RA software pipelining (`-Rpass` remarks, `-Wpass-failed`), `AIE_LOOP_UNROLL`, store-width/packing rules. Read the "inline wrapper + noinline core" pattern before touching the inner loop. |
| `vai-peak-bandwidth-analysis` | One method only, but it is the scoring method: achieved BW = bytes/time, classify the stall side (read starvation / write backpressure = DDR, read backpressure / write starvation = L2), then pick the lever family. For this task the classification is "DDR, latency-limited": family B (hide latency: real ping/pong on the B stream, deeper descriptors) and family C (spread the load across all 8 columns/channels), not family A (there are no bytes to cut). |
| `vai-perf-analysis` | Reference only; it assumes VAIML-compiled models, which these full-ELF designs are not. |
| `~/wt/npu-forge2` + `~/wt/ross-npu-product/nk` (`npu-kernels`) | the design generators (`npu/dx/designs/*.py`), `tools/dx_model.py` reference, and the fused per-layer design the sealed set was built from. |
| arXiv 2504.03083 (GPT-2 GEMMs on Phoenix XDNA) | one tile size for every GEMM shape, shape passed as runtime parameters (RTP) instead of a per-shape ELF, DMA-side layout transforms. A submission that folds the five phase tasks into one parameterised ELF is in the spirit of the task. |

Lane rules that still apply: Peano + Loom AIE2P only (never xchesscc/aietools), Apache header on
every new file, our code in our own files.
EOF
echo "task $T ready: $D (K=$K N=$N)"
