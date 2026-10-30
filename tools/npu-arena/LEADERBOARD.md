# Leaderboard

Scoring: `oracle.sh <PHASE> <submission>` — rebuild from source with the pinned toolchain,
byte-exact int32 against the frozen harness on 3 seeds, then GB/s = bytes of B streamed / median
of 5 launches on seed 1, with the current leader replayed first in the same box state. A row only
exists if the oracle printed ACCEPT; the "why" paragraph is written by hand and must explain both
the correctness argument and the reason it is faster.

Toolchain for every row: `mlir-aie@465853266718`, `peano21-c9c5ecb` (see TOOLCHAIN.lock).

## GU — gate/up projection, K=5120 N=17408, M=16 (89.13 MB streamed per launch)

| # | date | tree sha | GB/s | ms/launch | delta | technique |
|---|---|---|---|---|---|---|
| 2 | 2026-10-30 | `c463050957c4b0a6` | **8.13** | **10.97** | **+145%** vs #1 | **pre-blocked B**: the weight buffer is written as whole 8,192-byte tiles in the 8x8-microtiled order the microkernel already expects (tile-n-major), so each B tile is ONE contiguous DMA instead of 64 runs of 128 bytes strided by N. No microkernel change, no core-count change, same bytes. |
| 1 | 2026-10-30 | `0e806769ae6be499` | 3.32 | 26.85 | — (starting leader) | shipping M=16 twin: m=16 k=64 n=128, `-c 8 -r 1 -b 5`, in-ELF column-group loop |

Harness: frozen ternary (one magnitude per 32-row group, 0/+mag/-mag, a third of each column
zero — the container's real structure). Row 1 measured on it, all 3 seeds byte-exact plus the
5-launch timing run (`submissions/GU/leader.elf`, md5 `b662765e…` = the ELF the engine ships).
Earlier on the first (dense-column) harness the same ELF scored 3.13–3.19 GB/s; the harness was
re-frozen as ternary *before* any entry was accepted, because a dense column would have made the
lossless 2-bit packing lever look impossible.

Reference points for this phase: the fused per-layer gate design streams ~24 GB/s (3.7 ms for the
same 89.13 MB), the array's practical LPDDR5X ceiling is ~150 GB/s.

**Why #1 is correct and is the number to beat.** `GU.elf` built from `tasks/GU/kernel` by the
oracle is md5-identical (`b662765e9c497d227e3150ca15a6d870`) to the ELF the engine ships as the
decode twin of this role, and the engine's own bit-exact gate compared that kernel against the
M=128 design on 7,295 launches with `mismatched=0, worst int32 maxdiff=0` on identical quantized
activations and the same weight BO. It is *not* fast because it is a good streaming design: it
streams 8 KiB B tiles, each emitted as 64 runs of 128 contiguous bytes with row stride 17,408
(the descriptor survey in the bonsai-speed working notes (`kern-exp2/VERIFIED-DESCRIPTORS.md`, outside this repo)), which is why 89.13 MB takes 27.9 ms.

**Why #2 is correct and 2.45x faster.** `layout.sh` is a pure permutation of the frozen harness B
(verified: identical multiset of bytes, and three spot tiles round-trip exactly under the
microkernel's own `b_in[((kb*nb + nb)*8 + r)*8 + c]` indexing), so the maths is untouched and the
oracle checks it byte-exact on 3 seeds plus the timed run. The gain is the DMA descriptor shape that
`kern-exp2/VERIFIED-DESCRIPTORS.md` predicted and PLAN v2 §"Non-goals" wrongly dismissed: in the
row-major layout every 8 KiB tile arrives as `[<8,139264>,<16,8>,<8,17408>,<8,1>]` = 64 bursts of
128 contiguous bytes, and no amount of run-length tuning inside that layout helped; pre-blocking the
weights makes the same 8 KiB one contiguous burst (`strides=[1024,64,8,1]`), worth **2.45x**.
Engine-wide this is the whole-token lever: 25.60 GB/token at 8.13 GB/s = 3.15 s instead of 8.87.

### Rejected / not-a-technique so far

| attempt | result | why |
|---|---|---|
| `-b 4` (fifo depth 5, `1e081e75cd242581`) | 3.18 vs 3.13 GB/s = +1.6% | noise: within run-to-run spread, and the opposite sign to the M=128 measurement in FINDINGS §7 (−1.6%). Not promoted. |
| `-c 8 -r 4` (all 32 cores at M=16) | **build fails by construction** | `n1_core_i8_v27.py:69` asserts `(M//m) % n_aie_rows == 0`; at M=16 with m=16, `M//m = 1`, so the generator can only ever place ONE core row. Using 32 cores at M=16 requires the rows to partition **N** (column groups) instead of M — a generator change, and the first real task for a competitor. |
| identity `layout.sh` (`a2701dfe91950861`) | 3.22 vs 3.20 GB/s | plumbing demo for the layout hook, not a technique (it copies the harness B unchanged). |

## D — down projection, K=17408 N=5120, M=16 (89.13 MB)

| # | date | tree sha | GB/s | ms/launch | delta | technique |
|---|---|---|---|---|---|---|
| 2 | 2026-10-30 | `405750d9c284eedd` | **8.25** | **10.80** | **+159%** | the same pre-blocked B as GU #2 — and it reproduces on the transposed geometry (long K, short N), which is the proof that this is a layout/descriptor lever and not a quirk of one shape |
| 1 | 2026-10-30 | `4c4a591c931d5105` | 3.19 | 27.98 | — (starting leader) | shipping M=16 twin, row-major B |

The phase exists separately from GU because its long axis is K, so the A stream and the k-loop order
differ; blocked B wins on both.

## QKV — 5120 x 10240, M=16 (52.43 MB)

| # | date | tree sha | GB/s | ms/launch | delta | technique |
|---|---|---|---|---|---|---|
| 2 | 2026-10-30 | `e18078a81ce39812` | **8.15** | **6.44** | **+143%** | pre-blocked B (same transform as GU #2) |
| 1 | 2026-10-30 | `3739cbbf39b43a21` | 3.35 | 15.67 | — (starting leader) | shipping M=16 twin, row-major B |

## O — 6144 x 5120, M=16 (31.46 MB)

| # | date | tree sha | GB/s | ms/launch | delta | technique |
|---|---|---|---|---|---|---|
| 2 | 2026-10-30 | `434aecd451f84a5d` | **8.15** | **3.86** | **+156%** | pre-blocked B (same transform) |
| 1 | 2026-10-30 | `3938246286acb818` | 3.19 | 9.86 | — (starting leader) | shipping M=16 twin, row-major B |

## Z — 5120 x 6144, M=16 (31.46 MB)

| # | date | tree sha | GB/s | ms/launch | delta | technique |
|---|---|---|---|---|---|---|
| 2 | 2026-10-30 | `c9bcb0977606c4c3` | **8.36** | **3.76** | **+143%** | pre-blocked B (same transform) |
| 1 | 2026-10-30 | `7e45893bbd7f98f1` | 3.44 | 9.14 | — (starting leader) | shipping M=16 twin, row-major B |

### All five phases, one technique (summary)

| phase | B streamed | stock | blocked B | speedup |
|---|---|---|---|---|
| GU | 89.13 MB | 3.32 GB/s / 26.85 ms | **8.13 / 10.97** | 2.45x |
| D | 89.13 MB | 3.19 / 27.98 | **8.25 / 10.80** | 2.59x |
| QKV | 52.43 MB | 3.35 / 15.67 | **8.15 / 6.44** | 2.43x |
| O | 31.46 MB | 3.19 / 9.86 | **8.15 / 3.86** | 2.55x |
| Z | 31.46 MB | 3.44 / 9.14 | **8.36 / 3.76** | 2.43x |

Whole-token effect: 25.60 GB per decode token at 8.2 GB/s = **3.12 s/token** instead of 8.87 s, from a
weight-layout change with no microkernel edit. (The fused per-layer route would stack on top of it.)

**Plumbing lesson, recorded because it cost an hour:** the three small phases were first reported as
"blocked B does not move them". They had a stale `LAYOUT` (`W_BYTES` copied from the GU bundle), so
the runner refused the layout buffer, the submission never ran, and the oracle reported the *leader's*
line (its RESULT file was appended to by both the leader replay and the submission). Fixed on both
sides: every submission now computes its own `W_BYTES`, and `oracle.sh` truncates the RESULT file
before the submission runs so a leader line cannot be mistaken for a submission's. Any future
submission must make `W_BYTES` match what its `layout.sh` writes — the runner exits 3 if it does not,
and that failure is now impossible to misread.
