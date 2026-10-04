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

# KernelForge tasks for gfx1151

[KernelForge](https://github.com/AMD-AGI/Hyperloom) (AMD, MIT, part of ROCm Hyperloom) runs an agent that optimizes one GPU
kernel at a time. Each iteration proposes one change. KernelForge builds it, gates it on a
correctness check, benchmarks it, keeps it only if it is faster, and reverts it otherwise.

KernelForge is built for AMD Instinct GPUs (MI300X, MI355X). These tasks point it at Strix
Halo's gfx1151 instead. Each task packages one of the engine's kernels with:

- a standalone copy of the kernel;
- a float64 torch oracle;
- a driver that follows KernelForge's contract;
- a `program.md` that tells the agent what differs on this GPU.

| Task | Kernel | Where it came from |
|---|---|---|
| [`gdn-prefill/`](gdn-prefill/) | gated delta-net prefill (head size 128, optional rollback snapshots) | Qwen3.8 / Qwen3.6 prompt processing; from ROCmFPX's `gated_delta_net.cu` |
| [`gemv-q4_0/`](gemv-q4_0/) | Q4_0 x Q8_1 for 1-8 columns (`mul_mat_vec_q`) | speculative verification (and plain decode); from ROCmFPX's `mmvq.cu` |

The engine no longer builds ROCmFPX's ROCm llama.cpp (RFC #213 stage 3). Both tasks are
standalone HIP kernels, so they still run; a kept kernel's destination is now an HIP kernel
dispatched through HRX in our llama.cpp fork, not the ROCm backend. The results below were
measured on the ROCmFPX build and are kept as the record.

## Run one

You need three things on the machine with the GPU:

- a ROCm torch (strixhalo: `~/torch-rocm`);
- Hyperloom installed into that Python: `pip install -e '.[forge]'` in a checkout;
- a `claude` CLI that is logged in. KernelForge drives its agent through that CLI, and a
  campaign bills that account for its whole run.

Then:

```sh
tools/kernelforge/gdn-prefill/run.sh /tmp/forge-gdn 2      # workspace, hours
```

`run.sh` copies the task into a fresh git workspace and runs `kernelforge forge-loop` there. The
loop edits the kernel in place and commits each kept change to the `forge-optimize` branch.
When it finishes:

- the best kernel is checked out in the workspace;
- every candidate, measurement and profile is under `forge_experiments/`.

A kept kernel is only a candidate. It went back into ROCmFPX's `gated_delta_net.cu` by hand and
had to pass `test-backend-ops -o GATED_DELTA_NET`. Now it goes into the HRX dispatch (with its
matcher), passes `test-backend-ops -b HRX0`, and is measured through `1bit serve` with
[`tools/bench.py`](../../docs/bench.md).

## Results

**gdn-prefill, 2026-09-29.** A 2-hour campaign, about $33 of Claude usage, kept 2 changes: 7.47x on
the task's own benchmark (2.95 -> 0.40 ms). One part of them was wrong. A gate-factored state
turned NaN with hard gates: about 1.1 million non-finite values with log-gates in [-20, 0). The
task driver's gentler gate range had let it pass. The rest went into ROCmFPX#5:

- DPP reductions on RDNA;
- two state columns per lane;
- rows interleaved across banks;
- 16-byte staging.

| Result | Before | After |
|---|---|---|
| One layer, 512 tokens | 3.02 ms | 0.41 ms (7.4x) |
| Same, with 8 snapshots | 2.46 ms | 0.49 ms (5.0x) |
| Qwen3.8-27B-H32 serve prompt | 484-495 tok/s | 542-555 tok/s |
| Same, with DFlash2 | 456-465 tok/s | 510-522 tok/s |

**gemv-q4_0, 2026-09-29.** A 2-hour campaign, about $40, kept a row-tiled body for 5-8 columns:

- 4 rows per warp;
- activations loaded once per k-block;
- weights unpacked once for all columns;
- a reduce-scatter epilogue.

The mean case score is 1.18x. Byte for byte it gives the old kernel's output: 40 cases, with odd
row counts, extreme scales, edge nibbles, and saturated and zero activations. The task's driver
now carries those adversarial inputs itself.

| Result | Before | After |
|---|---|---|
| 5120 x 17408, 8 columns | 0.298 ms | 0.226 ms |
| Qwen3.8-27B pp8 | 71.3 tok/s | 78.3 tok/s |
| Serve + DFlash2 decode, code (median, 2 rounds) | 43.0 / 39.4 | 46.1 / 45.7 |
| Serve + DFlash2 decode, prose (median, 2 rounds) | 20.2 / 25.9 | 28.1 / 28.1 |

In the full model it helped only at 8 columns (5-7 were level or slower), so ROCmFPX#6 dispatches
it for exactly 8.

The gdn-prefill driver now draws gates from ggml's own test range. Two setup traps:

- **Login.** KernelForge runs its planning and implementer sessions as `claude --bare`, which
  ignores a logged-in subscription and accepts only an API key. Our Hyperloom checkout on strixhalo
  carries a local switch, `FORGE_CLAUDE_OAUTH=1`, that drops `--bare`.
- **Stale process check.** `pgrep -f "kernelforge forge-loop"` also matches the ssh command that
  runs it.

## Check a task without KernelForge

```sh
cd tools/kernelforge/gdn-prefill
~/torch-rocm/bin/python driver.py                 # correctness: SNR per case against float64
~/torch-rocm/bin/python driver.py --bench-mode    # case_ms: median kernel time per case
```
