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

| Task | Kernel | Where it runs in the engine |
|---|---|---|
| [`gdn-prefill/`](gdn-prefill/) | gated delta-net prefill (head size 128, optional rollback snapshots) | Qwen3.8 / Qwen3.6 prompt processing on ROCm, from ROCmFPX's `gated_delta_net.cu` |

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

A kept kernel is only a candidate. It goes back into ROCmFPX's `gated_delta_net.cu` by hand and
has to pass `test-backend-ops -o GATED_DELTA_NET`. Then it is measured through `1bit serve`
with [`tools/bench.py`](../../docs/bench.md).

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

The driver now draws gates from ggml's own test range. Two setup traps:

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
