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

# How the numbers are measured: tools/bench.py

`tools/bench.py` measures `1bit serve` configurations against each other. It is how the serve
numbers in these docs are taken, and how a [recipe](recipes.md) earns its place.

```sh
tools/bench.py -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
    --config base= \
    --config dflash='--dflash Qwen3.8-27B-DFlash2-q8_0.gguf' \
    --rounds 2 --lock ~/.cache/lax-decode/box.lock --json one-server.json
```

## Method

- **The tool runs every measurement.** Each configuration gets its own fresh `1bit serve`,
  started, measured and stopped by the tool. A number in a table is one the tool measured, not
  a prediction or an earlier run.
- **The first `--config` is the baseline**, measured in the same run as the others. Every other
  row reports its change against that baseline's median.
- **Interleaved rounds.** With `--rounds N` the configurations run A B A B ..., so drift on a
  shared box (thermal, another job on the memory bus) hits all of them alike.
- **One lock for a shared box.** `--lock FILE` holds that flock for the whole run. Other jobs
  that take the same lock (big model loads, other benchmarks) cannot overlap it.
- **Best and median.** Each cell is the best and the median over all runs. The median is what
  the change is computed on.

## The load mode is part of the measurement: `-lm mmap` versus `-lm none`

A prefill comparison whose arms differ in load mode is partly measuring the loader, not the kernel. Measured on one
`gpu-hold` ticket, ABBA order `A C C A` (A = `-lm mmap`, C = `-lm none`), `REPS=3` per slot, Qwen3-4B-Q4_K_M at
`-dev HRX0 -ngl 99 -fa 1 -p 512 -n 32` (engine#1668):

| arm | prefill pp512 | decode tg32 | own peak GTT |
|---|---:|---:|---:|
| `-lm mmap` (default) | 3 181.97 t/s | 77.98 t/s | 5 495 MiB |
| `-lm none` | **3 737.19 t/s** (1.1745×) | 76.69 t/s (0.9835×) | 6 099 MiB (+604) |

**Why, from the code rather than from the ratio.** llama.cpp's loader refuses to put a weight in the *device's host buffer
type* while mmap is on: `src/llama-model-loader.cpp:1277` redirects such a tensor to the plain CPU buffer type. HRX exposes
`HRX0_HOST` as exactly that host type (`ggml/src/ggml-hrx/ggml-hrx.cpp:885`, registered as the device's host buffer type at
`:1423`). So the two modes differ in *where three of the model's weights live and therefore where their matmuls run*:

- `-lm none`: those tensors keep `HRX0_HOST`, the backend materialises them into device buffers on first use, and their
  matmuls dispatch on HRX against device-resident weights — the +604 MiB of GTT and the faster prefill (prefill touches
  every weight, so it sees the difference; decode touches a few per token, which is why it moves only −1.6 %).
- `-lm mmap` (default): the same tensors are pushed to plain CPU buffers, so their matmuls execute on the CPU.

That also explains why the effect is invisible in the byte-for-byte identical decode rate and why the extra memory is BO
bytes rather than slack: three more ~306 MiB tensors become device-resident (3 → 6).

**The rule that follows:** record `-lm` in the row, hold it constant across the configurations of a comparison, and never
quote a prefill delta measured across it. `-lm none` is not a lever to ship on the strength of one measurement — it costs
GTT on a box where GTT is the scarce resource, and the honest statement is the mechanism above, not the percentage.

## What is measured

- **Prompt:** one long prompt (default: the first 8,000 characters of `docs/*.md`, about 1,800
  tokens; `--prompt-file` to change it), `--prompt-reps` times (default 5). It is reported as
  serve reports prompt tok/s, so context checkpoints and a drafter's own prompt pass are included.
- **Decode:** 256 greedy tokens (`--decode-tokens`) on three chat prompts, `--decode-reps` times
  each (default 3):
  - **code:** an ISO-8601 parser with tests;
  - **prose:** how a B-tree insert works;
  - **short:** a one-sentence translation.

  Drafters behave differently on the three, so all three are reported.
- Thinking is off and the prompt cache is off for every request.

`--json FILE` keeps every raw measurement with the configurations and the time. Serve's own log
for each configuration goes to `--log-dir` (default `bench-logs/`), including the
[recipes](recipes.md) it applied.

`tests/bench_selftest.sh` (ctest `bench_selftest`) runs the tool against `1bit serve` with
`tests/fake_backend.py` behind it, without a GPU.

## The prefill flash-attention ratio: `tools/fa_ratio.py`

Whether the HRX flash-attention path is worth anything for prefill is an end-to-end question here, because the two
instruments that would answer it per dispatch are closed on this hardware:

- `rocprofv3` records **nothing** for HRX — the runtime `dlopen`s its own ROCr and writes its own AQL packets, so a
  profiler that intercepts the system ROCr never attaches;
- the runtime's own HAL profiler (`HRX_PROFILE_MODE=dispatch` with `HRX_PROFILE_FILE`) produces **no file** on a shipped
  payload, and adding `HRX_GPU_DEBUG=1` — the only way its sink and begin paths report failure — prints nothing either,
  which means the sink-owning path is not executed there.

What is left is the comparison a user can make with the shipped binary: the same model and shape with `-fa 1` and with
`-fa 0`. `tools/fa_ratio.py` runs that comparison in **adjacent pairs** (one `-fa 1` and one `-fa 0` back to back, with
the pair order alternating so slow drift cancels) and reports the **median of the per-pair ratios**, not the level of any
arm — because on a shared box the levels move by an order of magnitude inside one ticket while a pair taken seconds
apart moves far less.

A run is only quoted when its own pairs agree: with fewer than three pairs, or with `max/min` of the ratios above
`--tolerance` (default 1.25), the tool prints `VOID` and exits 3. Measured on a Strix Halo box with a 0.6B model at
pp2048:

| run | pairs | median ratio | spread | verdict |
|---|---|---|---|---|
| quiet box | 4 | **1.273×** (1 774.17 / 1 388.34 tok/s) | 1.08× | ACCEPTED — the FA path is worth ~21 % of the `-fa 0` time, 0.156 ms per 2048-token forward |
| busy box (load 73, 20 GiB of foreign models resident) | 3 | — | 9.29× | VOID — the same arms spanned 161 to 1 773 tok/s |

Before it starts, the tool refuses a box that cannot produce a quotable run: a 1-minute load average above `--max-load`
(default 60), `/sys/class/drm/card0/device/gpu_busy_percent` above `--max-gpu-busy` (default 5), or another `llama-server`
**holding** more than `--max-resident-gtt-mib` of GTT (default 256) — three resident servers inflated an earlier measurement by up
to 15 %, and refusing beats averaging. The last test is by *held device memory*, not by existence, because the reference box runs
two permanent embed servers (`:8090`/`:8095`) that no lane may stop: measured there, six `llama-server` processes were up and only
three held GTT (5 492 MiB, 32 240 MiB, 298 MiB), so a server that exists but holds nothing does not block a run and
`--skip-preflight` stays the exception rather than the normal path. `--skip-preflight` exists for a deliberately contended run,
never for a number anyone intends to quote.

The verdict refuses a run that came out **short** as well as one that disagrees: if fewer pairs are usable than `--pairs` asked
for, it prints how many runs produced no rate row and returns VOID, because a silently short run is the same class of error as a
silently shifted pair.

`-fa 0` is usable only below roughly 5.4k context on this route (it segfaults above that), which is why the default shape
is pp2048. The self-test (`ctest fa_ratio_estimator`) exercises the estimator without a GPU: it accepts agreement, refuses
disagreement, and checks that a run whose row failed to parse cannot shift the arms of the pair after it.
