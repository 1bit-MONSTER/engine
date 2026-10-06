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
# Hyperloom on Strix Halo (gfx1151)

[Hyperloom](https://rocm.docs.amd.com/projects/hyperloom/en/latest/what-is-hyperloom.html) 1.0.0
is AMD's agentic inference optimizer (MIT). It ships tables for Instinct boards only (gfx942/gfx950)
and serving frameworks such as vLLM and SGLang. These files let it tune the engine's llama.cpp fork
on HRX (Strix Halo, gfx1151) through its `--framework custom` + bypass backend route. The first
workload is ZAYA1-8B Q4_K_M decode.

| File | What it does |
| --- | --- |
| `hl_patch_strixhalo.py [pkg]` | Adds a `strixhalo` GPU type (gfx1151, 40 CU, ~256 GB/s, 59.4 TFLOPS f16, 128 GB) next to each `mi355x` entry of an installed Hyperloom package. |
| `hl_cli_login.py [pkg]` | With `ONEBIT_HYPERLOOM_CLI_LOGIN=1`, agent turns use the logged-in `claude` CLI instead of an API key; also lets `RAY_VERSION` override the pinned Ray (2.44.1 has no Python 3.14 wheels). Unsupported upstream. |
| `custom_strixhalo.sh` | The benchmark entrypoint (`custom_<gpu-type>.sh`). Rebuilds the checkout, takes `~/.cache/lax-decode/box.lock`, waits for a cool idle GPU, gates on wikitext perplexity, then reports `llama-bench` tg128 as `output_throughput`. |
| `lemonade_bench.sh` | Lemonade's own benchmark (`lemonade bench`, the settings of Lemonade's nightly benchmark-regression CI: chat-short + code-short, 1 warmup, 5 runs, ctx 4096) on the fork's `llama-server`, routed through `llamacpp.vulkan_bin` of a private lemond. Lemonade reserves `-dev`, so the device goes in as `LLAMA_ARG_DEVICE` (default HRX0). |
| `launch.sh` | Starts `hyperloom ... optimize` for ZAYA1-8B, detached. |
| `zaya1-8b-q4_k_m.reference.json` | Gate reference from the unmodified tree (fork 3b954f0). |
| `hrx2kineto.py` | Converts an HRX dispatch profile to a Kineto trace (Hyperloom's profile step, TraceLens, rocprof studio). With the fork's shape log it names every dispatch by its ggml op, shapes and layer and doubles as a TraceLens extension (roofline, categories, TraceDiff). |
| `hrx-tracelens.sh <model> <prompt\|decode> [VAR=value ...]` | Profiles one run, converts it, and writes a TraceLens report (optionally with TraceDiff against another run). |

## Two benchmark standards

`output_throughput`, the number Hyperloom optimises, is ours: `llama-bench` tg128. The result also
carries Lemonade's standard under `bench_summary.lemonade_bench` (TPS and TTFT per scenario), for
numbers compared with Lemonade. On ZAYA1-8B at fork 3b954f0: llama-bench tg128 93.5 tok/s; lemonade
bench chat-short 65.4 TPS, code-short 75.0 TPS, TTFT about 361 ms (Lemonade reloads the model for
each run and generates short answers).

## Quality gate

Hyperloom's own quality check targets served models; here the script gates instead. Each candidate
must stay within 0.5% of the reference prefill perplexity (c512, 4 chunks) and within 2% of the
decode-shaped perplexity (`-ub 1`, c128, 4 chunks). Eight prefill chunks trip the 93 C thermal stop
on this box, so the gate uses four. Every step cools below 60 C first and stops at 93 C.

## Setup (strixhalo)

```sh
python3 -m venv ~/hyperloom-ws/.venv && . ~/hyperloom-ws/.venv/bin/activate
pip install "hyperloom-inference-optimizer[runtime]==1.0.0" "ray[default]==2.55.1"
PKG=~/hyperloom-ws/.venv/lib/python3.14/site-packages/hyperloom
python tools/hyperloom/hl_patch_strixhalo.py $PKG
python tools/hyperloom/hl_cli_login.py $PKG
# Hyperloom needs a Ray head with a GPU and a serving_slot resource
ray start --head --num-gpus=1 --resources='{"serving_slot":1}' --disable-usage-stats
mkdir -p ~/hyperloom-zaya/scripts && cp tools/hyperloom/custom_strixhalo.sh ~/hyperloom-zaya/scripts/
cp tools/hyperloom/zaya1-8b-q4_k_m.reference.json ~/hyperloom-zaya/reference.json
# ~/hyperloom-zaya/llama.cpp: a worktree of the fork built with GGML_HRX=ON
# ~/hyperloom-zaya/model -> the ZAYA1-8B Q4_K_M GGUF
cd ~/hyperloom-zaya && (setsid nohup bash /path/to/tools/hyperloom/launch.sh > hyperloom.log 2>&1 < /dev/null &)
```

Setup notes:

- TraceLens pins `xprof==2.20.1`, which does not install on Python 3.14. Install TraceLens with
  `--no-deps` plus `xprof>=2.20.2,<2.21`.
- Hyperloom's `install.sh` also installs the scriptable-quality dependencies (lpips pulls in torch);
  the custom workload does not need them.

## HRX profiles in TraceLens

rocprofv3 sees no HRX dispatches (HRX loads its own libhsa and IREE writes the AQL packets), so HRX
profiles itself (`HRX_PROFILE_MODE=dispatch HRX_PROFILE_FILE=x.prof`) and `iree-profile` reads the
bundle. `hrx2kineto.py events.jsonl out.trace.json [kernel_trace.csv]` turns the dispatch events into
Kineto kernel events; that alone gives TraceLens a kernel timeline, every op in "other".

A llama.cpp build with `GGML_HRX_DISPATCH_SHAPE_LOG=<file>` also logs, for every command of every HRX
command program, the dispatch match that emitted it (covered ggml nodes, tensor types and shapes) and,
per graph execution, the current graph's node names. With `--shapes <file> --commands <iree-profile
command jsonl>` the converter then emits one `hrx::<ggml op>` cpu_op per match execution
(`hrx::mul_mat`, `hrx::mul_mat_id`, `hrx::flash_attn_ext`, `hrx::rms_norm`, ...) with its kernels,
shapes and the fused op chain, nested under `hrx::step` (one forward pass, across graph splits) and
`hrx::layer`. FLOPs and bytes use each ggml type's real bits per weight (Q4_K 4.5, Q6_K 6.5625, Q8_0 8.5,
MXFP4 4.25, ...); a MUL_MAT_ID reads only the routed experts. Passed to TraceLens as
`--extension_file`, the same file maps these ops to GEMM / GroupedGEMM / SDPA / NORM / RoPE / reduce /
SSM / elementwise perf models; `--write-arch` writes a gfx1151 arch JSON (WMMA f16/bf16/iu8 1024 and
iu4 2048 ops/WGP/clk, 20 WGPs, clock from `pp_dpm_sclk`, 256 GB/s) for the roofline columns.

```sh
GGML_HRX_DISPATCH_SHAPE_LOG=x.shapes.jsonl HRX_PROFILE_MODE=dispatch HRX_PROFILE_FILE=x.prof llama-bench -p 512 -n 0 ...
iree-profile dispatch --format=jsonl --dispatch_events x.prof > x.events.jsonl
iree-profile command --format=jsonl x.prof > x.commands.jsonl
python3 tools/hyperloom/hrx2kineto.py x.events.jsonl x.trace.json --shapes x.shapes.jsonl \
  --commands x.commands.jsonl --phase prefill --skip 1 --write-arch gfx1151.json --summary
TraceLens_generate_perf_report_pytorch --profile_json_path x.trace.json --output_xlsx_path x.xlsx \
  --extension_file tools/hyperloom/hrx2kineto.py --gpu_arch_json_path gfx1151.json \
  [--comparison_json_path other.trace.json]
```

`hrx-tracelens.sh` does all of that for one model (prompt: `llama-bench -p 512 -n 0`; decode:
`llama-server`, 8 tokens; the dispatch ring holds 65,536 events and `llama-bench` generation fails
under profiling). Reading the numbers: HRX issues independent dispatches without a barrier, so
concurrent kernels (for example the Q/K/V projections) share the GPU and each one's duration includes
the others (`--summary` prints the overlapped share); activations that stay in the 32 MB Infinity Cache
show more than 100% of the DRAM roofline; KV lengths are the padded cache views; prefill attention
FLOPs include the masked half.

Never put credentials in these files.

## Known limitations (measured 2026-10-05, gfx1151, llama.cpp 522dab4)

The profiling chain has three failure modes that were only found by running it, not by reading it. All three are handled
so the chain still produces a trace, but two should be fixed rather than tolerated:

1. **`iree-profile` must match the IREE that wrote the profile.** An older tool aborts with
   `unsupported IREE HAL profile file version`. `hrx-tracelens.sh` now derives the tool from `HRX_BIN`'s own deps
   build (falling back to `IREE_PROFILE`, then `PATH`) instead of a hard-coded path that may not exist.
2. **The shape log needs a build carrying `GGML_HRX_DISPATCH_SHAPE_LOG`.** Without it the op-level half is empty.
   Build the tree you benchmark with that flag, or point `HRX_SHAPELOG_BIN` at one that has it —
   `custom_strixhalo.sh` warns loudly when the trace describes a different build.
3. **`hrx2kineto.py`'s op-level match assumes command-buffer order equals the logged program's build order, and the
   runtime does not guarantee that.** Observed: the dispatch *multiset* matches a program exactly while the *order*
   differs (`copy_strided*` hoisted ahead of `concat_dim0`; `set_rows` moved), so `map_command_buffers()` reports
   `matches no logged program` and no trace is written. When that happens `hrx-tracelens.sh` falls back to a
   **kernel-level** trace (raw dispatch keys), which TraceLens accepts; the roofline is then coarser (op names in
   "other") but real. The matcher still needs an order-independent key (the profile's `executable_id` /
   `function_ordinal`, or per-dispatch identity).
