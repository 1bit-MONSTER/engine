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
# ONNX Runtime in-process

`-DONEBIT_ORT=ON` links the distribution's ONNX Runtime into `1bit` and builds two things on top of
it: the `1bit ort-info` probe, which reports the pin and the execution providers this binary really
has, and `libonebit_ep.so`, a **plugin execution provider** that ORT loads by path at run time and
that runs the graph's matmuls through the engine's own kernels.

It is off by default, and CI builds without it: it needs the `libonnxruntime` dev package, and
nothing else in the engine depends on ONNX Runtime. ONNX Runtime is not modified, and the only
constants this repository uses are the ones its public headers declare.

## The pin

| what | value |
|------|-------|
| package | Ubuntu 26.04 `libonnxruntime1.23` + `libonnxruntime-dev`, `1.23.2+dfsg-6ubuntu1` |
| library | `/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23.2` |
| sha256 | `9c8f6112e9dcf09ed4e621a1d0a7b6ba49575c7d09f27a652d24a22c8564b13c` |
| OrtApi | `v23`, negotiated with `OrtGetApiBase()->GetApi(ORT_API_VERSION)` |
| pkg-config | `libonnxruntime` (`-I/usr/include/onnxruntime -lonnxruntime`) |
| licence | MIT, recorded in `NOTICE` (linked at build time, never redistributed) |

The pin is a contract, not a preference: the plugin EP hands ORT a vtable of C function pointers
whose layout is fixed by `ORT_API_VERSION`, so a host built for another version would call through
the wrong offsets instead of failing. `1bit ort-info` prints both numbers, and the version the
headers compile against is checked against the version the loaded library serves.

Same version, our own build: `onnx/build_ort.sh` clones tag `v1.23.2`, patches the one `#include
<cstdint>` GCC 15 needs in `core/common/semver.h`, and installs to `~/vendor/onnxruntime-1.23.2`, so
`pkg-config` finds it by `PKG_CONFIG_PATH` alone — nothing in `onnx/` changes. The MIT notice covers
that build too (see `NOTICE`).

## Building it

```bash
sudo apt install libonnxruntime-dev        # 1.23.2 on 26.04
cmake -S . -B build -DONEBIT_ORT=ON
cmake --build build -j"$(nproc)"
./build/1bit ort-info
```

That adds `onnx/` to the build: the probe, `libonebit_ep.so`, and the two tests below. The build
compiles against ORT 1.23's headers, so `-DONEBIT_ORT=OFF` produces a binary with no ORT symbols at
all — the option is a compile-time switch, not a runtime one.

## `1bit ort-info`

```text
onebit ort-info
  ortv    : 1.23.2
  api     : v23 (compile-time ORT_API_VERSION, negotiated above)
  lib     : /usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23
  builtin : 3
            [0] DnnlExecutionProvider
            [1] XnnpackExecutionProvider
            [2] CPUExecutionProvider
  eproster: 1 (partition order; first EP gets the node, no tie-break)
            [0] CPUExecutionProvider  vendor=Microsoft  device=CPU
```

The two lists answer different questions. `builtin` is what is compiled into the `.so` the loader
found — a distribution build carries DNNL and XNNPACK as well as the CPU EP, so "the CPU EP is the
only provider" is not an assumption this build may make. `eproster` is the `OrtEpDevice` roster, and
its order **is** the partition order: ORT offers a node to each provider in turn and the first one
that returns a kernel gets it, with no tie-break. That is why every integration in this repository
registers its own provider first and the CPU EP last.

## The plugin EP

`libonebit_ep.so` is a shared object ORT loads by path:

```c
api->RegisterExecutionProviderLibrary(env, "onebit_ep", "/path/to/libonebit_ep.so");
```

Registration must happen before the session is created; `GetEpDevices` then returns one
`OrtEpDevice` per hardware device the factory offers, and the caller appends them in the order it
wants nodes partitioned. Unregistering invalidates those handles.

What it claims today, and the four edges ORT enforces on any plugin:

- **Ops.** `ONEBIT_EP_OPS` (default `MatMul`). The rest of the target list — `Gemm`, `Add`,
  `Softmax`, `LayerNormalization`, `RMSNorm`, `Gather` — lands one at a time; anything unclaimed is
  offered to the next provider, so a graph never becomes unrunnable because of this EP.
- **Precision and rank.** float32 only, rank 2 or 3 (`ORT_NOT_IMPLEMENTED` otherwise). No sync
  streams yet: an execution stream request fails cleanly rather than pretending.
- **Data transfer must be non-null.** ORT calls through the vtable unconditionally, so a null
  `CreateDataTransfer` is a call to address 0. The allocator must be `OrtDeviceAllocator` and
  256-byte aligned — MLAS loads assume it, and an arena allocator is refused by name.
- **Teardown.** `ReleaseEpFactory` returns an `OrtStatus*` even though it "cannot fail"; ORT
  dereferences the result when it unloads the library, so returning void crashes at `ReleaseEnv`,
  long after the run that looked fine.

### MatMul on the NPU

With `-DONEBIT_EP_NPU=ON` and `-DONEBIT_NPU_RT_DIR=<dir with npu_rt.cpp and npu_rt_api.cpp>`, the
same MatMul kernel calls the NPU through the forward route's `I8Ctx` — the same weight packer and
the same activation quantiser the NPU lane uses (`packB`, per-row `amax/127`, readback and dequant
by `ascales[m] * group_scales[l][n]`), so a device result is comparable with the forward route's.
`docs/npu.md` covers the runtime and what ships in a release; the kernels are built one ELF per
projection shape, and the ELF names its own geometry, which is why the shape is configuration rather
than something the EP infers:

| variable | meaning |
|----------|---------|
| `ONEBIT_EP_NPU_ELF` | the full-ELF kernel for this K and N (e.g. `full_i8_O_K1024_N1024_m16.elf`) |
| `ONEBIT_EP_NPU_M`, `_K`, `_N` | the design dims that ELF was built for; a MatMul of another shape falls back to the host |
| `ONEBIT_EP_TRACE=1` | append one line per callback to `/tmp/onebit_ep_trace.<pid>.log` — the ownership evidence the tests read |

**The device lock.** The NPU is shared, and the rule is that a plugin never evicts the owner's
serve. The EP takes `/tmp/1bit-npu-device.lock` with `flock(2)` **non-blocking** and holds it for the
life of the process; when the lock is held it does not queue — it reports `device busy` and runs the
matmul on the host instead, because an inference session must not block on another process. The open
itself matches `app/npu_device_lock.h` (engine#640): the fixed path, `O_CREAT|O_RDONLY|O_NOFOLLOW`
at 0644, since a read fd is enough for `flock` and the path is not something an environment variable
may name. The difference is only the contention policy — that helper queues by default, this one
always gives up.

### Numerics

The two backends are held to different tolerances, because they compute different things: the host
path is float32 and must match the CPU EP to `1e-5` relative; the device path is int8 by int8 into
int32, and is held to `2e-2` of the result's magnitude. The proof prints which backend it got, so a
silent fall back cannot be mistaken for a device pass.

## Qwen3-0.6B

The first real model through this path is Qwen3-0.6B, exported by `onnx/export_qwen3.py` into a plain
graph — opset 20, ai.onnx domain only, float32, `input_ids`/`attention_mask` int64 `[1, seq]` in and
`logits` float32 `[1, seq, vocab]` out, weights in one external-data sidecar. Attention is exported
with `attn_implementation="eager"` on purpose: a fused/SDPA attention block hides the matmuls inside
an op our EP does not claim, and then there is nothing to schedule.

```sh
python3 onnx/export_qwen3.py --model-dir ~/models/Qwen3-0.6B-hf --out ~/models/qwen3-0.6b.onnx --check
1bit ort-run --model ~/models/qwen3-0.6b.onnx --ep ./libonebit_ep.so --ids "9707 11 1879" -n 8 \
             --dump-logits /tmp/ort-logits
python3 onnx/compare_logits.py --forward /tmp/fwd-logits --ort /tmp/ort-logits
```

`--check` prints the operator census and every distinct `MatMul` `(K, N)` in the graph, which is both
the EP's binding table and the coverage picture: 254 `MatMul` nodes in six shapes, of which this
engine's kernels cover `1024 x 2048` (q) and `1024 x 1024` (k, v) — 84 nodes — and the rest stay on
the CPU EP. `ort-run` therefore reports what it actually did: an EP-on-NPU run and an
`ONEBIT_EP_OPS=none` control on the same graph, both with the CPU EP last.

The graph keeps no KV cache, so one decoding step re-runs the whole sequence: the tokens/s it prints
is that graph's rate, and it is not comparable to a KV-cached server. The comparison that matters is
against the forward route on the same token ids, which is what `compare_logits.py` does — per-step
KL(forward || ort) in nats plus top-1 agreement, over the two logits dumps
(`decode_logits_idx<N>.bin` from `1bit npu-run --dump-logits`, `logits_<N>.bin` from `ort-run`). The
measurements themselves are recorded in the state file for the deliverable, and the pack design they
feed is in [onnx-pack.md](onnx-pack.md).

### Two engines in one session

The NPU EP is not the only accelerator worth registering. `1bit ort-run` takes `--ep` once per
library and applies them **in the order given**, with the CPU EP always last:

```bash
# our NPU EP first, a GPU EP (HIP/ROCM plugin, when one is installed) second, CPU last
1bit ort-run --model qwen3-0.6b.onnx \
    --ep ./libonebit_ep.so --ep /path/to/libhip_ep.so --ids "9707 11 1879 374" -n 16
```

Order is priority, and the two engines cover different halves of the same graph: the NPU takes the
projection shapes it has an ELF for, the GPU EP takes what it supports, the CPU EP mops up. Two
rules make this work, and one of them is a bug we hit:

- each library is appended with its own `SessionOptionsAppendExecutionProvider_V2` call — the API
  rejects a single call whose device list mixes two EPs (`All OrtEpDevice values in ep_devices must
  have the same execution provider`);
- an EP must claim only what it can actually run. Our EP claims a MatMul only when both operands are
  float32 with rank 2 or 3, and with `ONEBIT_EP_NPU` only for `(K, N)` shapes an ELF is configured
  for **and** only while the device lock is ours. Claiming wider turned a graph that ran at
  13 tokens/s on the CPU EP into one that ran at 0.18 tokens/s on our naive host kernel, and, before
  the rank check, into a graph that refused to run at all (`only rank-2 and rank-3 MatMul are
  supported yet` on a rank-4 attention matmul).

### Quantised graphs (Quark A8W8 QDQ)

A Quark-quantised ONNX model is a *QDQ* graph (`DequantizeLinear` → `MatMul` → `QuantizeLinear` with
int8 tensors and float32 scales), and the `MatMul` inside it is the same int8 × int8 → int32 product
our device kernel already computes. The EP claims the whole group — both `DequantizeLinear` nodes, the
`MatMul` and the `QuantizeLinear` — as ONE fused node, so the dequantised float tensors are never
materialised: int8 activations and int8 weights go into the kernel, int8 output comes back with the
scales folded into one multiply per element. Per-tensor scales, the two per-axis forms the epilogue
can fold (`s_w[n]` along the weight's output axis, `s_x[m]` along the activation's row axis) and
**asymmetric zero points** are all supported: `Σ_k (a−zx)(w−zw)` costs the GEMM's own sum plus a
column sum of the weight (computed once per session), a row sum of the activation, and `K·zx·zw`, and
the output zero point is added after the round. A per-axis scale or zero point along **K** cannot
fold — it weights the products inside the sum — so such a graph is refused and stays on the per-node
path (see [onnx-pack.md](onnx-pack.md#a8w8-qdq-the-shape-quark-emits)). `ctest -R onebit_ep_qdq_proof`
checks six cases end to end: per-tensor, per-axis, asymmetric and asymmetric-per-axis all
int8-identical to the CPU EP (`worst |d| = 0` LSB, also on a 256×1024×1024 case), the per-K scale
refused and correct anyway, and ~6-7 MB less peak RSS for the run, which is exactly the three float
tensors it no longer allocates. The control, `ONEBIT_EP_QDQ=0`, runs the same graph through the
per-node path and asserts that the float kernel — not the fused one — is what ran.

## Verifying

```bash
cmake -S . -B build -DONEBIT_ORT=ON && cmake --build build -j && ctest --test-dir build -R "ort_info|onebit_ep"
```

Five tests, each proving a different claim:

| test | proves |
|------|--------|
| `ort_info` | the binary negotiated OrtApi v23 and can list its EP roster |
| `onebit_ep_proof` | a two-node graph (`MatMul` then `Relu`) built in memory through the Model Editor API runs with the plugin EP registered first and the CPU EP last, our EP takes the `MatMul` (the trace has a `matmul` line, which only `Compute` can write), and the result matches the CPU EP |
| `onebit_ep_proof_cpu_fallback` | with `ONEBIT_EP_OPS=none` the same graph is partitioned to the CPU EP (`claimed=0`), so the match above cannot come from the CPU running both sessions |
| `onebit_ep_qdq_proof` | the four-node A8W8 QDQ group is claimed as one fused node: the trace has a `qdq` line and no `matmul` line, the int8 output is identical to the CPU EP's per-tensor, per-axis, asymmetric and asymmetric-per-axis (`worst |d| = 0` LSB), a per-axis scale along K is refused, and the run's peak RSS is ~6-7 MB smaller than the CPU EP's |
| `onebit_ep_qdq_proof_unfused` | with `ONEBIT_EP_QDQ=0` the same graph goes through the per-node path (a `matmul` line, no `qdq` line), so the numbers above can only come from the fused kernel |

`onebit_npu_gemm_probe` (NPU builds only) is registered as a fourth test and **skips** when the
device is busy rather than failing someone else's run.

## Status

The host path is done: it is what the tests assert, and it needs no hardware. The NPU path is
proven on XDNA 2 as well — the ELF loads, ORT hands the EP the `MatMul`, the kernel runs on the
device, and three consecutive runs give the same answer inside the int8 tolerance:
`max_abs = 9.893e-02` against `max|ref| = 20.414` (rel `4.846e-03`, gate `2e-2`), with the probe's
scale fit at `0.9999`. What is *not* done is coverage: an ELF exists per `(M, K, N)`, so on a real
model the EP takes the shapes it has a kernel for and the CPU EP takes the rest (for Qwen3-0.6B that
is q, k and v). The device backend is still off by default and is not part of any release, because
enabling it on a box where another process owns the NPU must stay an explicit decision.

The A8W8 QDQ group is claimed as one fused node and proven (`ctest -R onebit_ep_qdq_proof`, exact int8
match for per-tensor, per-axis and asymmetric zero points, ~6-7 MB less peak RSS per run). One thing
is still open and stays documented as such: the device leg of the fused path — today its int8 GEMM
runs on the host, so a real model's shapes are still a CPU-EP story until the NPU kernel takes int8
activations straight into its input BO.

The two-engine order is implemented and exercised with a single library; a *second* EP library was
not available to try it against (this ORT build ships no ROCM/HIP EP and there is no gfx1201 HIP EP
plugin on the box), so treat a real NPU+GPU session as designed but unmeasured.

The ops list is deliberately one entry long. A plugin EP that claims a node it cannot run makes a
graph fail where it used to work, so ops are added only with a passing proof beside them.

Not part of this, and never linked or shipped here: AMD's VitisAI execution provider wheel. This
engine talks to the NPU through its own runtime ([releases.md](releases.md#private-components)),
and an ORT graph served by `1bit` executes our EP's kernels, not VitisAI's.
