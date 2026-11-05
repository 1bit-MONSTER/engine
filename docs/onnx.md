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

The fused A8W8 QDQ path has its own entry beside `gemm_f32`: with `ONEBIT_EP_NPU_ELFS="MxKxN=path,…"`
(one ELF per shape, because a QDQ group is claimed per (K, N)) `gemm_q8` hands the graph's int8 activations
and int8 weights to the kernel unchanged — no `packB`, no `amax/127`, since those bytes already carry the
graph's scales — and reads int32 accumulators back into the **same epilogue** the host path runs, so a device
result differs from the host one only if the accumulator does. A shape with no ELF falls back with
`npu_qdq_fallback … why=no QDQ kernel for KxN`, and a run is only a device run if that line is absent for
the shape you care about: `qdq rows=` is written by the shared epilogue either way. **An entry the runtime
cannot load costs that design and nothing else**: `init_elf` returning false marks the shape unavailable
(`no QDQ kernel for KxN: I8Ctx init_elf failed for <path> (MxKxN)`) while the other configured designs keep
running, and only a configuration in which *every* design is refused disables the device as a whole.

**A shape with no design is a build, not a limit.** All seven of Qwen3-0.6B's projection shapes are covered
on this box: o_proj (2048×1024) and down_proj (3072×1024) come with the xclbins, and q_proj (1024×2048),
k_proj/v_proj (1024×1024) and gate_proj/up_proj (1024×3072) were emitted from the iso-build design generator —
one `-M 128 -K <K> -N <N> -c 4` run of `n1_core_i8_v26.py` through the IRON/MLIR-AIE `aiecc` per shape, with
the `load_pdi` injection `generators/elf_emit.sh` documents. The generator and the resulting ELFs live in that
tree, never here. The proof's `quark-kv` and `quark-gu` cases are exactly the two shapes that had neither a
case nor a design, so a new shape is one line in the `Case[]` table plus one build. On this toolchain
`aiecc --full-elf-name` exits **0 and writes nothing**; `--get-full-elf` is the flag that emits the ELF (its
bundle is larger than the xclbin-derived ELFs beside it, and the runtime accepts it — the `|d| = 0` results
above are the evidence, not the file size).

**What the device leg is worth, measured.** `onebit_npu_gemm_q8_probe` puts one shape through the same
`gemm_q8_raw` and through the EP's own host loop with int8 operands, requires their exact int32 accumulators to
be identical, and times both — `repeats` calls *timed* on each leg after one untimed device warm-up, and the
min of three runs per cell. strixhalo, M=32 designs, `-O2` build, load 9–10 on 32 cores, in an `npu-hold`
ticket. **The array ramps**: in a ticket whose *first* measurement is a small-row cell, every device figure comes
out a uniform ~1.8× high and settles by the sixth configuration (measured, same shapes, same binary), so the
first shapes of a cold run are not quotable — the table below was taken after nine warm-up configurations of the
same shape.

| shape | rows | device ms/call | host ms/call | device vs host |
|---|---|---|---|---|
| 1024×2048 (q) | 1 | 1.501 | 0.158 | 9.5× **slower** |
| | 2 | 1.550 | 0.317 | 4.9× slower |
| | 4 | 1.526 | 0.669 | 2.3× slower |
| | 8 | 1.515 | 1.319 | 1.1× slower |
| | 16 | 1.529 | 2.550 | 1.7× faster |
| | 32 | 1.566 | 5.062 | 3.2× faster |
| 1024×1024 (k, v) | 1 / 32 | 0.781 / 0.824 | 0.078 / 2.529 | 10× slower / 3.1× faster |
| 1024×3072 (gate, up) | 1 / 32 | 2.440 / 2.425 | 0.235 / 7.765 | 10.4× slower / 3.2× faster |
| 2048×1024 (o) | 1 / 32 | 1.499 / 1.493 | 0.180 / 5.017 | 8.3× slower / 3.4× faster |
| 3072×1024 (down) | 1 / 32 | 2.209 / 2.311 | 0.250 / 7.602 | 8.8× slower / 3.3× faster |

A call costs the same at 1 row and at 32 — q_proj is 1.501 ms against 1.566 ms, i.e. 32× the work for 4 % more
time — so it is **dispatch-bound**: a decode step pays one launch per projection (196 of them on Qwen3-0.6B) and
at one row that launch costs about 10× the host loop it replaces (8.3–10.9× across the three takes below). This is therefore a **prefill** leg: at a full
32-row design the five projection shapes are 2.9–3.6× ahead of the host loop and at 16 rows 1.4–1.7×, while
**12 rows is the count where the margin is smallest** — 1.13–1.23× on a quiet box (k/v the floor at 1.13×)
and as little as break-even on a loaded one (k/v 0.98–1.06×), which is the same range the 45-cell sweep behind
this table measured from the other side (its worst shape at 0.86× of the host loop *time*, i.e. 1.16×). The
threshold is 16, where the five shapes are 1.53–1.64×, so nothing sits near that crossover on a box whose
timing moves with its load. Design M is not the lever — an M=128
design at one row measured 1.233 ms against M=32's 1.499 ms on 2048×1024 in this run and 1.606 vs 1.441 in an
earlier one, i.e. the two are within run-to-run variance at decode width and neither changes the picture. The
host column is a **lower bound**: it is the accumulation loop only, while the EP's real fallback then runs
`epilogue_row` (a float multiply and a `q8_quantize` per element), so the 12–16-row crossover is optimistic *for
the device*. A decode-side win would need fewer, longer launches (one launch covering many projections), not a
smaller design — and "What the device leg is worth at model level" below is the four-arm measurement that shows
the same thing from the graph's side: kernel-level wins of 1.5–1.9× do not survive 196 dispatch cycles per pass.
`docs/onnx-pack.md` quotes that model-level result rather than a device speedup, because the sign of it is not
what a speedup would claim.

*(An earlier revision of this table was ~5 % low on the device column: the probe timed `repeats-1` calls and
divided by `repeats`. The figures above are the corrected ones, and `repeats < 2` — which used to print
`device 0.000 ms/call` and a four-digit speedup — is now refused with exit 2.)*

**Re-measured on a second boot** (`main` `9eb991f`, the five M=32 designs, three rounds of 30 timed calls per
cell, `~/wt/qdq-quiet-timing.sh` + `~/wt/qdq-quiet-medians.py`): the table above reproduces — one row
**8.5–10.9× slower** (table: 8.3–10.4×), 16 rows **1.40–1.65× faster** across the five shapes (the table's
single q figure there is 1.7×), 32 rows **2.86–3.63×** (3.1–3.4×), and the device column stays flat between 1
and 32 rows (q: 1.693 ms at one row against 1.724 ms at 32). The box was
not quiet — load 5.2 at round 1 rising to 13.8 at round 3, another lane's `ort_run` at 382 % CPU — and *both*
columns move with it (round 3 against round 1 is 1.15–1.98× slower on the device and 1.51–3.39× on the
host), so the 1-, 12- and 16-row cells are quoted from rounds 1–2 only. A second estimator agrees with the first: taking each cell's **minimum**
over all three rounds (the least-contended sample) gives k/v 1.06× at 12 rows, the other four 1.13–1.19×, 16 rows
1.49–1.63× and 32 rows 3.00–3.17×, so the boundary sentence changed above is not an artefact of which average was
picked. Every one of those 75 probe runs also asserts the device's exact
int32 accumulators equal the host loop's, so it is a correctness re-check of `gemm_q8_raw` on five shapes at
five row counts (1, 12, 16, 32, 64) as well as a timing one.

**And then the box did go quiet, which moved one cell and changed why the threshold is 16.** A watcher held no ticket
until load1 stayed under 4 for five minutes, then took one 24 s window at **load 1.72–2.73**, 60 timed calls per
leg, three rounds. Same designs, same shapes:

| shape (K×N) | 1 row | 12 rows | 16 rows | 32 rows |
|---|---|---|---|---|
| q 1024×2048 | 0.10× | 1.18× | 1.56× | 3.07× |
| k/v 1024×1024 | 0.10× | **1.13×** | 1.55× | 3.00× |
| gate/up 1024×3072 | 0.10× | 1.19× | 1.53× | 3.09× |
| o 2048×1024 | 0.10× | 1.17× | 1.60× | 3.19× |
| down 3072×1024 | 0.10× | 1.23× | 1.64× | 3.17× |

So on a quiet box **every** shape is faster at 12 rows, k/v included (1.13×) — the loaded take above put k/v at
0.98–1.06× there, and both are real: the difference is the box, not the kernel. What makes the quiet run the one
to trust is its own internal spread: within a cell the host loop varies **1.00–1.09×** at load 1.7–2.7 against
**1.51–3.45×** at load 5–14. That is also the honest reason the threshold is 16 rather than 12: at 12 the
margin (1.13–1.23×) is smaller than the spread a busy box adds, and at 16 it is not. Reproduce with
`~/wt/qdq-quiet-watch.sh` (it waits for the load, then spends one short `npu-hold` ticket).

**What `gemm_q8` does with that number.** It routes by rows: below `kDeviceMinRows` — 16, declared once in
`onnx/ep/onebit_ep_kernel.h` — the host loop runs the multiply and the call never reaches the array; at or above
it the device does. 12 rows is where the margin is smallest — **1.13–1.23× on a quiet box, break-even for k/v on a
loaded one** — and 16 is the first count that keeps more margin than the box's load adds (1.53–1.64×), so the
threshold is one constant rather than one per (K, N). A host-routed call is a choice, not a
failure, and it does not look like one: it writes
`npu_qdq_host_rows m=… k=… n=… min=16` to the trace. "The device ran this shape" therefore stays exactly
"neither `npu_qdq_host_rows` nor `npu_qdq_fallback` was written for it" — an assertion a green `|d| = 0` line
alone has never been (see `docs/onnx-pack.md`). `onebit_ep_qdq_proof` asserts the routing of *every* fused case
against that same constant, and it asserts that a case at or above it did **not** fall back, so a decode-width
routing regression, a prefill-width one and a device that quietly declined to run are three different red lines:
with designs configured, `quark-stress` (64×1024×2048) and the three 32-row projection cases (`quark-o32`,
`quark-d32`, `quark-gu32`) must reach the array, and every 5–8 row case must log `npu_qdq_host_rows`. Those three
exist because the 5-row cases are host-routed by design: without them o_proj, down_proj and gate/up had no device
gate at all, while this page claimed the shapes were covered.

**What the device leg is worth at model level, and it is not yet a win.** The rows table above times one
shape; this times the graph. Four arms on one graph — the real Quark A8W8 export of `qwen3-0.6b.onnx`
(196 fused groups, the 197th being `lm_head`, whose weight is a `Cast`, so it stays float) — a Release build of
this tree (the EP targets carry no `-O2` of their own: a bare `cmake -S onnx -B` compiles the host loop
unoptimized, which inflates the device-against-host ratio by about 25× — measured, 129.6 ms against 5.0 ms for the
same probe call), one thread, a 32-token prompt, 8 generated tokens, quiet box
(load 1.17–1.35 for every arm), two runs each and the second quoted:

| arm | prefill 32 tokens | decode ms/token | tok/s |
|---|---|---|---|
| float32, CPU EP | 207 / 212 ms | 225.6 / 226.0 | 4.43 |
| A8W8, CPU EP | 635 / 645 ms | 625.1 / 628.1 | 1.59–1.60 |
| our EP, host kernel | 843 / 845 ms | 947.4 / 950.6 | 1.05–1.06 |
| our EP, **array** | 805 / 809 ms | **1073.7 / 1105.1** | 0.90–0.93 |

So at model level the array is **1.13–1.16× slower than the host kernel it replaces**, 1.7× behind the A8W8 CPU
EP and 4.8× behind float32; only the 32-token prefill is ahead of the host leg, by 4–5 %, and still 1.3× behind
the CPU EP. Correctness is exact where it matters: the device arm generated **the same ids as the host arm** and
its KL against float is the same number (5.6787 nats both; the A8W8 CPU EP's own arm scores 7.2085 with a
different continuation), and its trace is `qdq runs=1764` (= 196 groups × 9 passes), `host-rows policy=0`,
`device fallbacks=0`, `device-busy=0` — every fused group went to the array, and nothing was dressed up as one.
**These are the figures as published then, on a tree before #1047; the same two arms re-measured on the current tree
are 930.5 against 972.9 ms/token in the device's favour — see "Where a pass actually goes, and the sign on the current
tree" at the end of this section, which is the number to quote, and which explains the difference rather than
overwriting it.**
The 5-token counter-example is the other half of that claim: same binary, same designs, 8 tokens in 2159 ms
(3.70 tok/s), trace `host-rows policy=1764` and no device call at all, because a 5-token prompt never reaches 16
rows.

**Why the kernels being 1.5–1.9× ahead does not reach the model.** One probe per design at the decode width
(m=36, which is two blocks per launch — 32 rows + 4 — so it measures what a decode step really issues), 20 timed
calls each, same build and same window:

| shape | groups/pass | device ms/call | host ms/call | device ms/pass |
|---|---|---|---|---|
| q 1024×2048 | 28 | 3.663 | 5.629 | 102.6 |
| k/v 1024×1024 | 56 | 1.670 | 2.874 | 93.5 |
| gate/up 1024×3072 | 56 | 5.287 | 8.769 | 296.1 |
| o 2048×1024 | 28 | 3.292 | 5.638 | 92.2 |
| down 3072×1024 | 28 | 4.560 | 8.626 | 127.7 |
| **196 groups** | | | | **device 712.0, host 1209.0 (1.7×)** |

At m=32 the same probe reproduces the rows table's q cell — `device 1.632 / host 5.166` against the published
`1.566 / 5.062`, i.e. 4 % and 2 % — so the two measurements describe the same kernel. And the host column is a
**unit** measurement, not an additive one: it sums to 1209 ms/pass, which is *above* the whole host arm's 947
ms/token. The likely reason is that the probe's operands are synthetic int8 while the graph's are not and the
host loop skips `aik == 0` — a hypothesis, not something measured here. Either way the host column is a unit
measurement: do not subtract the two rows above to explain the arms, compare arms to arms.

**Which means the cause of the model-level loss is not established, only its sign.** Two explanations fit every
number on this page, and these measurements do not separate them:

1. **Non-kernel work per call.** 196 calls per pass, two launches each at 33–40 rows, each paying the A upload, two
   syncs, a wait, the weight writeback and the shared epilogue. Three of those are now priced on the array (below) and
   come to **0.43 ms/call** — 84 ms of a 1074 ms pass — so this candidate is no longer a guess, but it is also not yet
   the answer: what is left of the call inside `dev_call_ms` is the kernel plus the A writeback, the launch, the two
   syncs, the wait and the C readback, which that knob times as one number.
2. **The host loop is faster on real data than on the probe's.** The probe's host column (5.629 ms for q at 36
   rows) is what makes "the kernels are 1.7× ahead" true in isolation; if the graph's sparser activations let the
   `aik == 0` skip cut that by 30 %, the host arm's kernel time falls to ~850 ms/pass, the device's 712 ms
   becomes a far smaller win, and the two arms land where they are measured — the device slightly behind.

**What the per-call costs actually are, measured.** `onebit_npu_gemm_q8_probe` now prices a weight cache on the
array as a unit measurement: it times the same call twice, once with one weight (the cache hits) and once with a
different weight every call (the cache misses), which is what a model pass does — 28 weights of one shape go through
one kernel in a row. On the M=32 designs, 40 repeats, load 9.2, both legs exact against the host accumulators over
32768 values:

| shape | one cache slot | one packed image per weight | delta |
|---|---|---|---|
| q 1024×2048 | 1.660 → 1.906 ms/call | 1.693 → 1.747 ms/call | **+0.246 → +0.054** |
| o 2048×1024 | 1.839 → 2.076 ms/call | 1.875 → 1.942 ms/call | **+0.238 → +0.067** |

So the one-slot cache costs **0.24 ms/call** — the BO writeback of the whole image — and the ~0.06 ms that survives
resident weights is the pack loop itself. Keeping every weight resident removes the 0.24 ms (**~35 ms/pass**) for
**568 MiB** of int8 images (588 MiB with the per-context A/C BOs; 140 MiB for the two shapes that have an ELF here),
measured on the real export's 197 projected MatMuls, and the QDQ proof is bit-identical with and without it — that
trade was built, verified and **not** taken, because 3 % of a token is not worth 0.5 GiB of a shared box.

The shared epilogue, from `ONEBIT_EP_QDQ_PROF=1` on the device arm of the same export (two runs, identical ids, trace
`qdq rows=1764  fallbacks=0  device-busy=0`):

```
qdq_prof calls_dev=1764 calls_host=0 rows_dev=63504 dev_call_ms=9964.1 dev_epi_ms=237.2 host_ms=0.0
```

`dev_call_ms` is **5.65 ms/call** (1107 ms/pass) and `dev_epi_ms` is **0.134 ms/call** (26.4 ms/pass) — taken at load
11–14, so those absolute figures are not quotable as throughput, but the split is, since both terms come from one run.
With the graph's other nodes (~194 ms/pass at that load) the three terms add to the measured 1327 ms/token exactly,
which is what makes this a split rather than a guess: **the epilogue is 2.4 % of the arm, not the gap.**

So of candidate 1's non-kernel work, 0.43 ms/call is measured (writeback 0.24, epilogue 0.134, pack 0.06) and the
rest is not. **Candidate 2, on the other hand, can be measured without the array at all** — the host loop skips
`aik == 0`, so what matters is how many zeros the graph's own activations carry. Exposing every projected MatMul's int8
activation as a graph output and running one 32-token prefill at `ORT_ENABLE_BASIC` (which keeps the Q/DQ nodes) gives
the real bytes, and the same host loop timed on them against the probe's synthetic draw:

| host column for a whole pass, 196 groups | ms/pass |
|---|---|
| synthetic operands (the published probe column) | 1098.0 |
| the graph's real activations | **770.6 (1.42× faster)** |

The real tensors are **62.1 % nonzero** against the probe's ~99 %; layer 0 alone is 93.7 %, which is why one tensor is
not a sample of the pass and the whole pass was dumped (112 distinct tensors behind 196 MatMuls — q/k/v share one,
gate/up share one). `down_proj` is the extreme: its input is the SwiGLU product at 55.8 % nonzero, and the loop is
**1.76×** faster on it.

That is enough to price candidate 2 against the arms' own numbers, with one ratio named: the published synthetic
column is 1209 ms/pass at m=36 while this dump is at its 32 rows, and on the synthetic draw the two agree at 1.10×
(1098 here), so the graph's own operands put the **host loop at ~848 ms/pass** against a device multiply of
712 + ~47 ms of weight writeback = 759. **The array's kernel advantage is therefore ~1.12× at model level, not the
1.7× the synthetic column implies** — which is the direction the bound in the next paragraph allowed, and it is why the
device arm does not win: it saves ~90 ms/pass on the multiply and the two arms differ by 127 ms/token in the other
direction. What that 127 ms is made of is candidate 1's unpriced remainder, and it needs **both** arms profiled in one
window (`dev_call_ms/call` against `host_ms/call`) rather than another single-leg profile: the multiply is no longer
the place to look.

The bound that *is* rigorous, and it is tight enough to matter: the probe's host column sums to 1209 ms/pass
while the whole host arm — kernel plus every other node in the graph — is 947 ms/token, so the graph's real
operands make the host loop **at least 1.28× faster** than the probe's synthetic ones. The device's 1.7×
kernel-level advantage is therefore at most **1.33×**, and it could be none at all, which is the whole reason the
cause is stated as open rather than as dispatch.

The next step is the one comparison these numbers cannot make, not a redesign: **both arms with
`ONEBIT_EP_QDQ_PROF=1` in one window** — the device arm's `dev_call_ms/call` against the host arm's `host_ms/call` on
the graph's own int8 bytes, which prices candidate 2 directly. The call's own sections are timed as well —
the same line carries `dev_split_wpack` (weight pack into the BO + its upload, a cache miss), `dev_split_aupload`,
`dev_split_async` (the A flush), `dev_split_launch` (launch + wait), `dev_split_readback` (C invalidation + the copy
back) and `dev_split_other` (the device lock, the kernel lookup and the EP's own frame around the call, i.e. what the
five named sections do not account for) — so `dev_call_ms` adds up on the face of the line rather than leaving a reader
to attribute one residue by subtraction. `NPU_C_SYNC_PROF=1` (engine#988) still prints the C-sync counters per call
if a section needs separating further. These sections are timed only when the knob is on (a plain call pays one branch
per section), and **no device window has printed them yet** — they appear from the next one on. What a *fix* would
look like if (1) dominates is one launch covering many projections — a resident weight layout and a pass-level submission, rather
than the per-projection ELF this leg uses — which is also why the threshold discussion above stands unchanged:
`kDeviceMinRows` is about one call, and the model-level question is about 196 of them. Worth knowing before anyone
starts that work: the resident-weight half of it has now been measured and is worth 35 ms/pass for 568 MiB, i.e. the
sign of the model-level result does **not** turn on the weight path.

**Where a pass actually goes, and the sign on the current tree.** The same two EP arms, two runs each, back to back in
one `npu-hold` ticket (load 4.9–6.9, identical ids, `fallbacks=0`, `device-busy=0`), with `ONEBIT_EP_QDQ_PROF=1` on
both:

| arm | decode | per pass, from the profile |
|---|---|---|
| our EP, **array** | 953.3 / **930.5** ms/token | device calls 802.5 (`dev_call_ms` 7222.2 / 9), epilogue 19.0 (`dev_epi_ms` 171.2 / 9), other 109 |
| our EP, host kernel | 970.4 / **972.9** ms/token | host loop 879.0 (`host_ms` 7911.2 / 9), other 94 |

Both arms are fully accounted for (802.5 + 19.0 + 109 = 930.5; 879.0 + 93.9 = 972.9), and "other" — attention, norms,
softmax, embed, ORT — is 94–109 ms/pass, the part neither leg touches. So on this tree the device arm is **4.4 % ahead**,
not 1.13× behind as the four-arm table far above reports, and both runs of each arm agree on the sign. The host
loop's 879 ms/pass also validates the CPU-only estimate of 848 from the section above to within 4 %.

What differs between those two sessions is **not** the code, as far as the repository can say: since the build those
figures came from, exactly two commits touch this path — #1032, which adds the profiling that produced the numbers above,
and #1047, the weight re-pack, 0.24 ms/call = 47 ms/pass (~5 % of a token) — the private NPU runtime the EP links against
is unchanged (files dated 2026-10-31), and the five design ELFs are byte-identical (md5s). It is not load in the obvious
direction either: this run was at load 4.9-6.9 against 1.2-1.35 then, and the device arm gets *slower* under load (its
calls measured 5.65 ms/call at load 11-14 against 4.09 here). So ~5 % of the 13 % is code and **the rest is
unestablished**: the plausible shape of it is CPU-side contention in the earlier session that a one-minute load average
did not capture, since half of a device call is host work (BO writes, two syncs, the readback), and the honest reading is
that those are two sessions under different conditions rather than two builds. The kernel is unchanged either way — the
probe in this same window still measures q at m=36 at 3.460 ms/call device against 5.836 host, the 1.69x the rows table
reports. What moved is the model-level per-call cost, and a quiet-box repetition is what this number needs
before anyone quotes it as the device leg's worth.

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
| `onebit_ep_qdq_proof_qdq_only` | with `ONEBIT_EP_QDQ_ONLY=1` (the mode a quantised model is measured in) the fused groups are still claimed and no single-node float MatMul is |

`onebit_npu_gemm_probe` (NPU builds only) is registered as a sixth test and **skips** when the
device is busy rather than failing someone else's run.

`ort-run --basic` keeps a quantised graph's `DequantizeLinear`/`QuantizeLinear` nodes instead of letting
ORT's graph transformations fold them back into float ops, which is what a quantised EP needs to see;
`ONEBIT_EP_QDQ_ONLY=1` restricts the EP to the fused groups so a measurement of a real quantised model
is not distorted by float matmuls landing on the host kernel. `ONEBIT_EP_QDQ_DEBUG=1` makes
`GetCapability` log why each `MatMul` was refused. `ONEBIT_EP_QDQ_PROF=1` has the fused path report where
its own wall time went, as one line per process at exit —
`qdq_prof calls_dev=… calls_host=… rows_dev=… rows_host=… dev_call_ms=… dev_epi_ms=… host_ms=…`, split into
the device call (kernel, BO writes, both syncs, the wait), the epilogue both legs share, and the host loop —
so two runs that differ only in the leg price that leg on the graph's own int8 bytes, which is the one thing
a per-call probe timing synthetic operands cannot do. On the device leg the call is split further, by the device
leg's own timers: `dev_split_wpack` (weight pack + upload, a cache miss — once per weight, not per call),
`dev_split_aupload`, `dev_split_async` (the A flush), `dev_split_launch` (launch + wait), `dev_split_readback`
(C invalidation + the copy back to the caller's buffer) and `dev_split_other`, which is `dev_call_ms` minus those
five. The sections are timed only when the knob is on. The line writes to the EP trace when `ONEBIT_EP_TRACE` is set
and to stderr when it is not: a knob whose only output can silently vanish is how a debugging session gets
lost, and this file already paid for that once.

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
