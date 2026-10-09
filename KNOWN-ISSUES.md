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

# Known issues

## engine#315 — intermittent all-NaN logits and format-500 on GLM-4.7-Flash under GTT pressure

**Status: OPEN, accepted as a documented limitation.** The issue stays open until this criterion is met.

Owner acceptance, recorded 2026-10-07: the owner authorised landing this entry in writing — *"you have gh
authorization and do what needs to be done"* — on the basis that no forward fix is available. The limitation text
below was drafted by the assistant; the owner has not edited it and may amend or withdraw it at any time.

**Affected:** `GLM-4.7-Flash-Q4_K_M` on `HRX0`, `llama-server`, at high GTT occupancy (measured from
`gtt_start` ~83 GB up to ~110 GB). Raises at roughly 10-20 % per run; not deterministic and not reproducible on
demand, which is why it is recorded as a limitation rather than fixed.

### Reproduction

3 detached neighbour `llama-server` processes at default context drive GTT into the ~100 GB band, then one
detector server is exercised with 8 identical greedy requests separated by different-length distractors,
followed by 50 mixed-length requests. Under pressure the logits buffer is never written, and the sampler guard
reports:

```
count=154880 of 154880, contiguous=yes
```

`154880` is the model's vocabulary width — the whole LM-head output. The guard fires on the consumer side,
which is why it reports the shape and not the producer.

### Both failure faces

The same underlying defect — output data that is wrong or unwritten under memory pressure — surfaces in two
places depending on where it lands:

1. **All-NaN logits.** The logits buffer is never written; the sampler guard fires with
   `count=154880 of 154880, contiguous=yes`. Example: `v315g-c3` (control arm).
2. **Format-500.** The model emits text the server rejects:
   `{"error":{"code":500,"message":"The model produced output that does not match the expected Content-only
   format"}}`. Example: `v315k-l1` (legacy arm), `identical: 1 distinct of 8` then mixed `10 of 50`, `nan=0`.

A NaN-only count records the second face as a clean run, which is what makes it dangerous as a metric. Scoring
must be: an arm is **valid only if it completes `mixed: 50 of 50 ok`**; an arm that aborts early has reduced
exposure and is excluded, not counted as clean.

### Levers tested and excluded

- **`GGML_HRX_TRANSIENT_REUSE=legacy`** — no effect. 15-arm matched study, alternating control and legacy
  inside one box state: **0 NaN in 15 control vs 1 in 14 legacy.** A 7-arm paired confirmation scored by
  completion rather than NaN: **0 NaN in 7 arms**, control 3 clean / 1 truncated, legacy 2 clean / 1 truncated.
  Truncation is not one-sided, and the unpaired aggregates on both sides are noise.
- **Placement advice on the transient arena's committed backing** — cannot be a fix. `iree_hal_buffer_params_t`
  has no placement member (`usage`, `access`, `type`, `queue_family_affinity`, `min_alignment`;
  `runtime/src/iree/hal/buffer.h:646`), and the only advice API is advisory by contract: *"incorrect hints may
  reduce performance but will not cause incorrect behavior"* (`iree/hal/allocator.h:495`). The HRX transient
  path never calls it, but a hint that cannot cause incorrect behaviour cannot resolve one either.
- **The staged host writeback path** — excluded. `download_prepared_host_staging` gates on `has_download`,
  which needs `staging.download = access.write && !binding.graph_input` **and** a binding with non-null
  `host_data` — set only for non-directly-bindable *host* allocations. No graph node writes into a host tensor
  on this path, so `PendingHostWriteback` is inert: 11 valid arms, 0 `HRX host writeback` lines, with the
  reporting path proven to fire under `=poison`.
- **The direct device-to-host copy not being covered by the wait** — falsified.
  `stream_synchronize_sleeping` (`runtime/hrx-sleeping-wait.cpp:127`) flushes **before** it waits, so the copy
  recorded by `backend_get_tensor_async` is submitted and then awaited.
- **Pinning `f2099e9b7` (the `1bit/amd-core-our-hrx` cell)** — not available as a fix.
  `git merge-base --is-ancestor f2099e9b7 26330cc20` is **true**: it is a strict ancestor of the pin
  `engine@main` already ships (`third_party/llama.cpp @ 26330cc20490a547f94c9e5c960bf93063453151`), so adopting
  it would be a revert, which this issue's constraint excludes. Its `322/322 MUL_MAT` result is therefore **not**
  a working configuration that the shipped pin lost — whatever makes that cell clean is already in `26330cc20`,
  which faults. The regression is in what the `1bit/hrx-vulkan-patched` merge added on top, and `f2099e9b7` is a
  known-clean lower bound to bisect from.
- **The `TRANSIENT_REUSE` knob is not a lever for this fault** beyond the arms above: truncation is not one-sided,
  and the unpaired aggregates on both sides are noise.
- **The sleeping wait returning before completion** — cleared: it falls through to `hrx_stream_wait` and returns
  that status.

### Traps this issue has already sprung

- **Wrong device.** Results are only meaningful with an explicit provider/device; a run against the wrong backend
  produces plausible-looking numbers that answer a different question.
- **Vacuous pass.** Zero NaN can be achieved by failing differently — an arm that aborts early logs no NaN. Both
  faces above must be counted, plus complete-arm counts.
- **Stale incremental builds.** Several earlier measurements on this issue were retracted as build artifacts: a
  nested build directory that had previously compiled a *different* `ggml-hrx` carried ~1.3 MB of stale objects.
  Fresh build directories with the engine's own `ExternalProject` arguments are required.
- **Silent arm loss.** `rt-det-direct.sh` takes the box lock on fd 9 and passes it to the `nohup`'d server, so an
  abnormal arm death leaves an orphan holding the flock and every later arm wedges without running. One-word fix
  at the launch site: `... 2>&1 9>&- &`.

### What remains unknown

The fault reproduces on the shipped pin — `pair 2 arm c: FAULT-NaN nan=1 req=34 gtt_start=102G`, build
`10604 = 3971f49bf`, `parse=0` ruling out the format-500 mode — so the shared-memory tile-visibility change does
not eliminate it. The rate study is complete, and its per-arm regime gate and truncation accounting are reported
here — both halves ran at `gtt_start = 102 GiB` on all 70 arms, and parse-500 truncations are counted separately,
never scored as clean arms:

```
A  shipped fix  (10604 = 3971f49bf)   40 scheduled, 40 in-regime, 37 complete
   clean 66/66: 35   NaN faults: 2 (both parse=0, inside the gate)   rate 2/37 = 5.4%  (95% CI 0.7-18.2%)   parse-500: 3
B  always-stage (10605)               30 scheduled, 30 in-regime, 25 complete
   clean 66/66: 24   NaN faults: 1                                    rate 1/25 = 4.0%  (95% CI 0.1-20.4%)   parse-500: 5
Fisher exact two-sided p = 1.000;  P(rate_A < rate_B) = 0.460
```

So the tile-visibility pin did not measurably change the pressure-dependent all-NaN rate, and the fault is still
live on the shipped pin. The reading is bounded, not directional: both intervals contain the ~10% figure that
motivated the change and neither excludes "no change" (and that ~10% comparator itself rested on 1 fault in 15
arms, 95% CI ~0.2-32%, so the lower rate needs no environmental explanation). No runtime-side ordering gap on the
device-to-host path could be named, and no forward-pin fix candidate
survives (the only one proposed is an ancestor of the shipped pin, i.e. a revert). The open candidates are in the
kernel/binding family: GLM is the only model whose decode splits into two 128 KB AQL command blocks (919 + 760
dispatches) with 1,399 barriers per token.

**Pin note (2026-10-09).** `engine@main` has since moved twice: first to `third_party/llama.cpp e44c9d01a4d5`
(#347), then to `0e255d774032` (#373, the fork tip after the NVFP4 / Q2_0 / IQ GET_ROWS merges), both with
`third_party/hrx-system 98d05d94`. That is the same runtime the study above ran against, and `26330cc20490` is an
ancestor of both, so the measurements above still describe what ships. Two things learned on the way:
a bump that moved `hrx-system` alone to `4ba76c18eafe` (2026-10-08) left `main` unbuildable with
`ONEBIT_HRX=ON` (the pinned `ggml/src/ggml-hrx/loom-jit.cpp` still used the LoomC API that revision removed; reverted
in #363, and #365 now holds `hrx-system` when LoomC symbols go missing); and with the port of `loom-jit.cpp`
(fork #97) that revision compiles but returns an immediate end-of-text on a 4.7K-token prompt where `98d05d94`
answers (#369), so `hrx-system` stays at `98d05d94` and moving it needs a multi-ubatch prompt check, not a symbol
check. The #373 pin passed that check: needle 20/20 on Qwen3-0.6B.

### Instrumentation available for the bisect (implemented, not yet built)

On `f2099e9b7`: a probe on the route the logits actually take (`backend_get_tensor_async` →
`hrx_stream_copy_buffer`) that names the producing graph `value`. It recovers the `ValueId` by searching the
retained `GraphProgram` value maps for the downloaded tensor pointer (`GraphProgram::graph()` exposes the
`ValueMap`; each `Value` holds `const ggml_tensor * tensor`), records at copy time, and scans after the
flush-and-wait that publishes the data, with a `=poison` mode so the reporting path can be proven to fire. It
compiles (single TU). It is deliberately unbuilt and unrun: relinking rebuilds every TU that includes
`backend-context.h`, and that CPU inside a GTT-bound measurement is the confound that already invalidated one run
of this experiment.
