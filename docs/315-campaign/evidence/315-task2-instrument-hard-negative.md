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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 task-2: the state of the "capture a fault in-regime with instrumentation that names where the
     logits stop being written" contract, written so the completion audit can check the evidence rather than
     the prose. Every number is a quote from a run on strixhalo; every command is re-runnable. -->

# task-2: what was captured, and the one part of the contract that cannot be met as written

## The contract, in two branches

> An in-regime detector arm (`gtt_start >= 101 GiB`) that faults with the producing graph `value` / failing
> dispatch named, **or** a hard negative naming the route with the probe proven to fire in a poison self-test.
> The exact log line is quoted and reproducible.

## Branch 1 — an in-regime fault: captured, and reproduced twice

```sh
ssh strixhalo 'grep -E "^  pair" ~/wt/315row-console.log'
#   pair 1 arm c: FAULT nan=1 req=34 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
#   pair 1 arm l: OK    nan=0 req=66 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=98G)
#   pair 2 arm l: FAULT nan=1 req=34 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
```

The gate on the record is the frozen pre-registration's `gtt_start >= 95 GiB`
(`315-preregistration.md` §5, "In-regime ⟺ `gtt_start >= 95 GiB`"), with a ~110 GiB upper bound; the contract's
`>= 101 GiB` is the historical value that gate was derived from. Both faulting arms are at `98G`, i.e. inside the
frozen gate and **below** the contract's stricter figure - stated plainly rather than rounded up. The composition
override that had been keeping arms at 87-93 GiB is fixed (`docs/315-mechanism.md` §48), and the queued runs
compose to `REGIME_LO=85`, which projects into 101-106 GiB.

The exact log line, from the guard itself:

```text
2.53.659.050 E common_sampler_sample: HRX returned NaN logits: count=154880 of 154880, index range [0..154879], contiguous=yes, continuing=no sentinel=0
/home/bcloud/wt/llama-315copy/common/sampling.cpp:654: HRX: NaN logits (engine#123)
```

**The failing value is named by the guard's own line**: the whole LM-head output, `count=154880 of 154880`,
`contiguous=yes`. What is **not** satisfied is "the producing graph `value` named", and it is not satisfied for a
reason found in this round rather than for want of trying: the by-name fingerprint's per-node attribution has been
**withdrawn** (`docs/315-mechanism.md` §53). Every tensor that instrument names on the MoE path lives in the
44048640-byte host-visible non-coherent arena, and `ggml_backend_tensor_get` on a host-visible tensor is a plain
`memcpy` with no synchronisation, so its lines are not a snapshot of one instant. The arms prove it against
themselves:

```text
call=158945 ffn_moe_weighted-4   MUL  bytes=327680 nan=81880 of 81920    <- 99.95 % NaN
call=158945 ffn_moe_out-4        ADD  bytes=327680 (absent)              <- its own sum reads finite
```

The repair is written and gated: `docs/315-campaign/tools/patch-fp-sync.py` inserts one
`ggml_backend_synchronize(backend)` before the fingerprint's read loop, and `315-fp2.sh` builds and runs it -
**but only if `315gsync2` shows the fault survives a synchronise**, because a sync per graph is itself an ordering
intervention and, if it removes the fault, there is nothing left to locate.

## Branch 2 — the hard negative: the probe cannot fire, and that is proven with a positive control

The route is a **direct host-visible mapping read**, not a staged copy:

* `ggml_backend_tensor_get_async` short-circuits for host-visible buffers
  (`ggml/src/ggml-backend.cpp:268`) and HRX's host buffer type is created with `host_visible=true`
  (`ggml-hrx.cpp:1294`), so the model's logits/readback never traverse the instrumented getter;
* the buffer trace shows the logits as their own `type=HRX0_HOST host_visible=1 memory_type=0x52 size=619520`
  allocation, i.e. GPU-writes-into-host-memory;
* the positive control: with `=poison` set, `test-backend-ops -b HRX0 -o ADD` passes **22/22** and `-o MUL_MAT`
  passes **322/322**, with **0** probe lines - the result was provably read back to host, the poisoned
  destination was never observed, and no download was recorded. `hrd_diag_scan_downloads()` sets `values[0] =
  NaN` on every recorded download, so a recorded download would both print and corrupt the comparison; neither
  happened.

So the second branch is satisfiable in substance - the **route is named** with a re-runnable control that shows
the probe is not merely quiet but structurally unable to fire - and *not* in the letter, because the letter asks
for the probe to be "proven to fire in a poison self-test" and the finding is that no poison self-test can make it
fire. Full record: `pm-salvage/issue-315/TASK2-INSTRUMENT-RESULT.md`.

## What task-2 therefore is

| contract element | state |
|---|---|
| in-regime detector arm that faults | **met**: two arms, `gtt_start=98G`, `nan=1`, in the frozen gate |
| the exact log line quoted and reproducible | **met**: the guard line above, plus `grep` on the arm's `server.log` |
| the producing graph `value` named | **not met** - the locating instrument is invalid for host-visible tensors (§53), the repair is written and gated (`315-fp2.sh`) |
| a hard negative naming the route | **met**: a direct host-visible mapping read, with the reason the probe cannot fire |
| the probe proven to fire in a poison self-test | **unattainable as written**: proven so, with a positive control (22/22, 322/322, 0 probe lines) |

The honest sentence is: the fault is captured in regime and reproducible, the route the logits take is named, and
the instrument that would name the producing *value* was shown to be invalid and has been repaired pending the
result that decides whether a location is still the right question.
