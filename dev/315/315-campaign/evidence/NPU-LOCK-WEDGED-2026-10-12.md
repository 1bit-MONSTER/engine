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
# `~/npu.lock` is wedged by a dead holder — every GPU consumer on strixhalo is blocked

**Found:** 2026-10-12 ~13:26–14:00 ADT, while tracing why an engine#315 trace campaign could
not start. **Not specific to #315** — this blocks every goal that takes the box-wide GPU lock.

## The condition

`/proc/locks` for the inode behind `~/npu.lock` (`103:02:80249278`). Unindented line = the
**granted** lock; `->` lines are **waiters**:

```
10: FLOCK  ADVISORY  WRITE 826940  103:02:80249278 0 EOF      <- HOLDER (no indent)
10: -> FLOCK  ADVISORY  WRITE 867745  103:02:80249278 0 EOF   <- waiting
10:  -> FLOCK  ADVISORY  WRITE 99830  103:02:80249278 0 EOF   <- waiting
10:   -> FLOCK  ADVISORY  WRITE 1155054 103:02:80249278 0 EOF <- waiting
```

**The holder `826940` is dead.** `/proc/826940` does not exist; its whole group
(826913 / 826939 / 826940, running `/tmp/probe_model.sh`) has exited. The kernel still lists
it as the granted holder and still refuses new acquirers.

## It is stuck, not busy — reproduced

```
flock -n ~/npu.lock      -> HELD at t=10s, 20s, 30s
flock -n box.lock        -> FREE immediately (same filesystem, same mechanism)
```

`box.lock` acquiring instantly proves `flock` itself is healthy; the condition is specific to
`npu.lock` and is permanent, not transient contention.

## Who it blocks

| pid | waiting | what |
|---|---|---|
| 867745 | 3,191 s | a #315 arm (`fx-c5-lockall-r1`) |
| 99830 | 8,462 s (2 h 21 m) | `batch3.sh` (loom-ub389 batch) |
| 1155054 | 289 s | a `bisect4.sh` worker |

Plus, queued at various depths: **the `dx long-context` goal's workers**
(`probe_model.sh` on q4/q8 kernel sets, `dxctx-prof/probe-*`) and **two other goals'
`npu-forge2-kvsplit` bisect/serve jobs**. So at least three goals are stalled by this.

## Cause — hypothesis, not a finding

The dead holder was started under `systemd-run --user --scope`. If a scope is torn down while
one of its processes holds a granted flock, the grant can outlive the process in
`/proc/locks`. If that is what happened, then

```sh
flock ~/npu.lock some-transient-command
```

is an **unsafe pattern on this box**: the advisory lock's holder can be reaped out from under
it, wedging the lock for everyone. The safer form is to hold the lock in a long-lived shell
(`exec 9>lock; flock 9`) whose lifetime you control, which is what `315-attrib.sh` and
`probe_model.sh` both do — the wedged holder was a *different* pattern.

**Not verified.** Verifying it means reproducing a scope teardown with a granted flock, which
would wedge the box again. Recorded as the most likely explanation so the next person can
test it deliberately rather than by accident.

## Not cleared, deliberately

`flock` has no userspace break. The options are:

- **`fuser -k ~/npu.lock`** — kills the three *innocent* waiters' slots; their work is lost,
  and one of them is another goal's arm.
- **reboot** — clears it, loses everyone's state.

Both are destructive and affect work that is not mine. Left for the user.

## Prior art / related

- `dev/315/315-campaign/evidence/315-evict-trace-status.md` — the trace campaign blocked by this
- `dev/315/315-campaign/evidence/315-evict-trace-instrument.md` §6g — why `npu.lock` is the real
  mutex (the older 315 harnesses take `box.lock`, which excludes nobody)
- `~/amd-oss/315-TOOLS.md` — the box-tooling inventory

## Correction: the wedge does not make a run impossible — it makes it a policy choice

After writing the above I checked whether a run is physically possible despite the wedge. It
**is**, and the blocked note overstated this. Measured:

| quantity | value |
|---|---|
| HRX0 occupancy | 0 processes, sampled over 60 s |
| baseline GTT | 26.7 GiB; **only two foreign holders**, both **Vulkan** (24.5 GiB at pid 846805, 3.4 GiB at 1618706) — not HRX |
| `MemAvailable` | 82 GiB |
| `Tctl` | 53–54 °C (well under the 72 °C gate) |

And critically, the stalled waiters hold **0 bytes of GTT**:

```
pid 867745   GTT=0B   flock 7
pid 99830    GTT=0B   flock ~/npu.lock ./batch3.sh
pid 1155054  GTT=0B   flock -w 7200 9
pid 826914   GTT=0B   probe_model.sh ...
pid 1155052  GTT=0B   bisect4.sh ...
```

So all six are parked on the lock with no GPU state at all. A campaign could reach the
95–110 GiB regime without clobbering anyone's resident memory.

**The only reason not to run is the decision itself.** Those three waiters are queued for
HRX0 work; a composition that takes HRX0 to 98–104 GiB would evict work they are stalled
waiting to do — using a kernel bug as an advantage over other goals. Two of them are the
`dx long-context` goal's own probes. That is a call for the user, not for me.

**Why this still blocks the goal as stated:** the campaign harness I built deliberately
refuses to run when `~/npu.lock` is unobtainable (§6g), because that lock is how this box
serialises GPU use. Overriding that guard would be me weakening my own safety check to get a
result — which the goal's constraints explicitly forbid ("do not weaken the existing guards
or instrumentation in a way that destroys failure evidence"). If the user wants the run, the
honest form is to say so explicitly, or to clear the wedge first.
