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

- `docs/315-campaign/evidence/315-evict-trace-status.md` — the trace campaign blocked by this
- `docs/315-campaign/evidence/315-evict-trace-instrument.md` §6g — why `npu.lock` is the real
  mutex (the older 315 harnesses take `box.lock`, which excludes nobody)
- `~/amd-oss/315-TOOLS.md` — the box-tooling inventory
