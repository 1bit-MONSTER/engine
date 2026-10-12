# task-8: eviction/preemption trace — live status

Run `315evict2` on strixhalo. Purpose: attach the KFD SMI eviction/preemption stream to
in-regime arms so a **faulting** arm can be correlated against the driver's own events.
This is NOT an acceptance run (task-9 owns acceptance).

## Setup (verified before the trace was trusted)

- watcher as root, TTL 10800 s (unprivileged it writes a header-only file and exits 0 —
  silently; see `315-evict-trace-instrument.md`)
- regime reached: GTT 98–104 GiB, gate is `>= 95`
- control cell `315instr3-build` (pin, `if (direct_host_binding)` intact) vs fix cell
  `315fix-build` (pin + HOST_COHERENT). Both report `e44c9d01a`, so this is the diagnostic
  pair — it tests the falsified coherence lever, not a candidate fix.
- arm→PID binding is exact (`tools/arm-pid.py`, via the port in the arm's own server.log)

## Eviction stream (the substantive measurement)

| quantity | value |
|---|---|
| Q_EVICT | **14,570** |
| — trigger=SVM | 8,594 |
| — trigger=USERPTR | 5,976 |
| Q_RESTORE | 18,769 |
| — `rescheduled=no` | **14,566** |
| — `rescheduled=yes` | 4,203 |
| VMFAULT | **0** |
| PAGE_FAULT | 0 |
| sustained rate | ~972/min (≈16/s) at GTT ≈ 104 GiB |

## Arms so far

```
pair 1 arm c: OK nan=0 req=66 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0
pair 1 arm l: (running) 152 requests served, 0 faults both faces
```

## What this does and does not establish

**Does:** the preemption/eviction window the #123/#140 lead describes is **open and
intense** during these arms — 14.5k queue evictions, of which 14.6k restores were *not*
rescheduled (i.e. the evicted queue's work was dropped and had to be redone). The
composition is genuinely exercising the driver's eviction path, not idling near it.

**Does NOT:** produce a fault. Both scored arms are clean under that pressure.

So the honest reading at this point is **negative for the lead as stated**: heavy
eviction/preemption in regime is present *and* the arm is clean. Eviction pressure is
therefore **not sufficient** on its own at this resolution. That is narrower than the
#123/#140 record predicts ("the window opens when a dispatch is preempted, and any
process's KFD queue eviction ... does that"), and it must not be written up as support for
the mechanism.

Two caveats against over-reading the negative, both required before it is treated as a
result:
1. the run is incomplete (1 of 12 arms scored), so the fault may simply not have occurred
   yet in this composition;
2. the fix cell here carries HOST_COHERENT — a lever already falsified — so a clean fix
   arm is expected and says nothing about the fault.

## Contract reminder (task-8)

Satisfied by EITHER `>=1` in-regime faulting arm with the correlation attached, OR the
trace recorded **UNINFORMATIVE with its clean-arm count**. A clean trace is not evidence
about the mechanism. If the run ends clean, that is what will be recorded.
