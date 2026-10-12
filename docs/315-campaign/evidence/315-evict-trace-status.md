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
pair 1 arm c: OK  nan=0 req=66 parse=0 gtt_start=98G   <- IN regime, clean
pair 1 arm l: OOR nan=0 req=66 parse=0 gtt_start=93G   <- OUT of regime, NOT COUNTED
```

Both arms served 66/66 with `utf8_fail=0 parse_fail=0 http_fail=0 transport_fail=0
truncated=False` and no NaN. But the driver's own gate ("out-of-regime `gtt_start` < 95 GiB
(reported, not counted)", `315-paired.sh:12`) excludes arm `l`: it started at 93 GiB, below
the frozen 95 GiB gate.

**So the tally to date is 1 in-regime arm, 0 faults — not 2 clean arms.** An `OOR` arm is
neither clean evidence nor a fault and must not be summed with the clean ones. The
composition does not guarantee every arm lands in band: `gtt_start` is sampled per arm
(`315-paired.sh:75`) and drifted by 5 GiB between the two arms of this pair.

If the drift persists, a 12-arm run yields roughly half in-regime arms — which is far short
of task-9's `>=60` but is enough for task-8, whose requirement is a single faulting arm or
an honest null.

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

---

## Run 1 (`315evict2`) — completed, but TRUNCATED, and why

The first real trace run ended after **2 arms, not 12**. Cause: my wrapper passed
`MINCLEAN=0` intending "no minimum", but `315-paired.sh:147` treats MINCLEAN as a goalpost
and stops when the clean count REACHES it — so `0 >= 0` stopped the run at pair 1. See
`315-evict-trace-instrument.md` §6a. Fixed (`MINCLEAN=$((PAIRS*2+1))`), plus §6b (a
`grep -c || echo 0` producing a two-line integer).

What the truncated run did establish, at full strength, is the **eviction environment**:

| quantity | value |
|---|---|
| Q_EVICT | **15,357** (8,936 SVM + 6,421 USERPTR) |
| Q_RESTORE | 19,649 |
| — `rescheduled=no` | **15,353** |
| VMFAULT | 0 |
| arms scored | 2 — `c` in-regime OK, `l` **OOR** (93 GiB < 95 gate) |
| in-regime arms | **1** |
| in-regime faults | **0** |

So: 15.4k evictions, 15.4k of them unrestored, sustained across the window, with one
in-regime clean arm. **This is a null for task-8, not a result** — and it is explicitly
*not* recorded as evidence about the mechanism.

## Why run 1 cannot be read as anything stronger

1. one in-regime arm is nothing like the `>=60` task-9 needs, and task-8 itself wants a
   *faulting* arm to correlate;
2. an `OOR` arm is neither clean nor faulted, so the fix cell contributed **no** in-regime
   data at all;
3. the campaign was truncated by a harness bug, so its length was not the designed length.

The honest statement is: **the eviction/preemption window was open and intense, and the
one in-regime arm was clean.** That is a datapoint about eviction pressure being present
without the fault, and nothing more.

## Run 2 — not launched: box contention

A relaunch on 2026-10-12 13:09+ found a peer agent's HRX workload on the box
(`~/wt/loom-ub389`, `guard.sh`-wrapped, 1800 s timeout each, first a 27B Qwen3.8 run then a
Zamba2-VL-2.7B run, `npu.lock` freshly touched). The campaign composes ~5 neighbours into
the 95–110 GiB band, which would have evicted that peer's memory. Per the goal's own
constraint (take the box lock, do not disturb other workloads) the relaunch **waits** for
the box rather than competing for it.

This is the second time the box has been unavailable for a long in-regime window (the
first was `repro336` holding 103–111 GiB). It is a real constraint on task-8, not an
excuse: the trace needs an uninterrupted composition, and the box is shared.

## Run 2: deferred, with the reason and the resume command

Not launched. The box is held by a peer agent's **batch** (`~/wt/loom-ub389/runs/batch2.log`,
active 13:10, jobs queued, several `-dev HRX0`) plus a CPU embedding pass. The campaign's
composition drives ~5 neighbours into the 95–110 GiB band, which would evict that peer's
memory. The goal's constraint is explicit — take the box lock, do not disturb other
workloads — so the trace waits.

Resume when the box is quiet, with the two launch bugs fixed:

```sh
ssh strixhalo
# confirm free: no "-dev HRX0" jobs of anyone else's, GTT near the ~26 GiB baseline
ps -eo pid,args | grep -- "-dev HRX0" | grep -v grep
awk '{printf "%.1f GiB\n", $1/1073741824}' /sys/class/drm/card*/device/mem_info_gtt_used

# then, with the corrected wrapper (MINCLEAN=$((PAIRS*2+1)), PAIRS cap ends the run):
nohup bash ~/wt/315-evict-trace.sh 8 315evict3 > ~/wt/315evict3-run.log 2>&1 &
```

8 pairs = 16 arms, which at the observed ~1-in-2 in-regime rate gives ~8 in-regime arms.
That is still short of task-9's `>=60`; task-8 needs only one **faulting** in-regime arm, or
an honest null with its clean-arm count.

**If the box stays contended, task-8's honest close is the null already in hand**: 1
in-regime clean arm under 15.4k evictions, recorded as uninformative, with the reason being
box contention rather than a design choice.
