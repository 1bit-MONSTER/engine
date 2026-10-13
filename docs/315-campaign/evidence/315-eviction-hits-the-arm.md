<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The eviction hits the ARM's own buffers, not only the forcer's — read from the trace

Written while arm B (`315ab1b2`) runs, to fix a misreading before it enters the record. Reading
`~/wt/evictonly.hip` suggests the intervention is self-contained ("only makes KFD evict and restore
**its own** queues"), and the mechanism statement's wording ("eviction of in-flight KFD work") could be
read either way. The trace settles it, and the settled reading is stronger.

## What `evictonly` does

    hipHostRegister(16 MiB anon range)          # a host-visible, KFD-managed range
    hipMalloc(4B); hipMemset; synchronize       # ensures a queue exists
    loop: mprotect(PROT_READ) <-> mprotect(PROT_READ|WRITE); touch a page; usleep(500)

The `mprotect` churn on a registered range is what makes KFD evict and restore queues. The comment in
the source frames this as the process's *own* queues — but the file is describing what it *causes*, not
who *receives* it.

## What the trace shows

The control run's SMI stream, per arm, attributed by the arm's own server pid
(`315-task9-session-state-2026-10-12.md`, P3 section):

    arm pid   Q_EVICT   Q_RESTORE
    21139         293         411
    44443         160         173
    68030         262         288
    91692         196         257
    114850        275         307

**Every arm's own buffers are evicted and then restored while the arm is live.** So the eviction is a
device-wide event under memory pressure, not something confined to the forcer process. `evictonly` is a
*pressure generator*; the arm is a *victim*.

## Why this sharpens the mechanism

The mechanism statement should be read as:

> generating KFD evict/restore pressure while the GLM server has in-flight work causes **the server's
> own queue memory** to be evicted and restored mid-flight, and something read across that boundary is
> consumed without re-validation, producing both faces.

That is a materially different and more specific claim than "the forcer's memory gets evicted." Two
things follow, both checkable:

1. **The stale data is in the server's own evicted-then-restored memory**, not in the forcer's — which
   is exactly why the per-arm correlator (`315-evict-per-arm.py`) compares the *arm's* pre-fault
   eviction count against a clean-arm baseline (~228/arm). The instrument is already aimed at the right
   pid.
2. **The read fix addressed a real incoherence but the wrong one.** The logits read is host-side; the
   eviction is of device-side queue memory. Routing the host read through a coherent transfer cannot
   repair a device-side restore that left different bytes behind — consistent with arm B's cell-`e`
   fault with the knob on.

## The pre-registered refutation test this enables

P3 as written: *"if an arm faults at cwsr=1 with ZERO Q_EVICT for its own pid, H2' is dead."* That test
is aimed at the arm's own counters and remains the right instrument. It is live on the current run: the
re-run's per-arm SMI stream should be read for arm `315ab1b2-r1e`, where a fault already landed at
10:33:19, to test P3 against a real faulting arm rather than assuming it.

## What this does not do

It does not locate the cause. It narrows the cause to "something the server reads from its own
evicted-then-restored memory without re-validating," and rules the forcer's memory out as the site.
The surviving candidate families are the same ones §55/§56 left open (executor-level cross-dispatch
ordering, or a different class entirely), not the read path this A/B tested.

---

# HARNESS GAP (same turn) — P3 cannot be tested on the A/B runs, because no SMI watcher was launched

The section above promises to "read the re-run's per-arm SMI stream for arm `315ab1b2-r1e`". That is
**not possible on these runs**, and the gap is recorded rather than worked around.

## Why not

The eviction counters come from a separate privileged watcher, launched by
`docs/315-campaign/315-evict-trace.sh`, not by the campaign driver:

    WATCH=$HOME/amd-oss/tools/315-trace/kfd_smi_watch                 # 315-evict-trace.sh:25
    sudo -n true || { echo "ABORT: the watcher needs passwordless sudo"; exit 2; }   # :40
    sudo -n nohup "$WATCH" -e all -t 10800 -o "$OUT/kfd_smi.tsv" -q & # :146

`~/wt/315-disc.sh` — the driver the A/B used — contains no SMI watcher and no `sudo` call. It records
only `side=<alive>/<launched>` per arm, which proves the forcer was *running*, not what KFD did.

Neither `315ab1b2` nor its discarded predecessor launched the watcher, so for both A/B arms:

    ~/wt/315ab1b2-evict/         does not exist
    pgrep kfd_smi_watch          nothing

The raw interface is also not open to me: `/sys/kernel/debug/kfd/kfd_smi_events` does not exist on this
box's mount, and the script's own guard requires passwordless sudo, which is the documented reason the
watcher is a separate opt-in step.

## What this costs, and what it does not

- **Costs**: P3 (the pre-registered "a faulting arm with zero Q_EVICT for its own pid kills H2'") cannot
  be evaluated on the A/B. The eviction refinement above rests on the *earlier* control run's counters,
  not on this one. A future run that wants P3 must launch the watcher with the driver.
- **Does not cost**: the A/B's own reading. The A/B tests the fix against the forcer, and `side=1/1`
  establishes the forcer ran for the whole arm, which is what the pre-registered cell design needs. The
  missing counters bear on *which pid* is evicted, not on whether arm B faulted.
- **Is not a reason to re-run the A/B.** Arm B's cell-`e` fault with the knob on already lands in the
  pre-registered "H-fix REFUTED" branch. Adding eviction counters would refine the mechanism story; it
  would not change the A/B verdict.

## Lesson

The campaign driver and the eviction tracer are separate entry points, and the pre-registration did not
say which one to use. For any future run whose reading depends on P3, the tracer must be named in the
pre-registration and launched with the driver — otherwise the run cannot answer the question it was
designed around, which is the same class of harness/measurement mismatch recorded in
`315-evict-trace-instrument.md` §6a–§6h.
