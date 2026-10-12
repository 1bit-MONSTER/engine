# After the size explanation: three surviving explanations, and their discriminating tests

**Date** 2026-10-12. Supersedes the causal story in `315-cwsr-hypothesis.md`, which is
falsified (`315-cwsr-sizes-excluded.md`).

## Where we are

- **Signal:** `cwsr_enable=0` → 0 faults in 11 in-regime arms, every arm at `gtt_start=97G`,
  against a 26% multi-session baseline (34 faulting / 131 in-regime arms) at
  `cwsr_enable=1`. Binomial p = 0.030; Fisher p = 0.107; rule-of-three bound 27.3% does not
  exclude 26%. **Suggestive, not significant.**
- **Falsified:** the area is *undersized*. Both sizes our runtime declares for gfx1151 are
  correct — VGPR 0x60000 (384 KB, the 1.5× value from `#1807`, present) and SGPR 0x4000
  (16 KB, standard). The kernel's `cwsr_size` check is also the fixed `<` form, so a
  mismatch can no longer silently under-allocate. **Size is excluded.**

That leaves exactly three explanations, each with a cheap discriminating test. They are
listed here *before* the 60-arm result so the next step is not chosen after the fact.

## H1 — fresh-boot effect (the confound)

The reboot changed more than the one cmdline token: run 7 ran on a long-uptime box with
fragmented device memory, the `cwsr=0` run on a fresh boot.

**Falsified by:** a **fresh boot with `cwsr_enable=1`**, running the identical campaign
(same PAIRS, same hour of day, same composition). If that also gives ~0 faults, CWSR is not
the cause and the 11-arm signal was a boot effect. If it gives ~26%, H1 is dead.
**Cost:** one reboot + ~50 min.

## H2 — CWSR save/restore is not bit-exact under preemption on gfx1151

Not a size bug: a hardware/firmware characteristic of the spill itself. A preempted wave
resumes with subtly wrong state, and what the damaged state holds decides the observable —
finite → Face B, exponent/NaN → Face A.

This is the explanation that fits **everything** measured, including what nothing else did:

- the GTT dependence: preemption happens under memory pressure;
- why coherence, ordering, AQL barriers, `-fa off`, transient reuse, PM4 and the MoE routing
  levers all did nothing — none of them changes whether a wave is saved and restored;
- why the corruption is already whole-batch at layer 0 — a damaged wave corrupts its whole
  tile, not one tensor's memory;
- why run 7 faulted 3× at `gtt_start=97G` while 8 clean arms sat at the same 98 GiB — the
  *occurrence* of preemption is the intermittent variable, not the pressure level.

**Discriminating test (cheap, no campaign):** count CWSR preemptions per arm and correlate
with faults. If faulting arms have preemptions and clean arms do not, that is direct
evidence. Candidate observables: the `hqds`/`mqds` debugfs nodes, the CWSR control-register
writes visible in a ctrlcode-style trace, or an eBPF probe on the MES/queue path.
**Cost:** instrumentation, no reboot.

## H3 — the partial kernel-side fix matters

Our module carries **one fixed and one unfixed** size check:

```
"queue cwsr size ... not sufficient for"   FIXED   (d15deaf)
"queue ctl stack size ... not equal to"    UNFIXED
```

Both sizes our runtime declares happen to be correct, so the `!=` check passes today — but
it is a latent constraint: any future change that makes userspace declare a *larger* control
stack would be rejected outright (EINVAL, loud) rather than accepted. That is a robustness
gap, not (on current values) a defect.

**Falsified by:** anything above. It is listed for completeness and because it is the only
actual code asymmetry found in the kernel path.

## What is NOT a candidate

- arena knobs, transient reuse — falsified earlier in the issue's record;
- MoE routing / partition tables — rebuilt in-plan, falsified;
- the routing-tensor and by-name locator routes — withdrawn (reused arena slots);
- **area size on either the VGPR or SGPR half — excluded today;**
- stream ordering / visibility / barriers — closed by run *and* at source.

## The order of the next steps

1. finish the 60-arm `cwsr=0` campaign (running) — establishes the rate, not the cause;
2. **H1 control**: fresh boot, `cwsr_enable=1`, identical campaign. Without this, step 1 is
   not attributable to CWSR at all;
3. only if H1 is survived, **H2 instrumentation** to catch the preemption directly.

Doing step 3 before step 2 would risk instrumenting a boot effect.
