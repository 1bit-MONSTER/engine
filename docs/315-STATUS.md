# engine#315 — STATUS / resume-here

**Updated** 2026-10-12. Single entry point. Everything else is referenced from here.

## Task state

| task | state |
|---|---|
| task-1..3 | complete (earlier sessions) |
| task-4 | completed against a FALSIFIED candidate; retracted in the record — read `docs/315-campaign/evidence/ISSUE-COMMENT-5-2026-10-10.md` addendum |
| task-5 | complete — falsified-lever table rebuilt from the run consoles |
| task-6 | complete — closed families cited at source with verified file:line |
| task-7 | complete — the preemption-window lead recorded as a lead, with its two unbuilt instruments |
| **task-8** | **complete** — run 7 gave 3 in-regime faults across both faces with the Face-B arm's PID correlated (323 Q_EVICT / 357 Q_RESTORE) |
| **task-9** | **gated — NOT started.** Conditional on a *named mechanism*, which does not exist. |

## The one live lead

`amdgpu.cwsr_enable=0` vs `1` (CWSR = compute wave save/restore on queue preemption).

| | `cwsr=1` | `cwsr=0` |
|---|---|---|
| in-regime arms | 11 (run 7) | 11 |
| faults | **3** (Face A ×2, Face B ×1) | **0** |
| regime | 97–98 GiB | 97 GiB |

First intervention in this issue's history to move the rate in the predicted direction.
Statistics at n=11: binomial p=0.030 (suggestive); Fisher p=0.107 (not significant);
rule-of-three bound 27.3% does **not** exclude the 26–27% multi-session baseline.

## What is FALSIFIED (do not re-propose)

- **area size, both halves.** libhsakmt returns the correct values for gfx1151 — VGPR
  `0x60000` (384 KB = 1.5×, the `rocm-systems#1807` fix IS present) and SGPR `0x4000` (16 KB).
  Verified at runtime by linking `queues.c.o` from `libhsakmt.a` and calling the exported
  functions. Both are plumbed into the allocation and topology paths.
- coherence, consumer ordering, AQL barriers, `-fa off`, transient reuse, PM4, MoE routing
  tables, the routing-tensor and by-name locator routes, arena knobs.
- the "half-applied patch pair" story: the kernel half is in, and the userspace half is too.

## The confound that must be settled FIRST

The reboot that set `cwsr_enable=0` **also** gave a fresh boot: unfragmented device memory,
no prior campaigns, cool box. Run 7 ran on a long-uptime box. So a clean `cwsr=0` result is
**not yet attributable to CWSR**. This issue has already been burned once by exactly this
class of error ("the rate followed memory-pressure settings").

**H1 CONTROL** — fresh boot with `cwsr_enable=1`, identical campaign. Scripted and ready:
`docs/315-campaign/315-cwsr-control.sh`. Sized to **20 arms, not 60**: at the 26% baseline,
20 arms give ~5.2 expected faults and P(0 faults) = 0.002, which is decisive; 60 would cost
~3 extra hours for no added discriminating power.

Reading key: **~26% → H1 dead, the `cwsr=0` signal is real. ~0% → H1 confirmed, the signal
was a boot effect.**

## Ordered next steps

1. **finish the running campaign** (`~/wt/315cwsr60-console.log`, tag `315cwsr60`, 48 pairs,
   `cwsr=0`). It establishes the RATE, not the cause.
2. **run the H1 control** (reboot required; kills any running campaign).
3. only if H1 survives, **H2 instrumentation** — count CWSR preemptions per arm and correlate
   with faults. Doing H2 before H1 risks carefully measuring a boot effect.
4. then the fix, the regression comparison, and the KNOWN-ISSUES update.

## The forward-only fix is UNRESOLVED

With size excluded, "bump the HSA runtime" is **not** the fix. The only remaining candidate is
an engine-side **startup guard** that detects the bad configuration and refuses rather than
silently faulting — that is a **workaround, not a repair**. Do not present it as solving the
issue while the underlying defect is unidentified.

The engine has no HSA pin to move: `IREE_HAL_AMDGPU_LIBHSA_PATH` points at whatever the ROCm
SDK ships.

## Key paths

| | |
|---|---|
| live treatment log | strixhalo `~/wt/315-cwsr-test/phase1-treatment.txt` (255+ lines) |
| campaign console | strixhalo `~/wt/315cwsr60-console.log` |
| eviction stream | strixhalo `~/wt/315cwsr60-evict/kfd_smi.tsv` |
| run 7 (the faulting baseline) | `~/wt/315evict7-console.log`, `~/wt/315evict7-evict/correlated-p3l.tsv` |
| harness | `docs/315-campaign/315-evict-trace.sh` (+ `tools/arm-pid.py`, `tools/lever-census.py`) |
| harness defects found+fixed | `docs/315-campaign/evidence/315-evict-trace-instrument.md` §6a–§6h |
| regression baselines | `docs/315-campaign/evidence/315-hrx-tests-baseline.md`, `315-engine-ci-baseline.md` |
| box-wide lock finding | `docs/315-campaign/evidence/NPU-LOCK-WEDGED-2026-10-12.md` |

## Honest bottom line

**task-8 is genuinely done** with real, correlated evidence. **task-9 is blocked on a named
mechanism that does not exist yet** — there is a strong signal (CWSR) and a falsified
explanation for it, and the confound that must be settled before the signal is even
attributable. The objective's criteria 1 and 2 are **not met**, and a clean 60-arm run would
satisfy criterion 3 while leaving 1 and 2 open.

---

# UPDATE (same day, later) — the size question is now PROVEN closed, and H1 is quantified

## Proven: the known gfx1151 CWSR size defect is FIXED here

AMD's own SRU cover letter (Mario Limonciello) names the defect and its two kernel halves
(`bump minimum vgpr size for gfx1151` in `kfd_queue.c`, `Export the cwsr_size and
ctl_stack_size` in `kfd_topology.c`) plus the matching ROCm commit. The arithmetic closes:

| vgpr | per-CU (vgpr+sgpr+lds+hwreg) | × 40 CUs | node requires 19,185,664 |
|---|---|---|---|
| `0x40000` unfixed | `0x55000` | 13,926,400 | **5,259,264 short** |
| `0x60000` fixed | `0x75000` | 19,169,280 | off by one SGPR unit |

Independent cross-check: a deployment report's real broken figure is **13,942,784**; my unfixed
total is 13,926,400 **+ 16,384** = exactly that. Three independent agreements ⇒ proof.

All five halves are present on our box. **So the documented defect cannot be our cause.** See
`315-cwsr-size-excluded-PROOF.md`. H3 and H4 are dead (both were sizing variants).

## H2 refined: mid-wave preemption, not spill correctness

The kernel's parameter doc: *"CWSR allows the GPU to preempt shader execution in the middle of
a compute wave... Setting 0 disables it."* So `cwsr_enable=0` removes mid-wave preemption
entirely. Refined mechanism: under pressure a wave is preempted mid-execution, its buffers are
evicted/migrated, and it resumes reading moved or recycled memory → Face A or B by content.
This does **not** depend on the (now-fixed) size defect, and it explains the GTT dependence, why
every prior lever returned null, the whole-batch layer-0 cascade, and the intermittency.

## H1 quantified **and a prediction recorded before the result**

Run 7 (`cwsr=1`, 3/11 faults) ran at **~32.5 h uptime**. The `cwsr=0` campaign started **692 s
after boot**. The campaign runs ~6 h → ends at ~7 h post-boot, so:

- **faults appearing LATE** in the campaign → supports H1 (uptime effect);
- **still 0 across 11 min → 7 h uptime** → weakens H1, leaves H2 standing.

Recorded in advance so it cannot be read post hoc.

## Current running state

`~/wt/315cwsr60-console.log`, tag `315cwsr60`, 48 pairs (~66 arms), `cwsr=0`, ~5.5 min/arm.
**Next actions after it completes, in order:** (1) read its rate; (2) **H1 control** — fresh
boot, `cwsr=1`, 20 arms (`docs/315-campaign/315-cwsr-control.sh`); (3) only then H2
instrumentation.
