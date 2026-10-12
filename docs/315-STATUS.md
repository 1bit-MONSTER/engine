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

---

# UPDATE 2 (late same day) — H1 FALSIFIED; the CWSR signal stands on its own

**Read `315-H1-FALSIFIED.md` first.** My recorded "fresh-boot confound" was an arithmetic error
of mine: I derived run 7's uptime (32.5 h) from a reading taken *before* the clean reboot at
14:03:53 that preceded run 7. Corrected from the SMI stream's `boot_ns`, two independent anchors
agree within 4 s that run 7 booted at **14:03:5x**.

| run | config | box age during its arms | result |
|---|---|---|---|
| **run 7** | `cwsr_enable=1` | **~6–40 min after a fresh boot** | **3 faults / 11 arms** |
| `315cwsr60` | `cwsr_enable=0` | ~53 min – 7 h after a fresh boot | 0 faults / 9 arms so far |

Both were fresh boots, same day, same hardware, same harness. **So a fresh boot does not suppress
the fault — run 7 was freshly booted and faulted anyway.** The strongest alternative explanation
for the CWSR signal is gone.

## Revised hypothesis ranking

| | | state |
|---|---|---|
| ~~H1~~ | fresh-boot effect | **FALSIFIED** — run 7 was fresh-booted and faulted 3/11 |
| **H2** | **mid-wave preemption under memory pressure** | **the leading and only surviving candidate** |
| ~~H3~~ | partial kernel-side fix | dead — a sizing variant |
| ~~H4~~ | firmware save/restore list mis-sized | dead — a sizing variant |

The mechanism as it now stands, and it is falsifiable:

> Under memory pressure a compute wave is preempted mid-execution (CWSR); the buffers that wave
> is reading are evicted/migrated by TTM/amdgpu; on resume it reads relocated or recycled memory,
> and the damaged values surface as Face A or Face B depending on what those bytes now hold.
> Disabling CWSR removes mid-wave preemption and closes the window.

**Discriminating experiment:** `amdgpu.cwsr_enable` 1 vs 0 — same harness, same gate, same day.
**Alternatives ruled out:** every prior lever (coherence, ordering, AQL barriers, `-fa off`,
transient reuse, PM4, MoE routing), the area-size explanation (proved fixed by AMD's own patch set
with exact arithmetic), and now the fresh-boot confound.

## What is still missing for criterion 1

**n.** 9 clean arms gives a rule-of-three upper bound of 33%, which does **not** exclude the
26–27% baseline. Removing H1 removes an *explanation*; it does not substitute for the ≥60 arms the
objective requires. At 60 clean arms the bound is 5% and the criterion's power statement holds.

## Unchanged

Criterion 2 (a forward-only fix) still has nothing to repair: the documented gfx1151 CWSR defect is
**proved fixed here**, and H2's remedy would be a driver/firmware change, not an engine change. The
best engine-side option remains a configuration guard — a **workaround**, not a repair.

Side quests completed while the campaign ran: a rate/CI/power tool (`315-rate.py`), the per-arm
eviction comparison tool (`315-evict-per-arm.py`, the H2 test), PID-capture wiring into the H1
control, the control's guard verified by dry-run, campaign QA (9/9 arms valid and in regime), and
three failed attempts at a retroactive H1 test — each of which is recorded as a negative with its
reason.

---

# UPDATE 3 — I under-sized the running campaign against the pre-registration's own target

**This prevents a false claim, so it goes at the top of the record.**

## The discrepancy

| source | says |
|---|---|
| the objective (user-provided) | "≥60 arms, every arm inside the declared regime gate, zero faults across both faces" |
| `docs/315-preregistration.md` §6 | "Target: **≥60 in-regime clean arms in the fix cell**, with paired control arms running in the same session… Hard cap: **80 scheduled pairs** (160 arms)" |
| `docs/315-preregistration.md` §7 | "The campaign passes only if, **in the fix cell**, ≥60 in-regime arms are clean…" |

The pre-registration counts **fix-cell arms**, not total arms. And the driver emits exactly **one
fix-cell (`l`) arm per pair** — measured, not assumed: the 8-pair run produced 8 `l` arms and 3
`c` arms (11 total).

So the running campaign, launched with **48 pairs**, will produce **~48 fix-cell arms** — **twelve
short of the pre-registered target**, even though its *total* arm count (~64) would satisfy the
objective's looser wording.

## What this means

1. **The running campaign cannot, by itself, satisfy §7.** It can establish the rate (and 9 clean
   arms so far is consistent with zero), but it must not be reported as meeting the ≥60 fix-cell
   bar.
2. **A follow-on is required.** Either:
   - a further campaign of **≥60 pairs** (the fix-cell count is what must reach 60), i.e. **~13
     additional pairs** beyond the 48 already scheduled would *not* suffice unless pooled; the
     clean route is a fresh run at ≥60 pairs (~6.5 h); or
   - pooling the `cwsr=0` arms across runs (`315cwsr0` gave 11, the campaign ~48, total ~59) —
     but pooling is **not provided for** in the pre-registration, which says "no arm pruning, no
     criterion changes" (§8) and frames the target as a single campaign with its paired controls.
     Pooling would need to be declared and justified, not assumed.
3. **The honest reading is that §7 needs a fresh ≥60-pair run**, and the current campaign is a
   useful but insufficient down-payment on it.

## How the mistake happened, stated plainly

I read the objective's "≥60 arms" and sized the run for ~64 **total** arms, never checking the
pre-registration's own definition of the target, which counts only the fix cell. I had read §6
earlier in the session (I quoted the "≥60 in-regime arms" bar in the task-8 work) and still sized
by total arms. The check takes one `grep` and I did not run it until now.

Recorded so the next run is sized from §6 — **60 fix-cell arms means ≥60 pairs** — rather than from
the objective's looser phrasing.

---

# UPDATE 4 — campaign of record relaunched at compliant size (70 pairs)

The 48-pair run was stopped at 10 scored arms (all clean) once I realised it could not reach the
pre-registration's ≥60 **fix-cell** arms. It is recorded as a **mis-sized pilot, not the campaign
of record** — no result was discarded and none was inconvenient (10/10 clean); the run simply could
not meet §6 as sized.

## The campaign of record

```
tag    : 315cwsr70
pairs  : 70  ->  ~70 fix-cell ('l') arms   (>=60 required by section 6; within the 80-pair cap)
config : amdgpu.cwsr_enable=0   (unchanged since the treatment reboot)
gates  : box lock -> npu lock -> GPU quiet -> watcher   (all four passed 16:54:11)
driver : with the PID capture installed, so its arms carry $O/server.pid and its data can
         support the H2 faulting-vs-clean eviction comparison
```

Expected duration ~7–8 h at the observed ~5 min/arm.

## Why relaunching rather than extending

Running 48 pairs and then another 60+ would cost ~11.9 h to reach one compliant campaign.
Relaunching once at 70 pairs costs ~7.8 h for the same thing. The stopped run's 10 clean arms are
not counted toward the target (pooling across runs is not provided for in the pre-registration),
so nothing of value was gained by letting it continue.

## Box-time budget

| | arms | ~GPU hours |
|---|---|---|
| earlier sessions (10-09…10-11) | 260 | — (predate this goal turn) |
| this goal turn, before the relaunch | ~24 | ~2 h |
| this campaign of record | ~93 | ~7.8 h |
| **total for this turn** | | **~10 h of the stated ≤12 h** |

Tight but inside. If the campaign needs a follow-on, the budget will be the binding constraint
before anything else.

## What to check when it lands

1. `tools/315-rate.py <console>` — the rate, exact CI and the power statement, applying §5/§6's
   counting rules from the driver's own lines;
2. **fix-cell count specifically** — must be ≥60 clean, not merely ≥60 total;
3. `tools/315-evict-per-arm.py` — the H2 comparison, now possible because the PIDs are captured;
4. a fresh-boot `cwsr=1` control if the rate holds, since H1's falsification was inferred from
   `boot_ns` rather than from a purpose-built control.

---

# UPDATE 5 — side quests while the campaign runs (all committed and pushed)

Run while waiting on `315cwsr70`, per "unblock by doing side quests". Five of the six are
correctness fixes to the **analysis and control** tooling, i.e. to the parts that decide whether the
campaign's result is believed — not to the frozen experiment itself.

**1. The console line cannot certify zero faults on its own — fixed.** (commit `43caadc`) Criterion 3
requires zero faults on **both faces**, but the driver's per-arm console line has no `http=` field,
and `detector.json` carries `http_failures` and `transport_failures` that the console cannot see. A
Face-B fault surfacing as HTTP 500 would therefore leave **no trace in the console** and a
console-only count would report that arm **clean** — a false negative on the one criterion the
campaign exists to certify. `315-rate.py` now reads every arm's `detector.json` and treats the
**detector as authoritative**: an arm the console passes but the detector fails is counted a **fault**
(`MISSED BY CONSOLE`), and an arm with no usable detector is **UNVERIFIED**, never silently clean.
Proven on a fixture where it changes the answer (`0/3 = 0%` without it → `1/2 = 50%` with it, the
difference being a real HTTP-500 fault). Re-checked on run 7: still 3/11, no missed arms, so it is a
guard and not a revision.

**2. The endgame runbook.** (`f3ac0a5`) `docs/315-ENDOGAME.md` — exact commands, the decision table
for the control, and the standing constraints plus seven traps already paid for. Written because the
session that started the campaign will not outlast it.

**3. The H1 control was hardened and dry-run safe.** (`ad0a494`) Sized 48 → 20 pairs (at ~26%, 20
gives ~5.2 expected faults and `P(0)=0.002`; 48 would cost ~3 h for no added power). It now
**verifies** the PID-capturing driver actually landed instead of trusting the copy — a run without the
capture silently produces H2-unusable data, and that is invisible until the analysis, after the GPU
hours are spent. Dry-run with `sudo`/`pgrep` neutered: drop-in PRESENT, `cwsr_enable` 0, campaign
alive, uptime unchanged. Its abort guards fire before any `sudo`.

**4. Pace measured, budget verdict: both stages fit.** (`80032c8`) From arm `server.pid` mtimes:
intervals **4:25, 4:26, 4:26, 4:27, 4:25** — a 2-second spread, so **4.43 min/arm** is not an estimate.
93 arms ≈ 6.87 h, finishing ~23:45. With ~2 h used before and ~2.2 h for the 20-pair control, the
total is **~11.1 h of the 12 h ceiling**. The decisive control does fit; margin is ~0.9 h.

**5. The H2 test was confounded — now it is a causal test.** (`76b7bdb`) I checked my own test and
found it could not distinguish *more evictions caused the fault* from *the fault caused more
evictions*: a faulting arm crashes, may be retried, and outlives its arm, so its window can simply be
longer. But **the fault is timestamped** (`0.07.279.709 E ... HRX returned NaN logits`, in
minutes.seconds.milliseconds.microseconds — confirmed against that arm's `req=1`, not assumed). So
the tool now also counts evictions **up to the fault instant**, which the fault cannot have caused.
Both faces are matched (Face B via `unparsed Content-only`), validated as 3/3 faults timed and **0 of
8 clean arms falsely flagged** — a parser that found faults in clean arms would fabricate the effect.

**6. Two of run 7's three faults were Face A *followed by a full server crash*** — the same commit.
p7c and p7l logged the NaN line and died (logs end in a gdb backtrace), which is why their detectors
recorded `RemoteDisconnected`: **the transport failure is downstream of the NaN, not a separate mode.**
The console's `FaceA` label is right. Run 7's faults are two NaN crashes (at 7.3 s and 57.8 s into
their arms) and one Face-B parse failure (81.2 s, `req=20`).

## Where that leaves the criteria

Unchanged: **1 open** (control), **2 open** (no repair landed), **3 in progress**, **4/5 not started**.
But criterion 3 is now much harder to pass *falsely* — the detector reconciliation and the fix-cell
count both stand between a clean-looking console and a claimed pass. That is the point.
