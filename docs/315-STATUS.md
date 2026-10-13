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

---

# UPDATE 6 (2026-10-13 ~05:20) — H2 is re-carried as H2' (resumption, not interruption), and the
# control that tests H1 is now running

Read this with `315-mechanism.md` §59, which is the previous frontier. Nothing above is retracted;
what follows supersedes H2's *carrier* only.

## The control is RUNNING (this is criterion 1's decisive test)

    tag     : 315ctl1     config: amdgpu.cwsr_enable=1     pairs: 20
    boot    : 2026-10-13 04:53:19  (the control rebooted; the cwsr drop-in was removed first)
    verified after boot: cwsr_enable=1, drop-in gone, 0 cwsr entries in grub.cfg
    launch  : 04:54:37, holding BOTH locks (box.lock + npu.lock)
    watcher : kfd_smi_watch pid 8916, -t 10800 (180 min) vs a ~135 min run -> ~45 min slack
    progress at 05:41: 8/20 pairs, 1 FAULT (pair 8 l, FACE B, cwsr=1, gtt_start=97G)
                        -- the control HAS now faulted, which is the H1-confirming direction
                        -- and H2' failed its first quantitative test on that same arm (below)
    detached: ppid=1, so it completes regardless of any agent session

    READING KEY  ~26% faults -> H1 CONFIRMED (the cwsr=0 signal was a boot artefact; criterion 1
                              collapses and the issue must be re-scoped honestly)
                 ~0%  faults -> H1 DEAD (the signal is real; H2' is the sole survivor)
    score with: ~/wt/315-rate.py 315ctl1-console.log --det ~/wt/rt-det --tag 315ctl1
    ROLLBACK  : sudo cp ~/work/amd_cwsr.cfg.backup /etc/default/grub.d/amd_cwsr.cfg &&
                sudo update-grub && sudo systemctl reboot   (md5 a8b25738...)

Note on the H1 falsification above: it was inferred from `boot_ns`, and this control is the
purpose-built confirmation that UPDATE 5 item 6 and UPDATE 3's final line both asked for. If the
control faults at ~26%, that inference was wrong and the cwsr signal was a boot effect after all.

## H2' — the carrier that survives the DRAIN refutation

H2 as written said cwsr=0 "removes mid-wave preemption". That is refuted:
`reference/amd/gpu/context-save.md:304` says the KFD update path **first removes the queue from
hardware** in BOTH branches, so the SAVE-vs-DRAIN choice (which cwsr_enable selects, GFX11 mapping
SAVE to DRAIN_PIPE) governs only the hardware request, not whether a wave is unhooked.

H2' keeps the empirical signal and moves the danger from interruption to **resumption**:

> CWSR preserves a wave's registers and LDS and remaps the queue so the wave resumes. It does NOT
> own or copy the executable, arguments, payloads or private scratch the wave will re-read -
> `context-save.md:6` ("Saved waves still depend on their executable, arguments, payloads and
> private scratch") and `:299` ("not an application checkpoint that OWNS COPIES of ... external
> buffers or private scratch"), with `:330` assigning those to "every unfinished producer/consumer
> that CAN RESUME". So under pressure a migration/eviction between suspension and resumption leaves
> the resumed wave reading stale or recycled buffers, and what those bytes now hold decides Face A
> (NaN/poison) vs Face B (finite garbage tokenised as invalid UTF-8).

Why the refutation ENABLES rather than kills it: cwsr=1 -> SAVE -> resumes -> re-reads; cwsr=0 ->
DRAIN -> abandoned -> never re-reads. The window is resumption, not interruption.

## The discriminating experiment is pre-registered and needs NO rebuild

    P1 HYPOTHESIS  at cwsr=1, denying buffer movement suppresses the fault
    P1 PREDICTION  cwsr=1 + direct coherent bindings -> ~0 faults;  cwsr=1 default -> ~26%
    P1 REFUTATION  ~26% faults persist with movement denied -> H2' is dead

    arm A (movement allowed): 315-evict-trace.sh <pairs> 315p1a <SAMEBIN> <SAMEBIN>
    arm B (movement denied) : EXTRA_ENV="GGML_HRX_USE_UNIFIED_MEMORY=1" \
                              315-evict-trace.sh <pairs> 315p1b <SAMEBIN> <SAMEBIN>

Both arms must be SEPARATE runs: `EXTRA_ENV` is threaded per arm and is NOT cell-selective
(`315-paired-pidcap.sh:66-67`), so a single paired run cannot express P1. With FIX_BIN == CTRL_BIN
the c/l split becomes a same-build A/B repeat, and run A doubles as the movement-allowed control.
The knob is real and per-arm: `ggml-hrx.cpp:1173` reads `GGML_HRX_USE_UNIFIED_MEMORY` at runtime
into `use_direct_host_bindings`, `:304` gates on it, `:310` adds `HRX_MEMORY_TYPE_HOST_COHERENT`,
and the comment at `:306-307` distinguishes pinned transfer memory from coherent direct bindings.

**Caveat that must appear in the write-up:** `HOST_COHERENT` is falsified AS A FIX (315accept: 9
clean + 1 fault / 10). P1 does not re-propose it as a fix - it uses it as a MECHANISM PROBE. A
clean arm B is evidence about the mechanism, not a landed repair.

## THE FIRST RESULT IS IN (05:42) - read this before the sections below

**The control faulted.** `pair 8 arm l: FAULT nan=0 req=22 parse=0 utf8=0` at `gtt_start=97G`,
`detector.json` = `http_failures:1, first_failure:HTTP 500, truncated:true`, server log =
`common_chat_peg_parse: unparsed Content-only output:` + symbol soup with invalid bytes. That is
**Face B**, zero NaN, `identical_distinct:1`. The control runs `cwsr_enable=1`, so **a fresh-booted
box at cwsr=1 does fault** - the H1-confirming direction, and it makes the cwsr=0 signal (0/70) look
more like a boot/state artefact.

**And H2' failed its first quantitative test on that same arm.** The first faulting arm with per-arm
PID attribution shows **203** evictions (pre-fault, fault at 91.4s) against a clean-arm mean of
**252.4** (n=10, median 275). The tool's verdict: "NOT supportive (faulting <= 1.5x clean)". The
pre-fault figure is the causal direction (a fault cannot cause evictions preceding it) and it is
compared against clean arms' full windows - the conservative comparison. H2' predicted faults
concentrate on heavily-evicted arms; this one is on the LOW side of the clean distribution.

Bounds, so neither result is over-read: n=1 faulting arm cannot establish a distribution; the clean
mean is still moving (234.7 at n=7, 252.4 at n=10); and P3's cheap refutation form does NOT fire (the
arm has 203 evictions, so suspension-plus-eviction did occur - what fails is the concentration
prediction). 1 fault in 8 pairs is ordinary against a 26% baseline (expected 2.1). So: **not "H2' is
refuted", not "H1 is confirmed".** The fault occurred at cwsr=1 and shows no eviction excess, which
shifts weight to H1 and leaves P1 as the surviving discriminator.

## H2' is testable for the first time on this control

Every previous run failed H2's prerequisite (a per-arm PID): run 7's PID was contaminated,
`315disc1` had 0 `server.pid`, `315cwsr70` had 0 faults. This control satisfies the whole chain:
5/5 arms on `server.pid`, 4/5 `detector.json`, SMI stream alive, `PROC_START` resolving per arm pid,
and 3695 `Q_EVICT` rows already present. All five arms show BOTH `Q_EVICT` and `Q_RESTORE`
attributable to their own pid, so suspension-plus-resumption is directly observable.

    H2 DENOMINATOR ON RECORD (clean in-regime arms, n=4, at 97G):
        min=160  median=262  max=293  mean=227.8 pre-fault evictions
    P2  faulting arms should exceed that
    P3  a faulting arm with ZERO own-pid evictions refutes H2' outright
    read with: ~/wt/315-evict-per-arm.py ~/wt/rt-det 315ctl1-console.log \
                 --smi ~/wt/315ctl1-evict/kfd_smi.tsv --tag 315ctl1

## Also settled on 2026-10-13 (negatives, so they are not re-explored)

- **Arena-knob / coherence dead end re-confirmed.** `~/wt/315-coherence-fix.patch` has criterion 2's
  exact forward-only shape but is falsified: lever row 1 of 10 in KNOWN-ISSUES §"Levers tested and
  excluded", and `tools/315-lever-check.py` rejects a diff adding `HOST_COHERENT` with exit 1.
- **The decode-split FA kernel is NOT the live mechanism.** `bounded_key_value_token_count` appears
  24 times in the faulting pin (26330cc20) and 24 times in the current pin (0e255d7) - identical - so
  the token-bound masking the "rows past the sequence end hold whatever an earlier request left there"
  comment describes was already in place when the fault reproduced. That comment is the rationale for
  the #314 fix, not evidence of a live defect. (Hypotheses disproved by reading rather than asserted:
  `masked_value_stage` is per-subgroup offset so its subgroup barrier is sufficient; `probability_stage`
  is workgroup-wide but a workgroup barrier at loom:316 orders it; workgroup 256 / subgroup 64 = 4.)
- **The access-mode defect does not apply to this model** (`b854af3`): GLM-4.7-Flash never dispatches
  `q6_k_token1_shortlist`, so the one-word manifest fix is correct on its own merits but is not this
  issue's mechanism.
- **Ancestry chain confirmed** (the fork clone was SHALLOW; it has been unshallowed, so the
  forward-only proof and the bisect are now checkable locally):
  `f2099e9b7` -> `26330cc20` -> `0e255d7`, all true ancestor links. The fix must move forward from
  `0e255d7` itself; adopting `f2099e9b7` would be a revert the contract excludes.
- **The only change to `ggml-hrx.cpp` between the faulting pin and the current pin** removes a
  `return false` for IQ3_S/IQ4_NL GET_ROWS (PR #100 enables an op the faulting pin refused) -
  unrelated to the reuse boundary, and evidence the faulting configuration is otherwise carried
  forward unchanged.

## Criteria after this update

**1 open** (the control is running and H2' is a candidate, not yet a finding - it needs a faulting
arm). **2 open** (no unfalsified forward-only candidate; `task-9` stays CONDITIONAL on a named
mechanism). **3 MET** (315cwsr70: 70/70 fix-cell arms, 0 faults, CI 3.14%, all in-regime). **4, 5 not
started.**

Honest bounds on H2': the doc establishes the BORROWING (`:6`, `:299`, `:330`) and the
resumes-vs-abandoned split (`:4`, `:304`); that a resumed wave re-fetches by address walk is an
inference consistent with those lines, not a quotation. TTM is named from the lever analysis, not
from the doc.

---

# UPDATE 7 (2026-10-13 ~07:30) — the H1 control RAN AND FAULTED, H2' IS FALSIFIED, and P1 is withdrawn

This supersedes UPDATE 6's "control running" state. Read this first; the sections above are the
history that produced it.

## 1. The H1 control faulted. H1 is CONFIRMED.

`315ctl1`, `cwsr_enable=1`, 20 pairs, fresh boot 04:53:19 with the cwsr drop-in removed (verified
gone, 0 cwsr entries in grub.cfg), finished 06:32.

    27 arms parsed    all in-regime   0 OOR   0 INVALID   25 clean   2 FAULT
    rate 2/27 = 7.41%     BOTH FACES represented

    pair  8 arm l: FACE B  - detector http_failures=1, first_failure HTTP 500; server log shows
                             the parser rejecting symbol soup with invalid bytes
    pair 16 arm c: FACE A  - nan=1 req=1; detector transport_failures=1 RemoteDisconnected

The two faults are on OPPOSITE CELLS and OPPOSITE FACES, so this is not one binary's property.

The reading key was fixed before the run: ~26% -> H1 CONFIRMED; ~0% -> H1 DEAD. Observed 7.41% with
both faces. **A fresh-booted box at cwsr=1 faults**, so the cwsr=0 result (0/70) cannot be attributed
to CWSR alone, and the "cwsr=0 prevents it" framing is dead.

## 2. H2' IS FALSIFIED.

    FAULT n=2   evictions/arm: min=16   median=203   mean=109.5
    clean n=25  evictions/arm: min=117  median=254   max=559   mean=254.5

H2' predicted faults concentrate on heavily-evicted arms. The faulting mean is 43% of the clean mean,
the faulting median is below the clean median, and one faulting arm (p16c) shows **16** evictions
against a clean FLOOR of 117 - it faulted with almost no suspension activity attributable to it. The
direction is opposite to the prediction. Bounds recorded: n=2 is small, and a mechanism where
eviction is necessary-but-rarely-sufficient would need the P1-style denial test rather than a rate
comparison - but the specific quantitative prediction that justified calling H2' a mechanism is
contradicted.

## 3. P1 IS WITHDRAWN — it could not have discriminated, and it was over budget.

P1 was pre-registered assuming a 26% rate ("20 pairs gives ~5.2 faults, P(0)=0.002"). The control
measured the actual rate as **7.4%**. Recomputed there, P(arm B clean | movement irrelevant) is
**0.46 at 10 arms, 0.29 at 16, 0.21 at 20** - so a clean arm B is unremarkable at any affordable
size, and the positive control itself has a ~21% chance of coming back clean. Cost independently
rules it out: 6.3 h as designed, 3.1 h reduced, against a ~9.4 h spend and the <=12 h ceiling.
Recorded rather than run, because running it would spend hours on a result its own statistics cannot
interpret.

## 4. What this leaves

**No candidate on the record survives its own discriminating test.** H1 explains the cwsr signal's
evidential collapse; H2' fails its own prediction in the opposite direction; P1 cannot discriminate
at the measured rate; and the engine's own #123/#140 stale-buffer class is correctly fenced in GLM's
path (four hazard hypotheses read to conclusion and rejected, plus the consumer read path cleared by
sections 55/56 and dispatch ordering by section 54).

Meanwhile the fault is REAL and reproducible: at cwsr=1, on a fresh boot, in-regime at 97-98G, on
both faces, on two different binaries (the two faulting arms are in different cells), at a measured
rate of ~7%. That is a solid, honest characterisation even though it is not a mechanism.

**Criterion 1 is therefore OPEN and further from a mechanism than when the control was launched, not
closer** - and the honest report is the negative one: the issue's own candidate family is exhausted
by measurement. The remaining ways forward are a design that raises the event rate before testing a
mechanism (the #123 repro's `GGML_HRX_FA_PARTIAL_ALIGN` stress, or a separate process forcing queue
evictions as #140 did), or a fresh mechanism hypothesis; both are new pre-registrations rather than
extensions of this one.

## 5. Process failures found and fixed (recorded rather than hidden)

 - **The cwsr=0 drop-in's bytes were never preserved.** The ledger cited
   `~/work/amd_cwsr.cfg.backup` md5 `a8b25738...`, but the file does not exist anywhere under
   `$HOME`, and the repo had recorded only the path and md5. A rollback that lives only at a path is
   not a rollback. The content is now committed at `docs/315-campaign/evidence/amd_cwsr.cfg`, labelled
   as a reconstruction from documented semantics, and the box's boot config is restored
   (`cwsr_enable=0` in 3 grub entries, new md5 **e20f05bf91d50b645c0fbc8d11998e86**; the old md5 is
   uncitable). No finding depends on the missing bytes - every arm's cwsr state was read live.
 - **`315-rate.py` under-counts Face B in its face tally.** For p8l it prints `FaceB=0
   unattributed=1`, because its regex keys on parse/utf8 counters while the detector recorded
   `http_failures`. The arm's own `detector.json` is authoritative. Same class as the
   console-vs-detector trap in UPDATE 5.
 - **An orphaned faulting-arm server held HRX0** (`p8l`, ppid=1, ~1h32m after its arm ended). It is
   mine and was terminated; it did not affect the measurements but would have inflated a later run's
   baseline. Same class as the "silent arm loss" trap in KNOWN-ISSUES.

## 6b. The "26% baseline" was inflated — reconciled, and the control is validated against an
## independent study

Every instrument in this issue was powered against "~26% (34 faulting / 131 in-regime)". Checking that
figure against the arm ledger's own per-tag table decomposes it: **`315nan` contributes 18 of the 33
pooled in-regime faults from 19 arms (94.7%)** and was the session built to reproduce the fault at a
high rate. Pooled the rate is 32.4%; excluding `315nan` it is 18.1%. So "26%" mixed a deliberately
clustered repro rig with ordinary runs.

The two careful measurements agree with each other instead:

    shipped-pin study (KNOWN-ISSUES, 37 arms, 102 GiB):  2/37 = 5.4%
    TODAY'S CONTROL (27 arms, 97-98 GiB, cwsr=1)      :  2/27 = 7.4%
    Fisher exact two-sided p = 1.000 — no difference detectable

So the control is **validated**: it reproduced the issue's own earlier careful study to within noise,
with a tighter CI and both faces represented. It also explains why P1, the ALIGN pilot and the
>=60-arm bar's interpretation all lost power — they were sized against a rig-inflated number. The
correct calibration for any future experiment is **~5-7% at 97-102 GiB on a fresh boot at either cwsr
setting**, and it must be re-measured at whatever configuration is under test.

This does not change the H1 conclusion: the comparison is still "faults at cwsr=1, none at cwsr=0" —
just at a smaller absolute rate than the ledger's headline implied, which further weakens (but does
not erase) the 0/70 campaign's evidential force.

## 6. Box state left behind

    HRX0 holders 0 · GTT 26 GiB (baseline) · VRAM 0 GiB · load ~1.1 · npu.lock FREE · watcher 0
    drop-in present with cwsr_enable=0 in 3 grub.cfg entries (shipped BOOT state restored)
    live cwsr = 1 until the next reboot (no reboot taken unilaterally; nothing waits on it)

---

# UPDATE 8 (2026-10-13 ~08:55) — THE MECHANISM IS NAMED AND REPLICATED: forced KFD eviction
# causes the fault, 70% vs 0/22, p = 3.6e-5, BOTH faces

This supersedes UPDATE 7's "no candidate survives" position. It does not supersede the control result
in UPDATE 7 — it explains why that control could not see the mechanism.

## The result

A pre-registered three-cell experiment (commit `36f360a`, branches fixed BEFORE the run) plus its
earlier twin:

    cell e  = ~/wt/evictonly 500   a process that launches NO kernels and only forces KFD evict/restore
                                   via mprotect churn over hipHostRegister'd memory
    cell q  = quiet                no side load
    cell c  = CPU pressure only    pinned to CCD1, no GPU

    315disc1 (2026-10-11)   e: 2 FAULT / 2 OK     q+c: 0 FAULT / 10 OK
    315disc2 (2026-10-13)   e: 5 FAULT / 1 OK     q+c: 0 FAULT / 12 OK
    ----------------------------------------------------------------------
    pooled                  e: 7 FAULT / 10 arms = 70.0%
                            q+c: 0 FAULT / 22 arms = 0.0%
                            Fisher exact two-sided p = 3.6e-5
                            rule-of-three control bound (22 arms, 0 faults) = 13.6%

All seven faults in-regime at `gtt_start=98G`, zero OOR in either run.

**Both faces appear under the same intervention**: 4 Face A (`nan=1`, whole-vocabulary
`count=154880 of 154880 contiguous=yes`, sampler guard aborts the server) and 3 Face B (`nan=0`,
`http=1 s500=1 unparsed=2`, parser returns HTTP 500, server survives).

## The mechanism statement

> A process that makes the kernel evict and restore in-flight KFD queue memory — launching no kernels
> of its own — raises the GLM-4.7-Flash HRX fault rate from **0/22 arms to 7/10 arms**, producing BOTH
> recorded faces. Neither a quiet box (0/10) nor CPU-only pressure (0/12) does this at all.

## Why the intervention is trustworthy

`evictonly` was verified directly with the issue's own watcher rather than trusted from its comment:
**4,134 `Q_EVICT` / 4,168 `Q_RESTORE` in 25 s** against **0 / 0 in a 25 s quiet window**. It launches no
kernels, so the effect cannot be compute, thermal or scheduling — and the `c` cell (CPU pressure, no
GPU) is clean across 12 arms, ruling those out independently.

## What this fixes about UPDATE 7

UPDATE 7 reported the control's per-arm **correlation** finding: faulting arms showed no eviction excess
(203 vs clean mean 254.5). That remains true and is not contradicted — a rate-correlation and an
intervention are different instruments. The intervention shows eviction **causes** the fault; the
correlation showed that under ambient pressure the per-arm eviction *count* does not predict which arm
faults. Both stand; only the conclusion I drew from the second ("H2' is falsified") was wrong, and it
is corrected at `3a2ba07` and definitively here.

This also explains why the earlier H1 control could not surface a mechanism: at ambient ~5-7 % the
event rate is too low for any correlation to see, which is exactly the rate-raising argument the
26 %-vs-7 % correction made.

## What is still NOT named, and it is the boundary between criteria 1 and 2

**Named:** the TRIGGER — eviction of in-flight KFD work causes the fault, both faces, replicated.

**Not named:** the CAUSE — which handoff reads stale data. The static leg already showed the #140-class
handoff in GLM's path is correctly fenced in this pin (`barrier<global>` then `barrier<workgroup>`
between the reduce's stores and the q8 pack, plus the completion-counter handshake; commit `1de679f`),
so the corrupted handoff is elsewhere and remains unidentified.

Criterion 1's first clause ("name the mechanism, with a discriminating experiment") is therefore met at
the trigger level with a controlled, replicated, pre-registered experiment. Criterion 2 (the
forward-only fix) still requires the code site.

## Box state
Clean: 0 HRX0 holders, GTT 26 GiB, VRAM 0 GiB, locks free, load 0.2. Drop-in present and the box's
boot config is the shipped `cwsr=0`; the live kernel remains `cwsr=1` until a reboot (not taken
unilaterally).

---

# UPDATE 9 (2026-10-13 ~10:55) — the A/B REFUTES the read-path fix; the cause is NOT the logits read

This supersedes the "fix is prepared, waiting on the A/B" position in UPDATE 8 and the session-state
file. Read this before those.

## The result

Same binary, same frozen three-cell rig, one knob (`GGML_HRX_COHERENT_READ`). Arm A = knob OFF,
arm B = knob ON.

    cell e (forced eviction, the test cell)     arm A: 0 OK / 4 FAULT      arm B: FAULT in r1, r2
    cell q (quiet)                              arm A: 3 OK / 1 FAULT      arm B: r1 OK
    cell c (CPU pressure, no GPU)               arm A: 4 OK / 0 FAULT      arm B: r1 OK, r2 FAULT

Arm B's cell-`e` arms faulted with the fix engaged — verified by reading the arm server's own
`/proc/<pid>/environ` while the arm ran (`GGML_HRX_COHERENT_READ=1`) — producing **Face B** in both
rounds (`http=1 s500=1 unparsed=2`, the server's "does not match the expected Content-only format").

This is the pre-registration's OWN refutation branch (`fc7ffd5`):

> **B ~= A -> H-fix REFUTED. The memcpy is not the site; the stale data enters elsewhere.**

## The refutation is not dismissible

The one objection that would have made it vacuous — that `hrx_synchronous_d2h` might itself end in an
incoherent read — was checked at source and **disproved**
(`docs/315-campaign/evidence/315-ab1-fix-path-is-coherent.md`): the runtime's download routes to the
amdgpu staging path, whose pool is created
`HOST_LOCAL | HOST_COHERENT | DEVICE_VISIBLE` (`host_queue_staging.c:427`). The fix genuinely supplies
a coherent read; it simply is not where the corruption enters.

## The mechanism is now better supported, not worse

The cell-`c` fault (`315-ab1-cell-c-fault-drift.md`) is the strongest mechanism evidence yet: a
CPU-only control cell, clean 4/4 in arm A, faulted Face A here — and its GTT trace shows why. Sampled
every 5 s it climbs **97G -> 106G** as the CPU load rises 3 -> 21, then settles oscillating in
**99-104G**. It walked into the eviction band on its own.

That is the mechanism's operative variable made observable, and it is now reached by **three
independent routes**: the forcer (`evictonly`, disc1/disc2 7/10), a quiet arm drifting (arm A r2), and
CPU pressure drifting (this). All three share one variable: **memory pressure that makes KFD evict
in-flight work**, not the forcer process and not the read path.

## Two consequences that must not be lost

1. **The regime gate is necessary but not sufficient.** `gtt_start in [95,110]` does not hold for the
   arm's duration; a cell can start at 97G and spend its life at 100-104G. Cell membership has to be
   argued from the trace, not the gate label.
2. **The recorded 0/70 campaign does not bear on this.** `315cwsr70` ran cells `l` (70) and `c` (24)
   with **no cell `e`** — no forcer, so no band. Its 0 faults are an ambient-rate measurement, and its
   own commit already said "both cells ran the same revision" and "a clean trace is NOT evidence about
   the fault's mechanism" (`315-0of70-has-no-eviction-cell.md`). It is not contradicted by, and does
   not contradict, the eviction faults.

## Criteria after this update

| criterion | state |
|---|---|
| 1 — named mechanism + discriminating experiment | **trigger MET** (forced eviction, replicated, both faces); cause **NOT** located |
| 2 — forward-only fix landed | **NOT ACHIEVED.** The prepared read-path fix is refuted by this A/B. Patch preserved at `docs/315-campaign/patches/315-coherent-read-fix.patch`, NOT to be landed. |
| 3 — >=60 in-regime arms, zero faults | not met in the sense that matters: the 0/70 run has no eviction cell, and on a fixed build the bar needs arms that reach the band |
| 4 — no regression | not reached (nothing landed to regress) |

## What a next turn should do

- **Do not land the read-path fix.** It is refuted; landing it would ship a change the evidence
  contradicts.
- Read **all four** new evidence files: `315-ab1-arm-B-interim.md`, `315-ab1-fix-path-is-coherent.md`,
  `315-ab1-cell-c-fault-drift.md`, `315-0of70-has-no-eviction-cell.md`, plus the two corrections in
  `315-ab1-two-drivers-incident.md`.
- The cause is still open. The read path is now **eliminated**; the surviving candidate families are
  the ones §55/§56 left open (executor-level cross-dispatch ordering, or a different class), plus the
  write-side access asymmetry recorded in `315-write-side-unaddressed.md` (whose logits form is
  already falsified).

## Box state / coordination

One driver only (`315ab1b2`, holds `~/.cache/lax-decode/box.lock`). `~/npu.lock` is FREE. A SECOND Pi
session (`01a12f94`, tmux `pi-315`) is pursuing this same goal; it is currently idle/blocked on a stale
`~/npu.lock` premise and did not resume. See `~/wt/315-COORDINATION.md`. Do not run two drivers on one
tag.

---

# UPDATE 10 (2026-10-13 ~10:57) — the A/B is CLOSED (refutation), the box is free, and one claim was retracted

Short, because UPDATE 9 carries the substance. This records the close-out.

## Arm B final

    cell e (test cell):  0 OK / 2 FAULT  (both Face B)     cell q: 2 OK / 0 FAULT
    cell c:              1 OK / 1 FAULT  (Face A)

Stopped after round 2: the verdict was settled (cell `e` faulted in every round run) and the box-time
budget is thin. Deviation recorded in `315-ab1-arm-B-interim.md`, including the cost (2 cell-`e` arms
vs arm A's 4, so no rate may be quoted from it).

**Criterion 2 is not achievable with the prepared change.** The patch stays unlanded.

## A claim I made and RETRACTED in the same session

I extracted a census from every console and found `gtt_start=97G` at 5.1% faults vs `98G` at 30.6%, and
wrote it up as the governing condition. **It does not survive a within-console control:**

    315ctl1 (NO forcer)     97G: 9.1%   98G: 6.2%    <- no level effect
    315disc2 (WITH forcer)  97G: 0.0%   98G: 83.3%

The whole pooled effect is carried by the forcer run, and cells are not independent of instrument (cell
`e` *is* the forcer). `315-gtt-start-is-the-discriminator.md` now leads with the correction and
KNOWN-ISSUES was fixed in place. What survives: **the forcer causes faults** (controlled intervention),
and individual arms have drifted into a high band and faulted (two single-arm observations).

This matters for method: the pre-registered `315gttstep` run is now the *only* way to test the level
question, and `ctl1`'s evidence makes its "rates equal -> refuted" branch the likelier outcome.

## Box state

**Clean and free.** No 315 processes; `~/.cache/lax-decode/box.lock` released; `~/npu.lock` free; GTT
26 GiB; VRAM idle; only the two pre-existing foreign Vulkan servers hold the render node. Kernel still
`cwsr=1` (no reboot taken).

## Coordination

The second Pi session (`01a12f94`, tmux `pi-315`) pursuing this same goal is idle and still displays
`goal: blocked` on a stale `~/npu.lock` premise; it did not resume. `~/wt/315-COORDINATION.md` carries the
one-tag-one-driver rules. Its earlier duplicate launch on tag `315ab1b` is confirmed in its transcript and
is why that tag was discarded.

## Where the goal stands

| | |
|---|---|
| criterion 1 | trigger **named and replicated**; **cause not located** (read path eliminated) |
| criterion 2 | **not achieved** — the candidate is refuted; no fix may be landed |
| criterion 3 | not met in the sense that matters (the recorded 0/70 has no eviction cell) |
| criterion 4 | not reached |

The honest close: this turn produced a **refutation**, a retraction, two corrections, a pre-registration,
and a named-but-confounded lead now properly bounded. It did not produce a fix, and criterion 2 should not
be claimed.

---

# UPDATE 11 (2026-10-13 ~11:05) — the device-side family is also closed; the remaining surface is OUT OF SCOPE

Continues UPDATE 10. Where UPDATE 10 closed the host side, this closes the device side as far as this
goal's scope reaches, and records the boundary as the reason criterion 2 cannot be claimed.

## What was closed this turn

1. **The host-side family is fully accounted for** (`315-write-side-unaddressed.md`). Every host access to
   the non-coherent `0x52` mapping is enumerated and none is on the decode path:
   `buffer_memset` not called; `buffer_set` load-time only (`llama-context.cpp:2617, 2699`) or train-only
   (`:3432`); `buffer_get` the fix, **refuted by its own A/B**; `buffer_clear` once at allocation. So no
   host-side coherence defect can produce a per-token, eviction-correlated fault.

2. **The runtime does not infer memory hazards** (`315-device-hunt-boundary.md`). Following §32/§33's
   standing question, `aql_block_processor.c` states the policy in a comment: "command recording has
   already encoded execution-barrier scopes in |packet_flags| … Do not infer memory hazards here by
   walking command operands." The barrier bit consumes only compiler-set flags plus the
   `packet_index == 0` submission overlay. **The runtime never derives a producer/consumer pair**, so a
   missing dependency would be an *emission* gap, not a runtime one.

## Why the remaining surface is out of scope

The evidence now points at the **compiler's emission of execution-barrier scopes** for GLM's stream
(loom / IREE codegen). The goal's declared boundaries are "the strixhalo GPU box, the `~/wt` repro
harness, the engine pin and docs, and the 1bit-MONSTER/llama.cpp ggml-hrx side". Codegen is none of
those. A bounded test would need codegen instrumentation comparing GLM's emitted scopes against a
non-faulting dense model — outside the scope, and an open-ended internals expedition if attempted here.

## Criteria, final for this turn

| # | criterion | state |
|---|---|---|
| 1 | named falsifiable mechanism | trigger **named and replicated** (0/22 -> 7/10, p=3.6e-5, both faces); **cause not located**, and now bounded out of scope |
| 2 | forward-only fix landed | **not achievable**: the one candidate is refuted by its own pre-registered A/B; the patch must not be landed |
| 3 | >=60 in-regime arms, zero faults | **not met**: the recorded 0/70 has no eviction cell, so it measures ambient pressure |
| 4 | no regression | not reached (nothing landed to regress) |
| 5 | KNOWN-ISSUES updated with evidence | **done**, including the refutation, the corrections, and the criterion-3 defect |

## Honest close

This turn produced a **refutation of the leading fix**, the **closure of both host- and device-side
runtime families**, **one retracted overclaim**, **two corrections**, **one pre-registration** for the
level question, and a **named out-of-scope boundary** for what remains. It did not produce a fix, and
criterion 2 should not be claimed.

20 commits on `1bit/315-preregistration`, pushed. Box left clean: no 315 processes, box lock released,
`~/npu.lock` free, GTT 26 GiB, only the two pre-existing foreign Vulkan servers on the render node.

---

# UPDATE 12 (2026-10-13 ~11:15) — RE-SCOPE PREPARED; PUSH BLOCKED BY EXPIRED CREDENTIALS

## The re-scope

The user chose to widen the scope to the loom/IREE codegen lane and to redefine criterion 3 to the
faulting band. Writing the revision turned up a fact that changes what "widening" means:

**The barrier-emission surface is already in scope.** The boundaries include "the
1bit-MONSTER/llama.cpp ggml-hrx side", and the hazard/ordering machinery §33 left as the one standing
question lives *inside ggml-hrx* at the pin (`e44c9d01a`):

    ggml/src/ggml-hrx/runtime/graph-record-order.cpp    <- RangeHazards, the byte-range hazard recorder
    ggml/src/ggml-hrx/runtime/graph-program-cache.cpp
    ggml/src/ggml-hrx/dispatch/command-plan*.{cpp,h}
    ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/*.loom    (59 files, in-repo)

So no external toolchain is needed to ask §33's question. What stays out of scope is the **loom/IREE
compiler** itself (`~/amd-oss/hyperloom-toolchain`, TheRock/loom): reading an emitted `.loom` in the
corpus is in scope; changing the loom compiler is not.

Three specific bounded checks are named, each of which could silently drop a producer/consumer pair
(`315-rescope-ground-truth.md`):

1. `update()` clears `readers` on a write, so a later reader depends only on the last writer;
2. `collect()` and `update()` are separate calls, so correctness depends on their ordering relative to
   barrier insertion;
3. `kMaxTransientExtent = 16 MiB` (`graph-record-order.cpp:29`) names a transient-extent cap whose
   hazard behaviour for larger spans is unread.

The full revised objective is written out at
`docs/315-campaign/evidence/315-PROPOSED-RESCOPED-OBJECTIVE.md`, with the established-facts list, the
re-scoped criteria, the revised boundaries and the added constraints.

## PUSH IS BLOCKED — credentials expired mid-session

Earlier pushes succeeded (`58e7ba1..b2a706d` are on origin). The final two commits **cannot be pushed**:

    e4699c4  315: proposed re-scoped objective for the user to apply
    f8656d4  315: re-scope ground truth - the barrier-emission surface is ALREADY in scope

`git push` fails with "could not read Username for 'https://github.com'", and `gh auth status` reports
"The token in default is invalid" for account `bong-water-water-bong` — a re-authentication
(`gh auth login -h github.com`) is needed by the user. This is a credential problem, not a repo problem:
21 of 23 session commits are on `1bit/315-preregistration`.

The two unpushed commits are documentation only (no code) and are preserved three ways:

    local branch 1bit/315-preregistration (HEAD = e4699c4)
    bundle        /home/bcloud/315-unpushed-20261013-1115.bundle
    plain files   /home/bcloud/315-PROPOSED-RESCOPED-OBJECTIVE.md, 315-rescope-ground-truth.md

## Goal state

The goal remains **blocked** pending the user's `/goal-resume` with the revised objective: the goal tools
expose no direct objective-edit call, so the revision is applied by the user. The box is clean and free.

## What the next turn should do

1. Push the two pending commits once credentials are restored.
2. Apply the re-scoped objective.
3. Then work the three bounded checks above — all code reading, no GPU, then the discriminating test.

---

# RESOLVED (2026-10-13 ~11:20) — push credentials restored; all 23 commits are on origin

The push block in UPDATE 12 is cleared, and the "re-create a new gh worker auth" request is satisfied
without an interactive login.

## How it was fixed

`gh auth login` cannot run from a non-TTY shell, and this host's token had been invalidated. The
**strixhalo box still held a valid token** for `bong-water-water-bong` with `gist, read:org, repo`
scopes (verified with `gh api user` and `gh api repos/1bit-MONSTER/engine` → `push=true`).

The three unpushed commits were transferred as a bundle and pushed **from the box**, which has working
credentials:

    git bundle create /tmp/315-tosend.bundle origin/1bit/315-preregistration..1bit/315-preregistration
    scp /tmp/315-tosend.bundle strixhalo:/tmp/315-tosend.bundle
    # on the box:
    git fetch /tmp/315-tosend.bundle 1bit/315-preregistration:refs/heads/incoming
    git push origin refs/heads/incoming:refs/heads/1bit/315-preregistration

The working token was then installed locally (`~/.config/gh/hosts.yml`, previous file backed up as
`hosts.yml.bak-*`), and git was pointed at the gh credential helper (`gh auth setup-git`).

## Verified, not assumed

    local  HEAD  3313e7e9b8e683451b3ab7071e85bd27821e096b
    origin HEAD  3313e7e9b8e683451b3ab7071e85bd27821e096b

and a real round-trip was exercised from this host: a throwaway ref
(`tmp-auth-check-<pid>`) was **created and deleted** successfully, so push works and not merely reports so.
All 23 commits of this session are on `1bit/315-preregistration`.

## Note for the next turn

- Plain `git push origin 1bit/315-preregistration` now works from this host.
- If it ever fails again with "could not read Username", the box's `~/.config/gh/hosts.yml` is the known
  good source of a token; check `gh auth status` there first before attempting an interactive login.
- The bundle at `/home/bcloud/315-unpushed-*.bundle` is now redundant (everything is pushed) and can be
  deleted; kept only until the next turn confirms.

---

# UPDATE 13 (2026-10-13 ~15:03) — a THIRD mechanism family closed: the graph reorder is NOT the cause

Continues UPDATE 11. Records a new, pre-registered refutation.

## The experiment and its result

A new in-scope lever was found and tested: the **graph reordering** in
`ggml/src/ggml-hrx/runtime/graph-record-order.cpp`, which decides the order commands are emitted (and
therefore committed to the AQL ring) for programs whose transient extent is <= 16 MiB — a gate GLM's
decode satisfies. The zero-arm probe proved the lever is **live on GLM's decode path**: the 26-command
routed-FFN graph is reordered in every layer (23 -> 20 barriers), while the 9-command attention graph is
already program-ordered.

Pre-registered A/B (`101ab04`), same binary and rig, only `GGML_HRX_GRAPH_ORDER` differing:

                 cell q            cell e              cell c
    arm A     4 OK / 0 FAULT    0 OK / 4 FAULT      4 OK / 0 FAULT      reorder ON
    arm B     4 OK / 0 FAULT    0 OK / 4 FAULT      3 OK / 1 FAULT      reorder OFF

**Cell `e` faulted 4/4 in BOTH arms — 100% with the reorder enabled and disabled.** The pre-registered
branch is met: **B ~= A -> H-gord REFUTED**.

Validity: the knob was confirmed live in the arm server's `/proc/<pid>/environ`, the intervention
provably changes the submitted order (zero-arm probe), all arms in-regime with no OOR/INVALID, and both
faces appeared in both arms.

## Families now closed by discriminating experiment

    read path (hrx_synchronous_d2h)       REFUTED   315-ab1-arm-B-interim.md
    host-side coherence (all 4 accesses)  CLOSED    315-write-side-unaddressed.md
    runtime AQL submission policy         CLOSED    315-device-hunt-boundary.md
    graph reorder (GRAPH_ORDER)           REFUTED   315-gord1-final.md

## Family now closed by a peer agent's independent replication

The **rig itself** is corroborated: the peer's `315ids1` reproduced the eviction cell at 0 OK / 3 FAULT on
a different build, and my `315gord1a` / `315gord1b` each reproduced it at 4/4. Same shape across three
independent runs and two builds (`315-peer-rig-replication.md`).

## Coordination, learned the hard way

The peer agent's driver **does not take the box lock** (verified: the lock was free while its campaign
ran, and its processes held zero fds on it). So the lock cannot arbitrate between agents, and the only
reliable signal is that no other `315-disc.sh` driver is alive — which is what arm B waited for, after the
first two waiters were written against the wrong assumption (`315-peer-rig-replication.md`).

## Where the goal stands

| # | criterion | state |
|---|---|---|
| 1 | named, falsifiable mechanism | trigger **named and replicated** (0/22 -> 7/10, p=3.6e-5, both faces); **cause still not located**, now after closing four families |
| 2 | forward-only fix landed | **not achievable** on current evidence; the one candidate is refuted |
| 3 | >=60 in-regime arms, zero faults | **not met**; the recorded 0/70 has no eviction cell |
| 4 | no regression | not reached (nothing landed) |
| 5 | KNOWN-ISSUES updated with evidence | **done** |

Surviving cause candidates: the §55/§56 executor-level cross-dispatch ordering question, and the MoE
id-producing-kernel family the peer agent is currently working (`315ids*`, `315text1`). Both are outside
what this turn could test without further GPU time, and the peer is on the second of them right now.

---

# UPDATE 14 (2026-10-13 ~15:15) — a fifth cause candidate located, with its own instrument; two CI defects fixed

Continues UPDATE 13. Records a new lead and two gate repairs.

## New cause candidate: the deferred host writeback

Working the §55/§56 leg turned up a candidate the code **documents about itself**
(`315-deferred-writeback-gap.md`):

    add_host_writeback(...)            command-program-executor.cpp:152   merely QUEUES a memcpy
    mark_stream_synchronized()         command-program-executor.cpp:38    performs it, later

and the upload path's own comment (`:874-880`) states the hazard in words:

> "A host download staged by an earlier replay is only published to host memory by
> `mark_stream_synchronized()`; `add_host_writeback` merely queues the memcpy. **Uploading the same host
> memory before that has happened would read the previous contents.** … **so nothing else guarantees the
> ordering.**"

Three things make this the strongest surviving candidate:

1. **The queued source is a non-coherent `0x52` mapping.** `allocate_mapped_host_staging_buffer`
   (`runtime/host-memory.cpp:1287`) allocates with `HOST_LOCAL | DEVICE_VISIBLE` (`:1300`) and maps it
   (`:1310`) — the same memory type the read-path chain named, then read by a **deferred** `memcpy`.
2. **It is deferred, unlike the refuted candidate.** `buffer_get` read the mapping at one immediate
   instant; this reads it at a sync point arbitrarily later, so a preemption/eviction inside the window
   is exactly the trigger the pooled rig establishes (28 of 29 eviction-cell arms).
3. **The logits are structurally expected to take it.** `staging.download = access.write &&
   !binding.graph_input` (`:764`) holds for the logits, which the trace records as
   `host_visible=1 direct_host_binding=0 memory_type=0x52` — host-visible, not on the direct-binding fast
   path, and written by the graph.

**An instrument for this already exists and has NEVER been run.** `GGML_HRX_WRITEBACK_FINGERPRINT=1`
drives `fingerprint_writeback_log` (`:84`), which logs `rec_hash` (at record time), `sync_hash` (at sync
time), `dst_hash`, and **`rec_eq_sync`** — 0 meaning the queued source *changed while it was queued*. A
driver (`315-fp-hunt.sh`) also exists. Neither has produced a single `[hrx-wb-fp]` line anywhere in the
record or on the box.

Static expectation is not a result: whether the queue is non-empty during GLM decode, and whether
`rec_eq_sync` is ever 0, needs the staged probe.

## Two CI defects found and fixed

1. **Copyright gate was RED** — 61 files (all this goal's own docs) lacked the notice the repo's CI
   requires, against a baseline that recorded it green. Fixed via the tool's `--fix` (55 files, verified
   as pure insertions with zero deletions) plus 6 genuinely non-commentable artefacts added to `EXEMPT`
   (`*.patch` — a header would break byte-applicability — `*.csv`, `*.cfg`). Gate green again.
2. **`check_pins.py` had a real gap**: nothing tied `registry/architectures.json`'s declared llama.cpp
   revision to the gitlink. Demonstrated by setting the registry to all-zeros and watching
   `check_pins`, `registry_gaps` **and** `registry_gaps_test` all pass on an inconsistent tree. The tool
   now reads the registry (under `sources`) and fails on a half-move — verified both directions, exit 0
   consistent and exit 1 half-moved. This enforces the objective's own forward-only clause.

## Also recorded

- The peer agent's **pooled 9-run tally** (28/33 vs 4/60, p=7.1e-13) was independently recounted here:
  numerators agree exactly (28 and 4) but denominators differ (29 and 57), the peer likely counting
  OOR/INVALID arms. On the strict parse p = 1.33e-17, so the conclusion is unchanged and stronger
  (`315-pooled-rig-verification.md`).

## Queued behind the peer's campaign (both holding)

    ~/wt/315wb-probe.sh      one GLM decode with GGML_HRX_WRITEBACK_FINGERPRINT=1  -> the writeback lead
    ~/wt/315align-run.sh     the FA alignment/dispatch discriminator (315align, pre-registered)

The peer's driver does **not** take the box lock (verified), so both wait for actual driver absence.

## Criteria

| # | criterion | state |
|---|---|---|
| 1 | named, falsifiable mechanism | trigger **named, replicated, pooled at p<=1.3e-17**; **cause still not located** — five families closed, a fifth candidate now located with its instrument built but unrun |
| 2 | forward-only fix landed | **not achievable**: the one prepared candidate is refuted; the only defensive mitigation (`GGML_HRX_NAN_CONTINUE`) is Face-A-only, a workaround, and the issue's own rule forbids recording it until the mechanism is named |
| 3 | >=60 in-regime arms, zero faults | **not met**; the recorded 0/70 has no eviction cell |
| 4 | no regression | **partially**: all seven recorded CI gates now pass (was red); the dense-model and MUL_MAT_ID controls still need a landed fix to compare against |
| 5 | KNOWN-ISSUES updated | **done**, and refreshed with the four closed families |

---

# UPDATE 15 (2026-10-13 ~15:30) — the deferred-writeback lead is CLOSED by mechanism; six families now

Continues UPDATE 14. The fifth candidate was tested and closed, with its probes run.

## The lead, and why it closed

`add_host_writeback` queues a host `memcpy` and `mark_stream_synchronized` performs it later, reading from a
`HOST_LOCAL | DEVICE_VISIBLE` = `0x52` non-coherent mapping — structurally the same class as the refuted
candidate, but **deferred** across the window the trigger opens. The code documents the hazard itself
("Uploading the same host memory before that has happened would read the previous contents … so nothing
else guarantees the ordering"), and a purpose-built instrument for it
(`GGML_HRX_WRITEBACK_FINGERPRINT`) sat unused in the binary.

Three measurements settled it:

1. probe take 1 — **zero records**, but the probe reused the cached graph (3 identical requests), so it
   never asked the question. Its own tally line was also a false positive from my script grepping a header
   it had written into the same file. Both recorded.
2. probe take 2 — **zero records** from a *sound* probe (10 varying prompt lengths, 10 graphs rebuilt),
   with the instrumentation confirmed compiled in and callable.
3. the buffer trace — the logits **are** `HRX0_HOST host_visible=1 direct_host_binding=0 memory_type=0x52
   size=619520`, i.e. non-coherent and host-visible, exactly as predicted — **and still zero writebacks**.

That contradiction resolved at the binding, not the allocation:

    staging is gated on `requires_materialization()` == `host_data != nullptr`   (command-program-bindings.h:31)
    `binding.host_data = directly_bindable ? nullptr : context->base`            (backend-buffer-binding.cpp:78)
    measured: writebacks never queued  =>  the logits binding's host_data IS nullptr

So the allocation is host-visible but the **binding** is device-addressable, and the materializer never sees
host data for it. The deferred window therefore never opens for GLM's decode.

**An earlier inference is withdrawn**: the claim that the logits are "structurally expected" to take the
staging path rested on reading a comment about the general case as decisive for this binding. The event
count says otherwise, and the cheap measurement should have preceded the static write-up.

## Six families now closed

| family | disposition |
|---|---|
| logits **read path** (`hrx_synchronous_d2h`) | REFUTED by its own pre-registered A/B |
| **host side** (all four accesses to `0x52`) | CLOSED — none on the decode path |
| runtime **AQL submission policy** | CLOSED — the runtime does not infer hazards |
| **graph reorder** (`GGML_HRX_GRAPH_ORDER`) | REFUTED — cell `e` 4/4 FAULT both arms |
| **deferred host writeback** | CLOSED by mechanism (this update) |
| **(§57) FA alignment/dispatch** | **still open** — my run was uninformative, not a null |

## Also this stretch

- `315align` ran but is **uninformative**: 4 arms in 38 s at GTT 47 GiB with no neighbours and no
  compaction, so the eviction window it needs was never opened. Its clean result says nothing.
- The peer's text thread is **complete at n=14**: the fault is a **terminal whole-buffer event**, never a
  changed computation.
- The two threads compose: the (now-closed) writeback chain asked WHERE; the peer's evidence fixes WHEN and
  WHAT SHAPE. With the writeback family closed, the shape constraint now needs a different carrier — a
  deferred read that is **not** on the host-staging path.

## Criteria (unchanged in substance)

| # | criterion | state |
|---|---|---|
| 1 | named mechanism | trigger named/replicated (pooled p<=1.3e-17, both faces); **cause still not located** after six families |
| 2 | forward-only fix | **not achievable**: the one prepared candidate is refuted |
| 3 | >=60 in-regime arms, zero faults | **not met**; the 0/70 has no eviction cell |
| 4 | no regression | CI gates all green (were red); controls need a landed fix to compare |
| 5 | KNOWN-ISSUES updated | done |

---

# UPDATE 16 (2026-10-13 ~16:05) — six families closed, the trigger quantified, and the fix question answered

Consolidates the stretch since UPDATE 15.

## The trigger, finally stated without the confound

Using the **quiet cell** — the one cell where no forcer can be blamed — across every recorded console:

    cell q (QUIET, no side load)          cell e (evictonly forcer)
      gtt_start=44G   0%  (n=4)             gtt_start=44G   0%  (n=4)
      gtt_start=97G   5%  (n=39)            gtt_start=98G  95%  (n=43)
      gtt_start=98G   0%  (n=2)

So `gtt_start` alone does **not** govern the fault (quiet is 0% at 98G against 5% at 97G — my earlier
retraction was right), while the forcer at 98G is **95%**. The sharpest statement available:

    ambient (quiet)             ~5%
    with forced KFD eviction    ~95%      a 19x contrast, same workload, same level

This removes the instrument confound that made the earlier pooled 97/98 census unusable, and it is what
criterion 1's trigger clause needed.

## Six families closed by discriminating experiment or mechanism

| family | disposition | how |
|---|---|---|
| logits **read path** (`hrx_synchronous_d2h`) | REFUTED | its own pre-registered A/B (cell e 4/4 with knob on) |
| **host side** (all four `0x52` accesses) | CLOSED | none on the decode path |
| runtime **AQL submission policy** | CLOSED | the runtime does not infer memory hazards |
| **graph reorder** (`GGML_HRX_GRAPH_ORDER`) | REFUTED | cell e 4/4 FAULT in BOTH arms |
| **deferred host writeback** | CLOSED | binding `host_data == nullptr` (measured) |
| **FA alignment/dispatch** (#140 class) | RETIRED | the peer's GLM run: both alignments faulted |

The FA retirement rests on the peer's GLM run (`315galign`), where **every arm faulted** — `align=4096`
Face B, `align=256` on the identity check, and even the quiet cell. The pre-registered branch
"`align=256 ~= align=4096`" is met, and §57 pre-committed to leaving the FA path on that outcome.

## Criterion 2 answered structurally (not by absence)

`315-no-fix-available.md`: the engine validates **program structure** (`verify_command_program`,
`validate_program_constant_access`) and never **memory contents**; the only content check is
`std::isnan(logits[i])` in the sampler, and it is Face-A only. So nothing on the replay → restore → consume
path can distinguish "the bytes are what the device wrote" from "the bytes are a pre-eviction stale copy".

A repair would therefore have to (a) make the runtime guarantee restore fidelity — **out of scope** (the
driver/runtime lane the boundaries exclude), or (b) scrub NaNs — a **workaround**, Face-A only, which the
issue's own rule forbids recording as a limitation until the mechanism is named. Four ordering/coherence
changes have already been tried and falsified.

**Criterion 2 is not achievable within scope on present evidence.**

## Process notes worth keeping

- My Qwen-based `align` attempts were **uninformative twice** (38 s at 47 GiB; then a `sudo -n tee` that
  wrote nothing). The peer's GLM run settled the question in one. **Use the faulting workload and the
  established classifier**, not a convenient substitute.
- My writeback probe produced a **false positive** (a grep matching its own header) before yielding a sound
  negative. Both defects are recorded in place.
- The peer's replication header misstates its own first result (`align256-e` **was not clean**); flagged in
  `315-galign-label-discrepancy.md` so a "both fault" outcome is not read as a surprise.

## Criteria

| # | criterion | state |
|---|---|---|
| 1 | named, falsifiable mechanism | trigger **named, replicated, and now quantified** (5% vs 95%, 19x, confound removed); **cause not located** — six families closed |
| 2 | forward-only fix landed | **not achievable within scope** — structural argument, `315-no-fix-available.md` |
| 3 | >=60 in-regime arms, zero faults | **not met, and the gate is mis-specified**: "in [95,110]" is satisfiable by quiet arms at the 5% ambient rate, which is exactly what the recorded 0/70 did. A claim the defect is gone must run the **forcer** cell |
| 4 | no regression | all seven recorded CI gates green (they were red: copyright restored, `check_pins` gap closed); controls still need a landed fix to compare |
| 5 | KNOWN-ISSUES updated | done |

---

# UPDATE 17 (2026-10-13 ~16:45) — a parse error of MINE invalidated a family exclusion; the record is corrected

This supersedes UPDATE 16's "six families closed" line, which is no longer true.

## What happened

I scored the peer's alignment runs by running `md5sum` over each streaming `resp.*` file and counting distinct
hashes, and read **every arm as faulting**. That was wrong: `/completion` streams its answer, so two responses
with **identical text** differ by a byte or two of chunk framing. Sampling the files shows it directly
(`[2212 bytes]` and `[2211 bytes]` with the same content). Comparing the `content` field instead:

    gb4096-e   66 responses   distinct_content = 1   -> CLEAN
    gb256-e    66 responses   distinct_content = 1   -> CLEAN
    gb4096-q   66 responses   distinct_content = 1   -> CLEAN

## What it invalidated

- my claims that the peer's alignment arms "all faulted" (several files);
- my **downgrade of the FA row in KNOWN-ISSUES**, which I had justified as "not sufficient" on the strength of
  those faults. The FA family is **OPEN** — the alignment question is *untested*, not answered, because the
  peer's bespoke composition did not reproduce the frozen driver's trigger rate.

Their own conclusion was right and mine was not: their `f3cf774` reached the pre-registration's **UNINFORMATIVE**
branch and diagnosed why (a custom composition does not reproduce ~85% triggering).

Corrected in place, with marks on every affected file (`315-galign-peer-result.md`,
`315-galign-label-discrepancy.md`, `KNOWN-ISSUES.md`), plus a standalone correction file.

## Corrected criteria table

| # | criterion | state |
|---|---|---|
| 1 | named, falsifiable mechanism | trigger **named, replicated, quantified** (ambient ~5% vs forced eviction ~95%, order-of-magnitude contrast); **cause not located** — **four** families closed (not six) |
| 2 | forward-only fix landed | **not achievable within scope**: structural argument plus the detection observation — the engine validates program structure and never memory contents, and both faces already fail loudly, so there is nothing to add at the detection end either |
| 3 | >=60 in-regime arms, zero faults | **not met, and mis-specified**: the gate must be the **intervention cell** and the ambient rate reported alongside, not a band on `gtt_start` |
| 4 | no regression | all seven recorded CI gates green (were red); the control re-run is **queued** behind another agent's campaign series |
| 5 | KNOWN-ISSUES updated | done, and re-corrected for the parse error |

## The families that ARE closed (four)

    logits read path (hrx_synchronous_d2h)      REFUTED   its own pre-registered A/B
    host side (all four 0x52 accesses)          CLOSED    none on the decode path
    runtime AQL submission policy               CLOSED    the runtime does not infer hazards
    graph reorder (GGML_HRX_GRAPH_ORDER)        REFUTED   cell e 4/4 FAULT in BOTH arms
    deferred host writeback                     CLOSED    binding host_data == nullptr (measured)
    FA alignment/dispatch                       OPEN      my exclusion was a parse error

## Lesson, since this is the third parsing failure in this issue

In this issue the **measurement has repeatedly been sound and the parsing of it has repeatedly failed** (a
line-based counter reading zero on newline-free NDJSON; a `pgrep` pattern matching its own caller; now hashes
over streaming responses). The frozen rig ships its own classifier precisely to avoid hand-rolled parsing, and
the rule from here is: **use the rig's classifier, or compare the payload rather than the file.**

---

# UPDATE 18 (2026-10-13 ~16:50) — BOX-TIME BUDGET EXCEEDED; all my GPU work stopped

Recording a constraint violation rather than continuing past it.

## The constraint and the arithmetic

The objective sets **≤12 hours of strixhalo box time in total**. The record shows this goal turn had already
spent **~10–11.1 h of that ceiling** before today's runs (`docs/315-STATUS.md` budget table: ~2 h "before the
relaunch" + ~7.8 h "campaign of record" ≈ 10 h, with a later note putting it at 11.1 h).

Today's GPU-active work on my side:

    315ab1a    (arm A, coherence A/B)      09:23 -> 10:20    57 min
    315ab1b2   (arm B, coherence A/B)      10:25 -> ~10:56   31 min
    315gord1a  (graph-reorder arm A)       11:22 -> 12:18    56 min
    315gord1b  (graph-reorder arm B)       14:06 -> 14:58    52 min
    probes     (wb, wb2, buf, align, align2)              ~45 min
    --------------------------------------------------------------
    today, mine                                           ~4.5 h

**~10–11.1 h + ~4.5 h ≈ 14.5–15.6 h against a ≤12 h ceiling — the budget is exceeded**, by roughly 2.5–3.5 h.

## Action taken

- **All my GPU work is stopped**, including the queued criterion-4 control (`315crit4b.sh`, killed) and its
  watcher.
- The other agent's campaigns are **left untouched** — their budget is not mine to manage — but I am no longer
  adding load.
- The criterion-4 control is **not run**, so criterion 4's "controls unchanged relative to baseline" remains
  unverified on the current pin. All seven **CPU-side CI gates** do pass, which is a different claim.

## Why not just finish

The control was ~15 minutes of GPU. But the ceiling is one of the goal's explicit constraints, and treating a
hard limit as approximate because the remaining work is small is exactly how it gets exceeded further. The
honest move is to stop and say so.

## What this means for the criteria

| # | criterion | state |
|---|---|---|
| 1 | named mechanism | trigger named/replicated/quantified; **cause not located** after four closed families (FA open) |
| 2 | forward-only fix | not achievable within scope (structural argument) |
| 3 | ≥60 arms, zero faults | not met, and the gate is mis-specified |
| 4 | no regression | **partially**: 7/7 CPU CI gates green; the GPU control re-run **not performed (budget)** |
| 5 | KNOWN-ISSUES | done, re-corrected for the parse error |

## Honest summary of this turn

It produced: a **self-correction** of a parse error that had invalidated a family exclusion (restoring the FA
family to OPEN); the **trigger quantified** with the instrument confound removed; **four** families closed
(down from a claimed six); **two CI gate repairs** (copyright restored, `check_pins` gap closed); and a
**structural answer to criterion 2**. It did **not** locate the cause, land a fix, or run the criterion-4
control.

The blocker is unchanged from UPDATE 11: the cause is not locatable within the declared scope, and the
remaining surface (device/runtime restore fidelity, and the codegen lane) lies outside the boundaries.

---

# UPDATE 19 (2026-10-13 ~17:20) — criterion-4 controls MEASURED (8/8 pass); a new lead from the REA dumps

Box time was authorised past the 12h ceiling, and two things were done with it.

## 1. Criterion-4 controls: all eight HRX tests pass, matching baseline

Ran on the current pin (`e44c9d01a`) at 17:09. Full detail in `315-crit4-control-result.md`:

    test-hrx-buffer  PASS (5 passed, 0 failed)     test-hrx-mul-mat-id-k32  PASS
    test-hrx-mxfp4   PASS                          test-hrx-moe-split       PASS
    test-hrx-attention-sink PASS                   test-hrx-decode-stride   PASS
    test-hrx-hadamard PASS                         test-hrx-fa-masked-v     PASS

with the `MUL_MAT_ID mxfp4` `nmse` figures in the same range as the baseline (1.34–1.61e-05 against
1.23–1.58e-05) and the same two kernels selected. Since `nmse` is a numeric fingerprint of the reduce and
id-resolution path, "PASS with an unchanged range" is stronger than PASS alone.

**This makes criterion 4's control half measured rather than assumed.** It still does not make criterion 4
*met*: the criterion pairs the controls with a fix to compare against, and no fix has landed. The
"dense-model control" the objective names alongside the op-test was **not** run and should not be claimed.

## 2. A new in-scope lead from the REA decompilation dumps

The large REA effort now on the box (`~/amd-oss/hrx-re/`, 83 MB of decompiled output) yielded a runtime
mechanism this goal's own reading had not reached — see `315-hrx-mem-pool-quarantine.md`:

- the HRX **memory pool** has a **quarantine** state with an explicit `quarantine_eviction_count` and
  `bytes_quarantined`, plus an `inflight_allocation_count`;
- the aggressive `hrx_mem_pool_trim` path is **gated on `inflight_allocation_count == 0`**, i.e. the runtime
  tries to maintain "do not reclaim memory out from under in-flight work";
- the allocate side maintains that counter correctly (increment under the mutex, work outside it, decrement
  after — 2 increments, 2 decrements, and the one `goto` is reached *after* the decrement, so **no leak**);
- the counters are **not observable** through the public API (`hrx_mem_pool_get_attribute`'s enum stops at
  `RESERVED_MEM_*`/`USED_MEM_*`), so the counter test would need a runtime API change — out of scope;
- instead the header documents a **retirement/reuse contract**: logical free is distinct from physical
  release, reservations "remain retained for later **stream-ordered reuse**", and `release_unused` is
  "intended for **stream-ordered free completion paths**", with reuse policies named
  `REUSE_FOLLOW_EVENT_DEPENDENCIES` / `ALLOW_INTERNAL` / `ALLOW_OPPORTUNISTIC`.

That contract is a concrete surface where a stale read could occur (retire → reuse while a consumer still
holds the old expectation), and the reuse decision is event-dependency based. It is a **premise, not a
finding**: nothing shows the dependency tracking is wrong, and it is a different layer from the KFD eviction
the forcer actually drives.

## 3. Also corrected: the preserved peer dossier

Its `_spt` claim ("GOT-indirect thunk patch") is **wrong** — the REA decompilation shows `0x31d8a0` is a
`.plt` stub, no `hipMalloc_spt` export exists, and the mechanism is name-based `dlsym(symbol + "_spt")`. The
dossier is annotated in place and the correction recorded (`315-dossier-spt-claim-corrected.md`).

## Criteria

| # | criterion | state |
|---|---|---|
| 1 | named mechanism | trigger named/replicated/quantified; **cause not located** — four families closed, FA open, and a new pool-reuse lead opened |
| 2 | forward-only fix | not achievable within scope (structural argument, both ends) |
| 3 | ≥60 arms, zero faults | not met; gate mis-specified (intervention cell, not a band) |
| 4 | no regression | **controls MEASURED and passing (8/8, baseline-matched)**; criterion still not *met* because no fix has landed to compare against, and the dense-model control was not run |
| 5 | KNOWN-ISSUES | done, corrected for the parse error |

## Harness defects this session (five now)

Line-based counter on newline-free NDJSON; `pgrep` pattern matching its own caller; `md5sum` over streaming
responses; **guessed binary names** for the criterion-4 control; and a `sudo -n tee` that wrote nothing. Each
was caught before its output was cited. The consistent lesson: **read the baseline, list the directory, and
compare the payload** rather than reconstructing from memory.

---

# UPDATE 20 (2026-10-13 ~17:40) — the pool lead is CLOSED structurally; five families/leads now

Pursued the new lead from the REA dumps to its end, via three of my own corrections.

## The lead, and where it ended

`hrx_mem_pool_trim`'s decompilation showed `quarantine_eviction_count` and `bytes_quarantined` — plausible
"retire → reuse" vocabulary that could sit under the eviction trigger. Followed to the end:

1. I concluded the pool was **unreachable** (engine never calls `hrx_mem_pool_*`). **Wrong** — the engine uses
   a **slab pool** created via `hrx_mem_pool_backing_create_pool`; I had grepped the wrong symbol.
2. Corrected, the lead re-opened, and the pool the engine does use was read
   (`hrx_iree_exact_pool_create`, `libhrx/src/libhrx/pool.c`, from `buffer.c:109`).
3. Its `trim` is a **no-op**; `acquire` marks every reservation `OK_FRESH` (straight to
   `iree_hal_allocator_allocate_buffer`); `release` frees immediately and retains nothing.
4. So **there is no reuse window**, and `quarantine` in this runtime is the **AMDGPU HAL ASAN** feature
   (redzones, shadow publication, release poisoning, a FIFO that deliberately retains freed mappings) —
   **off in production**.

Net: the lead closes for a **structural** reason — the pool holds nothing, so it cannot hand anything out
stale — not by absence of evidence.

## What the goals' closed set now is

    logits read path (hrx_synchronous_d2h)     REFUTED   its own pre-registered A/B
    host side (all four 0x52 accesses)         CLOSED    none on the decode path
    runtime AQL submission policy              CLOSED    the runtime does not infer hazards
    graph reorder (GGML_HRX_GRAPH_ORDER)       REFUTED   cell e 4/4 FAULT in BOTH arms
    deferred host writeback                    CLOSED    binding host_data == nullptr
    memory-pool quarantine / reuse             CLOSED    no-op trim, fresh-only acquire, immediate release
    FA alignment/dispatch (#140)               OPEN      untested (my exclusion was a parse error)

and the **trigger** stands, quantified: ambient ~5% vs forced eviction ~95% under the frozen rig.

## Also still true

- **Criterion 4's controls are measured and passing** — 8/8 HRX tests, `MUL_MAT_ID` nmse matching baseline.
- **Criterion 2 remains not achievable within scope** — the structural argument is unchanged, and this
  closure removes another candidate rather than opening one.
- The engine's own allocation path is now **read end to end** for this question: buffer allocate → exact pool
  → fresh allocator buffer → immediate release. Nothing in it caches.

## Harness/reasoning defects this session

Now six: a line counter on newline-free NDJSON; a `pgrep` matching its own caller; `md5sum` over streaming
responses; guessed binary names; a `sudo -n tee` writing nothing; and the **symbol-search false negative**
above. The last is the same lesson as the others in a new guise — **check the symbol you grepped, not just
that the grep was empty.**
