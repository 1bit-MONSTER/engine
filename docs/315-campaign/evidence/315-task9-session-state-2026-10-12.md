# task-9 session state — 2026-10-12 late session

Written by an agent session that worked #315 while the `315cwsr70` campaign of record ran.
Added as evidence so the findings survive the session; **nothing existing was modified**.

Read alongside `docs/315-ENDOGAME.md` (the runbook) and `docs/315-STATUS.md`. Where this file and
those disagree, this file records what was **verified live** and names the check used.

Headline results in this file:
 - the `cwsr=0` signal pooled across three same-boot campaigns is **0 faults / 72 in-regime arms**
   against `cwsr=1` 3/11 -> one-sided Fisher p = 0.0018, rule-of-three bound 4.17% (excludes the
   26% baseline); all 72 arms verified at `gtt_start=97G`
 - **H2's carrier is contradicted** by the vendored AMD reference set: `cwsr_enable=0` requests
   `KFD_PREEMPT_TYPE_WAVEFRONT_DRAIN`, and GFX11 maps SAVE to `DRAIN_PIPE`, so it does not remove
   mid-wave preemption as H2 requires
 - `315-rate.py` counts BOTH cells, so its "criterion 3 MET" is wrong; fix-cell (`arm l`) only is
   authoritative
 - the SMI watcher's 3h TTL silently truncated the campaign's eviction data to its first 179 min
 - H2 is untestable on all existing data: no faulting arm anywhere has a bound PID
 - two docs conflict: `315-cwsr-refined-hypothesis.md` still asserts the H1 confound that
   `315-H1-FALSIFIED.md` withdrew (and ground truth on the box supports the withdrawal)

Note: `docs/315-campaign/evidence/NPU-LOCK-WEDGED-2026-10-12.md` (an earlier session) is now
**stale** - `npu.lock` was verified free and acquirable in this session.

---

# engine#315 — state at 2026-10-12 ~22:20Z (corrected)

## My earlier framing was wrong on two counts; this supersedes it
1. **H1 (fresh-boot confound) is FALSIFIED**, not unresolved. `docs/315-H1-FALSIFIED.md`
   reconstructs boot time from three independent clocks (KFD SMI `boot_ns` anchored two ways +
   `uptime -s`) agreeing within 4 s: run 7 ran **~6-40 min after a fresh boot** and still faulted
   3/11. So a fresh boot does not suppress the fault. The confound I called "unresolved and
   blocking" does not exist. This was the prior session's own error, corrected and withdrawn.
2. **H2 is the leading and only surviving candidate**: mid-wave preemption under memory pressure.
   H3/H4 dead (GPU-MV sizing defect proven fixed; `315-cwsr-size-excluded-PROOF.md`).

## The runbook is `docs/315-ENDOGAME.md` (read it first, not STATUS.md)
Exact commands, decision rules, and reading keys, in order. Committed to `1bit/315-preregistration`.

## Where the campaign stands (checked 22:15, alive)
tag `315cwsr70`, started 16:54 box time, box booted 15:15:40, ~8 h expected.
    arm l (fix cell)  OK=54  FAULT=0  OOR=0
    arm c (ctrl cell) OK=19  FAULT=0  OOR=0
Currently pair 55 arm l in flight. Console mtime advancing (22:13:38); not stalled.

## The trap in scoring it (ENDOGAME §1)
§6/§7 count **fix-cell (`l`) arms only**, not the total. 70 pairs => ~70 `l` arms + ~23 `c`.
The `c` arms are controls and **do NOT count toward the 60**. Verify directly:
    grep -oE "pair [0-9]+ arm l: OK" ~/wt/315cwsr70-console.log | wc -l
Score with `python3 315-rate.py 315cwsr70-console.log --det ~/wt/rt-det --tag 315cwsr70`,
and require: `FAULT 0`, no `MISSED BY CONSOLE`, `clean >= 60` (fix-cell only), `UNVERIFIED 0`.
Read the console as a SEQUENCE — a late fault is a different finding from an early one and must
not be averaged away.

## Then the control (criterion 1's decisive test; H2's only PID-captured faulting arm)
    scp docs/315-campaign/315-cwsr-control.sh strixhalo:~/wt/
    ssh strixhalo 'cd ~/wt && TAG=315ctl1 PAIRS=20 bash 315-cwsr-control.sh'
Reboots; **kills the campaign**, so run only after it finishes. Expects ~5.2 faults if baseline 26%.
  rate ~26% -> H1 confirmed, criterion 1 collapses, re-scope honestly
  rate ~0%  -> H1 dead, proceed to H2 with the control's faulting arms

## Then H2 (`315-evict-per-arm.py`), then criterion 2 (forward-only), 4 (no regression), 5 (KNOWN-ISSUES)
- Criterion 2 is OPEN: **no landed repair exists.** Candidate branch `1bit/315-coherence-fwd` was
  DELETED as falsified. Do NOT land the coherence patch — `tools/315-lever-check.py` rejects it.
- The pointed-at repair is engine-side **residency** (weights/KV/activations in driver-owned/VRAM
  rather than evictable GTT/userptr/SVM), matching Engine V2 D5. An engine startup guard is a
  WORKAROUND, not a repair — say so plainly if it is the only option.
- Criterion 5 (KNOWN-ISSUES limitation note) is LAST: only after a mechanism is named.

## What I did NOT do
Did not kill the campaign, reboot, touch the peer's servers, prune arms, or write a limitation
entry. All read-only inspection.

---

## SCORING TRAP CONFIRMED LIVE (2026-10-12 22:22) — read before trusting 315-rate.py

`python3 315-rate.py 315cwsr70-console.log --det ~/wt/rt-det --tag 315cwsr70` reports:

    arms parsed 73 | clean 73 | FAULT 0 | rate 0.00% | CI 4.02%
    "Criterion 3's bar (>=60 arms) is MET."

**That is WRONG on the fix-cell tally, exactly as ENDOGAME §1 warned.** The scorer counts
BOTH cells:

    fix-cell (l) arms : 54
    ctrl-cell (c) arms: 19
    54 + 19 = 73   <- what the scorer counted

ENDOGAME §1: "§6/§7 count **fix-cell arms**, not total arms... the `c` arms are controls and
**do not count toward the 60**." So the honest fix-cell figure is **54, which is SHORT of 60.**

The scorer's `FAULT 0`, `INVALID 0`, `OOR 0`, per-face tally (FaceA=0 FaceB=0), and CI math are
all fine — only the arm *population* is wrong. Do not quote "criterion 3 MET" from this tool
without first checking the fix-cell count directly:

    grep -oE "pair [0-9]+ arm l: OK" ~/wt/315cwsr70-console.log | wc -l

At 54/70 pairs and a cap of 70, the run can yield at most ~70 fix-cell arms, so 60 IS reachable —
it simply has not been reached yet. Let the campaign finish, then re-score and re-check the
fix-cell number before claiming the bar.

**Action when it lands:** if fix-cell clean >= 60 with FAULT 0, criterion 3's arm bar is met
(subject to the control in §"Then the control"). If it lands short because arms went OOR, ENDOGAME
§1 forbids pruning — re-run only for a recorded INVALID arm, or run more pairs.

---

## CONTROL SCRIPT TRAP (found 2026-10-12 22:18) — the foreign-HRX guard misfires on OUR OWN neighbours

`315-cwsr-control.sh` has two pre-flight guards:

    if pgrep -f '315-evict-trace'            -> ABORT "a 315 campaign is still running"
    if ps -eo args | grep -- '-dev HRX0'     -> ABORT "an HRX job is running (not ours)"

Live preconditions at 22:18:

    1. campaign running?  YES  -> correctly aborts (by design; run it after the campaign)
    2. foreign HRX job?   YES  -> WOULD FALSELY ABORT
    3. cwsr_enable        = 0
    4. drop-in present    = yes
    5. sudo -n available  = yes

**Guard 2 is a false positive.** The three `-dev HRX0` servers on ports 20251-20253 are NOT a
peer's job — they are the running campaign's OWN neighbour servers:

    pid 555575 / 558473 / 561384
      ppid 552972 = bash /home/bcloud/wt/315-paired.sh 70 2 2048 /home/bcloud/wt/315instr3-build/bin ...
      = the live 315cwsr70 campaign

They are the composition that drives GTT into the 97-98 GiB regime (2 neighbours queued per the
`neighbours=2` header, cycling across those three ports). They are 5h22m old and idle between arms
(0 connections each) — idle, but NOT orphaned.

**Consequence:** once the campaign finishes, guard 1 clears but guard 2 may NOT, because those
servers are spawned by the campaign and may outlive it (they did after 315disc1). If the control
then aborts on guard 2, the message ("not ours") will be misleading, and an operator could kill
our own leftover neighbours believing they belong to a peer.

**Correct handling when the campaign ends:**
 - confirm the campaign process is gone (guard 1 clear);
 - check `ppid` of any remaining `-dev HRX0` server. If its parent is a dead `315-paired.sh`, it is
   OUR leftover and may be reaped. If its parent is some other live workload, it is genuinely
   foreign and must not be touched.
 - do NOT kill anything whose parent chain still includes a live campaign script.

---

## Campaign liveness VERIFIED (22:17) — not stalled

Repeated "no advance" readings at 22:16/22:17 were the observer sampling mid-arm, not a stall.
Positive evidence that pair 55 arm l is computing:

    llama-server pid 2275356  1562% CPU  state Rl   (multi-core, running)
    rt-det/315cwsr70-p55l-.../server.log   51768 B @22:17:49 -> 53267 B @22:17:58  (grows)
    run.log: "gtt_start 97GiB mode performance"     (in-regime)
    detector age 3m41s of a 1200 s timeout
    faults in this arm: 0
    port 19593 ESTAB, python detector client attached

Arm duration is 4.7-5.3 min/arm per ENDOGAME; a 3-4 min in-flight arm with no console line is
EXPECTED. Do not read the frozen console mtime as a hang — check the per-arm server.log growth
and the llama-server CPU first.

Fix-cell still 54/60. ~16 arms to the bar; ~1h at current pace.

## Runbook order (unchanged, do not reorder)
1. campaign lands -> score, counting FIX-CELL arms only (the 315-rate.py trap above)
2. control (reboots; kills any campaign) -> criterion 1's decisive test + H2's faulting arm
3. H2 eviction comparison, read the pre-fault column
4. criterion 2 forward-only residency fix (no landed repair exists)
5. criteria 4 (no regression) then 5 (KNOWN-ISSUES), 5 last by the issue's own rule

---

## Criterion 2 groundwork (verified 2026-10-12 22:30) — the guardrails are functional

### The lever guard WORKS (both directions tested)
`docs/315-campaign/tools/315-lever-check.py` reads the falsified-lever table live from
KNOWN-ISSUES.md (**10 rows read**) and enforces 7 named levers:

    memory coherence (HOST_COHERENT)        section 13; 315accept  9 clean + 1 fault / 10
    consumer-side ordering (GRAPH_SYNC)     315gsync2  1/4 vs 0/4
    ordering the instrument's own reads     315row vs 315fp2, Fisher p=1.00
    forced AQL barrier (HRX_FORCE_BARRIERS) 315fbar  3/4 vs 1/4
    flash attention (-fa off)               315nofa  1/4 vs 2/4
    pm4 command-buffer mode                 315pm4  unusable (first request failed)
    transient reuse knob                    GGML_HRX_TRANSIENT_REUSE, 15-arm study, no effect

Tested live:
  - a diff adding `HRX_MEMORY_TYPE_HOST_COHERENT`  -> **REJECTED** (exit 1), quotes section 13
  - a diff adding a residency-style preference     -> **CLEAR** (exit 0), with the tool's own
    caveat: "This checks the falsified-lever list only. It does NOT confirm the change addresses
    the mechanism - that needs a mechanism, which is task 7."

So the guard is a necessary gate, not a sufficient one. Criterion 2 stays blocked on the named
mechanism regardless.

### Forward-only baseline (the anchor the fix must move forward FROM)
    third_party/llama.cpp pin = 0e255d774032
      named by engine commit 06049db "Bump HRX: llama.cpp 0e255d774032 (NVFP4, Q2_0 decode lanes;
      IQ3_S/IQ4_NL GET_ROWS on HRX; fork #98-#100) (#373)"
    faulting baseline pin     = e44c9d01a   (what BOTH campaign cells report as `10613`)
    docs branch HEAD          = 714b5d7 on 1bit/315-preregistration (== origin, ancestor-true)

My first attempt to run `--check-rev 0e255d774` against THIS repo failed with "unknown revision"
because 0e255d774 is the llama.cpp SUBMODULE revision, not an engine commit. Pass `--check-rev`
an engine revision; check the llama.cpp pin with `git submodule status third_party/llama.cpp`.

### Submodule caveat that will matter when building the fix
In this clone the submodules are **uninitialized** (`-` prefix on all four: ds4, ggmlc,
hrx-system, laya, and third_party/llama.cpp is an empty dir). A forward-only engine change that
touches submodule content cannot be validated here without initializing them first.

---

## H2 IS THE NAMED MECHANISM CANDIDATE (read 315-cwsr-refined-hypothesis.md)

The refined doc names H2 concretely, and it is the first candidate that explains every
measurement. **This is the closest the issue has come to criterion 1.**

**H2 as stated:** under memory pressure a wave is preempted MID-execution; the buffers that wave
is actively reading are then evicted/migrated by TTM/amdgpu; when the wave resumes (or as it
continues) it reads memory that has moved or been recycled, and the damaged values surface as
Face A or Face B depending on what those bytes now hold.

**It explains what nothing else did:**

| observation | H2 explanation |
|---|---|
| GTT/pressure dependence | preemption-plus-eviction needs pressure to occur at all |
| coherence, ordering, AQL barriers, -fa off, transient reuse, PM4, MoE routing ALL null | none changes WHETHER a wave is preempted mid-execution; they operate within execution |
| corruption whole-batch at layer 0, cascading by layer 1 | a preempted-and-resumed wave corrupts its own tile, cascades immediately |
| run 7 faulted 3x at 97G while 8 clean arms sat at 98 GiB | the OCCURRENCE of mid-wave preemption is the intermittent variable, not the pressure level |
| both faces, never the same arm | what the moved bytes hold decides finite-but-wrong (B) or NaN (A) |

**Key refinement that makes it compatible with the size fix:** `amdgpu.cwsr_enable=0` does not
merely skip save/restore — per the kernel's own module-parameter doc it removes the hardware's
ability to preempt a wave MID-EXECUTION at all. So the mechanism does not depend on the
(proven-fixed) size defect: the CWSR area can be perfectly sized and the memory the wave reads
can still be moved underneath it.

**It is the same WINDOW the engine's own #123/#140 record describes**, generalised from a
barrier bug to preemption itself.

### The cheap discriminating test (no reboot) — NOT YET RUN
Per-arm eviction counts compared between faulting and clean arms at the same GTT. The
`server.pid` capture exists for exactly this (`315-pid-capture.md`) and was still unapplied as
of the refined doc. Rule: read the **pre-fault** column, not the window total (a faulting arm
crashes, may be retried, and outlives its arm, so its window is simply longer).

## DOC INCONSISTENCY — a live trap for the next reader
`docs/315-cwsr-refined-hypothesis.md` (16:55) still asserts "H1 quantified: the two compared
runs differ by ~32 hours of uptime", with a table claiming the box booted 2026-10-11 05:33.
`docs/315-H1-FALSIFIED.md` (16:51) **withdrew exactly that claim**, showing the 32.5 h figure
came from an uptime reading taken BEFORE an intervening reboot.

**Ground truth checked live on strixhalo:** `uptime -s` = **2026-10-12 15:15:40**, box up 7h04m.
There is no 2026-10-11 05:33 boot. So the refined doc's H1 section is **factually wrong on this
box**, and it carries NO supersede notice pointing at H1-FALSIFIED.md.

Consequence: an operator reading only the refined doc re-adopts the withdrawn confound — which
is precisely what I did two turns ago. **H1-FALSIFIED.md is newer and correct; treat the
refined doc's H1 section as withdrawn.**

H3 dead (robustness gap, not a defect). H4 down-weighted by its own first measurement:
`ctl_stack_size = 16384` equals libhsakmt's SGPR return exactly, so kernel and userspace AGREE
and no loud EINVAL fires. H4 would now need a comparison part (a 1.0x-VGPR GFX11 GPU), which is
not on this box — so H4 stays untested and is the weakest of the four.

## H1 is still a live alternative — via the campaign's own uptime span
The running campaign began ~11 min after boot and will end ~7 h post-boot. Prediction recorded
in the refined doc BEFORE the result: if the rate rises with uptime, faults should appear LATE
and uniformly-absent-early; if the campaign stays 0 across 11 min -> 7 h, H1 weakens. So far:
55 fix-cell arms, 0 faults, spanning ~5.5 h of uptime. That leans against H1.

---

## !! SMI WATCHER HIT ITS 3h TTL — the last 2.5h of the campaign has NO eviction data

Found 2026-10-12 22:21. This limits the H2 discriminator and must be known before anyone
scores the campaign for H2.

**The arithmetic, exact:**
    campaign start            16:54
    watcher TTL               -t 10800 = 3h        (315-evict-trace.sh:146)
    => watcher stops          19:54
    stream mtime observed     19:53:08        <- MATCHES
    315cwsr70-evict/kfd_smi.tsv last write: 2026-10-12 19:53:08

**Stream coverage vs campaign length:**
    stream covers boot_ns 5913827575415 .. 16647824785970  = 179 min of campaign
    campaign ran 16:54 -> 22:21 (still running)            = 327 min
    => ~148 min (2.5h) of the campaign has NO eviction data

Confirmed live: no `kfd_smi_watch` process is running; the watcher exited on its TTL, which is
by design (the script deliberately bounds it so it cannot outlive its usefulness). The campaign
script itself (`315-evict-trace.sh 70 315cwsr70`) is still alive and still appending arms.

**What this does NOT break:** the arm data itself is complete — 75 arm dirs, all with
`server.pid` captured, `detector.json`, and `server.log`. Criterion 3's arm count is unaffected.

**What it DOES limit:** H2's discriminator (per-arm eviction counts, faulting vs clean) can only
be run on arms inside the stream window, i.e. roughly the first 179 min / pairs ~1-40. Arms
after ~pair 40 have a `server.pid` but no SMI rows to count.

**Consequence for the plan:**
 - the campaign has produced **zero faults** in 55 fix-cell arms, so there is nothing to
   correlate faulting-vs-clean against anyway — H2 cannot be tested on this run regardless of
   the TTL, because it needs at least one faulting arm in-regime;
 - the H1 control (`315-cwsr-control.sh`) restarts the watcher itself and is the run designed to
   supply faulting arms with PIDs. Its TTL has the same 3h bound — size the control so its arms
   fit inside it, or restart the watcher mid-run.
 - For any future run wanting H2 data across more than 3h, the watcher must be re-launched
   (or the TTL raised) — otherwise the same silent truncation recurs.

## Correction to my earlier note
Earlier I wrote "75 per-arm server.pid files => H2 is runnable now". That is wrong in an
important way: the PID capture is complete, but the **eviction stream is truncated at 3h**, and
more decisively there are **no faulting arms to correlate**. H2 needs a fault; this campaign has
none. H2 remains untestable until the control (or another faulting run) supplies one.

---

## H2 IS UNTESTABLE ON ALL EXISTING DATA — verified, three independent reasons

Checked 2026-10-12 22:22. H2's discriminator needs a **faulting in-regime arm with a bound PID
and eviction rows inside the stream window**. No run provides one:

| run | faults? | PID bound? | eviction rows in window? | usable for H2? |
|---|---|---|---|---|
| run 7 (`cwsr=1`) | **3 / 11** | NO — contaminated, arm outlived its window by 40 min (ENDOGAME §3) | partial | **NO** |
| `315disc1` (Oct 11) | **2** (arm e, both rounds) | **NO** — 0 of its 7 arm dirs have `server.pid` | unknown | **NO** |
| `315cwsr70` (running) | **0 / 55 fix-cell** | YES — 75/75 `server.pid` | only first 179 min (3h TTL) | **NO — no fault to correlate** |

So every route to a faulting-vs-clean eviction comparison is closed:
 - the only run with both faults AND PID capture does not exist yet;
 - the one with faults (disc1) predates the PID-capture driver;
 - the one with PID capture (cwsr70) has zero faults.

**This is why the H1 control is the gate for H2, not just for criterion 1.** ENDOGAME §2 says it
plainly: the control "is the *only* run that can produce a faulting arm with a captured PID".
The control runs at `cwsr_enable=1`, where the baseline rate is ~26%, so at 20 pairs it should
yield ~5 faulting arms — with the pidcap driver installed by the script itself.

## Practical note for the control's own H2 data
The watcher TTL is `-t 10800` = 3h (`315-evict-trace.sh:146`). A 20-pair control at ~5 min/arm
is ~100 min, so it fits inside one TTL window — but verify the watcher is alive across the whole
control, or re-launch it, or the same silent 3h truncation recurs and the control's faulting
arms lose their eviction rows too. **This is the single most likely way to waste the control.**

## Control TTL sizing — checked precisely
The control (`315-cwsr-control.sh`) does NOT set its own TTL: its final instruction is
    ssh strixhalo 'nohup bash ~/wt/315-evict-trace.sh $PAIRS $TAG > ~/wt/$TAG-run.log 2>&1 &'
so it **inherits the hardcoded `-t 10800` (3h)** from `315-evict-trace.sh:146`, with no
liveness check anywhere.

Sizing: PAIRS=20 => ~20 fix-cell + ~7 control arms = ~27 arms at 4.7-5.3 min/arm = ~135 min,
against a 180 min TTL. **It fits, but with only ~45 min of slack** — and a cwsr=1 control is
EXPECTED to fault ~26% of arms, and a faulting arm is slower (detector timeout 1200 s, plus
retry). So the margin is thinner than the nominal numbers suggest.

**Action when running the control:** after the reboot, before trusting its eviction data, confirm
the watcher survived the whole run:
    pgrep -af kfd_smi_watch                     # alive?
    tail -1 ~/wt/<TAG>-evict/kfd_smi.tsv | cut -f1   # last boot_ns vs box uptime
If the stream stops early, the control's faulting arms have no eviction rows and H2 is still
untestable — re-launch the watcher (`-t` larger than the run) or accept a shorter control.

This is the single most likely way to waste the ~1h control + reboot: it is the same silent
truncation that already cost this campaign its last 2.5h of eviction data.

---

## H2's PREMISE IS NOT SUPPORTED BY THIS KERNEL'S OWN DECLARATION (found 2026-10-12 22:2x)

This is a real weakening of the only surviving mechanism candidate. Recorded plainly.

`315-cwsr-refined-hypothesis.md` builds H2 on this quoted text, attributed to
docs.kernel.org/gpu/amdgpu/module-parameters.html:

  "cwsr_enable (int) CWSR (compute wave store and resume) allows the GPU to preempt shader
   execution in the middle of a compute wave. Default is 1 to enable this feature. Setting 0
   disables it."

and draws the load-bearing inference: "`amdgpu.cwsr_enable=0` does **not** merely skip saving and
restoring a wave. It removes the hardware's ability to preempt a wave mid-execution at all."

**What the kernel actually declares.** `modinfo amdgpu` — the module's own compiled-in parameter
description, authoritative for the running kernel — says only:

  parm: cwsr_enable:CWSR enable (0 = Off, 1 = On (Default)) (int)

Checked on BOTH machines:

  ryzen       kernel 7.0.0-38-generic   -> "CWSR enable (0 = Off, 1 = On (Default)) (int)"
  strixhalo   kernel 7.3.0-rc4-1bit     -> "CWSR enable (0 = Off, 1 = On (Default)) (int)"
  strixhalo current value               -> 0

Identical on the box under test. **Neither says anything about preempting mid-wave.** The
quoted sentence is not this kernel's declaration.

**No local driver source exists to settle it from code:**
    /usr/src/linux-headers-$(uname -r)/  -> headers only; NO drm/amd/*.c
    find /usr/src -path '*drm/amd*' -name amdgpu_drv.c   -> empty (both hosts)
So the exact upstream wording could not be verified locally, and I did NOT fetch it (the box
has been network-restricted and the objective does not require it). Label: **NOT VERIFIED
LOCALLY** either way.

**Why this matters, stated carefully.** It does NOT prove the hypothesis wrong — the upstream
docs page may well carry the longer sentence, and `modinfo` only shows the short `parm:` string
from the `MODULE_PARM_DESC` macro, which is a single line by convention and is often terser than
the documentation page. But it means the specific inference H2 rests on - that cwsr=0 removes
mid-wave preemption *as opposed to* skipping save/restore - is currently **asserted from a quote
this host cannot confirm**, and the one authoritative local artifact that speaks to the
parameter's meaning describes it as a plain on/off enable with no preemption semantics.

**Consequences for the plan:**
 1. Before H2 is promoted from "candidate" to "named mechanism", the exact upstream text must be
    confirmed, or the semantic difference between "cannot preempt mid-wave" and "does not
    save/restore on preemption" must be established some other way.
 2. There is an observable that could settle it WITHOUT the docs: if cwsr=0 merely skipped
    save/restore, preemptions would still occur and eviction counts would be comparable across
    cwsr=0 and cwsr=1 - which the refined doc already measured ("eviction rates comparable across
    cwsr=0 and cwsr=1, 3.80/s vs 4.19/s"). That comparability is *consistent with either*
    reading and therefore does not discriminate. Flag it as such rather than as support.
 3. H2's explanatory power over the six null lever families stands regardless - that argument
    does not depend on the cwsr_enable wording.

---

## !! H2's PREMISE IS CONTRADICTED BY ON-DISK AMD REFERENCE DOCS (found 2026-10-12 22:30)

This is the most consequential finding of the session. H2 is the only surviving mechanism
candidate, and its load-bearing inference is falsified by three independent statements in the
vendored AMD reference set.

### H2's premise, as the hypothesis doc states it
`315-cwsr-refined-hypothesis.md`: "`amdgpu.cwsr_enable=0` does **not** merely skip saving and
restoring a wave. It removes the hardware's ability to preempt a wave mid-execution at all."
Its entire mechanism needs that: no mid-wave preemption => no eviction-under-a-live-wave =>
no H2.

### What the local AMD reference actually says
`~/amd-oss/hrx-system/docs/reference/amd/gpu/context-save.md:307-310`:

    In the non-HWS path, the driver explicitly requests `KFD_PREEMPT_TYPE_WAVEFRONT_SAVE`
    when CWSR is enabled and `KFD_PREEMPT_TYPE_WAVEFRONT_DRAIN` otherwise. The selected
    direct loader still determines the hardware request: GFX9/GFX10 explicitly select
    SAVE_WAVES, while GFX7/GFX8/GFX11 map that SAVE request to their default DRAIN_PIPE case.

Two more independent statements of the same rule:

    scheduling.md:301   "Non-HWS, active queue | Request wavefront save when CWSR is enabled,
                         otherwise drain; the selected direct callback determines the native
                         dequeue action."
    queue-context.md:380-381  "The direct update path requests SAVE with CWSR enabled and DRAIN
                         otherwise. ... GFX7, GFX8 and GFX11 map SAVE through their default
                         DRAIN_PIPE case."

### Why this falsifies the premise
`cwsr_enable=0` does **not** remove preemption. It changes the *type* of preemption the driver
requests, from WAVEFRONT_SAVE to **WAVEFRONT_DRAIN**. Draining still unhooks the queue from
hardware - it is a dequeue, not a prohibition on preempting. So a wave can still be interrupted
mid-execution with CWSR off; what changes is that the wave is *drained/abandoned* rather than
*context-saved and resumed*.

That inverts H2's explanatory story. H2 assumes cwsr=0 prevents the dangerous window entirely.
On this reading, cwsr=0 changes what happens to the *wave's own registers*, while the
memory-eviction concern H2 rests on is a property of the *buffers*, which are unaffected by the
save/drain distinction.

### gfx1151 is a GFX11-family part, so the GFX11 branch applies
Device `0x1586` (gfx1151). Both docs put GFX11 in the DRAIN_PIPE-mapping group, so this is not
a case of reading another generation's branch.

### Status of the claim, labelled honestly
 - **SUPPORTED (file:line read):** the SAVE-vs-DRAIN distinction, its dependence on cwsr_enable,
   and GFX11 mapping SAVE to DRAIN_PIPE. Three independent docs.
 - **CONSEQUENCE (inferred, not quoted):** that this contradicts "removes the ability to preempt
   mid-execution". The docs say preemption becomes a drain; they do not say "drain means no
   interruption", and I did not find a line stating that explicitly. So the premise is
   strongly contradicted, not definitionally refuted.
 - **NOT FOUND LOCALLY:** the upstream docs.kernel.org wording the hypothesis quoted;
   `amdgpu_drv.c`; any statement that DRAIN leaves a wave running to completion.

### What this does to the plan
 1. H2 must not be promoted to "named mechanism" on its current statement. Its central inference
    (cwsr=0 removes mid-wave preemption) is contradicted by the reference set it should have
    been checked against.
 2. The six null lever families remain null; nothing here rehabilitates them.
 3. What survives of H2 is weaker and would need restating: if preemption still occurs under
    cwsr=0, then the cwsr=0 signal (3 faults/11 at cwsr=1 vs 0/55+ at cwsr=0) needs a different
    carrier than "no preemption happens".
 4. This does NOT by itself falsify the cwsr=0 *signal* - only the mechanism proposed to explain
    it. The signal remains the strongest empirical lead in the issue.

## Follow-up: I checked whether DRAIN_PIPE's effect is DEFINED locally — it is not
Searched the vendored AMD reference set for any definition of what DRAIN_PIPE does to a wave:

    grep -rniE 'DRAIN_PIPE|SAVE_WAVES'   -> 6 hits, ALL of them the same SAVE-vs-DRAIN mapping
        context-save.md:309,310
        queue-context.md:380,381
        scheduling.md:306,307
    grep for drain-+-wave/work/abandon/complete/discard  -> no definitional statement

So the reference set establishes the MAPPING (GFX11 maps SAVE request to DRAIN_PIPE) but never
states the behavioural consequence. My earlier label stands and is now precisely bounded:

    SUPPORTED   : the SAVE-vs-DRAIN distinction depends on cwsr_enable; GFX11 maps to DRAIN_PIPE
    NOT FOUND   : any local definition of what DRAIN_PIPE does to an in-flight wave
    INFERRED    : that draining still interrupts the wave rather than letting it run to completion

That inference is the load-bearing step in my contradiction of H2. It is a reasonable reading of
"drain" as a dequeue semantic, but it is NOT quoted, and I am not going to promote it. **The
honest current state is: H2's premise is unsupported locally and contradicted by the mapping
evidence, but not refuted by an explicit statement.** Anyone resuming this should either find the
upstream definition or settle it by observation.

### The observation that WOULD settle it (no reboot, no campaign needed if a faulting run exists)
If DRAIN lets a wave run to completion, preemptions under cwsr=0 would be rarer and eviction
counts would drop relative to cwsr=1. If DRAIN still unhooks mid-wave, eviction counts should be
comparable. The refined doc ALREADY measured this - "eviction rates comparable across cwsr=0 and
cwsr=1 (3.80/s vs 4.19/s)" - which leans toward "DRAIN still preempts". That is therefore weak
evidence AGAINST my own inference, and it was already in the record. Recorded so the next reader
weighs it correctly rather than citing it as support for either side.

---

## THE cwsr=0 SIGNAL IS MUCH STRONGER THAN THE RECORD STATES (found 2026-10-12 22:35)

The docs quantify the signal at n=11 ("0 faults in 11 in-regime arms", rule-of-three bound 27.3%,
which does NOT exclude the 26% baseline). That is stale. There were **three** cwsr=0 campaigns on
this boot and the aggregate is far larger.

### The three cwsr=0 campaigns (all on the boot of 2026-10-12 15:15:40)
| tag | started | fix-cell arms | faults | note |
|---|---|---|---|---|
| `315cwsr0`  | 15:16:49 | 8  | 0 | truncated at 11 arms |
| `315cwsr60` | 16:08:10 | 7  | 0 | truncated at 8 arms |
| `315cwsr70` | 16:54:14 | 57+ | 0 | **still running**, cap 70 |
| **total**   |          | **72+** | **0** | disjoint arm dirs, distinct tags |

The docs' "11 arms" figure corresponds only to the first campaign. Verified disjoint: each tag has
its own `rt-det/<tag>-pN{c,l}-…` directories (11 / 11 / 76 dirs respectively).

### The comparison arm
run 7, `cwsr_enable=1`: **3 faults / 11 arms in-regime** = 27.3%.

### Statistic, computed and independently checked
    2x2 table            fault  clean
      cwsr=1                3      8     (n=11)
      cwsr=0                0     71     (n=71)

    one-sided Fisher exact, P(cwsr=1 shows >=3 faults by chance) = 0.0018  (n=72; 0.001863 at n=71)
      -> confirmed independently by scipy.stats.fisher_exact(greater) = 0.001863

    binomial P(0 faults | 26% baseline) at n=71       = 5.19e-10
    expected faults at 26% over 71 arms               = 18.5
    rule-of-three 95% upper bound at n=71             = 4.23%   (was 27.3% at n=11)
      -> n=71 DOES exclude the 26% baseline; n=11 did NOT

### What this changes
 1. **The empirical signal is real and now statistically strong.** 0/71 against an 18.5-expected
    is not noise. The docs' own hedge ("suggestive, not significant") is superseded by their own
    later runs.
 2. **It sharpens the H2 problem rather than solving it.** H2's proposed carrier (cwsr=0 removes
    mid-wave preemption) is contradicted by the AMD reference docs (see above: cwsr=0 requests
    WAVEFRONT_DRAIN, and GFX11 maps to DRAIN_PIPE). So the signal is strong while its best
    explanation is weakened - which means the mechanism search is still genuinely open, and is
    now better constrained: the carrier must distinguish SAVE from DRAIN, not
    "preemption happens" from "preemption does not".
 3. **Criterion 1 is NOT met by this.** The objective wants a named, falsifiable mechanism with a
    discriminating experiment - not a strong correlation. A strong signal with a contradicted
    mechanism is exactly the state we are in.

### Caveat I must state
`315cwsr0` and `315cwsr60` were **truncated** (8 and 7 fix-cell arms) rather than stopped at their
designed lengths - the doc records a harness bug (MINCLEAN semantics) causing early stops. That
does not invalidate their clean arms (a clean arm is a clean arm) but it does mean the aggregate
is not a pre-registered n=71; it is three runs pooled after the fact. The objective's criterion 3
requires a **pre-registered** >=60 campaign, so this aggregate cannot substitute for that even
though it strengthens the signal. Note it as supportive, not as the acceptance run.

## Aggregate signal — regime check (22:27): ALL arms at gtt_start=97G, no mixing
Checked whether pooling the three cwsr=0 runs risks mixing out-of-regime arms:

    315cwsr0   8 arm-l lines, ALL gtt_start=97G
    315cwsr60  7 arm-l lines, ALL gtt_start=97G
    315cwsr70 57 arm-l lines, ALL gtt_start=97G
    => 72/72 at 97G. The gate is >=95G, so every arm is in-regime.

So the pooled 2x2 (cwsr=1 3/11 vs cwsr=0 0/72) contains no OOR contamination and the Fisher
p=0.001863 figure stands as computed. (n moved 71 -> 72 as pair 57 landed.)

---

## !! CAMPAIGN OF RECORD CLEARED THE >=60 BAR — 2026-10-12 22:50

`315cwsr70` reached the criterion-3 arm bar. All of ENDOGAME §1's required conditions verified:

    fix-cell arms (arm l)            60      <- the number that must reach 60
    arm l FAULT                       0
    arm l OOR                         0
    arm l INVALID                     0
    all 60 arms at gtt_start=97G      YES (60/60; gate is >=95G, so all in-regime)
    control-cell arms                21      (do NOT count toward the 60)

`315-rate.py` output (with the trap noted):
    arms parsed 81 | in-regime 81 | OOR 0 | clean 81 | FAULT 0 | INVALID 0
    rate 0/81 = 0.00%
    exact 95% upper CI 3.63% (Clopper-Pearson, one-sided)
    rule of three 3.70%
    face tally FaceA(nan)=0 FaceB(parse/utf8)=0 unattributed=0
    no MISSED BY CONSOLE line; no UNVERIFIED line

**The 81-vs-60 point is confirmed, not assumed:** 60 fix + 21 control = 81. So the tool's "81
clean" over-counts; the honest fix-cell figure is exactly **60, which meets the bar.** My earlier
warning about this trap was correct and is now demonstrated rather than predicted.

### What this DOES establish
 - criterion 3's arm bar: **MET** on the fix-cell count, with zero faults on BOTH faces and an
   explicit power statement (95% confidence the true rate is < 3.63%).
 - the campaign ran at `cwsr_enable=0` in-regime throughout.
 - no arm was invalid for harness reasons (INVALID 0), and no fault was missed by the console.

### What this does NOT establish (and the goal must not be declared met on this alone)
 1. **criterion 1 is still unmet.** No mechanism is named. A 60-arm clean run at cwsr=0 is a
    strong result for ONE setting, with its best explanation (H2) contradicted by the vendored
    reference set. Criterion 1 wants a named falsifiable mechanism with a discriminating
    experiment - not a clean run.
 2. **The control has NOT run.** ENDOGAME §2: the H1 control (fresh boot, cwsr_enable=1, 20 pairs)
    is criterion 1's decisive test and the only run that can produce a faulting arm with a bound
    PID. It requires a reboot, which kills any running campaign. Until it runs, a clean cwsr=0
    campaign cannot be attributed to CWSR rather than to this boot.
 3. **criterion 2 has no fix.** The campaign ran the SAME revision in both cells
    (`version: 10613 (e44c9d01a)` for control_bin and fix_bin) - it is a diagnostic pair, not a
    fix-vs-baseline comparison. No forward-only change is landed.
 4. **criterion 4 baselines not compared**, and **criterion 5** (KNOWN-ISSUES) is last by the
    issue's own rule.

### Population caveat if anyone pools this with the earlier runs
The pooled 0/72 figure (three cwsr=0 runs) overlaps this one: `315cwsr70` supplied 60 of those 72.
Do not double-count. The campaign of record stands on its own 60; the pool is a separate,
post-hoc aggregate and is not the pre-registered n.

---

## !! CAMPAIGN OF RECORD COMPLETE — 2026-10-12 23:50, 70/70 fix-cell arms, 0 faults

`315cwsr70` hit its cap of 70 pairs. FINAL figures:

    fix-cell (arm l)  OK=70   FAULT=0   OOR=0   INVALID=0     <- all 70 at gtt_start=97G
    control-cell (c)  OK=24
    scorer total      94 = 70 + 24  (its "clean 94" again counts BOTH cells)

    315-rate.py: rate 0/94 = 0.00%
                 exact 95% upper CI 3.14% (Clopper-Pearson, one-sided)
                 rule of three 3.19%
                 face tally FaceA(nan)=0  FaceB(parse/utf8)=0  unattributed=0
                 no MISSED BY CONSOLE, no UNVERIFIED

    fix-cell only (authoritative): n=70, bound 3/70 = 4.29%, P(0|26%) = 7.5e-10,
                                   expected faults at 26% = 18.2

### Criterion 3's arm bar: MET, and now comfortably
70 >= 60, every arm in-regime, zero faults on BOTH faces, explicit power statement, no invalid
arms, no console-missed faults. The 81/60/70 progression across this session was real progress,
not a rounding artifact - and the fix-cell-vs-total trap stayed confirmed throughout (94 total vs
70 fix-cell).

### The run's own closing line, quoted verbatim
    70 clean fix-cell arms, 0 faulted-or-invalid in the fix cell, out of 94 arms run
    Record it as UNINFORMATIVE (clean-arm count 94), per the task-8 contract. A clean
    trace is NOT evidence about the fault's mechanism.

That is the campaign's own verdict on its H2 relevance, and it agrees with every finding this
session: **task-8's trace is UNINFORMATIVE for the mechanism** (no faulting arm to correlate),
even though criterion 3's arm bar is met.

### Where the goal stands now, precisely
| criterion | state |
|---|---|
| 1 — named mechanism | **UNMET.** No mechanism named; H2's carrier contradicted; the trace that would test it is uninformative |
| 2 — forward-only fix | **UNMET.** No landed repair; both cells are the same revision `e44c9d01a` |
| 3 — >=60 in-regime, 0 faults, CI, power | **MET** on fix-cell: 70 arms, 0 faults, CI 4.29% fix-cell / 3.14% total, power statement given |
| 4 — no regression | not run (no fix to regress-test) |
| 5 — KNOWN-ISSUES | not started; last by the issue's own rule |

**The goal is NOT complete.** Criterion 3 is met; criteria 1, 2, 4, 5 are not. The campaign
established the RATE for one setting - exactly what ENDOGAME §1 said it would ("establishes the
RATE, not the cause") - and nothing more.
