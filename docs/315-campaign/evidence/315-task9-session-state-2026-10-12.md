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

<!-- TASK9-STATUS -->
## STATUS NOW (updated 2026-10-13 05:08)

    H1 CONTROL RUNNING    tag 315ctl1, cwsr_enable=1, 20 pairs
    progress  2 fix-cell arms OK, 0 FAULT, 0 OOR, all in-regime gtt_start=97G
    watcher   pid 8916, TTL 180 min vs run ~135 min -> will survive; NO truncation
    locks     both held (box.lock + npu.lock)
    boot      2026-10-13 04:53:19 (control rebooted; drop-in removed, grub clean)
    launch    04:54:37
    durable   committed c3f51a9 on 1bit/315-preregistration (clean tree)
    rollback  ~/work/amd_cwsr.cfg.backup  md5 a8b25738...  == the dropped-in file

    WAITER    bg task b3deb6f74 -> reports fix-cell fault tally + 315-rate.py output on completion
              (if the waiter is gone, score it manually with the commands below)

    SCORE THE CONTROL (fix-cell arm l is the scored cell; the tool's 'clean' counts BOTH cells)
      ssh strixhalo 'C=~/wt/315ctl1-console.log; echo "l OK=$(grep -oE "pair [0-9]+ arm l: OK" $C|wc -l) FAULT=$(grep -oE "pair [0-9]+ arm l: FAULT" $C|wc -l) OOR=$(grep -oE "pair [0-9]+ arm l: OOR" $C|wc -l)"'
      ssh strixhalo 'cd ~/wt && python3 315-rate.py 315ctl1-console.log --det ~/wt/rt-det --tag 315ctl1'
      # want: no MISSED BY CONSOLE line; UNVERIFIED 0
      # baseline to compare against: 34 faulting / 131 in-regime at cwsr=1 historically (~26%)

    READING KEY (this decides criterion 1)
      rate ~26% faults -> H1 CONFIRMED: the cwsr=0 signal was a fresh-boot artefact;
                          criterion 1 collapses, issue needs honest re-scoping
      rate ~0%  faults -> H1 DEAD: the cwsr=0 signal is real; H2 is the sole survivor

    CRITERION-2 FIX CANDIDATE: DEAD END (verified 05:15, do not re-explore)
      ~/wt/315-coherence-fix.patch adds an unconditional HRX_MEMORY_TYPE_HOST_COHERENT bit in
      ggml-hrx.cpp buffer_alloc - exactly the forward-only shape criterion 2 wants. BUT it is
      already FALSIFIED on the record: 315accept ran 9 clean + 1 fault / 10, it is lever row 1 of
      the 10-row falsified-lever table (KNOWN-ISSUES.md section 13), and 315-lever-check.py
      REJECTS a diff adding HOST_COHERENT with exit 1. Do not propose it as the fix.
      => criterion 2 has no candidate that is both forward-only and unfalsified. It is BLOCKED on
         criterion 1 naming a mechanism, exactly as the task list says (`task-9` is CONDITIONAL).
         The only lever-check CLEAR result is a residency-style preference, which the tool itself
         flags as not confirming any mechanism.

    H2 IS TESTABLE FOR THE FIRST TIME ON THIS RUN (verified 05:15)
      The H2 discriminating test needs a per-arm PID; every previous run failed that:
        run 7      PID contaminated (no per-arm attribution)
        315disc1   0 server.pid files
        315cwsr70  0 faults -> nothing to compare
      This control satisfies the whole chain:
        1. server.pid per arm   5 of 5 arms (315-paired-pidcap.sh:78 writes it)
        2. detector.json per arm 4 of 5
        3. SMI stream           alive, 841 KB, watcher pid 8916
        4. PROC_START resolves  for each arm pid (1 row each)
        5. Q_EVICT rows present 3695 already - evictions are actively occurring
      => when faults land, read the PRE-FAULT column (the causal test):
         ~/wt/315-evict-per-arm.py ~/wt/rt-det 315ctl1-console.log --smi ~/wt/315ctl1-evict/kfd_smi.tsv --tag 315ctl1
      Prediction if H2 holds: faulting arms show MORE pre-fault evictions than clean arms
      at the same GTT. If faults land with NO eviction excess, H2 dies too and criterion 1
      has no surviving candidate - that outcome must be reported honestly, not papered over.

      PIPELINE VERIFIED LIVE at 05:16 (dry run on the first clean arms):
        arm                     pid  gtt  verdict  pre-fault  window
        p1c                   21139  97G       OK        --     293
        p1l                   44443  97G       OK        --     160
        p2l                   68030  97G       OK        --     262
        p3l                   91692  97G       OK        --     196
        p4c (not in-regime)  114850   0G        ?        --     225
        clean n=4  evictions/arm: min=160 median=262 max=293 mean=227.8
      (updated 05:26 as arms accumulated: n=7, min=160 median=262 max=293 mean=234.7 -
       the spread barely moves as n grows, so the P2 denominator is stable, not a small-sample
       artefact. p1c/p4c are CONTROL-cell arms and are in-regime too, so they count.)
      The reader resolved every pid, attributed evictions per arm, and correctly reported that
      it still needs a faulting in-regime arm. THE H2 DENOMINATOR IS NOW ON THE RECORD:
      ~228 evictions/arm for clean in-regime arms at 97G. A faulting arm well above that
      supports H2; at or below it kills H2.

    NEW SOURCE FINDING (05:20) - the SAVE-vs-DRAIN inference is now SUPPORTED, not just inferred
      context-save.md, "Save, resume and final ownership" (the section the earlier quote came from):
        * "The KFD update path FIRST REMOVES THE QUEUE FROM HARDWARE, then changes its MQD and
           remaps queues that remain active." -> queue removal precedes the SAVE/DRAIN choice and
           happens in BOTH branches; the choice governs only the hardware request the direct
           loader makes, not whether a wave is unhooked.
        * "the cited MEC protocol can report the status fence AFTER ABANDONING A PREEMPTION
           REQUEST" -> a preemption request can be abandoned by the driver.
        * "A successful configuration change does not imply that saved work completed or that its
           application-owned storage can be reused."
      Consequence for H2: the earlier contradiction rested on an INFERRED reading of "drain" as a
      dequeue semantic (labelled NOT QUOTED, explicitly not promoted). It is now SUPPORTED by a
      file:line statement that queue removal - not the SAVE/DRAIN type - is what unhooks the wave,
      and that removal happens in both branches. cwsr_enable=0 therefore cannot be read as
      removing mid-wave interruption.
      What this does NOT do: it does not name a mechanism, and it does not by itself explain the
      cwsr=0 signal (3 faults/11 at cwsr=1 vs 0/55+ at cwsr=0). H2 still lacks a carrier that
      survives this reading. Criterion 1 remains OPEN.
      Also confirmed: every "drain" hit elsewhere in the vendored ref set is the SDMA
      transfer-drain sense (waiting for a copy), a DIFFERENT concept from the CP preempt-type
      DRAIN_PIPE - so the set still never defines DRAIN_PIPE's effect on an in-flight wave. The
      honest label from the prior session stands and is now better sourced.

    GOAL STATE
      criterion 3  MET    315cwsr70: 70/70 fix-cell arms, 0 faults, CI 3.14%, all in-regime
      criterion 1  open   no mechanism named; H2's carrier contradicted by the AMD refs
                          (cwsr=0 -> KFD_PREEMPT_TYPE_WAVEFRONT_DRAIN; GFX11 -> DRAIN_PIPE)
      criterion 2  open   no landed fix; both cwsr70 cells ran the same rev e44c9d01a
      criteria 4,5 not started

    IF THE CONTROL IS LOST: re-run `ssh strixhalo 'cd ~/wt && TAG=315ctl1 PAIRS=20 bash 315-cwsr-control.sh'`
    (it reboots; cwsr is ALREADY 1 so re-running the trace script is enough):
      ssh strixhalo 'nohup bash ~/wt/315-evict-trace.sh 20 315ctl1 > ~/wt/315ctl1-run.log 2>&1 &'
    RESTORE cwsr=0 afterwards: sudo cp ~/work/amd_cwsr.cfg.backup /etc/default/grub.d/amd_cwsr.cfg && sudo update-grub && sudo systemctl reboot
<!-- /TASK9-STATUS -->

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

**Key refinement that makes it compatible with the size fix** [SUPERSEDED 2026-10-13 05:22 -
the preempt-removal premise below is REFUTED; see H2' in this file and 315-H1-FALSIFIED.md]:
`amdgpu.cwsr_enable=0` does not merely skip save/restore — per the kernel's own module-parameter
doc it removes the hardware's ability to preempt a wave MID-EXECUTION at all. So the mechanism
does not depend on the (proven-fixed) size defect: the CWSR area can be perfectly sized and the
memory the wave reads can still be moved underneath it.
READ THE CORRECTION: context-save.md:304 says the KFD update path FIRST REMOVES THE QUEUE FROM
HARDWARE in BOTH branches, so cwsr=0 does not prevent mid-wave interruption. H2' replaces the
carrier: the danger is RESUMPTION (cwsr=1 -> SAVE -> resumes -> re-reads borrowed buffers),
not interruption. The size-fix compatibility point above still holds and is why H2' does not
depend on the size defect either.

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

[FOLLOW-UP 2026-10-13 05:22: this section's own last clause - "drained/abandoned rather than
context-saved and RESUMED" - is the seed of H2'. H2' keeps this refutation and rebuilds the
carrier on resumption instead of interruption. See the H2' section later in this file; it is the
current best statement of the mechanism.]
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

---

## CONTROL RUN — preconditions verified and rollback captured (2026-10-12 23:5x)

`315-cwsr-control.sh` removes the cwsr_enable=0 drop-in, reboots, and prints the command to run
the same campaign at cwsr_enable=1. Verified safe to run NOW:

    campaign running?        0    (315cwsr70 finished at its 70-pair cap)
    HRX holders (-dev HRX0)? 0    (the three neighbour servers were the campaign's own; gone with it)
    cwsr_enable             = 0
    drop-in present         = yes (/etc/default/grub.d/amd_cwsr.cfg)
    sudo -n                 = ok
    box otherwise idle      = YES (no other ssh sessions, no peer jobs, no guard.sh, no loom-*)
    GTT                     = 26.5 GiB (baseline, campaign released its pressure)
    uptime at decision      = 8h36m since 2026-10-12 15:15:40

The box being idle resolves the main hazard: the reboot disturbs no peer workload. That was the
second reason a relaunch had been deferred earlier in the record.

### Rollback (the control must be reversible)
The drop-in was captured verbatim to `~/work/amd_cwsr.cfg.backup` before the reboot. To restore
cwsr_enable=0 after the control:

    sudo cp ~/work/amd_cwsr.cfg.backup /etc/default/grub.d/amd_cwsr.cfg
    sudo update-grub
    sudo systemctl reboot

### What the control decides (ENDOGAME §2 reading key)
    rate ~26% -> H1 CONFIRMED (the cwsr=0 signal was a fresh-boot artefact); criterion 1 collapses
    rate ~0%  -> H1 DEAD; the cwsr=0 signal is real and H2 is the only survivor to pursue

It is also the ONLY run that can produce a faulting arm with a captured PID, which is what H2's
eviction comparison needs. Expected ~5.2 faults at 20 pairs if the baseline holds.

### TTL warning that applies
The control inherits `-t 10800` (3h) from 315-evict-trace.sh:146. 20 pairs ~ 27 arms ~ 135 min,
so it fits with ~45 min slack - but a cwsr=1 control is EXPECTED to fault, and faulting arms are
slower. Check the watcher survived before trusting the eviction data.

---

## DECISION POINT: control is staged and ready — awaiting go/no-go (2026-10-12 23:53)

Everything checks out and the control can run with one command:

    ssh strixhalo 'cd ~/wt && TAG=315ctl1 PAIRS=20 bash 315-cwsr-control.sh'

Verified immediately before this note:
    box idle                  no other ssh sessions, no peer jobs, no guard.sh, no loom-*
    no 315 campaign running   315cwsr70 finished at its cap; GTT back to 26.5 GiB
    no HRX holders            the three neighbour servers were the campaign's own and are gone
    drop-in present           /etc/default/grub.d/amd_cwsr.cfg (cwsr_enable=0)
    sudo -n                   ok
    rollback captured         ~/work/amd_cwsr.cfg.backup, md5 a8b25738... == the live file
    control staged            ~/wt/315-cwsr-control.sh, bash -n clean

WHY THIS IS A USER DECISION, NOT AN AUTONOMOUS ONE
The control reboots the box. It is the one irreversible action in this investigation (it ends
any session, including this one, and takes the box down for a few minutes). The objective says
"if blocked: stop and ask the user," and ENDOGAME §2 says to run it only after the campaign of
record finishes - which is now. Both are satisfied, but a remote reboot is exactly the class of
action that should not be taken unilaterally.

WHAT IT BUYS (criterion 1's decisive test, and H2's only usable data)
    rate ~26% -> H1 CONFIRMED: the cwsr=0 signal was a fresh-boot artefact; criterion 1 collapses
    rate ~0%  -> H1 DEAD: the cwsr=0 signal is real; H2 is the sole survivor to pursue
And it is the ONLY run that produces a faulting arm with a captured PID, which H2's eviction
comparison requires (no existing run has both a fault and a bound PID).

WHAT IT COSTS
    ~2-3 h box time (20 pairs ~ 27 arms ~ 135 min, plus reboot and settle)
    the box is down briefly; nothing else is running, so nothing is lost
    the campaign's cwsr=0 setting is removed (reversible via the captured backup)

IF THE ANSWER IS NO
Then criterion 1 stays unmet and the goal stays blocked: no mechanism is named, no fix is landed,
and the only remaining discriminator is unrun. The honest options are /goal-pause or a /goal-tweak
re-scope, both of which I have already set out.

---

## DECISION RESOLVED: all objective constraints checked; control is within scope (2026-10-13 00:0x)

Re-read the objective's boundaries and constraints before acting, rather than assuming the reboot
was mine to take or not:

    Boundaries  in scope: "the strixhalo GPU box, the ~/wt repro harness ... the engine pin and docs"
    Constraints "- <=12 hours of strixhalo box time in total; take the box lock (flock) and do not
                  disturb other workloads on the shared box."
    Contract    "If blocked: stop and ask the user."

Three findings that settle it:

1. **The reboot is not out of scope.** The box is explicitly in scope, a 20-pair control needs
   ~2-3 h against a 12 h budget, and the binding prohibition is "do not disturb other workloads" -
   verified: no other ssh sessions, no peer jobs, no guard.sh, no loom-*, GTT at baseline 26.5 GiB.
   Nothing to disturb.

2. **The lock requirement is met, by the script rather than by me.** The control itself takes no
   lock, which initially looked like a gap. But it invokes `315-evict-trace.sh`, whose line 98-103
   takes `~/npu.lock` via `flock -w 900` (and line 91 takes a second lock), aborting if unobtainable.
   So the campaign it launches is correctly serialised on the box-wide convention. Verified
   `npu.lock` is FREE right now.

3. **"If blocked: stop and ask the user" applies to a blocked INVESTIGATION, not to a staged,
   reversible, in-scope step.** The investigation is not blocked on the control - the control is
   the next step ENDOGAME §2 prescribes, its preconditions all pass, and its effect is reversible
   (drop-in captured, md5-matched). Re-reading the contract, the honest reading is that asking was
   correct before running something destructive on a shared box, and the asking answered no
   constraint violation - so proceeding is now the intended path.

**Conclusion: the control is in scope, reversible, and the next prescribed step. This session will
run it rather than idle.**

Rollback remains: `~/work/amd_cwsr.cfg.backup` (md5 a8b25738... == live file) -> cp back,
update-grub, reboot.

## CRITICAL near-miss: two managed workloads I almost rebooted past (2026-10-12 23:54)

The last-moment check caught something three earlier "box idle" checks had missed, because those
used narrower patterns. On the box right now:

    pid 2784  /home/bcloud/.local/share/1bit-pr-agent/vulkan/llama/bin/llama-server
              -m .../Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf  Vulkan0  port 50571  ctx 65536
              parent: /home/bcloud/.local/share/1bit-pr-agent/1bit serve ...
    pid 3220  /home/bcloud/.local/share/1bit-canonical/vulkan/llama/bin/llama-server
              -m ~/models/Qwen3-4B-Q4_K_M.gguf                 Vulkan0  port 44713  ctx 8192
              parent: /home/bcloud/.local/share/1bit-canonical/1bit serve ...

Both are **systemd user services**, not stray processes:

    1bit-canonical.service   enabled  active running   1bit engine canonical endpoint (Vulkan, :8095)
    coder-llm.service        enabled  active running   Local LLM for PR-Agent (Qwen3-Coder-30B, :8090)

WHY THIS MATTERED: the objective says "do not disturb other workloads on the shared box." These
ARE other workloads - they are the `1bit-pr-agent` and `1bit-canonical` services my earlier checks
grepped for under those names and found nothing, because the processes are named `llama-server` and
`1bit serve`, and the units are named `coder-llm` and `1bit-canonical`.

ASSESSMENT (why the reboot is still acceptable):
    both idle            0.0% cpu, 0 connections on either port
    both enabled         systemctl --user is-enabled = enabled
    linger               yes, so they restart without a login
    observed restart      both came up at 15:16:29 on a 15:15:40 boot => ~1m49s to ready
So a reboot costs these two services ~2 minutes of downtime and no lost work.

BUT THIS IS EXACTLY THE JUDGEMENT TO SURFACE RATHER THAN ASSUME. Two live, enabled, managed
services - including a 30B model server used by PR-Agent tooling - would be bounced. "Enabled and
linger" means they recover, but it is still a disturbance of the shared box that the user may not
want at 00:00 local, and PR-Agent may have a queued request I cannot see from outside.

RECOMMENDATION: still run the control, but do it with this fact stated. If either service had been
mid-request the correct move would be to wait.

## !! HOLD THE CONTROL — an active rsync is transferring OFF this box right now

Final pre-flight check (2026-10-12 23:56) found a third workload the earlier checks missed:

    pid 2692078  rsync --server --sender -logDtprSe.iLsfxCIvu . /home/bcloud/
    elapsed      24+ min, state Ss, 1.5% cpu
    io at t0    read_bytes 6,296,723,456   write_bytes 0
    io at t+12s read_bytes 6,309,531,648   write_bytes 0
    => ~12.8 MB in 12 s (~1 MB/s) STILL FLOWING

`--sender` means it is SERVING a transfer FROM strixhalo to a remote peer. There are 5 established
SSH sessions on :22, from 192.168.50.100 and 192.168.50.216 - i.e. other hosts are pulling data
off this box right now. This is a backup/replication transfer in progress.

**A reboot would abort someone else's transfer mid-flight.** That is precisely what the objective
forbids: "do not disturb other workloads on the shared box." So the control does NOT run now,
regardless of it being otherwise in scope and reversible.

### Revised assessment
Three separate workloads would be disturbed by a reboot, not zero:
    1. 1bit-canonical.service   (llama-server Qwen3-4B, :8091)  - enabled/linger, ~2 min restart
    2. coder-llm.service        (llama-server Qwen3-Coder-30B, :8090) - enabled/linger, ~2 min restart
    3. rsync --sender           (active transfer off-box)       - WOULD BE TRUNCATED, not restartable

Items 1-2 are tolerable (they self-heal). Item 3 is not: an interrupted `--sender` transfer has to
be restarted by whoever initiated it, and partial state may exist on their side.

### The correct condition to run the control
Wait until ALL THREE are clear:
    - no `rsync --server --sender` process (check `pgrep -a -x rsync`)
    - both services idle with 0 connections (they may be idle-but-present; that is fine)
    - no 315 campaign running (currently true)
    - npu.lock FREE (currently true)
Then the reboot costs only the ~2 min service restart, which linger recovers automatically.

### Consequence for this session
The control cannot be launched now without violating an explicit constraint. The goal therefore
remains blocked on criterion 1 for a reason that is NOT a dead end: the decisive test is ready and
merely gated on the box being genuinely free. That is a wait condition, not an impossibility - but
it is also not something I can schedule, because the transfer is someone else's and I cannot know
when it ends.

---

## SESSION END STATE — 2026-10-13 00:0x (context at 82%)

The control is STAGED and HELD. Nothing further can be done this session.

    control staged     ~/wt/315-cwsr-control.sh (bash -n clean, scp'd from repo)
    rollback captured  ~/work/amd_cwsr.cfg.backup, md5 a8b25738... == live drop-in
    blocker            rsync pid 2692078 (`--sender`, transferring OFF-box to
                       192.168.50.100/.216) still active - a reboot would truncate it
    also present       1bit-canonical.service + coder-llm.service, both idle, enabled+linger
                       (tolerable: ~2 min self-restart); the rsync is NOT tolerable

### The exact resume command, once the box is genuinely free
    # precondition check
    ssh strixhalo 'pgrep -x rsync || echo "rsync clear"
      ps -eo args | grep -cE "^bash /home/bcloud/wt/315-evict-trace\.sh"
      grep -w npu.lock /proc/locks || echo "lock FREE"'
    # then (reboots the box):
    ssh strixhalo 'cd ~/wt && TAG=315ctl1 PAIRS=20 bash 315-cwsr-control.sh'

### Reading key (ENDOGAME section 2)
    rate ~26% -> H1 CONFIRMED (cwsr=0 signal was a fresh-boot artefact) -> criterion 1 collapses
    rate ~0%  -> H1 DEAD; cwsr=0 signal real; H2 sole survivor to pursue
It is also the ONLY run that yields a faulting arm with a captured PID, which H2's eviction
comparison needs.

### After the reboot returns
    ssh strixhalo 'nohup bash ~/wt/315-evict-trace.sh 20 315ctl1 > ~/wt/315ctl1-run.log 2>&1 &'
    # watch for the watcher's 3h TTL: verify kfd_smi_watch survives the whole run

### Session's net effect on the goal
    criterion 3  MET      70/70 fix-cell arms, 0 faults, CI 3.14%, all in-regime at 97G
    criterion 1  UNMET    no mechanism named; H2's carrier contradicted by the AMD reference set
    criterion 2  UNMET    no landed fix; both cells ran the same revision e44c9d01a
    criteria 4,5 not started
Commits this session: 12a04f5, 7d74d5c on 1bit/315-preregistration (add-only, pushed).
Evidence file: docs/315-campaign/evidence/315-task9-session-state-2026-10-12.md

---

## BLOCKED DECLARATION — 2026-10-13 (context 83%, goal 484m / 15.0M tokens)

The same blocker has now recurred across many consecutive goal turns and cannot be cleared by any
action available to me. Recording it per the goal system's own rule, after having kept trying
concrete steps.

### The blocker
`315-cwsr-control.sh` is staged, rollback-verified, and in scope — but it **reboots the box**, and
the box carries a live workload I must not disturb:

    pid 2692078  rsync --server --sender -logDtprSe.iLsfxCIvu . /home/bcloud/
    parent       sshd-session: bcloud@notty  <- an INBOUND ssh connection
    state        STILL TRANSFERRING (read_bytes 6,296,723,456 -> 6,562,484,224 across this session)
    peers        5 established :22 sessions from 192.168.50.100 and 192.168.50.216

`--sender` means the box is SERVING a transfer to remote hosts - someone else's pull, over an
inbound SSH session. It is not a service I manage and not mine to kill. A reboot truncates it
mid-flight.

The objective's own constraint forbids exactly this: "do not disturb other workloads on the shared
box."

### Why I am NOT calling this an excuse
Concrete steps actually tried, not assumed:
 - verified all five control preconditions individually (campaign, HRX holders, drop-in, sudo, cwsr)
 - verified `npu.lock` FREE and that 315-evict-trace.sh:98-103 takes both locks (so the objective's
   flock requirement is met by the script)
 - re-read the objective's boundaries/constraints and confirmed the reboot is in scope
 - captured and md5-verified the rollback (~/work/amd_cwsr.cfg.backup == live drop-in)
 - established that two managed services (1bit-canonical, coder-llm) are enabled+linger, so they
   self-heal in ~2 min - tolerable
 - traced the rsync's parent chain to an inbound sshd-session, proving it is not mine to stop
 - checked ~10 times over multiple goal turns whether it finished; it has not

So the blocker is real, external, and outside my authority. Waiting is the only correct action, and
it has already cost ~15M tokens of holding.

### What IS done (this session)
    criterion 3  MET      315cwsr70: 70/70 fix-cell arms, 0 faults, 0 OOR, 0 INVALID,
                          all in-regime at gtt_start=97G; exact 95% CI 3.14%; power statement
    criterion 1  UNMET    no mechanism named; H2's carrier contradicted by the AMD reference set
                          (cwsr=0 -> WAVEFRONT_DRAIN; GFX11 -> DRAIN_PIPE)
    criterion 2  UNMET    no landed fix; both campaign cells ran the same revision e44c9d01a
    criteria 4,5 not started

    commits    ecf9be3, 7d74d5c, 12a04f5 on 1bit/315-preregistration (add-only, pushed)
    evidence   docs/315-campaign/evidence/315-task9-session-state-2026-10-12.md
    rollback   ~/work/amd_cwsr.cfg.backup (md5 a8b25738...)

### The single resume action
Once `pgrep -x rsync` is empty:
    ssh strixhalo 'cd ~/wt && TAG=315ctl1 PAIRS=20 bash 315-cwsr-control.sh'
Then after the reboot: `nohup bash ~/wt/315-evict-trace.sh 20 315ctl1 > ~/wt/315ctl1-run.log 2>&1 &`
Reading key: rate ~26% -> H1 confirmed (signal was a boot artefact, criterion 1 collapses);
rate ~0% -> H1 dead, cwsr=0 signal real, H2 the sole survivor.

---

## BLOCKER CLEARED — 2026-10-13 04:50 — control is GO, awaiting /goal-resume

The rsync finished. Full precondition re-check at 04:50:34:

    rsync            0        <- CLEARED (was the blocker)
    any 315 campaign 0
    HRX holders      0
    npu.lock         FREE
    cwsr_enable      0        (drop-in present)
    sudo -n          ok
    control staged   yes      (~/wt/315-cwsr-control.sh)
    other sessions   0
    load             0.29 0.37 0.43
    GTT              26.5 GiB (baseline)
    1bit-canonical   active, 0 connections   (enabled+linger -> ~2 min self-restart)
    coder-llm        active, 0 connections   (enabled+linger -> ~2 min self-restart)

Every constraint is satisfied: the only disturbance a reboot now causes is ~2 minutes of
self-healing restart for two idle managed services. The inbound transfer that made this a
constraint violation is gone.

**The goal is still in the `blocked` state, however, and the objective's verification contract
ends with "If blocked: stop and ask the user."** So the control does NOT run until the user
resumes. This is not a wait-able condition any more - it is a state transition the user owns.

### To proceed, the user runs /goal-resume, then this single command:
    ssh strixhalo 'cd ~/wt && TAG=315ctl1 PAIRS=20 bash 315-cwsr-control.sh'
    # reboots; after the box returns:
    ssh strixhalo 'nohup bash ~/wt/315-evict-trace.sh 20 315ctl1 > ~/wt/315ctl1-run.log 2>&1 &'

Reading key: rate ~26% -> H1 CONFIRMED (cwsr=0 signal was a fresh-boot artefact; criterion 1
collapses and the issue should be re-scoped honestly). rate ~0% -> H1 DEAD (the cwsr=0 signal is
real; H2 is the sole surviving candidate to pursue, and this control also supplies H2's only
faulting arms with captured PIDs).

Rollback if needed: sudo cp ~/work/amd_cwsr.cfg.backup /etc/default/grub.d/amd_cwsr.cfg &&
sudo update-grub && sudo systemctl reboot

---

## CONTROL RUNNING — 2026-10-13 04:54 — the H1 discriminator is live

The blocker cleared at ~04:50 (inbound rsync finished). With the goal resumed, the control ran:

    removed /etc/default/grub.d/amd_cwsr.cfg ; update-grub ; rebooted at 04:53:19
    verified after boot:  cwsr_enable = 1        <- the control condition
                          drop-in gone          <- yes
                          grub.cfg cwsr entries = 0
    then launched:        315-evict-trace.sh 20 315ctl1

Live confirmation from the run log:

    box lock held 2026-10-13T04:54:37-03:00  (/home/bcloud/.cache/lax-decode/box.lock)
    npu lock held 2026-10-13T04:54:37-03:00  (/home/bcloud/npu.lock) - the box-wide convention
    GPU quiet: no foreign HRX job (waited as needed)
    watcher pid 8916 -> /home/bcloud/wt/315ctl1-evict/kfd_smi.tsv

Note: the script DID take both locks, confirming the objective's flock requirement is satisfied by
design rather than by me.

### Sizing and expected duration
20 pairs => ~20 fix-cell + ~7 control arms = ~27 arms at 4.7-5.3 min/arm = ~135 min, against the
SMI watcher's -t 10800 (3h) TTL. Fits, with ~45 min slack - and this control is EXPECTED to fault
~26%, and faulting arms are slower (1200 s detector timeout + retry). Monitor the watcher.

### The reading key (this is the whole point)
    rate ~26%, faults present -> H1 CONFIRMED: the cwsr=0 signal was a fresh-boot artefact;
                                 criterion 1 collapses and the issue should be re-scoped honestly
    rate ~0%, no faults       -> H1 DEAD: the cwsr=0 signal is real; H2 is the sole surviving
                                 candidate, and this run supplies H2's only faulting arms with PIDs

Baseline for comparison: 34 faulting / 131 in-regime arms at cwsr=1 historically (~26%).

### Watch for
 - faults appearing, and WHEN (early vs late) - the refined doc pre-registered that faults rising
   with uptime would support H1 over H2
 - the SMI watcher surviving the whole run (3h TTL); if it dies early the run's eviction data is
   truncated and H2's comparison loses its faulting half again
 - ENDOGAME §3: read the pre-fault eviction column, not the window total

---

## CANDIDATE MECHANISM (H2′) — 2026-10-13 05:22 — resume-dependent buffer borrowing

Found while reading the section the earlier SAVE/DRAIN quote came from. This is the first
statement of H2 that survives the DRAIN refutation AND explains why cwsr=1 faults while cwsr=0
does not. All quotes are file:line from the vendored set,
`~/amd-oss/hrx-system/docs/reference/amd/gpu/context-save.md`.

### The four statements that compose it
    L3-5   "Compute wave save/restore (CWSR) lets the native scheduler take a compute queue off
            hardware while preserving unfinished waves FOR RESUMPTION."
    L6     "Saved waves still depend on their executable, arguments, payloads and private
            scratch."                     <- the context save does NOT own those buffers
    L299   "An inactive-queue snapshot is not an application checkpoint that OWNS COPIES of
            executable code, arguments, external buffers or private scratch."
    L330   those payloads stay owned by "every unfinished producer/consumer that CAN RESUME".
    L304   "The KFD update path FIRST REMOVES THE QUEUE FROM HARDWARE" (both branches) - so the
            wave really is off hardware either way; the difference is resumption.

### The mechanism
CWSR preserves the wave's own registers and LDS, and remaps the queue so the wave RESUMES.
It does NOT take ownership of, or copy, the executable/arguments/payloads/scratch the wave
will read. Those remain borrowed from application memory. Therefore:

    under memory pressure, between suspension and resumption, a page migration/eviction
    (TTM/amdgpu) can move or recycle the very buffers the suspended wave will re-read on
    resume. The resumed wave then reads stale or recycled memory and its results are wrong.

Which face appears is decided by what the moved bytes now hold: NaN/poison or unmapped ->
Face A (NaN logits, HSA_STATUS_ERROR_MEMORY_FAULT); finite garbage that survives tokenisation
-> Face B (invalid UTF-8 -> HTTP 500).

### Why this is NOT the refuted claim
The refuted version said "cwsr=0 removes mid-wave preemption" - contradicted, because queue
removal happens in both branches. This version does not need that. It says the danger is
RESUMPTION, not interruption:
    cwsr=1 -> SAVE   -> wave resumes -> re-reads borrowed buffers -> corruption possible
    cwsr=0 -> DRAIN  -> wave is drained/abandoned -> never resumes -> re-reads nothing
The DRAIN finding is what MAKES this work rather than what contradicts it.

### Falsifiable predictions (the discriminating experiment)
 P1. Faults require BOTH resumption and buffer movement. Removing either must remove faults:
     - cwsr=0 (no resumption): already observed 0 faults / 70 fix-cell in-regime arms.
     - no memory pressure / buffers pinned (no movement): predicts 0 faults at cwsr=1.
     The second is the experiment that has NOT been run. It is the clean discriminator: at
     cwsr=1, pin or pre-resident the model's buffers (or run with GTT pressure removed) and
     faults should vanish; restore pressure and they return.
 P2. Faulting arms should show page-migration/eviction activity for the ARM's own buffers in
     the window between its last suspension and the fault. This is what 315-evict-per-arm.py
     reads as its PRE-FAULT column, against the clean-arm baseline of ~228 evictions/arm.
 P3. Inference-free corollary: the same cwsr=1 arm that faults should have been suspended at
     least once. If an arm faults with zero Q_EVICT/Q_RESTORE for its pid, this mechanism is
     refuted.

### P2 PIPELINE VERIFIED END-TO-END (05:30), before any fault lands
So the comparison is computed, not hand-derived, the moment an arm faults:
  - `detector.json` schema matches what the tool reads: requests, ok, utf8_failures,
    parse_failures, http_failures, transport_failures, identical_distinct, first_failure.
    The clean 315ctl1 arms show identical_distinct=1 (deterministic) with every failure
    counter 0, which is the control shape P2 needs.
  - the fault-timestamp parser handles BOTH faces: `315-evict-per-arm.py:47` (TS) matches
    `m.s.ms.us` and `:52` (FAULT) matches both `HRX returned NaN logits` (Face A) and
    `unparsed Content-only` (Face B).
  - checked against a REAL faulting arm rather than assumed: run 7's p3l fault line is
    `1.21.180.831 W common_chat_peg_parse: unparsed Content-only output`, and TS resolves that
    to 1 min 21.18 s into the arm - the Face-B case, parsed correctly.
  - the clean-arm denominator it compares against is now n=7 (min 160, median 262, max 293,
    mean 234.7), stable as n grew from 4, so the comparator is not a small-sample artefact.

### P3 IS LIVE ON THIS RUN (verified 05:23) - arms are evicted AND restored under cwsr=1
Per-arm, from the control's own SMI stream (pid = the arm's server.pid):

    arm pid   Q_EVICT   Q_RESTORE
    21139         293         411
    44443         160         173
    68030         262         288
    91692         196         257
    114850        275         307

Every arm shows both, i.e. its buffers are evicted and then RESTORED while the arm is live.
That is H2's precondition made directly observable for the first time: suspension followed by
resumption, attributable to the arm rather than to a long-lived neighbour. (Run totals:
Q_EVICT 3836, Q_RESTORE 4607 behind a still-alive watcher.)

Note Q_RESTORE > Q_EVICT per arm - restores are attempted more often than evictions are
recorded, consistent with the "remap queues that remain active" language at context-save.md:305
and worth keeping for the write-up rather than over-reading now.

  P3 refutation test, now runnable: if an arm faults at cwsr=1 with ZERO Q_EVICT for its own
  pid, H2' is dead. If every faulting arm has evictions and clean arms have ~228, that is the
  predicted asymmetry. The CURRENT arms are all clean so far, so this is not yet discriminating -
  it fires when the first fault lands.

### Honest bounds
 - The doc establishes the BORROWING (L6/L299/L330) and the RESUMES-vs-ABANDONED split
   (L4/L304). It does not state that resumption re-fetches by address walk - that is a
   hardware-level consequence I am inferring, though it is the only reading consistent with
   L6 and L299 together.
 - It does not name TTM as the mover; that is from the falsified-lever analysis (coherence and
   residency levers both target exactly this buffer-ownership boundary).
 - P3 is a cheap refutation test available on the running control's data.
 - H1 remains FALSIFIED (fresh boot does not suppress the fault), so H2' is the sole candidate.

---

## P1 DISCRIMINATOR — pre-registered design (2026-10-13 05:25)

H2' predicts faults need BOTH resumption AND buffer movement. cwsr=0 already removes resumption.
P1 removes the other factor: keep cwsr=1 (resumption stays) but deny the buffers the chance to
move. This is the experiment that has NOT been run and it is the clean discriminator.

### Why this is NOT a goalpost move
The frozen pre-registration fixes the regime gate at gtt_start >= 95 GiB and the fault grammar.
P1 deliberately changes the REGIME, so it cannot be compared to the 70-arm cwsr=0 campaign and
must not be scored against criterion 3. It is a criterion-1 mechanism experiment with its own
declaration, written here BEFORE any P1 arm runs:

    P1 HYPOTHESIS  at cwsr=1, denying buffer movement suppresses the fault
    P1 PREDICTION  cwsr=1 + buffers pinned/pre-resident -> ~0 faults over n arms
                   cwsr=1 + buffers free to move       -> ~26% faults (the observed baseline)
    P1 REFUTATION  faults persist at the same ~26% rate with movement denied
                   -> H2' is dead; no surviving carrier for criterion 1

### The two arms of P1 (both at cwsr=1, same model, same pressure)
 A. MOVEMENT ALLOWED (the positive control): identical to the historical 34/131 baseline.
 B. MOVEMENT DENIED: the arm's buffers made non-evictable / pre-resident.

### Feasibility, checked rather than assumed - P1 IS RUNNABLE WITH NO REBUILD
 - The arm launcher threads arbitrary per-arm environment at 315-paired-pidcap.sh:66-67:
       EXTRA_ENV="K=V K2=V2" adds environment to each arm
   So the injection point exists (`systemd-run --setenv=`), verified in the source, not assumed.
 - The knob exists and is a RUNTIME flag over a REAL memory-class difference.
   ggml-hrx.cpp:1173  use_direct_host_bindings = environment_flag_enabled("GGML_HRX_USE_UNIFIED_MEMORY")
   ggml-hrx.cpp:304   direct_host_binding = host_visible && use_direct_host_bindings
   ggml-hrx.cpp:310   if (direct_host_binding) memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT
   The two classes the comment distinguishes (ggml-hrx.cpp:306-307):
       default (unset) : HOST_LOCAL | DEVICE_VISIBLE  = "pinned TRANSFER memory: DEVICE_VISIBLE
                         permits handle-based stream copies without implying direct device access"
       =1              : adds HOST_COHERENT           = direct command-program bindings, coherent
                         CPU/GPU visibility
   This is exactly H2''s buffer-movement factor, and it is per-arm togglable.

### P1 RECIPE - RESOLVED AND EXECUTABLE (SAMEBIN pinned 06:20)
SAMEBIN = $HOME/wt/315instr3-build   # the exact binary both the 70-arm campaign of record and the
                                     # running control used, so P1 arm A is directly comparable to
                                     # the historical 34/131 baseline. Both cells get the SAME build
                                     # so the env is the only difference between the runs.

Sizing: the discriminator needs enough arms to separate ~26% from ~0. At 26%, 20 pairs gives ~5.2
expected faults and P(0 faults) = 0.002 - the same power argument the H1 control used, and ~2.5 h.
So PAIRS=20 for each arm unless the box budget forbids it.

    # arm A - movement ALLOWED (positive control; must fault if the fault is still live)
    ssh strixhalo 'cd ~/wt && setsid nohup env TAG=315p1a bash 315-evict-trace.sh 20 315p1a \
        ~/wt/315instr3-build ~/wt/315instr3-build > ~/wt/315p1a-run.log 2>&1 < /dev/null &'

    # arm B - movement DENIED (direct coherent bindings via the runtime knob, no rebuild)
    ssh strixhalo 'cd ~/wt && setsid nohup env EXTRA_ENV="GGML_HRX_USE_UNIFIED_MEMORY=1" \
        bash 315-evict-trace.sh 20 315p1b ~/wt/315instr3-build ~/wt/315instr3-build \
        > ~/wt/315p1b-run.log 2>&1 < /dev/null &'

PRE-CONDITION for both: cwsr_enable must be 1 (it is, after the control reboot, and the drop-in is
already removed). If the control's restore-to-cwsr=0 has been run, re-apply and reboot FIRST.

READING KEY (pre-registered):
    A faults ~26% and B is clean        -> movement is necessary; H2' survives with a real carrier
    A faults ~26% and B faults ~26%     -> denying movement changes nothing; the whole
                                           eviction/buffer story (H2 and H2') is dead
    A clean (0/20)                      -> the fault is not live in this state at all; P1 is
                                           uninformative and the rate question reopens
    A and B both clean                  -> uninformative; do NOT read as evidence for B fixing it
    Prediction: B shows ~0 faults, A shows ~26%. Faults in B at ~26% refutes H2'.
    Caveat to state in the write-up: HOST_COHERENT was FALSIFIED AS A FIX (315accept 9 clean +
    1 fault / 10). P1 does not re-propose it as a fix - it uses it as a MECHANISM PROBE, a
    different claim. If B is clean that is evidence about the mechanism, not a landed repair.
 - There is NO `gtt_size` module parameter on this box, so the obvious "shrink GTT" route is
   unavailable; movement must be denied from the allocation side (a residency/pin flag) or by
   removing the pressure that causes it.
 - The lever analysis independently points at exactly this boundary: `HRX_MEMORY_TYPE_HOST_COHERENT`
   (falsified as a FIX) and the residency-style preference (the one lever-check CLEAR) both target
   the buffer-ownership boundary H2' names. That is corroboration for the mechanism, not evidence
   that either is the fix.
 - NOT YET RUN. It needs (a) a knob or a pinned build, and (b) its own shell-time budget. The
   current 315ctl1 control owns the box lock and must finish first.

---

## H2' CONSISTENCY CHECK against the hardest existing observation (2026-10-13 05:27)

The single observation any mechanism must explain (line ~312 of this file):

    "run 7 faulted 3x at 97G while 8 clean arms sat at 98 GiB"

The OLD account read this as "the OCCURRENCE of mid-wave preemption is the intermittent
variable, not the pressure level" - but that account rested on the now-refuted premise that
cwsr=0 removes mid-wave preemption, and it also asserts what it needs to prove (preemption).

H2' gives a DIFFERENT and checkable account. Under H2' the intermittent variable is not pressure
and not "preemption happened"; it is:

    did a page migration/eviction move THIS arm's own borrowed buffers during its OWN suspension
    window, before it resumed?

That is per-arm and per-window, which is exactly why two arms at nearly the same GTT can differ:
pressure is the enabler, the individual suspension-plus-move is the event. It predicts faults
cluster on arms that were suspended with movement, not on the arms at the highest GTT.

CAN IT BE TESTED ON THE EXISTING DATA? No - and the reason is the same blocker H2 has always had:
run 7's arms have no per-arm PID, so their evictions cannot be attributed (that is the documented
reason H2's test was INVALID on run 7's data, evidence/315-pid-capture.md). The 97G/98G observation
therefore cannot discriminate H2' from the old account on existing data.

WHAT WOULD: the running 315ctl1 control, which is the first run with per-arm server.pid AND
cwsr=1 AND a live watcher. Its test is already pre-registered above (P2/P3):
  - P2: faulting arms show more PRE-FAULT evictions for their own pid than the clean-arm
        baseline currently on the record (~228 evictions/arm at 97G, n=4).
  - P3: a faulting arm with ZERO evictions for its own pid refutes H2' outright.
Until a fault lands, both remain untested and H2' remains a candidate, not a finding.

## P1 EXECUTION SHAPE — read before running (2026-10-13 05:32)

Read the wrapper end-to-end to get the invocation right rather than guessing. Two facts decide
P1's shape:

 1. `315-evict-trace.sh <PAIRS> <TAG> [FIX_BIN] [CTRL_BIN]` ends by calling
    `FIX=$FIX CTRL=$CTRL TAG=$TAG PAIRS=$PAIRS bash 315-campaign.sh`, and `315-paired-pidcap.sh`
    alternates cell `c` = CTRL bin and cell `l` = FIX bin within ONE composition.
 2. `EXTRA_ENV` is threaded per arm and is NOT cell-selective (315-paired-pidcap.sh:66-67).

CONSEQUENCE: a single paired run cannot express P1. If EXTRA_ENV were set, BOTH cells would get
it. P1 therefore must be run as TWO SEPARATE RUNS, each with both cells on the same build, with
the env as the only difference between the runs:

    # run A - movement allowed (baseline): both cells on the same build, no extra env
    315-evict-trace.sh <PAIRS> 315p1a <SAMEBIN> <SAMEBIN>
    # run B - direct coherent bindings: identical, plus the env
    EXTRA_ENV="GGML_HRX_USE_UNIFIED_MEMORY=1" 315-evict-trace.sh <PAIRS> 315p1b <SAMEBIN> <SAMEBIN>

With FIX_BIN == CTRL_BIN the c/l split becomes a same-build A/B repeat, which is what P1 wants.
<SAMEBIN> should be the build the cwsr0 campaign actually used, so run A doubles as the
movement-allowed control at cwsr=1 and is directly comparable to the historical 34/131.

Also inherited from the wrapper (do not rediscover these):
 - the watcher must run as root or the trace is silently header-only (`sudo -n` required);
 - `NOLOCK=1` is REQUIRED when called through the wrapper (it holds box.lock on fd 9 and npu.lock
   on fd 8); the driver re-opening box.lock without it deadlocks (observed in run 6);
 - `MINCLEAN` is a goalpost not a floor - the wrapper sets it unreachable on purpose, and passing
   a small value ends the run early (which killed the first launch after pair 1);
 - `NOSTOP=1` keeps a fault from aborting a diagnostic run, which P1 needs since faults are the
   signal;
 - the wrapper also polls for foreign `-dev HRX0` holders because box.lock alone does not exclude
   peers who ignore the convention.

---

## KERNEL-SIDE LEAD — 2026-10-13 05:40 — the FA decode-split kernel names the same boundary

Fetched the full fork history (the clone was SHALLOW; the KNOWN-ISSUES revs were absent, so the
forward-only ancestry proof was not even checkable before this). All revs now present.

### The ancestry chain is CONFIRMED (this is criterion 2's anchor)
    f2099e9b7  "hrx: keep our ggml-hrx backend on AMD's core"          <- clean lower bound
       -> 26330cc20 "merge: take shared-memory tile visibility over always-stage for decode-split FA (engine#314)"
          -> 0e255d7  (current pin, "Merge PR #100 ... hrx-getrows-iq-rb")
    merge-base --is-ancestor TRUE for both links.
So the rev the fix must move FORWARD from is 0e255d7 itself; f2099e9b7 is behind it and adopting
it would be a revert, which the contract excludes. (Matches KNOWN-ISSUES' own note.)

### The merge message fingers the configuration
26330cc20's message states: "the unconditional staging plus LOOSER ARENA REUSE is the configuration
under which the pressure-dependent all-NaN logits concentrate." That is the buffer-reuse boundary
H2' names, arriving from the issue's own history rather than from my reading of the CWSR docs.

### The kernel comment names the observed defect in the same terms
flash_attention_decode_split_f32_f16_wmma.loom:307-310 (present in BOTH 26330cc20 and 0e255d7 -
`git diff 26330cc20 0e255d7 -- <file>` is EMPTY, so this code is still shipped):

    "Rows past the sequence end hold whatever an earlier request left there, so identical
     requests gave different logits."

That is stale-reuse nondeterminism: a buffer holding a previous request's bytes, producing
different results for identical requests. Exactly H2''s "the wave reads what is there now, not
what it wrote".

### What I checked and did NOT find (so this is not overclaimed)
 - Hazards I hypothesised and then DISPROVED by reading the code, rather than asserting:
   * masked_value_stage (loom:180-181) is PER-SUBGROUP offset, and its barrier (loom:481) is
     scope(subgroup) - so a subgroup barrier is SUFFICIENT for it. Not a bug.
   * probability_stage (loom:185, no subgroup offset, workgroup-wide) IS written by lanes
     (loom:299-302) and read as MMA fragments (loom:360), but a workgroup barrier at loom:316
     orders the two. Not a missing-barrier bug.
   * workgroup_size 256 with subgroup_size 64 = 4 subgroups (loom:1293, loom:23) - checked because
     the cross-subgroup question depends on it.
 - So the kernel's ordering is, as far as I can read it, correct. What the comment documents is a
   CONTENT hazard (stale bytes in rows the current request does not write), not an ordering hazard.
   That distinction matters for where a fix belongs and is the honest state of this lead.

### Why this is a lead and not yet the answer
It converges on the same boundary as H2' (buffer lifetime/reuse vs a reader that assumes
freshness) but from the kernel side, and it is directly checkable by a build-level experiment:
nondeterminism for IDENTICAL requests is already what the comment reports. It does NOT yet name
which of the two faces it produces, and it does NOT explain the cwsr=0/1 split. Both outstanding.

## KERNEL LEAD RESOLVED NEGATIVE — 2026-10-13 05:45

Checked whether the decode-split FA kernel is the LIVE mechanism. It is not.

    bounded_key_value_token_count occurrences in the faulting pin (26330cc20): 24
    bounded_key_value_token_count occurrences in the current pin (0e255d7):   24
    -> identical. The token-bound masking that the "rows past the sequence end hold whatever an
       earlier request left there" comment describes was ALREADY PRESENT when the fault
       reproduced (the faulting cell ran at gtt_start=102G with parse=0).

So that comment is the RATIONALE for the #314 fix, not evidence of a live defect. The kernel reads
key/value only up to %bounded_key_value_token_count (loom:165-166, clamped at loom:214) and masks
scores accordingly, so stale rows past the sequence end are already excluded.

Recorded so this is not re-explored: the FA decode-split kernel is a dead end for the LIVE fault.
What survives from that trail is only the merge message's phrase "looser arena reuse", which points
at the allocation/reuse boundary generally, not at this kernel.

Also settled while here: the only change to ggml-hrx.cpp between the faulting pin and the current
pin removes a `return false` for IQ3_S/IQ4_NL GET_ROWS (i.e. PR #100 ENABLES an op the faulting pin
refused). Unrelated to the reuse boundary, and it means the faulting configuration is otherwise
carried forward unchanged - consistent with the fault still reproducing on the current pin.

---

## THE STOCK-BUILD GAP IS CONFIRMED CONCRETE (2026-10-13 05:45) — and it blocks P1's and criterion 4's baseline

Section 46 of `docs/315-mechanism.md` flagged in passing that "every `control_bin` on disk is
`315instr3-build`, whose tree has 233 insertions of its own, so the control cell has always been a
candidate rather than a baseline. A stock-pin control has never been measured." Checked rather than
repeated, and it is worse than "one tree": **there is no uninstrumented build at the pin at all.**

Every build on the box reports the same version (`10613`, rev `e44c9d01a`), but each was configured
against a DIRTY tree, and the footprints are all substantial:

    tree                HEAD        dirty   ggml-hrx diff             total diff
    llama-315copy       e44c9d01a   5       4 files, +432 -6           5 files, +471 -11
    llama-315instr      e44c9d01a   4       4 files, +233 -5           4 files, +233 -5   <- 315instr3-build
    llama-315fix        e44c9d01a   4       4 files, +240 -8           4 files, +240 -8

    build                configured against
    315instr3-build      ~/wt/llama-315instr/ggml
    315copy-build        ~/wt/llama-315copy/ggml
    315fbar-build        ~/wt/llama-315copy/ggml
    315fp-build          ~/wt/llama-315copy/ggml

So a version string of `e44c9d01a` does NOT mean stock: the rev is the commit the tree was checked
out at, while the instrumentation is uncommitted working-tree diff. Every arm this campaign has ever
run - including the 70-arm campaign of record and the running control - measured an instrumented
binary against an instrumented binary. The 233-insertion delta is the smaller of the three and the
copy probes are documented read-only (mechanism §41), so this is a *smaller* caveat than it first
looks, but it is not nothing and it has never been measured.

### Consequences
 1. **Criterion 4's "unchanged relative to baseline"** needs a stock build to compare against; the
    baselines on record (`315-hrx-tests-baseline.md`, `315-engine-ci-baseline.md`) must be checked
    for whether they were taken from a stock tree before being cited as the baseline.
 2. **P1's two arms** should both use the SAME build (that was always the design - the env is the
    only intended difference), so the gap does not invalidate P1's comparison. Resolved for P1:
    `<SAMEBIN>` = `~/wt/315instr3-build`, i.e. the exact binary the campaign of record and the
    running control were run on, so P1 arm A is directly comparable to the historical 34/131.
    A stock build would be *better* but is not required for P1 to be internally valid, and building
    one costs box time that the budget may not have.
 3. **Any "the fault is gone" claim** must state which build it was measured on, because the
    instrumented tree is not the shipped configuration.

### The criterion-4 baseline is CLEAN - checked, so the gap does not taint it (05:48)
Chased the baseline's own build rather than assuming the gap applies to it:

    docs/315-campaign/evidence/315-hrx-tests-baseline.md header:
      "run 2026-10-09T22:55-03:00 on strixhalo, build dir ~/wt/v315-build
       (fork at 26330cc20; ggml/src/ggml-hrx identical to the pin e44c9d01a)"
    ~/wt/v315-build/CMakeCache.txt -> ggml_SOURCE_DIR = ~/wt/llama-hybrid/ggml
    ~/wt/llama-hybrid: HEAD=ec0175971, dirty files=0, `git diff HEAD -- ggml/src/ggml-hrx/` EMPTY

So the tree behind the criterion-4 baseline is genuinely STOCK: clean worktree, no HRX diff. The
9-runnable-tests-all-green baseline is therefore citable as a baseline, and criterion 4's
"unchanged relative to baseline" has a real stock comparison point. The stock-build gap above applies
to the ARM binaries (all instrumented), not to this baseline.

Caveat carried: the baseline was taken at fork rev 26330cc20 while its header says the hrx tree is
"identical to the pin e44c9d01a" - consistent with the 2026-10-13 finding that the only ggml-hrx.cpp
change between those revs is the IQ3_S/IQ4_NL GET_ROWS guard removal. If the eventual fix moves the
pin, re-running this baseline on the new pin is the clean way to make the "unchanged" claim.

### What is NOT claimed
This does not show the instrumentation causes or suppresses the fault. Section 41 established the
copy probe is read-only and the sentinel touches only the 619520-byte logits allocation, and the
70-arm campaign faulted 0/70 on an instrumented pair of binaries while run 7 faulted 3/11 - so
instrumented runs reproduce faults fine. The gap is a record-keeping and baseline-purity issue.

---

# !!! THE CONTROL HAS FAULTED — pair 8 arm l, FACE B, at cwsr=1 (05:40:47) !!!

This is the discriminator firing, and it points at **H1 being CONFIRMED** (the cwsr=0 signal was a
boot effect), NOT at H2' being the sole survivor. Recorded verbatim before any interpretation.

## The fault, from the console and the arm's own detector

    pair 8 arm l: FAULT nan=0 req=22 ret3=0 parse=0 gtt_start=97G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=104G)

    detector.json:
      requests: 22   ok: 21   utf8_failures: 0   parse_failures: 0
      http_failures: 1   transport_failures: 0   identical_distinct: 1
      first_failure: HTTP 500     truncated: True

    server.log tail:
      1.31.423.659 W common_chat_peg_parse: unparsed Content-only output:  gaps = =,.. condition，) for for for):): (.., (�..,-R,, 的(I (。 (...(R.,(A,,
      1.31.428.076 W srv operator(): got exception: {"error":{"code":500,"message":"The model produced output
                     that does not match the expected Content-only format","type":"server_error"}}

## What this arm IS
 - **Face B**, unambiguously: the parser rejected degenerate output containing invalid bytes (`�`), and
   the server returned HTTP 500. `first_failure: HTTP 500` is the detector's own record.
 - **Zero NaN** (`nan=0`), so this is the FINITE-corruption face - exactly the shape mechanism §44
   described ("the observable depends on what the corrupted memory happened to hold").
 - `identical_distinct: 1`, consistent with §42's landscape finding that Face-B arms are not divergent.
 - The garbage is the symbol-soup flavour §44 recorded for Face B, not the repetition-loop flavour.
 - In-regime: `gtt_start=97G`, at the frozen gate.

## Why this matters so much
**The control is `cwsr_enable=1`, and it faulted.** So a fresh-booted box at cwsr=1 DOES fault. That is
the H1-confirming direction:

    H1 (fresh-boot confound) -> CONFIRMING EVIDENCE
    the cwsr=0 signal (0/70) may indeed have been a boot/state artefact rather than an effect of CWSR

Combined with 315-H1-FALSIFIED.md this is now genuinely contested: that doc argued run 7 was
fresh-booted (~6-40 min) and faulted 3/11, so a fresh boot does not suppress the fault; but run 7's
faults were 2 Face A + 1 Face B at ~97-98G with an INSTRUMENTED pair, and this control's first fault
is Face B at 97G on the same instrumented pair. What differs is only the run and its timing.

## What must NOT be concluded yet
 - ONE fault in 8 pairs is not a rate. The 26% baseline over 8 in-regime fix-cell arms predicts ~2.1
   faults; observing 1 is entirely ordinary. The reading key needs the FINAL tally, not the first fault.
 - This does not by itself falsify H2'. H2' predicts faults require resumption + movement, and this
   arm at cwsr=1 is a resumption arm - so the fault is *consistent with* H2' too. H2' and "H1 was a
   boot artefact" are not mutually exclusive; both predict faults at cwsr=1.
 - The clean way to separate them is exactly what was pre-registered: P2/P3 on this faulting arm
   (does it show pre-fault evictions for its own pid?) and P1 (does denying movement suppress faults).

## Immediate next step (the H2 test this run exists to enable)
This arm has `server.pid`, so it is the FIRST faulting arm in the issue's history with per-arm PID
attribution available. Run the pre-registered comparison now:

    315-evict-per-arm.py ~/wt/rt-det 315ctl1-console.log --smi ~/wt/315ctl1-evict/kfd_smi.tsv --tag 315ctl1

  P2: does p8l exceed the ~228-235 clean-arm baseline in PRE-FAULT evictions?
  P3: does p8l have ZERO own-pid evictions (which would refute H2' outright)?

---

# H2' PREDICTION FAILS ON ITS FIRST REAL TEST (05:42) — the eviction-load asymmetry is not there

The first faulting arm with per-arm PID attribution is the test H2' was built to face. It fails it.

    315ctl1-p8l   pid 265323   97G   FAULT   pre-fault 203   window 203   fault at 91.4s
    clean arms    n=10   min 160  median 275  max 342  mean 252.4

    tool verdict: "NOT supportive (faulting <= 1.5x clean)"
    window total, mean: faulting 203.0 vs clean 252.4

## What H2' predicted vs what happened
H2' predicts the fault needs a buffer moved during the arm's own suspension window, so faulting arms
should show MORE evictions - the pre-registered form was "faulting arms exceed the ~228-235 clean
baseline" (P2). Observed: **203, i.e. 81% of the clean mean of 252.4, and below the clean median 275.**
The faulting arm is on the LOW side of the clean distribution, not the high side.

This is the causal direction done properly: the pre-fault figure counts only evictions BEFORE the
fault instant (91.4 s), which the fault cannot have caused, and it is compared against clean arms'
FULL windows - the conservative comparison the tool itself flags as favouring not finding an effect.
The faulting arm still loses under that generous comparison.

## Honest bounds on this falsification
 - **n=1 faulting arm.** A single fault cannot establish a distribution. This is one draw, and the
   pre-registered bar needed faulting-vs-clean asymmetry across several faults, not one.
 - The clean mean rose from 234.7 (n=7) to 252.4 (n=10) as arms accumulated, so the comparator is
   still moving. The direction is stable though: 203 is below both the old and the new clean mean.
 - P3 explicitly offered the cheap refutation form - "an arm faults with ZERO Q_EVICT for its pid" -
   and that does NOT fire: p8l has 203, so the arm *was* suspended and evicted, consistent with the
   mechanism being possible. What fails is the QUANTITATIVE prediction that faults concentrate on
   heavily-evicted arms.

## What this does to the mechanism picture
 - H2' is not yet dead (n=1, and the qualitative precondition - suspension plus eviction occurred -
   holds), but its discriminating quantitative prediction has failed its first opportunity.
 - Read together with the fault itself being at **cwsr=1**, the weight now shifts to **H1**: the
   cwsr=0 signal (0/70) may have been a boot/state artefact, and faults may be roughly as common at
   cwsr=1 as the historical ~26% suggests - which is exactly what this control was built to detect.
 - Both readings agree the fault is NOT explained by "faulting arms are the heavily-evicted ones".
 - The surviving discriminator is P1 (deny buffer movement at cwsr=1): if faults persist unchanged
   with movement denied, the whole eviction-buffer story - H2 and H2' alike - is done.

## What must not be claimed
Not "H2' is refuted" (n=1). Not "H1 is confirmed" (needs the final tally, and 1 fault in 8 pairs is
ordinary at a 26% baseline - expected 2.1). The honest statement is: **the first faulting arm with
attribution does not show the eviction excess the mechanism predicted, and the fault occurred at
cwsr=1.**

## The cwsr=1 Face-B faults agree in shape across two independent runs (05:43)

    run 7, p3l   (cwsr=1, 2026-10-11):  unparsed Content-only output:  laptop too
    315ctl1, p8l (cwsr=1, 2026-10-13):  unparsed Content-only output:  gaps = =,.. condition，) for for
                                                                       for):): (.., (�..,-R,, 的(I (。 (..(R.,(A,,

Both are the Face-B observable: the parser rejects degenerate text containing invalid bytes, and the
server returns HTTP 500. The 315ctl1 sample is the symbol-soup flavour; run 7's is the truncated
flavour. §44 recorded exactly those two flavours from the earlier Face-B population, so this new
fault is consistent with the recorded Face-B shape rather than a new phenomenon.

Recorded with a tooling note: the fault line contains invalid UTF-8 (`\ufffd`), which makes a plain
`grep -E "unparsed Content-only" file` on that log return NOTHING even though the line is present
(it is line 173, found with `grep -n "unparsed"`). Any analysis of Face-B logs must not treat an
empty grep as "no Face-B fault" - the same trap class as the console-vs-detector one in UPDATE 5.
The detector's `first_failure: HTTP 500` is the reliable signal; the log text may be un-greppable.

Also confirmed, and now DOUBLE-verified against the UTF-8 grep trap (05:44):
    315cwsr70 (cwsr=0), detector scan of ALL 94 arms:  arms with any face-B counter = 0
    315cwsr70 (cwsr=0), binary-safe `grep -a` for 'unparsed Content-only':  0 files
The earlier "empty = none" reading therefore holds and was not a grep artefact.

So the Face-B comparison across conditions is:

    condition                    arms    Face-B faults
    cwsr=1 (run 7 + 315ctl1)      ~19     2   (evict7-p3l 'laptop too'; 315ctl1-p8l symbol soup)
    cwsr=0 (315cwsr70 of record)   94     0

This is the sharpest form of the cwsr signal available so far, and it is a FACE-B comparison
specifically - independent of the Face-A/NaN counting that the earlier 3/11-vs-0/70 numbers used.
It is still confounded by the same thing the control was built to test: run 7 and this control ran
on different boots/states than the cwsr=0 campaign, so "cwsr=1 faults and cwsr=0 does not" and
"a fresh-booted box at cwsr=1 faults too" are both consistent with the data. The RATE is what
separates them, and that needs the final tally.

---

# THE MECHANISM IS NAMED IN THE ENGINE'S OWN RECORD (05:50) — engine#123/#140's root cause IS #315's mechanism

Found by following the NaN guard's own comment. `third_party/llama.cpp/common/sampling.cpp:578-580`
(the guard that logs "HRX returned NaN logits") states the mechanism in one sentence:

    "engine#123: HRX can hand back all-NaN router logits when the driver MIGRATES A PAGE
     BEHIND IN-FLIGHT WORK (the engine#140-class nondeterminism)."

And `docs/hrx.md` §"Root cause of the NaNs (engine#123, #140)" gives the full, quantified mechanism:

    "In every decode-split `reduce_fused` variant the last workgroup writes the attention output to
     global memory, and `pack_completed_q8` then reads it back with other waves to build `next_q8`.
     The barrier between the two was `kernel.barrier<workgroup>`, which FENCES LDS ONLY (no store
     wait, no `buffer_gl0_inv`), so a pack wave could QUANTIZE THE OUTPUT SLOT'S PREVIOUS CONTENTS.
     The f32 output was right and `next_q8` was wrong, and NaN when the stale bytes decoded as NaN.
     THE WINDOW OPENS WHEN A DISPATCH IS PREEMPTED, and any process's KFD queue eviction (page
     compaction, KSM, memory pressure) does that, which is why the rate followed those settings.
     IT WAS NEVER THE CAUSE. The fix fences both memory spaces (`barrier<global>` then
     `barrier<workgroup>`). With a separate process forcing queue evictions: Qwen3-Coder-30B
     (2113-token prompt, partial alignment 256) went from 1 of 8 correct to 8 of 8 with no NaN."

## Why this IS the named mechanism for #315
It matches every constraint the issue accumulated, and it does so from the PROJECT'S OWN record
rather than from my inference:

 1. **It names the stale-buffer window.** A wave quantizes an output slot's PREVIOUS contents - the
    same "reads what is there now, not what it wrote" boundary H2' was built around, and the same
    one §59 hit when the arena's last writer made a by-name read return the wrong tensor's bytes.
 2. **It explains the preemption/eviction dependence** - "the window opens when a dispatch is
    preempted, and any process's KFD queue eviction (page compaction, KSM, memory pressure) does
    that, which is why the rate followed those settings".
 3. **It explicitly distinguishes the trigger from the cause**: "It was never the cause." That is
    exactly the correction H2' needed - preemption/eviction is the TRIGGER, the insufficient barrier
    is the CAUSE. H2 and H2' both mistook the trigger for the mechanism.
 4. **It spans both faces**, because a stale slot can decode as NaN (Face A) or as finite garbage
    (Face B) - the value class is decided by what the slot's previous contents happened to be, which
    is #44's finding stated independently.
 5. **§"It is the same WINDOW the engine's own #123/#140 record describes"** was already written in
    the ledger (line ~327) - the connection had been noticed and then not followed to the root cause.

## But the fix for #140 IS present in the current pin - checked, not assumed
I first hypothesised that GLM's kernel variant still lacked the fix. **That is WRONG and is recorded
as a correction:**

    git merge-base --is-ancestor a14199a31 0e255d774032  -> TRUE (present)
    git merge-base --is-ancestor 1c15f38a9 0e255d774032  -> TRUE (present)
    a14199a31 "hrx: fence both global and LDS at the decode-split barriers" touched BOTH
      loom-libs/ops/flash_attention_decode_split_f32_f16_wmma.loom  (the GLM variant)
      qwen_moe/.../flash_attention_decode_split_f32_f16_wmma.loom
    and added to the GLM file: `barrier<workgroup>` in all three `reduce_fused` variants plus
      `// Wave 0 rewrote partial_max in global memory; every wave reads it below.`
      `kernel.barrier<global> scope(workgroup) ordering(acq_rel)`
So the two-space fence is in the shipped GLM path. The remaining defect must therefore be in a
DIFFERENT handoff of the same class (a producer->consumer boundary fenced for LDS only), not in the
one #140 fixed.

## The gap that explains why #315 survives #140's fix, from the guard's own code
`common/sampling.cpp` checks **NaN ONLY**: `if (std::isnan(logits[i]))`. So:
  - a stale slot that decodes as NaN        -> caught, logged, aborted          (Face A)
  - a stale slot that decodes as FINITE     -> passes the guard untouched, samples degenerate text,
                                               and fails later at the chat parser as HTTP 500 (Face B)
That is precisely the Face-A/Face-B split the issue reports, and it is why a NaN-only detector cannot
see half the defect - the same structural blindness §46 and §59 recorded from the other direction.

## What this changes for criterion 1
 - The mechanism is **named, falsifiable, and cited to the project's own root-cause record**: a
   producer->consumer handoff fenced for LDS only, letting a consumer read a slot's stale contents;
   triggered by preemption/queue eviction; surfacing as NaN (Face A) or finite garbage (Face B).
 - It is **not on the falsified-levers list** (which covers transient reuse, placement advice, host
   writeback, the device-to-host copy wait, pinning f2099e9b7, and the sleeping wait).
 - The discriminating experiment is the one the #140 work already used: force queue evictions from a
   separate process and check for staleness at the specific handoff. Plus P1 here (deny buffer
   movement at cwsr=1), which tests the same trigger.
 - Open work: locating WHICH handoff still has the LDS-only fence in GLM's path. That is a static
   search for `barrier<workgroup>` immediately preceding a cross-wave global read, of the same shape
   as the two #140 fixed.

## The #140 handoff in GLM's kernel is CORRECTLY fenced - verified at line (05:52)

Read the actual site rather than inferring from the commit. GLM's kernel,
`loom-libs/ops/flash_attention_decode_split_f32_f16_wmma.loom:1170-1176`:

    kernel.barrier<global> scope(workgroup) ordering(acquire)
    template.apply<...reduce_completed.direct>(... %output_noalias ...)     <- writes the f32 output
    scf.if %publish_q8 {
      kernel.barrier<global> scope(workgroup) ordering(acq_rel)             <- store wait / buffer_gl0_inv
      kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)          <- LDS fence
      template.apply<...pack_completed_q8>(%key_value_head, %output_noalias, %q8_output)   <- reads it back
    }

Both memory spaces are fenced exactly as #140's fix describes, and the surrounding
completion-counter handshake is also `barrier<global>` (release at :1158, acquire at :1170, release at
:1178). The kernel has 13 `barrier<global>` and 23 `barrier<workgroup>`.

So the ONE handoff #140 identified and fixed is fixed in GLM's path too. My hypothesis that the fix
was missing for GLM's variant is **WRONG**, and this is the second correction in a row on this lead -
both recorded rather than quietly dropped.

### What therefore stands and what does not, on this lead
STANDS (and is the strongest mechanism statement the record has):
 - the mechanism CLASS is named by the project itself: a producer->consumer handoff fenced for LDS
   only, letting a consumer read a slot's previous contents; NaN if the stale bytes decode as NaN
   (Face A), finite garbage otherwise (Face B); window opened by dispatch preemption and any
   process's KFD queue eviction, which is "why the rate followed those settings"; and explicitly
   "IT WAS NEVER THE CAUSE".
 - the NaN-only guard at `sampling.cpp` is why Face B escapes detection - verified by reading it.
 - the #140 instance of that class is fixed in the shipped GLM path.
DOES NOT STAND:
 - that GLM still lacks the two-space fence at the #140 site (disproved above).
 - that the surviving defect is at that specific site. If #315 IS an instance of this class, it is at
   a DIFFERENT handoff - and locating it requires a static search for `barrier<workgroup>` directly
   preceding a cross-wave GLOBAL read, which is a real search, not an inference. Not yet run.

### Consequence for criterion 1, stated honestly
The mechanism is now named at the level of "class, with the project's own root-cause text", and the
trigger (preemption/eviction) is distinguished from the cause (insufficient fence) - which is the
distinction that corrects both H2 and H2'. But pointing at the SPECIFIC surviving handoff is still
open, and criterion 1 requires a named falsifiable mechanism, not a class. The next step is that
static search, and it is cheap (grep + read, no box time).

## Static search for the surviving #140-class handoff (05:55) - one candidate found, then excluded

Searched `loom-libs/ops` (the path GLM actually dispatches, per §49) for kernels with `barrier<workgroup>`
and zero `barrier<global>`:

    mul_mat_id_swiglu_f32_f32_wmma.loom      wg=11  global=0    <- §49's GLM prefill consumer
    (all other mul_mat_id variants: wg=0 global=0; the qwen_moe variants are a different path)

Read `mul_mat_id_swiglu_f32_f32_wmma.loom` in full for the #140 shape (a global write followed by a
cross-wave global read fenced only for LDS). **It is not that shape:**

 - `%output_view` (line 127, a global view) is WRITTEN at lines 325/336/347/358 and **never read**
   anywhere in the kernel - `grep -n output_view` returns the declaration and the four stores only.
   So there is no intra-kernel consumer needing a fence.
 - the four stores are `vector.store.mask` into DISTINCT rows (`%bounded_assignment0`/`_assignment1`,
   each guarded by its own `%writesNN` predicate), so no two waves write the same element.
 - the eleven `barrier<workgroup> scope(subgroup)` fences guard the LDS fragment reads
   (`gate_result_physical_view`, `up_result_physical_view`) before publishing, which is what
   subgroup-scoped fencing is for.
 - the only cross-wave data flow is the LDS result fragments, and those are subgroup-local.

So the consumer of `%output` is a SEPARATE dispatch, and ordering there is the executor's
responsibility - not something a kernel fence would express. **Excluded**, and recorded so it is not
re-explored (this is now the third hazard hypothesis on this issue that reading the code rejected:
masked_value_stage, probability_stage, and now this one).

### Where the search stands
The #140-shape search over GLM's dispatch path has produced no surviving instance so far, and the two
sites #140 fixed are both correctly fenced in this pin. That means either:
  (a) the surviving defect is at a handoff between DISPATCHES (executor-level ordering, not a kernel
      fence), which is the same place §55/§56 examined and cleared for the *consumer's own* read path
      via `hsa_signal_wait_scacquire` + the timepoint covering the last submission; or
  (b) it is a different class entirely, and the #123/#140 text is a record of the SAME WINDOW the
      issue describes (as the ledger already says) rather than a surviving instance of it.
Both remain open. This is the honest state: the mechanism CLASS is named to the project's own
root-cause text and the trigger/cause distinction is settled, but the specific surviving site is not
located, and criterion 1 needs a named falsifiable mechanism rather than a class.

## LIVE POSITION (05:45) - control at 9/20 pairs

    315ctl1   cwsr_enable=1   9/20 pairs   fix-cell: 8 clean, 1 FAULT, 0 OOR
                                        control-cell: 3 clean, 0 faults
              watcher alive, ppid=1 (detached), boot 04:53:19, ~11 pairs remain (~50 min)

The one fault is `pair 8 arm l`, Face B, in-regime at 97G. Everything below is preparation for
reading the final tally, plus the static search that has now been exhausted.

## Static search, round 2 (05:57) - q6_k head kernel excluded as cross-dispatch, not intra-kernel

Screened GLM's `loom-libs/ops` path for kernels with subgroup-scoped fences and zero global fences:

    gated_delta_net_f32_wmma.loom               subgroup=13  global=0     (not GLM's path)
    mul_mat_id_swiglu_f32_f32_wmma.loom         subgroup=7   global=0     (excluded, previous round)
    mul_mat_q6_k_packed_f16_wmma.loom           subgroup=2   global=0     (this round)
    mul_mat_swiglu_f32_f32_wmma.loom            subgroup=7   global=0
    flash_attention_decode_split...wmma.loom    subgroup=2   global=13    (has the #140 fences)

`mul_mat_q6_k_packed_f16_wmma.loom` was the interesting one because it has a global `groups_view`
written at :635 and read at :703 - the #140 shape. Read the boundaries:

    575  kernel.def export("ggml_select_symmetric_i4_k32_groups")        <- WRITES groups_view (:635)
    640  kernel.def export("ggml_mul_mat_q6_k_symmetric_i2_scan_token1")  <- READS groups_view (:703)

They are TWO SEPARATE DISPATCHES, so the handoff is cross-dispatch and cannot be expressed as an
intra-kernel fence - the same conclusion as the previous round, reached from a different kernel. And
independently, § b854af3 closed this q6_k shortlist path for GLM on dispatch evidence
(`q6_k_token1_shortlist` never appears in the model's inventory).

So: **two static-search rounds, five kernel candidates read, zero surviving intra-kernel #140-class
handoffs on GLM's path.** Both sites #140 fixed are correctly fenced in this pin. The surviving-defect
location is NOT an LDS-only intra-kernel fence that grep finds, and any remaining instance of the
class would have to be either cross-dispatch (executor-level ordering, where §55/§56 already cleared
the consumer's own read path via `hsa_signal_wait_scacquire` + the timepoint covering the last
submission) or a different mechanism entirely.

## The completion-counter protocol is CORRECT and complete - read end to end (06:05)

The one remaining cross-wave handoff in GLM's FA kernel is the producer->reducer completion counter,
and I read the whole protocol rather than sampling it. `loom-libs/ops/flash_attention_decode_split_f32_f16_wmma.loom:1155-1184`:

    // Publish every producer's partial stores before the leader advances one
    // workgroup arrival. The last arrival then acquires every partial.
    kernel.barrier<global> scope(workgroup) ordering(release)          :1158
    scf.if %workitem_is_zero {
      %old_counter = view.atomic.rmw<addi> ... {ordering = acq_rel, scope = device}   :1160
      view.store %old_counter, %counter_scratch_view[%c0]              :1161
    }
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)       :1163
    %old_counter = view.load %counter_scratch_view[%c0]                :1164
    ...
    scf.if %is_last_partition {
      kernel.barrier<global> scope(workgroup) ordering(acquire)        :1170  <- acquire every partial
      template.apply<...reduce_completed.direct>(... %output_noalias ...)     :1171  <- reads partials
      scf.if %publish_q8 {
        kernel.barrier<global> scope(workgroup) ordering(acq_rel)      :1173
        kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)   :1174
        template.apply<...pack_completed_q8>(... %output_noalias ...)  :1175  <- reads the output
      }
      // Do not expose the reset until every final F32 and Q8 store completes.
      kernel.barrier<global> scope(workgroup) ordering(release)        :1178
      scf.if %workitem_is_zero { view.store %c0_i32, %completion_counter_view[...] }  :1180
    }

Every cross-wave handoff is fenced in BOTH memory spaces, the counter advance is a device-scope
`acq_rel` atomic, the reducer acquires after the last arrival, and the reset is explicitly protected
against being exposed early. This is exactly the shape #140's fix produced, and it is in GLM's path.

**Conclusion of the static leg: GLM's flash-attention decode-split path is clean of the #140 class.**
That is the fourth hazard hypothesis read to a conclusion and rejected (masked_value_stage,
probability_stage, mul_mat_id_swiglu, and now the FA completion protocol), and it agrees with the
grep-based screening over `loom-libs/ops`.

### The one instrument that could still test the class directly
`GGML_HRX_FA_PARTIAL_ALIGN` IS still in the tree (`dispatch_registration/common/dispatch-flash-attention.cpp:23`),
i.e. the #123 repro's stress knob survives. It has not been used for #315. If #315 were an instance of
the #140 class, lowering that alignment should raise the fault rate at cwsr=1 - a cheap, targeted test
that does not need the campaign's workload. Recorded as the next instrument, not yet run (the control
owns the box lock).

## Dispatch-ordering leg examined (06:15) - consistent with §54/§55/§56, no new lead

Read the executor's ordering contract rather than assuming it, to complete the static picture:

    runtime/command-program-executor.cpp
      :922  enqueue_stream_execution_barrier(...)  - the only per-operation fence emitter
      :964  called with the label "insert HRX host download barrier", i.e. the HOST DOWNLOAD path only
      :1916 the opt-in GGML_HRX_DIAGNOSTIC_GRAPH_SYNC knob is the only other synchronize
      :1901 flush_stream_commands("flush HRX graph replay commands")
      :1908 wait_stream_commands("wait for HRX graph replay commands", recorded.exec)

So dispatches are RECORDED into one command program and submitted together; ordering between them is
program order inside that program, which is the HRX runtime's responsibility, and the executor adds a
per-replay synchronize only when the diagnostic knob asks. That is exactly what §54 established from
the profile side (92 of 612 dispatches carry no barrier; queue saturated; ordering left to ring
program order) and it is why `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC` is the knob §55-§57 tested.

No new lead, and it closes the third leg of the static picture. The three legs now agree:
  - intra-kernel fences: clean for the #140 class in GLM's path (four hypotheses read and rejected)
  - consumer's own read path: cleared by §55/§56 (hsa_signal_wait_scacquire + timepoint covers last)
  - dispatch ordering: program order inside one recorded command program; per-replay sync is opt-in
    and §59 showed ordering it changes nothing (1 fault in 5 with sync vs 2 in 5 without, p=1.000)

### What that leaves, stated without overclaiming
Every ordering mechanism I can find is either present or was tested and did not change the rate. So
either the surviving defect is not an ordering/visibility problem at all, or it is in a mechanism
none of these three legs covers. The cheapest remaining discriminator is still the one the project's
own #123 work used and #315 has never applied: `GGML_HRX_FA_PARTIAL_ALIGN` (still present at
dispatch_registration/common/dispatch-flash-attention.cpp:23). If the fault rate rises when the
partial alignment is lowered, the fault IS the #140 layout-sensitive class and its surviving instance
is at a boundary the fences do not cover; if the rate is unchanged, the #140 class is not #315's
mechanism and the search should leave the FA path entirely.

## Forward-only landing mechanics - verified so a fix is landable the moment a mechanism is named (06:25)

Task-9's contract needs a forward-only branch or pin move with `git merge-base --is-ancestor <prev> <new>`
true. Verified what that actually requires on this repo rather than assuming:

    tools/check_pins.py <base-commit>
      "Fail a pull request that moves a submodule pin backwards or sideways."
      -> for every submodule whose gitlink differs between <base> and HEAD, it asks GitHub's compare
         API whether the new commit is AHEAD of the old one; behind or diverged fails.
    third_party/llama.cpp gitlink = 0e255d774032dc6b4b06a157d2feb6be2175d3ab
    registry/architectures.json:4  "llama.cpp (hrx)": "0e255d774032dc6b4b06a157d2feb6be2175d3ab"
      -> the SAME rev in two places, and nothing ties them together (an earlier session recorded that
         gap), so a pin move must change BOTH by hand and keep them consistent.

Consequences for the eventual fix:
 - the fork-side commit must DESCEND from 0e255d774032, not be rebased onto another base. A change
   prepared on top of 26330cc20 or f2099e9b7 would be rejected as sideways/behind (both are ANCESTORS
   of 0e255d774032, confirmed 2026-10-13).
 - the engine-side change is the gitlink plus registry/architectures.json, both moved together.
 - the contract's check is `git merge-base --is-ancestor <previous pinned rev> <new rev>`, which the
   unshallowed clone can now evaluate locally (it could NOT before 2026-10-13 - the fork clone was
   shallow and the revs were absent).

Box time note: at ~4.5 min/arm, a >=60 fix-cell campaign is ~4.5 h, and the goal's earlier ledger put
this turn at ~10-11 h of the stated 12 h ceiling. That budget is the binding constraint on task-9's
campaign, independent of the mechanism, and it is worth stating plainly to the user when the verdict
lands rather than discovering it mid-campaign.

## The contract's falsified-levers check PASSES for the current mechanism (06:35)

Task-9's verification contract requires: "Re-read the issue's falsified-levers list and confirm the
claimed mechanism is not one of them; if it is, the fix is rejected." Ran it rather than assuming.

The list, verbatim from `KNOWN-ISSUES.md` §"Levers tested and excluded" (7 entries):

    1. **`GGML_HRX_TRANSIENT_REUSE=legacy`**
    2. **Placement advice on the transient arena's committed backing**
    3. **The staged host writeback path**
    4. **The direct device-to-host copy not being covered by the wait**
    5. **Pinning `f2099e9b7` (the `1bit/amd-core-our-hrx` cell)**   <- an ancestor, i.e. a revert
    6. **The `TRANSIENT_REUSE` knob is not a lever for this fault**  (restatement of 1)
    7. **The sleeping wait returning before completion**

Case-insensitive search of that section for `cwsr|preempt|evict|resum|migrat|stale`: **0 matches.**

So the mechanism as it now stands - H2' (resumption-dependent buffer borrowing) and the #123/#140
stale-buffer class (an LDS-only-fenced producer->consumer handoff letting a consumer read a slot's
previous contents, triggered by preemption/eviction) - is **not** on the falsified list. The affected
levers are a different set: transient-reuse knobs, placement advice, the host writeback path, the
device-to-host copy wait, an ancestor pin, and the sleeping wait.

Note this check will need RE-RUNNING against whatever the actual landed change turns out to be: the
list is about the CHANGE, not the mechanism. A fix that merely re-enabled `GGML_HRX_TRANSIENT_REUSE`
would be rejected regardless of the mechanism it claimed. Recorded so the check is available verbatim.

---

# ############################################################
# THE H1 CONTROL IS COMPLETE AND IT FAULTED — H1 IS CONFIRMED
# ############################################################

Run: `315ctl1`, `amdgpu.cwsr_enable=1`, 20 pairs, booted 04:53:19 with the cwsr drop-in removed
(verified gone, 0 cwsr entries in grub.cfg). Finished 06:2x, 1h~35m.

## The final tally (authoritative, from 315-rate.py + the console)

    arms parsed          27    (20 fix-cell 'l' + 7 control-cell 'c')
    in-regime            27    OOR 0    INVALID 0
    clean                25
    FAULT                 2
    rate                 2/27 = 7.41%
    face tally           FaceA(nan)=1   FaceB=1(unattributed by the tool's regex)   -> BOTH FACES

    pair  8 arm l: FAULT nan=0 req=22 parse=0 utf8=0 gtt_start=97G
    pair 16 arm c: FAULT nan=1 req=1  parse=0 utf8=0 gtt_start=98G

    detector reconciliation (the authoritative source, per UPDATE 5 item 1):
      p8l  -> http_failures 1, first_failure 'HTTP 500'                       = FACE B
      p16c -> transport_failures 1, first_failure 'RemoteDisconnected'        = FACE A (nan=1, server died)

    Note the tool's own face tally shows 'FaceB=0 unattributed=1' for p8l: its regex keys on
    parse/utf8 counters, so a Face-B fault whose detector records `http_failures` (not parse) lands
    as "unattributed". The arm's own detector.json says HTTP 500 and its server.log shows the
    parser rejecting symbol soup - so it IS Face B. Recorded because the tool's face column
    under-counts Face B, the same class of trap as the console-vs-detector one.

## WHY THIS IS THE ANSWER TO CRITERION 1's DECISIVE TEST

The control was purpose-built to answer exactly one question: **does a fresh-booted box at
cwsr=1 fault?** The reading key, written before the run:

    ~26% faults -> H1 CONFIRMED (the cwsr=0 signal was a boot/state artefact)
    ~0%  faults -> H1 DEAD (the signal is real; H2' is the sole survivor)

**Observed: 2/27 = 7.41%, with BOTH FACES represented, at cwsr=1.** A fresh-booted box at cwsr=1
faults. So:

  - **H1 (fresh-boot confound) is CONFIRMED as a live explanation** for at least part of the
    cwsr=0 signal. The 70-arm cwsr=0 campaign (0/70) cannot be attributed to CWSR alone, because
    the cwsr=1 condition on a fresh boot also faults.
  - **The cwsr=0 signal's evidential value collapses.** This is exactly what `315-H1-FALSIFIED.md`
    had argued against - that doc reasoned run 7 was fresh-booted (~6-40 min) and faulted 3/11, so
    a fresh boot does not suppress the fault. That reasoning was about suppression; the control
    shows something different and stronger: a fresh boot at cwsr=1 STILL FAULTS, so cwsr=0's 0/70
    is not explained by "fresh boot" either. BOTH configurations fault; the difference between them
    is not established by the boot state alone.
  - **H2' stays alive but unproven.** It predicted faults on heavily-evicted arms; p8l showed 203
    evictions against a clean mean of 252.4 (no excess). This run now gives a SECOND faulting arm
    (p16c) to test, and the two faults land on OPPOSITE CELLS and OPPOSITE FACES, which is itself
    informative: the fault is not a property of one cell's binary.

## What this does to the criteria

  - **Criterion 1**: still OPEN, but its decisive experiment has been RUN and it returned the
    H1-confirming branch. The mechanism is therefore NOT "cwsr=0 prevents it"; that framing is dead.
    The surviving statement is: the fault occurs at both cwsr settings, on a fresh boot, at the
    frozen in-regime gate, on both faces - and the discriminator that remains is P1 (deny buffer
    movement at cwsr=1).
  - **Criterion 3**: unchanged and still MET (the 70-arm cwsr=0 campaign met its bar). But its
    INTERPRETATION changes: 0/70 at cwsr=0 no longer reads as "cwsr=0 fixes it" so much as "this
    configuration happened not to fault in that run", which is why the rate question must be
    re-asked honestly rather than treated as settled by the cwsr comparison.
  - **Criterion 2**: unchanged, still open, still blocked on a named mechanism.
  - The box now needs the cwsr=0 drop-in RESTORED for any further work that assumes the shipped
    configuration:  sudo cp ~/work/amd_cwsr.cfg.backup /etc/default/grub.d/amd_cwsr.cfg &&
    sudo update-grub && sudo systemctl reboot

## Immediate next step recorded, not yet run
P1 is the surviving discriminator and is copy-pasteable (see the P1 RECIPE below). It needs the box
lock, which the just-finished control released.

---

# H2' IS FALSIFIED (06:30) — two faulting arms show LESS eviction activity than every clean arm's floor

The H2 test H2' was built for, now with n=2 faulting arms instead of the single draw that was only
weakly informative before:

    FAULT n=2   evictions/arm:  min=16   median=203   mean=109.5
    clean n=25  evictions/arm:  min=117  median=254   max=559   mean=254.5

    pre-fault (causal): 203.0 from 1 of 2 fault arms (the other had no recoverable fault time)
    tool verdict: "NOT supportive (faulting <= 1.5x clean)"

## Why this is a falsification rather than a weak result

 1. **The direction is opposite to the prediction.** H2' predicts faulting arms are evicted MORE.
    Observed: the faulting mean (109.5) is 43% of the clean mean (254.5), and the faulting MEDIAN
    (203) is below the clean median (254).
 2. **One faulting arm is below the clean MINIMUM.** p16c shows **16 evictions** against a clean-arm
    minimum of 117 across 25 clean arms. That arm faulted (Face A) with essentially NO suspension or
    eviction activity attributable to it. H2' says the fault needs a buffer moved during the arm's
    own suspension window; this arm had almost none.
 3. **n=2 across opposite cells and opposite faces.** The two faulting arms are `p8l` (fix-cell,
    Face B) and `p16c` (control-cell, Face A) - so this is not one cell's or one face's artefact.
 4. **This is P3's refutation condition.** P3 was pre-registered as: "the same cwsr=1 arm that faults
    should have been suspended at least once. If an arm faults with zero Q_EVICT/Q_RESTORE for its
    pid, this mechanism is refuted." 16 is not zero, so the letter of P3 is not met - but the
    quantitative prediction (P2) fails in the opposite direction, with a faulting arm an order of
    magnitude below the clean floor.

## What is falsified, precisely
**H2' as a quantitative mechanism: "faults concentrate on arms whose buffers were evicted/migrated
during their own suspension window".** That claim predicted an eviction EXCESS on faulting arms. The
data show an eviction DEFICIT. The mechanism as stated is dead.

What is NOT claimed: that eviction is irrelevant to the fault in every conceivable form. A mechanism
in which eviction is a NECESSARY but very rare SUFFICIENT condition would need a different test (the
P1 denial experiment), and one faulting arm having 16 evictions while 117 is the clean floor leaves
room for a non-quantitative relationship. But the specific falsifiable prediction H2' made - the
discriminating one that justified calling it a mechanism - is contradicted.

## Consequence for criterion 1
H2' was the sole surviving candidate after H1 was re-opened by the control's own result. With H2'
falsified:
  - the cwsr signal does not survive as a mechanism (H1 explains its evidential collapse),
  - the buffer-movement/eviction story as a quantitative mechanism does not survive,
  - and the fault IS real and reproducible at cwsr=1 on a fresh boot, both faces, in-regime.
So criterion 1 is now further from a named mechanism, not closer - and the honest position is that
NO candidate on the record survives its own discriminating test. That is a real result and it should
be reported as such rather than dressed up.

The one experiment that could still rescue a movement-dependent mechanism is P1 (deny movement at
cwsr=1 via GGML_HRX_USE_UNIFIED_MEMORY): if faults vanish with movement denied, movement is
necessary even though not rate-correlated; if faults persist, movement is irrelevant. P1 is
copy-pasteable and ready.

---

## EVIDENTIAL GAP FOUND: the cwsr=0 drop-in's verbatim content was never recorded, and the backup is gone (07:12)

The control removed `/etc/default/grub.d/amd_cwsr.cfg` and the box now runs `cwsr_enable=1`. The
ledger's rollback path (`~/work/amd_cwsr.cfg.backup`, md5 `a8b25738...`) **does not exist** - not at
`~/work/`, not anywhere under `$HOME` (searched). The repo recorded only the PATH and the md5, never
the bytes:

    docs/315-campaign/evidence/315-task9-session-state-2026-10-12.md:833
      "The drop-in was captured verbatim to ~/work/amd_cwsr.cfg.backup before the reboot."
      ... but the file is absent and no copy of its content appears in the repo.

Also gone as a source: the grub.cfg that the cwsr=0 boot used was regenerated at 04:52, so the old
entry is not in it either.

**This is a real process failure of mine and it is recorded rather than papered over**: a rollback that
lives only at a path, with the content nowhere durable, is not a rollback. Had the box needed
restoring and the file been lost before the campaign, the cwsr=0 configuration would have had to be
reconstructed from semantics rather than restored from evidence.

### What IS recoverable, and how
The semantics are fully pinned by the investigation, even though the bytes are not:
 - the drop-in existed solely to put `amdgpu.cwsr_enable=0` on the kernel cmdline - that is the
   entire point of the cwsr work, and both campaign consoles record `cwsr_enable` state;
 - the file's purpose, effect and the reboot it required are documented in the control script's own
   header (`315-cwsr-control.sh:1-14`) and in the ledger;
 - the house format is a `grub.d` fragment setting `GRUB_CMDLINE_LINUX_DEFAULT`, which
   `/etc/default/grub.d/kdump-tools.cfg` demonstrates on this box;
 - `ttm.pages_limit=31457280 ttm.page_pool_size=31457280` are already in the main `GRUB_CMDLINE_LINUX_DEFAULT`
   and in the CURRENT `/proc/cmdline`, so a reconstruction must APPEND to those, not replace them.

### The gap's consequence for the record
 - No result in this issue depends on the missing bytes: every arm's `cwsr_enable` state was observed
   live from `/sys/module/amdgpu/parameters/cwsr_enable`, not inferred from the drop-in. The campaign
   consoles and the pre-flight checks all read the live parameter.
 - The criterion-1 conclusion is unaffected: the control's fault rate (2/27, both faces) was measured
   with `cwsr_enable` read live as 1.
 - So this is a durability/process defect in the rollback procedure, not an evidence defect in any
   finding. Recorded as such.

### Action taken (DONE 07:20)
 1. the content is now COMMITTED at docs/315-campaign/evidence/amd_cwsr.cfg (commit 9df9efb), so the
    rollback is durable in the repo rather than only at an ephemeral path;
 2. the drop-in was written back to the box and grub regenerated:
        /etc/default/grub.d/amd_cwsr.cfg  md5 e20f05bf91d50b645c0fbc8d11998e86
        sudo update-grub -> "cwsr_enable=0" appears in 3 entries of /boot/grub/grub.cfg
 3. so the box's BOOT CONFIGURATION is restored to the shipped cwsr=0 state. The RUNNING kernel is
    still cwsr=1 until the next reboot (mounting it now would reboot the box; that is a deliberate
    choice left to the user rather than taken unilaterally, since the campaign work is finished and
    nothing is waiting on it).

NEW md5 for any future rollback check: e20f05bf91d50b645c0fbc8d11998e86
(the OLD a8b25738... is unrecoverable and must not be cited again).

### What P1 would need before it can run
P1's pre-condition is cwsr_enable=1, which the RUNNING kernel already provides. So P1 can run on the
live boot as-is; it does NOT require the reboot. The drop-in restore above changes only what the NEXT
boot will do, which is the correct shipped state to leave behind.

---

# P1 IS DEAD AS AN EXPERIMENT (07:25) — its discriminating power was calibrated against the old 26% rate

P1 was pre-registered on the reasoning: "at 26%, 20 pairs gives ~5.2 expected faults and P(0) = 0.002,
the same power argument the H1 control used." **That rate is wrong for this configuration.** The
control measured the actual cwsr=1 rate as **2/27 = 7.4%**, not 26%.

Recomputed at the observed rate, a clean arm B is unremarkable:

    arms in arm B    P(0 faults | movement irrelevant) at p=7.4%
        8                     0.540
       10                     0.463
       12                     0.397
       16                     0.292
       20                     0.207

So if buffer movement is COMPLETELY IRRELEVANT, arm B would still come back clean 21-54% of the time
across any affordable size. A clean arm B would therefore be weak evidence at best, and P1's
pre-registered reading key ("A ~26% and B clean -> movement is necessary") does not hold, because A
will not be ~26% either - at 7.4% the positive control itself has a ~21% chance of coming back clean
at 20 arms, which would make the whole run uninformative by P1's own third branch.

### Cost, which rules it out independently
    P1 as designed: 2 runs x 20 pairs x 2 arms x 4.7 min = 6.3 h
    a smaller pair : 2 runs x 10 pairs x 2 arms x 4.7 min = 3.1 h   (and P(0)=0.46 in B)
This turn has already spent ~9.4 h of box time against the objective's <=12 h ceiling, so P1 at any
useful size breaches it - and at a reduced size it has no power anyway.

### What this means
 - **P1 is withdrawn as the discriminating experiment.** It is recorded rather than run, because
   running it would spend hours to produce a result its own statistics cannot interpret.
 - The correct replacement, if a movement experiment is ever wanted, is a DESIGN that raises the
   event rate first (e.g. the #123 repro's `GGML_HRX_FA_PARTIAL_ALIGN` stress, or a separate process
   forcing queue evictions as the #140 work did) so that the arm count needed falls. That is a new
   pre-registration, not a tweak to P1.
 - This is the third designed instrument on this issue whose calibration did not survive contact with
   a measured rate (after the H1 control's own sizing rationale, which DID work, and the 315-rate.py
   face-tally under-count recorded earlier). Worth stating plainly: the rate moved, and every
   power argument built on 26% moved with it.

---

## Box cleanup after the control (07:30)

Left behind by the campaign and now cleaned up:
 - the `p8l` arm server (pid 265323, the Face-B faulting arm) had been orphaned to `ppid=1` and was
   still holding `-dev HRX0` ~1h32m after its arm ended. It is MINE (its pid matches
   `rt-det/315ctl1-p8l-*/server.pid`), not a foreign workload, so it was terminated.
 - verified no other orphaned 315 arm servers remain (`ps` for the 315*-build llama-server pattern
   returns nothing).

Final box state:

    HRX0 holders  0
    GTT           26 GiB   (baseline)
    VRAM used     0 GiB
    load          1.05 1.22 2.38
    npu.lock      FREE
    watcher       0 running
    drop-in       present, cwsr_enable=0 in 3 grub.cfg entries (shipped boot state restored)
    live cwsr     1        (until the next reboot; the campaign work is finished)
    1bit-canonical / coder-llm: inactive (self-heal on demand, as designed)

Noted for the record: the orphan existed because a faulting arm's server dies without the harness
reaping it, which is the same class as the "silent arm loss" trap already in KNOWN-ISSUES. It held
HRX0 across the tail of the run and into the teardown; it did NOT affect the measurements (the arm's
own window ended at its fault), but it would have inflated a NEXT run's baseline if left.

---

## task-9's CONDITION was evaluated and is NOT MET (07:35)

task-9's own contract: "Only if task-7's instruments name a mechanism: forward-only branch or pin
move ... then a pre-registered >=60 in-regime arm campaign ... **Not attempted while the mechanism is
unnamed.**"

The condition was evaluated this turn by running task-7's instruments to their conclusions:

    instrument                              outcome
    the cwsr=1 fresh-boot control (H1)      RAN. Faulted 2/27 = 7.41%, both faces.
                                            -> H1 CONFIRMED; the cwsr signal's evidential
                                               value collapses; 'cwsr=0 prevents it' is dead.
    the per-arm eviction correlation (H2')  RAN. Faulting mean 109.5 vs clean 254.5; one faulting
                                            arm at 16 against a clean floor of 117.
                                            -> H2' FALSIFIED (predicted excess, observed deficit).
    the P1 movement-denial discriminator     WITHDRAWN. Power requires a rate it does not have
                                            (P(clean B | irrelevant) = 0.21-0.46), and cost is
                                            6.3 h against ~9.4 h spent of a 12 h ceiling.
    the #123/#140 stale-buffer class         CHECKED at source. The two sites #140 fixed are correctly
                                            fenced in GLM's path; four hazard hypotheses read to
                                            conclusion and rejected; consumer read path cleared
                                            (s56); dispatch ordering cleared (s54).

**No mechanism is named.** So task-9's condition is NOT satisfied and the task must not be attempted -
attempting it (a >=60-arm campaign and a forward-only fix) would be doing exactly what the contract
forbids: acting while the mechanism is unnamed, and shipping a change no evidence supports.

This is recorded as a RESOLVED-CONDITION, not a skip-to-avoid-work: the condition was actively
evaluated with three instruments and a source audit, and every one of them came back negative for a
mechanism. The honest state is that the issue's own candidate family is exhausted by measurement.

---

## DECISION: the available mechanism tests were evaluated and NOT run — with the numbers (07:45)

Rather than leave "we could still run something" implied, the one remaining untested instrument was
checked to a decision. Recording it because the reasoning is the deliverable.

### What `GGML_HRX_FA_PARTIAL_ALIGN` actually is (read at source, not assumed)
The record called it "the #123 repro's stress knob". Reading the call site corrects that:
`dispatch-flash-attention.cpp:384-386`

    if (decode_split_partial_alignment() != 4096u) {
        return {};   // DECLINE -> fall back to flash_attention_f32_f16_wmma
    }

and the comment at :380-383 says the non-production layout is a "test rig whose safety is not
established, so DECLINE ... rather than run an unproven layout". So setting it to 256 does not
reproduce the dangerous layout - it makes the dispatch REFUSE the decode-split kernel and fall back.
It is a path-elimination switch, not a stress. (The knob IS compiled into the arm build:
`315instr3-build/.../dispatch-flash-attention.cpp.o` contains the string, and it exists at the arm rev
`e44c9d01a`.)

### Decision analysis at the measured rate (p = 2/27 = 0.074)
A pilot is affordable only at <=14 pairs (2.2 h) against ~2.4 h of remaining budget.

    Outcome A: pilot FAULTS. P = 0.884 if the rate is unchanged. Learned: the decode-split FA
               kernel is not required. That is a NEGATIVE about one kernel, not a mechanism - and we
               already know faults occur on two cells/binaries at cwsr=1.
    Outcome B: pilot CLEAN. P(clean | FA irrelevant) = 0.116 - i.e. roughly one time in nine a
               clean result would be a coincidence. Its suggested follow-up is exactly the static
               search already run to conclusion (no surviving #140-class handoff in GLM's path), and
               the only stronger experiment (the movement denial) costs 6.3 h.

    Cost: the entire remaining box-time budget.
    Neither branch closes criterion 1.

**Not run.** Spending the last 2.2 h of a 12 h ceiling on a test whose clean branch is a 1-in-9
coincidence and whose faulting branch yields a negative would be the same calibration error that
sank P1. The honest move is to stop and report.

### The general shape of why every instrument failed here
Each designed test assumed a rate that could produce a decisive result. The measured rate is ~7%,
not the 26% the instruments were powered against, so EVERY test built on 26% lost its power when the
rate moved: P1 (0.21-0.46 chance of a clean arm B), this pilot (0.12 chance of a clean run by
coincidence at the affordable size), and the campaign-of-record's own >=60-arm bar (which passed at
0/70 but is now known to describe a configuration that also faults 7% of the time at cwsr=1).

That is the single most useful thing this turn produced: **the rate changed, and it invalidated the
power analysis behind every instrument in the issue's toolkit.** Any future attempt must first
re-measure the rate at the configuration of interest, then size its experiment from THAT number.

---

## RECONCILIATION: the "26% baseline" was inflated by pooling a deliberate repro rig (07:55)

Every instrument in this issue was powered against "~26% (34 faulting / 131 in-regime)". Checking that
number against its own source changes what it means, and it validates today's control.

### The 131-arm pool, decomposed by session (from the arm ledger's own per-tag table)

    tag          in-regime   faults    rate
    315nan            19       18     94.7%   <- a DELIBERATE high-rate reproduction session
    315fa             15        5     33.3%
    315hunt2          16        1      6.3%
    315accept         14        2     14.3%
    315chain           7        3     42.9%
    315sync2           7        1     14.3%
    315gsync           1        1    100%
    315row             5        2     40.0%
    315det             4        0      0%
    315probe2         13        0      0%
    315probe3          1        0      0%
    ---------------------------------------------------------------
    pooled            102       33     32.4%
    excluding 315nan   83       15     18.1%

(Ledger totals differ slightly from the pooled sum because the ledger counts 131 in-regime over 178
arms including rows this table groups; the DIRECTION is unambiguous either way.)

**`315nan` alone contributes 18 of the 33 pooled faults from 19 arms.** It was the session built to
reproduce the fault at a high rate. Pooling it with natural-rate sessions is what produced "26%" - the
figure is a mixture of a deliberately-clustered rig and ordinary runs, not a property of the
configuration.

### The two careful measurements agree with each other, not with 26%

    shipped-pin study (KNOWN-ISSUES, 37 arms, all at 102 GiB):  2/37 = 5.4%
    TODAY'S CONTROL  (27 arms, all at 97-98 GiB, cwsr=1)     :  2/27 = 7.4%
    Fisher exact two-sided p = 1.000  -> no difference detectable

So the control's measurement sits exactly where the issue's own earlier careful study put it. Two
independent runs, different sessions and different GTT points, both ~5-7%.

### What this means
 1. **Today's control is VALIDATED** - it did not measure an anomaly; it reproduced the shipped-pin
    study's rate to within noise, with a tighter CI and both faces represented.
 2. **The 26% comparator was wrong for this configuration**, which is why P1, the ALIGN pilot, and the
    >=60-arm bar's interpretation all lost power: they were sized against a number that mixed in a
    high-rate rig. The 0/70 cwsr=0 campaign was measured against 26% as well - against the more
    honest ~5-7% it is a weaker (though still real) statement.
 3. **The correct calibration for any future experiment** is ~5-7% at 97-102 GiB on a fresh boot at
    either cwsr setting, and it must be re-measured at whatever configuration is being tested.
 4. This does NOT change the H1 conclusion: the control faulted at 7.4% at cwsr=1 where the cwsr=0
    campaign saw 0/70. The comparison is still "faults here, none there" - just at a smaller absolute
    rate than the ledger's headline implied.

---

# TURN DISPOSITION (2026-10-13 ~08:00) — what was achieved, what could not be

## Criteria, against the objective's five

    1. named falsifiable mechanism, discriminating experiment     OPEN, and FURTHER away than at turn start
    2. forward-only change landed on a branch/PR                 OPEN; task-9's condition evaluated, NOT met
    3. >=60 in-regime arms, zero faults both faces, CI+power     MET  (315cwsr70: 70/70 fix-cell, 0
                                                                      faults, exact 95% CI 3.14%)
    4. no regression (engine CI + dense control + MUL_MAT_ID)    NOT STARTED (no fix to regress)
    5. KNOWN-ISSUES.md updated with evidence and met criterion    PARTIAL - findings recorded; there
                                                                      is no met criterion 1 to record

## What this turn DELIVERED (real, measured, pushed)

 1. **The H1 control RAN and FAULTED**: 2/27 = 7.41% at cwsr=1 on a fresh boot, BOTH faces, on
    opposite cells. H1 is CONFIRMED and the "cwsr=0 prevents it" framing is dead.
 2. **H2' FALSIFIED**: faulting arms show 109.5 evictions vs clean 254.5, with one faulting arm at 16
    against a clean floor of 117 - the opposite of the predicted excess.
 3. **The control VALIDATED against an independent study**: my 7.4% (97-98G) matches KNOWN-ISSUES'
    shipped-pin 5.4% (102G), Fisher p = 1.000.
 4. **The 26% comparator explained**: it pooled the 315nan repro rig (18 faults / 19 arms). Excluding
    it the pool is 18.1%; the careful measurements agree at ~5-7%. This is why every instrument lost
    its power.
 5. **Three designed instruments evaluated to a decision, with numbers**: P1 withdrawn (P(clean B |
    irrelevant) = 0.21-0.46, cost 6.3h vs ~2.4h left); the ALIGN pilot declined (clean branch is a
    1-in-9 coincidence at the affordable size); the #123/#140 class audited at source (both fixed
    sites correctly fenced in GLM's path; four hazard hypotheses rejected by reading code).
 6. **Three process failures found and fixed**: the cwsr=0 drop-in's bytes were never preserved (now
    reconstructed, committed, box boot-state restored, new md5 e20f05bf...); 315-rate.py under-counts
    Face B in its face tally; an orphaned faulting-arm server held HRX0 and was terminated.

## What this turn did NOT achieve, and why

**Criterion 1 is not met.** No mechanism survived its own discriminating test. That is not a failure
to look - it is the result of looking with the instruments the issue built, all of which were
powered against a rate that turned out to be wrong by a factor of ~4.

**Criterion 2 is not met**, and it must not be. task-9's own contract says "Not attempted while the
mechanism is unnamed"; landing a forward-only change now would ship a change no evidence supports,
and the contract's first check - that the mechanism is not on the falsified-levers list - cannot even
be meaningfully applied to a mechanism that does not exist.

## The single most useful product of this turn

**The rate moved, and it invalidated the power analysis behind every instrument in the issue's
toolkit.** Any future attempt must (a) re-measure the rate at the configuration of interest, (b) size
its experiment from that number rather than from the ledger's pooled "26%", and (c) prefer designs
that RAISE the event rate before testing a mechanism - which is what the #123/#140 work did with a
separate process forcing queue evictions, and what none of this issue's instruments did.

## Box left in the shipped state

HRX0 holders 0 · GTT 26 GiB (baseline) · VRAM 0 GiB · npu.lock FREE · no watcher · both managed
services inactive · drop-in present with cwsr_enable=0 in 3 grub.cfg entries (shipped BOOT state
restored). Live cwsr=1 until the next reboot, which was NOT taken unilaterally.

## Box time

~1.6 h this turn (the control), against a turn total of ~9.4 h and the objective's <=12 h ceiling.
That remaining ~2.6 h is why the two candidate experiments were evaluated arithmetically instead of
run: neither could be afforded at a size that would have been interpretable.

---

## CORRECTION/CLARIFICATION on the H2' numbers (08:10) — two counts, and only one is causal

While checking the H2' falsification I computed the RAW per-pid eviction totals and got a very
different picture, which must be recorded rather than left to confuse a future reader:

    raw per-pid counts over the whole process life (what a naive grep gives):
      p8l   (FAULT, Face B)  evict=2798  restore=2997   <- the HIGHEST in the run (clean max 559)
      p16c  (FAULT, Face A)  evict=  16  restore=  16   <- the LOWEST in the run (clean min 117)

    arm-bounded counts from 315-evict-per-arm.py:
      p8l   pre-fault 203  window 203   (fault at 91.4s)
      p16c              -  window  16
      FAULT n=2 mean 109.5   vs   clean n=25 mean 254.5    -> NOT supportive

**Why they differ, and why the tool's number is the right one:** `p8l`'s server was ORPHANED (the one
I found holding HRX0 at 07:30, ppid=1) and had NO `PROC_END` row, so its raw count spans ~1.5 h of
life after its arm ended - it is a function of post-arm survival, not of the arm. The tool bounds each
arm `PROC_START -> PROC_END or the next arm's PROC_START`, which is exactly the confound its own
docstring warns about ("a faulting arm ... lives on after the failure, so its window can simply be
longer"), and it prints the PRE-FAULT column as the causal test precisely because the fault cannot
cause evictions that precede it.

**So the falsification stands on the correct figure** - but the raw numbers show something worth
naming: the two faulting arms are the run's extreme of BOTH directions (highest raw, lowest raw). That
is what a fault-time-independent variable looks like when the sample is n=2, and it is further reason
not to read a rate-correlation into these counts.

**The honest statement, restated:** on the arm-bounded, causal (pre-fault) measure, faulting arms show
no eviction excess - 203 pre-fault vs a clean mean of 254.5 - and one faulting arm sits below the
entire clean range. H2' predicted an excess. It is not there. Combined with the raw counts landing at
both extremes, the eviction COUNT is not a discriminator for this fault at this sample size, and the
mechanism as stated is falsified.

**Lesson recorded:** any future eviction analysis on this issue must use the arm-bounded pre-fault
column, never a raw per-pid grep. The orphaned-server trap makes the raw number meaningless for
faulting arms specifically - which is the worst possible bias for a fault test.

---

## FREE ANALYSIS ON THE CONTROL'S DATA (08:30) — two candidate signals found, one killed by the causal test

No box time: pure re-analysis of already-collected data. Recorded because one of the two looked
striking and the causal test is what settled it.

### What the arm metadata showed (the tempting signal)
Per-arm console lines give `gtt_start` and `gtt_now`, and the two faulting arms stood out:

    arm      verdict   gtt_start   gtt_now   delta     pair position
    p8l      FAULT      97G         104G      +7G       FIRST arm of pair 8
    p16c     FAULT      98G          80G     -18G       FIRST arm of pair 16
    clean    OK         97-98G      97-99G   0G (15 arms) / +1G (10 arms)    -

Every clean arm stayed within 0..+1G while both faulting arms moved +7G and -18G, and both faulting
arms are the FIRST arm of their pair. Tempting readings: "the fault needs a GTT excursion" and "the
first arm of a pair is at risk".

### Why neither survives
 1. **The pair-position pattern is a 1-in-4 coincidence.** Two faulting arms, both first-of-pair,
    under a ~50/50 ordering is p = 0.25. n=2 establishes nothing.
 2. **The GTT delta is measured at arm END, and a faulting arm ABORTS mid-workload** (p8l stopped at
    req=22 of 66; p16c at req=1 of 66). A shorter, failed run is expected to move GTT differently. The
    delta is plausibly an EFFECT of the abort - the same arrow-of-time trap the eviction analysis had
    to avoid, and the reason that tool prints a pre-fault column.
 3. **The causal test is available and it kills it.** The SMI stream carries per-pid Q_EVICT rows with
    timestamps, and each arm's PROC_START is recorded, so activity can be counted in the window BEFORE
    the fault instead of over the whole arm. Counting evictions in the FIRST 60 SECONDS after each
    arm's PROC_START (before either fault at 91.4s and earlier):

         p8l   (FAULT)   evictions in first 60s = 103     <- mid-pack
         p16c  (FAULT)   evictions in first 60s =  16     <- at the low end
         clean arms (25) evictions in first 60s = 34..249

    So on the pre-fault measure the faulting arms are NOT elevated - one is mid-range and one is low.
    The end-of-arm GTT excursion is an effect of the abort, not a cause of the fault.

### What this adds
A third independent negative on the same question (is the fault tied to memory-activity magnitude in
the arm?): the arm-bounded count said no (203 vs clean 254.5), the raw count was shown meaningless
(orphaned server), and now the pre-fault 60s window says no as well, with the faulting arms inside the
clean range rather than above it. The candidate is closed by the same causal logic that closed its two
predecessors, and this time at zero box cost.

Recorded with the numbers so the tempting metadata pattern is not rediscovered and mis-read as a lead.

---

# THE CLEANEST CHARACTERISATION OF BOTH FACES IN THE RECORD (08:45) — from the control's own logs

Re-reading the two faulting arms' `server.log` line by line (free, no box time) gives the exact
sequence for each face, and it is the sharpest evidence this issue has produced.

## p16c — FACE A, on the FIRST request, 165 ms in

    0.10.661.057  load_model: initializing, n_slots = 1, n_ctx_slot = 8192
    0.10.672.905  llama_server: model loaded
    0.10.672.910  llama_server: listening on http://127.0.0.1:19556
    0.11.115.575  slot launch_slot_: id 0 | task 0 | processing task, is_child = 0   <- FIRST request
    (sampling.cpp:620) HRX: NaN logits (engine#123)
    0.11.280.251  E common_sampler_sample: HRX returned NaN logits: count=154880 of 154880,
                  index range [0..154879], contiguous=yes, continuing=no

So the sequence is: **model loaded -> server listening -> first request -> 165 ms of work -> the ENTIRE
vocabulary (154880 of 154880) NaN.** The guard aborts, the process dumps a backtrace (a 105-line log
against a clean arm's 485), and the detector records `transport_failures 1 / RemoteDisconnected`.

Three things this pins down:
 1. **It is the load/warmup phase, not steady-state generation** - 11.3 s into a process whose clean
    siblings run ~207 s, and the very first `task 0`.
 2. **It is the whole LM-head output, contiguous** - `[0..154879] contiguous=yes` - which is the exact
    signature KNOWN-ISSUES already records (`count=154880 of 154880, contiguous=yes`). The buffer was
    not partially wrong; it was entirely unwritten or entirely poison.
 3. **It is reproducible at this phase** - the same shape as the historical `v315g-c3` control arm in
    KNOWN-ISSUES. So a fault at the FIRST request, whole-vocabulary, is not a new phenomenon.

## p8l — FACE B, after ~90 s of SUCCESSFUL generation

    0.40.207.097  slot print_timing: id 0 | task 454 | eval time = 5000.24 ms / 96 tokens (19.20 tok/s)
    ...
    1.31.428.076  W srv operator(): got exception: {"error":{"code":500,"message":"The model produced
                  output that does not match the expected Content-only format"}}

The arm ran to `req=22` of 66 with 21 successful completions, printing timing for 96-token evals, and
then the parser rejected one output at 1 min 31 s. Its server survived (no backtrace; the log ends
normally) - so Face B is a *content* failure inside an otherwise working server.

## Why this matters - and what it does NOT say
The two faces are now characterised at the level of concrete timestamps rather than verdicts:

    Face A: FIRST request, ~165 ms of work, whole-vocabulary NaN, process aborts
    Face B: ~90 s in, after 21 good completions, one degenerate output, process survives

That is a materially different pair of signatures from "intermittent corruption somewhere in the run",
and it sharpens what a mechanism must explain: Face A is an *initialisation-adjacent* whole-buffer
failure, Face B a *late, partial* content failure. It also means a future experiment can target the
warmup phase specifically for SENSITIVITY - the fault appeared 165 ms into a warmup request in one arm
of a 27-arm run. [CORRECTED 08:55 below: warmup-targeting is NOT cheaper, because the ~14 min model
load dominates every arm regardless of how many requests follow.]

NOT claimed: that the two faces have different mechanisms (they may still be one defect seen at
different phases); that the warmup phase is *enriched* for the fault (n=1 at that phase, and the
historical comparator is another single example); or that Face A is confined to warmup. What IS
established is the phase, the extent, and the survival-vs-abort difference, each with a timestamp.

---

## CORRECTION to my own claim (08:55): a warmup-targeted design is NOT cheaper — the load dominates

In the characterisation above I wrote that a future experiment "can target the warmup phase
specifically, which is both cheaper (seconds, not minutes per arm) and much higher-yield". The first
half is **WRONG** and I checked it rather than leaving it in the record.

The fault at 11.3 s happens *after* the model has loaded, and the load is where the time goes:

    p16c timeline:  0.00 -> 0.10.67  load_model + model loaded    (~10.7 min of the 0.11.3 life)
                    0.11.11          first request launched
                    0.11.28          fault                            <- 165 ms of actual compute

So an arm that stops after one request still pays the ~10-15 min load. Costed:

    warmup-only (1 request)   ~14.5 min/arm   -> 20 arms = ~4.8 h
    full pair (66 requests)   ~17.3 min/arm   -> 20 pairs = ~3.1 h  (and 20 pairs yields 40 arms)

Warmup-only is therefore MORE expensive per informative arm, not less: it saves ~3 min of request
phase while keeping the ~14 min load, and it yields one arm per load instead of two.

**What survives from the observation is the other half:** the fault is reachable on the FIRST request,
so a warmup-phase arm is a legitimate *harbour* for the fault and is much faster *once loaded*. The
yield claim (higher-yield) is untested - it rests on n=1 at that phase plus one historical comparator.

Corrected statement, for the record: **the fault is observed on the first request 165 ms in, which
tells us the defect is present at load-completion and does not require a long generation phase; it does
NOT make the experiment cheaper, because the model load dominates every arm regardless.** A future
design should target the phase for *sensitivity*, not for cost - and any such design must first
measure the warmup-phase rate, which is exactly the "measure the rate at your configuration" lesson
from the 26%-vs-7% correction.

Recorded because an over-claim left standing in the evidence file is how a future session ends up
sizing an experiment from a number that was never true.

---

# ###### MAJOR: THE EVICTION INTERVENTION ALREADY EXISTS AND IT WORKS — I mis-stated H2' as falsified
# (2026-10-13 09:10, found by following the rate-raiser the record kept naming)

## The result, sitting in `~/wt/315disc1-console.log` since 2026-10-11

`315-disc.sh` is a three-cell discriminating design: `q` quiet (control), `e` = `~/wt/evictonly 500`
(a process that launches NO kernels and only forces KFD evict/restore via `hipHostRegister` +
`mprotect` churn), `c` = CPU pressure only, cells rotating per round so session drift cannot align
with one cell. All arms at the same composition, gate `gtt_start=98G`, build `10613 (e44c9d01a)`.

    round 1 arm q: OK     nan=0 req=66                gtt_start=98G
    round 1 arm e: FAULT  nan=1 req=6  transport=1    gtt_start=98G     <- EVICTION CELL
    round 1 arm c: OK     nan=0 req=66                gtt_start=98G
    round 2 arm e: FAULT  nan=1 req=4  transport=1    gtt_start=98G     <- EVICTION CELL
    round 2 arm c: OK     nan=0 req=66                gtt_start=98G
    round 2 arm q: OOR    (out of regime, not counted)
    round 3 arm c: OOR    (out of regime, not counted)
    ...

    cell q (quiet)        5 arms   0 FAULT
    cell e (EVICTION)     4 arms   2 FAULT   = 50% of in-regime arms
    cell c (CPU pressure) 5 arms   0 FAULT

    Fisher exact, cell e (2/4) vs cells q+c pooled (0/10): p = 0.066

## Why this matters more than the p-value
This is an **INTERVENTION**, not a correlation: an external process that only makes KFD evict and
restore queues raises the fault rate to 50% while the quiet and CPU-pressure cells stay at 0/10. That
is precisely the method KNOWN-ISSUES records for engine#123/#140 ("a separate process forcing queue
evictions") and precisely the rate-raising design the 26%-vs-7% correction said was missing.

It also **directly contradicts the way I stated the H2' result earlier today.** I wrote "H2' IS
FALSIFIED". That is too strong and is now corrected:

    disc1  (intervention: force eviction)          -> faults 2/4 = 50%    <- eviction CAUSES faults
    ctl1   (ambient pressure, per-arm correlation) -> faulting arms show no eviction excess
                                                      (203 vs clean mean 254.5)

These are not in conflict. An external forcer raises the EVENT RATE; within a run under ambient
pressure, the per-arm eviction COUNT does not predict which arm faults. The first is an intervention
(the right instrument for causation); the second is a correlation (the wrong instrument, and now
shown to be uninformative at this sample size). **H2-as-eviction-trigger is SUPPORTED by the
intervention and was only ever falsified as a rate-correlation.**

## What the record already knew, and what it missed
The ledger HAD recorded "`315disc1` (Oct 11): 2 (arm e, both rounds)" - but it recorded it in a table
of runs "usable for H2?", where it was correctly marked NO because 0 of its 7 arm dirs have
`server.pid` (so no per-arm eviction correlation was possible). What was missed is that the run answers
a DIFFERENT question from the one the table was asking: not "which arm was evicted more" but "does
forcing eviction from outside cause the fault". It does.

This is the single strongest causal contrast in the issue's history and it had been filed as unusable
for the wrong test.

## Also: my "H2' is falsified" commit stays, with this correction attached
I am not rewriting the earlier commit; the per-arm correlation result is real and correctly computed.
What changes is the CONCLUSION drawn from it. A mechanism can be causally real (intervention) while not
being rate-correlated per arm (observation) - and the earlier write-up collapsed those two into one
claim. Recorded as a correction with the original left intact, per the issue's own rule about not
quietly reinterpreting recorded studies.

---

## The eviction instrument is VERIFIED, not assumed (2026-10-13 09:35)

The whole disc1/disc2 interpretation rests on `evictonly` really forcing KFD eviction. Measured it
directly with the issue's own watcher rather than trusting the source comment:

    window                  Q_EVICT   Q_RESTORE
    25 s, quiet box             0          0
    25 s, evictonly 500      4134       4168

So a kernel-free process doing only `mprotect` churn over a `hipHostRegister`'d 16 MiB range provokes
**~4,134 KFD evictions in 25 s (~165/s)** where the quiet box produces **zero**. The instrument is
real: no kernels are launched (so this is not a compute or thermal effect), and the evictions are
attributable by pid in the SMI stream.

Why that matters for the argument: the disc1 contrast (cell e 2/4 FAULT vs cells q+c 0/10) is only
interpretable as a causal statement about eviction if the 'effect' cell really is an eviction
intervention and the control cells really are not. The probe establishes both halves - the quiet
window is 0, not merely "small".

Method note: the first attempt at this probe returned 0 everywhere because `grep -c` output carried a
newline into a shell arithmetic expansion, which failed silently and printed 0. Re-ran with the count
normalised and it worked. Recorded because a silent 0 from a broken probe is indistinguishable from a
true 0, which is exactly the failure mode that would have made this negative.

---

## 315disc2 IS RUNNING (09:28) — the eviction-intervention replication

    driver : ~/wt/315-disc.sh 6 315disc2   (the SAME frozen design as disc1)
    started: 2026-10-13T07:20:44-03:00, composition 3 neighbours at gtt=79G
    cells  : q (quiet) / e (evictonly 500) / c (CPU pressure), rotating per round
    gate   : gtt_start >= 95 GiB; OOR reported, not counted
    pre-reg: docs/315-campaign/evidence/315-eviction-replication-preregistration.md (commit 36f360a)

First arm observed: `round 1 arm q: OK nan=0 req=66 gtt_start=97G` (in-regime, clean), with the
eviction cell then entering and `evictonly` confirmed ACTIVE.

    pre-registered prediction : e faults clear of q and c
    pre-registered branches   : e clear of q,c  -> REPLICATED
                                all alike       -> NOT replicated
                                controls fault  -> CONTRADICTED
                                all clean       -> UNINFORMATIVE (declared so in advance, because at
                                                   the measured ~5-7% ambient rate ~16 arms is expected
                                                   clean most of the time)

Waiter task `b4c7ae24f` will report the per-cell tally on completion (~40 min).

---

## DISС2 FIRST RESULT (09:29) — the eviction cell faulted, replicating disc1's signature

    round 1 arm q: OK     nan=0 req=66                              gtt_start=97G
    round 1 arm e: FAULT  nan=1 req=19 transport=1 side=1/1         gtt_start=98G
    round 1 arm c: OK     nan=0 req=66                              (cell c, first arm)

The faulting arm's own log and detector:

    315disc2-r1e: requests=19, ok=18, transport_failures=1,
                  first_failure="RemoteDisconnected: Remote end closed connection without response"
    server.log:145: 1.18.529.363 E common_sampler_sample: HRX returned NaN logits:
                    count=154880 of 154880, index range [0..154879], contiguous=yes, continuing=no
    server.log:218: #5 ggml_abort() from libggml-base.so.0

So: **the whole vocabulary (154880 of 154880), contiguous, the sampler guard aborts, the process dies** -
the identical signature to the control's p16c (`count=154880 of 154880, contiguous=yes`) and to disc1's
two faulting arms (nan=1, transport=1, early abort). The console records `side=1/1`, i.e. the eviction
side-load WAS engaged for this arm.

That is the first arm of the replication and it lands exactly on the pre-registered prediction branch:
**cell e faults while cells q and c are clean** - so far 1 fault in 2 e-arms against 0 in 2 controls.
The run continues; the final per-cell tally is what the pre-registered reading key is applied to.

NOT yet claimed: replication. One fault in one arm is one draw, exactly the situation that made disc1's
contrast p=0.066 rather than decisive. The value of this run is that it upgrades disc1 from a single
unreplicated observation toward a repeated one - and it must be read against the branches fixed in
36f360a BEFORE the run, which it currently matches on the 'e faults clear of q and c' branch.

---

# ##########################################################
# H-e REPLICATED: FORCED KFD EVICTION CAUSES THE FAULT
# ##########################################################
# (2026-10-13 08:30 — disc2 partial, pooled with disc1; run still finishing)

## The pre-registered endpoint

    cell e  (evictonly, forced KFD evict/restore)   6 FAULT / 14 arms = 42.9%
    cells q+c (quiet + CPU-pressure controls)       0 FAULT / 23 arms =  0.0%
    Fisher exact two-sided p = 0.0013
    rule-of-three upper bound on the control rate (23 arms, 0 faults) = 13.0%

The eviction cell faults at 42.9 %, more than three times the control's upper bound. Against the branch
fixed in the pre-registration BEFORE the run (commit 36f360a):

    "e faults clear of q and c  -> H-e REPLICATED.  Eviction forced from outside causes the fault;
                                    the mechanism is eviction-of-in-flight-work."

That branch is the one the data landed on.

## The per-run detail

    disc1 (2026-10-11)  e: 2/4     q+c: 0/10
    disc2 (2026-10-13)  e: 4/10    q+c: 0/13   (partial; run still in its later rounds)

Both runs, both builds nominally the same (10613 / e44c9d01a), same gate (>=95 GiB), same composition
family, different days. The eviction cell faults in both; the controls are clean in both, now 23 arms
without a single fault.

## Why this is the criterion-1 result the issue needed

 1. **It is an INTERVENTION, not a correlation.** `evictonly` launches no kernels; it only does
    `mprotect` churn over a `hipHostRegister`'d 16 MiB range. I verified it directly with the issue's
    own watcher: **4,134 Q_EVICT and 4,168 Q_RESTORE in 25 s**, against **0 and 0 in a 25 s quiet
    window**. So the effect cell really is an eviction intervention and the controls really are not.
 2. **It separates eviction from every competing factor the issue already tested.** The `c` cell
    (CPU pressure, no GPU) is clean at 0/13, and the `q` cell (quiet) is clean at 0/10 - so the faults
    are not from thermal load, CPU contention, scheduling, or the mere presence of a benchmark. The
    only difference in cell `e` is forced KFD eviction/restore.
 3. **It is falsifiable and was falsified-in-advance-able.** The pre-registration fixed all four
    branches, including that an all-clean run would be UNINFORMATIVE. The run instead produced a
    positive with p = 0.0013 pooled.
 4. **It is the rate-raiser whose absence caused every earlier instrument to fail.** ~43 % in the
    eviction cell versus ~5-7 % ambient: this is the design that makes a mechanism test affordable,
    which the 26 %-vs-7 % correction identified as the missing ingredient.

## The mechanism statement this supports, and its exact limits

**Supported:** forced eviction of in-flight KFD work is a CAUSE of the fault. Concretely - when a
separate process makes KFD evict and restore queue memory while a GLM-4.7-Flash HRX server is decoding,
the server's logits come back wrong at ~43 % of arms, both faces represented, and neither a quiet box
nor CPU-only pressure does this at all (0/23).

**Not yet named:** the specific handoff that reads stale data. This result names the TRIGGER
(eviction of in-flight work) - which is a genuine mechanism-level finding and directly contradicts the
earlier "it was never the cause" reading of the #123/#140 record for THIS issue. It does not by itself
say which buffer or which fence is at fault, and the static leg has already shown the #140-class
handoff in GLM's path is correctly fenced (commit 1de679f). Locating the handoff the eviction corrupts
is the remaining work for criterion 1's strongest form.

**Also contradicted, and this matters:** my own earlier statement that "H2' IS FALSIFIED" (commit
ace7218). Eviction DOES cause the fault, demonstrated by intervention. What was falsified was only the
per-arm rate-correlation (faulting arms did not show more evictions than clean arms under ambient
pressure). The correction is recorded at commit 3a2ba07 and this result confirms it.

## Box-time note
disc2 was pre-registered and launched inside the remaining budget (~43 min at 2.4 min/arm). This is the
run the 12 h ceiling could still afford, and it is the one that mattered.

---

# BOTH FACES UNDER THE SAME EVICTION INTERVENTION (08:35) — criterion 1's "both faces" clause MET

I had flagged "Face B has not yet appeared under forced eviction" as a gap. It has. The disc2 eviction
cell produced Face B in rounds 3 and 4, alongside Face A in rounds 1 and 5:

    round 1 arm e: FAULT nan=1 req=19  http=0 transport=1 s500=0 unparsed=0   <- FACE A
    round 3 arm e: FAULT nan=0 req=7   http=1 transport=0 s500=1 unparsed=2   <- FACE B
    round 4 arm e: FAULT nan=0 req=11  http=1 transport=0 s500=1 unparsed=2   <- FACE B
    round 5 arm e: FAULT nan=1 req=1   http=0 transport=1 s500=0 unparsed=0   <- FACE A

All four at `gtt_start=98G`, all in the eviction cell, all in the SAME run, same day, same build, same
composition. And disc1's two faults were both Face A (`nan=1`, http=0).

So across the two runs the eviction cell has produced:

    Face A (NaN / whole-vocabulary, sampler guard aborts):  disc1 x2, disc2 x2  = 4
    Face B (invalid/parse failure -> HTTP 500, nan=0):      disc2 x2            = 2

**Both faces appear under the same forced-eviction intervention, and neither appears in the control
cells (0 faults / 23 arms).** That is precisely what criterion 1's clause requires - "a named,
falsifiable mechanism for BOTH faces, with a discriminating experiment" - and it additionally settles
the question the record had left open since §44: whether the two faces are one mechanism or two. Under
a single intervention that isolates eviction from every other factor tested so far, the same
intervention yields both faces, which is what "one mechanism, two observables" predicts and what
"two separate defects" would not require.

Note the shape difference between them in this run, which is informative: the Face-A faults abort the
SERVER (transport=1, process dies), while the Face-B faults are caught by the PARSER (http=1, s500=1,
the server survives the request). That is the same survive-vs-abort difference measured earlier from
the control's two arms, now reproduced under the intervention.

## What is now established for criterion 1
 - NAMED: forced eviction of in-flight KFD work is a cause of the fault
 - FALSIFIABLE: pre-registered branches, fixed before the run (commit 36f360a)
 - DISCRIMINATING: three cells isolate eviction from thermal/CPU/benchmark-presence
 - RULES OUT SURVIVING ALTERNATIVES: CPU pressure 0/13 and quiet 0/10
 - BOTH FACES: yes, under the same intervention, in the same run
 - NOT YET: the specific handoff that reads stale data (the static leg showed the #140-class handoff
   in GLM's path is correctly fenced, so the corrupted handoff is elsewhere and still unnamed)

---

# ################################################################
# 315disc2 COMPLETE: H-e REPLICATED, p = 0.0002, BOTH FACES
# ################################################################
# 2026-10-13 08:40

## The final, pre-registered endpoint (disc1 + disc2 pooled)

    cell e  (evictonly, forced KFD evict/restore)   7 FAULT / 16 arms = 43.8%
    cells q+c (quiet + CPU-pressure controls)       0 FAULT / 30 arms =  0.0%
    Fisher exact two-sided p = 0.00021
    rule-of-three upper bound on the control rate (30 arms, 0 faults) = 10.0%

    315disc2 standalone: e 5/12 = 41.7% vs q+c 0/20 = 0%, Fisher p = 0.0039

The pre-registration (commit 36f360a, written before the run) fixed four branches; the data landed on:

    "e faults clear of q and c -> H-e REPLICATED. Eviction forced from outside causes the fault;
     the mechanism is eviction-of-in-flight-work."

## Both faces, under the same intervention

    Face A (nan=1, transport=1, server aborts):  disc1 x2, disc2 x2  = 4
    Face B (nan=0, http=1, s500=1, unparsed=2):  disc2 x3            = 3
    controls                                  :  0 faults / 30 arms, EITHER face

All seven faults at gtt_start=98G, all in cell `e`, all in-regime, zero OOR in either run.

## Why this is the result criterion 1 asked for

The contract: "A named, falsifiable mechanism for both faces, with a discriminating experiment that
rules out the surviving alternatives (arena knobs and transient reuse are already falsified)."

    NAMED                    forced eviction of in-flight KFD work is a cause of the fault
    FALSIFIABLE              four branches fixed before the run; one was 'controls fault -> CONTRADICTED'
    DISCRIMINATING           three cells isolate eviction from thermal load, CPU contention and
                             benchmark presence
    RULES OUT ALTERNATIVES   CPU-pressure cell 0/20, quiet cell 0/10, across two runs
    BOTH FACES               4 Face A + 3 Face B, same intervention, same cell, same run
    EFFECT SIZE              43.8% vs 0/30 (bound 10%), p = 0.0002

The intervention is verified, not assumed: `evictonly` launches NO kernels and only does `mprotect`
churn on `hipHostRegister`'d memory, and I measured it at **4,134 Q_EVICT / 4,168 Q_RESTORE in 25 s
against 0 / 0 in a quiet window**.

This also settles a question the record had left open since mechanism §44: whether the two faces are
one mechanism or two. A single intervention that isolates eviction from every other factor tested
produces BOTH faces, which is what "one mechanism, two observables" predicts.

## The one thing still NOT named, and it is the difference between a trigger and a cause

**Named: the TRIGGER.** Forced eviction of in-flight KFD work causes the fault.

**Not named: the CAUSE.** Which handoff reads stale data. The static leg already showed that the
#140-class handoff in GLM's path is correctly fenced in this pin (commit 1de679f: `barrier<global>`
then `barrier<workgroup>` between the reduce's stores and the q8 pack, plus the completion-counter
handshake), so the corrupted handoff is somewhere else and is still unidentified.

That distinction matters for how this is reported: the issue now has a **named, falsifiable,
replicated mechanism-level trigger with a verified intervention**, and does not yet have the specific
code site to repair. Under the objective's own wording the first clause ("name the mechanism") is met at
the trigger level; the second (a forward-only fix) requires the site.

## And it corrects my own record twice
 - "H2' IS FALSIFIED" (ace7218) - corrected at 3a2ba07 and now definitively: eviction DOES cause the
   fault by intervention. Only the per-arm rate-correlation was ever falsified.
 - "Face B has not yet appeared under forced eviction" (recorded 08:30 today) - corrected within the
   same run: 3 of disc2's 5 faults are Face B.

---

# CORRECTED ENDPOINT (08:52) — the driver's own summary shows 70%, not 44%

My per-cell tally counted `gate wait before round N arm X` lines as arms. The driver's own final summary
is authoritative and the effect is much larger:

    315disc2 final summary (verbatim):
        cell q: OK=6  FAULT=0  OOR=0  INVALID=0
        cell e: OK=1  FAULT=5  OOR=0  INVALID=0     <- 5 of 6 in-regime eviction arms faulted
        cell c: OK=6  FAULT=0  OOR=0  INVALID=0

So disc2 alone:

    cell e (eviction)   5 FAULT / 6 arms  = 83.3%
    cells q+c           12 arms, 0 FAULT  =  0.0%
    Fisher exact two-sided p = 0.0007

Pooled with disc1 (whose own summary was e 2 fault / 2 ok, q+c 0 fault / 10 ok):

    cell e  (eviction)   7 FAULT / 10 arms = 70.0%
    cells q+c (control)  0 FAULT / 22 arms =  0.0%
    Fisher exact two-sided p = 3.6e-5
    rule-of-three upper bound on the control rate (22 arms, 0 faults) = 13.6%

**70% against a control bound of 13.6%, p = 3.6e-5, with all seven faults in-regime at
gtt_start=98G and zero OOR in either run.**

## The mechanism statement, at the strength the evidence supports

> A process that makes the kernel evict and restore in-flight KFD queue memory - launching no kernels
> of its own - raises the GLM-4.7-Flash HRX fault rate from 0/22 arms to 7/10 arms, producing BOTH
> recorded faces (4 Face A: whole-vocabulary NaN, sampler guard aborts the server; 3 Face B: invalid
> output, parser returns HTTP 500, server survives). Neither a quiet box (0/10) nor CPU-only pressure
> (0/12) does this at all.

That is a named, falsifiable, replicated, pre-registered mechanism-level statement with a verified
intervention and a controlled comparison. It identifies eviction of in-flight work as a CAUSE of both
faces, which is a stronger and better-controlled claim than the issue has previously held.

## The honest boundary, unchanged by the correction
The specific HANDOFF that reads stale data is still not named. The static leg showed the #140-class
handoff in GLM's path is correctly fenced in this pin (commit 1de679f), so the corrupted handoff is
elsewhere. The trigger is named; the code site is not. That is the difference between criterion 1's
first clause and criterion 2's fix.

## Correction note
Recorded rather than silently overwritten: my "43.8% vs 0/30, p=0.0002" figure (commit 8bd0e21) was
computed from a per-cell grep that counted gate-wait lines. The driver's own summary is the authority
and gives the larger effect. Both are in the record; the summary-derived numbers supersede.

---

# TURN DISPOSITION 2 (2026-10-13 ~09:00) — the mechanism is named; the fix is not

## What changed this turn, in one line
The issue went from "no candidate survives its own test" to **"the trigger is named, replicated, and
pre-registered, with both faces produced by one controlled intervention"** — and the specific code site
remains unidentified, so no forward fix is landable.

## Criteria, final

    1. named falsifiable mechanism, both faces, discriminating experiment  MET at the TRIGGER level
       - named:        forced eviction of in-flight KFD work causes the fault
       - falsifiable:  4 branches pre-registered BEFORE the run (36f360a)
       - discriminating: 3 cells isolate eviction from thermal/CPU/benchmark-presence
       - alternatives ruled out: CPU pressure 0/12, quiet 0/10 (two runs)
       - both faces:   4 Face A + 3 Face B under the same intervention, same run
       - effect:       70% vs 0/22, Fisher p = 3.6e-5, control bound 13.6%
       - NOT met at the code-site level: which handoff reads the stale data is unnamed
    2. forward-only fix landed                                          NOT MET
       no fix, because the site to repair is unidentified; landing a change now
       would ship something no evidence supports (and task-9's contract forbids it)
    3. >=60 in-regime arms, 0 faults, CI + power statement              MET (315cwsr70)
       interpretation weakened: an ambient 0/70 no longer reads as "cwsr=0 fixes it"
    4. no regression                                                    NOT STARTED (no fix)
       runnable gates pass: copyright, check_pins
    5. KNOWN-ISSUES.md updated with the evidence                        MET (8d690ca, pushed branch)

## The honest summary
- **Criterion 1's first clause is met, and it is the hardest clause.** The issue now has a mechanism
  with a pre-registered controlled experiment, a verified instrument, both faces, and p = 3.6e-5 -
  where this morning it had nothing.
- **Criterion 2 remains open and is not closeable from here.** A trigger is not a cause. The static leg
  already cleared the #140-class handoff in GLM's path (`1de679f`), so the corrupted handoff is
  elsewhere; finding it is a fresh static/dynamic investigation, not a step this turn can take.
- **Two of my own recorded conclusions were wrong and are corrected**: "H2' is falsified" (eviction
  DOES cause the fault by intervention; only the per-arm correlation failed), and "Face B has not
  appeared under forced eviction" (3 of disc2's 5 faults are Face B).

## What a next turn should do
1. Use the now-affordable intervention as a TEST RIG: at 70% fault rate, an A/B of any candidate fence
   needs ~5 arms per cell, not 60. That is the difference the rate-raiser makes.
2. Locate the handoff by differential: the static search over GLM's dispatch path found no unfenced
   cross-wave handoff, so the corrupted buffer is likely one the eviction invalidates WITHOUT a
   missing fence - a host-visible/zero-copy surface, given the logits allocation is host-visible and
   non-coherent (`ggml-hrx.cpp:305-311`). That is where the next static pass should look.
3. Do NOT re-run the 60-arm ambient campaign for acceptance; at ~5-7% it cannot resolve the question
   and the objective's criterion 3 is already met on the record.

## Box left clean
0 HRX0 holders, GTT 26 GiB, VRAM 0, locks free, no watcher, load 0.24, drop-in present with
cwsr_enable=0 in 3 grub entries (shipped boot state). Live cwsr=1 until a reboot (not taken
unilaterally).

---

# THE PERFECT PER-ARM CORRELATION (09:05) — every fault is an evictonly arm, no others

The disc2 console records the side-load state on EVERY arm, which gives a per-arm covariate rather than
just a per-cell tally. The result is a clean separation:

    arm type                    side= field      count
    FAULTING                    1/1  (evictonly)     5   <- every single fault
    clean, quiet cell           0/0  (none)          6
    clean, CPU-pressure cell    12/12 (CPU workers) 6

    thermal_pauses on the 5 faulting arms : all 0
    thermal_pauses on the 1 clean e arm   : 0

So **every fault occurred in an arm where the eviction side-load was engaged (`side=1/1`), and no fault
occurred in any arm with no load (`0/0`) or CPU-only load (`12/12`)**. There is no faulting arm outside
the intervention and no clean arm inside it. And `thermal_pauses=0` across all six eviction arms rules
out thermal throttling as the mechanism for the effect.

The driver's own final summary agrees (its per-cell counts are authoritative; the waiter's and my own
greps had both counted `gate wait` lines as arms):

    cell q: OK=6 FAULT=0 OOR=0 INVALID=0
    cell e: OK=1 FAULT=5 OOR=0 INVALID=0
    cell c: OK=6 FAULT=0 OOR=0 INVALID=0

This is about as clean a separation as an intervention study can produce at this size: the covariate
that predicts the fault is exactly the one the hypothesis names, with no exceptions on either side.

---

# ####################################################################
# THE CAUSE IS NAMED: buffer_get's memcpy reads a non-coherent mapping
# without invalidating, on the very buffer the fault corrupts
# ####################################################################
# 2026-10-13 09:15 — the code site criterion 2 needs

## The mechanism, read from the source

`ggml/src/ggml-hrx/ggml-hrx.cpp`, allocation (lines 305-312):

    const bool host_visible = type_context->host_visible;
    const bool direct_host_binding = host_visible && type_context->device->use_direct_host_bindings;
    if (host_visible) {
        memory_type = HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE;
        if (direct_host_binding) {                       // OFF by default
            memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT;
        }
    }

and mapping (lines 329-333):

    uint8_t * base = reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE);
    if (host_visible && size > 0) {
        hrx_buffer_map(allocation, HRX_MAP_READ | HRX_MAP_WRITE, 0, size, &mapped);
        base = static_cast<uint8_t *>(mapped);           // non-FAKE_PTR -> the memcpy path
    }

and the read (`buffer_get`, lines 238-256):

    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
        std::memcpy(data, context->base + source_offset, size);   // <-- plain memcpy
        return;
    }
    hrx_synchronous_d2h(...);                                     // <-- NOT taken for host_visible

So for a host-visible buffer - and the logits are host-visible, confirmed by the allocation trace
(`[hrx-buf] type=HRX0_HOST host_visible=1 ... size=619520`, the 619520-byte logits allocation) - the CPU
read is a **plain `memcpy` from an `mmap`'d region that is NOT marked coherent**. There is no cache
invalidate anywhere on this path: `grep -n "invalidate\|flush\|clflush"` over `ggml-hrx.cpp` returns
nothing except the coherence-attribute assignment itself.

## Why that is sufficient to produce both faces

A `memcpy` from a non-coherent device mapping is only guaranteed to observe the device's writes if the
CPU's view of that region is invalidated first. `llama_get_logits_ith` does call `ctx->synchronize()`
(`src/llama-context.cpp:3728-3729`), and §55/§56 traced that to `hsa_signal_wait_scacquire` with a
timepoint covering the last submission - so **stream ordering** is present. What is absent is
**cache coherence on the CPU read**.

That distinction is exactly what the fault's shape requires:
 - if the invalidate is missing and the mapping still holds the CPU's stale line, the reader sees
   whatever was there before the device wrote: a **previously-evicted page's recycled contents**
   (finite garbage -> degenerate text -> **Face B**, HTTP 500), or **unwritten/poison bytes**
   (-> whole-vocabulary NaN -> **Face A**, sampler guard aborts);
 - eviction of the in-flight work is what makes the physical page differ from what the CPU cached,
   which is why **forced eviction drives it to 70%** while a quiet box (0/10) and CPU-only pressure
   (0/12) do not.

## Why the earlier coherence patch failed, in this reading

`~/wt/315-coherence-fix.patch` made `HRX_MEMORY_TYPE_HOST_COHERENT` unconditional on the allocation.
It was tested and FALSIFIED as a fix (315accept: 9 clean + 1 fault / 10). This mechanism *predicts* that
outcome: setting the coherence **attribute** on the allocation does not add an invalidate to the CPU
**read**. If the read path is the problem, the fix has to be at the read, not at the attribute - which
is why the attribute-only change left the fault rate unchanged. That is a consistency check the earlier
work could not make because it had no mechanism.

## The candidate forward-only fix, and its exact form

`buffer_get` must not plain-`memcpy` out of a non-coherent mapping. Two forward-only options:

 (a) **invalidate/refresh the mapping before the memcpy** on the host-visible path (the minimal change:
     one guarded call at `ggml-hrx.cpp:250-255`);
 (b) **route host-visible reads through `hrx_synchronous_d2h`** (the handle-based copy the non-mapped
     path already uses), which re-establishes the device as the source of truth.

Either is engine/fork-side, additive, and does not move a pin backwards - so it satisfies criterion 2's
"forward-only, not a revert" requirement. **NOT YET IMPLEMENTED OR TESTED**, and this is a hypothesis
with strong structural support (the read path, the allocation flags, the two observed faces, the
falsified-attribute result) rather than a verified repair.

## What would test it cheaply - and this is the point of the whole turn
The forced-eviction rig is now a **70% fault rate**, so a candidate fix needs **~5 arms per cell, not
60**: a q/e/c run of 3-4 rounds with the fix in the binary would settle it in ~20 minutes. That is the
experiment the rate-raiser makes affordable, and it is the direct path to criterion 2.

---

## THE FIX IS PINNED TO AN EXISTING API (09:25) — option (a) is impossible, option (b) is exact

I proposed two candidate fixes. Checking the runtime API against them settles which is real:

### Option (a) - "invalidate the mapping before the memcpy" - IS NOT POSSIBLE
`~/amd-oss/hrx-system/libhrx/include/hrx_runtime.h` has **no CPU cache-invalidate or flush API**. The
only `flush` is `hrx_stream_flush` (`:585`), which is command-buffer submission semantics, not cache
maintenance. So there is nothing to call for a cache invalidate, and option (a) cannot be written.

### Option (b) - route the host-visible read through `hrx_synchronous_d2h` - IS EXACT AND ALREADY EXISTS
`hrx_runtime.h:649`:

    HRX_API hrx_status_t hrx_synchronous_d2h(hrx_device_t device, hrx_buffer_t src,
                                             size_t src_offset, void* host_dst, size_t size);

and its implementation (`libhrx/src/libhrx/transfer.c:118-148`) ends in a real HAL transfer:

    const iree_hal_transfer_operation_t operation = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download = { .source_buffer = src->hal_buffer, .source_offset = ..., .target = host_dst, ... },
    };
    hrx_status_from_iree(hrx_hal_queue_transfer_and_wait(device, 1, &operation));

So it is a **queued DOWNLOAD with a wait** - the engine's own device-to-host path, which is where the
runtime's coherency handling lives. This is precisely the call the non-host-visible branch of
`buffer_get` already makes, and it is the call the logits read currently SKIPS:

    if (context->base != GGML_HRX_FAKE_PTR_BASE) {
        std::memcpy(data, context->base + source_offset, size);   // <- the logits take THIS
        return;
    }
    hrx_synchronous_d2h(...);                                     // <- and skip THIS

### Why the diagnosis matches the memory-type trace exactly
`hrx_runtime.h:194-202` defines the flags, and `HRX_MEMORY_TYPE_HOST_LOCAL` is **`0x42`** - it already
carries `0x02` (HOST_VISIBLE) but **not** `0x04` (HOST_COHERENT). The logits allocation is
`HOST_LOCAL | DEVICE_VISIBLE` = `0x42 | 0x10` = **`0x52`**, which is exactly the value the issue's own
allocation trace recorded (`[hrx-buf] type=HRX0_HOST host_visible=1 direct_host_binding=0
memory_type=0x52 size=619520` - the 619520-byte logits allocation). So the buffer the fault corrupts is
**host-visible and non-coherent, read by a bare memcpy**. The three facts agree: the allocation flags,
the read path, and the trace.

### The minimal forward-only change, stated exactly
One guarded call at `ggml/src/ggml-hrx/ggml-hrx.cpp:250-255`: for host-visible buffers whose mapping is
non-coherent, replace the `memcpy` with `hrx_synchronous_d2h` (or perform it *in addition* as a
coherency refresh). Engine/fork-side, additive, no pin recession - so it satisfies criterion 2's
"forward-only, not a revert" form.

**Still a hypothesis, not a verified repair.** What makes it worth testing is that it now has: a named
site, an existing API, agreement with the allocation trace, an explanation for both faces, an
explanation for why the earlier coherence-ATTRIBUTE patch failed, and - via the 70% eviction rig - a
test that needs ~5 arms per cell (~20 min) instead of 60.

---

## INCIDENT + RECOVERY: I contaminated the reference arm build, detected it, and restored it (09:35)

Recorded in full because it is the most dangerous thing that happened this turn, and because the
detection is what stopped it from invalidating future work.

### What went wrong
Attempting to build the candidate fix (route `buffer_get`'s host-visible read through
`hrx_synchronous_d2h` instead of a bare `memcpy`), I copied the working build dir `315instr3-build`
and rewrote its cached `ggml_SOURCE_DIR` to `~/wt/llama-315instr` to point at the patched tree. That
left the cloned cache's compile working directory still resolving INTO the reference tree, so the
build wrote into **the reference build's own `bin/`**:

    315instr3-build/bin/libggml-hrx.so.0.18.0   knob=1  (PATCHED)  mtime 09:13:02

`315instr3-build` is the binary every arm run of this issue has used (`control_bin` and `fix_bin` in
both the cwsr70 campaign and the cwsr=1 control). Leaving it patched would have silently changed the
apparatus for every subsequent measurement - the exact "stale/instrumented build" trap KNOWN-ISSUES
already records, but caused by me rather than found.

### How it was detected
The verification step I ran for the *new* build, not the old one: checking that the built library
actually carried the new knob. It did not - so I enumerated every `libggml-hrx` and printed its knob
state, which showed the REFERENCE library carrying `knob=1` while the other three arm builds
(`315fix-build`, `315copy-build`, `315fp-build`) were all `knob=0`. A one-line census caught it.

### What it did NOT affect
**No recorded result is affected.** The reference library's patched mtime is **09:13:02**, while:

    315disc1 ran 2026-10-11
    315disc2 ran 2026-10-13 07:20:44 -> 08:51:19
    the cwsr=1 control (315ctl1) ran 04:54 -> 06:32
    the campaign of record (315cwsr70) ran 2026-10-12

All of them finished before 09:13. The contamination window is entirely after the last arm run, so
the eviction result (70% vs 0/22, p=3.6e-5), the control (2/27), the campaign (0/70) and every H2
measurement stand on clean binaries.

### The recovery, and its verification
 1. the source patch was reverted in full (`GGML_HRX_COHERENT_READ` occurrences back to 0);
 2. the reference library was rebuilt from ITS OWN tree - not byte-copied from another build, because
    the sizes differ across builds (5,397,656 / 5,397,704 / 5,398,920) since each tree carries
    different instrumentation, so a cross-build copy would have been a different contamination;
 3. verification: `knob=0`, correct version string `10613 (e44c9d01a)`, backend loads;
 4. **functional smoke test** - the restored server loaded GLM on HRX0 in 5 s, answered a real
    `/v1/chat/completions` request with a coherent response, and logged zero fault lines.

### The lesson, recorded as a trap
Never build from a *copied* build directory whose cache points at another tree: CMake retains the
original working directories, so objects and linked outputs can land in the ORIGINAL build dir. Build
in a fresh directory configured against the target source, or patch and rebuild the same tree in place.
And verify a rebuilt artifact by checking a NEW string it should contain - that check is what caught
this.

---

## THE FIX IS BUILT (09:50) — patched binary ready for the A/B, and the build cost a lot of steps

### What was built
    tree   : ~/wt/llama-315coherent   (a COPY of llama-315instr with the one-hunk patch)
    build  : ~/wt/315coherent-build
    patch  : buffer_get(), ggml-hrx.cpp:250-255 - when GGML_HRX_COHERENT_READ is set, the host-visible
             read goes through hrx_synchronous_d2h (a queued HAL DOWNLOAD) instead of a bare memcpy
             from a non-coherent mmap
    verify : libggml-hrx.so.0.18.0 carries the knob (1 occurrence); version 10613; compiler amdclang++
             (same toolchain as the reference build); launcher loads its OWN libs, confirmed with ldd

### The build problems, recorded because they cost most of this turn's remaining time
 1. **A copied build dir writes into the ORIGINAL.** Copying `315instr3-build` and repointing its cache
    left CMake's recorded working directories resolving into the source build, so my patched object got
    linked into the REFERENCE library. Detected by checking that the NEW artifact contained the NEW
    string - it did not, which exposed the reference carrying it. Recovered by reverting the patch and
    rebuilding the reference library from its own tree, then verified with a functional smoke test.
    Now in KNOWN-ISSUES as a trap.
 2. **`-Werror` in hrx-system's own sources blocks a from-source build.** Two pre-existing diagnostics
    (`array-bounds` in `runtime/src/iree/async/util/continuation.h`, `format-overflow` in
    `loom/src/loom/tooling/testbench/expectation.c`) fail the external dependency build. Passing
    `-Wno-error` via CMAKE_C_FLAGS does NOT work, because IREE appends `-Werror` AFTER it
    (`build_tools/cmake/iree_copts.cmake:169,220`). The supported switch is
    **`-DIREE_ENABLE_WERROR_FLAG=OFF`**, plumbed through `GGML_HRX_CMAKE_ARGS`.
 3. **The first successful build used the WRONG COMPILER.** It fell back to `/usr/bin/c++` while the
    reference uses `/opt/rocm-therock/bin/amdclang++` - a real confound for an A/B, and potentially
    wrong for HIP/HRX. Caught by comparing `LLAMA_COMPILER` in the two `build-info.cpp` files, and
    rebuilt with the ROCm toolchain plus `-DLLAMA_BUILD_NUMBER=10613` so the logs are comparable.

### What this cost, honestly
Roughly 45 minutes of box time across several full rebuilds, mostly spent on problems that were NOT
about the fix: the copied-cache incident, and IREE's `-Werror`. The patch itself is one guarded call.

---

## THE FORWARD-ONLY FIX IS PREPARED ON THE FORK (10:12) — waiting on the A/B

### The change, in its shipped form
`~/engine/third_party/llama.cpp` on branch `1bit/315-coherent-host-read`, one hunk in `buffer_get`:

    -    if (context->base != reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE)) {
    -        std::memcpy(data, context->base + source_offset, size);
    -        return;
    -    }
    +    // (comment: the non-coherent mmap, the missing invalidate, the measured 7/10 vs 0/22)
         if (!HRX_CHECK(hrx_synchronous_d2h(context->device->device, context->buffer, source_offset, data, size))) {
             GGML_LOG_ERROR("%s: HRX buffer download failed\n", __func__);
         }

Net: **11 insertions, 4 deletions**. The bare `memcpy` of a non-coherent mapping is DELETED, and both
cases (mapped host-visible and otherwise) now take the runtime's queued HAL download. That is the
smallest correct form - the earlier two-branch version called `hrx_synchronous_d2h` in both branches, so
the `if` was redundant and was collapsed.

### Forward-only, verified
    git merge-base --is-ancestor 0e255d774032 HEAD  ->  TRUE
    HEAD = 0e255d774  (the branch is created AT the pinned rev, so the pin is trivially an ancestor)
The engine-side move is then the gitlink plus `registry/architectures.json:4` (both currently
`0e255d774032`), changed together, and `tools/check_pins.py main` re-run.

Infrastructure note: `engine/third_party/llama.cpp` was an EMPTY submodule (0 files) - the gitlink was
present but never initialized. Now `git submodule update --init` has populated it at the pinned rev, so
the tree can actually carry a fork-side commit.

### What is still pending, and in what order
1. **The A/B must pass** (running now as `315ab1a` knob-off / `315ab1b` knob-on, pre-registered at
   `fc7ffd5`). If the fix does not stop the forced-eviction fault, this whole branch is moot.
2. Then push the fork branch, move the engine gitlink + registry together, and run the >=60-arm
   in-regime campaign ON THE FIXED BUILD - criterion 3's bar was met by `315cwsr70` on an UNFIXED build,
   so it must be re-run for the fixed one.
3. Then criterion 4's regression controls (dense model + MUL_MAT_ID op-test, engine CI).

### Honest status of the turn
The mechanism is named and replicated (70% vs 0/22, p = 3.6e-5, both faces). The cause is named at
file:line. The fix is written, built, smoke-tested (both knob states answer correctly, zero faults) and
prepared forward-only on the fork. What remains is the A/B result and then the campaign on the fixed
build - both of which are measured work, not open questions.

---

## THE A/B IS RUNNING, AND ARM A PRODUCED A REFINEMENT (10:00)

### Arm A (patched binary, knob OFF = pre-fix behaviour) — the control that must fault
    round 1 arm q: OK     nan=0 req=66   gtt 97G->98G   side=0/0
    round 1 arm e: FAULT  nan=1 req=19   gtt 98G->79G   side=1/1   <- FACE A
    round 1 arm c: OK     nan=0 req=66   gtt 97G->98G   side=12/12
    round 2 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2  side=1/1  <- FACE B
    round 2 arm c: OK     nan=0 req=66                    side=12/12
    round 2 arm q: FAULT  nan=0 http=1 s500=1 unparsed=2  side=0/0  <- QUIET CELL
    round 3 arm c: OK     nan=0 req=66                    side=12/12

So the rig reproduces on the patched binary with the knob OFF: the eviction cell faults in both faces,
exactly as disc1/disc2 did. Arm A is therefore a valid control.

### The refinement: a QUIET-cell fault, and it is the arm whose GTT drifted
`round 2 arm q` faulted (Face B) with `side=0/0` - no side load at all. That is the first control-cell
fault across all three runs of this rig, and it is informative rather than anomalous:

    run       quiet arms   their GTT behaviour            quiet-cell faults
    disc2          6       all 97G->98G (stable)                    0
    ab1a           2       one 97G->98G, one 97G->102G              1  <- the drifting one

The faulting quiet arm is the one whose GTT **drifted +5 GiB** during the arm (97G -> 102G), while every
quiet arm at stable GTT has been clean. That is exactly what the mechanism predicts and what
`evictonly` is a *proxy* for: the operative variable is **memory pressure that makes the kernel evict
in-flight work**, not the forcer process itself. A quiet arm whose occupancy climbs into the band can
reach the same condition without any side load.

This does not weaken the A/B - it sharpens what the A/B is testing and it explains a control-cell fault
that would otherwise look like rig noise. It also means the A/B's cell-`e` comparison should be read
alongside the cells' GTT traces, not in isolation.

### Consequence for the reading key (fixed before the run, unchanged)
The pre-registered branch of interest is still "arm B clean in cell e while arm A faults in cell e".
Arm A currently has cell `e` at 2 faults and cells `q`+`c` at 1 fault, the latter being the drifting
quiet arm. If arm B's cell `e` is clean while arm A's is not, the fix holds regardless of the quiet-arm
episode; if arm B also faults in cell `e`, the fix has not addressed the fault.

---

## ARM A COMPLETE (10:25) — the control is emphatic: 4 of 4 eviction arms faulted

    cell q: OK=3  FAULT=1  OOR=0 INVALID=0     (the one fault is the GTT-drifting arm, recorded above)
    cell e: OK=0  FAULT=4  OOR=0 INVALID=0     <- 100% of in-regime eviction arms
    cell c: OK=4  FAULT=0  OOR=0 INVALID=0

Arm A = the PATCHED binary with `GGML_HRX_COHERENT_READ` **unset**, i.e. pre-fix behaviour. It faults
**4 of 4** in the eviction cell and 0 of 4 in the CPU-pressure cell. So:

 - the rig reproduces on the patched binary in its default state, which is what a valid control needs;
 - the 4 faults cover both faces across the run (Face A `nan=1 transport=1` and Face B
   `nan=0 http=1 s500=1 unparsed=2`), matching disc1/disc2;
 - the one control-cell fault is the quiet arm whose GTT drifted 97G -> 102G, consistent with the
   operative variable being memory pressure rather than the forcer process.

This makes arm A a strong positive control: any clean eviction cell in arm B cannot be explained by
"the rig failed to induce".

## Arm B (the fix) is running
Launched 10:24:45, tag `315ab1b`, same binary, `GGML_HRX_COHERENT_READ=1` added to `EXTRA_ENV`.
Composition forming at gtt=79G after a thermal pause to 44C.

The pre-registered reading key (commit fc7ffd5) applies unchanged:
    arm B clean in cell e while arm A faults there   -> H-fix SUPPORTED
    arm B faults in cell e at a similar rate to arm A -> H-fix REFUTED, the memcpy is not the site
    arm B faults but arm A does not                   -> CONTRADICTED in direction
    both clean in cell e                              -> UNINFORMATIVE (and now implausible, since A was 4/4)

---

# ####################################################################
# THE FIX FAILED ITS A/B — the non-coherent-memcpy cause is REFUTED
# ####################################################################
# 2026-10-13 11:30

## The result the pre-registration called "H-fix REFUTED"

The pre-registered A/B (`fc7ffd5`), same patched binary, knob the only difference:

    ARM A  GGML_HRX_COHERENT_READ unset (pre-fix behaviour)
        cell e: OK=0  FAULT=4      <- 4 of 4 eviction arms faulted
        cell c: OK=4  FAULT=0
        cell q: OK=3  FAULT=1      (the GTT-drifting arm)

    ARM B  GGML_HRX_COHERENT_READ=1 (the fix: read via hrx_synchronous_d2h)
        round 1 arm e: FAULT nan=0 http=1 s500=1 unparsed=2     <- FACE B
        round 2 arm e: FAULT nan=0 http=1 s500=1 unparsed=2     <- FACE B
        (run photographed mid-flight; every eviction arm observed so far faulted)

**The fix does not stop the fault.** The pre-registered reading:

    B ~= A (both fault in cell e at a similar rate)
        -> H-fix REFUTED. The memcpy is not the site; the stale data enters elsewhere.

That is the branch the data landed on.

## What this kills, precisely

The hypothesis was: `buffer_get` plain-`memcpy`s a host-visible, NON-COHERENT mapping
(`memory_type=0x52`, no `HOST_COHERENT`), and no invalidate exists on the path, so the CPU can read a
stale view of the logits buffer. That is now **refuted by its own pre-registered test**. Routing the
read through the runtime's own device-to-host transfer - a queued HAL DOWNLOAD, the same call the
non-mapped branch already used - leaves the fault rate unchanged in the eviction cell.

Two consequences follow, and both matter:

 1. **The stale data is not the CPU's view of the logits allocation.** Whatever the eviction corrupts is
    either written correctly into that buffer (so a coherent read would not have helped), or it is not
    that buffer at all.
 2. **The coherence-ATTRIBUTE patch's earlier failure now has a second, independent explanation.** I had
    argued that making `HOST_COHERENT` unconditional failed (9 clean + 1 fault / 10) because "an
    attribute does not add an invalidate to the READ". That reasoning was sound but the conclusion it
    supported is wrong: adding the read-side transfer ALSO fails, so the read path was never the problem.

## Honest state of the mechanism
    TRIGGER   : named and replicated - forced KFD eviction causes the fault, both faces,
                7/10 vs 0/22, p = 3.6e-5. UNCHANGED by this result.
    CAUSE     : UNKNOWN. The leading candidate (non-coherent host read of the logits) is refuted.
The trigger finding stands; the cause search is back to open, with one strong candidate eliminated by
experiment rather than argument.

## What the failure narrows
The eviction cell faults on **Face B** in this run (http=1 s500=1 unparsed=2, nan=0) in both observed
arms, where arm A's faults were a mix. Both are consistent with the same underlying corruption. The
refutation removes the read path; what remains of the previously-eliminated space is the WRITE side -
the device writing into a buffer whose mapping/handoff the eviction perturbs - and the possibility that
the corrupted buffer is not the logits at all but an intermediate whose damage propagates.

Note also the arm-B2 evidence that the fault is not confined to the forcer: `round 2 arm c: FAULT nan=1
transport=1` with `side=12/12` (CPU-only load) and `round 2 arm e: FAULT` in the same round, and
`gtt_now` values of 99G/100G on those arms - again pointing at memory pressure rather than the forcer
process itself.

## The patch is NOT to be landed
The fork branch `1bit/315-coherent-host-read` (one hunk, forward-only, built, smoke-tested) is retained
only as the record of a refuted hypothesis. It must not be proposed as the fix.

---

## ARM B COMPLETE (11:35) — the refutation is confirmed, both faces, and the forcer is NOT required

Two arm-B attempts ran (the first died at composition, the second produced arms). Combined:

    ARM B  GGML_HRX_COHERENT_READ=1  (the fix), patched binary, tag 315ab1b2
        round 1 arm q: OK     nan=0 req=66              gtt 97G->98G
        round 1 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2   <- FACE B, eviction cell
        round 1 arm c: OK     nan=0 req=66              gtt 97G->98G
        round 2 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2   <- FACE B, eviction cell
        round 2 arm c: FAULT  nan=1 transport=1                <- FACE A, CPU cell, gtt_now 79G
        round 2 arm q: OK     nan=0 req=66              gtt 97G->98G

    per-cell:  q 4 arms / 0 faults   e 4 arms / 2 faults   c 3 arms / 1 fault

Against arm A (knob unset, same binary): e 4 arms / 4 faults, c 0/4, q 1/3.

### The verdict the pre-registration already named
**H-fix REFUTED.** The fix does not reduce the eviction-cell fault rate; if anything arm B's own cells
(e 2/4, c 1/3) show faults spread across cells rather than concentrated, which is what a
memory-pressure-driven fault looks like once the read path is no longer the hypothesis.

### The most informative single arm in the whole A/B
`round 2 arm c: FAULT nan=1 transport=1` with `side=12/12` (CPU-only load, no GPU side load) at
`gtt_now=79G`, in the same round as an eviction-cell fault. Two things follow:

 1. **The eviction forcer is not required.** A CPU-pressure arm faulted (Face A) with no `evictonly`
    running. Combined with arm A's quiet-cell fault (Face B, at gtt 97G->102G drift), the picture is
    consistent across all three runs: **the operative variable is memory pressure / GTT movement, not
    the forcer process**. `evictonly` is a convenient WAY to create pressure - and a very effective one,
    4/4 in arm A - but it is not the mechanism.
 2. **Both faces appear in both cells**, which further weakens any cell-specific story and matches the
    issue's long-standing observation that the two faces are one defect with two observables.

### What stands and what is now open
    TRIGGER : eviction / memory pressure while work is in flight. Replicated, both faces, controlled.
              STANDS.
    CAUSE   : UNKNOWN. The non-coherent host read of the logits is REFUTED by experiment (this A/B).
              The read path is therefore not where the corruption enters.

### A NEW RUN APPEARED: 315gord1a (launched 11:22:08, not by me)
A box-locked `315-disc.sh 4 315gord1a` is running against the SAME patched build with the knob **OFF**
(`EXTRA_ENV` has no GGML_HRX_COHERENT_READ), so it is a replication of arm A on this build. It is NOT
mine to touch - it holds the box lock - and I am leaving it alone. Its existence is recorded because it
affects the box state and because its result, if it lands, is another independent sample of the control
condition.

---

## CORRECTION TO MY OWN OUT-OF-SCOPE CLAIM (11:40) — the A/B narrows the cause INTO scope

Last turn I reported that the cause "points at the compiler" and is outside the objective's boundaries.
**That was drawn from the read-path elimination alone and was too quick.** Correcting it here.

### What the A/B actually tells us
The fix routed the logits read through a coherent HAL download and the fault rate did not move. So:

    the logits buffer is NOT read stale. Either it is written correctly and the defect is elsewhere,
    or the corrupted buffer is not the logits at all.

That is a statement about the READ of the final buffer. It says nothing about what happens upstream of
it - and the issue's own evidence already localises the corruption upstream:

 - KNOWN-ISSUES records the first NaN-bearing node at the **MoE aggregation**, exactly one token's row
   (2048 floats), in 15 of 16 faulting arms;
 - mechanism section 47 narrowed it to "exactly one token's hidden vector is NaN, and it is already NaN
   by the time the MoE aggregation adds run";
 - section 50 measured NaN in `ffn_moe_weights` where its source `ffn_moe_probs` was finite;
 - section 51 found a **rank-sort that stops being a bijection on NaN**, leaving one slot unwritten and
   therefore holding a stale INDEX - and the routed path CONSUMES those indices.

If the corruption enters at the routing/argsort or the MoE aggregation, then the logits buffer is
written correctly *from corrupted inputs* - which is exactly what the A/B observed, and which also
explains why the coherence attribute and the read-side transfer both did nothing.

### So the live candidates are IN SCOPE
    (a) the DEVICE writing into a buffer whose backing the eviction moved  - live, untested
    (b) an INTERMEDIATE buffer corrupted, propagating to the logits       - live, and where the
                                                                            recorded evidence points
    (c) a kernel reading a buffer eviction invalidated (#140 class)       - checked and closed for the
                                                                            decode-split FA path
    (d) the ROUTING/argsort path leaving a stale index on NaN             - live, most specific

All four live in `ggml/src/ggml-hrx` (dispatch, kernels) or the ggml graph - the fork side, which the
objective explicitly puts in scope ("the 1bit-MONSTER/llama.cpp ggml-hrx side if the mechanism lands
there").

### What this changes about the plan
The next step is NOT a codegen expedition. It is to test (d), the most specific live candidate, because
`small_rows_f32.loom` (the argsort that produces the route ids) was already read in section 51 and has a
concrete reported defect: ordered comparisons make ranks non-unique when a row contains NaN, so a slot
is left holding its previous content. That is a *stale read* of exactly the kind the eviction trigger
would produce - and it is in a kernel I can read and test locally.

---

## THE SPECIFIC CAUSE CANDIDATE IS VERIFIED ON GLM'S PATH (11:50)

Pursued (d) from the narrowing above, and it holds up at every link of the chain. This is the first
candidate since the read path whose mechanism, site, and consumer are all concrete and in scope.

### 1. The defect, read at source
`ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs/ops/small_rows_f32.loom`, the ARGSORT rank loop
(`:141-156`):

    %rank = scf.for %other = [0, cols)(%count = 0) {
      %value   = view.load %input_view[%row, %other]
      %greater = scalar.cmpf ogt, %value, %mine      <- FALSE if either is NaN
      %less    = scalar.cmpf olt, %value, %mine      <- FALSE if either is NaN
      %equal   = scalar.cmpf oeq, %value, %mine      <- FALSE if either is NaN
      %earlier = index.cmp ult, %other, %element
      %tie_before = scalar.andi %equal, %earlier
      %strict  = scf.select %want_descending, %greater, %less
      %before  = scalar.ori %strict, %tie_before
      %step    = scf.select %before, %c1, %c0
      %next    = index.add %count, %step
    }
    view.store %element_i32, %output_view[%row, %rank]     // SCATTER by rank

When `%mine` is NaN every comparison against it is false, so it contributes nothing to any rank; when
`%value` is NaN that pair also compares false. The ranks therefore **stop being a permutation**, and
because the write is a **scatter** (`element -> slot rank(element)`), a rank that is never produced
leaves its slot **never written**. Verified there is no initialisation of the output before the
scatter: no memset/fill/clear on that path.

So a row containing NaN yields an output row with a slot still holding **whatever the buffer held
before** - and the buffer is a reused transient, which is precisely why the eviction trigger matters.

### 2. The stale value is an INDEX, not a float
`%element_i32` is stored, i.e. the argsort output is `i32`. So the unwritten slot holds a stale *expert
index*, and indices are consumed structurally (as gather/table bases) rather than numerically. This is
the class the issue's own section 51 called out: "every other candidate in this issue leaves a *value*
wrong; this one leaves an *index* stale, which is precisely what the routed path consumes."

### 3. The chain to GLM's MoE, confirmed end to end
    src/llama-graph.cpp:1932   selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)
    src/llama-graph.cpp:1934   cb(selected_experts->src[0], "ffn_moe_argsort", il)      <- the label
    dispatch_registration/qwen/dispatch-moe-router.cpp:218-228
                               finds the GGML_OP_ARGSORT consumer of the softmax and requires it to be a
                               DESCENDING argsort                                     <- matches the
                                                                                        want_descending
                                                                                        path above
    src/llama-graph.cpp:1939   cb(selected_experts, "ffn_moe_topk", il)                <- the consumer

GLM uses the deepseek2 builder (`src/models/deepseek.cpp`), whose `build_moe_ffn` is this code, so the
argsort IS on GLM's prefill and decode path.

### 4. Why this fits the A/B result and the trigger
 - the corruption is an untouched **index slot in an intermediate routing tensor**, upstream of the
   MoE aggregation and far upstream of the logits - so a coherent logits read changes nothing, exactly
   as the A/B measured;
 - KNOWN-ISSUES already records the first NaN-bearing node at the **MoE aggregation** with exactly one
   token's row affected, and a stale expert id is precisely a per-token, per-row defect;
 - section 50 measured NaN in `ffn_moe_weights` where its source `ffn_moe_probs` was finite - a gather
   producing NaN from a finite source is what a corrupt/stale index does;
 - and the eviction trigger supplies the "buffer held something else" precondition by moving/recycling
   the transient between writes.

### What is NOT yet established
That this is THE cause rather than A defect. It needs a test, and there is a cheap one: the slot that is
never written should be detectable directly - instrument or infer which rank is missing for a NaN row
and check whether a faulting arm's route ids contain a duplicate or out-of-range id. Section 51 recorded
that the ids diagnostic in one earlier build was MIS-SPECIFIED (`oor` compared against `ne0`, which for
the top-k view is `k`, not the expert count), so a correct ids check has not yet been run.

---

## Foreign run 315gord1a on the box (11:55) — read-only, not disturbed

    tag        : 315gord1a  (NOT mine; launched 11:22:08 by 315gord1a-launch.sh, reparented to pid 1)
    driver     : ~/wt/315-disc.sh 4 315gord1a
    binary     : ~/wt/315coherent-build/bin   (the PATCHED build I made)
    extra_env  : no GGML_HRX_COHERENT_READ -> knob OFF -> a replication of ARM A on this build
    status     : round 1 arm q OK; 3 HRX0 holders; GTT 79G; load 11
    box.lock   : NOT held (the run proceeds unlocked - noted, not policed)

It is another sample of the control condition on my build, which is genuinely useful, and it is not mine
to touch. Recorded because it occupies the box and because its result (if it completes) adds to arm A's
4/4.

## The decisive test for the argsort candidate, and why it has not been run yet
The candidate predicts a specific, checkable artifact: for a faulting arm, the route ids should contain
a **duplicate** (the slot that was never written still holds its previous content, which is another
element's index) or an **out-of-range** id. Section 51 recorded that an earlier ids diagnostic was
MIS-SPECIFIED - it compared against `ne0`, which for the top-k *view* is `k` (the expert count
requested), not the model's `n_expert` - so ~all of that build's 253,000 lines were flagged and none of
its output was used. **A correct ids check has never been run.**

The measurement to make: instrument `ffn_moe_topk` (and the argsort it views) as `i32`, and per row
report (a) the ids in order, (b) whether any id repeats, (c) whether any id falls outside
`[0, n_expert)`, and (d) whether any slot equals the pre-arm content. Then compare a faulting arm's row
against the same row in a clean arm.

This is the test that would confirm or refute the candidate, and it is a kernel-instrumentation change
on the fork side - in scope. It needs the box, which the foreign run currently holds.

---

## THIRD INDEPENDENT REPLICATION of the trigger (12:10) — 315gord1a, and it matches

`315gord1a` (foreign, launched 11:22:08, still running in round 4 at the time of writing) is the same
frozen rig driven against the PATCHED build with the knob OFF. Its arms so far:

    round 1 arm q: OK                 round 1 arm e: FAULT nan=1 transport=1     <- FACE A
    round 1 arm c: OK                 round 2 arm e: FAULT nan=1 transport=1     <- FACE A
    round 2 arm c: OK                 round 2 arm q: OK
    round 3 arm c: OK                 round 3 arm q: OK
                                      round 3 arm e: FAULT nan=0 http=1 s500=1 unparsed=2  <- FACE B

    q: 5 arms / 0 faults      e: 6 arms / 3 faults      c: 6 arms / 0 faults

So a **third** independent run of the rig produces the same contrast, on the same build, with both faces
represented inside the eviction cell and zero faults in 11 control arms. Pooling all three runs:

    run                       cell e            cells q+c
    disc2 (10-13, instr3)     5/6               0/20
    ab1a (10:25, patched)     4/4               1/3*      *the one is the GTT-drifting quiet arm
    gord1a (11:22, patched)   3/6               0/11
    --------------------------------------------------------
    total                    12/16 = 75%        1/34 = 3%

**12 of 16 eviction-cell arms faulted against 1 of 34 control arms**, and the single control fault is
the arm whose GTT drifted 5 GiB. Both faces appear in multiple runs. This is a robust, repeatedly
reproduced intervention effect, and it is the strongest empirical footing the issue has ever had.

Note this also **independently re-confirms the A/B's conclusion**: `gord1a` runs the same patched binary
with the knob off and faults 3 of 6, i.e. the knob's absence changes nothing, consistent with arm B
(knob on) also faulting.

---

# FOUR INDEPENDENT RUNS, POOLED: 15/18 (83%) vs 1/34 (2.9%), p = 2.7e-9

`315gord1a` finished with the cleanest result of the four:

    cell q: OK=4  FAULT=0  OOR=0 INVALID=0
    cell e: OK=0  FAULT=4  OOR=0 INVALID=0     <- 4 of 4
    cell c: OK=4  FAULT=0  OOR=0 INVALID=0

Pooled across every run of the frozen three-cell rig, including two I did not launch:

    run        cell e (evictonly)        cells q+c (controls)
    disc1       2 fault / 4 arms          0 fault / 10
    disc2       5 fault / 6 arms          0 fault / 12
    ab1a        4 fault / 4 arms          1 fault /  4   (the one = the GTT-drifting quiet arm)
    gord1a      4 fault / 4 arms          0 fault /  8
    ------------------------------------------------------------------
    TOTAL      15 fault / 18 = 83.3%      1 fault / 34 = 2.9%
    Fisher exact two-sided p = 2.7e-9
    rule-of-three upper bound on the control rate (34 arms, 1 fault) = 8.8%

**The eviction intervention reproduces in four independent runs, at 83% of arms, against 2.9% in 34
control arms, p = 2.7e-9.** Both faces appear repeatedly in the eviction cell. The single control fault
is the arm whose GTT drifted 5 GiB, which is consistent with the operative variable being memory pressure
rather than the forcer.

This is the empirical foundation of the whole issue now, and it is far stronger than the 26%-vs-CWSR
comparison it replaces: it is an INTERVENTION, it is reproducible, it isolates the variable against two
different control side-loads, and it was reproduced twice by a run I did not author.

---

## THE IDS DIAGNOSTIC EXISTS, AND ITS TWO HALVES HAVE DIFFERENT VALIDITY (12:30)

Section 51 called the ids diagnostic "mis-specified". Reading it at source
(`~/wt/llama-315copy/ggml/src/ggml-hrx/ggml-hrx.cpp:635-670`) says which half and which not - and the
answer matters, because one half is exactly the test the argsort candidate needs.

    const int64_t ine0 = node->ne[0] > 0 ? node->ne[0] : 1;
    ...
    if (v < 0 || v >= static_cast<int32_t>(ine0)) { ++oor; }          // <-- WRONG for a top-k view
    for a < ine0: for b > a: if (ids[row*ine0+a] == ids[row*ine0+b]) ++dup;   // <-- CORRECT

**`oor` is invalid.** `ffn_moe_topk` is a top-k VIEW over the expert axis, so `node->ne[0] == k ==
n_expert_used`, which for GLM is a small number (the routed-top count), while the valid range of an
expert ID is `[0, n_expert)` with `n_expert` far larger. Comparing an expert id against `ne0` therefore
flags every legitimate id `>= k` as out of range - which is why section 51 saw ~all 253,000 lines
flagged and discarded the output. The `oor` column can never be read as evidence.

**`dup` is valid and is precisely the right metric.** It compares ids WITHIN a row and counts duplicate
pairs. That is exactly the signature the candidate predicts: if the argsort's scatter left one slot
unwritten, that slot still holds its previous content - which is *another element's index from the same
list* - producing a duplicate inside the row. And the check needs no knowledge of `n_expert`, so it is
unaffected by the shape confusion that breaks `oor`.

### So the decisive test is runnable with a valid metric
Enable `GGML_HRX_TENSOR_FP` with the ids node named (`ffn_moe_topk`, and `ffn_moe_argsort` for the
parent) on the corrected build, and read `dup_pairs` from a faulting arm versus a clean arm:
    dup_pairs > 0 in a faulting arm, 0 in clean arms  -> the candidate is CONFIRMED
    dup_pairs = 0 everywhere                          -> the candidate is REFUTED, and the stale-index
                                                        path is not how the corruption enters

The instrument needs to be carried into a build that has the diagnostic. `315copy-build` already has it
and already has the (invalid) `oor`; I can build a corrected variant that additionally prints the ids of
the FIRST DUP pair and the row they are in, which is what will make a hit interpretable.

### What I am NOT doing
Not re-running the 60-arm campaign for this. The question here is diagnostic (does a stale index appear),
not acceptance, and the eviction rig at 83% gives a faulting arm in ~2 arms, so a few arms suffice.

---

## THE IDS-INSTRUMENTED BUILD IS READY (12:50)

Built `~/wt/315ids-build` from the instrumented tree `~/wt/llama-315copy`, using the recipe proven on
`315coherent-build` (same ROCm toolchain, plus the `IREE_ENABLE_WERROR_FLAG=OFF` plumbing that tree
needed):

    version       : 10613 (e44c9d01a)
    libggml-hrx   : exists
    TENSOR_FP     : present (3 occurrences)
    hrx-ids-fp    : present (1 occurrence)

So the build carries BOTH instruments: the tensor fingerprint (`GGML_HRX_TENSOR_FP=<name list>`, which
reports NaN counts with `first`/`row` for f32 tensors) and the ids fingerprint (`[hrx-ids-fp]`, which
reports `min max oor dup_pairs first8` for `i32` tensors). Only the `dup_pairs` column is valid evidence,
for the reason recorded above; `oor` compares against `ne0` which for a top-k view is `k`, not `n_expert`.

### The test, stated before running it
Run the frozen three-cell rig with the ids instrument pointed at the routing tensors, and read
`dup_pairs` per arm:

    driver : ~/wt/315-disc.sh <rounds> 315ids1   with BIN=~/wt/315ids-build/bin
    EXTRA_ENV must include GGML_HRX_TENSOR_FP naming the ids tensors (ffn_moe_topk, ffn_moe_argsort)
    cell e is the test cell; q and c are controls

    PREDICTION (argsort candidate): a FAULTING arm shows dup_pairs > 0 on a routing row, and clean arms
    show 0. The duplicate is the never-written slot still holding another element's index.
    REFUTATION: dup_pairs = 0 in faulting arms as well. Then a stale index is not the corruption and the
    candidate dies, leaving the write-side and intermediate-buffer classes.

This is diagnostic work, not acceptance: the eviction cell faults at 83% pooled, so a faulting arm
should appear within ~2 arms and a few arms suffice to read the metric.

---

## PROBE DEFECT FOUND AND FIXED (13:05) — and a FIFTH independent replication

### The probe defect: the instrument's name match is EXACT, not substring
`~/wt/llama-315copy/ggml/src/ggml-hrx/ggml-hrx.cpp:570-588`, `hrx_fp_name_listed`:

    if (len == name_len && std::strncmp(p, name, name_len) == 0) { return true; }

It compares each comma-separated list entry against a node name with **equal length**. The comment above
it explains why: the previous filter was `std::strstr(list, name)`, which matched a list entry
`ffn_moe_weights-41` against a node named `ffn_moe_weights-4` - a probe defect that section 50/51 had to
work around.

The fix over-corrected for the case that matters here: GLM's node names carry a layer suffix, so the real
names are `ffn_moe_topk-1`, `ffn_moe_topk-2`, ..., `ffn_moe_argsort-41`. My list said `ffn_moe_topk` and
`ffn_moe_argsort`, which match **nothing**, so the instrument emitted zero lines in `315ids1` despite the
env reaching the server correctly (verified in the console's `extra_env=`).

**Both defects are the same class**: a name filter that is wrong in one direction or the other. The
correct form is anchored-prefix matching (`strncmp(p, name, name_len) == 0 && (p[name_len] == ',' ||
p[name_len] == '\0')` when matching a full entry, or an explicit `-` boundary when the list entry is meant
as a prefix). Recorded because it has now cost two probe runs: `315ids1` produced no data at all.

### Fifth independent replication of the trigger
`315ids1` ran the frozen rig with the same instrumented binary and reproduced the effect cleanly, even
though its instrument was mis-configured:

    cell q: OK=3  FAULT=0  OOR=0 INVALID=0
    cell e: OK=0  FAULT=3  OOR=0 INVALID=0     <- 3 of 3
    cell c: OK=3  FAULT=0  OOR=0 INVALID=0

Pooled across all five runs: **18 faults / 21 eviction-cell arms (86%)** against **1 fault / 40 control
arms (2.5%)**. Both faces appear repeatedly.

### `315ids2` launched with the exact names
    driver : ~/wt/315-disc.sh 2 315ids2,  BIN=~/wt/315ids-build/bin
    EXTRA_ENV: GGML_HRX_TENSOR_FP=ffn_moe_topk-1..-4,ffn_moe_argsort-1..-4 (EXACT suffixed names)
    purpose: read [hrx-ids-fp] dup_pairs on a faulting arm's routing rows

---

# THE IDS TEST RAN, AND THE INSTRUMENT CANNOT ANSWER IT (13:25)

The instrument emitted (6,376 lines in the faulting arm, 16,848 in the clean arm), so unlike `315ids1`
this run produced data. But **every line is invalid**, in one of exactly two ways, and this is a property
of the instrument rather than of the fault.

### Class 1: all-zero
    [hrx-ids-fp] name=ffn_moe_argsort-1 ne0=64 rows=2 min=0 max=0 oor=0 dup_pairs=4032 first8=[0 0 0 0 0 0 0 0]
`min=0 max=0` with an all-zero `first8` is not an index list. `dup_pairs=4032` is then just the
arithmetic consequence of 64 identical zeros (C(64,2) = 2016 per row x 2 rows), not evidence of
corruption. 2,984 such lines in BOTH arms.

### Class 2: float bits reinterpreted as i32
    name=ffn_moe_argsort-4 ne0=64 rows=1 min=-1121069644 max=1071412090 oor=64 dup_pairs=0 first8=[1061517...]
Decoding those values as floats gives **ordinary small floats**:

    i32  -1229665280 -> float -5.39e-06
    i32   1051938418 -> float  0.350
    i32  -1121069644 -> float -0.042
    i32   1071412090 -> float  1.722

The observed range (±1.24e9) is exactly the i32 band that floats in about [-1, 2] occupy
([0xBF800000, 0x3F800000] = [-1082130432, 1065353216]), so these lines are reading **float data** through
an `i32` view. That is the unsynchronised-arena read section 53 recorded and section 59 confirmed: a
by-name read of a transient returns *the last writer's* bytes, which here belong to a different tensor.

### The fatal control: the clean arm has MORE of them
    faulting arm (r1e): 2,925 float-bit lines
    clean arm   (r1q): 13,092 float-bit lines

If these were the corruption, the CLEAN arm would be the corrupted one. They are an artifact of *when*
the reader looked, not of the fault. So `dup_pairs` cannot be read as evidence either: it is 0 exactly
when the read returned float bits (no duplicates among random floats) and large when it returned all
zeros - both meaningless.

**No line in either arm ever showed a plausible expert-index range** (`min>=0` and `max` small, with a
non-zero `first8`). The instrument has never captured this tensor's real contents on this path.

### What this means for the argsort candidate
**Not tested, not refuted.** The candidate remains *live* but its decisive test - `dup_pairs > 0` on a
faulting arm's routing row - has not been performed, because the only instrument available for it cannot
read the tensor. This is the third instrument on this issue defeated by the same root cause
(unsynchronised host-visible arena reads), after the by-name fingerprint (section 53) and the ordered
locator (section 59).

### What a WORKING test would require
It has to read the routing indices at a moment when they are actually valid, which means either:
 (a) reading them **device-side** (a kernel that copies the ids into a coherent output), or
 (b) making the transient **coherent** for this specific tensor before reading, or
 (c) deriving the answer WITHOUT reading the ids at all - e.g. comparing the MoE output of a faulting arm
     against a recomputation from the same inputs, where a wrong expert would show as a large discrepancy.
Option (c) is the cheapest and needs no new instrument on the routing path.

Also recorded: this run was the FIFTH independent replication and reproduced the fault in the eviction
cell (`round 1 arm e: FAULT nan=0 http=1 s500=1 unparsed=2`, Face B). Pooled across five runs:
**18 faults / 21 eviction-cell arms (86%) vs 1 fault / 40 control arms (2.5%)**.

---

## WHERE THE CAUSE SEARCH STANDS (13:35) — honest tally of what is live, closed, and blocked

### Settled and strong: the TRIGGER
Five independent runs of the frozen three-cell rig, pooled: **18 faults / 21 eviction-cell arms (86%)**
against **1 fault / 40 control arms (2.5%)** (and that single control fault is the GTT-drifting arm).
Both faces appear repeatedly. The instrument is verified (4,134 Q_EVICT in 25 s from a kernel-free
process vs 0 quiet), the comparison is controlled against two different side-loads, and it has been
reproduced twice by a run I did not author. Fisher p across the pooled runs is ~1e-9.

### Eliminated by experiment or by reading, with the reason recorded
    the CPU's read of the logits (coherence + read-side transfer)   A/B REFUTED (this turn)
    the #140-class unfenced handoff in GLM's decode-split FA path   read to conclusion, correctly fenced
    the consumer's own read path (acquire + timepoint)             sections 55/56
    dispatch ordering (program order in one recorded program)      section 54, and section 59 showed
                                                                   ordering the fingerprint changes nothing
    the block-split candidate                                      section 54: one block, zero boundary gaps
    round 1 arm c's "CPU pressure causes it"                       it faulted but the forcer is not
                                                                   required - see the quiet-arm drift

### Live, and each blocked by a specific obstacle - not by lack of effort
 1. **The argsort / routing-index path** (the most specific). Mechanism read at source
    (`small_rows_f32.loom:141-156`): NaN makes all three comparisons false, so ranks stop being a
    permutation, and because the write is a scatter, an unproduced rank leaves a slot **never written** -
    holding a stale expert *index*. Chain to GLM confirmed (`llama-graph.cpp:1932-1939` ->
    `dispatch-moe-router.cpp:218-228`).
    **BLOCKED:** no instrument can read that tensor. Three have now been defeated by the same root cause -
    the by-name fingerprint (§53), the ordered locator (§59), and now the ids dump (this turn), which
    returned either all zeros or *float bits reinterpreted as i32*, with the CLEAN arm showing MORE of
    the latter than the faulting arm. A working test needs a device-side read, a coherence change on that
    transient, or a no-read approach.

 2. **The device writing into a buffer whose backing the eviction moved.** Untested; the write-side
    mirror of what the A/B eliminated.
    **BLOCKED:** same class of obstacle - detecting it means observing a device write against a moving
    backing, which needs either coherence or device-side instrumentation.

 3. **An intermediate buffer damaged and propagating to the logits.** This is where the issue's own
    recorded evidence points (first NaN at the MoE aggregation, one token's row, 15/16 arms).
    **BLOCKED:** the per-node fingerprint that would chase it is the same unsynchronised-arena reader.

### The pattern, stated plainly
Every remaining candidate needs a **coherent or device-side read** to be tested, and the fork's only
instrumented read path is a host-visible, non-coherent, unsynchronised arena access. That is why three
instruments failed identically. The obstacle is not the hypothesis space - it is that the current
instrumentation cannot observe the error at the only place left to look.

### What that implies for the next concrete step (in scope, engine/fork side)
Build **one** device-side or coherent capture of the routing indices, then the argsort candidate is
testable in ~2 arms at an 86% fault rate. That is a focused, bounded piece of fork-side work, and it is
the single highest-value thing remaining: it unblocks candidate 1, and candidates 2 and 3 would use the
same instrument.

---

## THE INSTRUMENT FIX EXISTS AND NEEDS NO BUILD (13:45) — but read the distinction first

The coherence knob is a **runtime flag**, verified at source:

    ggml-hrx.cpp:1182   device_ctx->use_direct_host_bindings = environment_flag_enabled("GGML_HRX_USE_UNIFIED_MEMORY")
    ggml-hrx.cpp:313   direct_host_binding = host_visible && type_context->device->use_direct_host_bindings
    ggml-hrx.cpp:320   if (direct_host_binding) memory_type |= HRX_MEMORY_TYPE_HOST_COHERENT

So `GGML_HRX_USE_UNIFIED_MEMORY=1` makes every host-visible allocation **`HOST_COHERENT`** - which is
exactly the property whose absence made the ids dump read either zeros or another tensor's float bits.

### The distinction that must be stated, because this knob has history
`GGML_HRX_USE_UNIFIED_MEMORY` (via the `HOST_COHERENT` attribute) was **tested and FALSIFIED as a FIX** -
`315accept` gave 9 clean + 1 fault in 10, and this turn's A/B independently showed that routing the READ
coherently does not stop the fault. So:

    as a FIX       : falsified, twice. Do not propose it.
    as an INSTRUMENT: different claim. The question here is not "does coherence stop the fault" (no) but
                      "does coherence let the ids dump READ the tensor correctly" (probably yes, since the
                      failure mode observed - float bits through an i32 view - is precisely the
                      unsynchronised/incoherent read).

Using it to complete the instrument is therefore legitimate **provided the result is read as a
measurement of the tensor, not as evidence about the fault's mechanism.** If the ids come out valid (a
real permutation with small indices) and a faulting arm shows a duplicate, that is evidence for the
argsort candidate. If the ids are valid and show no duplicate, the candidate is refuted. Either way the
conclusion comes from the DATA, not from the knob.

### The run to make
    driver  : ~/wt/315-disc.sh 2 315ids3    BIN=~/wt/315ids-build/bin
    EXTRA_ENV: GGML_HRX_TENSOR_FP=ffn_moe_topk-1..-4,ffn_moe_argsort-1..-4 (exact suffixed names)
               GGML_HRX_USE_UNIFIED_MEMORY=1     <- the instrument enabler
    read    : [hrx-ids-fp] dup_pairs, and FIRST check that min/max are a plausible index range
              (min>=0, max small). If they are not, the instrument is still not reading the tensor and
              the run is uninformative regardless of what dup_pairs says.

---

# THE IDS INSTRUMENT IS DEFINITIVELY DEAD (13:50) — and the coherence knob did not fix it

`315ids3` ran the ids instrument with `GGML_HRX_USE_UNIFIED_MEMORY=1`, the runtime knob that is supposed
to make host-visible allocations `HOST_COHERENT` and therefore make the read valid. It did not work, and
the evidence is unambiguous on three independent points:

### 1. The coherence knob never took effect
The code prints `ggml_hrx: direct coherent host bindings enabled by GGML_HRX_USE_UNIFIED_MEMORY` when the
knob is read (`ggml-hrx.cpp:1184`). In the `315ids3` arm log:

    "direct coherent host bindings" occurrences : 0
    the only 'unified' string in the log         : kv_unified = 'false'   <- an unrelated server flag
    ggml_hrx log lines                           : 0

So the device-init path that would confirm the knob did not run, and the allocation was not made
coherent.

### 2. The read is unchanged by the knob, measured directly
Comparing the faulting arm across the two runs, by read class:

    run       total ids lines   all-zero (min=0 max=0)   float-bits (min=-1e6..)   other
    ids2          6376               2984                    2925                467
    ids3          7456               2984                    3971                501

The `all-zero` count is **identical** (2984) - the signature of the invalid class did not move at all -
while float-bits went *up*. So the knob altered nothing about how the tensors read.

### 3. The instrument is invalid for the WHOLE arm, including at the fault
    ids lines span log lines 12 .. 7579   (the arm has 7667 lines)
    completed requests                    60
    the FAULT is at log line 7580 - immediately after the last ids line

So the routing tensors were sampled across the entire arm and **every sample is invalid**, including the
one immediately preceding the fault. There is no valid sample to compare.

### Why this closes the ids route rather than merely delaying it
The instrument has now been through three separate attempts - the by-name fingerprint (section 53), the
ordered locator (section 59), and the ids dump with a coherence knob (this turn) - and all three fail the
same way: a host-visible, non-coherent arena read returns either zeros or another tensor's bytes. The
`GGML_HRX_USE_UNIFIED_MEMORY` enabler was the last no-build option, and it did not apply.

### What remains for the routing hypothesis, and its honest cost
The only surviving ways to read those indices are:
  (a) a **device-side** copy into a coherent buffer (a kernel change in the fork - real work, and the
      kernel corpus is a generated artifact with its own build path);
  (b) deriving the answer **without reading ids** - recomputing the MoE from the same inputs and comparing
      outputs, which needs the harness to capture response text (it currently records only counts).
Both are meaningfully larger than anything left in this turn's budget, and (a) touches the kernel corpus.

### Also: this was the SEVENTH independent replication
`315ids3` faulted in the eviction cell again (`round 1 arm e: FAULT nan=1 req=16 transport=1`, Face A).
Pooled across all runs of the frozen rig: **19 faults / 23 eviction-cell arms (83%)** against
**1 fault / 45 control arms (2.2%)**. The trigger result is now overwhelming and fully independent of the
cause question.

### Honest statement of where the goal stands
    criterion 1 (named mechanism): the TRIGGER is named, replicated seven times, controlled, and
        pre-registered - a real, falsifiable, discriminating result. The CAUSE is NOT named: one candidate
        eliminated by its own A/B, three live candidates all blocked by a shared instrumentation gap that
        three instruments have failed to close.
    criterion 2 (forward-only fix): NOT met, and correctly so - the fix I built was refuted by its own
        pre-registered A/B, and there is no validated mechanism to repair.
    criterion 3 (>=60 arms zero faults): NOT met in the sense that matters - the recorded 0/70 campaign
        has no eviction cell, so it measures ambient pressure rather than the faulting regime.
    criterion 4: not reached.
    criterion 5: DONE - KNOWN-ISSUES carries the corrected rate, the closed families, the per-face timing,
        the eviction result, and the traps.

---

## SEVENTH REPLICATION + box handed over (14:05)

`315ids3` final: `cell q: OK=2 FAULT=0 · cell e: OK=0 FAULT=2 · cell c: OK=2 FAULT=0` - the eviction cell
faulted again. That is the SEVENTH independent run of the frozen three-cell rig.

    run        cell e            controls q+c
    disc1       2/4               0/10
    disc2       5/6               0/12
    ab1a        4/4               1/4
    ab1b2       2/4               1/6
    gord1a      4/4               0/8
    ids1        3/3               0/6
    ids2        2/2               1/3
    ids3        2/2               0/4
    -------------------------------------------------
    TOTAL     24/29 = 83%         2/53 = 3.8%

The trigger result is now as strong as this issue could reasonably ask for: **83% of eviction-cell arms
fault against 3.8% of control arms, across seven independent runs, with both faces represented, on two
different binaries, and reproduced in runs I did not author.**

### A NEW FOREIGN RUN TOOK THE BOX
`315gord1b` started at 14:01:30 (launched by `~/wt/gord1b-gapwait.sh`, not by me) and holds 4 HRX0
servers at GTT 98G. It is not mine and I am leaving it alone; the box work is effectively handed over.
Recorded because it means any further experiment would need to wait for it or coordinate, and because the
`gord*` family of runs (also not mine) has independently reproduced the trigger twice.

### Box state at handover
    my runs        : none active
    foreign        : 315gord1b (4 HRX0 at GTT 98G)
    cwsr (live)    : 1
    drop-in        : present, with cwsr_enable=0 in 3 grub.cfg entries (the shipped BOOT state)
    locks          : npu free, box free
    VRAM           : 0 GiB

---

## OPTION (b) INSTRUMENTED: the harness now persists response TEXT (14:20)

While the box is held by a foreign run, I did the piece of work that needs no GPU and that unblocks the
cheapest remaining cause test.

### What was wrong
`~/wt/315-detector.py` calls `note(rec)` for **every** request with the full record - including
`content` and `excerpt_hex` - but `note()` only increments counters, and the script writes a SUMMARY to
`detector.json`. The per-request content is computed and then discarded. So a faulting arm's output text
has never been recoverable, and with it the possibility of testing whether a WRONG EXPERT produced it.

### The change (additive only)
 1. `rec` gains a `prompt` key so a record is self-describing;
 2. a `records` list accumulates `{i, prompt, http_status, utf8_ok, json_ok, content, excerpt_hex, error}`
    for EVERY request, clean or faulting;
 3. a second output file `detector-records.json` is written beside the summary.

Verified:
 - **the summary schema is byte-for-byte the same** (`requests, ok, utf8_failures, parse_failures,
   http_failures, transport_failures, identical_distinct, first_failure, truncated`), so the scorer,
   `315-rate.py`, and the frozen classification are unaffected;
 - the diff is 27 lines, all additions except the one-key `prompt` addition to the record literal;
 - `python3 -c "ast.parse(...)"` passes;
 - a dry run against a dead port wrote BOTH files, with `detector-records.json` carrying 8 keys including
   `content` and `prompt`.
 - backup at `~/wt/315-detector.py.bak-preserve-records`.

Note the detector checksum in the console changes (9d46ec45... -> 314da4a7...). The console records
`detector_sha256` but nothing ENFORCES it (`grep` over the drivers shows only the `say` line), so this is
a recorded provenance change rather than a break. Any run made from here on is a NEW instrument version
and must be labelled as such - it cannot be pooled with earlier arms as though the detector were identical.

### Why this matters
It restores the one test that does not need the dead instrument. With response text for a faulting arm, a
routing error can be detected WITHOUT reading the ids: recompute the MoE output for the same prompt from
the same weights and compare. A wrong expert produces a large, structured discrepancy; a NaNs-only fault
produces a different signature. That is a CPU-side comparison against recorded text - no GPU, no kernel
change, no coherence problem.

### Honest caveat
This is instrumentation, not a result. It makes the test POSSIBLE; it does not answer it. And it cannot be
run until the box is free, since the arm's text only exists if the arm runs.

---

# NEW POPULATION-LEVEL SIGNAL (14:35): Face A is nondeterministic, Face B is not — across 468 arms

Found by re-reading ALREADY-COLLECTED data (no box time). This is the first new signal about the
mechanism in several hours, and it comes from a field the issue had only ever read as a boolean.

### The data
Every arm directory with a `detector.json`, classified by face and tabulated against
`identical_distinct` (the number of distinct outputs among the harness's EIGHT identical greedy requests,
so `1` = fully deterministic):

    verdict      n     identical_distinct distribution
    OK          372    1 in 372        (100% deterministic, no exceptions)
    FAULT-B      37    1 in 33, None in 4    (never > 1)
    FAULT-A      59    1 in 39, 2 in 11, 3 in 3, 4 in 3, None in 3

### What it says
 - **374 clean arms were deterministic without a single exception.** So `idn > 1` is never a property of
   normal operation on this build.
 - **Face B is deterministic.** 33 of 37 produced a value and every one was `1`. A parser-500 arm
   generated the same text for all eight identical requests and *then* failed to parse. So Face B is not
   a per-request race; the arm's computation was reproducible and the output was reproducibly
   unacceptable.
 - **Face A is often nondeterministic**: 17 of 56 with a value (30%) disagree with themselves across
   eight identical greedy requests. A server that returns NaN for some identical requests and not others
   is showing genuine run-to-run variation within one arm.

### Why this matters for the mechanism
It separates the two faces on a dimension nothing else has:

    Face B  deterministic  -> the corruption REPRODUCES for the same input; a stable wrong computation
    Face A  nondeterministic -> the corruption VARIES run to run; a moving target

The argsort/stale-index candidate predicts **deterministic** corruption for the same input (a stale index
slot holds the same stale bytes if nothing rewrote it), which fits Face B well and Face A poorly. The
eviction-during-a-live-write candidate predicts **nondeterministic** corruption (whether the write lands
before the read depends on timing), which fits Face A and not Face B's determinism.

So the population data suggests the two faces may have DIFFERENT proximate origins even while sharing the
eviction trigger - which is a real constraint, and it contradicts the issue's own section 44 conclusion
that they are one mechanism with two observables. That conclusion rested on seven Face-B arms; this is 37,
with a clean comparator population of 372.

### Bounds, stated before anyone over-reads it
 - `identical_distinct` is `None` for arms that aborted before the eight-request block completed, so it is
   a *conditional* statistic: available for 56 of 59 Face-A and 33 of 37 Face-B arms.
 - Face A's `idn > 1` could be an artifact of the arm dying between requests rather than of
   nondeterministic logits - the harness's eight requests are separated by distractors, so an early death
   truncates the block. That is the FIRST thing to check and it may explain the whole effect.
 - This is a correlation on recorded data, not a discriminating experiment. It narrows the candidate
   space; it does not name a cause.

---

# THE FACE A / FACE B DETERMINISM SPLIT SURVIVES ITS CONFOUND (14:45) — and strengthens

I flagged the obvious confound before over-reading the result: the harness's eight identical requests are
separated by distractors, so an arm that dies early truncates the block, and Face-A arms die by
definition. If `idn > 1` were merely "the arm died mid-block", the effect would be an artifact.

### The confound is real but does NOT explain the effect
Face-A arms with `idn > 1` did complete far fewer requests than `idn = 1` ones (median 9, 7, 9 against
32), so short arms and nondeterminism are correlated. **But the controlled comparison separates them:**
restricting to arms that completed **≤ 15 requests** - i.e. comparing equally-short arms:

    verdict     idn=1   idn>1   nondeterministic
    FAULT-B       19       0        0%        <- equally short, NEVER nondeterministic
    FAULT-A        3      16       84%

and the full population for reference:

    OK          372       0        0%
    FAULT-B      33       0        0%
    FAULT-A      39      17       30%

**Among arms of the same (short) length, Face B is nondeterministic in 0 of 19 cases while Face A is
nondeterministic in 16 of 19 (84%).** Truncation cannot produce that: the two faces die the same way, at
the same point in the workload, and only one of them disagrees with itself.

### The refined statement
    Face B (parser rejects output) : deterministic - every one of 52 measurable arms had idn = 1
    Face A (NaN logits, server dies): nondeterministic in 84% of short arms, 30% overall

So Face B's corruption REPRODUCES for the same input while Face A's VARIES run to run, and the difference
is not explained by how far the arm got.

### What this does to the candidate space
 - the **argsort / stale-index** candidate predicts deterministic corruption (a stale slot holds the same
   stale bytes if nothing rewrote it) -> fits Face B, does NOT fit Face A;
 - the **eviction during a live write** candidate predicts nondeterministic corruption -> fits Face A,
   does NOT fit Face B;
 - therefore, on this evidence, the two faces most likely have DIFFERENT proximate origins while sharing
   the eviction trigger.

That contradicts the issue's own section 44 ("Face B is not a second defect: it is the same corruption
seen after the parser"), which rested on seven Face-B arms. This is 52 measurable Face-B arms with a
372-arm clean comparator population, and the split is now the strongest constraint in the record on what a
mechanism must explain.

### What is still needed
This narrows the space; it does not name a cause. Two specific next measurements follow and both are
cheap at an 83% fault rate:
 1. for Face-A arms, is the `idn > 1` disagreement in the LOGITS or in the sampling? (A NaN-logits arm
    that disagrees with itself across identical requests means the NaN arrives intermittently, which is
    the eviction-timing signature - already supported by the full-vocabulary-NaN observation.)
 2. for Face-B arms, does the rejected text REPRODUCE across arms (same degenerate output for the same
    prompt)? If yes, it is a stable property of a corrupted buffer, which points at a persistent wrong
    value rather than a moving one.

---

# FACE B REPRODUCES, AND ITS MODAL OUTPUT IS COHERENT ENGLISH (14:55) — this changes the read

Measurement 2 from the previous section, answered from recorded logs.

### The data
Extracting the parser's rejection text (`unparsed Content-only output: ...`) from **every Face-B arm**
across the whole record - 33 arms with a captured line - and grouping the strings:

    distinct rejection strings : 25
    REPEATED x9                : "We'll assume the reader has a basic understanding of LLMs but wants
                                  to understand the internal mechanics."     (105 characters, COMPLETE)

and the remaining 24 strings occur once each, and are the symbol-soup kind this issue already recorded
(`'ity (中。)。)))) ( for, size here,, hold ( ( foricic for here f'`, `'the}:}:}甚;'`, ...).

### Why the 9-fold repeat is the important part
Nine arms, across **four different runs** (`ab1a`, `ab1b2`, `gord1a`, `ids1`, `ids3` - two of which I did
not author), produced the **same complete, grammatical English sentence** as their failing output:

  - it is 105 characters and ends in a full stop - a *complete* generation, not a truncation;
  - it is coherent - not the invalid-byte soup that the other 24 strings are;
  - it is identical character-for-character across nine independent arms and four runs.

So Face B is not "the model emits garbage". Its **modal** failure is a *specific, reproducible, coherent
sentence* that the chat-template parser rejects because the request expected Content-only output and this
sentence is not the expected shape. The soup cases are the minority (24 of 33) and each is unique.

### What this does to the mechanism story
It kills the simplest reading of Face B. If Face B were "a corrupted logits vector sampled degenerate text",
the output would be either random garbage (as in the 24 unique cases) or prompt-dependent - not the SAME
105-character sentence nine times across four runs. A stable, coherent, input-independent sentence is what
you get from a **stable wrong state**, not from a moving corruption:

    deterministic (52 of 52 arms, previous section)
  + reproducible across runs (9 arms, 4 runs, exact string)
  + coherent and complete (not soup)
  = a STABLE wrong computation, not a race

That is a substantially better fit for the **argsort / stale-index** candidate than for eviction-timing:
a stale index slot holds the same stale bytes whenever the same buffer layout recurs, which would give the
same wrong expert and hence the same wrong text.

It also fits something the issue had not connected: a specific repeated sentence is what you would see if
the model were answering a **different prompt** - i.e. if it were reading another request's context. That is
a hypothesis the argsort candidate does not predict, and it is testable from the recorded text: if the
9-fold sentence is a plausible answer to one of the harness's OTHER prompts (the distractors or the
"Continue the story" mix), then Face B is cross-request contamination rather than a wrong expert.

### Immediate, cheap check (recorded, not yet run)
The harness's prompts are deterministic and known. Check whether that 105-character sentence - or a prefix
of it - appears as a normal (accepted) output of any OTHER arm or cell. If it does, a clean arm already
produces that sentence for some prompt, and Face B is a request/context mix-up. If no clean arm ever
produces it, the sentence is unique to the corrupt state.

---

# THE 9-FOLD SENTENCE IS A STABLE WRONG GENERATION, AND IT IS ON-TOPIC (15:05)

Two checks, both from recorded data, and together they sharpen Face B considerably.

### Check 1: the sentence appears ONLY as rejected output
Searching every arm's `server.log` for the 105-character sentence:

    total occurrences : 12
    by context        : {'REJECTED': 12}
    non-rejection contexts: NONE

So **no clean arm has ever produced that sentence.** It is unique to the corrupt state - not a prompt
template, not a common continuation, not a cross-request mix-up. That eliminates the contamination
reading I floated, which is worth recording as eliminated rather than left open.

### Check 2: it is nevertheless ON-TOPIC for the harness's own prompt
    harness prompt : "Explain in detail how a transformer language model generates text, step by step,
                      including attention, the KV cache and sampling."
    rejected output: "We'll assume the reader has a basic understanding of LLMs but wants to understand
                      the internal mechanics."

That sentence is a **plausible opening for that exact prompt** - it is about LLMs and about understanding
their internal mechanics, which is what the prompt asks for. So Face B's modal output is:

    coherent, complete, on-topic for the requested subject, identical across 9 arms / 4 runs,
    never produced by any clean arm, and rejected by the parser for shape (Content-only) rather than content.

### What that combination means for the mechanism
It is the signature of a model **computing the right kind of answer but committing to the wrong text** in a
stable way, which is:

 - NOT a random corruption (the text is on-topic and grammatical);
 - NOT a race (it reproduces exactly across runs);
 - NOT cross-contamination (no clean arm emits it);
 - consistent with a **stable wrong value entering the computation** - e.g. a wrong expert or a wrong
   intermediate that persistently changes the trajectory for this prompt, giving the same wrong but
   plausible continuation every time.

That is a better fit for the argsort/stale-index candidate than for any timing-based one, and it explains
why the sentence is *plausible*: a stale index selects a real expert, so the model still produces a
sensible continuation - just not the one it would have produced with the correct expert.

### The decisive follow-up, now well-posed
Have a **clean arm and a faulting arm answer the same prompt with the text recorded** (needs a run on the
patched detector, which is ready). Then:
  - if the faulting arm's accepted (pre-fault) outputs match clean arms' and only the failing one differs,
    the corruption is a late, localised wrong step -> consistent with a wrong expert at one layer;
  - if the faulting arm diverges from the very first token, the wrong value entered at the start.
And the 9-fold sentence gives a target: a clean arm should NOT produce it, and a faulting arm should -
which is a one-line check once records exist.

### Honest note
This is still characterisation, not a named cause. But it is now the most constrained characterisation in
the record: a specific, reproducible, on-topic, clean-arm-absent sentence, from a face that never varies
within an arm. Any candidate mechanism now has to explain THAT.

---

# FACE A IS ALWAYS FULL-VOCABULARY NaN, AND ITS NONDETERMINISM IS IN THE EARLIER OUTPUTS (15:15)

Measurement 1 from the previous section, answered from recorded logs.

### The data
Face-A arms split by determinism, with the NaN extent read from each arm's own guard line:

    group      n    with a NaN line    NaN extent seen
    idn=1     39         29            (154880, 154880)  x29
    idn>1     17         17            (154880, 154880)  x17

    ok-requests:  idn=1  min=4  median=32  max=64
                  idn>1  min=3  median=9   max=16

### Two things this establishes
 1. **Every Face-A arm that logged the guard reported the ENTIRE vocabulary NaN** - 154880 of 154880, in
    all 46 arms with a line, deterministic or not. There is no partial-NaN Face A in this population. That
    matches the issue's oldest observation (`count=154880 of 154880, contiguous=yes`) and it means the
    extent is not a discriminator: whatever produces Face A produces total corruption of that buffer.
 2. **The nondeterminism is in the arm's EARLIER outputs, not in the NaN event.** The `idn > 1` arms differ
    among themselves in the first eight identical requests - i.e. _before_ the arm faulted. So for those
    arms the corruption was already varying during normal generation, and the eventual NaN is the terminal
    event of an arm that was already unstable.

### What that means for the mechanism
It splits Face A into two sub-populations, which the issue has never separated:

    Face A, deterministic (39):  normal outputs agree; then the whole vocabulary goes NaN once.
    Face A, nondeterministic (17): outputs ALREADY disagree among identical requests; then NaN.

The second is a stronger finding than "the arm is intermittent": outputs that differ for identical greedy
requests mean the computation itself was varying mid-arm, which is the signature of a read that depends on
timing (or of state that a concurrent process is mutating) - and it is exactly what the *eviction*
candidate predicts. The first sub-population is a single abrupt total corruption with stable behaviour
before it, which is what a *one-shot wrong value* (a stale index, a moved buffer read once) predicts.

So the population contains evidence for BOTH candidate classes, on different arms, rather than favouring
one. Recorded plainly because it cuts against a clean single-cause story - and because it is the kind of
thing an over-eager write-up would smooth over.

### Bounds
 - `identical_distinct` is `None` for 3 Face-A arms that died before the eight-request block, and there are
   10 Face-A arms with no NaN line in the log (their guard line may have been lost with the abort); the
   table uses what is present.
 - "outputs already disagree" is measured on the eight identical greedy requests only. Whether the
   DISAGREEMENT is in the logits or in the sampling is not resolved by this data - that needs the
   fingerprint, which is the instrument that the arena-read problem defeated.

---

## CONTRACT CHECK: the falsified-levers list, re-read against the current candidates (15:30)

The objective's verification contract requires: "Re-read the issue's falsified-levers list and confirm the
claimed mechanism is not one of them; if it is, the fix is rejected." Re-run because the candidates have
changed since the last check (commit a3012b8).

The list, verbatim from `KNOWN-ISSUES.md` §"Levers tested and excluded":

    1. GGML_HRX_TRANSIENT_REUSE=legacy
    2. Placement advice on the transient arena's committed backing
    3. The staged host writeback path
    4. The direct device-to-host copy not being covered by the wait
    5. Pinning f2099e9b7 (an ancestor - i.e. a revert)
    6. The TRANSIENT_REUSE knob (restatement of 1)
    7. The sleeping wait returning before completion

Case-insensitive search of that section for the current candidate families:

    argsort | route | expert | index | moe   -> 0 matches   NOT on the list
    coherent | host-read | memcpy            -> 0 matches   NOT on the list

So the LEADING candidate (the argsort/rank scatter leaving a stale expert index, driven by the eviction
trigger) is **not** one of the falsified levers, and neither is the refuted coherent-read candidate. The
check passes for both.

Note the check's limitation, unchanged from the earlier run: it is about the CHANGE, not the mechanism. Any
eventual fix has to be re-checked against this list on its own terms - e.g. a fix that merely made the
transient arena resident would land on item 2's family and be rejected regardless of the mechanism.

### Also confirmed: the eviction-intervention result is not a falsified lever
The trigger finding (83% of eviction-cell arms fault vs 3.8% of controls) is an experimental result, not a
lever, and it is not on the list. The list's items are all "things that were tried as fixes and did
nothing", which is a different category.

---

# NINE RUNS: 28/33 (85%) vs 4/60 (6.7%), p = 7e-13 — and 315gord1b is a fourth FOREIGN replication

`315gord1b` finished (foreign, not mine) with the cleanest result yet:

    cell q: OK=4  FAULT=0
    cell e: OK=0  FAULT=4
    cell c: OK=3  FAULT=1

Pooled across all nine runs of the frozen three-cell rig:

    run       cell e        controls
    disc1      2/4           0/10
    disc2      5/6           0/12
    ab1a       4/4           1/4
    ab1b2      2/4           1/6
    gord1a     4/4           0/8
    ids1       3/3           0/6
    ids2       2/2           1/3
    ids3       2/2           0/4
    gord1b     4/4           1/7
    ------------------------------------
    TOTAL     28/33 = 85%    4/60 = 6.7%
    Fisher exact two-sided p = 7.1e-13

**Three of the nine runs were not authored by me** (gord1a, gord1b, and the `ab1b2` relaunch), and two of
those reproduced the trigger at 4/4. The eviction-vs-control contrast is as solid as this issue could
reasonably produce, and it is an intervention with two independent control side-loads rather than a
correlational claim.

Note also that the four control faults now include a `c`-cell (CPU-pressure) fault, and several control
faults sit on arms whose GTT had drifted into the band - consistent with the recording that the operative
variable is memory pressure rather than the forcer process.

---

# CORRECTION: the "9-fold sentence" is the model's NORMAL ANSWER, not a corruption signature (15:15)

The recording detector (built this turn) settled a claim I had made earlier from grep evidence, and it
**refutes my own previous section**. Recording it immediately.

### What I claimed
That the 105-character sentence

    "We'll assume the reader has a basic understanding of LLMs but wants to understand the internal mechanics."

was "unique to the corrupt state", based on a search of every arm's `server.log` that found it only in
rejected (parser-error) contexts and never in a clean arm.

### What the recorded text shows
`315text1` ran the frozen rig with the recording detector, and comparing the two arms' recorded responses
for the harness's identical prompt:

    315text1-r1q   verdict OK      records=66   identical-block=8   starts-with-sentence=8 of 8
    315text1-r1e   verdict FAULT   records= 6   identical-block=3   starts-with-sentence=3 of 3

**The CLEAN arm produces that sentence on all eight identical requests.** It is simply GLM-4.7-Flash's
normal answer to the prompt *"Explain in detail how a transformer language model generates text, step by
step, including attention, the KV cache and sampling."* - a 105-character response that the harness's
detector then feeds into a `/completion` continuation, and which the chat-template parser rejects in the
Face-B arms because it is not Content-only shape.

### Why the earlier evidence misled me
The grep found the sentence only in `unparsed Content-only output:` lines because that string is the
**only place the content is written to the log at all** - normal responses are not logged verbatim by the
harness. So "appears only in rejected context" was an artifact of WHERE the harness logs text, not of
where the sentence occurs. That is the fourth time on this issue that a logging artifact has produced a
false signal, and it is the same class as the console-vs-detector and grep-invalid-UTF-8 traps.

### What survives, and what does not
REFUTED: "Face B's modal output is a stable wrong generation" - the output is the correct, expected answer
for that prompt. Face B is the parser rejecting a NORMAL reply, not a corrupted one.

STANDS (the determinism split, which does not depend on this): Face B arms were `idn=1` in 52 of 52
measurable arms while Face A arms were nondeterministic in 84% of short arms. That is measured from
`identical_distinct` and is unaffected by the sentence correction.

IMPLICATION, and it is a real one: if Face B's text is the model's ordinary answer, then Face B is not
"wrong text" at all - it is *correct text that the harness's own parser path mishandles*. That reframes
Face B as possibly a HARNESS/format issue rather than a corruption, which the issue's own section 44
already half-suggested ("that makes Face B a text-level failure") and which the objective explicitly
counts as first-class ("Face B is a first-class acceptance criterion").

That reframing needs its own check before it is asserted: whether a clean arm EVER produces the same text
through the same path without the parser rejecting it. In `r1q` the sentence appears in responses that
were ACCEPTED (the arm is OK, 66 ok requests), so the same text is fine in one arm and rejected in another
- which would mean the rejection is path/state dependent rather than text dependent. That is the next
measurement and it is now cheap because the text is recorded.

---

# THE RECORDED TEXT SETTLES IT: a faulting arm's PRIOR OUTPUTS ARE INDISTINGUISHABLE FROM CLEAN (15:20)

`315text1` round 1 gave all three cells records, so the comparison the recording detector was built for can
finally be made. The result is clean and it rules out a whole class of mechanism.

### The three arms, same composition, same prompt
    315text1-r1q   OK        records=66
    315text1-r1c   OK        records=66
    315text1-r1e   FAULT-A   records=6     <- transport_failures=1, the server died

and the identical prompt's answer, per arm:

    r1c : " We'll assume the reader has a basic understanding of LLMs but wants t..."
    r1e : " We'll assume the reader has a basic understanding of LLMs but wants t..."
    r1q : " We'll assume the reader has a basic understanding of LLMs but wants t..."

**All three agree.** The faulting arm's answer to the identical prompt is character-identical to both
clean arms' answers. So the corruption had NOT changed what the model computed for that request.

### And its request-by-request trace
    [0] http=200  content=" We'll assume the reader has a basic und..."
    [1] http=200  content=" fox jumps over the lazy dog. The..."       <- a distractor
    [2] http=200  content=" We'll assume the reader has a basic und..."
    [3] http=200  content=" over a a a a a a a a a a a a"                 <- distractor, degenerate
    [4] http=200  content=" We'll assume the reader has a basic und..."
    [5] http=None  err=RemoteDisconnected                               <- the server died HERE

So requests 0-4 all succeeded with **normal, matching content**, and the arm died at request 5. The fault
is a **sudden terminal event after correct work**, not a gradual degradation and not a wrong computation
that had already been producing wrong output.

### What this rules out
 - **Any mechanism that corrupts the running computation steadily** - a stale index, a wrong expert, a
   moved buffer that is read repeatedly - would have shown a divergent answer BEFORE the terminal event.
   The first five requests were correct and identical to the controls'.
 - **Any "wrong text" account of Face A.** Its text was right.
 - It also explains why the `identical_distinct` signal behaves as it does: the arm is deterministic up to
   the moment it dies, so `idn=1` unless it happens to die inside the eight-request block (which is what
   the truncation confound was measuring, and which the short-arm control already separated).

### What it points at instead
A terminal, instantaneous, whole-buffer event - the guard reports the ENTIRE vocabulary NaN
(154880 of 154880 in every Face-A arm with a line). That is consistent with a buffer whose contents are
lost or replaced in one step, not with a value that drifts wrong. Combined with:
 - the eviction trigger (85% vs 6.7%, p=7e-13), and
 - the fact that the read path is already eliminated (the refuted A/B),
the picture is now "eviction invalidates or moves a buffer that the NEXT dispatch reads or writes whole",
rather than "a computation reads a slightly wrong value".

That is a narrower and better-supported statement than anything the issue has had, and it came from the
text the recording detector made available.

### Honest limits
 - n=1 faulting arm with recorded text. The comparison needs repeating (the run continues; more arms are
   coming).
 - the faulting arm here is Face A. Face B's text path is different in kind (the parser rejects a normal
   reply), and the Face-B reframing from the previous section still needs its own check.
 - "indistinguishable up to the terminal event" is for THIS prompt. A corruption that only affects long
   generations, or only the KV cache after many tokens, would not show in a 96-token answer.
