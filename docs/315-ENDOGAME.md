# engine#315 — endgame runbook

Written 2026-10-12 ~18:40 (box time) while the campaign of record was running, because the session
that started it will not outlast it. **Read this before doing anything else on #315.** Exact
commands, in order, with what to check and what each outcome means.

Everything below is committed to `1bit/315-preregistration`. Nothing is only in a session.

---

## 0. Where things stand

| fact | value |
|---|---|
| box | `strixhalo` (`ssh strixhalo`); engine work on `ryzen` in `~/wt315-prereg` |
| branch | `1bit/315-preregistration` of `1bit-MONSTER/engine` |
| campaign of record | `315cwsr70` — 70 pairs → ~70 fix-cell arms, `amdgpu.cwsr_enable=0` |
| campaign started | 2026-10-12 ~16:54 box time; box booted 15:15:40 |
| expected duration | ~8 h at ~4.7–5.3 min/arm (measure, don't assume — see §1) |
| H1 | **FALSIFIED** — run 7 ran on a fresh boot too (two independent clocks; `docs/315-H1-FALSIFIED.md`) |
| H2 | **the leading and only surviving candidate** — mid-wave preemption under memory pressure |
| H3/H4 | dead (the GPU-MV sizing defect is proven fixed on this box; `docs/315-cwsr-size-excluded-PROOF.md`) |

Criterion status at the time of writing: **1 open** (needs the control), **2 open** (no repair landed
yet), **3 in progress** (campaign running), **4 not started**, **5 not started**.

---

## 1. When the campaign lands

```bash
ssh strixhalo
P=$(pgrep -f "315-evict-trace.sh 70" | head -1)
ps -o etime= -p "$P"                       # if empty, it finished
tail -30 ~/wt/315cwsr70-console.log        # the last arms
```

**Check specifically whether faults appear LATE.** The recorded prediction: if faults begin in the
second half rather than the first, that supports H2 (pressure accumulates) and weakens any
"the setting fixed it" reading. Read the console as a sequence, not as a total — a late fault is a
different finding from an early one and must not be averaged away.

Then score it:

```bash
cd ~/wt
python3 315-rate.py 315cwsr70-console.log --det ~/wt/rt-det --tag 315cwsr70
```

**Read the tool's output, not the console, for the criterion.** It applies §5/§6's counting rules and
reconciles against every arm's `detector.json`. Three things must be true:

1. `FAULT  0` — and **no `MISSED BY CONSOLE` line.** That line would mean an arm faulted via HTTP 500
   where the console could not see it; if it appears, the campaign has failed and the arm is a fault.
2. **`clean` ≥ 60 counting only fix-cell (`l`) arms.** This is the trap: §6/§7 count **fix-cell arms**,
   not total arms. A 70-pair run yields ~70 `l` arms plus ~23 `c` (control) arms; the `c` arms are
   controls and **do not count toward the 60**. Verify the fix-cell count directly rather than
   reading the total:
   ```bash
   grep -oE "pair [0-9]+ arm l: OK" ~/wt/315cwsr70-console.log | wc -l
   ```
   Compare that to the tool's `clean` figure. If `clean` counts both cells, the fix-cell number is the
   one that must reach 60.
3. `UNVERIFIED  0` — any unverified arm is clean from the console alone and is **not certified**.

Expected shape of a good result: `clean ≥60`, `FAULT 0`, CI upper bound at or under ~5%.

If it lands **short of 60 fix-cell arms** (e.g. some went OOR), do **not** prune arms and do **not**
change the criteria — §8 forbids both. Re-run only for a **recorded invalid** arm, or run more pairs.

## 2. Then the control (this is criterion 1's decisive test, and H2's only faulting arm)

`docs/315-campaign/315-cwsr-control.sh` removes the `cwsr_enable=0` drop-in, reboots, and prints the
command to run the identical campaign at `cwsr_enable=1`. It is now sized at **20 pairs** — at the
26% baseline that gives ~5.2 expected faults and `P(0) = 0.002`, so it is decisive for H1, and it
supplies the few faulting arms H2 needs.

It installs the PID-capturing driver itself, which is **mandatory** — see §3. Run it only when the
campaign of record is finished, because the reboot ends it:

```bash
# on ryzen, from ~/wt315-prereg
scp docs/315-campaign/315-cwsr-control.sh strixhalo:~/wt/
ssh strixhalo 'cd ~/wt && TAG=315ctl1 PAIRS=20 bash 315-cwsr-control.sh'
```

**Reading key — this is the whole point of the control:**

| control result | meaning | consequence |
|---|---|---|
| rate ≈ 26%, faults present | **H1 confirmed** — the `cwsr=0` signal was a fresh-boot artefact | criterion 1 collapses; re-scope honestly |
| rate ≈ 0%, no faults | **H1 dead, the `cwsr=0` signal is real** | proceed to H2 with the control's faulting arms |

The control is the *only* run that can produce a faulting arm with a captured PID (see §3), so it is
not optional for H2 either.

## 3. H2: the eviction comparison

```bash
cd ~/wt
python3 315-evict-per-arm.py ~/wt/rt-det ~/wt/<tag>-console.log \
        --smi ~/wt/<tag>-evict/kfd_smi.tsv --tag <tag>
```

It bounds each arm by its captured PID's `PROC_START` → `PROC_END` (else the next arm's start) and
counts only that PID's `Q_EVICT` rows inside the window, then compares faulting vs clean means.

**Why the PID capture is mandatory:** a finished arm's PID cannot be recovered afterwards, and its
post-arm lifetime contaminates the count — proved on run 7, where a fault arm's total was inflated
to 1,461 evictions because the server outlived its arm by 40 minutes. Run 7's data is therefore
unusable for H2 and only the control (and the campaign of record) can supply it.

The tool refuses to invent a number when the PID is absent. Do not "fix" that by estimating.

**Interpretation:** read the **`pre-fault` column**, not the window total. The tool counts evictions
up to the instant the fault was logged, because a fault cannot cause evictions that precede it —
whereas the window total is confounded (a faulting arm crashes, may be retried, and outlives its arm,
so its window can simply be longer).

The fault time comes from the arm's own `server.log`, in `minutes.seconds.milliseconds.microseconds`:
`0.07.279.709 E common_sampler_sample: HRX returned NaN logits`. Both faces are matched (Face B via
`unparsed Content-only`), or a Face-B arm would look like it had no recoverable fault time. Validated
against run 7: all 3 faulting arms yield a time (81.181 s, 57.838 s, 7.280 s) and **0 of the 8 clean
arms do** — a parser that found faults in clean arms would fabricate the H2 effect.

Compare the faulting arms' **pre-fault** mean against the **clean arms' full windows** — a clean arm
has no fault to truncate at, so that comparison is conservative and works against finding an effect.

If faulting arms show more evictions pre-fault, that is consistent with H2 and with the residency
argument in §4. If they do not, H2's stated mechanism is wrong as written and must be revised rather
than defended. Note that run 7's faults landed **very early** — 7.3 s and 57.8 s into their arms — so
a pre-fault window can be short; check the parsed time rather than assuming a fault lands late.

A supporting (not decisive) check already done: eviction **rates** are comparable across
`cwsr=0` and `cwsr=1` (3.80/s vs 4.19/s), so `cwsr=0` does not reduce pressure — the variable is
isolated to wave-preemption capability, not to memory pressure. Recorded in commit `6db2fd0`.

## 4. Criterion 2: the forward-only change

H2's mechanism points at an engine-side **residency** fix: place weights/KV/activations in
driver-owned/VRAM memory rather than evictable GTT/userptr/SVM, so a mid-wave preemption cannot
resume onto relocated memory. This is the same conclusion Engine V2's D5 reached independently.

Rules that bind:

* **forward-only**: the pinned revision it moves forward from must be an **ancestor** of the new one.
  No reverts. Check with `git merge-base --is-ancestor <old> <new>`.
* **Do not land the coherence patch.** It is already falsified by §13; `tools/315-lever-check.py`
  rejects a diff that reintroduces it. Run that tool on the diff before proposing anything.
* An **engine startup guard** (refuse to run above some GTT watermark) is a **workaround, not a
  repair** — it hides the fault instead of removing it. If that is the only option, say so plainly
  and re-scope rather than presenting it as a fix.

The candidate patch branch `1bit/315-coherence-fwd` was **deleted** as falsified. There is currently
**no landed repair**, and criterion 2 is open.

## 5. Criterion 4: no regression

```bash
# both baselines are committed; compare against them, do not re-derive
docs/315-campaign/evidence/315-engine-ci-baseline.md
docs/315-campaign/evidence/315-hrx-tests-baseline.md
```

Two gates: (a) `1bit-MONSTER/engine` CI green; (b) the **dense-model control** and the
**`MUL_MAT_ID` op-test** unchanged vs baseline — byte-identical output where the baseline records it.

Builds on the box: `~/wt/315instr3-build/bin` (control, pin `e44c9d01a`, `if (direct_host_binding)`
intact) and `~/wt/315fix-build/bin`. Both report `version: 10613 (e44c9d01a)`. The faulting baseline
pin is `third_party/llama.cpp = e44c9d01a`; current engine pin `0e255d774`.

## 6. Criterion 5: KNOWN-ISSUES.md

An entry gets a limitation note **only after a mechanism is named** — that is the issue's own rule,
so this is last. If the defect is genuinely gone, **retire** the entry instead of annotating it.
Include the reproduction and the arm-campaign evidence paths.

---

## Standing constraints (do not violate)

* RE material is **source material only** — never copy code, constants, kernels or layouts. Facts go
  in docs; own implementations are written fresh.
* NPU runtime, NPU kernels, HIP kernels, the tuning DB and control code stay **private**.
* Every dossier ends with `Dossier: public | private | split`.
* Every behavioural claim cites an instruction/symbol address; dossiers ship a machine-checked bundle.
* Never prove absence with a truncated `strings | sort | head` — enumerate.
* Record unresolved questions with `record_unknown`. An honest unknown is a deliverable; a guess is a bug.
* **Never weaken a guard or the instrumentation to get a result.** If a guard blocks the conclusion,
  the conclusion is not yet available. (This is how the detector/console gap in §1 item 1 was found —
  the fix was to make the tool *stricter*, not to loosen the criterion.)
* Box-time budget: **≤12 h** of strixhalo GPU time. ~2 h was used before the campaign of record and
  the campaign costs ~8 h, so a 20-pair control is the last thing that fits comfortably. Check the
  budget before adding anything.
* The box is shared and thermally limited: NPU under `flock ~/npu.lock` (**the real box mutex** —
  `box.lock` excludes nobody), builds under `~/lb/thermal-run.sh`, never kill a peer, report XRT
  version / load / Tctl.

## Traps already paid for — do not re-step

1. **The console can't see an HTTP-500 fault.** Reconcile against `detector.json` (§1). A
   console-only count can manufacture a passing campaign.
2. **§6 counts fix-cell arms, not total arms.** 70 pairs ≠ 70 counting arms for the criterion.
3. **A finished arm's PID is unrecoverable.** Install the PID capture before the run, not after.
4. **`~/npu.lock` is the real mutex.** `box.lock` excludes nobody. A dead holder wedges the box;
   `NPU-LOCK-WEDGED-2026-10-12.md`.
5. **`kfd_smi_watch` needs root.** Without it, it writes a header-only file and **exits 0** —
   silently. A header-only eviction stream looks like a valid empty result.
6. **Do not edit a frozen driver mid-campaign.** bash reads scripts incrementally; create a new file
   instead (`315-paired-pidcap.sh` exists for exactly this reason).
7. **H1's falsification came from `boot_ns`, not a purpose-built control.** The control in §2 makes
   it direct. Until then, treat H1 as strongly weakened but formally open.

---

## Appendix — measured pace and the budget verdict (2026-10-12 17:22 box time)

Measured from the arm `server.pid` mtimes, i.e. from when each arm's server actually started:

```
16:55:18  p1c      16:59:45  p1l      17:04:11  p2l
17:08:37  p3l      17:13:02  p4c      17:17:27  p4l
```

Interval between consecutive arm starts: **4:25, 4:26, 4:26, 4:27, 4:25** — a spread of **2 seconds**
across five intervals. So the pace is not an estimate; it is **4.43 min/arm** with almost no variance,
which also means the arm time is set by the fixed request sequence and not by contention.

| | |
|---|---|
| arms started | 6 |
| measured pace | **4.43 min/arm** |
| projected total (93 arms) | **6.87 h** |
| elapsed so far | 0.42 h |
| projected completion | ~**23:45 box time** |

**Budget verdict — both stages fit.** ~2 h used before the campaign + 6.87 h campaign + ~2.2 h for a
20-pair control (30 arms × 4.43) ≈ **11.1 h of the 12 h ceiling.** So the decisive control in §2 does
fit after the campaign, which was the open question in Update 4. Margin is ~0.9 h: enough, but not
enough to also add a stage without re-checking.

Note the `c` arms appear only at pairs 1 and 4 so far, so the pairing is sparse in controls as
designed — 70 pairs should yield ~70 `l` and ~20 `c`, comfortably clearing the 60 **fix-cell** bar
with ~10 arms of slack for any that fall out of regime.
