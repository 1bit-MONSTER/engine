<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

engine#315 pre-registration. Written BEFORE the run so the reading cannot be chosen post hoc.
-->

# Pre-registration: the FA alignment question on the FROZEN rig (`315alignfrozen`)

The FA family is the one still-open lead, and it is open **because the attempt to test it did not reproduce
the trigger**: the peer's `galign`/`galign_repl` used a bespoke composition that produced mostly *clean*
eviction arms, so it could not compare alignments (correctly reported as UNINFORMATIVE — see
`315-galign-CORRECTION-my-parse-was-wrong.md`, and my own parse error that briefly misread those arms as all
faulting).

The fix is to use the **frozen driver**, which reproduces the trigger at ~95% in cell `e`, and simply vary the
alignment through `EXTRA_ENV`. That is what this pre-registers.

## Hypothesis

**H-align-frozen.** If the #140 layout/dispatch class carries the fault, then forcing
`GGML_HRX_FA_PARTIAL_ALIGN=256` (the layout the #140 oracle used to *expose* the divergence) should fault
**more** than the default `4096` on the frozen rig's cell `e`.

## Design

    driver : ~/wt/315-disc.sh <rounds> <tag>, unchanged - the SAME frozen three-cell rig
    BIN    : ~/wt/315coherent-build/bin
    cells  : e (evictonly forcer) is the test cell; q and c remain controls
    env    : EXTRA_ENV carries the standard detector env PLUS GGML_HRX_FA_PARTIAL_ALIGN
    arms   :  A = align 4096 (default)   B = align 256
    rounds : 3 per arm, interleaved by the driver's own rotation

Note the `EXTRA_ENV` exported to the arm is also inherited by the neighbours, so both are consistent within
an arm; the comparison is between arms.

## Reading, fixed BEFORE the run

    B (align 256) faults materially more than A (4096) in cell e
        -> the #140 layout/dispatch class is implicated; follow the FA substitution as the carrier.
    A ~= B (both at the rig's ~95%, overlapping)
        -> the alignment does not carry the fault, and the FA family closes.
    B clean while A faults
        -> contradicted in direction; report as such.
    both clean in cell e
        -> UNINFORMATIVE (the rig failed to induce); do not read as a null.

Because the expected cell-`e` rate is ~95%, **both arms faulting is the uninformative-looking but expected
outcome** and does not by itself distinguish the hypotheses — that is the power point recorded earlier. The
discriminating signal would be a **rate difference**, which needs the arm counts below.

## Arms, and the power problem stated up front

At ~95% per arm, distinguishing 95% from, say, 60% needs roughly **20 arms per alignment** at 80% power. With
3 rounds per arm the driver gives **3 cell-`e` arms per alignment** — enough to catch a *large* difference
(e.g. 256 clean where 4096 faults) but **not** enough to bound a moderate one.

So this is explicitly a **coarse** test: it can falsify "the alignment is required" (if 256 faults freely) but
cannot establish "the alignment makes no difference". If 256 faults at the rig's normal rate, the honest
conclusion is *not sufficient*, the same strength as the coherence and GRAPH_SYNC rows — **not** a measured
null.

## Stopping rule

3 rounds per arm (the driver's default for this form), or earlier only if one alignment is unanimous in cell
`e` while the other has also produced at least 2 scored cell-`e` arms. No arm pruning; INVALID/OOR arms are
recorded as invalid.

## What this does and does not settle

- **Would settle (one-sided)**: whether the FA alignment is *required* for the fault.
- **Would not settle**: whether it makes a *rate* difference, at this arm count.
- **Does not touch criteria 2–4.**
