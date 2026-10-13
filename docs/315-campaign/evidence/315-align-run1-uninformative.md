<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# `315align` run 1 — executed, but UNINFORMATIVE. Do not read it as a null.

The pre-registered FA alignment/dispatch discriminator ran at 15:16:23 and completed in **38 seconds**.
Recording it as uninformative rather than as a null, because the condition it needs was never established.

## What it produced

    === 315align start 2026-10-13T15:16:23-03:00 gtt=47GiB ===
      arm align4096-e align=4096 cell=e responses=120 distinct=1 nan=0
      arm align256-e  align=256  cell=e responses=120 distinct=1 nan=0
      arm align4096-q align=4096 cell=q responses=120 distinct=1 nan=0
      arm align256-q  align=256  cell=q responses=120 distinct=1 nan=0
    === 315align done 2026-10-13T15:17:01-03:00 ===

All four arms: every one of 120 responses identical (`distinct=1`), no NaN. The `align=256` + eviction cell —
the cell the #140 oracle used — was clean.

## Why this does NOT retire the FA path

The pre-registered branch for "align=256 ~= align=4096, both clean" is "the #140 class is NOT #315's
mechanism; leave the FA path". **That branch does not apply, because the arms could not have faulted.**

1. **No regime, no neighbours, no compaction.** GTT was **47 GiB of 120 GiB** — low pressure by
   construction. The prior #140 recipe (`docs/315-mechanism.md`) required `align=256` **plus** compaction
   (`tee /proc/sys/vm/compact_memory`) and still fired only once in five runs, and the record notes that
   the one faulty run coincided with foreign load 8-12 while the four clean repeats ran at 4-7.
2. **38 seconds is too short.** Four arms in 38 s gives ~10 s per arm. The eviction window this class needs
   opens on queue preemption under sustained pressure; a 10-second arm does not open it.
3. **The forcer alone was never shown sufficient for this small-model path.** `evictonly` drove the GLM rig
   to 85-96% fault rates, but that was under the full regime (3 neighbours, GTT 97-100 GiB). Whether it
   opens the window at 47 GiB with a 378 MB model is exactly what was *not* tested.

So the run is **uninformative by the pre-registration's own branch 3** ("both clean in cell e →
UNINFORMATIVE … most likely the rig failed to induce"), applied to the FA probe: a clean result under
conditions that could not produce the event says nothing about the event.

## What the run DID establish

The probe harness works mechanically: it starts a 378 MB model on HRX0, drives 120 identical greedy
requests per arm, counts distinct responses and NaN lines, and tears down cleanly — in ~10 s per arm. That
makes it a usable instrument once the conditions are right.

## What a decisive version needs

Match the prior recipe's conditions rather than a simplified version of them:

    align=256      vs   align=4096
    cell e         = evictonly 500  +  compact_memory looping  +  the GLM neighbours for GTT
    requests       >= 180 per arm (the prior recipe's N), not 120 short ones
    arms           several per cell, since the prior rate was 1 in 5 under ideal conditions

That is a materially bigger run — the prior work spent an evening on it and got one event — and it should
be attempted only with the box-time budget to spend, not as a 38-second probe.

## Bookkeeping

The harness has been improved in place (`~/wt/315align-run.sh` should gain the compaction loop and the
neighbour composition before any re-run). This file exists so the 38-second run is not later cited as
evidence that the FA layout class is innocent.
