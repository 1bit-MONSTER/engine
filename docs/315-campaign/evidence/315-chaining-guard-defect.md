<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Defect #16: "wait while the predecessor runs" cannot tell *finished* from *never ran*

## What happened

To run three probes back to back (the §40 chain, the layer-0 `ffn_inp` split, the shape measurement) I wrote three
small scripts, each of the form:

    while ps -eo cmd | grep -F "3 315chain" | grep -qv grep; do sleep 30; done   # wait for the predecessor
    ...clean orphans, cool down...
    launch the next probe

Three disc sessions then appeared on the box at once, including two copies of `315lout4`:

    1495773 19:30  bash /home/bcloud/wt/315-disc.sh 3 315lout4
    1583913 04:57  bash /home/bcloud/wt/315-disc.sh 3 315lout4
    1605854 01:16  bash /home/bcloud/wt/315-disc.sh 3 315inp0     <- stage 2, launched while stage 1 never ran

`315chain-run.log` was **empty** — stage 1 never started — and stage 2 launched anyway, because its guard is
**"wait *while* the predecessor is running"**, which is **satisfied immediately when the predecessor does not exist
at all**. The guard answers "is it running now?", not "did it ever run and finish?".

Two sessions then composed the regime simultaneously, so neither arm's cell means what its label says, and every arm
in that window is unusable.

## Why this is the same defect, one layer up

It is defect #14 — *a `NANONLY` sweep cannot distinguish "every layer clean" from "no Face-A arm occurred"* — with
the sign flipped: there, absence of output was read as a clean result; here, absence of a process is read as
completion. Both are a **predicate on a proxy** standing in for the event of interest, and both silently proceed.

The campaign now has four instances of this shape: #13 (an `EXTRA_ENV` naming a feature the build does not
implement — absence of effect read as absence of an effect), #14, #16, and the `pkill`-on-name traps (#11/#12) where
killing the wrapper was read as stopping the run.

## The fix, and what I did instead

The correct guard records **evidence of the predecessor's completion** — a sentinel line the predecessor writes
after its last arm — and additionally requires that it *started*. Chaining is not worth that machinery here.

**I abandoned the chaining and ran the stages directly, one at a time, verifying each launch by its own log header
before starting the next.** That is slower but it cannot silently skip a stage, which is the failure mode that cost
this session two composed regimes and ~20 minutes of box time it is already over budget on.

## Cost

Two simultaneous regime compositions, whose arms are discarded; the §40 chain probe (stage 1) never ran and must be
re-issued; one duplicate `315lout4` session. No conclusions were drawn from any arm in that window.
