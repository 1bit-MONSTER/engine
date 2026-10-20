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
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Defect #15: orphaned `evictonly` forcers accumulate, and the "quiet" cell is not quiet

## What was observed

While the `315lout2` session was still running, its side-processes were counted on the box:

    $ pgrep -fc evictonly        ->  4      (it had been 2 twenty minutes earlier)
    $ pgrep -fc "busy=70"        ->  2      (the rig's CPU-pressure workers)
    $ awk '{print $1}' /proc/loadavg; sensors | awk '/Tctl/'   ->  5.04 , 75.9 C

and the oldest forcer's elapsed time (28 min at that point) places its start **before `315lout2` began** — i.e. it
was left over from the previous session (`315lout`).

So during `315lout2`'s arms there was at least one foreign `evictonly` applying **forced KFD eviction to every
cell**. The count then grew, because each arm spawns a forcer for its cell and the teardown does not always
reap it.

## Why this matters: it changes what a cell *means*

The campaign's cells are defined by their pressure — `q` quiet, `e` eviction, `c` CPU — and the arm line's cell
letter is used as the label for which condition the arm ran under. **With an orphaned forcer alive, an arm
labelled `q` is not a quiet arm**: it ran under eviction pressure that the label does not record. The rig's own
gating does not catch this, because it measures thermal and load state, not *what the side-processes are doing*.

This is the second instance of one family, and it is worth naming it as such:

- **defect #10** — an orphaned neighbour (parent PID 1) that the composition step **found already running and
  counted toward the target GTT**, so the composition baseline was not "box at rest";
- **defect #15** — orphaned forcers that the arms' cell semantics do not record, so a "quiet" arm is not quiet.

Both are the same failure: **a leftover from a previous run changes the meaning of a later run's label, silently.**

## What it does to the Face-B amplifier claim

`315-faceb-amplifier.md` reports **6 of 6 arms as Face B "in all three cells"**, and offers that as a
near-deterministic reproducer. **That claim is now qualified**: the `q` and `c` arms were not cell-isolated, so
"Face B appears in the quiet cell too" is **not** established — it may be eviction pressure leaking into a cell
labelled quiet, which would be a much less interesting statement (and one already consistent with the campaign's
recorded eviction trigger).

The part that survives: **the configuration faults as Face B at `req=1…11`, repeatedly, in regime.** The part that
does not: **that it does so without eviction.**

Corrected in place in that file's own terms — the amplifier's value as a tool is unchanged (a fault on the first
request is still cheap to test against), but the claim about which cells it spans is withdrawn until a session is
run whose forcer count is verified zero at every arm boundary.

## The operational fix

Kill by **process family, in a script file, before and after every session**, and assert the count is zero before
the composition is trusted:

    for P in $(pgrep -f evictonly);    do kill -9 $P; done
    for P in $(pgrep -f "busy=70/1000.0"); do kill -9 $P; done
    # then assert:  [ "$(pgrep -fc evictonly)" = 0 ] || abort

Patterns must live in a **script file** rather than a command line, because a command line carrying them is
matched by its own `pgrep`/`pkill` — a trap that cost this session three separate incidents, two of them killing
my own SSH session mid-cleanup.

---

# Addendum: the rig's gates assume a dedicated box, and the box is now shared

Chasing defect #15 turned up a second operability problem, distinct from the leftovers.

The frozen rig gates every arm on, among other things, **`load1 < 4`** (`315-disc.sh:91`). Held next to the
current tenancy of this box that threshold is **unreachable**:

    $ cat /proc/loadavg        ->  4.72  ...      (tctl 44.5 C, our own forcers and workers at zero)
    $ ps -eo pcpu,cmd --sort=-pcpu | head
      clickhouse-server, redis, dockerd, node ...   <- other agents' workloads, not ours

with our own side-processes verified at **zero** (`evictonly=0`, CPU workers `=0`) and **GTT at 26 GiB** — i.e. the
box is as quiet as *we* can make it, and the gate still refuses.

Two consequences worth recording:

1. **A run can stall indefinitely without failing.** A swap script of mine sat in exactly this loop for minutes
   with every one of *our* preconditions satisfied, and only a self-imposed iteration cap let it proceed. The rig's
   own gate has no such cap. An arm that never starts is indistinguishable, from the outside, from an arm that
   started and produced nothing — the same ambiguity this campaign has hit twice before (defect #13's inert
   `EXTRA_ENV`, and #14's `NANONLY` sweep that could not distinguish "clean" from "not run").
2. **The gate is a property of the original tenancy.** `load1 < 4` and `tctl < 80` were calibrated for a box this
   campaign had to itself. With other agents running servers and CPU work, the gate measures *their* load, so it
   silently changes what "in-regime" means for our arms — not the GTT regime itself (arms carry their own
   `gtt_start`), but the *ambient* condition the campaign has been treating as constant, which is the baseline
   against which §39's inducing effect and the whole ambient-vs-forced comparison are drawn.

**Not resolved here.** The honest statement is that this campaign's ambient baseline and its arm-starts are now
conditional on a box it no longer exclusively owns, and no arm in this record was tagged with the foreign load
that was present when it ran. The rig's newer `resident_inventory (foreign, not ours)` line is a step toward
recording that, and it should be read as required context for any arm from this session onward.
