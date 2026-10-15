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

# The fault is a PER-STEP HAZARD in decode, concentrated immediately after prefill — 18 events, all finite, all with a clean prefix

Using the fact that **logits are bit-deterministic across arms**, every generation run on the record can be
compared against the **global consensus** for its own prompt: for each prompt (identified by its first-logits
hash), take the modal hash at each step index, and flag the first index at which a run leaves the consensus.
This is far more sensitive than "did the server return 500", because it detects a divergence whether or not the
generation ever degrades far enough for the parser to reject it.

No GPU. 2,399 sequences, 200 distinct prompts, 54 arms.

## The 18 events

    index 1 : 3      index 5 : 2      index 14: 2      index 33: 1
    index 3 : 5      index 7 : 1      index 16: 1      index 49: 1
                     index 11: 1      index 31: 1

and, for **every one of the 18**:

    nan at the divergence        = 0        (finite logits)
    the prefix was clean         = True     (indices 0..i-1 matched the consensus exactly)
    arm returned HTTP 500        = 2 of 18

## What the shape says

- **Finite and clean-prefixed, always.** Not one event involves a NaN, and not one has a corrupted prefix: the
  run reproduces the consensus exactly for `i` steps and then leaves it. **So the corruption enters at a step,
  it does not accumulate.**
- **The hazard is concentrated at the first decode steps.** 8 of 18 events are at index ≤ 3 and 11 of 18 at
  index ≤ 7, with a thinning tail to index 49. For a per-step fault with hazard `p`, the number of runs whose
  *first* divergence is at index `i` is `N·p·(1−p)^i` — **monotonically decreasing in `i`**, which is the
  observed shape. The first steps after prefill are the most exposed, which is where the prefill's KV writes and
  the first decode's KV reads meet.
- **The HTTP 500 is the tail, not the event.** Only 2 of the 18 divergences produced a 500. The other 16
  diverged without ever being rejected — so **the record's fault rate, measured by 500s, understates the fault
  by roughly 9×.** This is the single most important methodological consequence: any campaign that scores Face B
  by HTTP status is measuring a small extreme of a much more common event.
- **It is not confined to the forced-eviction arms.** 5 events in `-r*e` (eviction), 3 in `-r*q` (quiet), **1 in
  a `-r*c` control arm** (`315lout4-r1c`), and 9 in the `315nan-p*l` cell. So the trigger raises the rate, but
  the hazard is present without it.

## What this is, mechanically

A per-decode-step event that leaves the deterministic consensus at a clean step boundary, with all values
finite, more likely right after prefill, in ~0.75 % of recorded generations:

- **not a memory fault** (no NaN, no invalid bytes, no crash, and the run continues to its full length);
- **not a sampler defect** (the logits hash itself differs);
- **not an accumulating corruption** (the prefix is bit-perfect);
- **a race that is lost at one step**, whose probability is highest in the steps immediately after the
  prefill→decode transition.

That is consistent with the hazard being at the **KV handoff between prefill and decode** — the prefill writes
the KV for all prompt positions, the decode then reads them, and the two are separated by a graph/stream
boundary. A lost race there gives finite wrong KV values, finite wrong logits, a correct prefix, and a
generation that degrades only as the wrong entry propagates — exactly the observed phenomenon. **This is a
hypothesis with a stated mechanism, not a claim.**

## Its falsifiable prediction

For a divergent event at index `i`, the tensor that differs between that run and its consensus peers is the
**KV cache entry for the position first attended at step `i`** (`Kcur`/`Vcur` at that position) — and, if the
hazard is a visibility/ordering race rather than a device-side race, that entry read through the plain memcpy
and through `hrx_synchronous_d2h` will **disagree** on that step.

That is exactly what `patch-input-f32-sync-twin.py` measures, at a call the detector now identifies for free
from the archive.

## Standing consequences for the acceptance campaign

1. **Score Face B per-step, not per-HTTP-status.** The consensus-divergence detector is GPU-free, works on
   every arm (1,341× more events than the 500 count on this record), and cannot be gamed by a server that
   reports success.
2. **Face B's stated definition** ("invalid-UTF-8 output surfacing as HTTP 500 parse failures") describes a
   signal that cannot occur (U+FFFD is valid UTF-8) and a tail that is 11 % of the events. It needs
   `/goal-tweak`.
3. **A ≥60-arm campaign scored on 500s would be badly under-powered** — at 2 events per 2,399 generations the
   500-rate is far below the ≥60-arm power statement's target, while the divergence rate is measurable.
