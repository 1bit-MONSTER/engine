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

# Both faces, one trigger: 11 eviction-cell faults split 7 Face A / 4 Face B

Criterion 1 asks for a mechanism covering **both faces**. This records the numeric basis for that claim
under the single intervention that moves the rate, re-derived from the consoles rather than from prose.

## The split, per run, in the forced-eviction cell

    run          Face A (nan=1)   Face B (nan=0, content/500)
    315gord1a          2                  2
    315gord1b          3                  1
    315ids1            2                  1
    total              7                  4

So **11 forced-eviction faults split 7 Face A / 4 Face B**, and both faces occur under the *same* cell of the
*same* frozen rig. Neither face requires a different intervention:

- **Face A** is the whole-vocabulary NaN signature
  (`HRX returned NaN logits: count=154880 of 154880 … continuing=no`), which the sampler guard catches.
- **Face B** is the format failure (`common_chat_peg_parse: unparsed Content-only output` → HTTP 500), which
  the guard does not catch because the logits are finite.

## Why this matters for the criterion, and what it does NOT claim

- **It satisfies the "both faces" clause**: one intervention, both observable outcomes, replicated across
  three runs and two builds. That is stronger than the record had before, where the faces were sometimes
  attributed to different conditions.
- **It does not claim the faces share a cause.** What is established is that the trigger produces both; the
  *cause* is unlocated after six closed families. In particular, the peer's n=14 text constraint shows that
  faulting arms still emit the correct answer for the identical prompt, so Face B is a **terminal
  whole-buffer** event rather than a changed computation — which is a constraint on any future cause, not a
  cause itself.
- **The face that appears is a property of the session, not the event.** An earlier session faulted 16 of 17
  arms with `nan=1` each while another faulted half its arms with zero NaN anywhere
  (`dev/315/315-campaign/evidence/315-face-landscape.md`). So the two faces are not two mechanisms so much as
  two observables of one event, selected by whether the corrupted bytes happen to decode as NaN or as
  parseable-but-wrong text.

## The trigger, quantified (for the same claim)

From the quiet cell — the only cell with no instrument confound — across every recorded console:

    cell q (QUIET)        ~5%   (2 faults in 39 arms at 97G; 0% at 44G and 98G)
    cell e (FORCER)       ~95%  (41 faults in 43 arms at 98G)

a **19x contrast on the same workload at the same GTT level**. That is the intervention the both-faces claim
rests on, and it is why the mechanism statement can be made at trigger level with confidence while the cause
stays open.
