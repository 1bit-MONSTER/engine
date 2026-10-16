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
<!-- engine#315: the raw console of the instrumented nan-dump arm hunt (session 315nan, 2026-10-10
     06:08:04 to 09:37:17), kept verbatim so the pre-registered analyser can be re-run on the primary
     record rather than on a summary of it. The lines below are unindented on purpose: they are exactly
     the grammar 315-paired.sh emitted, so `315-analyse.py <this file>` parses this file as a console.
     Read it with the caveats in ../evidence/315-instrumented-arm-hunt.md: the fix cell carries the
     nan-dump instruments (cleared of fabricating NaN in section 41, but still a timing perturbation),
     the control arm is a single draw, and pair 19's fix arm was destroyed by my own SIGKILL. -->

# The instrumented arm hunt's console, verbatim

## What the pre-registered analyser makes of it

```
arms parsed: 21 from 1 console(s)

cell control (c): scheduled=1 in-regime=1 clean=0 fault=1 invalid=0 out-of-regime=0
    fault rate 1/1 = 100.0%  95% CI 20.7-100.0%
    under the recorded study's rule (Face B = harness-side): clean=0 fault=1 trunc=0
cell fix (l): scheduled=20 in-regime=17 clean=1 fault=16 invalid=1 out-of-regime=2
    fault rate 16/17 = 94.1%  95% CI 73.0-99.0%
    if clean count stands: 95% upper bound on the true rate = 100.0% (rule of three over 1 clean arms)
    under the recorded study's rule (Face B = harness-side): clean=1 fault=17 trunc=2

control vs fix: Fisher exact two-sided p = 1.000; P(rate_control < rate_fix) = 0.807
recorded baseline for reference: A 2/37 (5.4%), B 1/25 (4.0%)

acceptance (fix cell): >=60 in-regime clean arms AND zero faults on both faces
  clean=1 (need 60)   faults=16 (need 0)   -> NOT MET
```

Reading: the fix cell ran 20 arms, of which 17 were in-regime (2 out-of-regime, 1 invalid): 16 faulted and 1
was clean, for a rate of **16/17 = 94.1% (95% CI 73.0-99.0%)** against the recorded 2/37 = 5.4%. The single
control arm (the stock build, which implements none of the instruments) faulted as well. Acceptance is
**NOT MET** and could not be met by this run: it is a diagnostic run (`NOSTOP=1`), its faults are expected, and
`MINCLEAN` was never reached.

The `identical_distinct` freeze note of 2026-10-10 does not change this tally: every divergent arm in this
console was already a fault, which is the population claim the freeze note rests on. This file also serves as
the backward-compatibility check for that note - it has no `idn=` field, and the analyser still classifies it
exactly as it did before the note.

## The console, raw

```
=== session 2026-10-10T06:08:04-03:00 tag=315nan pairs=20 neighbours=2 ctx=2048 gtt_before=26G ===
model_sha256 29837ed2c0fc5f51981adf8ac8083fcf80743c598381f13e9f06cbad0498b174
detector_sha256 9d46ec453fc9640e33e6d6eed857c3e263bfa3df9c1c151a352e1f6f62a66a48
control_bin /home/bcloud/wt/315instr3-build/bin -> version: 10613 (e44c9d01a)
fix_bin /home/bcloud/wt/315fp-build/bin -> version: 10613 (e44c9d01a)
resident_inventory (foreign servers, not ours):
  pid=2705531 /home/bcloud/.local/share/1bit-canonical/vulkan/llama/bin/llama-server -m /home/bcloud/models/Qwen3-
  pid=2705616 /home/bcloud/.local/share/1bit-pr-agent/vulkan/llama/bin/llama-server -m /home/bcloud/.local/share/1
box lock acquired 2026-10-10T06:08:40-03:00
neighbours up gtt=79G
  pair 1 arm c: FAULT nan=1 req=34 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
  pair 1 arm l: FAULT nan=1 req=10 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 1: fix-cell clean=0 fault-or-invalid=1 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=1 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 2 arm l: FAULT nan=1 req=33 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 2: fix-cell clean=0 fault-or-invalid=2 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=2 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 3 arm l: FAULT nan=1 req=33 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 3: fix-cell clean=0 fault-or-invalid=3 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=3 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 4 arm l: FAULT nan=1 req=33 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 4: fix-cell clean=0 fault-or-invalid=4 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=4 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 5 arm l: FAULT nan=1 req=8 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 5: fix-cell clean=0 fault-or-invalid=5 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=5 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 6 arm l: FAULT nan=1 req=10 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 6: fix-cell clean=0 fault-or-invalid=6 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=6 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 7 arm l: FAULT nan=1 req=33 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 7: fix-cell clean=0 fault-or-invalid=7 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=7 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 8 arm l: FAULT nan=1 req=33 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 8: fix-cell clean=0 fault-or-invalid=8 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=8 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 9 arm l: FAULT nan=1 req=10 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 9: fix-cell clean=0 fault-or-invalid=9 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=9 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 10 arm l: FAULT nan=1 req=10 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 10: fix-cell clean=0 fault-or-invalid=10 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=10 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 11 arm l: FAULT nan=1 req=46 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 11: fix-cell clean=0 fault-or-invalid=11 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=11 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 12 arm l: FAULT nan=1 req=35 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 12: fix-cell clean=0 fault-or-invalid=12 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=12 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 13 arm l: FAULT nan=1 req=41 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 13: fix-cell clean=0 fault-or-invalid=13 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=13 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 14 arm l: FAULT nan=1 req=33 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 14: fix-cell clean=0 fault-or-invalid=14 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=14 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 15 arm l: FAULT nan=1 req=6 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 15: fix-cell clean=0 fault-or-invalid=15 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=15 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 16 arm l: OK nan=0 req=66 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=98G)
after pair 16: fix-cell clean=1 fault-or-invalid=15 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=15 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 17 arm l: FAULT nan=1 req=54 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=80G)
after pair 17: fix-cell clean=1 fault-or-invalid=16 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=16 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 18 arm l: FAULT nan=0 req=0 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=26G)
after pair 18: fix-cell clean=1 fault-or-invalid=17 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=17 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 19 arm l: OOR nan=0 req=5 ret3=0 parse=0 gtt_start=44G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=26G)
after pair 19: fix-cell clean=1 fault-or-invalid=17 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=17 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
  pair 20 arm l: OOR nan=1 req=22 ret3=0 parse=0 gtt_start=44G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=26G)
after pair 20: fix-cell clean=1 fault-or-invalid=17 (target >=60 clean, cap 20 pairs)
NOTE: fix-cell fault(s)=17 recorded and the run continues because NOSTOP=1 — this is a DIAGNOSTIC run, not an acceptance run, and its counts must not be read as one
1 clean fix-cell arms, 17 faulted-or-invalid in the fix cell, out of 21 arms run
```
