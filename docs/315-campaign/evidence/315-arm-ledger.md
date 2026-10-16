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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 evidence: the complete arm ledger on the box, the population statement the objective asks for
     ("arm-by-arm counts"), with the two attribution traps that came out of building it. Generated from every
     arm directory under ~/wt/rt-det; the CSV beside this file is the raw table. -->

# The arm ledger: 178 arms, 131 in-regime, 34 faulting in-regime

```sh
ssh strixhalo 'cd ~/wt/rt-det; for d in 315*-GLM-4.7-Flash-Q4_K_M/; do t=$(basename "$d" | sed "s/-GLM.*//"); \
  g=$(grep -m1 -oE "gtt_start [0-9]+" "$d/run.log" 2>/dev/null | awk "{print \$2}"); \
  n=$(grep -c "HRX returned NaN logits" "$d/server.log" 2>/dev/null); \
  r=$(python3 -c "import json;j=json.load(open(\"$d/detector.json\"));print(j[\"requests\"],j.get(\"http_failures\",0),j.get(\"identical_distinct\",1) or 1)" 2>/dev/null | tr " " ","); \
  echo "$t,${g:-0},${n:-0},${r:-,,-}"; done'
```

Columns, in order: `tag,gtt_start,nan_guard_lines,requests,http_fail,idn`. (Five values, not seven - the tag already
carries the arm suffix, e.g. `315row-p1c`. A six-name header shifts every column and turns the whole table into
noise; that mistake was made and corrected once while writing this.)

## The population

| bucket | arms |
|---|---|
| all arm directories under `~/wt/rt-det` | 178 |
| in-regime (`gtt_start` 95-110 GiB) | **131** |
| faulting in-regime | **34** |

Fault classes, from the ledger (an arm counts as faulting if any of `nan_guard_lines > 0`, `http_fail > 0`, or
`idn > 1`):

| tag | arms | in-regime | faults |
|---|---|---|---|
| `315nan` | 21 | 19 | 18 |
| `315fa` | 18 | 15 | 5 |
| `315hunt2` | 16 | 16 | 1 |
| `315accept` | 14 | 14 | 2 |
| `315chain` | 7 | 7 | 3 |
| `315sync2` | 7 | 7 | 1 |
| `315gsync` | 6 | 1 | 1 |
| `315row` | 5 | 5 | 2 |
| `315det` | 4 | 4 | 0 |
| `315probe2` | 13 | 13 | 0 |
| `315probe3` | 16 | 1 | 0 |

Both faces appear in the in-regime population and never in the same arm: the `nan=1` arms (Face A, the sampler
guard) and the `http=1` arms (Face B, the server's own 500). Twelve of the 34 in-regime faults are Face B.

## Two traps this table exposed - both worth knowing before it is used as evidence

1. **A tag can be reused across days, so a tag is not a session.** `~/wt/rt-det/315fp2-*` held five arms scored
   `req=66, nan=0, gtt=98G` whose mtimes were `2026-10-09 20:00-20:19` - a different session's run under the same
   tag - while the current run's first arm sat beside them with no `detector.json` yet. Read the mtime, or the
   run's own console, before attributing an arm to the session you are running.
2. **"No fault in the ledger" is not "clean".** An arm still in flight has no `detector.json`, and an arm the
   harness scored `INVALID` (server never served, or the detector could not allocate) has no fault signal of its
   own; both look like `nan=0, http=0` in this table. The verdict is in each run's console
   (`pair <n> arm <c>: OK|FAULT|INVALID|OOR`), and the ledger is a population statement, not a verdict list.

With those two caveats applied, the table is the direct answer to "how often does this reproduce": **34 faulting
arms in regime, across a dozen independent sessions, at `gtt_start` 98-99 GiB**, split between the two faces.
