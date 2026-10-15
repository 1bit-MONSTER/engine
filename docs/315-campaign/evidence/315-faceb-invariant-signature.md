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

# Face B: an invariant signature across 15 arms, present in every cell — and what the fields actually count

Extracted from the arm verdict lines in every `315*-console.log`. No GPU time.

## The signature is identical in all 15 arms

    nan=0 req=<5..27> parse=0 utf8=0 http=1 transport=0 s500=1 unparsed=2 hsa=0

Every Face-B fault on the record reads **exactly** `nan=0 http=1 s500=1 unparsed=2` with `parse=0`, `utf8=0`,
`transport=0`, `hsa=0`. There are no near-misses and no partial variants: Face B is one failure mode with an
invariant fingerprint, which is a stronger starting point than Face A had.

## It occurs in every cell, including the control cells

| cell | arms |
|---|---|
| `e` (eviction forcer) | rounds 1–4, `side=1/1` |
| `q` (**quiet**, `side=0/0`) | rounds 2 — three separate runs |
| `c` (CPU load, `side=12/12`) | round 1 |

So Face B — like Face A — is **not eviction-specific**. The quiet cell, with no forcer running at all, produced
the same signature in at least three runs (09:50:50, 18:34:37, and the `OOR` one at 09:34:57), and the CPU cell
produced it too. Whatever the trigger is, **the eviction forcer is not necessary for either face**, which is the
same conclusion the record already reached for Face A from the control-cell faults and which Face B now
independently reproduces.

One Face-B arm is even `OOR` (`gtt_start=71G`, below the 95G gate) yet still carries the full signature — so the
face does not require the declared regime.

## What the fields count — three subtleties that matter before these numbers are reused

Read from `315-disc.sh:139-148`:

- **`unparsed=2` is a line count, not an event count.** `unr=$(grep -cE "unparsed Content-only output|does not
  match the expected" …)` counts *log lines* matching that pattern. A constant 2 across fifteen arms means the
  one event is logged on two lines (detector and server), not that two responses were unparsed. This is the
  record's own lesson — count occurrences, not lines — appearing again.
- **`http=1 s500=1` is the same failure counted twice**: once by the detector's `http_failures` and once by
  grepping `"code":500` in the server log. So the Face-B event is **one** HTTP 500.
- **`FAULT` is also assigned to truncated arms.** Line 148 makes `req < 66` a fault condition on its own. Every
  Face-B arm has `req = 5…27`, so `FAULT` here mixes "the run failed" with "the run stopped early" — which is
  correct for the rig's purposes but means a `FAULT` verdict alone is not evidence of a *content* failure. The
  content evidence is `http=1 s500=1 unparsed=2`.

## The `utf8=0` question, RESOLVED: the two detectors measure the same event differently

The objective describes Face B as *"invalid-UTF-8 output surfacing as HTTP 500 parse failures"*, while every
Face-B arm in the disc rig shows `utf8=0 parse=0`. Read both detectors and the two are consistent — the fields
differ because the **measurement path** differs, not the event.

**`315-detector.py`** (the disc rig) separates the two:

    line 59-60   text = raw.decode("utf-8");  rec["utf8_ok"] = True
    line 95-97   if not rec["utf8_ok"]:  # Face B: body is not valid UTF-8
                     summary["utf8_failures"] += 1
    line 99-100  if not rec["json_ok"]:  # body decoded but is not the API JSON
                     summary["parse_failures"] += 1

so its `utf8_failures` counts bodies that reached the client but failed to decode, and `parse_failures`
counts bodies that decoded but were not the API's JSON. **Both are 0 in every Face-B arm because the response
never arrived as a 200 body at all** — the server answered `HTTP 500`, which the detector records as
`http_failures` instead. `utf8=0` therefore does **not** mean the output was valid UTF-8; it means the invalid
body was never handed to the decode step.

**`rt-det.sh`** (the original rig) has no such split — it does `json.loads(r.read())` directly, so a non-UTF-8
body and a non-JSON body both raise out of the same call and are recorded as one failure. That single failure
is what the objective's phrase describes.

**Reconciled:** the recorded Face-B event is *the server producing Content-only (non-JSON) output and answering
HTTP 500* — which matches the record's own description of the text as *"valid UTF-8 and a mid-essay fragment"*
and matches `unparsed=2` counting the server's own `unparsed Content-only output` log lines. Whether the bytes
are also invalid UTF-8 is **not determined by these arms**, because the status code short-circuits the client's
decode; `rt-det.sh`'s `identical.json` outputs are where that would show up.

This supersedes an earlier draft of this section, which reported the `utf8=0` field as a discrepancy with the
objective. It is not one: it is a difference in where the failure is caught.

## Why this matters for the goal

Face B now has, for the first time, a **uniform, box-free characterization** to test an instrument against: any
run claiming to localise Face B must produce `http=1 s500=1 unparsed=2` in at least one arm, and its
finite-hash divergence must be sought in *that* arm. It also confirms the structural point from
`315-faceb-has-no-capable-instrument.md` from a second direction: because the arms with these markers carry
**zero** fingerprint lines (the NaN gate), Face B's uniformity has never been matched to a tensor-level
measurement.

## Caveats

- These are arm-level fields from the rig's own classifier; none is a tensor measurement.
- The arms span several weeks and two rigs, and the later ones ran on a thermally saturated box.
- No rate is given: the count of Face-B arms on this record is 15, with no denominator extracted here.
