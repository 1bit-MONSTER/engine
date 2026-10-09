# engine#315 — pre-registration for the mechanism-and-fix campaign

Status: **pre-registered, not yet run.** This document fixes the criterion *before* any arm of the
verification campaign is executed, so that a later "clean" result cannot be a post-hoc reading.

Depends on: KNOWN-ISSUES.md (`engine#315` entry), the landed rate study (`docs/`, PR #358), and the
issue's instrumentation and falsification record.

## 1. Inputs, pinned

| input | value |
|---|---|
| engine revision | this branch's base, `origin/main` = `f87ef91` (docs: record the completed engine#315 rate study) |
| `third_party/llama.cpp` pin | `e44c9d01a4d5110cecaf45472a717f46cd1abbf9` |
| fork | `1bit-MONSTER/llama.cpp`, `ggml/src/ggml-hrx/` (`runtime/`, `graph/`, `dispatch/`, `hip/`, `loom-jit.*`) |
| model | `GLM-4.7-Flash-Q4_K_M.gguf`, 18 GB, sha256 recorded in the run log |
| device | `HRX0`, `-ngl 99 -fa on -c 8192 -np 1 --no-webui`, power mode `performance` |
| box | strixhalo; box lock `~/.cache/lax-decode/box.lock` (flock) taken for every arm |

The fix cell and the control cell must each name their build explicitly, e.g.
`llama-server version: 10605 (86c33b5d5)`, so no arm is attributed to an unnamed binary.

## 2. Shape set (per arm, 66 requests)

As implemented by the existing inner detector `~/wt/rt-det.sh`:

1. 8 identical greedy requests: `n_predict=96`, `temperature=0`, `cache_prompt=false`, separated by
   8 distractor requests of different lengths.
2. 50 mixed-length requests: distractor 3–900 words, `n_predict` 4–64.

An arm is *complete* only if all 66 requests return.

## 3. Regime gate (pre-registered)

- **In-regime ⟺ `gtt_start >= 95 GiB`**, exactly as fixed in `~/wt/issue-315-analysis.py`
  (`GATE = 95`). Arms below the gate are reported as out-of-regime and **are not counted**.
- Pressure is produced by **3 detached neighbour `llama-server` processes** on the same model and
  device (the mechanism that brought `gtt_start` to ~101.7 GiB = `109196881920` in the recorded arms).
- Arm watchdog: abort if GTT grows ≥24 GiB during the arm or if package temperature reaches 93 °C
  (the existing `rt-det.sh` guard). A watchdog kill is recorded as an invalid arm, not a pass.
- Neighbours are torn down after every session; no neighbour is left resident.

### 3b. Session preconditions and regime composition

Added after the first smoke arm (2026-10-09) failed in a way the earlier text did not cover. It is a
*tightening* of what may be counted, and it does not touch the acceptance bar in §7.

- **No unaccounted resident model server.** The smoke arm reached `gtt_start = 115.8 GiB` instead of
the recorded ~101.7 GiB because an unrelated GLM server (`~/wt/v315-bisect-build`, not started by this
campaign) was already resident, adding ~20 GB. Every session must record the resident server
inventory (pid, binary, port) and the GTT budget **before** the neighbours start, and choose the
neighbour count so the arm lands inside the gate with that inventory taken into account.
  - That foreign server was **not** killed: it is not this campaign's process and may belong to other
    work. It is recorded instead, and the neighbour count is tuned around it.
- **An upper bound, not just the gate.** The gate is `gtt_start >= 95 GiB`, but ~116 GiB is too hot:
  the detector failed while loading. Arms above ~110 GiB are invalid (allocation failure). The target
  band is the recorded one, ~100-105 GiB.
- **The run must actually serve requests.** An arm is classifiable only if the request phase ran
  (`identical:` and `mixed:` lines present, `gtt.log` non-empty). A server that dies during load
  produces no data and can be counted neither clean nor faulted.

## 4. Fault faces and detectors

| face | signature | detector |
|---|---|---|
| A — NaN logits | `HRX returned NaN logits` (guard reports `count=… contiguous=…`) | count > 0 in `server.log` |
| A — memory fault | `HSA_STATUS_ERROR`, `Queue error`, `wait for HRX graph replay commands failed` | case-insensitive count > 0 |
| B — truncation / format-500 | invalid UTF-8 in the completion body ⇒ parse failure | `parse > 0` or verdict `TRUNCATED` or `req < 66` |

**Change from the earlier rule, and why.** `issue-315-analysis.py` currently categorises
`parse > 0 / TRUNCATED / req < 66` as `trunc` and treats it as harness-side. The issue's own record
now characterises that face as the second real defect (raw invalid UTF-8 output, first-class). This
campaign therefore **counts Face B as a fault face**: an arm is clean only with `nan=0`, `hsa=0`,
`parse=0`, `req=66`. "Zero NaN" alone is explicitly *not* an acceptable acceptance result.

## 5. Arm classification (fixed in advance)

```
fault  if nan > 0 or hsa > 0 or parse > 0 or verdict == TRUNCATED or req < 66
clean  otherwise (66/66 served, both faces silent, gtt_start >= 95 GiB)
invalid  harness death: no server.log, no identical.json, watchdog kill, or an arm that never
         reached the regime gate — recorded, excluded from the denominator, never counted clean
         ALSO invalid: failure to allocate at model load / server start, specifically
         `hrx_allocator_allocate_buffer(...) failed` (ggml-hrx.cpp) or any startup OOM, and any arm
         whose request phase did not run (no `identical:`/`mixed:` lines, empty or absent `gtt.log`)
out-of-regime  gtt_start < 95 GiB — reported separately, not counted
```
Fault takes precedence over incomplete: an arm that faulted *and* then died is a fault, because the
fault is the datum (the existing rule already fixes this ordering).

## 6. Campaign size, stopping rule, and power

- Target: **≥60 in-regime clean arms in the fix cell**, with paired control arms running in the same
  session so that regime drift affects both cells.
- Hard cap: **80 scheduled pairs** (160 arms). If the cap is reached without 60 clean fix-cell arms,
  the campaign reports what it has and the goal is *not* satisfied — no extension chosen after seeing
  the data.
- Power statement: with 60 clean arms and 0 faults, the observed rate is 0 and the 95% upper bound is
  below 5% (`0.95^60 = 4.6%`); the bound is reported as a one-sided 95% CI (rule of three) plus Wilson.
- Comparison against the recorded baseline uses the pre-registered arms only
  (baseline: shipped-fix 2/37 = 5.4%, always-stage 1/25 = 4.0%; Fisher exact two-sided and
  `P(rate_fix < rate_baseline)` by Beta Monte-Carlo with seed 20261007, as in the existing script).
- No arm pruning, no criterion changes, no re-running an arm because its result was inconvenient.
  A re-run is allowed only for a recorded *invalid* arm, and the invalidation reason must be logged.

## 7. Acceptance

The campaign passes only if, in the fix cell, **≥60 in-regime arms are clean by §5 — zero Face-A and
zero Face-B faults**. Anything less is reported as not met, with the observed counts and CI.

## 8. Campaign log requirements (reproducibility)

For every session, append to the campaign console: the producer's own path and sha256, the copy of
`rt-det.sh` and `issue-315-analysis.py` used (with sha256), the model sha256, the binary identity of
both cells, `gtt_start`/`gtt_max` per arm, and the raw log paths.

Recorded from the first smoke session (2026-10-09), for reuse:

| item | value |
|---|---|
| model | `GLM-4.7-Flash-Q4_K_M.gguf` sha256 `29837ed2c0fc5f51981adf8ac8083fcf80743c598381f13e9f06cbad0498b174` |
| `~/wt/rt-det.sh` | sha256 `9657695645b697b51014c86ef30db31f9348ac2c394ab5e9ce75fe2be34043f8` |
| `~/wt/issue-315-analysis.py` | sha256 `acfbc4ea980c7c2848cbfd2ff0a837ec871340d6bdddbaafc09ffdfdc96d5309` |
| binary identity | `llama-server version: 10605 (86c33b5d5)` |
| smoke outcome | **invalid**: `gtt_start 124357271552` (~115.8 GiB), detector died at load, `hrx_allocator_allocate_buffer(...) failed` at `ggml-hrx.cpp:326`, request phase never ran |

Teardown note: `systemctl --user stop <unit>` alone did **not** stop the `systemd-run --user --scope`
neighbours; they had to be terminated by matching their own command lines (ports 20201-20203). Session
scripts must verify teardown and report the GTT afterwards — this smoke session left 118 GiB resident
until it was cleaned up by hand.

The producer is **committed**, not anonymous: `docs/315-campaign/315-paired.sh` (the paired driver) and
`docs/315-campaign/315-detector.py` (the request phase). The earlier producer was never located on the
box — only its consumers were on disk — so it was re-implemented here. Both scripts pass a syntax
check, and their emitted lines were tested against the **existing** analyser's regex: all four verdicts
(`OK`, `FAULT`, `INVALID`, `OOR`) match, because the extra `utf8=`/`alloc=` fields are appended after
`gtt_start=…G` and that regex is not end-anchored.

Why the detector is not just `rt-det.sh`: the existing harness decodes responses with
`json.loads(r.read())` inside a bare `except`, so a body that is not valid UTF-8 is indistinguishable
from any other request failure — Face B would be invisible. `315-detector.py` issues the *same* 66
requests (same seed, so arms stay comparable) but inspects the raw bytes, separating transport
failures, non-UTF-8 bodies (Face B), JSON parse failures, and HTTP status. The driver's classification
follows §5 exactly.

## 9. Budget

≤12 hours of strixhalo box time in total for this goal, shared with other work through the box lock.

## 10. Explicitly out of scope

- AMD driver/kernel changes (recorded as a proposal with evidence if the mechanism lands there).
- Any pin that moves backwards: a candidate that is an ancestor of the shipped pin is a revert and is
  not a fix.
- Weakening the guards or the instrumentation in a way that destroys failure evidence.
