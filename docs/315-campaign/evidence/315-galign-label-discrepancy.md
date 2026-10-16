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

# Discrepancy: the peer's replication misstates its own first result

Recorded because the running replication (`/tmp/galign_repl.sh`, tag `315galign2`) is premised on a baseline
their own data does not show, and a replication of the wrong thing would be wasted.

## What the replication says it is testing

`/tmp/galign_repl.sh`'s header:

> "REPLICATION of the alignment result. **n=1 showed align4096-e FAULT-B while align256-e was CLEAN.** At an
> ~85% base rate one clean arm is ~15% likely by chance, so this run repeats BOTH in the same composition
> to test whether the asymmetry holds."

## What their first run actually recorded

`~/wt/315galign/console.log`, verbatim:

    arm ga4096-e align=4096 cell=e FAULT-B req=66 nan=0 http=4
    arm ga256-e  align=256  cell=e FAULT-A req=66 nan=0 http=0      <- NOT clean
    arm ga4096-q align=4096 cell=q FAULT-A req=66 nan=0 http=0
    === 315galign done 2026-10-13T16:00:04-03:00 ===

and re-derived from the raw per-response files, independently of the console:

    ga256-e   66 distinct responses of 66      (all differ -> faulted)
    ga4096-e  63 distinct responses of 66      (3 differ -> Face B)

So **`align256-e` faulted**, by their own classifier and their own raw files. The results are the opposite of
an asymmetry:

    align=4096, cell e   FAULT (Face B)
    align=256,  cell e   FAULT (identity check)
    -> BOTH alignments fault -> the pre-registered "no difference" branch, i.e. the FA class is not
       implicated

## Why this matters

The replication is testing "does the 4096-faults / 256-clean asymmetry hold?" — an asymmetry that does not
exist in the run it cites. At the pooled 95% cell-`e` rate, two faults at both alignments is the expected
outcome, and the correct replication is of **"both fault"**, not of a contrast.

Two readings are possible and the evidence here cannot choose between them:

1. the replication header is simply wrong about its own baseline (a bookkeeping slip), and the run will still
   produce useful data — if both arms fault again, that *confirms* the no-difference reading;
2. the author read a different run, or a different field, as "clean".

Either way the **substance is unaffected**: `315-align-run2-result.md` and `315-galign-peer-result.md` both
record that both alignments faulted, and the FA path is retired on that basis. What is at risk is only the
replication's framing, and it is worth stating so that a "both fault" outcome is not then read as a
surprise.

## Recommendation for whoever reads the replication

Score it against the **pre-registered branches**, not against the header's premise:

    both alignments fault  -> FA class not implicated  (consistent with the first run)
    only one faults        -> would be a new asymmetry worth its own test
    both clean             -> uninformative, since n=1 showed faults at both

---

# Unblocking the replication: galign1's orphaned neighbours were holding the box

The replication sat with an empty console for ~4 minutes because **galign1 left three `llama-server`
processes behind**, all reparented to init:

    pid 3931486  ppid=1  21:18 elapsed
    pid 3931487  ppid=1  21:18 elapsed
    pid 3931488  ppid=1  21:18 elapsed

Their parent run had **finished** (`315galign done` was present in its console), so these were genuine
leftovers, and between them they held both `~/.cache/lax-decode/box.lock` and ~80 GiB of GTT — which is why
the replication's `flock -w 1800` was waiting rather than running.

Cleared, after checking they were orphaned and their run finished. The replication acquired the lock
immediately (its own four processes), and started at 16:04:53 with GTT climbing to 98 GiB.

Two coordination notes:

- **I also cleared my own dead `align3` waiter** from the lock queue (pid 3963038), so it was not squatting
  ahead of the peer's live run.
- **The teardown gap is systemic, not specific**: both my `315align2` and the peer's `315galign` left
  neighbours behind when the driving shell exited, because `setsid` + a killed parent skips the `trap ...
  EXIT` handler. Any future run on this box should either avoid `setsid` for the neighbours or check for
  ppid-1 `llama-server` processes before waiting on the lock — otherwise a finished run's orphans will block
  the next one for its full `flock` timeout.

---

# SUPERSEDED (2026-10-13): the counts used here were a parse error, and the peer was closer to right

This file argued that the replication misstated its own first result, on the basis of byte-level
`md5sum` counts showing `ga256-e` at "66 distinct of 66". **Those counts were wrong** — scoring streaming
responses by hash treats identical text with different chunk framing as distinct (see
`315-galign-CORRECTION-my-parse-was-wrong.md`).

So:

- the file's factual claim — that `align256-e` "faulted" and was therefore not clean — is **not supported by
  the evidence it cites**;
- the peer's replication premise is **not** shown to be a misstatement by anything here;
- the substantive outcome is the peer's own: their run was **UNINFORMATIVE** because their composition did not
  reproduce the frozen driver's trigger rate, which is a real finding about the design rather than about
  alignment.

The one part of this file that stands is its closing recommendation — score against the pre-registered
branches rather than any header's premise — and that recommendation is exactly what led to the correct
disposition.
