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
<!--
SUPERSEDED IN PART (2026-10-13): the "both alignments faulted / every arm faulted" sections below are
WRONG. They were produced by scoring arms with md5sum over streaming response files, so framing differences
read as content divergence. The true reading is distinct_content=1 for the arms checked, i.e. CLEAN. See
315-galign-CORRECTION-my-parse-was-wrong.md. The only claim below that survives is the one-sided statement
about sufficiency, and even that must be re-derived from a correct parse before use.
-->

# The peer's GLM alignment run: every arm faulted at BOTH alignments — the FA class is retired

This is the other agent's run (`/tmp/galign.sh`, tag `315galign`, 15:43:08–16:00:04), recorded here because
it answers the §57 discriminator on **GLM itself** and therefore supersedes my Qwen-based attempt. The
per-arm evidence is theirs; the parse below is re-derived from their raw files rather than taken from their
console.

## The result

    === 315galign start 2026-10-13T15:43:08-03:00 gtt=80G neighbours=3 ===
      arm ga4096-e align=4096 cell=e FAULT-B req=66 nan=0 http=4   gtt_start=98G
      arm ga256-e  align=256  cell=e FAULT-A req=66 nan=0 http=0   gtt_start=98G
      arm ga4096-q align=4096 cell=q FAULT-A req=66 nan=0 http=0   gtt_start=98G
    === 315galign done 2026-10-13T16:00:04-03:00 ===

## What the labels actually mean — re-derived from the resp.* files

The console's `FAULT-A`/`FAULT-B` labels are not server-error labels, so they were re-derived from the
66 per-response files each arm wrote:

    arm        distinct responses (of 66)   NaN logits   Content-only hits
    ga4096-e            63                       0              8      <- Face B (3 responses diverged)
    ga256-e             66                       0              0      <- identity check failed entirely
    ga4096-q            66                       0              0      <- same

and every arm has **66 non-empty responses and zero server errors**. So:

- `ga4096-e` is a genuine **Face B** fault: 3 of 66 responses diverged, with the Content-only parse
  failures in the server log.
- `ga256-e` and `ga4096-q` are flagged `FAULT-A` on the **response-identity check** — all 66 responses
  differ, which the rig's classifier scores as a fault. No NaN and no parse failure.

**Every arm faulted.** Both alignments, and even the quiet cell.

## The pre-registered reading

The pre-registration fixed, before any run:

> `align=256 ~= align=4096` (both fault, or both clean, in cell `e`) → **the #140 class is NOT #315's
> mechanism.** Per §57's own instruction, **leave the FA path entirely** and do not spend further budget
> there.

Both alignments faulted in cell `e` (and `q` faulted too). So the branch is met: **the FA layout/dispatch
class is not this issue's mechanism, and the FA path should be left.**

## Why this run counts where mine did not

| | my Qwen take 2 | the peer's GLM run |
|---|---|---|
| model | Qwen3-0.6B — does not fault in this regime | **GLM-4.7-Flash — faults** |
| alignment | all clean, uninformative | **both alignments faulted** |
| classifier | my own parser (broken; reported 0 everywhere) | the rig's identity check |
| compaction | silently absent (my `sudo -n tee` bug) | neighbours at GTT 80→98G, in-regime |

So the discriminator is answered, and it is answered **against** the #140 hypothesis — on the right model,
in-regime, with a working classifier.

## Caveat, stated rather than glossed

The `q`-cell fault is unexpected (the quiet cell should be clean) and is **not** explained by this file. It
may be the same drift-into-band effect the earlier cell-`c` fault showed, or a different harness condition.
It does not change the cell-`e` comparison (both alignments faulted there), but it does mean this run's
quiet cell is not a clean control and the result should be cited for the **alignment contrast**, not as a
three-cell rig result.

## What this closes

The sixth family. With it, the closed set is:

    logits read path (hrx_synchronous_d2h)     REFUTED   (its own pre-registered A/B)
    host side (all four 0x52 accesses)         CLOSED    (none on the decode path)
    runtime AQL submission policy              CLOSED    (runtime does not infer hazards)
    graph reorder (GGML_HRX_GRAPH_ORDER)       REFUTED   (cell e 4/4 both arms)
    deferred host writeback                    CLOSED    (binding host_data == nullptr)
    FA alignment / dispatch (#140 class)       RETIRED   (both alignments fault, on GLM)

---

# Correction: n=1 per alignment cannot TEST the contrast — it can only fail to contradict it

Re-reading the FA retirement's basis before relying on it further, and it is weaker than the section above
implies. This is recorded as a correction, not a caveat.

## The power problem

At the rate the forced-eviction cell actually runs (~95%: 41 faults in 43 arms), a **single** arm per
alignment has almost no chance of *not* faulting:

    P(fault | align=4096) = 0.95
    P(fault | align=256)  = 0.95
    -> observing 1 fault at each alignment is the EXPECTED outcome under the hypothesis that the
       alignment makes no difference

So the run's result is consistent with "no difference" — but it is *equally* consistent with "a real
difference smaller than the noise floor of a single arm". To distinguish "256 faults 95% and 4096 faults 95%"
from "256 faults 95% and 4096 faults 60%", one arm each is nowhere near enough: the second case also
frequently produces one fault per arm.

Concretely, with n=1 per cell, the experiment has essentially **no power to detect an alignment effect**
unless the effect were large enough to make one of the arms *clean* — and at a 95% base rate even a clean arm
is only ~5% likely, which is exactly the fluke the peer's replication header mistook for a result
(`315-galign-label-discrepancy.md`).

## What the retirement can and cannot rest on

- **Can rest on**: the FA family is not *sufficient* to explain the fault, because in the same session at
  `align=256` — the layout the #140 oracle used to *expose* the divergence — cell `e` still faulted. That is
  a one-sided statement and it is what the data supports.
- **Cannot rest on**: "the two alignments have the same rate." That requires several arms per alignment, and
  the pooled 95% base rate makes the required n larger than it first appears: distinguishing 95% from 60% at
  80% power needs roughly 20 arms per arm-of-the-experiment, not 1.

## What the running replication would establish

The peer's `315galign2` repeats both alignments in one composition, which is the right structure. But if it
again yields one arm each, it inherits the same power problem and its result should be cited as "again
consistent with no difference", not as a measured null. A decisive version needs the arm count the power
calculation above implies.

## Effect on the six-family table

The FA row is downgraded from **RETIRED** to **"not sufficient"** — the same strength as the earlier
coherence and GRAPH_SYNC rows, which were also one-sided nulls. It is not re-opened as a *candidate* (nothing
positive supports it), but it is no longer presented as retired on a measured null, because no measured null
exists.

---

# Replication result: both alignments faulted AGAIN — n=2 per alignment, no difference

`315galign2` completed at 16:19. Raw evidence (distinct-response counts, re-derived from the per-response
files, not the console labels):

    run         align=4096                     align=256
    galign1     63 distinct of 66  (FAULT)     66 distinct of 66  (FAULT)
    galign2     66 distinct of 66  (FAULT)     66 distinct of 66  (FAULT)

with both of galign2's arms at `gtt_start=98G` and `nan_guard=0`.

## What this changes

**n is now 2 per alignment, and no arm was clean.** That matters because the replication was framed as
testing a 4096-faults / 256-clean asymmetry (`315-galign-label-discrepancy.md` showed that premise was
wrong about galign1). The useful outcome is the opposite of what it was looking for: it **strengthens** the
one-sided conclusion.

    align=4096, cell e:  2 faults in 2 arms
    align=256,  cell e:  2 faults in 2 arms

## What it still does not do

The power correction stands (`315-galign-peer-result.md`, earlier section): at the ~95% base rate, n=2 per
alignment still cannot distinguish "both 95%" from "95% vs 60%" at reasonable power — the required n for that
contrast is roughly 20 per condition. So the honest statement remains **one-sided**, and is now simply
better supported by more arms:

> the FA alignment/dispatch class is **not sufficient** to explain the fault: at `align=256` — the layout the
> #140 oracle used to *expose* the divergence — the forced-eviction cell still faulted, in 2 of 2 arms
> across two independent runs.

That is the same strength as the coherence and `GRAPH_SYNC` rows, and it is what the entry now claims. It is
**not** a measured null.

## Note on the replication's framing

Its header said it was testing whether the "align4096-e FAULT-B / align256-e CLEAN" asymmetry held. Neither
run shows that asymmetry: galign1 had `align256-e` faulting on the identity check, and galign2 has both arms
faulting. So the replication answered a question it did not intend to ask, and the answer is the pre-registered
**"no difference"** branch — which §57 pre-committed to abandoning the FA path on.

---

# Update: the replication extended to 4 arms — all four faulted, both alignments 2/2

The peer's run continued past its first two arms. Full tally from its console:

    gb4096-e   align=4096  cell=e  FAULT-A  resp=66  http500=0  content_only=0  nan_guard=0
    gb256-e    align=256   cell=e  FAULT-A  resp=66  http500=0  content_only=0  nan_guard=0
    gb4096-e2  align=4096  cell=e  FAULT-A  resp=66  http500=0  content_only=0  nan_guard=0
    gb256-e2   align=256   cell=e  FAULT-A  resp=66  http500=1  content_only=2  nan_guard=0

so, pooled with the first run:

    align=4096, cell e:   2 of 2 arms faulted   (galign1 63/66 distinct; galign2 66/66 ×2)
    align=256,  cell e:   2 of 2 arms faulted   (galign1 66/66; galign2 66/66, one with Face-B content)

**Every arm faulted. No arm was clean.** Notably `gb256-e2` carries a genuine **Face B** signature
(`http500=1 content_only=2`) — so the forced-eviction cell produced Face B at `align=256` too, i.e. the face
that appears does not track the alignment either.

## What this settles, and at what strength

- **Settled (one-sided)**: the FA alignment/dispatch class is **not sufficient**. At `align=256` — the layout
  the #140 oracle used to *expose* the divergence — the forced-eviction cell faulted in **2 of 2 arms**, with
  a Face-B signature in one of them. Replicated across two independent runs.
- **Still not settled**: whether the two alignments fault at the *same rate*. n=2 per alignment cannot
  distinguish "both ~95%" from "95% vs 60%"; that needs roughly 20 per condition. So this remains a
  one-sided exclusion of sufficiency, not a measured null — the same strength as the coherence and
  `GRAPH_SYNC` rows.

## Why the extension was the right move

The replication was extended from two arms to four precisely because a single arm per alignment has no power
at a ~95% base rate (the power point raised in the correction above). Four arms with zero clean results does
not by itself reach the n a two-sided contrast needs, but it does make "the FA class is not sufficient"
robust: four independent opportunities for `align=256` to come back clean, and none did.

---

# Final tally (5 arms) — and the quiet cell faulted at 98G, twice now

The peer's run completed with five arms:

    gb4096-e   align=4096  cell=e  FAULT-A  resp=66  http500=0  content_only=0  nan_guard=0
    gb256-e    align=256   cell=e  FAULT-A  resp=66  http500=0  content_only=0  nan_guard=0
    gb4096-e2  align=4096  cell=e  FAULT-A  resp=66  http500=0  content_only=0  nan_guard=0
    gb256-e2   align=256   cell=e  FAULT-A  resp=66  http500=1  content_only=2  nan_guard=0
    gb4096-q   align=4096  cell=q  FAULT-A  resp=66  http500=0  content_only=0  nan_guard=0

Pooled with the first run, cell `e`:

    align=4096:  2 of 2 arms faulted
    align=256:   2 of 2 arms faulted
    -> every eviction-cell arm faulted, both alignments, across two runs

## The quiet-cell fault, and why it now needs a second look

`gb4096-q` faulted **in the quiet cell at `gtt_start=98G`**. My pooled tally
(`315-quiet-vs-forcer-by-level.md`) has the quiet cell at **0% at 98G (n=2)** and 5% at 97G (n=39), and
treated the *earlier* peer quiet-cell fault as an outlier on that basis.

With a **second** quiet-cell fault at 98G, that reading needs revising: quiet at 98G is now 1 fault in 3
arms pooled (this file's arm plus the earlier `ga4096-q`, against 2 clean). That is still consistent with a
low rate, but it is **no longer zero**, and the n=2 on which the "0% at 98G" rested is now the weaker leg of
that file's argument.

What is unaffected: the **cross-cell contrast at matched level**. Cell `e` at 98G is ~95% (41/43) against a
quiet rate that is now bounded well below it regardless of the exact quiet figure. The claim "the forcer
dominates by roughly an order of magnitude" survives; the specific "quiet is 0% at 98G" does not, and is
corrected here.

## Reading against the pre-registration

For the alignment question the branch is unchanged: **both alignments fault, 2 of 2 each**, meeting the
pre-registered "no difference" outcome that §57 pre-committed to abandoning the FA path on — stated
one-sidedly, since n=2 per alignment still cannot bound a *rate* difference.

---

# **SUPERSEDED IN PART (2026-10-13)** — the "every arm faulted" sections above are WRONG

Everything above that asserts the peer's alignment arms faulted — the "both alignments faulted AGAIN" section,
the 4-arm and 5-arm tallies, and the "distinct responses (of 66)" table — **is incorrect**, and the error is
mine.

The arms were scored with `md5sum` over each streaming `resp.*` file, so responses carrying **identical text**
differed by a byte or two of chunk framing and counted as distinct. Sampling them directly shows it:

    [2212 bytes] {"index":0,"content":" We'll assume the reader has a basic understanding of LLMs but wants…
    [2211 bytes] {"index":0,"content":" We'll assume the reader has a basic understanding of LLMs but wants…

Comparing the `content` field instead gives the true picture:

    gb4096-e   66 responses   distinct_content = 1   -> CLEAN
    gb256-e    66 responses   distinct_content = 1   -> CLEAN
    gb4096-q   66 responses   distinct_content = 1   -> CLEAN

So the peer's own reading is the correct one: their replication showed mostly **clean** eviction arms, which is
why they concluded **UNINFORMATIVE** — their bespoke composition did not reproduce the frozen driver's trigger
rate, so the run cannot test layout sensitivity either way.

**What survives here**: the power argument (n=1, later n=2, cannot bound an alignment *rate* difference at a
~95% base rate) — that reasoning is independent of the parse. **What does not survive**: every claim that the
FA family was excluded as "not sufficient" *on the basis of these arms*, because those arms were clean.

See `315-galign-CORRECTION-my-parse-was-wrong.md` for the full correction and the downstream files affected.
