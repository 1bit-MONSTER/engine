<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Two defects in the acceptance harness found by fixing Face B: an unattributable face, and a bound over 100%

Found while implementing the user's ruling that Face B is *Content-only / HTTP 500 **or** a genuine
non-UTF-8 body, whichever occurs*. Both defects are in `docs/315-campaign/315-campaign.sh`, the acceptance
campaign summary. Only the first is fixed here; the second is recorded because it is a smaller change to
**how a criterion reads** and should not be made silently.

## Defect 1 — the harness cannot attribute Face B, in two independent ways

**1a. The wrong condition.** The summariser computed

    if r["parse"] > 0 or r["utf8"] > 0:            # <- Face B, as written
    ...
    face_b = sum(1 for r in faults if r["parse"] > 0 or r["utf8"] > 0)

but `parse_failures` and `utf8_failures` are **0 in every Face-B arm on the record** — Face B's recorded
form is `http=1 s500=1 unparsed=2` (`315-faceb-invariant-signature.md`). A real Face B fault could therefore
**never** be counted as Face B; it landed in the `unattributed` bucket, whose own text says "read the server
log", i.e. the harness told its reader to go and find by hand what it was supposed to count.

**1b. The rig does not emit the fields at all.** Worse, the acceptance rig's arm line carries no
`http`/`s500`/`unparsed`:

    acceptance rig (paired):  pair 1 arm c: FAULT nan=0 req=13 ret3=0 parse=0 gtt_start=98G utf8=0 alloc=0 hv52=1 hv56=0
    disc rig:                 round 1 arm q: OK nan=0 req=66 parse=0 utf8=0 http=0 transport=0 s500=0 unparsed=0 hsa=0 ...

So even a corrected condition cannot attribute Face B from a paired-rig console. **This is empirical, not
hypothetical**: `315accept`'s own recorded fault — `pair 1 arm c: FAULT nan=0 req=13` with every counter 0 —
is exactly a Face-B-shaped arm, and the harness labelled it `unattributed`.

**Impact on the criteria.** Criterion 3 asks for *zero faults on both faces*. The **rate** accounting is
sound — it runs over all in-regime arms, so a fault fails acceptance regardless of face, and the test above
confirms `ACCEPTANCE FAILS` still fires. What is broken is the **face attribution**, which criterion 1 needs
(a named mechanism for *both* faces) and which any "zero on both faces" reading depends on. A campaign could
pass while Face B was never once attributed.

**Fix applied to the summariser** (the part that can be fixed here):

- it now parses `http=`, `s500=`, `unparsed=` from the tail **when present**, storing `None` when absent so a
  missing field can never masquerade as a zero;
- `is_face_b(r)` tests `parse > 0 or utf8 > 0 or http > 0 or s500 > 0 or unparsed > 0` — the disjunction the
  user ruled for, covering both the recorded form and the objective's full-body UTF-8 form;
- when the fields are absent it now prints an explicit note that Face B cannot be tested from that line,
  instead of silently bucketing the fault as unattributed.

**Verified on both branches**, running the extracted summariser:

    Test A (line WITHOUT the fields, from 315accept-console.log):
        NOTE: this line carries no http/s500/unparsed field, so Face B's recorded form cannot be tested
        from it - the paired rig does not emit them. ...
        face tally: FaceA=0 FaceB=0 unattributed=1
        ACCEPTANCE FAILS: 1 in-regime arm(s) faulted; ...

    Test B (a line WITH http=1 s500=1 unparsed=2):
        FAULT pair=1 FaceB(parse=0,utf8=0,http=1,s500=1,unparsed=2) req=7 gtt=98G
        FAULT pair=2 unattributed - FAULT with nan=0 and no parse/utf8 counter; ...
        face tally: FaceA=0 FaceB=1 unattributed=1

So the corrected test attributes a real Face-B signature and does **not** over-attribute an all-zero fault.

**Still to do, and it is not in this repository's hands:** the *paired* rig's verdict printer
(`315-paired.sh`, per `docs/preregistration.md`) must emit `http=`, `s500=`, `unparsed=` for the fix to bite
in an acceptance run. Until then the harness is correct but the rig is silent, and the note says so rather
than implying Face B was tested.

## Defect 2 — the "rule of three" bound prints probabilities above 100%

`315-campaign.sh` prints, for a clean cell of `n` arms:

    one-sided 95% upper bound (rule of three, 3/n) = {3.0/n*100:.2f}%

which for small `n` is **not a probability**:

| n | printed |
|---|---|
| 1 | **300.00%** |
| 2 | **150.00%** |
| 3 | 100.00% |
| 60 | 5.00% (correct and the useful case) |

The rule of three is only a bound for `3/n ≤ 1`, i.e. `n ≥ 3`. For the acceptance criterion's `n = 60` the
number is right (5%, matching the objective's own power statement), so this does not affect the intended
reading — but a partial or aborted campaign printing `300%` invites a reader to take the bound as satisfied.
**Not changed here:** capping it is trivial (`min(100.0, 3.0/n*100)`) but deciding what an under-powered
campaign should *say* is a criterion question, and the objective is immutable.

## Evidence

    sed -n '52,152p' docs/315-campaign/315-campaign.sh > /tmp/sum.py   # the extracted summariser
    python3 /tmp/sum.py /tmp/t_paired.txt    # test A
    python3 /tmp/sum.py /tmp/t_disc.txt      # test B

Both test consoles and the full outputs are reproduced in this session's transcript; the patch is in
`315-campaign.sh` and the summariser is syntax-checked (`ast.parse`) and run on both inputs.
