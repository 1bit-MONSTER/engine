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

# The `l_out` sweep: two setup defects, and a configuration that faults as Face B

## Defect 13: the first attempt read a build that does not implement the instrument

The first sweep (`315lout`) set `EXTRA_ENV` with 64 `l_out-*` names and got **zero fingerprint lines in every
arm**. The reason is the same class as the id probe's stale-build defect:

    BIN=${BIN:-/home/bcloud/wt/315instr3-build/bin}      # the rig's default
    strings 315instr3-build/bin/libggml-hrx.so | grep -c hrx-tensor-fp   ->  0
    strings 315fp-build/bin/libggml-hrx.so     | grep -c hrx-tensor-fp   ->  1

**The rig's default build does not implement `GGML_HRX_TENSOR_FP` at all.** The `315nan` session reached the
fingerprint only because its *fix* cell used `315fp-build`; the record already says as much ("that build not
implementing `GGML_HRX_TENSOR_FP`"). My `EXTRA_ENV` was therefore inert. Fixed by overriding `BIN`, verified
before reading results: all four arm servers are `315fp-build/bin/llama-server`.

## Defect 14: the arms in this configuration fault as Face B, which a `NANONLY` sweep cannot see

With the correct build the run produced **four arms, all in regime (`gtt_start=98G`), all faults** — and **still
zero fingerprint lines**. Not a defect this time:

    round 1 arm q: FAULT nan=0 ... http=1 s500=1 unparsed=2   gtt_start=98G
    round 1 arm e: FAULT nan=0 ... http=1 s500=1 unparsed=2   gtt_start=98G
    round 1 arm c: FAULT nan=0 ... http=1 s500=1 unparsed=2   gtt_start=98G
    round 2 arm e: FAULT nan=0 ... http=1 s500=1 unparsed=2   gtt_start=98G

**`nan=0` on every arm: these are all Face B** — invalid-UTF-8 / HTTP 500, with finite logits. The sweep counts
NaN in `l_out-*` and sets `GGML_HRX_TENSOR_FP_NANONLY=1`, so a run with no NaN anywhere **correctly prints
nothing**. The pre-registration assumed Face A arms; this configuration produced none.

**The sweep is therefore uninformative for Face A as run, and says so.** Its NANONLY design has a blind spot the
pre-registration did not anticipate: a null output is ambiguous between "every layer clean" and "no Face-A fault
occurred", and only the arm's `nan=` field distinguishes them.

## What this run does show, which is new

**A configuration that faults as Face B, reproducibly, in every cell, at `req=1…3`.** Face B is the rarer face in
this campaign and the one with the least evidence; here it appears in the quiet, eviction *and* CPU-pressure
cells alike, and the arms fault almost immediately (`req=1`, `3`, `1`) rather than after tens of requests. That
is distinct from the Face-A arms this campaign has been chasing, which fault at `req=6…54`.

Recorded, not interpreted: whether the 64-tensor name list, the `315fp-build` binary, or ordinary run-to-run
variation is responsible is **not** established by four arms, and the instrument is a known inducer (§39). Three
candidates, none excluded.

## Correction to the pre-registration's own claims

It states "an absent layer is a zero-NaN layer, not a missing measurement". **That is only true once the
instrument is verified live** — and in this run it was not, at least not before the `BIN` fix, and after the fix
the arms carried no NaN for it to report. The claim as written was too strong.

---

# Correction to the correction: these ARE Face B — and Face B's output is valid UTF-8

The section above claims the arms "are all Face B" and then, two paragraphs later, that they are not, on the
grounds that the recovered text is valid UTF-8. **The first claim is right and the second is wrong**, and the way
it was wrong is worth recording because it turned on which definition is authoritative.

**The campaign's operational definition of Face B is the format failure, not a UTF-8 check.**
`evidence/315-both-faces-one-trigger.md:40`:

> **Face B** is the format failure (`common_chat_peg_parse: unparsed Content-only output` → HTTP 500)

and `evidence/315-ab1-arm-A-result.md:53` tags exactly this signature as Face B:

    round 2 arm e: FAULT  nan=0 http=1 s500=1 unparsed=2  side=1/1  <- FACE B

The `315lout2` arms carry that signature verbatim (`nan=0 http=1 s500=1 unparsed=2`). **They are Face B.** The
objective's summary phrase ("invalid-UTF-8 output surfacing as HTTP 500 parse failures") describes one
manifestation; the campaign's own definition is the `Content-only` parse failure, and that is what the rig scores.

## What the replaced text actually is — the useful part

Checked, because the distinction decides whether Face B involves a UTF-8 problem at all:

    $ grep -a "unparsed Content-only" server.log | head -1 | iconv -f UTF-8 -t UTF-8   ->  VALID UTF-8
    od -c  ->  plain ASCII throughout the recovered fragment

    unparsed Content-only output:  We'll assume the reader has a basic understanding of LLMs but wants to
    understand the internal mechanics.

Two things stand out and both are informative:

1. **A Face B output is valid UTF-8.** So Face B is not, in this arm, a byte-encoding failure — it is the
   sampler producing a **finite but wrong distribution**, whose text the template parser then rejects. That is
   the campaign's own recorded reading of Face B (`ISSUE-COMMENT-3`: *"if it held the previous token's finite
   logits → Face B: the sampler draws from a stale distribution"*), now with a concrete example.
2. **The recovered fragment starts mid-essay** — "We'll assume the reader has a basic understanding of LLMs but
   wants to understand the internal mechanics." is a *continuation*, not an opening line, and the log shows the
   double space of a stripped prefix. That is what a corrupted opening looks like from the parser's side: the
   response does not begin where a valid response would, so no template matches it.

## Confirmed: this configuration is not the baseline

    clean arm 315row-p1l (66/66)                  unparsed Content-only: 0
    shipped-pin baselines v315x-c1/c2/c3          unparsed Content-only: 0
    315lout2 arms (req=1, 3, 1)                   unparsed Content-only: 2 each

So the signature is **absent in clean and baseline arms and present in every instrumented arm** — a genuine
deviation of this configuration, not a harness constant.

## And why the `l_out` sweep still reads nothing

Unchanged by the above: **Face B has `nan=0`.** A NaN-counting sweep over `l_out-*` with `NANONLY=1` is
uninformative for Face B *by construction*, and this configuration produced no Face A arm. The sweep's null is
therefore **not** evidence that any layer is clean — it is evidence that no Face A arm occurred to sweep.
