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
