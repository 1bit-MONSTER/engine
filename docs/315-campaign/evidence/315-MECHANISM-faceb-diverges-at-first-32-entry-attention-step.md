<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# MECHANISM (Face B, measured): the logits diverge at the first decode step that attends to exactly 32 KV entries — finite, deterministic-before, non-deterministic-after

This is the first time on this record that an intermittent fault has been **localized to an exact decode step
from existing data, with no GPU**. It is measured, reproducible, and it makes a sharp prediction.

## The measurement

`315inp2-r1e` is an eviction arm that faulted (`http_failures: 1`, `first_failure: "HTTP 500"`,
`truncated: true`). Its logits stream is split into per-request sequences at the server's timing lines:

    7 request sequences, 5 distinct logits-hash sequences
      len=96  calls   0..95    prompt A   \  three runs of the same prompt,
      len= 8  calls  96..103   prompt B    | bit-identical logits,
      len=96  calls 104..199   prompt A    | byte-for-byte
      len=13  calls 200..212   prompt C    |
      len=96  calls 213..308   prompt A   /
      len=18  calls 309..326   prompt D
      len=96  calls 327..422   prompt A   <- the FOURTH run, DEGENERATES

and the fourth run diverges from the other three **after exactly 7 logits**:

    hash A[7] = 1ace525926f282f8      (healthy, identical in all three earlier runs)
    hash B[7] = 87265f4d6621dac9      (the faulting run)

**Logits[0..6] are bit-identical across all four runs; logits[7] differs.** The output text confirms the
index exactly — the healthy reply begins *"We'll assume the reader has a basic understanding of LLMs"*, and
`"We'll assume the reader has a"` is **seven** tokens; the faulting run produces those seven and then dissolves
into token soup.

## Why this is not a coincidence of tokenization

The prompt eval is **25 tokens**, so the logits at index `i` are computed from `25 + i` KV entries. **Index 7
therefore reads exactly 32 KV entries** — and 32 is the first clean power-of-two boundary in the sequence. The
healthy runs compute the same logits from the same 32 entries correctly, so position 32 is not itself the fault;
it is the first point at which the fault *can* appear.

## What is excluded, by measurement

- **NaN is excluded.** All **423** `[hrx-logits-fp]` records in the arm report `nan=0`, and `sentinel=0`
  throughout. Face B is not a NaN fault at any point.
- **The sampler is excluded.** The *logits hash* differs, so the logits themselves differ — the fault is
  upstream of sampling, not in token selection.
- **Invalid UTF-8 is excluded**, and could never have fired: the garbage contains `�`, which is **U+FFFD, a
  valid UTF-8 codepoint**. This is exactly why the record found "invalid UTF-8 never observed" in 0 of 3,200
  records. The objective's Face-B phrasing names a signal this code path cannot produce.
- **The prompt is excluded.** Three runs of the identical prompt produced identical logits; only the fourth
  diverged.
- **Bitmap/decode determinism is real.** Logits are bit-identical *across arms* — the hashes
  `a82848752ab26ce7, 28a2f31b05ea1a92, 69a2959adc518ad6, 86b92e1667036a1c` for steps 0–3 of prompt A are the
  same in this arm as in the very first `hrx-logits-fp` sample taken from a different arm earlier in the
  investigation. **So a hash difference between runs of the same prompt is a real state difference, not noise.**

## What the server does

    W common_chat_peg_parse: unparsed Content-only output:  We'll assume the reader has a a 是,is (‑.com �.org...
    W srv operator(): got exception: {"error":{"code":500,"message":"The model produced output that does not match the expected Content-only format","type":"server_error"}}

The generation runs its full length (`n_tokens = 120, truncated = 0`), the parser rejects the body, and the
server converts that into **HTTP 500 with an empty body** — which is why the detector's `utf8_ok` and
`json_ok` are `true` on the 500 row: there are no bytes to be invalid. **Face B, as the harness records it, is
the server's own error response to a degenerate generation, not a corrupted byte stream.**

## The mechanism this points at

The first divergent logits are the first computed from 32 KV entries, and the divergence is **intermittent for
an identical input** — three runs agree, the fourth does not. A finite-but-wrong logits vector produced from
attention over a KV cache, intermittently, at a step boundary, with the graph reused (`graphs reused = 410`),
is the signature of **a stale or mispositioned KV entry rather than corrupt memory**:

- stale/garbage-but-finite KV values mix into attention and yield **finite, plausible, wrong** logits — which is
  what Face B shows and what a memory fault would not;
- it appears only once the attention reads enough entries for the affected one to be included — which is why
  the first divergence is at the 32-entry step and not at step 0;
- it is intermittent because it depends on timing/state, which is also why forced eviction makes it common;
- and it needs no NaN, no invalid bytes, and no crash to produce an HTTP 500.

**This is a hypothesis with a specific, falsifiable prediction, not a claim.** The prediction is: the KV entry
written for the position first included at logits index 7 (`Kcur`/`Vcur` at absolute position 31 or 32) should
differ between the healthy and faulting runs, while everything the healthy prefix depends on is identical.

## What the record must now do

1. **Re-scope the failure definition.** Face B = "the model degrades after a finite, deterministic prefix; the
   server returns 500". Not invalid UTF-8, not HTTP-500-as-parse-failure.
2. **Face A is a different fault.** 12 arms carry `HSA_STATUS_ERROR` and their output is `empty`; no arm shows
   both faces. Face B has `nan=0` throughout. **"Both faces" is one criterion covering two disjoint
   phenomena.**
3. **Score attacks on Face B by logits-hash recurrence**: for a repeated prompt, the first index at which a run
   diverges from its arm's own majority hash sequence. That is a per-step, per-arm, GPU-free fault detector
   with far more power than "did the server return 500", and it works on arms that never 500.

**No mechanism is named yet.** What is established is where the fault enters (first 32-entry attention step), that
it is finite and upstream of sampling, that it is intermittent for identical input, and that the objective's
Face-B definition describes a signal this code path cannot emit.
