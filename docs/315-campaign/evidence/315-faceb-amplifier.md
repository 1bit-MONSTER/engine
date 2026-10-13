<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# A near-deterministic Face-B configuration — and the limit of what it proves

## What was observed

The `315lout2` session set `EXTRA_ENV="GGML_HRX_TENSOR_FP=l_out-0..63 GGML_HRX_TENSOR_FP_NANONLY=1"` against
`BIN=~/wt/315fp-build/bin`, 3 rounds on the frozen rig. Every arm:

| arm | cell | result | signature | `gtt_start` |
|---|---|---|---|---|
| r1q | quiet | FAULT | `nan=0 http=1 s500=1 unparsed=2` | 98G |
| r1e | eviction | FAULT | `nan=0 http=1 s500=1 unparsed=2` | 98G |
| r1c | CPU pressure | FAULT | `nan=0 http=1 s500=1 unparsed=2` | 98G |
| r2e | eviction | FAULT | `nan=0 http=1 s500=1 unparsed=2` | 98G |
| r2q | quiet | FAULT | `nan=0 http=1 s500=1 unparsed=2` | 98G |

**5 of 5 arms, all in regime, all Face B, and every one at `req=1` or `3`** — where the Face-A arms this campaign
has chased fault at `req=6…54`. Face B is the campaign's rarer face, and here it appears in **all three cells**,
including the quiet one, on the *first request*.

Verbatim from a faulting arm's own records:

    prompt:  "Explain in detail how a transformer language model generates text, step by step, including
              attention, the KV cache and sampling."
    http_status: 500     utf8_ok: true     content: ""
    server:  common_chat_peg_parse: unparsed Content-only output:  We'll assume the reader has a basic
             understanding of LLMs but wants to understand the internal mechanics.

The model's text is **valid UTF-8** and a **mid-essay fragment**: the response does not begin where a valid
response would, so no template matches it. That is consistent with the campaign's recorded reading of Face B —
the sampler drawing from a finite but wrong distribution — now with a concrete, quotable instance.

## Why this is useful

**A fault that appears on the first request in any cell is far cheaper to test against than one that appears in
~5 % of arms.** If it shares its mechanism with the production fault, this configuration is an amplifier of the
kind the campaign needs to validate a candidate fix without a 60-arm campaign per attempt — which is the single
biggest practical obstacle in the objective's criterion 3.

## Why it is *not* yet evidence about the production fault

Stated plainly, because this campaign has already been bitten by exactly this:

1. **The relationship to the production fault is unestablished.** The signature is **absent in clean arms**
   (`315row-p1l`, 66/66, `unparsed=0`) and in the shipped-pin baselines (`v315x-c1/c2/c3`, all `unparsed=0`), and
   present in every instrumented arm. It is induced by this configuration.
2. **The instrument is a known inducer** (§39): the fingerprint already raises the fault rate by nearly two
   orders of magnitude, and this run reads **64 tensors per graph**, far more heavily than `tail:16`. The record's
   own standing warning applies — "until an instruments-off arm is run, this run measures the instrument".
3. **Candidates not excluded:** the 64-name list, the `315fp-build` binary, and ordinary run-to-run variation.
   Five arms in one configuration separate none of them.

So: **a reproducible Face-B configuration, not a measured production mechanism.** It is a tool, and its value
depends on a hypothesis — that the amplifier and the production fault share a mechanism — that has not been
tested. The way to test it would be a candidate fix that suppresses both; none exists yet.

## Correction carried

An earlier draft of `315-lout-sweep-result.md` argued these arms are not Face B because the text is valid UTF-8.
That was wrong: the campaign's operational definition of Face B **is** the `Content-only` parse failure
(`315-both-faces-one-trigger.md:40`), not a UTF-8 check. Corrected there; recorded here because the distinction
between the objective's summary phrase and the campaign's operational definition is exactly the kind of thing
that silently changes a tally.
