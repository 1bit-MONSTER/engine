<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Invalid UTF-8 has never been observed on this record — and Face B's recorded form is HTTP 500 + Content-only

Sweep of every per-request record and every greedy-output file in the archive. No GPU time. This bears on how
the objective defines Face B, so it is recorded precisely.

## The sweep

| source | what it holds | result |
|---|---|---|
| `~/wt/rt-det/*/identical.json` (267 files) | the 8 identical greedy responses per arm, as text | **0** files contain invalid UTF-8; **0** contain any non-ASCII byte |
| `~/wt/rt-det/*/detector-records.json` (87 files, 3,200 records) | every request: `http_status`, `utf8_ok`, `json_ok`, `content`, `excerpt_hex` | **3** records whose `excerpt_hex` is not valid UTF-8 — all three are **excerpt artifacts** (below) |
| the `utf8_ok` field (computed on the **full** body, `315-detector.py:59-60`) | — | `False` in exactly **34** records, and all 34 have `http_status = None`, i.e. transport failures with no body at all |

Status distribution over the 3,200 records: **200 → 3,149**, **500 → 17**, **None → 34**.

## The three candidates are truncation artifacts, not invalid UTF-8

    arm              req  http  utf8_ok  json_ok  excerpt_hex (32 bytes)
    315asan2-r1c      15  200   True     True     7b22696e646578... 20 e7
    315gord1b-r2e     32  200   True     True     7b22696e646578... ef
    315inp2-r3e        4  200   True     True     7b22696e646578... 2e e7 9a 84 e3

**`excerpt length bytes: 32` for all three**, and every decode error is `unexpected end of data` at position
31 — the excerpt cuts a multi-byte character in half. Meanwhile `utf8_ok = True` and `json_ok = True`, both of
which are computed on the **whole** body, and `content` is a proper string:

    ' lazy -. 的 for，_  if if of of for for for 的，，，。-r for for(R for for-g-r...'

So **the bodies are valid UTF-8**; only the fixed 32-byte excerpt is not. This is the record's own recurring
lesson in a new place — *the excerpt is not the body*, exactly as *the line count is not the occurrence count*
and *the baseline must be read rather than reconstructed*. Had this been read as "3 invalid-UTF-8 outputs" it
would have manufactured the very Face B the objective names.

## A control that failed, recorded so it is not re-proposed

The three bodies are visibly degenerate (`for for for for`, `的，，，。`), which suggested a third manifestation —
*wrong but well-formed output at HTTP 200, flagged by neither detector*. A crude repetition ratio
(`word tokens ÷ distinct word tokens`) over all 2,930 records with ≥8 word tokens **does not support it**: the
maximum value (64.0, i.e. every token identical) is reached by **clean quiet-cell arms** —
`315row-r1q`, `315size-r1q`, `315text1-r1q`, `315lout4-r2c` — as well as by the three candidates, and the median
across all records is 14.25. Degenerate repetition is a property of the prompts the detector sends, not a fault
signal. **The third-manifestation idea is withdrawn before it was written up as a finding.**

## What this leaves

- **Face B as recorded is `http=1 s500=1 unparsed=2`** — i.e. the server produced **Content-only (non-JSON)
  output and answered HTTP 500** — plus the record's own description of the text as *"valid UTF-8 and a
  mid-essay fragment"*. That description is **consistent with every measurement in this sweep**.
- **The objective's phrase "invalid-UTF-8 output" has no supporting measurement on this record.** Not one
  response body — among 3,200 records and 267 greedy-output files — is invalid UTF-8. The two detectors differ
  only in *where* the failure is caught (`rt-det.sh`'s bare `json.loads(r.read())` collapses "bad UTF-8" and
  "not JSON" into one exception; `315-detector.py` separates them), which is why the same event can be
  described either way.
- **The rig cannot presently distinguish a content fault from a truncated arm.** `315-disc.sh:148` makes
  `req < 66` a `FAULT` condition by itself, and every Face-B arm has `req = 5…27`, so `FAULT` conflates "the run
  failed" with "the run stopped early".

## Recommendation, for the user rather than an edit here

The objective is immutable and I am not changing it. But on this evidence, **Face B's operative definition
should be "Content-only / non-JSON output surfacing as HTTP 500"**, with "invalid UTF-8" either dropped or
explicitly noted as unevidenced on this record. That matters for acceptance: a fix campaign must detect
Face B as the HTTP-500/Content-only event, because a UTF-8-validity check would **never fire** and would pass a
faulting arm as clean. If the user wants the objective's wording changed, that needs `/goal-tweak`.

## Reproducing

    python3 - <<'PY'
    import glob, json, os
    bad = 0
    for f in glob.glob(os.path.expanduser("~/wt/rt-det/*/detector-records.json")):
        for r in json.load(open(f)):
            hx = r.get("excerpt_hex")
            if hx:
                try: bytes.fromhex(hx).decode("utf-8")
                except UnicodeDecodeError as e:
                    print(os.path.basename(os.path.dirname(f)), r["i"], r["http_status"], r["utf8_ok"], e)
                    bad += 1
    print("invalid:", bad)
    PY
