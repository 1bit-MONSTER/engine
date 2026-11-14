#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""token_rate_floor.py: fail when a model generates slower than a floor, so a silent slowdown is caught.

engine#374 is why this exists. `model_buffer_from_mmap()` retagged a CPU buffer as `mapped_buft` while leaving
its context a CPU context. The result was **correct output at 8.4 t/s instead of 31.3** -- a 3.7x regression that
no test caught, because every test that watched the buffer was watching the *assertion* it satisfied, not the
speed the machine was doing it at. `hrx-buffer-test` went red much later, on an alignment assert, and the rate
had been quietly wrong for as long as the retag had.

A rate floor is the direct measurement of the thing users feel, and it fails in the direction a correctness
check cannot: a run that is 3x too slow still produces the right tokens.

    token_rate_floor.py --log run.log --floor 20            # judge a saved llama-cli / 1bit run
    token_rate_floor.py --log run.log --floor 20 --json      # machine-readable verdict
    token_rate_floor.py --self-test                         # the four arms, no device needed

**Set the floor below the MINIMUM of at least three interleaved runs, not below a median or a single good one.** A single
rate is not a stable subject on this box: four *identical* 128-token decode runs at one config on engine#679 measured
7.94 / 11.78 / 10.39 / 12.57 t/s -- median 11.09, **spread 1.58x**. A floor judged on one sample flaps for reasons that
have nothing to do with a regression, and a flapping row gets ignored, which is how noisy checks die. Judging a range
instead (the way tools/ci/medians.py decides A/B by disjoint ranges) is more correct and more work; this file takes the
cheap honest path: one sentence here, and a caller who runs three times.

The log is llama-cli's own output ("llama_perf_context_print: prompt eval time = ... 31.3 tokens per second"),
or a one-line `N tokens per second`; the LAST rate in the file is used, because a warmup line is not the run.
Exit: 0 met the floor, 1 below it or unparseable, 77 no rate in the log at all (a skip, not a red).
"""
import argparse
import json
import re
import sys

SKIP = 77

# llama-cli: "eval time =     200.00 ms /    32 runs   (    6.25 ms per token,   160.00 tokens per second)"
# and shorter forms ("31.3 t/s", "31.3 tok/s") for one-line harnesses.
_RATE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:tokens per second|tok/s|t/s)")


def parse_rates(text):
    """Every rate in the log, in order. Multiline, because llama-cli prints prompt and eval rates separately."""
    return [float(m) for m in _RATE.findall(text or "")]


def verdict(rates, floor, subject=""):
    """-> (exit_code, dict). The LAST rate is the generation rate; earlier ones are prompt eval."""
    if not rates:
        return SKIP, {"status": "skip", "subject": subject,
                      "why": "no token rate in the log -- the run did not report one"}
    last = rates[-1]
    # A floor of 0 means "record, do not judge" -- useful while calibrating on a new box.
    if floor <= 0:
        return 0, {"status": "recorded", "rate": last, "floor": floor, "subject": subject}
    if last < floor:
        return 1, {"status": "below_floor", "rate": last, "floor": floor, "subject": subject,
                   "why": f"generated {last:g} tokens/s, floor is {floor:g} -- a slowdown, not a failure"}
    return 0, {"status": "ok", "rate": last, "floor": floor, "subject": subject}


def self_test():
    """Four arms. The second is the one this file exists for: #374's own before/after pair."""
    arms = [
        # (name, log text, floor, expected exit)
        ("#374 after the fix (31.3 t/s) meets a 20 floor",
         "llama_perf_context_print: eval time = 200.00 ms / 32 runs (6.25 ms per token, 31.3 tokens per second)\n", 20.0, 0),
        ("#374 before the fix (8.4 t/s) fails the same floor",
         "llama_perf_context_print: eval time = 200.00 ms / 32 runs (119 ms per token, 8.4 tokens per second)\n", 20.0, 1),
        ("a log with no rate is a SKIP (77), not a pass",
         "llama-cli: loading model\nerror: failed to load\n", 20.0, SKIP),
        ("the last rate wins: prompt 400 then eval 30 is judged on 30",
         "prompt eval: 400.0 tokens per second\neval: 30.0 tokens per second\n", 20.0, 0),
        # engine#2426's lesson applied to rates: a floor without its subject is a recollection, not evidence.
        ("the verdict carries its subject, so a pasted failure is checkable",
         "eval: 8.4 tokens per second\n", 20.0, 1),
    ]
    bad = 0
    SUBJECT = "selftest q4_k_m ctx2048"
    for name, text, floor, want in arms:
        got, detail = verdict(parse_rates(text), floor, SUBJECT)
        ok = got == want
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'FAIL'} {name}{'' if ok else f'  (got {got}, want {want})'}")
        if ok:
            subj = detail.get("subject", "MISSING")
            if subj != SUBJECT:
                print(f"  FAIL arm has no subject in its verdict (got {subj!r})")
                bad += 1
        if ok and got == 1:
            # the failure must NAME the number, or it is a red nobody can act on
            msg = json.dumps(detail)
            if "8.4" not in msg or "floor" not in msg:
                print("  FAIL the below-floor verdict does not name the rate and the floor")
                bad += 1
    print(f"{len(arms)} arms, {bad} failed")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="Fail when a run generates below a token-rate floor.")
    ap.add_argument("--log", help="llama-cli or one-line harness output to judge")
    ap.add_argument("--floor", type=float, default=0.0, help="minimum generation tokens/s (0 = record only)")
    ap.add_argument("--subject", default="", help="model/quant/ctx that produced the rate, e.g. 'qwen3-0.6b Q4_K_M ctx2048'")
    ap.add_argument("--json", action="store_true", help="print the verdict as JSON")
    ap.add_argument("--self-test", action="store_true", help="run the four arms, no device needed")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.log:
        ap.error("--log is required unless --self-test is given")
    try:
        with open(args.log, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"token_rate_floor: cannot read {args.log}: {exc}", file=sys.stderr)
        return SKIP
    code, detail = verdict(parse_rates(text), args.floor, args.subject)
    if args.json:
        print(json.dumps(detail))
    else:
        # A floor is a number tied to a model, quant, context and box. Without the subject a pasted failure is a
        # recollection, not evidence (engine#2426's AUDITED <sha> lesson, applied to rates).
        print(f"JUDGED {args.subject or '<subject not given>'}")
        print(f"{detail['status']}: {detail.get('rate', '-')} tokens/s (floor {args.floor:g})"
              + (f" -- {detail['why']}" if detail.get("why") else ""))
    return code


if __name__ == "__main__":
    sys.exit(main())
