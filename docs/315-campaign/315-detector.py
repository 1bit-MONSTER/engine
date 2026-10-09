#!/usr/bin/env python3
"""engine#315 detector: the pre-registered 66-request shape set, with Face B measured.

Why this exists: `~/wt/rt-det.sh` runs the same requests but decodes the response with
`json.loads(r.read())` inside a bare `except`, so a body that is not valid UTF-8 is
indistinguishable from any other request failure. Face B (invalid-UTF-8 output surfacing as
HTTP 500 / parse failures) is a first-class acceptance criterion in the pre-registration, so
the raw bytes have to be inspected.

Shape set (identical to rt-det.sh, so arms stay comparable with the recorded study):
  1. 8 identical greedy requests, n_predict=96, temperature=0, cache_prompt=false, each followed
     by a distractor request of a different length;
  2. 50 mixed-length requests: distractor 3-900 words, n_predict 4-64.
Total 8 + 8 + 50 = 66 requests.

Usage: 315-detector.py <port> <outdir> <nmix>
Writes <outdir>/detector.json and prints one summary line.
"""
import json
import os
import random
import sys
import urllib.error
import urllib.request

PROMPT = ("Explain in detail how a transformer language model generates text, step by step, "
          "including attention, the KV cache and sampling.")
WORDS = "the quick brown fox jumps over a lazy dog while seven tired engineers measure kernel launch gaps on a hot laptop".split()


def request(port: int, prompt: str, n_predict: int) -> dict:
    """One request, recording the raw bytes rather than trusting json.loads.

    The timeout is deliberately bounded: an arm whose server stops responding must be recorded as a
    transport failure and end quickly. The inherited 600 s per request let a stalled server hold the
    shared box for hours (observed on the second smoke arm).
    """
    timeout = float(os.environ.get("DET_TIMEOUT", "180"))
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                       "cache_prompt": False}).encode()
    rec = {"http_status": None, "utf8_ok": False, "json_ok": False, "content": None,
           "excerpt_hex": None, "error": None}
    try:
        with urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                                       {"Content-Type": "application/json"}), timeout=timeout) as resp:
            rec["http_status"] = resp.status
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        rec["http_status"] = exc.code
        raw = exc.read() if hasattr(exc, "read") else b""
    except Exception as exc:                                    # connection died mid-arm
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec

    if rec["excerpt_hex"] is None:
        rec["excerpt_hex"] = raw[:32].hex()
    try:
        text = raw.decode("utf-8")
        rec["utf8_ok"] = True
    except UnicodeDecodeError as exc:
        # Face B: the body itself is not valid UTF-8. Record where it starts.
        rec["error"] = f"UnicodeDecodeError at byte {exc.start}"
        return rec
    try:
        rec["json_ok"] = True
        rec["content"] = json.loads(text).get("content", "")
    except Exception as exc:
        rec["error"] = f"JSONDecodeError: {exc}"
    return rec


def main() -> int:
    port, outdir, nmix = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    rng = random.Random(7)                     # same seed as rt-det.sh: comparable arms
    summary = {"requests": 0, "ok": 0, "utf8_failures": 0, "parse_failures": 0,
               "http_failures": 0, "transport_failures": 0, "identical_distinct": None,
               "first_failure": None, "truncated": False}

    def note(rec: dict) -> bool:
        summary["requests"] += 1
        if rec["http_status"] is None:                      # connection died / timeout: not Face B
            summary["transport_failures"] += 1
            summary["first_failure"] = summary["first_failure"] or (rec["error"] or "transport")
            return False
        if not rec["utf8_ok"]:                               # Face B: body is not valid UTF-8
            summary["utf8_failures"] += 1
            summary["first_failure"] = summary["first_failure"] or (rec["error"] or "non-utf8 body")
            return False
        if not rec["json_ok"]:                               # body decoded but is not the API JSON
            summary["parse_failures"] += 1
            summary["first_failure"] = summary["first_failure"] or (rec["error"] or "json parse")
            return False
        if rec["http_status"] != 200:
            summary["http_failures"] += 1
            summary["first_failure"] = summary["first_failure"] or f"HTTP {rec['http_status']}"
            return False
        summary["ok"] += 1
        return True

    outs = []
    for i in range(8):
        rec = request(port, PROMPT, 96)
        if not note(rec):
            summary["truncated"] = True
            break
        outs.append(rec["content"])
        if not note(request(port, "Summarize: " + " ".join(rng.choice(WORDS) for _ in range(20 + 37 * i)),
                          8 + 5 * i)):
            summary["truncated"] = True
            break
    summary["identical_distinct"] = len(set(outs)) if outs else None

    if not summary["truncated"]:
        for _ in range(nmix):
            prompt = "Continue the story: " + " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 900)))
            if not note(request(port, prompt, rng.randint(4, 64))):
                summary["truncated"] = True
                break

    with open(f"{outdir}/detector.json", "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"detector: req={summary['requests']}/66 ok={summary['ok']} "
          f"utf8_fail={summary['utf8_failures']} parse_fail={summary['parse_failures']} "
          f"http_fail={summary['http_failures']} transport_fail={summary['transport_failures']} "
          f"identical_distinct={summary['identical_distinct']} truncated={summary['truncated']} "
          f"first_failure={summary['first_failure']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
