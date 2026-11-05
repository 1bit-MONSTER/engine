#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Can a *batched* cell be repeated? The half of the determinism question that a single-request
repeat does not answer.

`tools/logprob_agree.py` repeats one request at a time, which is the solo case. A scoreboard's
`--parallel N` cell is measured by issuing N concurrent requests, so its reproducibility question is:
if you replay the same N concurrent requests, do you get the same N answers? Measured here as R
rounds of N byte-identical requests issued through a barrier, each round fingerprinted per copy.

Reported per width: within-run (do the N copies of one round agree with each other) and
round-to-round (does copy i of round r carry the same bits as copy i of round r+1), plus the number
of distinct fingerprints seen at all. Usage:

  batched_repeat.py --url http://127.0.0.1:18120 --width 4 --rounds 4 --max-tokens 32 --json out.json
  batched_repeat.py --selftest
"""
import argparse
import hashlib
import itertools
import json
import struct
import sys
import threading
import urllib.request


def ask(url, model, prompt, max_tokens, top):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "top_k": 1, "seed": 1234, "max_tokens": max_tokens,
        "cache_prompt": False, "logprobs": True, "top_logprobs": top,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        ch = json.loads(r.read())["choices"][0]
    toks = (ch.get("logprobs") or {}).get("content") or []
    return [(t.get("id"), t.get("logprob"), [(x.get("id"), x.get("logprob")) for x in (t.get("top_logprobs") or [])])
            for t in toks]


def f32bits(x):
    """The wire carries a JSON decimal; the engine's value is float32. Returns the float32 bits, or
    None when the value does not survive the cast (counted by the caller)."""
    if x is None:
        return None
    try:
        return struct.unpack("<I", struct.pack("<f", float(x)))[0]
    except (OverflowError, ValueError):
        return None


def fingerprint(copy):
    """One hash per copy over (token id, chosen logprob bits, top-5 bits).

    Every field is tagged before it is hashed, so a missing value can never collide with a real one:
    a route that omits token ids (the NPU forward path) or log-probabilities stays distinguishable
    from a route whose id is 0 or whose log-probability is +0.0, instead of both hashing to the same 0.
    """
    def tag(kind, payload=b""):
        h.update(bytes([kind]) + payload)

    h = hashlib.sha256()
    for tid, lp, top in copy:
        tag(1, repr(tid).encode()) if tid is not None else tag(0)
        bits = f32bits(lp)
        tag(2, struct.pack("<I", bits)) if bits is not None else tag(3)
        for tid2, lp2 in top:
            tag(1, repr(tid2).encode()) if tid2 is not None else tag(0)
            bits2 = f32bits(lp2)
            tag(2, struct.pack("<I", bits2)) if bits2 is not None else tag(3)
    return h.hexdigest()[:16]


def compare(a, b):
    """(ids_equal, first_lp_bit_divergence, max_abs_dlogp) for two copies of equal-token sequences."""
    if len(a) != len(b):
        return (False, 0, float("inf"))
    ids_ok, first_bad, worst = True, None, 0.0
    for i, ((id_a, lp_a, top_a), (id_b, lp_b, top_b)) in enumerate(zip(a, b)):
        if id_a != id_b:
            ids_ok = False
        if lp_a is not None and lp_b is not None:
            d = abs(float(lp_a) - float(lp_b))
            if d:
                worst = max(worst, d)
                if first_bad is None:
                    first_bad = i
    return (ids_ok, first_bad, worst)


def vs_solo(solo, copy):
    """How far one concurrent copy is from the SOLO answer: the number a batched cell has to report when
    its row next to a solo row is supposed to mean the same computation."""
    n = min(len(solo), len(copy))
    if n == 0:
        return {"positions": 0, "mean_abs_dlogp": 0.0, "same_token": 0.0, "exact_bits": 0.0}
    deltas = [abs(float(x[1]) - float(y[1])) for x, y in zip(solo[:n], copy[:n])
              if x[1] is not None and y[1] is not None]
    return {
        "positions": n,
        "mean_abs_dlogp": sum(deltas) / len(deltas) if deltas else 0.0,
        "same_token": sum(1 for x, y in zip(solo[:n], copy[:n]) if x[0] == y[0]) / n,
        "exact_bits": sum(1 for x, y in zip(solo[:n], copy[:n])
                          if x[0] == y[0] and f32bits(x[1]) == f32bits(y[1])) / n,
    }


def round_of(url, model, prompt, max_tokens, top, width, join_timeout=900, request=None):
    """Issue `width` byte-identical requests through a barrier so they land in one decode batch.

    The barrier is passed *before* the request is made, so a copy that fails cannot strand the others
    waiting on it; the joins are bounded anyway and a copy that never returns is recorded as an error
    rather than hanging the round, after which the barrier is aborted so any straggler raises instead of
    waiting forever. `request` lets the selftest inject a failing copy without touching module state.
    """
    out = [None] * width
    err = [None] * width
    barrier = threading.Barrier(width)

    def one(i):
        try:
            barrier.wait(timeout=join_timeout)
            out[i] = (request or ask)(url, model, prompt, max_tokens, top)
        except Exception as e:  # noqa: BLE001 - a failed copy is a result, not a crash
            err[i] = repr(e)

    ts = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(width)]
    for t in ts:
        t.start()
    for i, t in enumerate(ts):
        t.join(join_timeout)
        if t.is_alive() and err[i] is None:
            err[i] = f"copy {i} did not finish within {join_timeout}s"
    barrier.abort()  # a straggler raises BrokenBarrierError instead of waiting for a partner that left
    return out, err


def selftest():
    base = [(5, -0.5, [(5, -0.5), (6, -1.25)]), (7, -0.25, [(7, -0.25)])]
    assert fingerprint(base) == fingerprint(list(base)), "stable"
    moved = [(5, -0.50000012, [(5, -0.5), (6, -1.25)]), (7, -0.25, [(7, -0.25)])]
    assert fingerprint(base) != fingerprint(moved), "1-ulp logprob change must change the fingerprint"
    assert compare(base, base) == (True, None, 0.0)
    ids, first, worst = compare(base, [(5, -0.5, []), (9, -0.25, [])])
    assert ids is False and worst == 0.0, "id divergence without a delta"
    assert compare(base, [(5, -0.5000001, []), (7, -0.25, [])])[1] == 0, "first divergence index"
    assert f32bits(-0.5) == struct.unpack("<I", struct.pack("<f", -0.5))[0]
    # a missing field must not collide with a real one (the reviewer's point on the old `or 0`)
    assert fingerprint([(None, None, [])]) != fingerprint([(0, 0.0, [])]), "None vs 0/+0.0 must differ"
    assert fingerprint([(5, None, [])]) != fingerprint([(5, 0.0, [])]), "missing logprob vs +0.0"
    # a failing copy is a result, not a hang, and it must not strand its partners; injected, not swapped
    # into module state (the review point about monkeypatching `ask`)
    def flaky(url, model, prompt, max_tokens, top):
        if getattr(flaky, "n", 0) == 1:
            flaky.n += 1
            raise RuntimeError("simulated backend error")
        flaky.n = getattr(flaky, "n", 0) + 1
        return base

    out, err = round_of("http://x", "m", "p", 2, 5, 3, join_timeout=30, request=flaky)
    assert sum(e is None for e in err) == 2, f"two copies ok, one failed: {err}"
    assert sum(c is not None for c in out) == 2, "good copies recorded, failed copy is None"
    assert any("simulated backend error" in (e or "") for e in err), "the failure is reported, not swallowed"
    # the vs-solo scorer: identical copy scores 1.0/1.0/0.0, a moved bit scores 1.0 on tokens but 0 exact
    sc = vs_solo(base, base)
    assert (sc["same_token"], sc["exact_bits"], sc["mean_abs_dlogp"]) == (1.0, 1.0, 0.0), sc
    sc2 = vs_solo(base, [(5, -0.50000012, [(5, -0.5)]), (9, -0.25, [])])
    assert sc2["same_token"] == 0.5 and sc2["exact_bits"] == 0.0 and sc2["mean_abs_dlogp"] > 0, sc2
    assert vs_solo([], [])["positions"] == 0, "empty copies do not divide by zero"
    print("selftest: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18120")
    ap.add_argument("--model")
    ap.add_argument("--prompt", default="Explain in one paragraph why the sky appears blue.")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--top-logprobs", type=int, default=5)
    ap.add_argument("--width", type=int, default=4, help="concurrent identical requests per round")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--vs-solo", action="store_true",
                    help="also score every concurrent copy against one solo request (the scoreboard number)")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0

    model = a.model
    if not model:
        with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
            model = json.loads(r.read())["data"][0]["id"]

    print(f"{a.url} model {model}: {a.rounds} rounds x {a.width} identical concurrent requests, "
          f"{a.max_tokens} tokens, greedy")

    solo, solo_err = None, None
    if a.vs_solo:
        one, solo_err = round_of(a.url, a.model or model, a.prompt, a.max_tokens, a.top_logprobs, 1)
        solo = one[0]
        if solo is None or solo_err[0]:
            print(f"  solo baseline failed: {solo_err[0]}")
            a.vs_solo = False

    rounds, fps = [], []
    scores = []
    for r in range(a.rounds):
        copies, err = round_of(a.url, a.model or model, a.prompt, a.max_tokens, a.top_logprobs, a.width)
        bad = [(i, e) for i, e in enumerate(err) if e]
        if bad:
            print(f"  round {r}: {len(bad)} failed copies: {bad[:2]}")
        ok = [c for c in copies if c is not None]
        rounds.append(copies)
        fps.append([fingerprint(c) if c is not None else None for c in copies])
        within = [compare(ok[0], c) for c in ok[1:]] if ok else []
        same = sum(1 for w in within if w[0] and w[2] == 0.0)
        worst = max([w[2] for w in within if w[2] != float("inf")] or [0.0])
        extra = ""
        if solo is not None:
            got = [vs_solo(solo, c) for c in ok]
            scores.extend(got)
            extra = (f", vs solo mean |dlogp| {min(x['mean_abs_dlogp'] for x in got):.3e}"
                     f"..{max(x['mean_abs_dlogp'] for x in got):.3e}"
                     f", same token {min(x['same_token'] for x in got):.3f}"
                     f"..{max(x['same_token'] for x in got):.3f}")
        print(f"  round {r}: copies agree with copy0: {same}/{max(len(ok)-1,0)} bitwise"
              f"{'' if not within else f', worst |dlogp| {worst:.3e}'}"
              f", distinct fingerprints {len(set(f for f in fps[r] if f))}/{len(ok)}{extra}")

    n_cmp = n_same = 0
    worst_x = 0.0
    for r in range(1, a.rounds):
        for i in range(a.width):
            a_, b_ = rounds[r - 1][i], rounds[r][i]
            if a_ is None or b_ is None:
                continue
            n_cmp += 1
            same, first, worst = compare(a_, b_)
            if same and worst == 0.0:
                n_same += 1
            elif worst != float("inf"):
                worst_x = max(worst_x, worst)
    all_fps = set(f for row in fps for f in row if f)
    print(f"\n  round-to-round, same copy index: {n_same}/{n_cmp} bitwise identical"
          f" (worst |dlogp| {worst_x:.3e}); distinct fingerprints over all rounds: {len(all_fps)}"
          f" of {a.rounds * a.width} copies")
    verdict = "REPEATABLE (batched)" if n_cmp and n_same == n_cmp else "NOT repeatable (batched)"
    print(f"  ==> width {a.width}, {a.rounds} rounds: {verdict}")
    if scores:
        mid = lambda k: sorted(x[k] for x in scores)[len(scores) // 2]
        ident = sum(1 for x in scores if x["exact_bits"] == 1.0)
        print(f"  vs the SOLO answer ({len(scores)} copies): identical {ident}/{len(scores)},"
              f" mean |dlogp| min {min(x['mean_abs_dlogp'] for x in scores):.3e}"
              f" median {mid('mean_abs_dlogp'):.3e} max {max(x['mean_abs_dlogp'] for x in scores):.3e},"
              f" same token median {mid('same_token'):.3f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"url": a.url, "model": model, "width": a.width, "rounds": a.rounds,
                       "max_tokens": a.max_tokens, "fingerprints": fps,
                       "round_to_round_identical": [n_same, n_cmp],
                       "distinct_fingerprints": len(all_fps), "verdict": verdict,
                       "vs_solo": scores or None}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
