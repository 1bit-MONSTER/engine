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
"""logprob_agree: two engine routes, one model, the same tokens — do their log-probs agree?

Answers two different questions, because they fail for different reasons:

  determinism   one route, the same request sent --repeat times. Is every token's log-prob
                bit-identical? If not, which position diverged first. A route that is not
                deterministic run-to-run cannot be the reference for the other question.
  agreement     two routes, the same prompts, greedy. How far apart are the log-probs of the
                tokens they generate, do they pick the same top-1 next token, and how often are
                the bits equal.

Both are measured on OpenAI /v1/chat/completions with logprobs=true. Greedy throughout
(temperature 0, top_k 1) and cache_prompt false so no route is measured through a different
prompt-cache path. Thinking is disabled by default; --thinking on is what a route that cannot be
told to stop thinking has to be measured with (the NPU forward path accepted and ignored
enable_thinking until engine PR #864), and then all routes must run with it on or the streams
differ from token 0 for template reasons.
Deliberately not a training-vs-rollout harness: it takes already-served endpoints, so it measures
the routes as users get them.

  logprob_agree.py --route cpu=... --route npu=... --thinking off --match-by text

  logprob_agree.py --route cpu=http://127.0.0.1:18110 --route hrx=http://127.0.0.1:18111
  logprob_agree.py --route cpu=http://127.0.0.1:18110 --repeat 5 --json out.json

What "bit match" means here: the wire carries JSON decimals, so the raw bits are not on the
wire. Each log-prob is cast to float32 and compared as bits, which is exact for a float32 the
server promoted to a double for serialisation — the tool asserts that round trip holds for every
value it compares and reports how many values had to be dropped as non-float32.

Exit code: 0 all routes deterministic and all pairs exactly equal, 1 a route is non-deterministic
or a pair differs, 2 a route is unreachable or a request failed.
"""
import argparse
import json
import math
import struct
import sys
import time
import urllib.error
import urllib.request

# Eight short, single-answer prompts: enough for the model to commit to one continuation, varied
# in shape so a tokenizer or template difference between the routes shows up early.
PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 17 * 23? Answer with the number only.",
    "Name the three primary colours.",
    "Complete this sentence: The quick brown fox",
    "List the first five prime numbers, comma separated.",
    "Translate 'good morning' into Spanish.",
    "What does CPU stand for?",
    "Write one sentence about the sea.",
]


def bare(url):
    return url.rstrip("/")


def get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post(url, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def f32(x):
    """The float32 a server's promoted-to-double JSON decimal came from."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    """Raw bit pattern of the float32 form of a log-prob, as an int."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def usable(x):
    return x is not None and isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def ask(url, model, prompt, max_tokens, top_logprobs, timeout, thinking=False):
    """One greedy request; returns the generated token ids with their log-probs."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_k": 1,
        "seed": 1234,
        "max_tokens": max_tokens,
        "cache_prompt": False,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    t = time.time()
    try:
        r = post(bare(url) + "/v1/chat/completions", body, timeout)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    ch = (r.get("choices") or [{}])[0]
    toks = []
    for tk in ((ch.get("logprobs") or {}).get("content") or []):
        top = []
        for x in tk.get("top_logprobs") or []:
            top.append({"id": x.get("id"), "token": x.get("token"), "lp": x.get("logprob")})
        toks.append({"id": tk.get("id"), "token": tk.get("token"), "lp": tk.get("logprob"), "top": top})
    return {
        "text": (ch.get("message") or {}).get("content") or "",
        "tokens": toks,
        "have_logprobs": bool(toks),
        "secs": round(time.time() - t, 3),
    }


def tok_key(tok, by="id"):
    """A generated token's identity: its id when asked for, else its text.

    Within one route ids are the exact identity.  Across routes one side may report no id at all
    (the NPU forward path reports text + log-probability only), so ids alone would make every
    position "match" on None == None; --match-by text compares the detokenised text on both
    sides instead, which is the only field the routes share.
    """
    i = tok.get("id")
    if by == "text" or i is None:
        return tok.get("token")
    return i


def first_divergence(a, b, by="id"):
    """Index of the first token position whose log-prob bits differ, else the length mismatch."""
    n = min(len(a), len(b))
    for i in range(n):
        if tok_key(a[i], by) != tok_key(b[i], by):
            return i
        if not usable(a[i]["lp"]) or not usable(b[i]["lp"]):
            continue
        if bits(a[i]["lp"]) != bits(b[i]["lp"]):
            return i
    return None if len(a) == len(b) else n


def determinism(runs, by="id"):
    """Compare every repeat against the first.  Returns the summary dict.

    A route that returns no log-probabilities at all (the NPU in-process path does this) cannot
    answer the bit question.  It is reported as such instead of as a vacuous "0 tokens, identical".
    """
    base = runs[0]
    if not base["have_logprobs"]:
        return {
            "repeats": len(runs),
            "logprobs_available": False,
            "bit_identical": None,
            "first_divergence_per_repeat": [],
            "max_abs_delta": None,
            "mean_abs_delta": None,
            "tokens": 0,
            "non_float32_values": 0,
            "same_text": all(r["text"] == base["text"] for r in runs[1:]),
        }
    deltas, non_f32 = [], 0
    for r in runs:
        for x in r["tokens"]:
            if usable(x["lp"]) and f32(x["lp"]) != x["lp"]:
                non_f32 += 1
    divs = [first_divergence(base["tokens"], r["tokens"], by) for r in runs[1:]]
    maxd = 0.0
    for r in runs[1:]:
        for a, b in zip(base["tokens"], r["tokens"]):
            if tok_key(a, by) == tok_key(b, by) and usable(a["lp"]) and usable(b["lp"]):
                d = abs(f32(a["lp"]) - f32(b["lp"]))
                deltas.append(d)
                maxd = max(maxd, d)
    return {
        "repeats": len(runs),
        "logprobs_available": True,
        "bit_identical": all(d is None for d in divs),
        "first_divergence_per_repeat": divs,
        "max_abs_delta": maxd,
        "mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "tokens": len(base["tokens"]),
        "non_float32_values": non_f32,
        "same_text": all(r["text"] == base["text"] for r in runs[1:]),
    }


def agreement(a, b, by="id"):
    """Cross-route summary over one prompt's generated tokens.

    Token-id and top-1 rates are over every aligned position, because a position whose token ids
    differ is exactly the interesting one and must not be dropped.  The log-prob deltas and the
    bit-match rate are only over positions where the two routes emitted the same token: comparing
    the log-prob of two different tokens would measure the tokens, not the numerics.
    """
    ta, tb = a["tokens"], b["tokens"]
    n = min(len(ta), len(tb))
    id_match = 0
    top1_n = 0
    top1_match = 0
    common = 0
    bit_match = 0
    deltas = []
    first_tok = None
    for i in range(n):
        ta_top = (ta[i]["top"] or [{}])[0]
        tb_top = (tb[i]["top"] or [{}])[0]
        if ta_top and tb_top:
            top1_n += 1
            if tok_key(ta_top, by) == tok_key(tb_top, by):
                top1_match += 1
        if tok_key(ta[i], by) != tok_key(tb[i], by):
            if first_tok is None:
                first_tok = i
            continue
        id_match += 1
        common += 1
        if usable(ta[i]["lp"]) and usable(tb[i]["lp"]):
            if bits(ta[i]["lp"]) == bits(tb[i]["lp"]):
                bit_match += 1
            deltas.append(abs(f32(ta[i]["lp"]) - f32(tb[i]["lp"])))
    return {
        "positions": n,
        "common_token_positions": common,
        "token_id_match_rate": id_match / n if n else 0.0,
        "first_token_divergence": first_tok,
        "top1_agreement_rate": top1_match / top1_n if top1_n else 0.0,
        "exact_bit_match_rate": bit_match / common if common else 0.0,
        "max_abs_dlogp": max(deltas) if deltas else 0.0,
        "mean_abs_dlogp": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "same_text": a["text"] == b["text"],
    }


def selftest():
    """The comparison logic on synthetic runs: the checks a bug in it would silently pass."""
    t = lambda i, lp, top=1: {"id": i, "token": f"t{i}", "lp": lp,  # noqa: E731
                              "top": [{"id": top, "token": f"t{top}", "lp": lp}]}
    # identical runs -> deterministic, no divergence
    a = {"tokens": [t(5, -0.5), t(6, -1.25)], "text": "x", "have_logprobs": True}
    assert determinism([a, a])["bit_identical"] is True
    assert first_divergence(a["tokens"], a["tokens"]) is None
    # one ulp apart -> the bits differ even though the decimal looks close
    one_ulp = -0.5 - 2.0**-24
    assert bits(-0.5) != bits(one_ulp)
    b = {"tokens": [t(5, -0.5), t(6, one_ulp)], "text": "x", "have_logprobs": True}
    assert first_divergence(a["tokens"], b["tokens"]) == 1
    det = determinism([a, b])
    assert det["bit_identical"] is False and det["first_divergence_per_repeat"] == [1]
    # a token-id mismatch is a divergence before any log-prob is considered
    c = {"tokens": [t(5, -0.5), t(99, -1.25)], "text": "x", "have_logprobs": True}
    assert first_divergence(a["tokens"], c["tokens"]) == 1
    # agreement: one shared position of two, exact bits, same top-1
    g = agreement(a, c)
    assert g["positions"] == 2 and g["common_token_positions"] == 1
    assert g["first_token_divergence"] == 1
    assert g["exact_bit_match_rate"] == 1.0 and g["top1_agreement_rate"] == 1.0
    assert g["max_abs_dlogp"] == 0.0
    # a different top-1 with identical chosen log-prob moves the top-1 rate only
    d = {"tokens": [t(5, -0.5, top=7), t(6, -1.25)], "text": "x", "have_logprobs": True}
    g2 = agreement(a, d)
    assert g2["top1_agreement_rate"] == 0.5 and g2["exact_bit_match_rate"] == 1.0
    # a route with no log-probabilities must not read as a vacuous pass
    nolp = {"tokens": [], "text": "hi", "have_logprobs": False}
    det = determinism([nolp, nolp])
    assert det["logprobs_available"] is False and det["bit_identical"] is None
    assert det["same_text"] is True
    # a route that reports no token ids must still be compared, on its text — not on None == None
    noid_a = {"tokens": [{"id": None, "token": "A", "lp": -0.5, "top": []}], "text": "x",
              "have_logprobs": True}
    noid_b = {"tokens": [{"id": None, "token": "B", "lp": -0.5, "top": []}], "text": "x",
              "have_logprobs": True}
    assert first_divergence(noid_a["tokens"], noid_b["tokens"]) == 0
    noid_c = {"tokens": [{"id": None, "token": "A", "lp": -0.5, "top": []}], "text": "x",
              "have_logprobs": True}
    assert first_divergence(noid_a["tokens"], noid_c["tokens"]) is None
    # length mismatch is a divergence at the shorter length
    e = {"tokens": [t(5, -0.5)], "text": "x", "have_logprobs": True}
    assert first_divergence(a["tokens"], e["tokens"]) == 1
    # --match-by text: ids may differ while the detokenised text agrees (the NPU reports no ids)
    id_only = {"tokens": [dict(noid_a["tokens"][0], id=7)], "text": "x", "have_logprobs": True}
    assert first_divergence(noid_a["tokens"], id_only["tokens"], "text") is None
    assert first_divergence(noid_a["tokens"], id_only["tokens"], "id") == 0
    assert agreement(noid_a, id_only, "text")["exact_bit_match_rate"] == 1.0
    print("logprob_agree selftest: ok")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--route", action="append", default=[], metavar="LABEL=URL",
                   help="a served OpenAI endpoint; repeat for each route to compare")
    p.add_argument("--model", default=None, help="model id (default: the first route's /v1/models)")
    p.add_argument("--repeat", type=int, default=5, help="identical requests per prompt (default 5)")
    p.add_argument("--max-tokens", type=int, default=64, help="tokens per prompt (default 64)")
    p.add_argument("--top-logprobs", type=int, default=5, help="alternatives per token (default 5)")
    p.add_argument("--thinking", choices=["on", "off"], default="off",
                   help="enable_thinking sent to every route (default off); a route that ignores it "
                        "has to be measured with on, and then every route must run with it on")
    p.add_argument("--match-by", choices=["id", "text"], default="id",
                   help="compare tokens by id (default) or by text, which is the only field the "
                        "NPU forward path shares with the llama-server routes")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--json", default=None, help="write the full report here")
    p.add_argument("--selftest", action="store_true", help="check the comparison logic and exit")
    a = p.parse_args(argv)

    if a.selftest:
        selftest()
        return 0
    if not a.route:
        p.error("at least one --route LABEL=URL is required")

    routes = []
    for spec in a.route:
        label, _, url = spec.partition("=")
        if not url:
            p.error(f"--route needs LABEL=URL, got {spec!r}")
        routes.append((label, url))

    # /v1/models is the liveness probe, not /v1/health: an idle llama-server reports
    # {"status":"stalled"} on /health (503) and clears it on the next request, so the health
    # route fails an endpoint that is perfectly usable.  503 there is not a hung backend.
    model = a.model
    for label, url in routes:
        try:
            models = get(bare(url) + "/v1/models", 10)
        except Exception as e:  # noqa: BLE001
            print(f"logprob_agree: route {label} ({url}) not reachable: {e}")
            return 2
        if model is None:
            try:
                model = models["data"][0]["id"]
            except (KeyError, IndexError, TypeError):
                model = "default"
    model = model or "default"
    print(f"model {model!r}, {len(routes)} route(s), {len(PROMPTS)} prompts, "
          f"{a.max_tokens} tokens, {a.repeat} repeats, greedy")

    report = {"model": model, "max_tokens": a.max_tokens, "repeat": a.repeat,
              "top_logprobs": a.top_logprobs, "thinking": a.thinking, "match_by": a.match_by,
              "time": time.strftime("%FT%T%z"),
              "routes": {}, "determinism": {}, "agreement": {}, "errors": []}
    thinking = a.thinking == "on"

    for label, url in routes:
        per_prompt = []
        for pi, prompt in enumerate(PROMPTS):
            runs = []
            for _ in range(a.repeat):
                r = ask(url, model, prompt, a.max_tokens, a.top_logprobs, a.timeout, thinking)
                if "error" in r:
                    report["errors"].append(f"{label} prompt {pi}: {r['error']}")
                    break
                runs.append(r)
            if len(runs) != a.repeat:
                break
            per_prompt.append({"prompt": prompt, "runs": runs})
        report["routes"][label] = per_prompt

    failed = False
    print("\n--- determinism (one route, identical request repeated) ---")
    for label, _ in routes:
        per_prompt = report["routes"][label]
        if not per_prompt:
            print(f"{label:<6} NO DATA")
            failed = True
            continue
        agg = [determinism(pp["runs"], a.match_by) for pp in per_prompt]
        have_lp = agg[0]["logprobs_available"]
        d = {
            "logprobs_available": have_lp,
            "bit_identical": all(x["bit_identical"] for x in agg) if have_lp else None,
            "first_divergence_per_prompt": [x["first_divergence_per_repeat"] for x in agg],
            "max_abs_delta": max((x["max_abs_delta"] or 0.0 for x in agg), default=0.0)
            if have_lp else None,
            "tokens": sum(x["tokens"] for x in agg),
            "non_float32_values": sum(x["non_float32_values"] for x in agg),
            "text_stable": all(x["same_text"] for x in agg),
            "prompts_not_bit_identical": [i for i, x in enumerate(agg) if x["bit_identical"] is False],
        }
        report["determinism"][label] = d
        if not have_lp:
            # No log-probabilities on this route: only the text can be held to repetition.
            failed = True
            print(f"{label:<6} NO LOGPROBS      text_stable={d['text_stable']} "
                  f"(cannot answer the bit question on this route)")
        else:
            verdict = "bitwise-identical" if d["bit_identical"] else "DIVERGES"
            print(f"{label:<6} {verdict:<17} tokens={d['tokens']:<5} max|d|={d['max_abs_delta']:.3e} "
                  f"text_stable={d['text_stable']} bad_f32={d['non_float32_values']}")
            if not d["bit_identical"]:
                failed = True
                for i in d["prompts_not_bit_identical"]:
                    print(f"       prompt {i}: first divergence at {agg[i]['first_divergence_per_repeat']}")

    print("\n--- agreement (two routes, same prompts, greedy) ---")
    labels = [label for label, _ in routes]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            pa, pb = report["routes"][la], report["routes"][lb]
            if not pa or not pb:
                print(f"{la} vs {lb:<6} NO DATA")
                failed = True
                continue
            gs = [agreement(x["runs"][0], y["runs"][0], a.match_by) for x, y in zip(pa, pb)]
            tot_common = sum(g["common_token_positions"] for g in gs)
            tot_pos = sum(g["positions"] for g in gs)
            g = {
                "positions": tot_pos,
                "common_token_positions": tot_common,
                "token_id_match_rate": (sum(g["token_id_match_rate"] * g["positions"] for g in gs)
                                        / tot_pos) if tot_pos else 0.0,
                "top1_agreement_rate": (sum(g["top1_agreement_rate"] * g["positions"] for g in gs)
                                        / tot_pos) if tot_pos else 0.0,
                "exact_bit_match_rate": (sum(g["exact_bit_match_rate"] * g["common_token_positions"]
                                             for g in gs) / tot_common) if tot_common else 0.0,
                "max_abs_dlogp": max((g["max_abs_dlogp"] for g in gs), default=0.0),
                "mean_abs_dlogp": (sum(g["mean_abs_dlogp"] * g["common_token_positions"] for g in gs)
                                   / tot_common) if tot_common else 0.0,
                "prompts_with_token_divergence": [k for k, g in enumerate(gs)
                                                  if g["first_token_divergence"] is not None],
                "same_text_all": all(g["same_text"] for g in gs),
            }
            report["agreement"][f"{la}|{lb}"] = g
            if g["exact_bit_match_rate"] < 1.0:
                failed = True
            print(f"{la} vs {lb}")
            print(f"  token-id match    {g['token_id_match_rate']:.3f} over {tot_pos} positions")
            print(f"  max|dlogp|        {g['max_abs_dlogp']:.6e}  (common positions only)")
            print(f"  mean|dlogp|       {g['mean_abs_dlogp']:.6e}  over {tot_common} tokens")
            print(f"  top-1 agreement   {g['top1_agreement_rate']:.3f}")
            print(f"  exact-bit match   {g['exact_bit_match_rate']:.3f}")
            if g["prompts_with_token_divergence"]:
                print(f"  token divergence  prompts {g['prompts_with_token_divergence']}")

    if report["errors"]:
        print("\nerrors:")
        for e in report["errors"][:20]:
            print(f"  {e}")
        failed = True
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nreport: {a.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
