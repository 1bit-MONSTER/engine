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


def _position_distribution(tok, by="id"):
    """One position's next-token distribution from its top-N, renormalised over what it carries.

    Two approximations, both stated rather than hidden: the entries below top-N are not on the
    wire, so their mass is dropped and the rest renormalised; and a log-prob the server sent as a
    non-finite value is skipped rather than guessed.
    """
    top = tok.get("top") or []
    ps = {}
    for t in top:
        lp = t.get("lp")
        if lp is None or not math.isfinite(lp):
            continue
        ps[tok_key({"id": t.get("id"), "token": t.get("token")}, by)] = math.exp(lp)
    z = sum(ps.values())
    return {k: v / z for k, v in ps.items()} if z > 0 else {}


def kld(ref_tokens, cand_tokens, by="id", eps=1e-12):
    """Mean KL(reference || reference) in nats, over positions where both routes emitted the same token,
    plus the share of positions whose top-1 the two routes agree on.

    THE FORMULA, because a KLD whose direction is inferred from prose can be flipped by accident and the
    number then looks plausible forever:

        mean_kld = SUM_k p_ref(k) * ln( p_ref(k) / p_cand(k) )        over the union of the two top-N sets,
                                                                     each renormalised to that union

    i.e. **KL(reference || candidate)**: `kld(cpu, hrx)` is "the mass CPU puts somewhere the hybrid route
    would not" -- the same direction as `tools/logits_kld.py`'s `kl(p=baseline, q=candidate)`, which the
    P4 gate already thresholds at `--max-mean-kld 0.01`. That is not a coincidence to state in a body: two
    instruments feeding one gate must agree on which argument is p, or the thresholds are meaningless. The
    first version of this function computed the opposite direction (KL(candidate || reference), the
    distillation convention), and a review asked for the formula in the source rather than in a comment
    -- an arm below imports the sibling tool and asserts the two agree on the same two distributions.

    Positions where the routes emitted DIFFERENT tokens are excluded rather than scored: a different
    token is a divergence the agreement section already reports, and folding it in would make one number
    answer two questions.
    """
    # Accepts a run dict (`{"tokens": [...]}`, what ask() returns and what the caller has) or a bare token
    # list (what the self-test has). The dict form is not a convenience: the first wiring of this function
    # passed the run dicts straight through and iterated their KEYS, which the fake-route end-to-end test
    # caught as `AttributeError: 'str' object has no attribute 'get'` and no self-test arm could see.
    if isinstance(ref_tokens, dict):
        ref_tokens = ref_tokens.get("tokens") or []
    if isinstance(cand_tokens, dict):
        cand_tokens = cand_tokens.get("tokens") or []
    n = same_top = truncated = 0
    total = 0.0
    for a, b in zip(ref_tokens, cand_tokens):
        if tok_key(a, by) != tok_key(b, by):
            continue
        pa, pb = _position_distribution(a, by), _position_distribution(b, by)
        if not pa or not pb:
            continue
        union = set(pa) | set(pb)
        za = sum(pa.get(k, 0.0) for k in union)
        zb = sum(pb.get(k, 0.0) for k in union)
        if za <= 0 or zb <= 0:
            continue
        if len(union) > len(pa) or len(union) > len(pb):
            truncated += 1
        for k in union:
            p = pa.get(k, 0.0) / za          # reference -- p, as logits_kld.py defines it
            q = pb.get(k, 0.0) / zb          # candidate
            if p > eps:
                total += p * math.log(p / max(q, eps))
        n += 1
        if max(pb, key=pb.get) == max(pa, key=pa.get):
            same_top += 1
    return {"positions": n,
            "mean_kld": (total / n) if n else 0.0,
            "same_top_token_pct": (100.0 * same_top / n) if n else 0.0,
            "positions_with_unseen_entries": truncated}


def kld_summary(per_pair):
    """Aggregate pairwise kld() results into the number a row quotes.

    `mean_kld` and `same_top_token_pct` are **None** when no pair contributed a position -- which is
    exactly the case when the two routes diverged at every position, i.e. the case the accuracy row
    exists to catch. Reporting 0.0 there would be a cannot-tell rendered as a pass: "0.000000 nats"
    is what a reader and a JSON consumer quote as "the routes agree". The reviewer of the first
    version of this function caught that; the helper exists so the rule has an arm instead of living
    in a print statement.
    """
    positions = sum(k["positions"] for k in per_pair)
    unseen = sum(k["positions_with_unseen_entries"] for k in per_pair)
    out = {"positions": positions, "runs_compared": len(per_pair),
           "mean_kld": None, "same_top_token_pct": None,
           "positions_with_unseen_entries": unseen}
    if positions:
        out["mean_kld"] = sum(k["mean_kld"] * k["positions"] for k in per_pair) / positions
        out["same_top_token_pct"] = (sum(k["same_top_token_pct"] * k["positions"] for k in per_pair)
                                     / positions)
    return out


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
    # --- KLD (engine#3158): the accuracy row's number, which no arm above computes ---
    def dist_tok(i, pairs):
        """One position carrying a full top-N: pairs = [(id, logprob), ...]."""
        return {"id": i, "token": f"t{i}", "lp": pairs[0][1],
                "top": [{"id": k, "token": f"t{k}", "lp": lp} for k, lp in pairs]}

    # identical distributions -> exactly zero, and every top-1 agrees
    same = [dist_tok(1, [(1, math.log(0.5)), (2, math.log(0.5))])]
    k = kld(same, same)
    assert k["positions"] == 1 and k["mean_kld"] == 0.0 and k["same_top_token_pct"] == 100.0, k
    # hand-computed, in the formula's direction: ref {A .5, B .5} vs cand {A .75, B .25}
    #   KL(ref || cand) = .5*ln(.5/.75) + .5*ln(.5/.25) = -0.2027326 + 0.3465736 = 0.1438410 nats
    ref = [dist_tok(1, [(1, math.log(0.5)), (2, math.log(0.5))])]
    cand = [dist_tok(1, [(1, math.log(0.75)), (2, math.log(0.25))])]
    k = kld(ref, cand)
    assert abs(k["mean_kld"] - 0.1438410) < 1e-6, k["mean_kld"]
    assert k["same_top_token_pct"] == 100.0         # top-1 is the same token, mass moved under it
    # KL is not symmetric and the reverse direction is asserted SEPARATELY, at its own value, so a flip
    # cannot be absorbed by a tolerance: .75*ln(1.5) + .25*ln(0.5) = 0.1308120
    assert abs(kld(cand, ref)["mean_kld"] - 0.1308120) < 1e-6
    # CROSS-TOOL: the same two distributions through the sibling KLD already in the tree, which the P4
    # gate thresholds at --max-mean-kld. Two instruments feeding one gate must agree on which argument
    # is p; this arm is what makes that a test instead of a claim. (logits_kld.kl takes LOGITS, so the
    # log-probabilities are fed straight in: log_softmax of a normalised log-distribution is itself.)
    import importlib.util as _ilu, pathlib as _pl
    _spec = _ilu.spec_from_file_location("logits_kld", _pl.Path(__file__).with_name("logits_kld.py"))
    _lk = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_lk)
    _ref_logits = [math.log(0.5), math.log(0.5)]
    _cand_logits = [math.log(0.75), math.log(0.25)]
    assert abs(_lk.kl(_ref_logits, _cand_logits) - k["mean_kld"]) < 1e-9, (
        f"cross-tool disagreement: logits_kld says {_lk.kl(_ref_logits, _cand_logits)}, kld says {k['mean_kld']}")
    # a position where the routes emitted DIFFERENT tokens is excluded, not scored
    other = [dist_tok(2, [(2, math.log(0.5)), (1, math.log(0.5))])]
    k = kld(ref, other)
    assert k["positions"] == 0 and k["mean_kld"] == 0.0, k
    # an entry one route carries and the other does not is counted, and renormalisation is what makes
    # the number finite -- the truncation is reported rather than silently absorbed
    wider = [dist_tok(1, [(1, math.log(0.5)), (2, math.log(0.25)), (3, math.log(0.25))])]
    k = kld(ref, wider)
    assert k["positions"] == 1 and k["positions_with_unseen_entries"] == 1, k
    assert k["mean_kld"] > 0.0
    # the ENTRY SHAPE: a run dict, which is what the caller has, must give the same answer as its token list
    assert kld({"tokens": ref}, {"tokens": cand})["mean_kld"] == kld(ref, cand)["mean_kld"] > 0.0
    # a missing or non-finite log-prob is skipped, not turned into a probability
    broken = [{"id": 1, "token": "t1", "lp": None, "top": [{"id": 1, "token": "t1", "lp": None}]}]
    assert kld(broken, broken)["positions"] == 0
    # --- kld_summary: a cannot-tell must never print as 0.0 (the review of #3225) ---
    # the diverged-everywhere case: kld() already returns positions 0 for it, and the summary must
    # carry that through as "no answer" rather than averaging an empty set into a perfect score
    zero = kld(ref, other)                      # different emitted tokens -> positions 0
    sm = kld_summary([zero])
    assert sm["positions"] == 0 and sm["mean_kld"] is None and sm["same_top_token_pct"] is None, sm
    # an empty pair list is the same state, not a different one
    assert kld_summary([])["mean_kld"] is None
    # and a real list aggregates weighted by positions, not by pair
    sm = kld_summary([kld(ref, cand), zero])
    assert sm["positions"] == 1 and abs(sm["mean_kld"] - 0.1438410) < 1e-6, sm
    assert sm["runs_compared"] == 2
    # truncation is carried through to the summary, because the printed line shows it from here
    sm = kld_summary([kld(ref, wider)])
    assert sm["positions_with_unseen_entries"] == 1, sm
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
            # engine#3158: the accuracy row asks for a mean KLD against CPU, which none of the above
            # answers -- they are all comparisons of the tokens the two routes DID emit. `--route` order
            # is the direction: the first route is the reference, so `--route cpu=… --route hrx=…` gives
            # KL(hrx || cpu). Truncated to the top-N both routes carry; see _position_distribution.
            # EVERY run of both routes, paired run-with-run, not `runs[0]`: the determinism section above
            # holds each route to repeating itself, and the row's claim is about the route, so a mean over
            # the repeats is what it implies. Weighted by positions, so a run that produced fewer shared
            # positions does not count as much as one that produced more.
            ks = [kld(rx, ry, a.match_by) for x, y in zip(pa, pb)
                  for rx, ry in zip(x["runs"], y["runs"])]
            k = kld_summary(ks)
            k.update({"reference": la, "candidate": lb})
            g["kld"] = k
            if k["mean_kld"] is None:
                # cannot-tell, and it must read as one: this is the diverged-everywhere case.
                print(f"  mean KLD          cannot tell -- no position had both routes on the same token with a "
                      f"top-N (KL direction {la} || {lb}, {k['runs_compared']} run pair(s))")
                failed = True
            else:
                unseen_note = (f", {k['positions_with_unseen_entries']} with entries below top-N"
                               if k["positions_with_unseen_entries"] else "")
                print(f"  mean KLD          {k['mean_kld']:.6f} nats  (KL({la} || {lb}), "
                      f"{k['positions']} positions{unseen_note})")
                print(f"  same top token    {k['same_top_token_pct']:.1f} %")
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
