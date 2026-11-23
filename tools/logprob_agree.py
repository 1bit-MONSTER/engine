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

Exit code: 0 all routes deterministic and all pairs exactly equal, 1 a route is non-deterministic or a
pair differs -- including a pair that had no position to compare, which prints *cannot tell* rather than
"0.000000 nats", 2 a route is unreachable or a request failed.

A route that carries no log-probs **at all** is a cannot-tell on both the bit question and the KLD, and by
default that is exit 1 like any other unanswered question. `--text-only` says the caller knows: the route's
shape is not a finding, so those two halves read *cannot tell* and no longer fail the run, while a route whose
log-probs ARE present is still held to them and two routes that both have them but diverge at every position
still fail. That is what makes an A/B against a route like loom-serve -- whose chat response emits the literal
`"logprobs":null` (hrx_loom pin 0cd370f2ba, `chat.c:1743`) -- exit 0 on a correct run, which a gate reading
the exit code needs.
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


def parse_route(spec):
    """`LABEL=URL[:minimal|:full]` -> (label, url, body-or-None).

    The two arms of the Loom gate need DIFFERENT bodies for the same meaning, and that is not a shortcut: our serve
    reads thinking only from `chat_template_kwargs.enable_thinking` (app/unified.cpp:222-223, default true) and
    loom-serve takes `enable_thinking` top level and 400s on `chat_template_kwargs` (app/loom_route.h:340-346). So one
    body cannot pin thinking on both -- measured: an empty answer on both arms, 5/5 "identical", 100 % (engine#3158,
    row27b-ud-v8 and row27b-bart). A per-route body lets each arm be told in its own dialect, which is the only shape
    in which the comparison measures the engines rather than the configuration.
    """
    label, _, rest = spec.partition("=")
    if not rest:
        return label, "", None
    body = None
    for suffix in (":minimal", ":full"):
        if rest.endswith(suffix):
            return label, rest[:-len(suffix)], suffix[1:]
    return label, rest, None


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
    """Mean KL(candidate || reference) in nats, over positions where both routes emitted the same token,
    plus the share of positions whose top-1 the two routes agree on.

    THE FORMULA, because a KLD whose direction is inferred from prose can be flipped by accident and the
    number then looks plausible forever:

        mean_kld = SUM_k p_cand(k) * ln( p_cand(k) / p_ref(k) )       over the union of the two top-N sets,
                                                                     each renormalised to that union

    i.e. **KL(candidate || reference)**: `kld(cpu, hrx)` is the mass the hybrid route puts somewhere CPU
    would not -- the direction the gate clause "mean KLD ... vs CPU" names, and the direction
    `tools/logits_kld.py`'s own docstring states ("the candidate's mass that the baseline would not have
    put there") even though its current call site returns the opposite. L2 owns that discrepancy and the
    fix (`~/.briefs` #3158, 18:40Z); this function implements the stated intent, and the arm below
    compares against `logits_kld.kl(cand, base)` so it keeps holding when the call site is corrected.

    This was flipped once already: the first version computed KL(candidate || reference), a review asked
    for the formula in the source rather than in a comment, and the flip to the sibling's *call site*
    direction was reverted when the sibling turned out to disagree with its own docstring. Which is the
    argument for the formula being on this line rather than in a review thread.

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
            p = pb.get(k, 0.0) / zb          # candidate  (see the formula above)
            q = pa.get(k, 0.0) / za          # reference
            if p > eps:
                total += p * math.log(p / max(q, eps))
        n += 1
        if max(pb, key=pb.get) == max(pa, key=pa.get):
            same_top += 1
    return {"positions": n,
            "mean_kld": (total / n) if n else 0.0,
            "same_top_token_pct": (100.0 * same_top / n) if n else 0.0,
            "positions_with_unseen_entries": truncated}


def pair_summary(per_prompt):
    """Aggregate one route pair's per-prompt agreement() dicts into the numbers a row quotes.

    Two rules live here rather than in the print statement, both because a rule written at a print site has
    no arm (the KLD aggregate moved here for the same reason):

      * a rate whose denominator is zero is **None**, never 0.0. `exact-bit match 0.000` and
        `token-id match 0.000 over 0 positions` are the cannot-tell-rendered-as-zero shape this tool has
        already been sent back for twice -- and they are what a route with no log-probs on the wire (Looms'
        chat response returns `logprobs:null` at hrx_loom pin 0cd370f2ba) produces on every prompt;
      * **greedy agreement is over prompts, on generated text**, because it is the only half of the gate's
        accuracy clause that survives a route which emits no log-probs. It is deliberately weaker than the
        token comparison above -- a whitespace or detokenisation difference counts as a disagreement -- and
        it is reported with its own denominator so it cannot be read as a token-level number.
    """
    empty = [k for k, p in enumerate(per_prompt) if p.get("both_empty")]
    comparable = [p for p in per_prompt if not p.get("both_empty")]
    n = len(comparable)
    positions = sum(p["positions"] for p in per_prompt)
    common = sum(p["common_token_positions"] for p in per_prompt)
    top1_n = sum(p["top1_positions"] for p in per_prompt)

    def rate(total, denom):
        return (total / denom) if denom else None

    out = {
        "prompts": len(per_prompt),
        "positions": positions,
        "common_token_positions": common,
        "token_id_match_rate": rate(sum(p["token_id_match_rate"] * p["positions"] for p in per_prompt),
                                    positions),
        "top1_agreement_rate": rate(sum(p["top1_agreement_rate"] * p["top1_positions"] for p in per_prompt),
                                    top1_n),
        "exact_bit_match_rate": rate(sum(p["exact_bit_match_rate"] * p["common_token_positions"]
                                         for p in per_prompt), common),
        "max_abs_dlogp": max((p["max_abs_dlogp"] for p in per_prompt), default=None) if common else None,
        "mean_abs_dlogp": rate(sum(p["mean_abs_dlogp"] * p["common_token_positions"] for p in per_prompt),
                               common),
        "prompts_with_token_divergence": [k for k, p in enumerate(per_prompt)
                                          if p["first_token_divergence"] is not None],
        "same_text_prompts": sum(1 for p in comparable if p["same_text"]),
        "prompts_compared": n,
        "prompts_both_empty": empty,
        "greedy_agreement_pct": (100.0 * sum(1 for p in comparable if p["same_text"]) / n) if n else None,
        "text_prompt_divergence": [k for k, p in enumerate(per_prompt)
                                   if not p.get("both_empty") and not p["same_text"]],
    }
    return out


def no_logprobs_fails(text_only):
    """Whether a route that carries **no log-probs at all** fails the run.

    It cannot answer the bit question, which is cannot-tell by this tool's own contract -- but it is the *expected*
    shape of a route that never puts log-probs on the wire (loom-serve at hrx_loom pin 0cd370f2ba emits the literal
    `"logprobs":null`), and a gate reading this tool's exit code cannot use the text half if a correct run always
    exits non-zero. So the caller declares it with `--text-only`; without the flag the old behaviour stands.
    """
    return not text_only


def missing_kld_fails(text_only, any_no_logprobs):
    """Whether a `mean_kld` of None fails the run.

    Two different states produce that None and only one of them is the caller's to accept: a route that carries no
    log-probs (declared with `--text-only`, and then the KLD was never answerable) versus two routes that both have
    them and diverged at every position (a finding, which fails with or without the flag). The distinction is read
    from the determinism section, not assumed from the flag.
    """
    return not (text_only and any_no_logprobs)


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


def request_body(model, prompt, max_tokens, top_logprobs, thinking=False, minimal=False):
    """The chat request. `minimal` sends only the fields the strictest route accepts.

    A route that validates its body against an OpenAI subset refuses the whole request on any field it does not
    know, so a full body is not a superset a strict server tolerates -- it is a 400 before any comparison happens.
    Measured on this box against loom-serve (the Loom arm of engine#3158's gate): `chat_template_kwargs` and `seed`
    alone were enough to answer `400 invalid model selector or request body`, and `/v1/completions` is refused
    outright, so the arm is chat-only. Hence an explicit switch rather than a silent downgrade: the run has to say
    which body it sent, because `--body minimal` cannot ask for log-probs and its cannot-tell KLD is then
    *unasked*, not *unavailable*.
    """
    if minimal:
        # `enable_thinking` is a TOP-LEVEL field on a strictly-validating route, not a chat_template_kwargs member:
        # loom-serve's serving docs pass "the boolean `enable_thinking`, string `reasoning_effort`, and
        # `model_options` object ... to source chat policy", and its chat response returns `reasoning_content`
        # separately from answer `content` (hrx_loom pin 0cd370f2ba, http-profiles.md:263-270). Dropping thinking
        # control from the minimal body is what made a real row read as a perfect match: the card defaults to
        # `reasoning_effort: "xhigh"`, the whole 16-token budget went to reasoning, `content` came back empty on BOTH
        # arms, and two empty strings compared identical (engine#3158, row27b-ud-v8).
        return {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0, "stream": False,
                "enable_thinking": bool(thinking)}
    return {
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


# --with-reasoning joins a reply's reasoning and its answer with this line, so the two halves cannot run together
REASONING_SEP = "\n<<answer>>\n"


def reply_reasoning(message):
    """The reply's reasoning, from either field name a server uses (one helper, so the JSON records what was compared)."""
    # whitespace at the thinking's edges is the server's parse, not the model's tokens: loom-serve keeps the newline before
    # the closing tag, llama.cpp's reasoning parser strips it (engine#3158, Qwen3.5-4B 23 Nov: 3 of 8 identical replies
    # differed by that one character and scored as divergences)
    return (message.get("reasoning_content") or message.get("reasoning") or "").strip()


def reasoning_one_sided(routes):
    """--with-reasoning: the prompts where some routes carry reasoning and others carry none.

    Such a pair cannot be scored by joined text. If one server parses the thinking out (`reasoning_content`) and the
    other leaves it inline in `content`, the joined texts differ at character 0, a split that never happened. If one
    model simply did not think, the turns differ (docs/loom-route.md: that voids a thinking-on row). Either way the
    run reports it as an error instead of a score (cl-nvfp4 review, #3386).
    """
    out = []
    labels = sorted(routes)
    n = min((len(routes[l]) for l in labels), default=0)
    for k in range(n):
        has = {l: any(r.get("reasoning") for r in routes[l][k]["runs"]) for l in labels}
        if len(set(has.values())) > 1:
            out.append("prompt %d: reasoning from %s only (none from %s): not scored by joined text"
                       % (k, ", ".join(l for l in labels if has[l]), ", ".join(l for l in labels if not has[l])))
    return out


def reply_text(message, with_reasoning=False):
    """The text a run is compared on: the answer `content`, or with --with-reasoning the reasoning, then the answer.

    engine#3158, Inkling-2-Mini thinking on: both arms answered every prompt correctly but in different words
    ("2, 3, 5, 7, 11" vs "2,3,5,7,11"), because they split somewhere INSIDE the thinking. The answer alone says that
    they diverged, not where, so whether the split was a near-tie could not be told. Both loom-serve and llama-server
    return the thinking as `reasoning_content` beside `content`.
    """
    content = message.get("content") or ""
    if not with_reasoning:
        return content
    return reply_reasoning(message) + REASONING_SEP + content


def ask(url, model, prompt, max_tokens, top_logprobs, timeout, thinking=False, minimal=False, with_reasoning=False):
    """One greedy request; returns the generated token ids with their log-probs."""
    body = request_body(model, prompt, max_tokens, top_logprobs, thinking, minimal)
    t = time.time()
    try:
        r = post(bare(url) + "/v1/chat/completions", body, timeout)
    except urllib.error.HTTPError as e:
        detail = e.read()[:300].decode(errors="replace")
        if e.code == 400 and not minimal:
            # The useful half of a strict-route 400 is the remedy, not the status.
            return {"error": f"HTTP 400: {detail} -- a route that validates strictly refuses unknown fields; "
                             f"re-run with --body minimal (it cannot ask for log-probs, so the KLD half reads "
                             f"cannot tell because it was unasked)"}
        return {"error": f"HTTP {e.code}: {detail}"}
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
        "text": reply_text(ch.get("message") or {}, with_reasoning),
        "reasoning": reply_reasoning(ch.get("message") or {}),
        "tokens": toks,
        "have_logprobs": bool(toks),
        "finish": ch.get("finish_reason"),
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


def route_has_token_ids(per_prompt):
    """Whether this route reports an `id` on its generated tokens at all.

    Measured on the two arms of engine#3158's gate: llama.cpp's chat logprobs carry `id`, and our own serve does not --
    `app/forward_serve.cpp`'s `logprob_json` emits `token`, `logprob` and `top_logprobs` and no id, on every backend.
    """
    for pp in per_prompt:
        for run in pp["runs"]:
            for tok in run["tokens"]:
                if tok.get("id") is None:
                    return False
    return bool(per_prompt)


def effective_match_by(requested, a_has_ids, b_has_ids):
    """The key one pair can actually be compared on.

    `--match-by id` against a pair where only one side carries ids compares an integer with a string at every
    position, so every position "differs" -- measured: `token-id match 0.000`, `token divergence` on all prompts and
    `mean KLD cannot tell`, i.e. a complete-disagreement report for two routes that are token-for-token identical
    under text. That is a wrong number in the most expensive direction (it reads as an accuracy failure and sends a
    model back to the fallback), so the pair degrades to the one key both sides have, and says so.
    """
    return "text" if requested == "id" and not (a_has_ids and b_has_ids) else requested


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
        "top1_positions": top1_n,
        "top1_agreement_rate": top1_match / top1_n if top1_n else 0.0,
        "exact_bit_match_rate": bit_match / common if common else 0.0,
        "max_abs_dlogp": max(deltas) if deltas else 0.0,
        "mean_abs_dlogp": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "same_text": a["text"] == b["text"],
        # engine#3158: two empty answers are identical strings and compare equal, which is not a measurement. With
        # thinking on and a small --max-tokens a route can spend the whole budget on reasoning and return no content
        # at all -- measured against two routes that answer exactly that way: `greedy agreement 100.0 %`, rc 0.
        "both_empty": a["text"] == "" and b["text"] == "",
    }


def split_margin(text, ref_run):
    """engine#3158 near-tie rule (orchestrator 23 Nov ~20:2xZ, tightened ~21:1xZ after the #3420 reviews): where `text`
    (the route's reply) first leaves `ref_run`'s (the CPU arbiter's), the reference's log-prob gap from its own top token
    to the token the ROUTE emitted there. A small top-2 margin alone is not enough: a drifting kernel that picks the
    CPU's Nth choice at a small-margin split must still count.

    The route returns text only, so its token at the split is read from the reference's candidates: the longest of the
    reference's top-N tokens that the route's text continues with from that token's start. When one text is a prefix of
    the other the split is at the end of the shorter one; if the reference ended first, its token there is the
    end-of-turn token (an empty string) and the route's is a candidate that matches its continuation.

    Returns (char index, gap or None, reference token, route token or None, top-2 margin or None). gap is None when the
    route's token is not among the reference's top-N (the split then counts)."""
    rt = ref_run.get("text") or ""
    i = next((j for j in range(min(len(text), len(rt))) if text[j] != rt[j]), min(len(text), len(rt)))
    acc = 0
    for t in ref_run.get("tokens") or []:
        s = t.get("token") or ""
        if acc + len(s) > i or (s == "" and acc == i):
            cands = [x for x in (t.get("top") or []) if usable(x.get("lp"))]
            lps = sorted((f32(x["lp"]) for x in cands), reverse=True)
            margin = (lps[0] - lps[1]) if len(lps) > 1 else None
            rest = text[acc:]
            if rest == "":   # the route ended here: its token is the end-of-turn token
                match = [x for x in cands if (x.get("token") or "") == ""]
            else:
                match = [x for x in cands if (x.get("token") or "") and rest.startswith(x["token"])]
            match.sort(key=lambda x: -len(x.get("token") or ""))
            if not match or not lps:
                return i, None, t.get("token"), None, margin
            return i, lps[0] - f32(match[0]["lp"]), t.get("token"), match[0].get("token"), margin
        acc += len(s)
    return i, None, None, None, None


def near_tie_score(cand_runs, ref_runs, nats=1.0):
    """Over prompts: identical text, near-tie splits (the route's token within `nats` of the reference's top token),
    truncated prompts and counted splits (further, or the route's token not among the reference's top-N).
    Score = identical + near-tie + truncated, out of the prompts both routes answered.

    Truncated: one text is a prefix of the other and the shorter run stopped on its token budget (finish "length").
    Neither route chose a different token; one ran out of tokens. Counting it would charge the budget to the model:
    with thinking on, Inkling's replies routinely end inside the thinking at max_tokens (engine#3423 review)."""
    out = {"threshold_nats": nats, "identical": 0, "near_tie": 0, "truncated": 0, "counted": 0, "unreadable": 0,
           "splits": []}
    n = 0
    for p, (c, r) in enumerate(zip(cand_runs, ref_runs)):
        c, r = c["runs"][0], r["runs"][0]
        if not c.get("text") and not r.get("text"):
            continue
        n += 1
        if c["text"] == r["text"]:
            out["identical"] += 1
            continue
        short = c if len(c["text"]) < len(r["text"]) else r
        other = r if short is c else c
        if other["text"].startswith(short["text"]) and short.get("finish") == "length":
            out["truncated"] += 1
            out["splits"].append({"prompt": p, "char": len(short["text"]), "margin": None, "top2_margin": None,
                                  "ref_token": None, "route_token": None, "kind": "truncated",
                                  "cut": "route" if short is c else "reference"})
            continue
        i, gap, tok, rtok, margin = split_margin(c["text"], r)
        kind = "counted" if (gap is None or gap > nats) else "near_tie"
        out[kind] += 1
        if gap is None:
            out["unreadable"] += 1
        out["splits"].append({"prompt": p, "char": i, "margin": gap, "top2_margin": margin, "ref_token": tok,
                              "route_token": rtok, "kind": kind})
    out["prompts"] = n
    out["score"] = out["identical"] + out["near_tie"] + out["truncated"]
    return out


def selftest():
    """The comparison logic on synthetic runs: the checks a bug in it would silently pass.

    The KLD arms below are named and counted, because an auditor reading `selftest: ok` cannot see which
    properties are covered -- the sibling tools print `7 ok, 0 failed` and `10 ok, 0 failed`, and a gate
    is only as legible as its instruments (raised on #3225).
    """
    named = [0, 0]

    def ok(cond, name):
        named[0 if cond else 1] += 1
        print(f"{'ok  ' if cond else 'FAIL'} {name}")
        assert cond, name
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
    ok(k["positions"] == 1 and k["mean_kld"] == 0.0 and k["same_top_token_pct"] == 100.0,
       "identical distributions -> exactly 0.0 nats and 100 % same-top")
    # hand-computed, in the formula's direction: ref {A .5, B .5} vs cand {A .75, B .25}
    #   KL(cand || ref) = .75*ln(.75/.5) + .25*ln(.25/.5) = 0.1013663 - 0.1732868 -> 0.1308120 nats
    # (the reverse, KL(ref || cand), is .5*ln(.5/.75) + .5*ln(.5/.25) = 0.1438410 -- the arm below)
    ref = [dist_tok(1, [(1, math.log(0.5)), (2, math.log(0.5))])]
    cand = [dist_tok(1, [(1, math.log(0.75)), (2, math.log(0.25))])]
    k = kld(ref, cand)
    ok(abs(k["mean_kld"] - 0.1308120) < 1e-6,
       f"a hand-computed KL(cand||ref) is 0.1308120 nats (got {k['mean_kld']:.7f})")
    ok(k["same_top_token_pct"] == 100.0, "top-1 unchanged while the mass moves under it -> same-top 100 %")
    # KL is not symmetric and the reverse direction is asserted SEPARATELY, at its own value, so a flip
    # cannot be absorbed by a tolerance: .5*ln(.5/.75) + .5*ln(.5/.25) = 0.1438410
    ok(abs(kld(cand, ref)["mean_kld"] - 0.1438410) < 1e-6,
       f"…and the reverse direction is its own number, 0.1438410 (got {kld(cand, ref)['mean_kld']:.7f})")
    # the docstring IS the direction spec, so it is pinned like one: this PR has already had it read
    # `KL(reference || reference)` (zero by definition) after a flip, and once been `None` because code
    # was inserted above the prose. A stale first line is exactly what a reviewer skims and trusts.
    ok(bool(kld.__doc__) and "KL(candidate || reference)" in kld.__doc__.splitlines()[0],
       "kld's docstring first line names the direction the arithmetic computes")
    # CROSS-TOOL: the same two distributions through the sibling KLD already in the tree, which the P4
    # gate thresholds at --max-mean-kld. Two instruments feeding one gate must agree on which argument
    # is p; this arm is what makes that a test instead of a claim. (logits_kld.kl takes LOGITS, so the
    # log-probabilities are fed straight in: log_softmax of a normalised log-distribution is itself.)
    import importlib.util as _ilu, pathlib as _pl
    _spec = _ilu.spec_from_file_location("logits_kld", _pl.Path(__file__).with_name("logits_kld.py"))
    _lk = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_lk)
    _ref_logits = [math.log(0.5), math.log(0.5)]
    _cand_logits = [math.log(0.75), math.log(0.25)]
    # The sibling's CALL SITE is kl(base, cand) and returns KL(base||cand); its DOCSTRING describes
    # KL(cand||base) -- "the candidate's mass that the baseline would not have put there" -- which is the
    # direction the gate clause names and the one this function implements. L2 owns fixing the call site,
    # so this compares against kl(cand, base): correct today, and still correct after that fix, because
    # kl(p, q) is KL(p||q) either way.
    # …and the sibling IS direction-sensitive, so "they agree" cannot be an accident of a symmetric fixture
    ok(abs(_lk.kl(_ref_logits, _cand_logits) - 0.1438410362258905) < 1e-9,
       "cross-tool: swapping the pair changes logits_kld's number (0.1438410, its other direction)")
    _sibling = _lk.kl(_cand_logits, _ref_logits)
    ok(abs(_sibling - k["mean_kld"]) < 1e-9,
       f"cross-tool: logits_kld.kl(cand, base) agrees to 1e-9 ({_sibling:.7f} vs {k['mean_kld']:.7f})")
    # a position where the routes emitted DIFFERENT tokens is excluded, not scored
    other = [dist_tok(2, [(2, math.log(0.5)), (1, math.log(0.5))])]
    k = kld(ref, other)
    ok(k["positions"] == 0 and k["mean_kld"] == 0.0,
       "a position with a different emitted token is excluded, not scored")
    # an entry one route carries and the other does not is counted, and renormalisation is what makes
    # the number finite -- the truncation is reported rather than silently absorbed
    wider = [dist_tok(1, [(1, math.log(0.5)), (2, math.log(0.25)), (3, math.log(0.25))])]
    k = kld(ref, wider)
    ok(k["positions"] == 1 and k["positions_with_unseen_entries"] == 1 and k["mean_kld"] > 0.0,
       "an entry only one route carries is counted, and renormalisation keeps the number finite")
    # the ENTRY SHAPE: a run dict, which is what the caller has, must give the same answer as its token list
    ok(kld({"tokens": ref}, {"tokens": cand})["mean_kld"] == kld(ref, cand)["mean_kld"] > 0.0,
       "a run dict and its token list give the same answer")
    # a missing or non-finite log-prob is skipped, not turned into a probability
    broken = [{"id": 1, "token": "t1", "lp": None, "top": [{"id": 1, "token": "t1", "lp": None}]}]
    ok(kld(broken, broken)["positions"] == 0, "a non-finite log-prob is skipped, not a probability")
    # --- kld_summary: a cannot-tell must never print as 0.0 (the review of #3225) ---
    # the diverged-everywhere case: kld() already returns positions 0 for it, and the summary must
    # carry that through as "no answer" rather than averaging an empty set into a perfect score
    zero = kld(ref, other)                      # different emitted tokens -> positions 0
    sm = kld_summary([zero])
    ok(sm["positions"] == 0 and sm["mean_kld"] is None and sm["same_top_token_pct"] is None,
       "kld_summary: no positions -> None, never 0.0")
    # an empty pair list is the same state, not a different one
    ok(kld_summary([])["mean_kld"] is None, "kld_summary: an empty pair list is the same cannot-tell")
    # and a real list aggregates weighted by positions, not by pair
    sm = kld_summary([kld(ref, cand), zero])
    ok(sm["positions"] == 1 and abs(sm["mean_kld"] - 0.1308120) < 1e-6 and sm["runs_compared"] == 2,
       "kld_summary: weighted by positions, and runs_compared counts the pairs")
    # truncation is carried through to the summary, because the printed line shows it from here
    sm = kld_summary([kld(ref, wider)])
    ok(sm["positions_with_unseen_entries"] == 1, "kld_summary: truncation is carried through")
    # pair_summary: the same rule for the agreement rates, and the one half that survives a route with no
    # log-probs. Both are live shapes: Loom's chat response is `logprobs:null` for every token, so a Loom arm
    # produces zero positions on every prompt while still producing text.
    no_lp = {"positions": 0, "common_token_positions": 0, "top1_positions": 0, "token_id_match_rate": 0.0,
             "top1_agreement_rate": 0.0, "exact_bit_match_rate": 0.0, "max_abs_dlogp": 0.0,
             "mean_abs_dlogp": 0.0, "first_token_divergence": 0, "same_text": True}
    ps = pair_summary([no_lp, dict(no_lp, same_text=False)])
    ok(ps["token_id_match_rate"] is None and ps["exact_bit_match_rate"] is None
       and ps["max_abs_dlogp"] is None and ps["mean_abs_dlogp"] is None,
       "pair_summary: a rate with no denominator is None, never 0.000")
    ok(ps["greedy_agreement_pct"] == 50.0 and ps["text_prompt_divergence"] == [1],
       "pair_summary: greedy agreement is over prompts, so it answers where the rates cannot (50 %)")
    had = {"positions": 2, "common_token_positions": 1, "top1_positions": 2, "token_id_match_rate": 0.5,
           "top1_agreement_rate": 0.5, "exact_bit_match_rate": 1.0, "max_abs_dlogp": 0.25,
           "mean_abs_dlogp": 0.25, "first_token_divergence": 1, "same_text": True}
    ps = pair_summary([had, had])
    ok(ps["token_id_match_rate"] == 0.5 and ps["exact_bit_match_rate"] == 1.0
       and ps["max_abs_dlogp"] == 0.25 and ps["greedy_agreement_pct"] == 100.0,
       "pair_summary: a real rate is unchanged, and identical text is 100 % greedy agreement")
    # --body minimal: measured against loom-serve, which 400s on `chat_template_kwargs` or `seed` alone and is
    # chat-only. The switch must drop every field it does not know AND say so, because its KLD is unasked.
    full = request_body("m", "p", 8, 5, thinking=True, minimal=False)
    ok({"logprobs", "top_logprobs", "seed", "top_k", "cache_prompt", "chat_template_kwargs"} <= set(full),
       "request_body: the full body is the OpenAI shape (log-probs and sampling pinning included)")
    bare = request_body("m", "p", 8, 5, thinking=True, minimal=True)
    ok(set(bare) == {"model", "messages", "max_tokens", "temperature", "stream", "enable_thinking"},
       "request_body: the minimal body is the strict route's documented field set")
    ok(bare["enable_thinking"] is True and request_body("m", "p", 8, 5, thinking=False, minimal=True)
       ["enable_thinking"] is False,
       "request_body: --thinking is carried TOP-LEVEL in the minimal body, because dropping it is how the Loom "
       "arm answered with empty content and scored 100 %")
    # --text-only: the exit contract for a route that carries no log-probs. A Loom A/B is exactly that route, and a
    # correct run of it used to exit 1 -- so a gate reading the exit code could not use the text half at all.
    ok(no_logprobs_fails(False) and not no_logprobs_fails(True),
       "no_logprobs_fails: fatal by default, declared-and-accepted with --text-only")
    ok(not missing_kld_fails(True, True) and missing_kld_fails(True, False) and missing_kld_fails(False, True),
       "missing_kld_fails: --text-only excuses a KLD that had no log-probs to compute, never a diverged-everywhere one")
    # effective_match_by: the mixed-ids trap. Measured on the gate's own two arms -- llama.cpp emits `id`, our serve
    # does not -- where `--match-by id` reported 0.000 token-id match and "diverged on all 8 prompts" for two routes
    # that are identical under text.
    ok(effective_match_by("id", True, True) == "id" and effective_match_by("text", True, True) == "text",
       "effective_match_by: an id-vs-id pair keeps the id key, and an explicit text request is untouched")
    # reply_text: --with-reasoning puts the thinking first, so a split inside it is a text difference; without the
    # flag only the answer counts (the default every earlier row was measured with)
    m = {"content": "2,3,5", "reasoning_content": "primes: 2 then 3"}
    ok(reply_text(m) == "2,3,5" and reply_text(m, True) == "primes: 2 then 3" + REASONING_SEP + "2,3,5"
       and reply_text({"content": "x"}, True) == REASONING_SEP + "x",
       "reply_text: answer alone by default; reasoning + separator + answer with --with-reasoning")
    ok(reply_text({"content": "same", "reasoning_content": "a"}, True)
       != reply_text({"content": "same", "reasoning_content": "b"}, True),
       "reply_text: two equal answers after different reasoning differ under --with-reasoning")
    # reasoning_one_sided: one route parsed the thinking out, the other returned none -> an error, never a score
    run = lambda r: {"runs": [{"reasoning": r}]}
    ok(reasoning_one_sided({"loom": [run("t"), run("")], "cpu": [run("t"), run("t")]})
       == ["prompt 1: reasoning from cpu only (none from loom): not scored by joined text"]
       and reasoning_one_sided({"loom": [run("a")], "cpu": [run("b")]}) == []
       and reasoning_one_sided({"loom": [run("")], "cpu": [run("")]}) == [],
       "reasoning_one_sided: flags only the prompts where the routes disagree on carrying reasoning")
    ok(reply_text({"content": "x", "reasoning_content": "think.\n"}, True) == reply_text({"content": "x", "reasoning_content": "think."}, True),
       "reply_text: a trailing newline inside the reasoning (loom-serve keeps it, llama.cpp strips it) is not a divergence")
    ok(reply_reasoning({"reasoning": "r"}) == "r" and reply_reasoning({"reasoning_content": "c", "reasoning": "r"}) == "c",
       "reply_reasoning: reasoning_content first, then reasoning (the same field the JSON records)")
    # near-tie rule (engine#3158, 23 Nov ~21:1xZ): the gap from the reference's top token to the ROUTE's token decides
    cand = lambda *pairs: [{"id": k, "token": tt, "lp": lp} for k, (tt, lp) in enumerate(pairs)]
    tok3 = lambda s, top: {"id": 0, "token": s, "lp": top[0]["lp"], "top": top}
    ref = {"text": "ab", "tokens": [tok3("a", cand(("a", -0.1), ("q", -5.0))),
                                    tok3("b", cand(("b", -0.3), ("X", -0.9), ("Z", -3.5))),
                                    tok3("", cand(("", -0.2), (".", -1.0)))]}
    i, gap, rt, wt, m2 = split_margin("aX", ref)
    ok(i == 1 and wt == "X" and abs(gap - 0.6) < 1e-6 and abs(m2 - 0.6) < 1e-6,
       "split_margin: the route took the reference's 2nd choice, 0.6 nat below its top: a near-tie")
    i, gap, rt, wt, m2 = split_margin("aZ", ref)
    ok(wt == "Z" and abs(gap - 3.2) < 1e-6 and abs(m2 - 0.6) < 1e-6,
       "split_margin: a 3rd choice 3.2 nat down counts, though the top-2 margin there is only 0.6 (the drifting-kernel case)")
    i, gap, rt, wt, m2 = split_margin("aW", ref)
    ok(gap is None and wt is None, "split_margin: a route token outside the reference's top-N has no gap (it counts)")
    i, gap, rt, wt, m2 = split_margin("ab.", ref)
    ok(i == 2 and rt == "" and wt == "." and abs(gap - 0.8) < 1e-6,
       "split_margin: the reference ended first: its end-of-turn token vs the route's continuation token (0.8 nat)")
    runs = lambda *texts_refs: [{"runs": [x]} for x in texts_refs]
    sc = near_tie_score(runs({"text": "aX"}, {"text": "ab"}, {"text": "aZ"}, {"text": "aW"}), runs(ref, ref, ref, ref), 1.0)
    ok(sc["identical"] == 1 and sc["near_tie"] == 1 and sc["counted"] == 2 and sc["score"] == 2 and sc["prompts"] == 4,
       "near_tie_score: identical + near-tie; a far choice and an out-of-top-N token are counted")
    # a reply cut by max_tokens (finish "length") that agrees up to its cut is truncated, not a disagreement, either way
    cut_ref = dict(ref, text="a", finish="length")
    sc = near_tie_score(runs({"text": "ab", "finish": "stop"}, {"text": "a", "finish": "length"}, {"text": "aX", "finish": "length"}),
                        runs(cut_ref, ref, ref), 1.0)
    ok(sc["truncated"] == 2 and sc["counted"] == 0 and sc["near_tie"] == 1 and sc["score"] == 3
       and [x.get("cut") for x in sc["splits"] if x["kind"] == "truncated"] == ["reference", "route"],
       "near_tie_score: a reference or a route cut by its token budget at an agreeing prefix is truncated, not counted")
    sc = near_tie_score(runs({"text": "a", "finish": "stop"}), runs(ref), 1.0)
    ok(sc["truncated"] == 0 and sc["near_tie"] + sc["counted"] == 1,
       "near_tie_score: a route that STOPPED early (finish stop) is still judged at the reference's token, not excused")
    ok(effective_match_by("id", True, False) == "text" and effective_match_by("id", False, False) == "text",
       "effective_match_by: one route without ids degrades the pair to text instead of reporting total divergence")
    # both routes answering with NO content is not agreement: with thinking on and a small --max-tokens a route can
    # spend the budget on reasoning and return "". Two of those are identical strings, and an earlier version scored
    # them 100 % with rc 0 -- the rubber stamp this arm exists to stop.
    blank = {"positions": 0, "common_token_positions": 0, "top1_positions": 0, "token_id_match_rate": 0.0,
             "top1_agreement_rate": 0.0, "exact_bit_match_rate": 0.0, "max_abs_dlogp": 0.0,
             "mean_abs_dlogp": 0.0, "first_token_divergence": None, "same_text": True, "both_empty": True}
    ok(pair_summary([blank, blank])["greedy_agreement_pct"] is None,
       "pair_summary: two routes that both answered with no content are cannot-tell, not 100 %")
    ok(pair_summary([blank, dict(blank, both_empty=False, same_text=True)])["greedy_agreement_pct"] == 100.0
       and pair_summary([blank, dict(blank, both_empty=False, same_text=True)])["prompts_compared"] == 1,
       "pair_summary: an empty-on-both-sides prompt leaves the denominator instead of counting as agreement")
    # per-route body: the two arms of the Loom gate need different spellings of the same setting
    ok(parse_route("loom=http://127.0.0.1:8080:minimal")[:2] == ("loom", "http://127.0.0.1:8080")
       and parse_route("loom=http://127.0.0.1:8080:minimal")[2] == "minimal",
       "parse_route: a :minimal suffix sets the body WITHOUT eating the port or the path")
    ok(parse_route("cpu=http://h:1/v1/x")[2] is None and parse_route("cpu=http://h:1/v1/x")[1] == "http://h:1/v1/x",
       "parse_route: no suffix leaves the url alone, so a plain --route is unchanged")
    ok(parse_route("a=u:full")[2] == "full" and parse_route("a=u:minimal")[2] == "minimal",
       "parse_route: both bodies can be named per route")
    ok(not route_has_token_ids([{"runs": [{"tokens": [{"id": None, "token": "a"}]}]}])
       and route_has_token_ids([{"runs": [{"tokens": [{"id": 7, "token": "a"}]}]}]),
       "route_has_token_ids: our serve's id-less entries are detected, a fork-style entry is not flagged")
    print(f"logprob_agree selftest: {named[0]} ok, {named[1]} failed (KLD arms named; the rest assert)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--route", action="append", default=[], metavar="LABEL=URL",
                   help="a served OpenAI endpoint; repeat for each route to compare")
    p.add_argument("--model", default=None, help="model id (default: the first route's /v1/models)")
    p.add_argument("--repeat", type=int, default=5, help="identical requests per prompt (default 5)")
    p.add_argument("--max-tokens", type=int, default=64, help="tokens per prompt (default 64)")
    p.add_argument("--top-logprobs", type=int, default=5, help="alternatives per token (default 5)")
    p.add_argument("--text-only", action="store_true",
                   help="accept a route that carries no log-probs: the bit question and the KLD are then reported as "
                        "cannot tell and do not fail the run (a route whose log-probs are present is still held to "
                        "them, and diverged-everywhere still fails)")
    p.add_argument("--body", choices=["full", "minimal"], default="full",
                   help="chat body to send: `minimal` is the four fields a strict route accepts (no log-probs, so "
                        "the KLD half reads cannot tell because it was unasked); default full")
    p.add_argument("--thinking", choices=["on", "off"], default="off",
                   help="enable_thinking sent to every route (default off); a route that ignores it "
                        "has to be measured with on, and then every route must run with it on")
    p.add_argument("--match-by", choices=["id", "text"], default="id",
                   help="compare tokens by id (default) or by text, which is the only field the "
                        "NPU forward path shares with the llama-server routes")
    p.add_argument("--with-reasoning", action="store_true",
                   help="compare each reply's reasoning_content followed by its answer, not the answer alone: with "
                        "thinking on, two routes can give the same answer in different words after splitting inside "
                        "the thinking, and only the joined text shows where. It changes the TEXT comparison only: "
                        "two routes that both return log-probs under --match-by id are still decided by their id "
                        "streams. A prompt where some routes return reasoning and others none is an error, not a score")
    p.add_argument("--arbiter", default="cpu",
                   help="route whose log-probs score the near-tie rule (engine#3158): default cpu, when such a route is given")
    p.add_argument("--near-tie-nats", type=float, default=1.0,
                   help="a split counts as a disagreement when the arbiter's log-prob gap from its top token to the route's token there exceeds this, or the route's token is not in its top-N")
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
        label, url, mode = parse_route(spec)
        if not url:
            p.error(f"--route needs LABEL=URL[:minimal|:full], got {spec!r}")
        routes.append((label, url, mode))

    # /v1/models is the liveness probe, not /v1/health: an idle llama-server reports
    # {"status":"stalled"} on /health (503) and clears it on the next request, so the health
    # route fails an endpoint that is perfectly usable.  503 there is not a hung backend.
    model = a.model
    for label, url, _ in routes:
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
          f"{a.max_tokens} tokens, {a.repeat} repeats, greedy, body={a.body}")

    report = {"model": model, "max_tokens": a.max_tokens, "repeat": a.repeat,
              "top_logprobs": a.top_logprobs, "thinking": a.thinking, "match_by": a.match_by,
              "with_reasoning": a.with_reasoning,
              "body": a.body, "logprobs_requested": a.body == "full", "text_only": a.text_only,
              "routes_body": {},
              "time": time.strftime("%FT%T%z"),
              "routes": {}, "determinism": {}, "agreement": {}, "errors": []}
    thinking = a.thinking == "on"

    for label, url, mode in routes:
        minimal = (mode or a.body) == "minimal"
        report["routes_body"][label] = "minimal" if minimal else "full"
        per_prompt = []
        for pi, prompt in enumerate(PROMPTS):
            runs = []
            for _ in range(a.repeat):
                r = ask(url, model, prompt, a.max_tokens, a.top_logprobs, a.timeout, thinking, minimal, a.with_reasoning)
                if "error" in r:
                    report["errors"].append(f"{label} prompt {pi}: {r['error']}")
                    break
                runs.append(r)
            if len(runs) != a.repeat:
                break
            per_prompt.append({"prompt": prompt, "runs": runs})
        report["routes"][label] = per_prompt
    if a.with_reasoning:
        report["errors"].extend(reasoning_one_sided(report["routes"]))

    failed = False
    print("\n--- determinism (one route, identical request repeated) ---")
    for label, _, _ in routes:
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
            # No log-probabilities on this route: only the text can be held to repetition. cannot-tell -- fatal
            # unless the caller declared the route with --text-only, because then it is the route's shape and not
            # an unanswered question.
            if no_logprobs_fails(a.text_only):
                failed = True
            print(f"{label:<6} NO LOGPROBS      text_stable={d['text_stable']} "
                  f"(cannot answer the bit question on this route"
                  f"{'' if no_logprobs_fails(a.text_only) else '; accepted with --text-only'})")
        else:
            verdict = "bitwise-identical" if d["bit_identical"] else "DIVERGES"
            print(f"{label:<6} {verdict:<17} tokens={d['tokens']:<5} max|d|={d['max_abs_delta']:.3e} "
                  f"text_stable={d['text_stable']} bad_f32={d['non_float32_values']}")
            if not d["bit_identical"]:
                failed = True
                for i in d["prompts_not_bit_identical"]:
                    print(f"       prompt {i}: first divergence at {agg[i]['first_divergence_per_repeat']}")

    print("\n--- agreement (two routes, same prompts, greedy) ---")
    labels = [label for label, _, _ in routes]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            pa, pb = report["routes"][la], report["routes"][lb]
            if not pa or not pb:
                print(f"{la} vs {lb:<6} NO DATA")
                failed = True
                continue
            mb = effective_match_by(a.match_by, route_has_token_ids(pa), route_has_token_ids(pb))
            gs = [agreement(x["runs"][0], y["runs"][0], mb) for x, y in zip(pa, pb)]
            g = pair_summary(gs)
            g["match_by_effective"] = mb
            report["agreement"][f"{la}|{lb}"] = g
            if (g["exact_bit_match_rate"] is not None and g["exact_bit_match_rate"] < 1.0) \
                    or g["text_prompt_divergence"] or g["greedy_agreement_pct"] is None:
                failed = True
            fmt = lambda v, spec: "cannot tell" if v is None else format(v, spec)  # noqa: E731
            print(f"{la} vs {lb}")
            if mb != a.match_by:
                lacking = [n for n, p in ((la, pa), (lb, pb)) if not route_has_token_ids(p)]
                print(f"  match key         {mb}, not {a.match_by}: {', '.join(lacking)} reports no token ids, and an "
                      f"id-vs-text comparison would report every position as differing")
            print(f"  token-id match    {fmt(g['token_id_match_rate'], '.3f')} over {g['positions']} positions")
            print(f"  max|dlogp|        {fmt(g['max_abs_dlogp'], '.6e')}  (common positions only)")
            print(f"  mean|dlogp|       {fmt(g['mean_abs_dlogp'], '.6e')}  over {g['common_token_positions']} tokens")
            print(f"  top-1 agreement   {fmt(g['top1_agreement_rate'], '.3f')}")
            print(f"  exact-bit match   {fmt(g['exact_bit_match_rate'], '.3f')}")
            # engine#3158: the clause's *other* half, and the only half a route that emits no log-probs can
            # answer -- Loom's chat response returns logprobs:null at the pinned hrx_loom, so the whole
            # comparison above is unanswerable there while this one is not. Text-level on purpose: it counts
            # a whitespace or detokenisation difference as a disagreement, which is why it is reported with
            # its own denominator rather than folded into the token rates.
            if g["greedy_agreement_pct"] is None:
                print(f"  greedy agreement  cannot tell -- no prompt was answered with content on both routes "
                      f"({len(g['prompts_both_empty'])} of {g['prompts']} had none on either)")
            else:
                print(f"  greedy agreement  {g['greedy_agreement_pct']:.1f} % over {g['prompts_compared']} "
                      f"prompt(s) (generated text)")
            if g["prompts_both_empty"]:
                # Both arms said nothing: identical strings, no comparison. Counted, never scored.
                print(f"  empty answers     prompts {g['prompts_both_empty']} answered with no content on EITHER "
                      f"route -- not comparable, excluded from the percentage")
            if g["text_prompt_divergence"]:
                print(f"  text divergence   prompts {g['text_prompt_divergence']}")
            # engine#3158 near-tie rule (orchestrator 23 Nov ~20:2xZ): against the arbiter, count only splits whose
            # arbiter top-2 margin exceeds --near-tie-nats; report the near-tie count beside the score
            if a.arbiter in (la, lb) and la != lb:
                cand, ref = (pb, pa) if la == a.arbiter else (pa, pb)
                if all(x["runs"][0].get("tokens") for x in ref):
                    nt = near_tie_score(cand, ref, a.near_tie_nats)
                    g["near_tie"] = nt
                    det = ", ".join(f"p{x['prompt']}@{x['char']}:" + (f"cut({x['cut']})" if x["kind"] == "truncated" else
                                    f"{'?' if x['margin'] is None else round(x['margin'], 3)}{'*' if x['kind'] == 'counted' else ''}")
                                    for x in nt["splits"])
                    print(f"  near-tie rule     {nt['score']}/{nt['prompts']} (identical {nt['identical']}, near-tie splits "
                          f"{nt['near_tie']}, truncated {nt['truncated']}, counted {nt['counted']}; {a.arbiter} log-prob gap to the route's token > {a.near_tie_nats} nat counts)"
                          + (f"  [{det}]  (* = counted)" if det else ""))
                else:
                    print(f"  near-tie rule     cannot tell -- {a.arbiter} carried no tokens with log-probs")
            # engine#3158: the accuracy row asks for a mean KLD against CPU, which none of the above
            # answers -- they are all comparisons of the tokens the two routes DID emit. `--route` order
            # is the direction: the first route is the reference, so `--route cpu=… --route hrx=…` gives
            # KL(hrx || cpu). Truncated to the top-N both routes carry; see _position_distribution.
            # EVERY run of both routes, paired run-with-run, not `runs[0]`: the determinism section above
            # holds each route to repeating itself, and the row's claim is about the route, so a mean over
            # the repeats is what it implies. Weighted by positions, so a run that produced fewer shared
            # positions does not count as much as one that produced more.
            ks = [kld(rx, ry, mb) for x, y in zip(pa, pb)
                  for rx, ry in zip(x["runs"], y["runs"])]
            k = kld_summary(ks)
            k.update({"reference": la, "candidate": lb})
            g["kld"] = k
            if k["mean_kld"] is None:
                # cannot-tell, and it must read as one: this is the diverged-everywhere case.
                any_no_lp = not all(report["determinism"].get(x, {}).get("logprobs_available", True)
                                    for x in (la, lb))
                if missing_kld_fails(a.text_only, any_no_lp):
                    failed = True
                print(f"  mean KLD          cannot tell -- no position had both routes on the same token with a "
                      f"top-N (KL direction {lb} || {la}, {k['runs_compared']} run pair(s))")
            else:
                unseen_note = (f", {k['positions_with_unseen_entries']} with entries below top-N"
                               if k["positions_with_unseen_entries"] else "")
                print(f"  mean KLD          {k['mean_kld']:.6f} nats  (KL({lb} || {la}), "
                      f"{k['positions']} positions{unseen_note})")
                print(f"  same top token    {k['same_top_token_pct']:.1f} %")
            if g["prompts_with_token_divergence"]:
                print(f"  token divergence  prompts {g['prompts_with_token_divergence']}")

    if a.text_only:
        excused = [label for label in labels
                   if not report["determinism"].get(label, {}).get("logprobs_available", True)]
        if excused:
            print(f"\n--text-only: {', '.join(excused)} carried no log-probs, so the bit question and the KLD are "
                  f"cannot tell BY THE ROUTE'S SHAPE, not because the routes agreed; the text comparison above is what "
                  f"decided this run")
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
