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
"""markovian_rsa.py — Markovian RSA test-time compute against a `1bit serve` endpoint.

    tools/markovian_rsa.py --url http://127.0.0.1:8080 "PROBLEM"            (or --problem-file F)
    tools/markovian_rsa.py --url ... --eval problems.jsonl --out results.jsonl   ({"problem","answer"} per line)

Markovian RSA is the test-time compute method Zyphra published with ZAYA1-8B (ZAYA1-8B Technical
Report, arXiv:2605.05365). It combines Recursive Self-Aggregation (Venkatraman et al., "Recursive
Self-Aggregation Unlocks Deep Thinking in Large Language Models", arXiv:2509.26626) with the
Markovian-thinking idea of carrying only a bounded tail of each trace forward:

  round 0      N independent traces of the problem, each at most BUDGET tokens (thinking on);
               keep the last TAIL tokens of each trace (reasoning and answer together);
  rounds 1..T  N new traces, each from an aggregation prompt holding the problem and C tails drawn
               at random from the previous round; keep their tails;
  answer       the \\boxed{} answers of round T, by majority (ties: the earliest).

The context of any request is bounded by prompt + C * TAIL + BUDGET, however many rounds run.
Zyphra's headline setting is N=16, C=4, T=2, BUDGET=40000, TAIL=4000 (AIME'25 91.9, HMMT'25 89.6),
about 1.9 M generated tokens per problem; the defaults here are that setting, and --preset local is
a small one for a single Strix Halo. The aggregation prompt is our own wording of the report's
description ("consider the candidates and produce the best solution"). Sampling follows the model's
generation_config (ZAYA1-8B: temperature 1.0, top_p 0.95).

Requests run --concurrency at a time; start the server with at least that many slots
(`1bit serve --parallel K`), each with room for prompt + C*TAIL + BUDGET tokens.
Tails are cut on exact token boundaries through the server's /tokenize and /detokenize
(`1bit serve` passes both through to llama-server); without them a tail is TAIL*4 characters.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request

PRESETS = {
    "zyphra": {"n": 16, "c": 4, "rounds": 2, "budget": 40000, "tail": 4000},
    "local": {"n": 4, "c": 2, "rounds": 2, "budget": 12000, "tail": 2000},
}

AGGREGATE = (
    "Below is a math problem followed by {k} attempts at it, each showing only the last part of its "
    "reasoning. Any of them may be wrong or unfinished. Weigh them against each other: keep what is "
    "right, repair what is wrong, and where they disagree work out which is correct. If none of them "
    "holds up, solve the problem another way. Finish with the final answer in \\boxed{{}}.\n\n"
    "Problem:\n{problem}\n\n{attempts}\n"
    "Now write one complete solution and end with the final answer in \\boxed{{}}."
)
SINGLE = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


class Server:
    def __init__(self, url, model, temperature, top_p, timeout):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout
        self.sampling = {"temperature": temperature, "top_p": top_p}
        self.exact_tails = True
        self.tokens = 0

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def chat(self, prompt, budget):
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": budget, **self.sampling,
                "chat_template_kwargs": {"enable_thinking": True}}
        if self.model:
            body["model"] = self.model
        for attempt in range(3):
            try:
                r = self.post("/v1/chat/completions", body)
                break
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        msg = r["choices"][0]["message"]
        self.tokens += (r.get("usage") or {}).get("completion_tokens", 0)
        reasoning = msg.get("reasoning_content") or ""
        content = msg.get("content") or ""
        return (reasoning + ("\n\n" if reasoning and content else "") + content), content, r["choices"][0].get("finish_reason")

    def tail(self, text, n):
        if self.exact_tails:
            try:
                toks = self.post("/tokenize", {"content": text})["tokens"]
                if len(toks) <= n:
                    return text
                return self.post("/detokenize", {"tokens": toks[-n:]})["content"]
            except Exception:  # noqa: BLE001 — server without /tokenize: fall back to characters
                self.exact_tails = False
        return text[-4 * n:]


def boxed(text):
    """the last \\boxed{...} in text, braces balanced"""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k].strip()
    return None


def norm(ans):
    if ans is None:
        return None
    a = re.sub(r"\\(left|right|,|;|!|\s)", "", ans)
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a).replace("$", "").replace(" ", "")
    a = re.sub(r"^\{(.*)\}$", r"\1", a)
    return a.rstrip(".")


def vote(answers):
    got = [norm(a) for a in answers if norm(a)]
    if not got:
        return None, {}
    counts = collections.Counter(got)
    best = max(counts.items(), key=lambda kv: (kv[1], -got.index(kv[0])))[0]
    return best, dict(counts)


def markovian_rsa(srv, problem, n, c, rounds, budget, tail, concurrency, seed, log=print):
    rng = random.Random(seed)
    history = []
    with cf.ThreadPoolExecutor(concurrency) as pool:
        prompts = [SINGLE.format(problem=problem)] * n
        for t in range(rounds + 1):
            t0 = time.time()
            outs = list(pool.map(lambda p: srv.chat(p, budget), prompts))
            tails = list(pool.map(lambda o: srv.tail(o[0], tail), outs))
            answers = [boxed(o[1]) or boxed(o[0][-4000:]) for o in outs]
            best, counts = vote(answers)
            history.append({"round": t, "answers": answers, "vote": counts, "finish": [o[2] for o in outs],
                            "seconds": round(time.time() - t0, 1)})
            log(f"  round {t}: {counts or 'no boxed answers'} ({time.time() - t0:.0f} s, {srv.tokens:,} tokens so far)")
            if t == rounds:
                return best, history
            prompts = []
            for _ in range(n):
                picks = rng.sample(tails, min(c, len(tails)))
                attempts = "\n".join(f"---- Attempt {i} (last part) ----\n{p.strip()}\n" for i, p in enumerate(picks, 1))
                prompts.append(AGGREGATE.format(k=len(picks), problem=problem.strip(), attempts=attempts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("problem", nargs="?")
    ap.add_argument("--problem-file")
    ap.add_argument("--eval", help="JSONL of {problem, answer}: run each, report accuracy")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="write per-problem results (JSONL)")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", default="")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="zyphra")
    for k in ("n", "c", "rounds", "budget", "tail"):
        ap.add_argument(f"--{k}", type=int)
    ap.add_argument("--single", action="store_true", help="baseline: one trace per problem, no aggregation")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=7200)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) if getattr(a, k) is not None else v for k, v in PRESETS[a.preset].items()}
    if a.single:
        cfg.update(n=1, rounds=0)
    srv = Server(a.url, a.model, a.temperature, a.top_p, a.timeout)

    if a.eval:
        items = [json.loads(x) for x in open(a.eval) if x.strip()]
        items = items[:a.limit] if a.limit else items
        right, out = 0, open(a.out, "a") if a.out else None
        for i, it in enumerate(items, 1):
            t0, tok0 = time.time(), srv.tokens
            print(f"[{i}/{len(items)}] {it['problem'][:70]!r}")
            best, hist = markovian_rsa(srv, it["problem"], cfg["n"], cfg["c"], cfg["rounds"], cfg["budget"],
                                       cfg["tail"], a.concurrency, a.seed + i)
            ok = best is not None and norm(str(it["answer"])) == best
            right += ok
            print(f"  -> {best} (expected {it['answer']}) {'OK' if ok else 'WRONG'} · {time.time() - t0:.0f} s · "
                  f"{srv.tokens - tok0:,} tokens · running {right}/{i}")
            if out:
                out.write(json.dumps({"i": i, "answer": it["answer"], "got": best, "ok": ok, "config": cfg,
                                      "seconds": round(time.time() - t0, 1), "tokens": srv.tokens - tok0,
                                      "history": hist}) + "\n")
                out.flush()
        print(f"accuracy {right}/{len(items)} = {right / max(1, len(items)):.1%} · config {cfg} · {srv.tokens:,} tokens")
        return 0
    problem = open(a.problem_file).read() if a.problem_file else a.problem
    if not problem:
        ap.error("give a problem, --problem-file or --eval")
    best, hist = markovian_rsa(srv, problem, cfg["n"], cfg["c"], cfg["rounds"], cfg["budget"], cfg["tail"],
                               a.concurrency, a.seed)
    print(f"answer: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
