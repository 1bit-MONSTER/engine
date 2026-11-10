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
"""spec_decode_acceptance.py -- acceptance of a DRAFT model against a TARGET model, both greedy, in TOKENS.

Why this exists rather than a one-off loop: "acceptance" is the number a speculative-decoding row is judged on, and it has
three traps that have already cost this project real numbers:

  1. **tokens/round is not accepted-drafts/round**, and neither is tokens/base-pass. The row prints both, named.
  2. **two processes are not two arms.** Both models are driven from this one process (two local llama-server instances), so
     they see the same box, and the verifier is always the TARGET's own greedy continuation -- never a second sample of the
     draft.
  3. **a degenerate prompt proves nothing.** The per-round accept list is printed, and the tool warns when every round
     accepts everything (copy-heavy prompt) or nothing (the pair does not draft for this text).

Acceptance is counted in token ids (`return_tokens: true`), not in characters: an earlier version of this harness compared
the returned text and then reported "chars per round" beside a k in tokens.

This measures ACCEPTANCE. For the token-rate consequence use `tools/spec_decode_speedup.sh`, which puts both models in one
runtime -- and read its header first: on a CPU-only backend speculation cannot pay at all, because a batched verify costs
~M token-equivalents while a round can only return 1 + accepted <= 1 + drafted.

    tools/spec_decode_acceptance.py --target http://127.0.0.1:18120 --draft http://127.0.0.1:18121 \\
        --prompt-file p.txt --tokens 32 --k 4
    tools/spec_decode_acceptance.py --selftest        # the accept-prefix rule and the units, no server needed

A target that is not a llama-server (engine#1371: an OGA pack served by `1bit forward-serve` on the NPU has no
`/completion`, and its ticket should not be held while a drafter runs) is measured in two steps. Greedy decode is
deterministic, so the target's continuation is recorded once and the loop is replayed against it:

    # inside the device ticket: the target's own greedy text for each prompt, through the OpenAI-shaped endpoint
    tools/spec_decode_acceptance.py --record-openai http://127.0.0.1:18172 --prompts prompts.json --tokens 40 \
        --out target.json
    # afterwards, no device: the draft proposes k tokens from every committed prefix of that recording
    tools/spec_decode_acceptance.py --target-recorded target.json --draft http://127.0.0.1:18173 --k 4 \
        --pass-ms 1037.5 --step-ms 1020.8

`prompts.json` is `{"name": "prompt text", ...}`. Token ids come from the DRAFT server's tokenizer, so the pair must
share a vocabulary (the tool refuses a recording whose prompt does not tokenize to a prefix of prompt + continuation).
With `--pass-ms` (one batched verify pass of k+1 rows) and `--step-ms` (one plain decode step), both measured on the
target, the row also carries the break-even bar and the projected speed-up for a one-pass round and for a two-pass
round. Those two lines are ARITHMETIC over measured parts, and are labelled so: they are not a measured loop.
"""
import argparse
import json
import sys
import urllib.request

EXIT_OK, EXIT_BAD_INPUT, EXIT_SERVER = 0, 2, 3


def _post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def greedy(url, prompt, n):
    """The model's own greedy continuation of `prompt`: (token ids, text). Token ids are what acceptance is counted in."""
    r = _post(url, "/completion", {"prompt": prompt, "n_predict": n, "temperature": 0.0, "top_k": 1, "top_p": 1.0,
                                   "repeat_penalty": 1.0, "seed": 0, "cache_prompt": False, "stream": False,
                                   "return_tokens": True, "stop": []})
    return (r.get("tokens") or []), (r.get("content") or "")


def detok(url, toks):
    if not toks:
        return ""
    return _post(url, "/detokenize", {"tokens": toks}).get("content", "")


def accept_prefix(draft_ids, truth_ids):
    """The accepted prefix length: the longest common TOKEN prefix. Pure, so --selftest can check it without a server."""
    n = 0
    while n < len(draft_ids) and n < len(truth_ids) and draft_ids[n] == truth_ids[n]:
        n += 1
    return n


def replay_rounds(truth_ids, draft_fn, k):
    """The verify loop against a FIXED greedy continuation `truth_ids` (the target's own, recorded once).

    Each round asks `draft_fn(i, k)` for up to k draft ids continuing the committed prefix of length i, accepts the
    longest common prefix with truth_ids[i:], and commits accepted + 1 tokens: the accepted drafts plus the target's
    own token at the first mismatch, which the same verify pass produced. Returns (per-round accepts, offered).
    Pure: `draft_fn` is the only thing that touches a server."""
    i, per_round, offered = 0, [], 0
    while i < len(truth_ids):
        d = list(draft_fn(i, k))[:k]
        offered += len(d)
        a = accept_prefix(d, truth_ids[i:])
        per_round.append(a)
        i += a + 1
    return per_round, offered


def loop_economics(tokens_per_round, k, pass_ms, step_ms, draft_ms_per_token):
    """Break-even and projected speed-up from measured parts. A one-pass round is one batched verify pass plus k
    draft tokens; a two-pass round (mtp::run's shape) adds the plain step that consumes the committed token. The bar
    is the tokens/round at which a round costs what the same tokens would cost as plain steps."""
    out = {}
    for name, cost in (("one_pass", pass_ms + k * draft_ms_per_token),
                       ("two_pass", pass_ms + step_ms + k * draft_ms_per_token)):
        out[name] = {"round_ms": round(cost, 1), "bar_tokens_per_round": round(cost / step_ms, 3),
                     "projected_speedup": round(tokens_per_round * step_ms / cost, 3)}
    return out


def selftest():
    """The parts that can be wrong without a model: the prefix rule and the unit arithmetic."""
    fails = 0

    def check(cond, what):
        nonlocal fails
        print(f"{'PASS' if cond else 'FAIL'}  {what}")
        if not cond:
            fails += 1

    check(accept_prefix([1, 2, 3], [1, 2, 4]) == 2, "a shared prefix of 2 is accepted")
    check(accept_prefix([1, 2, 3], [1, 2, 3]) == 3, "a fully matching draft is accepted entirely")
    check(accept_prefix([9], [1]) == 0, "a first-token miss accepts nothing")
    check(accept_prefix([], [1, 2]) == 0, "an empty draft accepts nothing")
    check(accept_prefix([1, 2], [1]) == 1, "a draft longer than the target's continuation is capped by the target")
    # the units: tokens/round is 1 committed + accepted drafts per round, NOT the accepted drafts alone
    rounds, emitted, accepted = 9, 24, 19
    tpr = emitted / rounds
    check(abs(tpr - 2.667) < 0.001, f"tokens/round is emitted/rounds ({tpr:.3f})")
    check(abs(accepted / rounds - 2.111) < 0.001, "accepted drafts/round is a different number, printed beside it")
    # the replay against a recorded continuation (engine#1371): a + 1 tokens per round, the target's own at the miss
    truth = [10, 11, 12, 13, 14, 15, 16]
    pr, off = replay_rounds(truth, lambda i, k: truth[i:i + k], 4)
    check(pr == [4, 2] and off == 6, f"replay: a perfect drafter at k=4 takes 7 tokens in 2 rounds ({pr}, offered {off})")
    pr, off = replay_rounds(truth, lambda i, k: [0] * k, 4)
    check(pr == [0] * 7 and off == 28, "replay: a drafter that is always wrong commits exactly 1 token per round")
    pr, _ = replay_rounds(truth, lambda i, k: [truth[i], 99, 99, 99][:k] if i < len(truth) else [], 4)
    # the last round has no "+1" when its accepted drafts reach the end of the recording, so the identity is
    # accepted + rounds == tokens or tokens + 1 -- which is why tokens/round divides the TOKENS, not that sum
    check(pr == [1, 1, 1, 1] and sum(pr) + len(pr) in (len(truth), len(truth) + 1),
          f"replay: accepted + rounds is the tokens committed, give or take the final round's own token ({pr})")
    pr, _ = replay_rounds(truth[:6], lambda i, k: [truth[i], 99, 99, 99][:k], 4)
    check(pr == [1, 1, 1] and sum(pr) + len(pr) == 6, f"replay: and exactly the tokens when the recording ends on a miss ({pr})")
    pr, off = replay_rounds(truth, lambda i, k: truth[i:i + 50], 3)
    check(max(pr) <= 3 and off <= 3 * len(pr), "replay: a draft longer than k is cut to k before it is counted")
    # the economics: a verify pass that costs one step puts the one-pass bar just above 1 and the two-pass bar above 2
    e = loop_economics(3.0, 4, 1000.0, 1000.0, 5.0)
    check(e["one_pass"]["bar_tokens_per_round"] == 1.02 and e["two_pass"]["bar_tokens_per_round"] == 2.02,
          f"economics: bars are round cost / step cost ({e['one_pass']['bar_tokens_per_round']}, "
          f"{e['two_pass']['bar_tokens_per_round']})")
    check(e["one_pass"]["projected_speedup"] == 2.941 and e["two_pass"]["projected_speedup"] == 1.485,
          "economics: projected speed-up is tokens/round over the bar")
    e = loop_economics(1.0, 4, 1200.0, 300.0, 50.0)
    check(e["one_pass"]["projected_speedup"] < 1.0, "economics: a drafter that accepts nothing is a slow-down, not 1.0x")
    return fails


def _post_openai(url, prompt, n):
    r = _post(url, "/v1/completions", {"prompt": prompt, "max_tokens": n, "temperature": 0})
    return r["choices"][0].get("text", ""), r.get("usage", {})


def record_openai(url, prompts_path, n, out_path):
    """Step 1 of the recorded mode: the target's greedy continuation of each prompt, as text."""
    prompts = json.load(open(prompts_path))
    if not isinstance(prompts, dict) or not prompts:
        print("spec_decode_acceptance: --prompts must be a non-empty JSON object {name: prompt}", file=sys.stderr)
        return EXIT_BAD_INPUT
    rec = {}
    for name, prompt in prompts.items():
        text, usage = _post_openai(url, prompt, n)
        rec[name] = {"prompt": prompt, "text": text, "usage": usage}
        print(f"recorded {name}: {usage.get('completion_tokens', '?')} tokens", file=sys.stderr)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return EXIT_OK


def recorded(a):
    """Step 2: replay the loop against the recording, the draft server proposing from every committed prefix."""
    rec = json.load(open(a.target_recorded))
    rows, tot_tokens, tot_rounds, tot_acc, tot_off, d_ms, d_n = {}, 0, 0, 0, 0, 0.0, 0

    def tok(text):
        return _post(a.draft, "/tokenize", {"content": text, "add_special": False}).get("tokens") or []

    for name, r in rec.items():
        p_ids = tok(r["prompt"])
        t_ids = tok(r["prompt"] + r["text"])
        if not p_ids or t_ids[:len(p_ids)] != p_ids:
            print(f"spec_decode_acceptance: {name}: the prompt does not tokenize to a prefix of prompt + continuation "
                  "under the draft's tokenizer, so token positions cannot be compared; refusing this recording",
                  file=sys.stderr)
            return EXIT_BAD_INPUT
        truth = t_ids[len(p_ids):]
        timing = {"ms": 0.0, "n": 0}

        def draft_fn(i, k, _ids=t_ids, _base=len(p_ids), _t=timing):
            rr = _post(a.draft, "/completion", {"prompt": _ids[:_base + i], "n_predict": k, "temperature": 0.0,
                                                "top_k": 1, "top_p": 1.0, "repeat_penalty": 1.0, "seed": 0,
                                                "cache_prompt": True, "stream": False, "return_tokens": True,
                                                "stop": []})
            tm = rr.get("timings") or {}
            _t["ms"] += tm.get("predicted_ms", 0.0)
            _t["n"] += tm.get("predicted_n", 0)
            return rr.get("tokens") or []

        per_round, offered = replay_rounds(truth, draft_fn, a.k)
        rows[name] = {"tokens": len(truth), "rounds": len(per_round),
                      "tokens_per_round": round(len(truth) / len(per_round), 3) if per_round else 0.0,
                      "accepted_tokens_per_round": round(sum(per_round) / len(per_round), 3) if per_round else 0.0,
                      "per_round_accept_tokens": per_round}
        tot_tokens += len(truth); tot_rounds += len(per_round); tot_acc += sum(per_round); tot_off += offered
        d_ms += timing["ms"]; d_n += timing["n"]
    out = {"mode": "recorded target, replayed", "k": a.k, "prompts": rows, "tokens": tot_tokens, "rounds": tot_rounds,
           "accepted_tokens": tot_acc, "offered_tokens": tot_off,
           "tokens_per_round": round(tot_tokens / tot_rounds, 3) if tot_rounds else 0.0,
           "accepted_per_offered": round(tot_acc / tot_off, 3) if tot_off else 0.0,
           "draft_ms_per_token": round(d_ms / d_n, 2) if d_n else None,
           "note": "token ids are the draft tokenizer's over the recorded TEXT; a round commits accepted + 1 tokens."}
    if a.pass_ms and a.step_ms and tot_rounds:
        out["economics_from_measured_parts_not_a_measured_loop"] = loop_economics(
            tot_tokens / tot_rounds, a.k, a.pass_ms, a.step_ms, out["draft_ms_per_token"] or 0.0)
    print(json.dumps(out, indent=2))
    return EXIT_OK


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="llama-server base URL of the TARGET model")
    ap.add_argument("--draft", help="llama-server base URL of the DRAFT model")
    ap.add_argument("--prompt-file", help="file holding the prompt text")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=0, help="0 = until --tokens emitted")
    ap.add_argument("--selftest", action="store_true", help="check the prefix rule and the units; needs no server")
    ap.add_argument("--record-openai", help="record mode: base URL of a target that serves /v1/completions")
    ap.add_argument("--prompts", help="record mode: JSON object {name: prompt}")
    ap.add_argument("--out", help="record mode: where the recording is written")
    ap.add_argument("--target-recorded", help="replay mode: a recording written by --record-openai")
    ap.add_argument("--pass-ms", type=float, default=0.0, help="replay mode: one batched verify pass of k+1 rows, ms")
    ap.add_argument("--step-ms", type=float, default=0.0, help="replay mode: one plain decode step on the target, ms")
    a = ap.parse_args(argv)
    if a.selftest:
        return EXIT_BAD_INPUT if selftest() else EXIT_OK
    if a.record_openai or a.target_recorded:
        if a.record_openai and not (a.prompts and a.out):
            print("spec_decode_acceptance: --record-openai needs --prompts and --out", file=sys.stderr)
            return EXIT_BAD_INPUT
        if a.target_recorded and not a.draft:
            print("spec_decode_acceptance: --target-recorded needs --draft", file=sys.stderr)
            return EXIT_BAD_INPUT
        if a.target_recorded and bool(a.pass_ms) != bool(a.step_ms):
            print("spec_decode_acceptance: --pass-ms and --step-ms go together (the bar is their ratio)", file=sys.stderr)
            return EXIT_BAD_INPUT
        try:
            if a.record_openai:
                return record_openai(a.record_openai, a.prompts, a.tokens, a.out)
            return recorded(a)
        except (urllib.error.URLError, OSError) as e:
            print(f"spec_decode_acceptance: a server did not answer: {e}", file=sys.stderr)
            return EXIT_SERVER
    if not (a.target and a.draft and a.prompt_file):
        print(__doc__)
        return EXIT_BAD_INPUT
    prompt = open(a.prompt_file, errors="replace").read()

    ctx = prompt
    emitted = rounds = accepted_total = offered_total = 0
    per_round = []
    try:
        while emitted < a.tokens and (a.rounds == 0 or rounds < a.rounds):
            t1_ids, t1_txt = greedy(a.target, ctx, 1)
            if not t1_ids:
                print("STOP: the target produced nothing (EOS or an empty completion)", file=sys.stderr)
                break
            ctx += t1_txt
            emitted += 1
            draft_ids, _ = greedy(a.draft, ctx, a.k)
            if not draft_ids:
                rounds += 1
                per_round.append(0)
                continue
            truth_ids, _ = greedy(a.target, ctx, a.k)
            n = accept_prefix(draft_ids, truth_ids)
            take = min(n, max(0, a.tokens - emitted))
            ctx += detok(a.draft, draft_ids[:take])
            emitted += take
            accepted_total += n            # acceptance is a property of the pair, not of the caller's budget
            offered_total += len(draft_ids)
            rounds += 1
            per_round.append(n)
    except (urllib.error.URLError, OSError) as e:
        print(f"spec_decode_acceptance: a server did not answer: {e}", file=sys.stderr)
        return EXIT_SERVER

    out = {
        "tokens": emitted, "k": a.k, "rounds": rounds,
        "accepted_tokens": accepted_total, "offered_tokens": offered_total,
        "tokens_per_round": round(emitted / rounds, 3) if rounds else 0.0,
        "accepted_tokens_per_round": round(accepted_total / rounds, 3) if rounds else 0.0,
        "per_round_accept_tokens": per_round,
        "note": "counted in TOKENS (return_tokens + /detokenize to advance the context). tokens_per_round is the loop's "
                "unit (1 committed + accepted drafts); accepted_tokens_per_round is the drafts alone.",
    }
    print(json.dumps(out, indent=2))
    if rounds and all(p == 0 for p in per_round):
        print("WARN: every round accepted 0 -- this pair does not draft for this prompt, which is not a measurement of "
              "its ceiling", file=sys.stderr)
    if rounds and all(p >= a.k for p in per_round):
        print("WARN: every round accepted k -- a copy-heavy prompt inflates acceptance; try freeform text",
              file=sys.stderr)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
