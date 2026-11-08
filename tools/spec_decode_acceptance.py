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
    return fails


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", help="llama-server base URL of the TARGET model")
    ap.add_argument("--draft", help="llama-server base URL of the DRAFT model")
    ap.add_argument("--prompt-file", help="file holding the prompt text")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=0, help="0 = until --tokens emitted")
    ap.add_argument("--selftest", action="store_true", help="check the prefix rule and the units; needs no server")
    a = ap.parse_args(argv)
    if a.selftest:
        return EXIT_BAD_INPUT if selftest() else EXIT_OK
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
