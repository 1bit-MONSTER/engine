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
"""Unsloth-NPU-TEST harness: the brief's protocol against ANY OpenAI-compatible endpoint, so arm A
(FastFlowLM behind Lemonade) and arm B (our engine behind Lemonade) are measured by one program.
Black-box HTTP only - it never looks inside the server.

  unsloth_probe.py --base http://127.0.0.1:PORT --model TAG --rounds 4 --out result.json

Measures per prompt length (short / ~2k / ~8k tokens):
  decode tok/s, prefill tok/s, first-token time - >=4 interleaved rounds, reports min..max and median
and the three agent-app checks: one tool-calling request (OpenAI tools schema), one streamed chat,
one JSON-mode request. Also 5 fixed greedy prompts -> text, for cross-arm agreement.
"""
import argparse, json, os, re, statistics, sys, time, urllib.request

PROMPTS = {  # ~word counts; tokens are counted from the server's own usage.prompt_tokens
    "short": "Name three primary colours.",
    "2k": "Summarise the following notes in one sentence: " + ("the array schedules cores per column; " * 420),
    "8k": "Summarise the following notes in one sentence: " + ("the array schedules cores per column; " * 1700),
}
GREEDY = ["What is 17*23?", "Name the capital of France.", "List two prime numbers above 100.",
          "Say hello in three languages.", "What does NPU stand for?"]


def call(base, path, body, stream=False, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            return json.loads(r.read().decode()), time.time() - t0
        first, chunks, dumped = None, [], 0
        for raw in r:
            line = raw.decode().strip()
            if os.environ.get("UNSLOTH_DUMP_STREAM") == "1" and dumped < 3:
                print("RAW>", line[:220], file=sys.stderr)   # names the stream shape instead of inferring it
                dumped += 1
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = obj.get("choices", [{}])[0]
            delta = (ch.get("delta") or {}).get("content") or (ch.get("delta") or {}).get("reasoning_content") \
                or (ch.get("message") or {}).get("content") or ch.get("text")
            if delta:
                first = first if first is not None else time.time() - t0
                chunks.append(delta)
        return {"text": "".join(chunks), "first_token_s": first}, time.time() - t0


REASONING_OFF = os.environ.get("UNSLOTH_NO_THINK") == "1"


def _body(model, prompt, max_tokens, stream):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
         "temperature": 0, "stream": stream}
    if REASONING_OFF:
        b["think"] = False          # FLM 1.0.13 accepts `think`; harmless for a server that ignores it
    return b


def one_round(base, model, name, prompt, max_tokens):
    """Streamed on purpose: the first arm's first run measured wall-clock over the whole request, which made
    "decode tok/s" fall from 16.5 at a short prompt to 0.85 at 8k - prefill, not decode, was in the number. With
    streaming the first token splits the request, so decode = tokens/(wall - ttft) and prefill = prompt_tokens/ttft."""
    res, wall = call(base, "/v1/chat/completions", _body(model, prompt, max_tokens, True), stream=True)
    res2, _ = call(base, "/v1/chat/completions",
                   _body(model, prompt, max_tokens, False))   # usage (token counts) comes non-streamed
    u = res2.get("usage", {}) or {}
    ptok, ctok = u.get("prompt_tokens"), u.get("completion_tokens")
    ttft = res.get("first_token_s")
    dec_s = (wall - ttft) if (ttft is not None and wall > ttft) else None
    return {"prompt": name, "wall_s": round(wall, 3), "ttft_s": round(ttft, 3) if ttft else None,
            "prompt_tokens": ptok, "completion_tokens": ctok,
            "decode_tok_s": round(ctok / dec_s, 2) if (ctok and dec_s) else None,
            "prefill_tok_s": round(ptok / ttft, 2) if (ptok and ttft) else None}


FLM_LINE = re.compile(r"prompt (\d+) tokens \(prefill ([0-9.]+) s, ([0-9.]+) tok/s\), generated (\d+) tokens "
                      r"\(decode ([0-9.]+) s, ([0-9.]+) tok/s\)")


def from_flm_log(path, out_path):
    """Numbers the SERVER reports. Cannot be distorted by a client-side parse, which is why it exists: FLM 1.0.13's
    SSE shape is not `choices[0].delta.content` for every model, and a stream parse that silently yields nothing
    turned every throughput column into null on qwen3:1.7b while the server was measuring 25.6 tok/s."""
    rows = []
    for line in open(path, errors="replace"):
        m = FLM_LINE.search(line)
        if m:
            ptok, pf_s, pf_rate, ctok, dec_s, dec_rate = m.groups()
            rows.append({"prompt_tokens": int(ptok), "prompt_s": float(pf_s), "prompt_tok_s": float(pf_rate),
                         "completion_tokens": int(ctok), "decode_s": float(dec_s), "decode_tok_s": float(dec_rate)})
    json.dump({"source": f"flm log {path}", "requests": rows}, open(out_path, "w"), indent=1)
    by_len = {}
    for r in rows:
        key = "short" if r["prompt_tokens"] < 100 else ("2k" if r["prompt_tokens"] < 6000 else "8k")
        by_len.setdefault(key, []).append(r)
    for k, v in by_len.items():
        d = [r["decode_tok_s"] for r in v]
        p = [r["prompt_tok_s"] for r in v]
        print(f"{k:5s} n={len(v)}  decode {min(d):.2f}-{max(d):.2f} (median {statistics.median(d):.2f})  "
              f"prefill {min(p):.1f}-{max(p):.1f}")
    print(f"{len(rows)} requests -> {out_path}")


def self_test():
    """Device-free: the parser against a stub that streams the same words in each shape a server may use. This is
    the check that separates "the server sent nothing my client reads" from "my client cannot read what it sent" -
    the distinction that cost a retracted measurement row when FLM streamed `delta.reasoning_content`
    (engine#3193). Exit 0 = every shape read, 1 = a shape was lost."""
    import os, subprocess, sys, time
    here = os.path.dirname(os.path.abspath(__file__))
    bad = 0
    for shape in ("delta.content", "delta.reasoning_content", "message.content"):
        port = 53990 + len(shape)
        srv = subprocess.Popen([sys.executable, os.path.join(here, "sse_stub.py"), str(port), shape],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(0.6)
            res, _ = call(f"http://127.0.0.1:{port}", "/v1/chat/completions", {"stream": True}, stream=True)
        except Exception as e:                     # a stub that will not start is a failure, not a skip
            print(f"FAIL {shape}: {e}")
            bad += 1
            continue
        finally:
            srv.terminate()
            srv.wait(timeout=5)
        ok = (res.get("text") or "").strip() == "one two three" and res.get("first_token_s") is not None
        print(f'{"ok  " if ok else "FAIL"} {shape:<24s} text={res.get("text")!r} first_token={res.get("first_token_s")}')
        bad += 0 if ok else 1
    print(f"unsloth_probe self-test: {'PASS' if bad == 0 else 'FAIL'} ({3 - bad} of 3 stream shapes read)")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--model")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", help="where to write the JSON result (required unless --self-test)")
    ap.add_argument("--from-flm-log", help="parse FLM's own per-request log instead of driving HTTP")
    ap.add_argument("--self-test", action="store_true", help="check the stream parser against the stub, no server")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.out:
        ap.error("--out is required unless --self-test")
    if a.from_flm_log:
        return from_flm_log(a.from_flm_log, a.out)
    out = {"base": a.base, "model": a.model, "rounds": a.rounds, "timings": [], "agent": {}, "greedy": {}}
    for r in range(a.rounds):                       # interleaved: every length once per round
        for name, prompt in PROMPTS.items():
            try:
                out["timings"].append(dict(round=r, **one_round(a.base, a.model, name, prompt, a.max_tokens)))
            except Exception as e:                  # a failing probe is data, not a crash
                out["timings"].append({"round": r, "prompt": name, "error": str(e)[:200]})
    for name in PROMPTS:
        vals = [t["decode_tok_s"] for t in out["timings"] if t.get("prompt") == name and t.get("decode_tok_s")]
        if vals:
            out.setdefault("summary", {})[name] = {"n": len(vals), "min": min(vals), "max": max(vals),
                                                  "median": round(statistics.median(vals), 2)}
    # agent-app checks
    tools = [{"type": "function", "function": {"name": "get_time", "description": "current time",
              "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}, "required": ["tz"]}}}]
    try:
        res, _ = call(a.base, "/v1/chat/completions", {"model": a.model, "messages": [
            {"role": "user", "content": "What time is it in Tokyo? Use the tool."}], "tools": tools,
            "tool_choice": "auto", "max_tokens": 128})
        msg = res.get("choices", [{}])[0].get("message", {})
        out["agent"]["tool_calling"] = {"pass": bool(msg.get("tool_calls")), "raw": str(msg.get("tool_calls"))[:300]}
    except Exception as e:
        out["agent"]["tool_calling"] = {"pass": False, "error": str(e)[:200]}
    try:
        s, _ = call(a.base, "/v1/chat/completions", {"model": a.model, "stream": True, "max_tokens": 48,
             "messages": [{"role": "user", "content": "Count from 1 to 10."}]}, stream=True)
        out["agent"]["streaming"] = {"pass": bool(s.get("text")), "first_token_s": s.get("first_token_s"),
                                     "text_head": (s.get("text") or "")[:80]}
    except Exception as e:
        out["agent"]["streaming"] = {"pass": False, "error": str(e)[:200]}
    try:
        res, _ = call(a.base, "/v1/chat/completions", {"model": a.model, "max_tokens": 64,
             "response_format": {"type": "json_object"},
             "messages": [{"role": "user", "content": "Reply with a JSON object with key ok and value true."}]})
        txt = res.get("choices", [{}])[0].get("message", {}).get("content", "")
        json.loads(txt)
        out["agent"]["json_mode"] = {"pass": True, "text_head": txt[:80]}
    except Exception as e:
        out["agent"]["json_mode"] = {"pass": False, "error": str(e)[:200]}
    for i, p in enumerate(GREEDY):
        try:
            res, _ = call(a.base, "/v1/chat/completions", {"model": a.model, "temperature": 0, "max_tokens": 64,
                 "messages": [{"role": "user", "content": p}]})
            out["greedy"][f"p{i}"] = res.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            out["greedy"][f"p{i}"] = f"ERROR {str(e)[:120]}"
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out.get("summary", {}), indent=1))
    print("agent:", {k: v.get("pass") for k, v in out["agent"].items()})


if __name__ == "__main__":
    # engine#3193: the row ctest runs is `--self-test`, and add_python_test judges it by EXIT STATUS. self_test()
    # returns 1 for a shape the parser could not read, and main() propagates it, so discarding that value makes the
    # row green no matter what the parser does -- measured: with the `reasoning_content` branch removed it prints
    # "FAIL (2 of 3 stream shapes read)" and exits 0 without this line, and 1 with it.
    sys.exit(main())
