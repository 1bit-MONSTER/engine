#!/usr/bin/env python3
"""AMD-protocol numbers for a server on the NPU (strixhalo), for the ONNX comparison table.

The engine's NPU route serves the same OpenAI-compatible API as `flm`, so the three stages AMD
reports (prompt processing, token generation, E2E) can be measured from the stream itself. Token
counts come from the response's own `usage`, not from a guess, so the rates are per real token.

Ours (the NPU row):      ~/src/engine-laya-npu/build-npu/1bit serve -m ~/lax-private-test/q06 --device npu -p 52625 &
FastFlowLM (reference):  flm serve qwen3:0.6b --port 52625 &

  python3 bench_npu_serve.py --lens "2048,1024,512,256,128" --gen 128 -w 1 -r 5
"""
import argparse
import json
import statistics
import time
import urllib.request

PROMPT_WORD = "The quick brown fox jumps over the lazy dog, and then it does it again. "


def prompt_of_tokens(n_tokens: int) -> str:
    # ~13 tokens per repetition for this sentence; the real count is read back from `usage`.
    return PROMPT_WORD * max(1, n_tokens // 13)


def one_run(host: str, model: str, prompt: str, gen: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": gen,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{host}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first = None
    stamps = []
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                if choice.get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    stamps.append(now)
    end = time.perf_counter()
    if first is None:
        raise SystemExit("no tokens streamed")
    return {
        "ttft_ms": (first - t0) * 1000.0,
        "token_ms": [(b - a) * 1000.0 for a, b in zip(stamps, stamps[1:])],
        "e2e_ms": (end - t0) * 1000.0,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", len(stamps)),
    }


def stat_of(vals):
    if not vals:
        return 0.0, 0.0, 0.0, 0
    avg = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    return avg, statistics.median(vals), sd, len(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:52625")
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--lens", default="2048,1024,512,256,128")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("-w", "--warmup", type=int, default=1)
    ap.add_argument("-r", "--repeat", type=int, default=5)
    args = ap.parse_args()

    print("-----------------------------")
    for want in [int(x) for x in args.lens.replace(",", " ").split()]:
        prompt = prompt_of_tokens(want)
        ttft, tok, e2e = [], [], []
        n_prompt = 0
        for i in range(args.warmup + args.repeat):
            r = one_run(args.host, args.model, prompt, args.gen)
            if i < args.warmup:
                continue
            n_prompt = r["prompt_tokens"]
            ttft.append(r["ttft_ms"] * 1000.0)
            tok.extend(x * 1000.0 for x in r["token_ms"])
            e2e.append(r["e2e_ms"])
        ta, tp, ts, tn = stat_of(ttft)
        ga, gp, gs, gn = stat_of(tok)
        ea, ep, es, en = stat_of(e2e)
        print(f"Prompt\n  Number of Tokens: {n_prompt}")
        print(f"Batch size: 1, prompt tokens: {n_prompt}, tokens to generate: {args.gen}")
        print("  Prompt processing (time to first token):")
        print(
            f"      avg (us): {ta:.0f}  avg (tokens/s): {1e6 * n_prompt / ta if ta else 0:.3f}"
            f"  p50 (us): {tp:.0f}  stddev (us): {ts:.0f}  n: {tn} * {n_prompt} token(s)"
        )
        print("  Token generation:")
        print(
            f"      avg (us): {ga:.1f}  avg (tokens/s): {1e6 / ga if ga else 0:.4f}"
            f"  p50 (us): {gp:.1f}  stddev (us): {gs:.1f}  n: {gn} * 1 token(s)"
        )
        print("  E2E generation (entire generation loop):")
        print(
            f"      avg (ms): {ea:.2f}  p50 (ms): {ep:.2f}  stddev (ms): {es:.2f}  n: {en}"
        )
        print("-----------------------------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
