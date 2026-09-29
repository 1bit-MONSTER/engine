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
"""tools/bench.py: A/B-measure `1bit serve` configurations the way the engine's numbers are
measured (docs/bench.md).

The first --config is the baseline. Every config gets its own fresh `1bit serve`, the configs
run interleaved round by round (A B A B ...), so drift on a shared box hits all of them alike,
and each result is reported against the baseline measured in the same run. The tool, not the
person reading the numbers, runs every measurement: prompt speed on one long prompt, decode
speed on three chat prompts (code, prose, short), greedy, as `serve` reports them.

  tools/bench.py -m Qwen3.8-27B-Q4_0-H32.gguf \\
      --config base= --config dflash='--dflash Qwen3.8-27B-DFlash2-q8_0.gguf' \\
      --lock ~/.cache/lax-decode/box.lock --json out.json
"""
import argparse, contextlib, fcntl, json, os, pathlib, shlex, signal, socket, statistics, subprocess, sys, time
import urllib.error, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECODE_PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 timestamp without using datetime, with tests.",
    "prose": "Explain how a B-tree insert works, step by step.",
    "short": "Translate to French: The quick brown fox jumps over the lazy dog, and then it goes home to sleep.",
}


def default_prompt_text(chars):
    # the engine's own docs, a fixed text anyone with the repo has (about 1,800 tokens at 8,000 chars)
    text = "".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
    return text[:chars]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ask(port, content, max_tokens, timeout):
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": max_tokens,
            "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        reply = json.load(r)
    if "timings" not in reply:
        raise RuntimeError("the backend reported no timings (a llama-server route is needed)")
    return reply["timings"]


@contextlib.contextmanager
def serve(args, name, cfg_args, log_dir):
    port = free_port()
    cmd = [args.onebit, "serve", "-m", args.model, "--port", str(port)] + cfg_args
    log = open(log_dir / f"{name}.log", "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.time() + args.load_timeout
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"{name}: `1bit serve` exited ({proc.returncode}); see {log_dir / (name + '.log')}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, OSError):
                pass
            if time.time() > deadline:
                raise RuntimeError(f"{name}: not healthy after {args.load_timeout} s")
            time.sleep(1)
        yield port
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()


def measure(port, args, prompt):
    ask(port, "hi", 4, args.request_timeout)  # warm-up
    pp = [ask(port, prompt, 8, args.request_timeout)["prompt_per_second"] for _ in range(args.prompt_reps)]
    tg = {k: [ask(port, p, args.decode_tokens, args.request_timeout)["predicted_per_second"]
              for _ in range(args.decode_reps)] for k, p in DECODE_PROMPTS.items()}
    return {"prompt": pp, **tg}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True, help="the model every config serves")
    ap.add_argument("--config", action="append", required=True, metavar="NAME=ARGS",
                    help="a configuration: its name and its extra `1bit serve` arguments; the first is the baseline")
    ap.add_argument("--onebit", default=str(ROOT / "build" / "1bit"), help="the 1bit binary (default: build/1bit)")
    ap.add_argument("--rounds", type=int, default=1, help="interleaved rounds; each round restarts every config")
    ap.add_argument("--prompt-reps", type=int, default=5)
    ap.add_argument("--decode-reps", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--prompt-file", help="the long prompt (default: the first --prompt-chars of docs/*.md)")
    ap.add_argument("--prompt-chars", type=int, default=8000)
    ap.add_argument("--lock", help="hold this flock for the whole run (a box other jobs share)")
    ap.add_argument("--load-timeout", type=int, default=900)
    ap.add_argument("--request-timeout", type=int, default=900)
    ap.add_argument("--log-dir", default="bench-logs")
    ap.add_argument("--json", help="write every measurement here")
    args = ap.parse_args()

    configs = []
    for c in args.config:
        name, sep, rest = c.partition("=")
        if not sep or not name:
            ap.error(f"--config {c!r}: expected NAME=ARGS")
        if name in (n for n, _ in configs):
            ap.error(f"--config {name} given twice")
        configs.append((name, shlex.split(rest)))
    prompt = pathlib.Path(args.prompt_file).read_text() if args.prompt_file else default_prompt_text(args.prompt_chars)
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    lock = None
    if args.lock:
        lock = open(os.path.expanduser(args.lock), "a")
        print(f"waiting for {args.lock} ...", file=sys.stderr)
        fcntl.flock(lock, fcntl.LOCK_EX)

    results = {name: {"prompt": [], **{k: [] for k in DECODE_PROMPTS}} for name, _ in configs}
    for rnd in range(args.rounds):
        for name, cfg_args in configs:
            print(f"round {rnd + 1}/{args.rounds}: {name} {shlex.join(cfg_args)}", file=sys.stderr)
            with serve(args, name, cfg_args, log_dir) as port:
                for k, v in measure(port, args, prompt).items():
                    results[name][k] += v

    base = configs[0][0]
    cols = ["prompt"] + list(DECODE_PROMPTS)
    print(f"\n{args.model}: best / median over {args.rounds} round(s), tok/s, "
          f"change against {base} (the first config) on the median\n")
    print("| config | " + " | ".join(("prompt" if c == "prompt" else f"decode {c}") for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for name, _ in configs:
        cells = []
        for c in cols:
            v = results[name][c]
            med, bmed = statistics.median(v), statistics.median(results[base][c])
            delta = "" if name == base else f" ({(med / bmed - 1) * 100:+.1f}%)"
            cells.append(f"{max(v):.1f} / {med:.1f}{delta}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({
            "model": args.model, "configs": {n: shlex.join(a) for n, a in configs}, "baseline": base,
            "rounds": args.rounds, "prompt_chars": len(prompt), "results": results,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    if lock:
        lock.close()


if __name__ == "__main__":
    main()
