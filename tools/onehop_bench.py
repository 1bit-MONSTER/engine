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
"""One hop from Lemonade, measured (docs/serve.md, "One hop from Lemonade").

Starts a private lemond (its own cache and config directories, its own port; nothing else is
touched), loads one onebit model on HRX, and measures through Lemonade's API and against the
backend llama-server directly:
  - load time; the processes Lemonade runs for the model;
  - /tokenize (no GPU work: the pure cost of the hops) and a 1-token chat, each on a fresh
    connection, via Lemonade and direct;
  - time to first token: a repeated prompt (prompt-cache hit) and fresh prompts of one length;
  - with --arm one, a reload: the model loaded again (pinned) beside a second model, then unloaded
    and loaded, with the time to first token of a prompt length the server has seen before.

usage: tools/onehop_bench.py --arm two|one --out DIR --lemond PATH --onebit PATH
           [--llama-server PATH] [--model NAME] [--second CHECKPOINT] [--port 13405]
--arm two sets LEMONADE_ONEBIT_ONE_HOP=0 (lemond -> 1bit serve -> llama-server), one leaves it on.
Run both arms with the same binaries; the result is <out>/result-<arm>.json.
"""
import argparse, http.client, json, os, signal, statistics, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--arm", choices=["two", "one"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lemond", required=True, help="lemond from the Lemonade fork (1bit/one-hop or later)")
ap.add_argument("--onebit", required=True, help="the 1bit binary (with serve --print-backend)")
ap.add_argument("--llama-server", default="", help="ONEBIT_LLAMA_SERVER for the engine (default: the build's)")
ap.add_argument("--model", default="Qwen3-4B-Instruct-2507-1bit")
ap.add_argument("--second", default="unsloth/Qwen3-0.6B-GGUF:Q4_0", help="checkpoint of the model loaded beside it")
ap.add_argument("--port", type=int, default=13405)
args = ap.parse_args()

ARM, OUT, PORT, MODEL = args.arm, args.out, args.port, args.model
API = f"http://127.0.0.1:{PORT}"
os.makedirs(OUT, exist_ok=True)
res = {"arm": ARM}


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def chat_latency(port, path, prompt, max_tokens=1):
    """non-streamed request on a fresh connection (as Lemonade's client does): total seconds"""
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    t0 = time.perf_counter()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", path, body, {"Content-Type": "application/json"})
    r = c.getresponse(); data = r.read(); c.close()
    dt = time.perf_counter() - t0
    assert r.status == 200, (r.status, data[:300])
    return dt


def ttft(port, path, prompt, max_tokens=8):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    t0 = time.perf_counter()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    c.request("POST", path, body, {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200, r.status
    first = None; text = ""
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            break
        d = json.loads(payload)
        ch = d.get("choices") or [{}]
        delta = ch[0].get("delta", {})
        piece = delta.get("content") or delta.get("reasoning_content")
        if piece and first is None:
            first = time.perf_counter() - t0
        text += piece or ""
    c.close()
    return first, text


def procs():
    out = subprocess.run(["ps", "-eo", "pid,ppid,args", "--no-headers"], capture_output=True, text=True).stdout
    rows = [l.split(None, 2) for l in out.splitlines()]
    return [(int(p), int(pp), a) for p, pp, a in rows if len((p, pp, a)) == 3]


def tree(root):
    ps = procs(); kids = {}
    for p, pp, a in ps:
        kids.setdefault(pp, []).append((p, a))
    out, stack = [], [root]
    while stack:
        x = stack.pop()
        for p, a in kids.get(x, []):
            out.append((p, a)); stack.append(p)
    return out


model_file = ""   # the model's file, from the llama-server command line once loaded


def llama_port(lemond_pid):
    for p, a in tree(lemond_pid):
        if "llama-server" in a and "--port" in a and model_file in a:
            toks = a.split(); return p, int(toks[toks.index("--port") + 1])
    return None, None


def words_prompt(word, n):
    return "Count the words: " + " ".join([word] * n)


env = dict(os.environ)
env.update({"LEMONADE_ONEBIT_BIN": args.onebit, "LEMONADE_ONEBIT_ONE_HOP": "0" if ARM == "two" else "1"})
if args.llama_server:
    env["ONEBIT_LLAMA_SERVER"] = args.llama_server
cfg = os.path.abspath(f"{OUT}/lemond-config-{ARM}")
os.makedirs(cfg, exist_ok=True)
json.dump({"max_loaded_models": 2}, open(f"{cfg}/config.json", "w"))
log = open(f"{OUT}/lemond-{ARM}.log", "w")
lem = subprocess.Popen([args.lemond, os.path.abspath(f"{OUT}/lemond-cache-{ARM}"), cfg, "--port", str(PORT),
                        "--log-file", "disabled", "--no-broadcast"], env=env, stdout=log, stderr=subprocess.STDOUT,
                       start_new_session=True)
try:
    for _ in range(300):
        try:
            get(f"{API}/api/v1/health"); break
        except Exception:
            time.sleep(0.2)
    t0 = time.perf_counter()
    post(f"{API}/api/v1/load", {"model_name": MODEL, "onebit_backend": "hrx", "ctx_size": 8192, "pinned": True})
    res["load_s"] = round(time.perf_counter() - t0, 2)
    tr = tree(lem.pid)
    for _, a in tr:
        if "llama-server" in a and " -m " in a:
            model_file = a.split(" -m ", 1)[1].split()[0]
    res["processes"] = [a[:160] for _, a in tr]
    lp, lport = llama_port(lem.pid)
    res["llama_pid"], res["llama_port"] = lp, lport
    print(json.dumps(res, indent=1), flush=True)

    # 1. a cold graph shape (first prompt of this length), then the same shape with other words
    shape_n = 333
    res["ttft_cold_shape_s"], txt = ttft(PORT, "/api/v1/chat/completions", words_prompt("apple", shape_n))
    res["first_answer"] = txt[:80]
    res["ttft_warm_shape_s"] = [ttft(PORT, "/api/v1/chat/completions", words_prompt(w, shape_n))[0]
                                for w in ("banana", "cherry", "grape", "lemon", "mango")]
    # 2. per-request overhead: tiny prompt, 1 token, fresh connection, via Lemonade vs llama-server direct
    for _ in range(3):
        chat_latency(PORT, "/api/v1/chat/completions", "Hi")
    via, direct = [], []
    for i in range(30):
        via.append(chat_latency(PORT, "/api/v1/chat/completions", "Hi"))
        direct.append(chat_latency(lport, "/v1/chat/completions", "Hi"))
    res["tiny_via_lemonade_ms"] = sorted(round(x * 1000, 2) for x in via)
    res["tiny_direct_llama_ms"] = sorted(round(x * 1000, 2) for x in direct)
    # 2b. pure hop cost: /tokenize does no GPU work (llama-server's HTTP thread answers it)
    def tok(port, path):
        body = json.dumps({"model": MODEL, "content": "The quick brown fox jumps over the lazy dog."}).encode()
        t0 = time.perf_counter()
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request("POST", path, body, {"Content-Type": "application/json"})
        r = c.getresponse(); data = r.read(); c.close()
        assert r.status == 200, (r.status, data[:200])
        return (time.perf_counter() - t0) * 1000
    for _ in range(10):
        tok(PORT, "/api/v1/tokenize"); tok(lport, "/tokenize")
    tv, td = [], []
    for _ in range(200):
        tv.append(tok(PORT, "/api/v1/tokenize")); td.append(tok(lport, "/tokenize"))
    res["tokenize_via_lemonade_ms"] = sorted(round(x, 3) for x in tv)
    res["tokenize_direct_llama_ms"] = sorted(round(x, 3) for x in td)
    # 3. TTFT, warm: same prompt (prompt cache hit) and a fresh prompt of a seen shape, via Lemonade
    res["ttft_same_prompt_s"] = [ttft(PORT, "/api/v1/chat/completions", "Name the capital of France in one word.")[0]
                                 for _ in range(10)]
    res["ttft_fresh_prompt_s"] = [ttft(PORT, "/api/v1/chat/completions", words_prompt(w, shape_n))[0]
                                  for w in ("pear", "plum", "kiwi", "lime", "peach", "melon", "olive", "onion", "carrot", "potato")]
    if ARM == "one":
        # 4. reload: Lemonade asked to load the model again, and a second model loaded beside it
        post(f"{API}/api/v1/load", {"model_name": MODEL, "onebit_backend": "hrx", "ctx_size": 8192, "pinned": True})
        try:
            post(f"{API}/api/v1/pull", {"model_name": "user.onehop-second", "checkpoint": args.second,
                                        "recipe": "onebit"})
            post(f"{API}/api/v1/load", {"model_name": "user.onehop-second", "onebit_backend": "hrx", "ctx_size": 4096})
            res["second_model"] = "loaded"
        except Exception as e:
            res["second_model"] = f"skipped: {e}"
        lp2, _ = llama_port(lem.pid)
        # the first model's llama-server is the one on its old port
        alive = any(p == lp for p, _ in tree(lem.pid))
        res["reload_same_pid"] = alive
        res["ttft_after_reload_seen_shape_s"] = [ttft(PORT, "/api/v1/chat/completions", words_prompt(w, shape_n))[0]
                                                 for w in ("tomato", "garlic", "pepper")]
        # 5. contrast: unload + load starts a new llama-server, its program cache cold
        post(f"{API}/api/v1/unload", {"model_name": MODEL})
        post(f"{API}/api/v1/load", {"model_name": MODEL, "onebit_backend": "hrx", "ctx_size": 8192, "pinned": True})
        res["after_unload_reload_new_pid"] = llama_port(lem.pid)
        res["ttft_after_unload_reload_seen_shape_s"] = [ttft(PORT, "/api/v1/chat/completions", words_prompt(w, shape_n))[0]
                                                        for w in ("radish", "celery", "squash")]
finally:
    left = [p for p, a in tree(lem.pid)]
    os.killpg(lem.pid, signal.SIGTERM)
    try:
        lem.wait(20)
    except Exception:
        os.killpg(lem.pid, signal.SIGKILL)
    time.sleep(2)
    for p in left:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    json.dump(res, open(f"{OUT}/result-{ARM}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
