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
"""forward-serve --spec-lookup returns the plain loop's reply, token for token (engine#1781).

usage: tests/serve_spec_lookup_test.py <1bit binary>        (needs numpy)

The speculative path (app/forward_serve.cpp, `spec`) decodes a greedy request through mtp::run_from() with the
prompt-lookup drafter. It is only allowed where it cannot change a token, and this holds it to that end to end: the
two-layer qwen3 container tests/batch_check_state_ci.py fabricates, a 256-token byte-level tokenizer written here, and
`1bit forward-serve` in host-GEMM mode (NPU_INFER_HOST_GEMM=1: no device is opened, no lock is taken) started twice,
plain and with --spec-lookup 4. The same id prompts go to both: several budgets, stop sequences cut out of the plain
replies (so they do end their replies), and one seeded sampled request. Text, finish_reason and completion_tokens
must be equal; the spec server must take the speculative path on exactly the greedy requests and draft (its `[spec]`
lines).

A dev-box row: forward-serve is built only with ONEBIT_NPU. Exits 77 (skip) without numpy.
"""
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "safetensors_to_q4nx.py")


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def write_tokenizer(d):
    m = bytes_to_unicode()
    vocab = {m[b]: b for b in range(256)}
    tok = {"version": "1.0", "added_tokens": [],
           "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
           "decoder": {"type": "ByteLevel"},
           "model": {"type": "BPE", "vocab": vocab, "merges": []}}
    with open(os.path.join(d, "tokenizer.json"), "w") as f:
        json.dump(tok, f)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def start(binary, model, port, extra, log):
    env = dict(os.environ, NPU_INFER_HOST_GEMM="1", OMP_WAIT_POLICY="PASSIVE")
    p = subprocess.Popen([binary, "forward-serve", "-m", os.path.join(model, "model.q4nx"), "--kernels",
                          os.path.join(model, "kernels"), "-p", str(port)] + extra,
                         env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(600):
        if p.poll() is not None:
            return p
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1).read()
            return p
        except Exception:
            time.sleep(0.1)
    return p


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2])
        return 2
    try:
        import numpy as np
    except ImportError:
        print("SKIP: numpy is not installed")
        return 77
    spec = importlib.util.spec_from_file_location("bcs", os.path.join(HERE, "batch_check_state_ci.py"))
    bcs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bcs)
    fails = 0
    with tempfile.TemporaryDirectory(prefix="serve_spec_lookup.", dir=bcs.tmp_root()) as tmp:
        src, out = os.path.join(tmp, "hf"), os.path.join(tmp, "q4nx")
        bcs.make_model(src, np, "qwen3")
        r = subprocess.run([sys.executable, TOOL, "pack", src, out], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL  packing the fabricated container: {r.stderr[-600:]}")
            return 1
        write_tokenizer(out)
        prompts = [[40, 41, 42, 43, 40, 41, 42, 43, 40, 41], list(range(60, 90)), [7, 7, 7, 7, 7, 7], [200, 3, 200, 3, 200]]
        base = [{"prompt": p, "max_tokens": n, "temperature": 0} for p in prompts for n in (1, 5, 24)]
        # A sampled request (seeded, so both servers draw the same tokens) must stay on the plain loop.
        sampled = [{"prompt": prompts[1], "max_tokens": 16, "temperature": 0.8, "seed": 7}]
        replies, logs, cases = {}, {}, []
        for name, extra in (("plain", []), ("spec", ["--spec-lookup", "4"])):
            port = free_port()
            logp = os.path.join(tmp, name + ".log")
            with open(logp, "w") as log:
                p = start(sys.argv[1], out, port, extra, log)
                try:
                    if p.poll() is not None:
                        print(f"FAIL  {name} server exited: " + open(logp).read()[-800:])
                        return 1
                    if name == "plain":
                        got = [post(port, b) for b in base]
                        # Stop sequences that occur: a piece of the plain reply itself, taken a few characters in, so
                        # each one cuts its reply mid-way (finish "stop") and the hold-back runs on both paths.
                        stops = []
                        for j, b in zip(got, base):
                            t = j["choices"][0]["text"]
                            # printable ASCII only: a replacement character (a split UTF-8 byte) never matches
                            at = next((i for i in range(3, len(t) - 1) if t[i:i + 2].isascii() and t[i:i + 2].isprintable()), -1)
                            if b["max_tokens"] == 24 and at >= 0:
                                stops.append({"prompt": b["prompt"], "max_tokens": 24, "temperature": 0,
                                              "stop": [t[at:at + 2]]})
                        cases = base + stops + sampled
                    replies[name] = []
                    for b in cases:
                        j = post(port, b)
                        c = j["choices"][0]
                        replies[name].append((c["text"], c["finish_reason"], j["usage"]["completion_tokens"]))
                finally:
                    p.terminate()
                    try:
                        p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
            logs[name] = open(logp).read()
        for b, a, c in zip(cases, replies["plain"], replies["spec"]):
            ok = a == c
            fails += not ok
            what = (f"prompt {prompts.index(b['prompt'])} max_tokens {b['max_tokens']}"
                    + (f" stop {b['stop']!r}" if "stop" in b else "")
                    + (f" temperature {b['temperature']}" if b["temperature"] else ""))
            print(f"{'PASS' if ok else 'FAIL'}  {what}: {c[2]} tokens, finish {c[1]}"
                  + ("" if ok else f" (plain {a!r}, spec {c!r})"))
        n_stop = sum(1 for b, c in zip(cases, replies["spec"]) if "stop" in b and c[1] == "stop")
        ok = n_stop >= 2
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {n_stop} stop-sequence requests ended on their stop sequence")
        spec_lines = [l for l in logs["spec"].splitlines() if l.startswith("[spec]")]
        drafted = sum(int(l.split(" of ")[1].split()[0]) for l in spec_lines)
        accepted = sum(int(l.split(" drafts accepted")[0].split(", ")[-1].split(" of ")[0]) for l in spec_lines)
        greedy = sum(1 for b in cases if not b["temperature"])
        ok = len(spec_lines) == greedy and drafted > 0 and "[spec]" not in logs["plain"]
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  the spec server took the speculative path on {len(spec_lines)} requests, the "
              f"{greedy} greedy ones ({accepted} of {drafted} drafts accepted); the sampled one and the plain server never")
    print("ALL PASS" if not fails else f"FAILED {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
