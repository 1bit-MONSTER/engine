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
"""Host-side checks of the HRX hardware gate's judges (docs/hw-gate.md), no GPU needed:
tools/kat.py against a fake OpenAI server (pass, cache-path near-tie = WARN, cache-path wrong
answer = FAIL, nondeterminism = FAIL, golden drift = FAIL, a case the golden also missed = not a
FAIL) and tools/hw_gate_manifest.py (KLD verdicts, placement verdicts, manifest verification)."""
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAT = os.path.join(ROOT, "tools", "kat.py")
MAN = os.path.join(ROOT, "tools", "hw_gate_manifest.py")

STATE = {"mode": "good", "n": 0}


def answer(prompt, cached):
    if "capital of France" in prompt:
        return ["Paris", "."]
    if "17 * 23" in prompt:
        return ["391"]
    if "def add" in prompt or "Python function" in prompt:
        return ["def add", "(a, b):", " return a + b"]
    if "color is the box" in prompt:
        return ["green"]
    if STATE["mode"] == "golden-missed" and "secret code" in prompt and len(prompt) < 3000:
        return ["21"]           # a small model missing the short needles, at the release pin too
    if STATE["mode"] == "cache-wrong" and cached and "secret code" in prompt:
        return ["21"]
    if STATE["mode"] == "cache-tie" and cached and "secret code" in prompt:
        return ["7341", " was", " it"]
    if STATE["mode"] == "flaky" and "secret code" in prompt:
        STATE["n"] += 1
        return ["7341"] if STATE["n"] % 2 else ["7341", "!"]
    if STATE["mode"] == "drift" and "secret code" in prompt:
        return ["The", " code", " is", " 7341"]
    return ["7341", " is", " the", " code"] if "secret code" in prompt else ["?"]


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        toks = answer(prompt, body.get("cache_prompt", True))
        # top-2 margin 0.3 at token 1 (a near-tie), 5.0 elsewhere
        content = [{"token": t, "logprob": -0.1,
                    "top_logprobs": [{"logprob": -0.1}, {"logprob": -0.4 if i == 1 else -5.1}]}
                   for i, t in enumerate(toks)]
        out = {"choices": [{"message": {"content": "".join(toks)}, "logprobs": {"content": content}}],
               "usage": {"prompt_tokens": len(prompt) // 4}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def kat(url, *args):
    p = subprocess.run([sys.executable, KAT, *args, "--url", url, "--repeat", "2", "--needle-words", "600"],
                       capture_output=True, text=True)
    return p.returncode, p.stdout


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    bad = []
    with tempfile.TemporaryDirectory() as d:
        gold = os.path.join(d, "g.json")
        STATE["mode"] = "good"
        rc, out = kat(url, "run", "--out", gold)
        if rc != 0 or "warn" in out:
            bad.append("good server should pass with no warning:\n" + out)
        rc, out = kat(url, "check", "--golden", gold)
        if rc != 0:
            bad.append("check against its own golden should pass:\n" + out)
        STATE["mode"] = "cache-tie"
        rc, out = kat(url, "check", "--golden", gold)
        if rc != 0 or "WARN cache path" not in out:
            bad.append("a cached run diverging at a near-tie is a WARN, not a FAIL:\n" + out)
        STATE["mode"] = "cache-wrong"
        rc, out = kat(url, "check", "--golden", gold)
        if rc != 1 or "misses the answer the fresh run gave" not in out:
            bad.append("a cached run missing the fresh run's answer must FAIL:\n" + out)
        STATE["mode"] = "flaky"
        rc, out = kat(url, "run")
        if rc != 1 or "nondeterministic" not in out:
            bad.append("different texts for the same fresh request must FAIL:\n" + out)
        STATE["mode"] = "drift"
        rc, out = kat(url, "check", "--golden", gold)
        if rc != 1 or "diverges from golden" not in out:
            bad.append("text that leaves the golden early must FAIL:\n" + out)
        STATE["mode"] = "golden-missed"
        gold2 = os.path.join(d, "g2.json")
        kat(url, "run", "--out", gold2)
        rc, out = kat(url, "check", "--golden", gold2)
        if rc != 0 or "so did the golden" not in out:
            bad.append("a case the golden also missed must not FAIL on the expected answer:\n" + out)

        # KLD verdicts
        man = os.path.join(d, "manifest.json")
        with open(man, "w") as f:
            json.dump({"kld": {"m": {"limits": {"max_mean_kld": 0.01, "max_kld_p99": 0.2, "min_same_top_p": 95.0}}}}, f)
        for mean, p99, top, want in ((0.0, 0.0, 100.0, 0), (0.002, 0.05, 99.1, 0), (0.08, 0.9, 91.0, 1)):
            log = os.path.join(d, "ppl.log")
            with open(log, "w") as f:
                f.write(f"Mean    KLD: {mean:10.6f} ±   0.000010\nMaximum KLD:   1.0\n99.0%   KLD: {p99:10.6f}\n"
                        f"Same top p: {top:6.3f} ± 0.100 %\n")
            rc = subprocess.run([sys.executable, MAN, "judge", man, "m", log], capture_output=True).returncode
            if rc != want:
                bad.append(f"judge mean={mean} p99={p99} top={top}: rc {rc}, want {want}")
        with open(os.path.join(d, "ppl.log"), "w") as f:
            f.write("main: llama_decode() failed\n")
        if subprocess.run([sys.executable, MAN, "judge", man, "m", os.path.join(d, "ppl.log")],
                          capture_output=True).returncode != 1:
            bad.append("a log without KLD statistics must fail")

        # placement verdicts (engine #390's report; the gate fails on any node off HRX)
        clean = "placement_observer: placement: 512 nodes, 0 off HRX (none), 1 graph splits\n"
        off = "placement_observer: placement: 354 nodes, 20 off HRX (MUL_MAT x 20), 5 graph splits\n"
        abort = ('ggml-hrx.cpp:444: placement_observer: GGML_HRX_STRICT=1: "attn_out-0" is MUL_MAT f32 '
                 'ne=[1024,7,1,1] and is not placed on HRX; 1 of 512 nodes off HRX (MUL_MAT x 1), 3 splits\n')
        for name, text, want, says in (("clean", clean * 3, 0, "all on HRX in 3 graphs"),
                                       ("worst graph", clean + off + clean, 1, "20 of 354 nodes off HRX"),
                                       ("no report", "srv  load_model: loaded\n", 1, "no placement report"),
                                       ("strict abort", clean + abort, 1, "attn_out-0")):
            log = os.path.join(d, "server.log")
            with open(log, "w") as f:
                f.write(text)
            p = subprocess.run([sys.executable, MAN, "placement", log], capture_output=True, text=True)
            if p.returncode != want or says not in p.stdout:
                bad.append(f"placement {name}: rc {p.returncode}, want {want}; said {p.stdout.strip()!r}")

        # manifest set / verify
        base, model, text = (os.path.join(d, n) for n in ("b.kld", "m.gguf", "t.txt"))
        for p, c in ((base, b"base"), (model, b"GGUF" * 100), (text, b"text")):
            with open(p, "wb") as f:
                f.write(c)
        man2 = os.path.join(d, "m2.json")
        subprocess.run([sys.executable, MAN, "set", man2, "m", base, model, text], check=True)
        if subprocess.run([sys.executable, MAN, "verify", man2, "m", base, model, text], capture_output=True).returncode != 0:
            bad.append("verify must accept what set recorded")
        with open(base, "ab") as f:
            f.write(b"x")
        if subprocess.run([sys.executable, MAN, "verify", man2, "m", base, model, text], capture_output=True).returncode != 1:
            bad.append("verify must refuse a changed base")
    srv.shutdown()
    for b in bad:
        print("FAIL:", b)
    print("hw_gate_test:", "FAIL" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
