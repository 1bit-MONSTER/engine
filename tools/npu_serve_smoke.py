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
"""npu_serve_smoke: does `1bit serve --device npu` serve a small model on the FAST route, at speed, with an answer?

The release gate's check for engine#2106 and engine#2102. A serve that falls back to the slow per-op forward still
answers, so "did it answer" passes on the wrong route (1.9 tok/s measured against 87-101). This asserts the four things
the fallback fails, as ~/briefs/DEMO-SMOKE.md specified them:

  route        the serve log names the fast route (and does not name the per-op forward)
  rate         median decode rate >= --floor tok/s (default 60: the two routes are 45x apart, so this is not a tolerance)
  first-token  median prompt time <= --first-token-ms (default 600)
  content      every reply has non-empty content that is not one character repeated

  npu-hold -l npu-smoke -t 10 -- tools/npu_serve_smoke.py --bin <1bit> --model <model dir> [--out <dir>]

It starts the serve, waits for it, sends --replies requests (default 3) with thinking off, stops the serve, and prints
one line per assertion and `RESULT: PASS` or `RESULT: FAIL` (exit 0 / 1). It takes no device ticket itself: run it
under npu-hold. The serve ends with this script however the script ends: a TERM, HUP or INT stops the serve first
(exit 128 + signal), and the kernel TERMs the serve if the script is killed outright. Logs go to --out (default
~/.cache/1bit/npu-serve-smoke), never to /tmp.

  tools/npu_serve_smoke.py --self-test       the judging on recorded shapes of each failure; no device, no binary
  tools/npu_serve_smoke.py --judge <dir>     judge a finished run's serve.log and reply_*.json again
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

FAST_ROUTE = "on the private route dx"
SLOW_ROUTE = "per-op forward"
PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number.",
    "What is 17 times 23? Reply with the number only.",
]


def decode_rate(reply):
    """tok/s of one reply from the server's own timings, or None when it reports none."""
    t = reply.get("timings") or {}
    if isinstance(t.get("predicted_per_second"), (int, float)) and t["predicted_per_second"] > 0:
        return float(t["predicted_per_second"])
    n = (reply.get("usage") or {}).get("completion_tokens")
    ms = t.get("predicted_ms")
    if isinstance(n, int) and n > 0 and isinstance(ms, (int, float)) and ms > 0:
        return 1000.0 * n / ms
    return None


def content_of(reply):
    try:
        c = reply["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        return None
    return c if isinstance(c, str) else None


def degenerate(text):
    """One character repeated (the '//////' symptom of engine#1810), ignoring whitespace."""
    s = "".join(text.split())
    return len(s) >= 4 and len(set(s)) == 1


def judge(log_text, replies, floor=60.0, first_token_ms=600.0, fast_route=FAST_ROUTE, slow_route=SLOW_ROUTE):
    """[(name, ok, detail)] for the four assertions. Pure: a log and the replies in, verdicts out."""
    out = []
    fast, slow = fast_route in log_text, slow_route in log_text
    if fast and not slow:
        out.append(("route", True, "the serve log names the fast route"))
    elif slow:
        out.append(("route", False, "the serve log names the %s: this run is on the slow route" % slow_route))
    else:
        out.append(("route", False, "the serve log never says '%s'" % fast_route))

    rates = [decode_rate(r) for r in replies]
    if not replies or any(r is None for r in rates):
        out.append(("rate", False, "%d of %d replies carry no decode timing" % (sum(r is None for r in rates), len(replies))))
    else:
        med = statistics.median(rates)
        out.append(("rate", med >= floor, "median %.1f tok/s (floor %.1f; each: %s)" % (med, floor, ", ".join("%.1f" % r for r in rates))))

    firsts = [(r.get("timings") or {}).get("prompt_ms") for r in replies]
    if not replies or any(not isinstance(f, (int, float)) for f in firsts):
        out.append(("first-token", False, "a reply carries no prompt_ms"))
    else:
        med = statistics.median(firsts)
        out.append(("first-token", med <= first_token_ms, "median %.0f ms (limit %.0f)" % (med, first_token_ms)))

    bad = []
    for i, r in enumerate(replies):
        c = content_of(r)
        if c is None or not c.strip():
            bad.append("reply %d: empty content%s" % (i, " (the tokens went to reasoning_content: is thinking off?)"
                                                      if (r.get("choices") or [{}])[0].get("message", {}).get("reasoning_content") else ""))
        elif degenerate(c):
            bad.append("reply %d: one character repeated (%r)" % (i, c[:12]))
    if not replies:
        bad.append("no replies")
    out.append(("content", not bad, "; ".join(bad) if bad else "%d replies, all with content" % len(replies)))
    return out


def report(verdicts):
    ok = all(v[1] for v in verdicts)
    for name, good, detail in verdicts:
        print("%-12s %s  %s" % (name, "ok  " if good else "FAIL", detail))
    print("RESULT: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def die_with_parent():
    """In the serve's process, before exec: ask the kernel to TERM it if this script dies, however it dies."""
    try:
        import ctypes
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG
    except Exception:
        pass
    if os.getppid() == 1:   # the script was already gone when this ran
        os._exit(1)


def run(args):
    # A ticket that ends TERMs this script, not the serve (the serve is this script's child, and in the gate a flock
    # sits between the ticket and the script). Python's default action for SIGTERM ends the process without running
    # `finally`, which is the only code that stops the serve -- so a hung run would leave `1bit serve` holding the NPU
    # after its ticket (review of #2125). The handlers turn the signal into SystemExit, which unwinds through it.
    def stop(signum, _frame):
        sys.exit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "serve.log")
    cmd = [args.bin, "serve", "-m", args.model, "--device", "npu", "--host", "127.0.0.1", "-p", str(args.port)]
    replies = []
    with open(log_path, "wb") as log:
        serve = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, preexec_fn=die_with_parent)
        try:
            deadline = time.time() + args.ready_seconds
            ready = False
            while time.time() < deadline and serve.poll() is None:
                try:
                    with urllib.request.urlopen("http://127.0.0.1:%d/health" % args.port, timeout=2) as r:
                        if r.status == 200:
                            ready = True
                            break
                except OSError:
                    time.sleep(0.5)
            if ready:
                for i in range(args.replies):
                    prompt = PROMPTS[i % len(PROMPTS)]
                    body = json.dumps({"model": "m", "max_tokens": args.max_tokens, "temperature": 0,
                                       "chat_template_kwargs": {"enable_thinking": False},
                                       "messages": [{"role": "user", "content": prompt}]}).encode()
                    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % args.port, data=body,
                                                 headers={"Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=args.request_seconds) as r:
                            reply = json.loads(r.read().decode())
                    except (OSError, ValueError) as e:
                        reply = {"error": str(e)}
                    with open(os.path.join(out, "reply_%d.json" % i), "w") as f:
                        json.dump(reply, f)
                    replies.append(reply)
        finally:
            # the serve must not outlive this script (and so not outlive the ticket it runs under)
            if serve.poll() is None:
                serve.send_signal(signal.SIGTERM)
                try:
                    serve.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    serve.kill()
                    serve.wait()
    with open(log_path, errors="replace") as f:
        log_text = f.read()
    if not replies:
        print("serve       FAIL  it never became ready within %d s (exit %s); see %s" % (args.ready_seconds, serve.returncode, log_path))
    return report(judge(log_text, replies, args.floor, args.first_token_ms, args.fast_route, args.slow_route))


def rejudge(args):
    d = os.path.expanduser(args.judge)
    with open(os.path.join(d, "serve.log"), errors="replace") as f:
        log_text = f.read()
    replies = []
    i = 0
    while os.path.exists(os.path.join(d, "reply_%d.json" % i)):
        with open(os.path.join(d, "reply_%d.json" % i)) as f:
            replies.append(json.load(f))
        i += 1
    return report(judge(log_text, replies, args.floor, args.first_token_ms, args.fast_route, args.slow_route))


FAKE_SERVE = r'''#!/usr/bin/env python3
# A stand-in for `1bit serve`, written by npu_serve_smoke.py --self-test: it prints a route line, answers /health and
# /v1/chat/completions with the timings FAKE_TPS gives, and exits on SIGTERM. FAKE_MODE=dead exits before listening.
import http.server, json, os, signal, sys
port = int(sys.argv[sys.argv.index("-p") + 1])
mode, tps = os.environ.get("FAKE_MODE", "fast"), float(os.environ.get("FAKE_TPS", "90"))
if mode == "dead":
    print("1bit serve: cannot open the device", flush=True); sys.exit(3)
print("1bit unified: on the private route dx: dense Qwen3 on the dx kernels" if mode in ("fast", "stall") else
      "1bit serve: a model-generic container: serving it through the per-op forward", flush=True)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self): self._send({"status": "ok"})
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if mode == "stall":
            open(os.environ["FAKE_MARK"] + ".asked", "w").write("asked")
            import time; time.sleep(600)
        self._send({"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Paris."}}],
                    "timings": {"prompt_ms": 200.0 if mode == "fast" else 1100.0, "predicted_ms": 1000.0 * 8 / tps, "predicted_per_second": tps},
                    "usage": {"completion_tokens": 8}})
srv = http.server.HTTPServer(("127.0.0.1", port), H)
def bye(*a):
    if os.environ.get("FAKE_MARK"): open(os.environ["FAKE_MARK"], "w").write("terminated")
    os._exit(0)
signal.signal(signal.SIGTERM, bye)
srv.serve_forever()
'''


def self_test_run(fails):
    """run() end to end against the stand-in: it starts the serve, waits, asks, stops it, and returns the verdict."""
    import contextlib
    import io
    import shutil
    import socket
    import tempfile
    d = tempfile.mkdtemp(prefix="npu_serve_smoke_selftest.", dir=os.getcwd())   # the working directory, not /tmp
    try:
        fake = os.path.join(d, "fake1bit")
        with open(fake, "w") as f:
            f.write(FAKE_SERVE)
        os.chmod(fake, 0o700)
        for mode, tps, want_exit, want_line in (("fast", "90", 0, "RESULT: PASS"), ("slow", "1.9", 1, "route        FAIL"),
                                                ("dead", "90", 1, "it never became ready")):
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            mark = os.path.join(d, "mark." + mode)
            os.environ.update(FAKE_MODE=mode, FAKE_TPS=tps, FAKE_MARK=mark)
            args = argparse.Namespace(bin=fake, model="model-dir", out=os.path.join(d, mode), port=port, floor=60.0, first_token_ms=600.0,
                                      fast_route=FAST_ROUTE, slow_route=SLOW_ROUTE, max_tokens=8, ready_seconds=5 if mode == "dead" else 30,
                                      request_seconds=30, replies=3)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = run(args)
            text = buf.getvalue()
            if rc != want_exit or want_line not in text:
                fails.append("run() against a %s stand-in: exit %d, output %r" % (mode, rc, text[-300:]))
            if mode != "dead" and not os.path.exists(mark):
                fails.append("run() against a %s stand-in left the serve running (no SIGTERM seen)" % mode)
            if mode == "fast" and not os.path.exists(os.path.join(d, mode, "reply_2.json")):
                fails.append("run() did not keep the three replies")
        # The ticket-ends case: this script is TERMed while a request hangs. The serve must be stopped too, and the
        # script must exit 128 + SIGTERM. Run as a real process, because the signal's default action is the bug.
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        mark = os.path.join(d, "mark.stall")
        env = dict(os.environ, FAKE_MODE="stall", FAKE_TPS="90", FAKE_MARK=mark)
        tool = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--bin", fake, "--model", "model-dir", "--out",
                                 os.path.join(d, "stall"), "--port", str(port), "--request-seconds", "300"],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while time.time() < deadline and not os.path.exists(mark + ".asked"):
            time.sleep(0.1)
        if not os.path.exists(mark + ".asked"):
            fails.append("the stalling stand-in was never asked (the run did not reach its first request)")
        tool.send_signal(signal.SIGTERM)
        try:
            rc = tool.wait(timeout=40)
        except subprocess.TimeoutExpired:
            tool.kill()
            rc = None
        if rc != 128 + signal.SIGTERM:
            fails.append("a TERMed run exits %r, want %d" % (rc, 128 + signal.SIGTERM))
        deadline = time.time() + 10
        while time.time() < deadline and not os.path.exists(mark):
            time.sleep(0.1)
        if not os.path.exists(mark):
            fails.append("a TERMed run left the serve running: the stand-in never saw its SIGTERM")
    finally:
        for k in ("FAKE_MODE", "FAKE_TPS", "FAKE_MARK"):
            os.environ.pop(k, None)
        subprocess.run(["pkill", "-f", fake], check=False)   # nothing of this test's may outlive it, whatever failed
        shutil.rmtree(d, ignore_errors=True)


def self_test():
    fails = []

    def reply(content, tps, prompt_ms=220.0, reasoning=None, n=20):
        m = {"role": "assistant", "content": content}
        if reasoning is not None:
            m["reasoning_content"] = reasoning
        return {"choices": [{"index": 0, "finish_reason": "stop", "message": m}],
                "timings": {"prompt_ms": prompt_ms, "predicted_ms": 1000.0 * n / tps, "predicted_per_second": tps},
                "usage": {"completion_tokens": n}}

    def expect(name, verdicts, want):
        got = {v[0]: v[1] for v in verdicts}
        if got != want:
            fails.append("%s: got %s, want %s" % (name, got, want))

    fast_log = "1bit unified: on the private route dx: dense Qwen3 on the dx kernels, one design per layer (/x/1024x3072x16x8)\n"
    slow_log = "1bit unified: loading on the NPU\n[forward] decode family: 0 role(s) at M=16, 4 on M=128\n1bit serve: per-op forward\n"
    good = [reply("The capital of France is Paris.", 101.4, 224), reply("def fib(n): ...", 87.5, 225), reply("391", 93.0, 263)]
    all_ok = {"route": True, "rate": True, "first-token": True, "content": True}

    # the measured demo row passes
    expect("the measured fast-route row", judge(fast_log, good), all_ok)
    # engine#2102 as it was seen: the slow route answers correctly, at 1.9 tok/s. Three of four assertions catch it.
    slow = [reply("The capital of France is Paris.", 1.9, 4100), reply("def fib(n): ...", 1.87, 4200), reply("391", 1.9, 4000)]
    expect("the silent fallback", judge(slow_log, slow), {"route": False, "rate": False, "first-token": False, "content": True})
    # a log that names neither route is not a pass: the evidence has to be there
    expect("a log that names no route", judge("1bit serve: ready\n", good), dict(all_ok, route=False))
    # ... and a log that names BOTH (a retry that fell back) is the slow route
    expect("a log naming both routes", judge(fast_log + slow_log, good), dict(all_ok, route=False))
    # the fast route with thinking left on: every token in reasoning_content, content empty
    thinking = [reply("", 95.0, 220, reasoning="Okay, the user asks ...") for _ in range(3)]
    expect("thinking left on", judge(fast_log, thinking), dict(all_ok, content=False))
    # one degenerate reply among good ones (engine#1810's '//////')
    expect("one degenerate reply", judge(fast_log, good[:2] + [reply("//////", 90.0)]), dict(all_ok, content=False))
    # the rate is the median: one slow reply of three does not fail it, two do
    expect("one slow reply of three", judge(fast_log, [good[0], good[1], reply("391", 20.0)]), all_ok)
    # ... and it IS the median: (70, 70, 1.9) has a mean of 47, under the floor, and still passes
    expect("the median, not the mean", judge(fast_log, [reply("a b", 70.0), reply("c d", 70.0), reply("391", 1.9)]), all_ok)
    expect("two slow replies of three", judge(fast_log, [good[0], reply("x y", 20.0), reply("391", 20.0)]), dict(all_ok, rate=False))
    # just under and at the floor
    expect("just under the floor", judge(fast_log, [reply("a b", 59.9)] * 3), dict(all_ok, rate=False))
    expect("at the floor", judge(fast_log, [reply("a b", 60.0)] * 3), all_ok)
    # a slow first token alone
    expect("slow first token", judge(fast_log, [reply("a b", 90.0, 900.0)] * 3), dict(all_ok, **{"first-token": False}))
    # replies without timings cannot pass the rate or the first-token assertion
    bare = [{"choices": [{"message": {"content": "Paris."}}]}] * 3
    expect("no timings", judge(fast_log, bare), {"route": True, "rate": False, "first-token": False, "content": True})
    # timings without predicted_per_second: the rate comes from tokens and milliseconds
    derived = [{"choices": [{"message": {"content": "Paris."}}], "timings": {"prompt_ms": 200.0, "predicted_ms": 100.0}, "usage": {"completion_tokens": 9}}] * 3
    expect("rate from tokens / ms", judge(fast_log, derived), all_ok)
    # an error reply, and no replies at all (the serve never came up)
    expect("an error reply", judge(fast_log, good[:2] + [{"error": "timed out"}]), {"route": True, "rate": False, "first-token": False, "content": False})
    expect("no replies", judge(fast_log, []), {"route": True, "rate": False, "first-token": False, "content": False})
    if not degenerate("//////") or degenerate("391") or degenerate("aa") or not degenerate("a a a a"):
        fails.append("degenerate(): wrong on a fixture")

    self_test_run(fails)
    for f in fails:
        print("FAIL " + f)
    print("npu_serve_smoke self-test: %d case(s) failed" % len(fails) if fails
          else "npu_serve_smoke self-test: 16 judging cases, 3 runs against a stand-in serve and a TERM mid-request ok")
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--judge", metavar="DIR")
    p.add_argument("--write-stand-in", metavar="PATH", help="write the self-test's stand-in serve to PATH (for testing a caller)")
    p.add_argument("--bin")
    p.add_argument("--model")
    p.add_argument("--out", default="~/.cache/1bit/npu-serve-smoke")
    p.add_argument("--port", type=int, default=18173)
    p.add_argument("--floor", type=float, default=60.0)
    p.add_argument("--first-token-ms", type=float, default=600.0)
    p.add_argument("--fast-route", default=FAST_ROUTE, help="the text the serve log prints on the fast route")
    p.add_argument("--slow-route", default=SLOW_ROUTE, help="the text the serve log prints on the slow route")
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--replies", type=int, default=3, help="how many requests to send (the three prompts, repeated); the rate is their median")
    p.add_argument("--ready-seconds", type=int, default=120)
    p.add_argument("--request-seconds", type=int, default=120)
    a = p.parse_args()
    if a.write_stand_in:
        with open(a.write_stand_in, "w") as f:
            f.write(FAKE_SERVE)
        os.chmod(a.write_stand_in, 0o700)
        return 0
    if a.self_test:
        return self_test()
    if a.judge:
        return rejudge(a)
    if not a.bin or not a.model:
        p.error("give --bin and --model (or --self-test, or --judge <dir>)")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
