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
"""The Hyperloom 1.2.0 Inkling-on-Loom workload, end to end without the box (tools/hyperloom/inkling-loom).

One Hyperloom round is: custom_radeon8060s.sh validates the levers, ships box_inkling.sh + gauntlet.py over ssh, runs
ONE `box-hold gpu sudo -n timeout ... env <levers> bash box_inkling.sh` ticket, fetches the output and lets hl_score.py
write inferencex_result.json. Here `ssh` runs the remote command locally, `box-hold` / `sudo` / `timeout` are pass-through
shims that record they were used, and loom-serve is a fake OpenAI server that streams fixed greedy replies. Checked:

  * a baseline round writes a passing report with the five scored medians and establishes the text reference;
  * a lever reaches the box as the overlay model.json's draft_depth, and through box-hold (never around it);
  * a lever outside its allowed values is refused BEFORE anything crosses ssh, as a failing report (exit 0);
  * a model that never registers is a failing report, not a crash and not a pass.

Run directly (`python3 tests/hyperloom_inkling_loom_test.py`) or through ctest.
"""
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
WL = ROOT / "tools/hyperloom/inkling-loom"

FAKE_SERVE = r'''#!/usr/bin/env python3
import json, sys, http.server, os
listen = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--listen=")][0]
model = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")][0]
open(os.environ["FAKE_ARGS"], "w").write(json.dumps({"argv": sys.argv, "model_json": json.load(open(model))}))
if os.environ.get("FAKE_NEVER_REGISTER"):
    import time; time.sleep(3); sys.exit(1)
host, port = listen.rsplit(":", 1)
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"data": [{"id": "Inkling-2-Mini"}]}).encode() if self.path == "/v1/models" else b"{}"
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = req["messages"][0]["content"]
        pt = max(1, len(content.split()) * 4 // 3)
        n = min(req["max_tokens"], 6)
        text = ("COBALT-FALCON-7429" if "vault" in content else "391" if "17 * 23" in content else "The user wants x")
        pieces = [text] + [" t%d" % i for i in range(n - 1)]
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        import time
        for p in pieces:
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": p}}]}) + "\n\n").encode()); self.wfile.flush()
            time.sleep(0.002)
        self.wfile.write(("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": n}}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
http.server.ThreadingHTTPServer((host, int(port)), H).serve_forever()
'''


def write_exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def setup(td):
    bindir = td / "bin"
    bindir.mkdir()
    log = td / "shims.log"
    # ssh HOST [-o ...] CMD: run CMD locally. box-hold/sudo/timeout: log and pass through.
    write_exe(bindir / "ssh", '#!/bin/bash\nwhile [ "${1:-}" = -o ]; do shift 2; done\nshift\nexec bash -c "$*"\n')
    write_exe(bindir / "box-hold", '#!/bin/bash\necho "box-hold $1" >> %s\nshift\nexec "$@"\n' % log)
    write_exe(bindir / "sudo", '#!/bin/bash\n[ "$1" = -n ] && shift\necho "sudo" >> %s\nexec "$@"\n' % log)
    card = td / "card"
    (card / "parameter").mkdir(parents=True)
    (card / "CARD_REVISION").write_text("a8354e77test\n")
    (card / "parameter" / "m.gguf").write_text("weights")
    (card / "model.json").write_text(json.dumps({"presets": {"chat": {"execution": {
        "rows": 4, "context_capacity": 262144, "prefill_capacity": 4096, "max_tokens": 4096, "draft_depth": 3}}}}))
    serve = td / "loom-serve"
    write_exe(serve, FAKE_SERVE)
    env = dict(os.environ, PATH="%s:%s" % (bindir, os.environ["PATH"]), HL_BOX="fakebox", HL_BOX_DIR=str(td / "box"),
               HL_BOX_ROOT=str(td / "box"), HL_CARD=str(card), HL_LOOM_SERVE=str(serve), HL_STATE=str(td / "state"),
               HL_RAM_NEED_GB="0", HL_RAM_FLOOR_GB="0", HL_RUNS="1", HL_OUT_OWNER=str(os.getuid()), FAKE_ARGS=str(td / "args.json"))
    return env, log


def round_(env, td, name, **levers):
    rd = td / name
    rd.mkdir()
    e = dict(env, RESULT_DIR=str(rd), RESULT_FILENAME="inferencex_result", **levers)
    p = subprocess.run(["bash", str(WL / "custom_radeon8060s.sh")], env=e, capture_output=True, text=True, timeout=300)
    rep = rd / "inferencex_result.json"
    return p, (json.loads(rep.read_text()) if rep.exists() else None), rd


def main():
    fails = []

    def check(name, cond, detail=""):
        if not cond:
            fails.append("%s %s" % (name, detail))

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        env, log = setup(td)

        p, rep, rd = round_(env, td, "baseline")
        check("baseline exits 0", p.returncode == 0, p.stderr[-800:])
        check("baseline writes a report", rep is not None, p.stdout[-800:] + p.stderr[-800:])
        if rep:
            g = rep["quality_gate"]
            check("baseline passes the gate", g["passed"], json.dumps(g)[:600] + (rd / "box/status").read_text())
            check("baseline establishes the text reference",
                  g.get("halves", {}).get("text", {}).get("reference", "").startswith("established"))
            check("baseline has a positive score", rep["output_throughput"] > 0, str(rep.get("bench_summary", {}).get("metrics")))
            check("the report says no KLD was measured", g.get("kld") is None and "logits" in g.get("kld_note", ""))
            check("provenance names the card revision",
                  rep["bench_summary"]["provenance"]["card_revision"] == "a8354e77test")
        shims = log.read_text() if log.exists() else ""
        check("the measurement ran through box-hold gpu", "box-hold gpu" in shims, shims)

        p, rep, rd = round_(env, td, "depth5", HL_DRAFT_DEPTH="5")
        args = json.loads((td / "args.json").read_text())
        check("the lever reaches the served model.json",
              args["model_json"]["presets"]["chat"]["execution"]["draft_depth"] == 5, json.dumps(args)[:400])
        check("the candidate compares against the reference", rep and rep["quality_gate"]["halves"]["text"].get("agree") == 1.0,
              json.dumps(rep and rep["quality_gate"])[:400])
        check("its report records the requested lever", rep and rep["bench_summary"]["levers_requested"].get("HL_DRAFT_DEPTH") == "5")

        before = log.read_text().count("box-hold")
        p, rep, rd = round_(env, td, "refused", HL_DRAFT_DEPTH="3; rm -rf /")
        check("a bad lever value exits 0 with a failing report", p.returncode == 0 and rep and not rep["quality_gate"]["passed"],
              p.stderr[-400:])
        check("a bad lever never reaches the box", log.read_text().count("box-hold") == before
              and not (rd / "box" / "ticket.log").exists())

        p, rep, rd = round_(dict(env, FAKE_NEVER_REGISTER="1"), td, "noreg")
        check("a model that never registers is a failing report", p.returncode == 0 and rep and not rep["quality_gate"]["passed"]
              and rep["output_throughput"] == 0.0, json.dumps(rep)[:400])
    for f in fails:
        print("FAIL:", f)
    print("hyperloom_inkling_loom_test: %d failed" % len(fails) if fails else "hyperloom_inkling_loom_test: ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
