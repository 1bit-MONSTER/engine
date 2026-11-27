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
"""The Hyperloom 1.2.0 box-ticket workloads without the box (tools/hyperloom/box_ticket.sh, inkling-loom/, custom/).

One Hyperloom round is: custom_radeon8060s.sh validates the levers, ships box_inkling.sh + gauntlet.py over ssh, runs
ONE `box-hold gpu sudo -n timeout ... env <levers> bash box_inkling.sh` ticket, fetches the output and lets hl_score.py
write inferencex_result.json. Here `ssh` runs the remote command locally, `box-hold` / `sudo` / `timeout` are pass-through
shims that record they were used, and loom-serve is a fake OpenAI server that streams fixed greedy replies. Checked:

  * a baseline round writes a passing report with the five scored medians and establishes the text reference;
  * a lever reaches the box as the overlay model.json's draft_depth, and through box-hold (never around it);
  * a lever outside its allowed values is refused BEFORE anything crosses ssh, as a failing report (exit 0);
  * a box without the temperature sensor (CI: no k10temp) still writes provenance (its fields are null);
  * a model that never registers is a failing report, not a crash and not a pass; when loom-serve refuses the
    registration (it stays up), its own reason reaches the round's status;
  * box_gate.sh (the restricted key's forced command): a full Inkling round passes through it once the box side is
    allowlisted; it refuses a shell, a ship of new scripts, a non-allowlisted or tampered rev, and a name outside its list;
  * a round the box cannot serve is refused before any ticket: server args from a model proposal (EXTRA_CUSTOM_ARGS)
    and a patched framework checkout each give a failing report naming why; a clean checkout still measures;
  * an unreachable box (ssh exits 255 every time) is retried, then the round exits 75 with NO report -- infrastructure,
    never a graded measurement;
  * a ticket box-hold never runs (the per-lane queue cap or a queue timeout, exit 4) also exits 75 with NO report;
  * the box's RAM guard stops the SERVER while the gauntlet is still running: MemAvailable (a fake /proc/meminfo,
    HL_MEMINFO) drops under the floor mid-gauntlet and the server's process group is gone within seconds, not after the
    gauntlet's foreground python returns (bash defers a trapped signal until then);
  * custom/ (the HRX route): a bad lever is refused before ssh, a ticket above the fleet's 45-minute cap is refused,
    and the box side refuses a job whose declared RAM need does not fit -- each as a failing report.

Run directly (`python3 tests/hyperloom_box_ticket_test.py`) or through ctest.
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
        fail = os.environ.get("FAKE_REG_FAIL")
        if fail and self.path == "/v1/status":
            body = json.dumps({"data": [{"id": "Inkling-2-Mini", "loom": {"registration": "failed",
                                "error": {"code": 3, "message": "card/program/llm.loom:67719:1: " + fail}}}]},
                              separators=(",", ":")).encode()   # loom-serve's own compact JSON
        elif fail:
            body = json.dumps({"data": []}).encode()
        else:
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
            time.sleep(float(os.environ.get("FAKE_TOKEN_SLEEP", "0.002")))
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
        "rows": 4, "context_capacity": 262144, "prefill_capacity": 1024, "max_tokens": 4096, "draft_depth": 3}}}}))
    serve = td / "loom-serve"
    write_exe(serve, FAKE_SERVE)
    env = dict(os.environ, PATH="%s:%s" % (bindir, os.environ["PATH"]), HL_BOX="fakebox", HL_BOX_DIR=str(td / "box"),
               HL_BOX_ROOT=str(td / "box"), HL_CARD=str(card), HL_LOOM_SERVE=str(serve), HL_STATE=str(td / "state"),
               HL_RAM_NEED_GB="0", HL_RAM_FLOOR_GB="0", HL_HWMON_ROOT=str(td / "no-hwmon"), HL_RUNS="1", HL_OUT_OWNER=str(os.getuid()), FAKE_ARGS=str(td / "args.json"))
    return env, log


def round_(env, td, name, wl=WL, **levers):
    rd = td / name
    rd.mkdir()
    e = dict(env, RESULT_DIR=str(rd), RESULT_FILENAME="inferencex_result", **levers)
    p = subprocess.run(["bash", str(wl / "custom_radeon8060s.sh")], env=e, capture_output=True, text=True, timeout=300)
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
        if not (rep and rep.get("quality_gate", {}).get("passed")):
            for f in ("status", "ticket.log", "ram.txt", "gauntlet.log", "serve.log"):
                q = rd / "box" / f
                print("== baseline box/%s: %s" % (f, q.read_text()[-800:] if q.exists() else "(absent)"))
            print("== entrypoint stdout/stderr:", p.stdout[-800:], p.stderr[-800:])
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
                  rep["bench_summary"].get("provenance", {}).get("card_revision") == "a8354e77test",
                  "status: %s" % rep["bench_summary"].get("status"))
        shims = log.read_text() if log.exists() else ""
        check("the measurement ran through box-hold gpu", "box-hold gpu" in shims, shims)
        # unset levers keep the CARD's preset (the shipped route serves it as it is), and no spin by default (#3395)
        args = json.loads((td / "args.json").read_text())
        ex = args["model_json"]["presets"]["chat"]["execution"]
        check("an unset lever keeps the card's own value (prefill_capacity 1024, not a script default)",
              ex["prefill_capacity"] == 1024 and ex["draft_depth"] == 3 and ex["rows"] == 4, json.dumps(ex))
        dev = [a for a in args["argv"] if a.startswith("--device=")]
        check("the default device is plain amdgpu:// (no completion spin), as the shipped routes",
              dev == ["--device=amdgpu://"], str(dev))
        if rep:
            check("provenance records the effective preset values",
                  rep["bench_summary"]["provenance"]["levers"]["prefill_capacity"] == 1024, json.dumps(rep["bench_summary"]["provenance"]["levers"]))

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
        before_cn = log.read_text().count("box-hold")
        p, rep, rd = round_(env, td, "bad-card-name", HL_CARD_NAME="../../etc")
        check("a card name with a path in it is refused before the box", p.returncode == 0 and rep
              and not rep["quality_gate"]["passed"] and log.read_text().count("box-hold") == before_cn, p.stderr[-300:])
        check("a bad lever never reaches the box", log.read_text().count("box-hold") == before
              and not (rd / "box" / "ticket.log").exists())

        p, rep, rd = round_(dict(env, FAKE_NEVER_REGISTER="1"), td, "noreg")
        check("a model that never registers is a failing report", p.returncode == 0 and rep and not rep["quality_gate"]["passed"]
              and rep["output_throughput"] == 0.0, json.dumps(rep)[:400])
        # the RAM guard during the gauntlet: the server must go when MemAvailable drops, not when python returns
        import time
        mi = td / "meminfo"
        mi.write_text("MemAvailable:   104857600 kB\n")   # 100 GB at the start: passes need 0 + floor 16
        rd = td / "ramguard"
        rd.mkdir()
        e = dict(env, RESULT_DIR=str(rd), RESULT_FILENAME="inferencex_result", HL_MEMINFO=str(mi), HL_RAM_FLOOR_GB="16",
                 FAKE_TOKEN_SLEEP="0.4")
        t0 = time.time()
        gp = subprocess.Popen(["bash", str(WL / "custom_radeon8060s.sh")], env=e, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        spid = None
        while time.time() - t0 < 90 and spid is None:
            for f in (td / "box" / "out").glob("*/serve.pid"):
                if f.stat().st_mtime >= t0 - 1:
                    try:
                        spid = int(f.read_text().strip())
                    except ValueError:
                        pass
            time.sleep(0.2)
        check("ram guard: the server started", spid is not None)
        alive = lambda: os.path.exists("/proc/%d" % spid) and "zombie" not in open("/proc/%d/status" % spid).read().lower()
        if spid:
            while time.time() - t0 < 90 and not (td / "box" / "out").glob("*/gauntlet.log"):
                time.sleep(0.2)
            time.sleep(2)                                       # well into the gauntlet's foreground python
            mi.write_text("MemAvailable:    1048576 kB\n")      # 1 GB: under the 16 GB floor
            t1 = time.time()
            while time.time() - t1 < 30 and alive():
                time.sleep(0.2)
            gone = time.time() - t1
            check("ram guard: the server is stopped within 10 s of the drop (not after the gauntlet)", gone < 10,
                  "server still alive after %.1f s" % gone)
        try:
            gp.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            gp.kill()
            check("ram guard: the round ends", False)
        ram = (rd / "box" / "ram.txt").read_text() if (rd / "box" / "ram.txt").exists() else ""
        check("ram guard: ram.txt records the guard", "RAM-GUARD" in ram, ram[-300:])

        # box_gate.sh: the same round, every ssh command routed through the gate as sshd's command= would
        gate = ROOT / "tools/hyperloom/box_gate.sh"
        gbin = td / "gbin"
        gbin.mkdir()
        write_exe(gbin / "ssh", '#!/bin/bash\nwhile [ "${1:-}" = -o ]; do shift 2; done\nshift\n'
                  'SSH_ORIGINAL_COMMAND="$*" exec bash %s\n' % gate)
        genv = dict(env, PATH="%s:%s" % (gbin, env["PATH"]), HL_GATE_ROOT=env["HL_BOX_DIR"],
                    HL_GATE_DIR=str(td / "gate"))
        (td / "gate").mkdir()
        p, rep, rd = round_(genv, td, "gate-unlisted")
        check("gate: a ship of a rev that is not allowlisted is refused", p.returncode != 0 and "refused" in p.stderr,
              p.stderr[-300:])
        rev = [d.name for d in (td / "box" / "hl").iterdir() if (d / "box_inkling.sh").exists()][0]
        (td / "gate" / "allowed-revs").write_text(rev + "\n")
        p, rep, rd = round_(genv, td, "gate-round", HL_DRAFT_DEPTH="4")
        check("gate: an allowlisted round runs end to end through the gate", p.returncode == 0 and rep
              and rep["quality_gate"]["passed"], p.stderr[-400:] + json.dumps(rep)[:300])
        glog = (td / "gate" / "gate.log").read_text()
        check("gate: the round used ship(no-op), ticket and fetch", all(w in glog for w in ("ship(no-op)", "ticket", "fetch")),
              glog[-500:])

        def gated(cmd):
            return subprocess.run(["bash", str(gate)], env=dict(genv, SSH_ORIGINAL_COMMAND=cmd), capture_output=True,
                                  text=True, timeout=60)
        bd = env["HL_BOX_DIR"]
        tk = "hl-20261123T000000Z-1"
        base = ("mkdir -p %s/out/%s && BOX_HOLD_WHO='cl-hyperloom %s' BOX_HOLD_WAIT=10 box-hold gpu sudo -n timeout -k 30 60 "
                "env %%s bash %s/hl/%%s/box_inkling.sh %s/out/%s" % (bd, tk, tk, bd, bd, tk))
        check("gate: a shell is refused", gated("bash -i").returncode == 126)
        check("gate: chained commands are refused", gated("tar -C %s/out/%s -cf - . 2>/dev/null; id" % (bd, tk)).returncode == 126)
        check("gate: a name outside the list is refused",
              gated(base % ("HL_LOOM_SERVE=/bin/sh", rev)).returncode == 126)
        check("gate: a ticket above 45 minutes is refused",
              gated(base.replace("timeout -k 30 60", "timeout -k 30 3600") % ("HL_RUNS=1", rev)).returncode == 126)
        check("gate: a rev that is not allowlisted is refused", gated(base % ("HL_RUNS=1", "0" * 12)).returncode == 126)
        with open(td / "box" / "hl" / rev / "box_inkling.sh", "a") as f:
            f.write("\n# tampered\n")
        check("gate: a tampered allowlisted rev is refused", gated(base % ("HL_RUNS=1", rev)).returncode == 126)

        # rounds the box cannot serve: refused before any ticket, as a failing report that says why
        before = log.read_text().count("box-hold")
        p, rep, rd = round_(env, td, "server-args", EXTRA_CUSTOM_ARGS="--model /x/model-d0.json --preset chat")
        check("server args are refused, not measured", p.returncode == 0 and rep and not rep["quality_gate"]["passed"]
              and "EXTRA_CUSTOM_ARGS" in rep["quality_gate"]["reason"] and log.read_text().count("box-hold") == before,
              json.dumps(rep)[:300])
        repo = td / "fw"
        repo.mkdir()
        git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                                        check=True, capture_output=True)
        git("init", "-q")
        (repo / "f.txt").write_text("base\n")
        git("add", "f.txt")
        git("commit", "-q", "-m", "base")
        p, rep, rd = round_(env, td, "clean-repo", FRAMEWORK_REPO_PATH=str(repo))
        check("a clean framework checkout still measures", rep and rep["quality_gate"]["passed"], json.dumps(rep)[:300])
        (repo / "f.txt").write_text("patched by a specialist\n")
        before = log.read_text().count("box-hold")
        p, rep, rd = round_(env, td, "patched-repo", FRAMEWORK_REPO_PATH=str(repo))
        check("a patched framework checkout is refused, not measured", p.returncode == 0 and rep
              and not rep["quality_gate"]["passed"] and "uncommitted changes" in rep["quality_gate"]["reason"]
              and log.read_text().count("box-hold") == before, json.dumps(rep)[:300])

        msg = "SYMBOL/008: symbol '@llm.tokens' value 16384 of type index violates constraint 'range'"
        p, rep, rd = round_(dict(env, FAKE_REG_FAIL=msg), td, "regfail", HL_PREFILL_CAP="16384")
        st = (rd / "box" / "status").read_text()
        check("a refused registration names loom-serve's reason", rep and not rep["quality_gate"]["passed"]
              and "violates constraint 'range'" in st and "violates" in json.dumps(rep), st)

        # an unreachable box: ssh always fails with 255 (the 23 Nov NIC loss)
        dbin = td / "deadbin"
        dbin.mkdir()
        write_exe(dbin / "ssh", '#!/bin/bash\necho "ssh: connect to host fakebox port 22: No route to host" >&2\nexit 255\n')
        p, rep, rd = round_(dict(env, PATH="%s:%s" % (dbin, env["PATH"]), HL_SSH_RETRY_SEC="0"), td, "unreachable")
        check("an unreachable box: retried, then exit 75 and no report", p.returncode == 75 and rep is None
              and p.stderr.count("retry") == 2 and "unreachable" in p.stderr, "rc=%s rep=%s %s" % (p.returncode, rep, p.stderr[-300:]))

        # box-hold never runs the ticket (27 Nov: the per-lane queue cap, #3579, refuses with exit 4; a BOX_HOLD_WAIT
        # timeout exits 4 too): unmeasured, exit 75 and no report -- never graded as the lever failing its gate
        cbin = td / "capbin"
        cbin.mkdir()
        write_exe(cbin / "box-hold", '#!/bin/bash\necho "box-hold: lane cap: 3 queued for cl-hyperloom (at most 3 per lane): test refused, not queued" >&2\nexit 4\n')
        p, rep, rd = round_(dict(env, PATH="%s:%s" % (cbin, env["PATH"])), td, "lanecap")
        check("a ticket box-hold refuses: exit 75 and no report", p.returncode == 75 and rep is None
              and "box-hold did not run the ticket (rc 4" in p.stderr and "lane cap" in p.stderr,
              "rc=%s rep=%s %s" % (p.returncode, rep, p.stderr[-300:]))

        # reachable for the ship, lost for the ticket: the ticket's own retry path
        write_exe(dbin / "ssh", '#!/bin/bash\nwhile [ "${1:-}" = -o ]; do shift 2; done\nshift\n'
                  'case "$*" in *box-hold*) echo "Timeout, server fakebox not responding." >&2; exit 255;; esac\n'
                  'exec bash -c "$*"\n')
        p, rep, rd = round_(dict(env, PATH="%s:%s" % (dbin, env["PATH"]), HL_SSH_RETRY_SEC="0"), td, "lost-mid-ticket")
        check("box lost during the ticket: retried, then exit 75 and no report", p.returncode == 75 and rep is None
              and (rd / "box" / "ticket.log").read_text().count("retry") == 2, "rc=%s %s" % (p.returncode, p.stderr[-300:]))

        # custom/: the HRX route over the same ticket library
        hrx = ROOT / "tools/hyperloom/custom"
        henv = dict(env, MODEL="/data/dev/x/Qwen3.8-27B-UD-Q4_K_XL.gguf")
        henv.pop("HL_RAM_NEED_GB")
        before = log.read_text().count("box-hold")
        p, rep, rd = round_(henv, td, "hrx-bad", wl=hrx, HL_UB="1024 -m /etc/passwd")
        check("custom/: a bad lever is a failing report and no ticket", p.returncode == 0 and rep
              and not rep["quality_gate"]["passed"] and log.read_text().count("box-hold") == before, p.stderr[-300:])
        p, rep, rd = round_(henv, td, "hrx-long", wl=hrx, HL_TICKET_MIN="60")
        check("custom/: a ticket above 45 minutes is refused", p.returncode != 0 and "45" in p.stderr, p.stderr[-300:])
        p, rep, rd = round_(henv, td, "hrx-ram", wl=hrx, HL_RAM_NEED_GB="999")
        check("custom/: the box refuses a RAM need that does not fit, as a failing report",
              p.returncode == 0 and rep and not rep["quality_gate"]["passed"]
              and "RAM need 999" in (rd / "box" / "status").read_text(), json.dumps(rep)[:300])
        check("custom/: the refused job still went through box-hold", log.read_text().count("box-hold") == before + 1)
    for f in fails:
        print("FAIL:", f)
    print("hyperloom_box_ticket_test: %d failed" % len(fails) if fails else "hyperloom_box_ticket_test: ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
