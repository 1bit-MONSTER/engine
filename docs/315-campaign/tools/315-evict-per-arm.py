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
"""engine#315: per-arm eviction counts, faulting vs clean - the H2 discriminating test.

WHY THIS TOOL EXISTS. The mechanism candidate (H2) is that a wave preempted mid-execution has
its buffers evicted underneath it. Its discriminating prediction is: **faulting arms show more
evictions in their own arm window than clean arms at the same GTT**.

That test was attempted on run 7's existing data and found INVALID (see
evidence/315-pid-capture.md): an arm server keeps being evicted after its arm ends, so its total
is a function of post-arm survival, and the long-lived neighbours cannot be told apart from arm
servers after the fact. It therefore REQUIRES the per-arm PID captured at arm start, which the
driver now writes to $O/server.pid.

HOW THE WINDOW IS BOUNDED. For each arm:
  start = that PID's PROC_START boot_ns in the KFD SMI stream
  end   = that PID's PROC_END, or the next arm's PROC_START, whichever comes first
Only Q_EVICT rows for THAT pid inside THAT window are counted. This is what makes the count
attributable to the arm rather than to the process's whole life.

THE CONFOUND, AND WHY THERE IS ALSO A PRE-FAULT COUNT. A whole-window total cannot tell
"more evictions caused the fault" from "the fault caused more evictions": a faulting arm's server
crashes, may be retried, and lives on after the failure, so its window can simply be longer. The
fix is to count evictions in the window only up to the moment the fault actually happened, which
the fault cannot have caused. server.log timestamps the fault line, e.g.

    0.07.279.709 E common_sampler_sample: HRX returned NaN logits: ...

in minutes.seconds.milliseconds.microseconds, i.e. 7.28 s into the arm (the `pre-fault` column).
Both counts are printed: `pre-fault` is the causal test, `window` is kept for continuity with the
earlier analysis and because a large gap between them is itself informative.

    ./315-evict-per-arm.py <outdir> <console.log> --smi <smi.tsv> [--tag TAG] [--gate 95]

<outdir> is the driver's arm root (e.g. ~/wt/rt-det), and each arm directory must contain
server.pid (written by 315-paired-pidcap.sh) and detector.json.
"""
import argparse
import glob
import json
import os
import re
import sys

# `0.07.279.709 E common_sampler_sample: HRX returned NaN logits` -> the fault time.
# Fields are minutes.seconds.milliseconds.microseconds (verified against arm req counts and
# against the listening-on-http line landing ~5.6 s in).
TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\s")
# BOTH faces must be matched, or a Face-B arm looks like it has no recoverable fault time.
# Face A: the HRX NaN logits line.  Face B: the chat-parse failure that surfaces as HTTP 500
# ("unparsed Content-only output"). Whichever comes first in the log wins, which is the first
# fault the arm actually suffered.
FAULT = re.compile(r"HRX returned NaN logits|NaN logits|unparsed Content-only|"
                   r"does not match the expected")


def fault_time_s(dirpath):
    """Seconds from arm start to the first recorded fault, or None if none is timestamped.

    Returns (seconds, raw) so the caller can show what it matched rather than trusting a number.
    """
    path = os.path.join(dirpath, "server.log")
    if not os.path.exists(path):
        return None, None
    with open(path, errors="replace") as fh:
        for line in fh:
            if not FAULT.search(line):
                continue
            m = TS.match(line)
            if not m:
                return None, line.strip()[:120]
            mi, s, ms, us = (int(g) for g in m.groups())
            return mi * 60 + s + ms / 1e3 + us / 1e6, line.strip()[:120]
    return None, None


def load_smi(path):
    """Return {pid: (start_ns, end_ns)} for llama-server, and a list of Q_EVICT rows."""
    procs, evicts = {}, []
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            try:
                ns, pid = int(p[0]), int(p[4])
            except ValueError:
                continue
            ev = p[2]
            if ev == "PROC_START":
                procs[pid] = [ns, None]
            elif ev == "PROC_END" and pid in procs:
                procs[pid][1] = ns
            elif ev == "Q_EVICT":
                evicts.append((ns, pid))
    return procs, evicts


def verdicts(console, gate):
    """{pair,cell -> (verdict, gtt)} from the console's per-arm lines."""
    out = {}
    pat = re.compile(r"pair\s+(\d+)\s+arm\s+([cl]):\s+(OK|FAULT|OOR|INVALID).*?gtt_start=(\d+)G")
    for line in open(console, errors="replace"):
        m = pat.search(line)
        if m:
            out[(int(m.group(1)), m.group(2))] = (m.group(3), int(m.group(4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("console")
    ap.add_argument("--smi", required=True)
    ap.add_argument("--gate", type=int, default=95)
    ap.add_argument("--tag", default="",
                    help="restrict to arm dirs for one campaign tag (strongly recommended: "
                         "the arm root holds every past run, and mixing tags mixes recipes)")
    a = ap.parse_args()

    pat = f"{a.tag}-p[0-9]*[cl]-*" if a.tag else "*-p[0-9]*[cl]-*"
    dirs = sorted(glob.glob(os.path.join(a.outdir, pat)))
    if not dirs:
        print(f"no arm directories under {a.outdir}", file=sys.stderr)
        return 1

    procs, evicts = load_smi(a.smi)
    verd = verdicts(a.console, a.gate)

    # Order arms by their PID's PROC_START so the window can be bounded by the next arm.
    arms = []
    for d in dirs:
        pidf = os.path.join(d, "server.pid")
        if not os.path.exists(pidf):
            arms.append((d, None, None))
            continue
        try:
            pid = int(open(pidf).read().split()[0])
        except (ValueError, IndexError):
            arms.append((d, None, None))
            continue
        st, en = procs.get(pid, (None, None))
        arms.append((d, pid, st))
    arms.sort(key=lambda t: (t[2] is None, t[2] or 0))

    rows = []
    for i, (d, pid, st) in enumerate(arms):
        base = os.path.basename(d)
        m = re.search(r"-p(\d+)([cl])-", base)
        key = (int(m.group(1)), m.group(2)) if m else None
        v, gtt = verd.get(key, ("?", 0))
        if pid is None or st is None:
            rows.append(dict(arm=base, pid=pid, n=None, verdict=v, gtt=gtt, note="no PID captured"))
            continue
        # window end: this pid's PROC_END, else the next arm's start
        en = procs[pid][1]
        if en is None:
            nxt = next((s for (_, p2, s) in arms[i + 1:] if s and s > st), None)
            en = nxt if nxt else (max((ns for ns, _ in evicts), default=st) + 1)
        n = sum(1 for ns, ep in evicts if ep == pid and st <= ns < en)
        # pre-fault count: the causal test. A fault cannot cause evictions that precede it.
        fsec, raw = fault_time_s(d)
        pre = None
        if fsec is not None:
            dur = (en - st) / 1e9
            if fsec > dur + 1.0:
                note = f"fault time {fsec:.1f}s exceeds window {dur:.1f}s - NOT used"
                pre, fsec = None, None
            else:
                pre = sum(1 for ns, ep in evicts if ep == pid and st <= ns < st + int(fsec * 1e9))
                note = f"fault at {fsec:.1f}s"
        else:
            note = ""
        rows.append(dict(arm=base, pid=pid, n=n, pre=pre, fault_s=fsec, verdict=v, gtt=gtt,
                         note=note))

    print(f"{'arm':46} {'pid':>8} {'gtt':>4} {'verdict':>8} {'pre-fault':>9} {'window':>7}")
    for r in rows:
        n = "n/a" if r["n"] is None else str(r["n"])
        pre = "" if r.get("pre") is None else str(r["pre"])
        print(f"{r['arm'][:46]:46} {str(r['pid'] or '-'):>8} {r['gtt']:>3}G {r['verdict']:>8} "
              f"{pre:>9} {n:>7}" + (f"   {r['note']}" if r["note"] else ""))

    inreg = [r for r in rows if r["n"] is not None and r["gtt"] >= a.gate]
    fault = [r for r in inreg if r["verdict"] == "FAULT"]
    clean = [r for r in inreg if r["verdict"] == "OK"]
    print()
    if not inreg:
        print("no in-regime arms with a captured PID - the comparison cannot be made.")
        print("(apply the PID capture BEFORE the run: see evidence/315-pid-capture.md)")
        return 1
    for label, sel in (("FAULT", fault), ("clean", clean)):
        if sel:
            vals = sorted(r["n"] for r in sel)
            mean = sum(vals) / len(vals)
            print(f"  {label:5} n={len(sel):2}  evictions/arm: min={vals[0]} median={vals[len(vals)//2]} "
                  f"max={vals[-1]} mean={mean:.1f}")
    if fault and clean:
        fm = sum(r['n'] for r in fault) / len(fault)
        cm = sum(r['n'] for r in clean) / len(clean)
        print(f"\n  H2 prediction: faulting arms are evicted MORE than clean arms.")
        print(f"  window total, mean: faulting {fm:.1f} vs clean {cm:.1f}  ->  "
              f"{'CONSISTENT with H2' if fm > cm * 1.5 else 'NOT supportive (faulting <= 1.5x clean)'}")
        pf = [r for r in fault if r.get("pre") is not None]
        if pf:
            pfm = sum(r['pre'] for r in pf) / len(pf)
            pc = [r['pre'] for r in clean if r.get("pre") is None]
            print(f"  pre-fault, mean:    faulting {pfm:.1f} ({len(pf)} of {len(fault)} fault arms had a "
                  f"recoverable fault time)")
            print(f"  >>> the pre-fault figure is the causal test: a fault cannot cause evictions")
            print(f"      that precede it. Compare it against the clean arms' FULL windows")
            print(f"      (mean {cm:.1f}), since a clean arm has no fault to truncate at - it is a")
            print(f"      conservative comparison that favours NOT finding an effect.")
        else:
            print(f"  pre-fault: no fault arm had a recoverable fault time; only the confounded")
            print(f"  window total is available. Do not draw a causal conclusion from it.")
    else:
        print("\n  both a faulting and a clean in-regime arm are needed for the comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
