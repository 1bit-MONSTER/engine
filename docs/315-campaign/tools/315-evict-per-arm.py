#!/usr/bin/env python3
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
        rows.append(dict(arm=base, pid=pid, n=n, verdict=v, gtt=gtt, note=""))

    print(f"{'arm':46} {'pid':>8} {'gtt':>4} {'verdict':>8} {'evict in window':>16}")
    for r in rows:
        n = "n/a" if r["n"] is None else str(r["n"])
        print(f"{r['arm'][:46]:46} {str(r['pid'] or '-'):>8} {r['gtt']:>3}G {r['verdict']:>8} {n:>16}"
              + (f"   {r['note']}" if r["note"] else ""))

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
        print(f"  observed mean: faulting {fm:.1f} vs clean {cm:.1f}  ->  "
              f"{'CONSISTENT with H2' if fm > cm * 1.5 else 'NOT supportive (faulting <= 1.5x clean)'}")
    else:
        print("\n  both a faulting and a clean in-regime arm are needed for the comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
