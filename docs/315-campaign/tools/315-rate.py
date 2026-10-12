#!/usr/bin/env python3
"""engine#315: compute the campaign rate, exact CI and the power statement from a console.

Criterion 3 requires ">=60 arms, every arm inside the declared regime gate, zero faults across
both faces, reported with CI and an explicit power statement (60 clean arms => ~95% confidence
the true rate is <5%)". This produces exactly that, from the driver's own per-arm lines, so the
numbers cannot be transcribed by hand.

    ./315-rate.py <console.log> [--gate 95]

Reads lines of the form the arm driver writes:
  pair <N> arm <c|l>: <OK|FAULT|OOR|INVALID> nan=<n> req=<n> ret3=<n> parse=<n> gtt_start=<G>G ...

Counting rules, taken from the frozen pre-registration:
  * only arms with gtt_start >= <gate> count (out-of-regime arms are reported, never counted)
  * BOTH faces count as faults: Face A = nan>0, Face B = parse>0 or utf8>0
  * an arm that faults on either face is a fault; an arm with neither and verdict OK is clean
  * OOR and INVALID are neither clean nor faulted and are excluded from the rate

The console line is NOT sufficient on its own to certify zero faults: it carries nan=, req=,
parse= and utf8= but NOT the detector's http_failures or transport_failures. A Face-B fault that
surfaces as HTTP 500 therefore leaves no trace in the console and a console-only count would report
a FALSE clean arm on the very criterion it must certify. So when --det points at the arm root, every
arm's detector.json is read and the detector is treated as authoritative for faults: any arm the
console calls clean but the detector calls failed is counted as a fault and named "MISSED". An arm
with no detector.json is reported as unverifiable rather than assumed clean.
"""
import argparse
import json
import os
import re
import sys
from math import comb

LINE = re.compile(
    r"(?:pair\s+(\d+)\s+)?arm\s+([cl]):\s+(OK|FAULT|OOR|INVALID)\s+nan=(\d+)\s+req=(\d+)\s+"
    r"ret3=\d+\s+parse=(\d+)\s+gtt_start=(\d+)G(?:\s+utf8=(\d+))?"
)

ARM = re.compile(r"-p(\d+)([cl])-")

# detector fields that mean a failure; http_failures/transport_failures are the ones the
# console cannot see at all.
DET_FIELDS = (("parse_failures", "parse"), ("utf8_failures", "utf8"),
              ("http_failures", "http"), ("transport_failures", "transport"))


def detector_verdicts(arm_root, tag):
    """{(pair, cell): {parse,utf8,http,transport,failed,error}} for every arm dir present.

    Enumerates the whole arm root rather than sampling: a missed arm here is a missed fault.
    """
    out = {}
    if not arm_root or not os.path.isdir(arm_root):
        return out
    for name in sorted(os.listdir(arm_root)):
        if tag and not name.startswith(tag):
            continue
        m = ARM.search(name)
        if not m:
            continue
        key = (int(m.group(1)), m.group(2))
        path = os.path.join(arm_root, name, "detector.json")
        if not os.path.exists(path):
            out[key] = dict(error="no detector.json", failed=None)
            continue
        try:
            d = json.load(open(path))
        except Exception as e:
            out[key] = dict(error=f"unreadable: {e}", failed=None)
            continue
        rec = {short: int(d.get(field) or 0) for field, short in DET_FIELDS}
        rec["failed"] = any(rec[s] > 0 for _, s in DET_FIELDS) or d.get("first_failure") is not None
        rec["error"] = None
        out[key] = rec
    return out


def parse(path, gate):
    rows = []
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        pair, cell, verdict, nan, req, parse, gtt, utf8 = m.groups()
        rows.append(dict(pair=int(pair) if pair else None, cell=cell, verdict=verdict,
                         nan=int(nan), req=int(req), parse=int(parse), gtt=int(gtt),
                         utf8=int(utf8 or 0)))
    in_reg = [r for r in rows if r["gtt"] >= gate]
    oor = [r for r in rows if r["gtt"] < gate]
    def is_fault(r):
        return r["nan"] > 0 or r["parse"] > 0 or r["utf8"] > 0
    clean = [r for r in in_reg if r["verdict"] == "OK" and not is_fault(r)]
    fault = [r for r in in_reg if is_fault(r) or r["verdict"] == "FAULT"]
    invalid = [r for r in in_reg if r["verdict"] == "INVALID" and not is_fault(r)]
    return rows, in_reg, oor, clean, fault, invalid


def reconcile(clean, fault, dets, tag):
    """Move arms the detector failed but the console passed from clean to fault.

    Returns (clean, fault, missed, unverified). The detector is authoritative: its http and
    transport failure counters exist only in detector.json, so the console is blind to them.
    """
    if not dets:
        return clean, fault, [], []
    missed, unverified, keep = [], [], []
    by_key = {(r["pair"], r["cell"]): r for r in clean}
    for key, r in by_key.items():
        d = dets.get(key)
        if d is None:
            unverified.append(r)
        elif d.get("failed") is True:
            missed.append((r, d))
        elif d.get("failed") is None:
            unverified.append(r)
        else:
            keep.append(r)
    keep = [r for r in clean if (r["pair"], r["cell"]) not in by_key or dets.get((r["pair"], r["cell"]), {}).get("failed") is False]
    return keep, fault + [r for r, _ in missed], missed, unverified


def cp_upper(k, n, alpha=0.05):
    """Exact Clopper-Pearson one-sided upper bound, by bisection (no scipy)."""
    if n == 0:
        return float("nan")
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        tail = sum(comb(n, i) * mid ** i * (1 - mid) ** (n - i) for i in range(k + 1))
        if tail > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("console")
    ap.add_argument("--gate", type=int, default=95)
    ap.add_argument("--det", default=os.path.expanduser("~/wt/rt-det"),
                    help="arm root; each arm's detector.json is read and the detector is "
                         "treated as authoritative for faults. Pass '' to disable.")
    ap.add_argument("--tag", default="",
                    help="only reconcile arms whose directory name starts with this")
    a = ap.parse_args()
    rows, in_reg, oor, clean, fault, invalid = parse(a.console, a.gate)

    dets = detector_verdicts(a.det, a.tag) if a.det else {}
    clean, fault, missed, unverified = reconcile(clean, fault, dets, a.tag)

    print(f"console : {a.console}")
    print(f"gate    : gtt_start >= {a.gate} GiB   (the frozen pre-registration's gate)")
    print()
    print(f"  arms parsed          {len(rows)}")
    print(f"  in-regime            {len(in_reg)}")
    print(f"  out-of-regime (OOR)  {len(oor)}   reported, NOT counted")
    print(f"  clean                {len(clean)}")
    print(f"  FAULT                {len(fault)}")
    print(f"  INVALID              {len(invalid)}")
    if dets:
        print(f"  detector.json read   {len(dets)} arm(s) under {a.det}"
              f"{f' (tag {a.tag})' if a.tag else ''}")
        if missed:
            print(f"  !! MISSED BY CONSOLE  {len(missed)} arm(s) the console passed but the")
            print(f"     detector failed -- these are counted as faults (http/transport are not in the")
            print(f"     console line at all):")
            for r, d in missed[:10]:
                print(f"       pair {r['pair']} cell {r['cell']} "
                      f"http={d.get('http', 0)} transport={d.get('transport', 0)} "
                      f"parse={d.get('parse', 0)} utf8={d.get('utf8', 0)}")
        if unverified:
            print(f"  UNVERIFIED           {len(unverified)} clean arm(s) with no usable")
            print(f"     detector.json -- reported as clean from the console alone; NOT certified")
    else:
        print("  WARNING: no detector.json read (--det disabled or root absent); a Face-B fault")
        print("  surfacing as HTTP 500 is invisible to the console and would be counted CLEAN.")
    print()

    n, k = len(clean) + len(fault), len(fault)
    if n == 0:
        print("rate: undefined (no in-regime scored arms)")
        return 1

    faces_a = [r for r in fault if r["nan"] > 0]
    faces_b = [r for r in fault if r["parse"] > 0 or r["utf8"] > 0]
    unattr = [r for r in fault if not (r["nan"] > 0 or r["parse"] > 0 or r["utf8"] > 0)]

    print(f"  rate                 {k}/{n} = {100.0*k/n:.2f}%")
    if k == 0:
        ub = cp_upper(0, n)
        print(f"  exact 95% upper CI   {100*ub:.2f}%   (Clopper-Pearson, one-sided)")
        print(f"  rule of three (3/n)  {100*3.0/n:.2f}%   (approximation; the exact value is above)")
        print(f"  P(0 faults | 5%)     {(0.95)**n:.4f}")
        print()
        if n >= 60:
            print(f"  POWER STATEMENT: {n} clean in-regime arms with zero faults on either face gives")
            print(f"  ~95% confidence the true fault rate is below {100*3.0/n:.1f}% (rule of three).")
            print(f"  Criterion 3's bar (>=60 arms) is {'MET' if n >= 60 else 'NOT met'}.")
        else:
            print(f"  POWER STATEMENT: {n} clean arms is BELOW the pre-registered 60-arm bar.")
            print(f"  The upper bound does not yet reach the ~5% target (it is {100*ub:.1f}%).")
    else:
        print(f"  exact 95% CI         [{100*cp_upper(k, n, 0.975)!r}, ...]  k>0: acceptance has FAILED")
        print(f"  ACCEPTANCE FAILS: {k} in-regime arm(s) faulted; the criterion is zero on BOTH faces.")
    print()
    print(f"  face tally           FaceA(nan)={len(faces_a)}  FaceB(parse/utf8)={len(faces_b)}"
          f"  unattributed={len(unattr)}")
    for r in fault[:10]:
        f = []
        if r["nan"] > 0:
            f.append(f"FaceA nan={r['nan']}")
        if r["parse"] > 0 or r["utf8"] > 0:
            f.append(f"FaceB parse={r['parse']} utf8={r['utf8']}")
        print(f"    fault: cell {r['cell']} gtt_start={r['gtt']}G req={r['req']} {' '.join(f) or 'see server log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
