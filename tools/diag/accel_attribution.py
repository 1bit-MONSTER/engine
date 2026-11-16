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
"""Read the accel sampler's logs and say what they can and cannot attribute (engine#2542).

`~/.cache/accel-sampler/accel-sampler.sh` (pi-pkg) samples which process holds /dev/accel/accel0 and, for each
IO_PAGE_FAULT, records the samples within +-WINDOW_S. Two properties of the raw logs decide how much a burst can be
attributed, and neither is visible in them:

  * `attributed.log` records the samples in the window as `sort -u` of `pid,cmd,scope,ticket`, with no timestamps and
    no order. When a window spans a holder change it therefore names BOTH holders and cannot say which was live at
    the fault. The sample epochs are in `samples.log`, so this tool rebuilds the timeline.
  * a `-` in that field means "no row in the window", which is not the same as "sampled, nobody held the node": the
    sampler writes an explicit `-`-row for the latter. Collapsing the two turns "I did not look" into "nobody was
    there", which is the failure this repository keeps removing.

Usage:
    tools/diag/accel_attribution.py                 # the live logs
    tools/diag/accel_attribution.py --dir DIR
    tools/diag/accel_attribution.py --self-test
"""

import argparse
import collections
import os
import sys
import shutil
import io
import tempfile


def read_samples(path):
    """[(epoch, pid, fds, cmd, scope, ticket)]; pid '-' is the sampler's 'nothing holds it' row."""
    out = []
    try:
        with open(path, errors="replace") as f:
            for i, line in enumerate(f):
                f6 = line.rstrip("\n").split("\t")
                if len(f6) < 6:
                    continue                      # a partially written row: the sampler appends without locking
                try:
                    out.append((int(f6[0]), f6[1], f6[2], f6[3], f6[4], f6[5]))
                except ValueError:
                    print(f"  {os.path.basename(path)}:{i + 1}: no epoch, skipped", file=sys.stderr)
    except FileNotFoundError:
        pass
    return out


# The sampler watches ONE device node (/dev/accel/accel0, the amdxdna NPU), so a fault belonging to another PCI
# function says nothing about its holders. Measured 2026-11-16 on this box: 189 of 189 fault records in one capture
# were `nvme 0000:c4:00.0 domain=0x0015`, and every one was attributed to whatever held the NPU at the time --
# a category error, not a weak answer (the same correction engine#2623 made to the sibling reader).
ACCEL_DEVICE = "amdxdna"


def is_accel_fault(line):
    """True when the fault line belongs to the device whose holders the sampler records."""
    return ACCEL_DEVICE in line


def read_attributions(path):
    """[(epoch, hits, fault_line)] as written by the sampler's watch loop, or None when there is no file yet.

    None is not a read failure. The sampler creates attributed.log on the FIRST IO_PAGE_FAULT
    (accel-sampler.sh's watch loop appends to it), so a box that has recorded no fault this boot has no such file --
    and that is the healthy state (measured 2026-11-16: 5894 samples, zero faults). Swallowing the error and returning
    [] answered "no attributions" correctly but said nothing about why; raising would report a working capture as
    broken. Returning None lets main() print the reason once and still answer zero, rc 0.
    """
    out = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                f3 = line.rstrip("\n").split("\t", 2)
                if len(f3) < 3:
                    continue
                try:
                    out.append((int(f3[0]), f3[1], f3[2]))
                except ValueError:
                    continue
    except FileNotFoundError:
        return None
    return out


def intervals(samples):
    """Seconds between consecutive DISTINCT sample epochs -- the effective interval, which is what the window sees.

    It is not SAMPLE_S: the sampler scans every /proc/<pid>/fd set, which takes seconds on a busy box (measured
    6-9 s against a configured 2 s), so the tuning knob and the observable are two different numbers."""
    ts = sorted({s[0] for s in samples})
    return [b - a for a, b in zip(ts, ts[1:])]


def holder_table(samples):
    """{pid: {cmd, spans, first, last, n}} for every real holder (the '-' rows excluded).

    `spans` is the ORDERED list of (ticket, first, last) for that pid. It matters because the sampler records the
    token as it stood at sample time, not the ticket the process belongs to: a process that outlives its own ticket
    is relabelled with whatever ticket is live next, so a single-ticket view silently launders a leaked holder into
    "part of the next ticket". Four cl-1077 PIDs did exactly this on 2026-11-15 18:20 (three boundaries) and were
    invisible until the spans were kept.
    """
    t = {}
    for epoch, pid, _fds, cmd, _scope, ticket in samples:
        if pid == "-":
            continue
        e = t.setdefault(pid, {"cmd": cmd, "spans": [], "first": epoch, "last": epoch, "n": 0})
        e["first"] = min(e["first"], epoch)
        e["last"] = max(e["last"], epoch)
        e["n"] += 1
        tk = ticket.rstrip()
        if e["spans"] and e["spans"][-1][0] == tk:
            e["spans"][-1] = (tk, e["spans"][-1][1], epoch)
        else:
            e["spans"].append((tk, epoch, epoch))
    return t


def spans_ticket_boundaries(table):
    """[(pid, spans)] for holders seen under more than one ticket.

    An fd held across a release is not proof that two AIE2 contexts were live -- a process can hold the node open
    without submitting -- but it is the precondition the fleet's reset guard exists for (two contexts on one array is
    what produced the double TDR on 2026-11-02), so it is reported rather than inferred away.
    """
    return [(pid, e["spans"]) for pid, e in table.items() if len(e["spans"]) > 1]


def in_window(samples, ts, window):
    return [s for s in samples if ts - window <= s[0] <= ts + window]


def explain(samples, attributions, window, any_device=False):
    """One block per fault: what was sampled around it, in order, and whether that identifies a single holder.

    A fault on another PCI function is reported as NOT THIS DEVICE: attributing an NVMe fault to the process holding
    /dev/accel is a category error, and a reader that does it silently produces a confident wrong holder.
    """
    lines = []
    for ts, _hits, fault in attributions:
        if not any_device and not is_accel_fault(fault):
            lines.append(f"  {ts}  NOT THIS DEVICE: the fault is not {ACCEL_DEVICE}'s, so the holders cannot explain it")
            lines.append(f"      {fault[:110]}")
            continue
        w = sorted(in_window(samples, ts, window), key=lambda s: s[0])
        if not w:
            verdict = "NO SAMPLE in the window: nothing can be attributed (this is not 'nobody held it')"
        else:
            held = [s for s in w if s[1] != "-"]
            empty = [s for s in w if s[1] == "-"]
            pids = {s[1] for s in held}
            if not held:
                verdict = f"sampled {len(w)}x, nothing held the node (explicit '-' rows): not an accel client"
            elif len(pids) == 1:
                s = held[0]
                verdict = (f"one holder across {len(held)} sample(s): pid {s[1]} "
                           f"({s[3].split()[0] if s[3] else '?'}...) ticket={s[5].rstrip()[:40]}")
            else:
                verdict = (f"AMBIGUOUS: {len(pids)} holders inside +-{window}s "
                           f"({', '.join(sorted(pids))}) -- the raw 'hits' field names them all and cannot "
                           f"say which was live at the fault")
            if empty:
                verdict += f"; {len(empty)} of {len(w)} samples had nothing holding the node"
        lines.append(f"  {ts}  {verdict}")
        lines.append(f"      {fault[:110]}")
        for s in w:
            lines.append(f"      {s[0]:>11}  pid={s[1]:<8} {s[3][:56]}  {s[5].rstrip()[:34]}")
    return lines


def self_test():
    # A window that spans a holder change: the raw record cannot separate them, the timeline can.
    samples = [
        (100, "111", "1", "/a serve -m X", "scope-a", "gpu-hold cl-a 1"),
        (106, "222", "1", "/b serve -m Y", "scope-b", "gpu-hold cl-b 2"),
        (112, "-", "-", "-", "-", "gpu-hold cl-c 3"),      # sampled, nobody holds the node
    ]
    assert intervals(samples) == [6, 6], intervals(samples)
    t = holder_table(samples)
    assert set(t) == {"111", "222"}, t
    assert t["111"]["n"] == 1 and t["222"]["spans"][0][0].startswith("gpu-hold cl-b"), t

    # The window decides the answer, which is the point of printing it. Hand-checked against the samples above:
    #   window 3, fault 100 -> [97,103]  -> {111}            one holder
    #   window 3, fault 106 -> [103,109] -> {222}            one holder
    #   window 3, fault 109 -> [106,112] -> {222, '-'}       one holder AND an explicit "nothing holds it"
    #   window 1, fault 112 -> [111,113] -> {'-'}            sampled, nothing held the node
    #   window 5, fault 103 -> [98,108]  -> {111, 222}       AMBIGUOUS: the same fault is unambiguous at 3
    #   window 3, fault 200 -> [197,203] -> {}               nothing sampled: not evidence of absence
    # The fault text names the device, as every real line does -- the fixtures used to carry a bare "A"/"B", which
    # is why none of them could see the category error engine#2623/#2542 found: a line with no device cannot be
    # classified as ours or not.
    A_ = "amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0003 address=0x4128000]"
    narrow, wide = "\n".join(explain(samples, [(100, "", A_), (106, "", A_), (109, "", A_),
                                               (200, "", A_)], 3)), "\n".join(explain(samples, [(103, "", A_)], 5))
    assert narrow.count("one holder across 1 sample(s): pid 111") == 1, narrow
    assert narrow.count("one holder across 1 sample(s): pid 222") == 2, narrow   # faults B and C
    assert narrow.count("; 1 of 2 samples had nothing holding the node") == 1, narrow   # fault C
    assert "NO SAMPLE in the window" in narrow, narrow
    dot = "\n".join(explain(samples, [(112, "", A_)], 1))
    assert "sampled 1x, nothing held the node" in dot, dot
    assert "AMBIGUOUS: 2 holders inside +-5s (111, 222)" in wide, wide
    assert "not 'nobody held it'" in narrow, narrow

    # A holder that outlives its ticket is RELABELLED by the sampler (it records the token as it stood), so the
    # spans are what make the leak visible: same pid, two tickets.
    leaky = [(100, "111", "1", "/a", "s", "gpu-hold cl-a 1"), (106, "111", "1", "/a", "s", "gpu-hold cl-b 2")]
    lt = holder_table(leaky)
    assert [x[0] for x in lt["111"]["spans"]] == ["gpu-hold cl-a 1", "gpu-hold cl-b 2"], lt
    assert [p for p, _ in spans_ticket_boundaries(lt)] == ["111"], spans_ticket_boundaries(lt)
    assert spans_ticket_boundaries(holder_table(samples)) == [], "the plain fixture must not trip the arm"

    # THE RENDERER, not only the analysis (engine#2542): the "holders seen" line read a `ticket` key that
    # holder_table() stopped building when it grew `spans`, so the merged tool raised KeyError on the first real log
    # (3,500+ samples) and had only ever run on fixtures where that line was not reached. Rendering is asserted here
    # because a reader that crashes on real data is not a reader.
    rendered = render_holders(holder_table(leaky))
    assert "gpu-hold cl-b 2" in rendered, rendered          # the LAST span: what the process was relabelled with
    assert "cl-a" not in rendered, rendered

    # engine#2542: a fault on another PCI function must be NAMED, not attributed to the NPU's holders. Measured on
    # this box: 189 of 189 records in one capture were `nvme 0000:c4:00.0 domain=0x0015` and every one was given the
    # process holding /dev/accel -- a category error that reads as a confident answer.
    foreign = ("1794798810\t215633\t/serve\t-\tscope\tgpu-hold cl-a\t"
               "[  321.308031] strixhalo kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged "
               "[IO_PAGE_FAULT domain=0x0015 address=0xe4240000 flags=0x0000]\n")
    ppath = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    ppath.write(foreign); ppath.close()
    try:
        at = read_attributions(ppath.name)
        assert len(at) == 1, at
        out = "\n".join(explain(samples, at, 5))
        assert "NOT THIS DEVICE" in out, out
        assert "one holder" not in out, out
        # ... and the override still lets a reader ask the cross-device question deliberately
        out2 = "\n".join(explain(samples, at, 5, any_device=True))
        assert "NOT THIS DEVICE" not in out2, out2
    finally:
        os.unlink(ppath.name)

    # A partially written row must not kill the reader.
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write("100\t111\t1\t/a serve\tscope\tticket\n")
        f.write("106\t22")                                  # torn
        p = f.name
    try:
        assert [s[0] for s in read_samples(p)] == [100], read_samples(p)
    finally:
        os.unlink(p)
    # engine#2610 review: the ONLY arm that can fail for this change. samples.log missing already answers rc 2 through
    # main()'s `if not samples` guard (that behaviour predates this branch, and an arm for it cannot fail until the
    # guard is removed) -- so the arm here is for the file whose absence is *healthy*: with samples present and no
    # attributed.log (no fault recorded yet, the state this box is in) the answer must be zero attributions and rc 0,
    # with one line saying why.
    _d = tempfile.mkdtemp()
    try:
        with open(os.path.join(_d, "samples.log"), "w") as _f:
            _f.write("100\t111\tx\tA\t1\tk\n")
        _argv, _err = sys.argv, sys.stderr
        sys.argv = ["accel_attribution.py", "--dir", _d]
        sys.stderr = io.StringIO()
        try:
            assert main() == 0, "no attributed.log is the healthy state: zero attributions, rc 0"
            assert "no attributed.log yet" in sys.stderr.getvalue(), "the absence must be explained, not silent"
        finally:
            sys.argv, sys.stderr = _argv, _err
    finally:
        shutil.rmtree(_d, ignore_errors=True)
    print("accel_attribution self-test: ok")


def render_holders(t):
    """The "holders seen" block. A separate function so the self-test can drive it: this line read a `ticket` key
    that holder_table() stopped building when it grew `spans`, and the crash only appeared on the first real log."""
    out = []
    for pid, e in sorted(t.items(), key=lambda kv: -kv[1]["n"]):
        last = e["spans"][-1][0] if e["spans"] else "-"   # the LAST span: what the process was relabelled with
        out.append(f"  {pid:<9} {e['first']}..{e['last']} ({e['last'] - e['first']:>5}s) n={e['n']:<4} "
                   f"{e['cmd'][:46]}  {last[:30]}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=os.path.expanduser("~/.cache/accel-sampler"))
    ap.add_argument("--window", type=int, default=5, help="the sampler's WINDOW_S (default 5)")
    ap.add_argument("--any-device", action="store_true",
                    help="also attribute faults from devices the sampler does not watch (default: name and skip them)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0

    samples = read_samples(os.path.join(args.dir, "samples.log"))
    attrs = read_attributions(os.path.join(args.dir, "attributed.log"))
    if attrs is None:
        print("accel_attribution: no attributed.log yet: the sampler has recorded no fault "
              "(it creates the file on the first one)", file=sys.stderr)
        attrs = []
    if not samples:
        print(f"accel_attribution: no samples in {args.dir}/samples.log", file=sys.stderr)
        return 2

    iv = intervals(samples)
    span = max(s[0] for s in samples) - min(s[0] for s in samples)
    med = sorted(iv)[len(iv) // 2] if iv else 0
    print(f"samples: {len(samples)} rows over {span} s; effective interval median {med} s, "
          f"min {min(iv) if iv else 0}, max {max(iv) if iv else 0}")
    gaps = [d for d in iv if med and d > 3 * med]
    print(f"gaps > 3x median (sampler down or a restart): {len(gaps)}"
          + (f" {gaps[:5]}" if gaps else ""))
    if med > 0 and args.window < med:
        print(f"  note: +-{args.window}s window is shorter than the effective interval ({med}s): some faults "
              f"will be sampled zero times")
    nobody = sum(1 for s in samples if s[1] == "-")
    print(f"'nothing holds the node' rows: {nobody} of {len(samples)}")

    t = holder_table(samples)
    crossed = spans_ticket_boundaries(t)
    if crossed:
        print(f"\nholders that outlived their ticket ({len(crossed)} pid(s)) -- an fd held across a release is the "
              f"precondition for two contexts on one array, not proof of one:")
        for pid, spans in crossed:
            print(f"  {pid:<9} {t[pid]['cmd'][:44]}")
            for tk, a, b in spans:
                print(f"      {a}..{b} ({b - a:>4}s)  {tk[:46]}")
    if t:
        print("\nholders seen (pid, first..last, samples, ticket):")
        print(render_holders(t))

    print(f"\nattributions: {len(attrs)}")
    for line in explain(samples, attrs, args.window, args.any_device):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
