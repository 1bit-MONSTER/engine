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
"""Attribute each IOMMU fault to the holder that was live at its own timestamp.

engine#2542 needs to know which ticket submitted the command behind a `/dev/accel` fault. The sampler
records holders on a cadence (epoch, pid, fds, cmd, scope, ticket); the kernel records faults with a
monotonic timestamp. Joining them **per fault** is not the same as sampling a window around one:

  measured on 1816 samples over an hour of this box, a +-5 s window names two or more holders for
  **33 %** of samples, and widening to +-15 s makes it 34.7 % -- 18 tickets held the node inside that
  hour, so the limit is holder turnover, not sampling density. A window that names two lanes has
  named none.

So this walks the faults, not the windows: for a fault at time t, the holder is the last sample at or
before t (that is the only interval in which the node was observed held by it), and the answer carries
its own uncertainty rather than hiding it:

  ATTRIBUTED         the last sample before t holds the node, and the first sample after t names the
                     same holder (or there is no later sample) -- the fault sits inside one tenure
  HANDOVER-ADJACENT  the next sample names a DIFFERENT holder: the fault fell in a gap where the node
                     changed hands, so the preceding sample is a best guess, not an attribution
  NOTHING HELD       the last sample before t is an explicit '-' row: sampled, nothing held the node
  NO SAMPLE          no sample within --max-gap of t: nothing was observed, which is not "nobody held it"

The `-` distinction is the one that turns "I did not look" into "nobody was there" if collapsed, and
the HANDOVER-ADJACENT case is the one a window cannot express: it would name both holders and stop.

    # boot epoch: seconds-since-boot stamps are relative to it
    tools/diag/iommu_fault_attribution.py --samples ~/.cache/accel-sampler/samples.log \
        --dmesg capture.txt --boot-epoch "$(stat -c %Y /proc/1)" --max-gap 15
    sudo dmesg | tools/diag/iommu_fault_attribution.py --samples ... --boot-epoch ...

Nothing here reads a device or starts anything: dmesg text on stdin or from a file, the sampler log as
a file, and `--self-test` runs entirely on fixtures.
"""

import argparse
import os
import re
import subprocess
import sys

# `[  1234.567890] nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x…`
STAMP = re.compile(r"^\[\s*(\d+\.\d+)\]")


def faults(text):
    """[(boot_seconds, address)] for every IO_PAGE_FAULT line, in log order."""
    out = []
    for line in text.splitlines():
        if "IO_PAGE_FAULT" not in line:
            continue
        m = STAMP.match(line)
        addr = re.search(r"address=(0x[0-9a-fA-F]+)", line)
        out.append((float(m.group(1)) if m else None, addr.group(1) if addr else "?"))
    return out


def rows_from_text(text):
    """[(epoch, pid, ticket)] from sampler-log text; a '-' pid is its 'nothing holds it' row."""
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 6 or not parts[0].strip().isdigit():
            continue
        rows.append((int(parts[0]), parts[1].strip(), parts[5].strip()))
    rows.sort()
    return rows


def samples(path):
    """[(epoch, pid, ticket)] from the sampler log; a '-' pid is its 'nothing holds it' row."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return rows_from_text(f.read())


def attribute(rows, boots, max_gap):
    """[(boot_seconds, address, verdict)] — one verdict per fault, decided by its own timestamp."""
    out = []
    for sec, addr in boots:
        if sec is None:
            out.append((sec, addr, "NO SAMPLE: the line carries no kernel timestamp"))
            continue
        before = [r for r in rows if r[0] <= sec]
        after = [r for r in rows if r[0] > sec]
        if not before:
            out.append((sec, addr, "NO SAMPLE: no sample at or before the fault"))
            continue
        epoch, pid, ticket = before[-1]
        if sec - epoch > max_gap:
            out.append((sec, addr, f"NO SAMPLE within {max_gap}s (last sample {sec - epoch:.0f}s earlier)"))
            continue
        if pid == "-":
            out.append((sec, addr, "NOTHING HELD: sampled at the node, no accel client held it"))
            continue
        who = f"pid {pid} {ticket[:64]}"
        if after and after[0][1] != pid:
            # Name the other side: a window would name both and stop; this says which one is the guess
            # and what the next observation saw. '-' there is "nothing held it", not a process named '-'.
            nxt = "nothing held the node" if after[0][1] == "-" else f"pid {after[0][1]}"
            out.append((sec, addr, f"HANDOVER-ADJACENT: {who} (next sample: {nxt})"))
        else:
            out.append((sec, addr, f"ATTRIBUTED: {who}"))
    return out


def yield_line(res):
    """The number that decides whether a sampling run can answer the question at all."""
    k = {"ATTRIBUTED": 0, "HANDOVER-ADJACENT": 0, "NOTHING HELD": 0, "NO SAMPLE": 0}
    for _, _, v in res:
        for key in k:
            if v.startswith(key):
                k[key] += 1
                break
    n = len(res)
    return ("faults: %d; attributable %d, ambiguous %d, nothing-held %d, no-sample %d"
            % (n, k["ATTRIBUTED"], k["HANDOVER-ADJACENT"], k["NOTHING HELD"], k["NO SAMPLE"]))


def outlook(rows, windows):
    """How many faults a given window COULD attribute, from the holder timeline alone.

    Measured on a 1816-sample hour of this box: +-5 s names two holders for 33 % of samples and
    widening makes it worse, because 18 tickets held the node in that hour. Printing that before a
    capture starts is the difference between "run it for a day and hope" and knowing the sampler's
    window cannot answer the question.
    """
    out = []
    for w in windows:
        one = amb = nothing = none = 0
        for epoch, _pid, _ticket in rows:
            near = [(e, p) for e, p, _t in rows if abs(e - epoch) <= w]
            holders = {p for _e, p in near if p != "-"}
            if not near:
                none += 1
            elif not holders:
                nothing += 1
            elif len(holders) == 1:
                one += 1
            else:
                amb += 1
        n = len(rows) or 1
        out.append((w, 100.0 * one / n, 100.0 * amb / n, 100.0 * nothing / n, 100.0 * none / n))
    return out


def attributed_rows(text):
    """[(epoch, hits)] from the sampler's attributed.log: `<epoch>\t<hits|->\t<raw fault line>`.

    The sampler writes this per fault (accel_sampler.sh: `printf '%s\t%s\t%s\n' "$ts" "$hits" "$line"`),
    where `hits` is its windowed holder list. So both readers have the same evidence and differ only in
    which field they trust -- which makes the disagreement set computable.
    """
    out = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].strip().isdigit():
            continue
        out.append((int(parts[0]), parts[1].strip()))
    return out


def disagreements(rows, pairs, max_gap):
    """[(epoch, window_hits, timestamp_verdict, agree)] for each fault in attributed.log."""
    out = []
    for epoch, hits in pairs:
        res = attribute(rows, [(epoch, "-")], max_gap)[0][2]
        # agreement is not string equality: the window names holders, this names one and says how sure.
        # A window with two holders and a HANDOVER-ADJACENT verdict AGREE about the situation.
        win_n = 0 if hits in ("", "-") else len([h for h in hits.replace(",", " ").split() if h])
        one = res.startswith("ATTRIBUTED")
        if one and win_n in (0, 1):
            agree = True
        elif res.startswith("HANDOVER-ADJACENT") and win_n >= 2:
            agree = True        # both saw a change; this one names which side is the guess
        elif res.startswith(("NOTHING HELD", "NO SAMPLE")) and win_n == 0:
            agree = True
        else:
            agree = False
        out.append((epoch, hits or "-", res, agree))
    return out


def fmt_attr(epoch, hits, verdict, agree):
    """One --attributed line. A function so the self-test can execute it: a format error here would pass
    every analysis assertion and crash on the first real log (engine#2602)."""
    return "%d\twindow=%s\t%s\t%s" % (epoch, hits, "agree" if agree else "DISAGREE", verdict)


def fmt_outlook_row(w, one, amb, nothing, none):
    """One --outlook row, for the same reason."""
    return "%5.0fs  %9.1f%%  %8.1f%%  %11.1f%%  %8.1f%%" % (w, one, amb, nothing, none)


def fmt_fault(sec, addr, verdict):
    """One per-fault line, for the same reason."""
    return "%s\t%s\t%s" % ("-" if sec is None else "%.0f" % sec, addr, verdict)


def self_test():
    # One hour of holders, one sample per 100 s, handing over at 300.
    rows = [(0, "111", "gpu-hold cl-a pid 1 t0"), (100, "111", "gpu-hold cl-a pid 1 t0"),
            (200, "-", "-"), (300, "222", "gpu-hold cl-b pid 2 t1"),
            (400, "222", "gpu-hold cl-b pid 2 t1")]
    boots = [(50.0, "0x1000"),     # held by 111, next sample also 111 -> ATTR
             (150.0, "0x2000"),    # held by 111, next sample is '-'    -> HANDOVER-ADJACENT
             (250.0, "0x3000"),    # last sample before is '-'          -> NOTHING HELD
             (305.0, "0x4000"),    # first sample of 222's tenure       -> ATTR
             (900.0, "0x5000")]    # 500 s after the last sample        -> NO SAMPLE
    res = attribute(rows, boots, max_gap=150)
    def kind(v):
        # the key is the verdict before any detail; "NO SAMPLE within 150s" is still NO SAMPLE
        head = v.split(":")[0]
        return "NO SAMPLE" if head.startswith("NO SAMPLE") else head
    got = [kind(v) for _, _, v in res]
    assert got == ["ATTRIBUTED", "HANDOVER-ADJACENT", "NOTHING HELD", "ATTRIBUTED", "NO SAMPLE"], res
    assert "cl-a" in res[0][2] and "pid 111" in res[0][2], res[0]
    # The handover verdict must NAME both sides: naming one is the failure mode of a window. Here the
    # handover is to 'nobody', which must read as words rather than as a process called '-'.
    assert "pid 111" in res[1][2] and "nothing held the node" in res[1][2], res[1]

    # And the case that motivates not using a window at all: two real holders either side of the fault.
    hand = attribute([(0, "111", "gpu-hold cl-a pid 1"), (100, "111", "gpu-hold cl-a pid 1"),
                      (200, "222", "gpu-hold cl-b pid 2")], [(150.0, "0x9000")], max_gap=150)
    assert hand[0][2].startswith("HANDOVER-ADJACENT"), hand
    assert "pid 111" in hand[0][2] and "pid 222" in hand[0][2], hand
    assert "no accel client held it" in res[2][2], res[2]      # not "I did not look"
    assert yield_line(res) == ("faults: 5; attributable 2, ambiguous 1, nothing-held 1, no-sample 1"), yield_line(res)

    # --max-gap decides freshness, and a smaller gap converts ATTRIBUTED into NO SAMPLE rather than
    # silently attributing across a long unobserved stretch.
    tight = [kind(v) for _, _, v in attribute(rows, [(50.0, "0x1")], max_gap=5)]
    assert tight == ["NO SAMPLE"], tight

    # The line shape the kernel actually writes: without a leading stamp there is no time to join on,
    # so the verdict is NO SAMPLE and says why -- it must never default to the nearest holder.
    no_stamp = faults("nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT address=0x4128000]")
    assert no_stamp == [(None, "0x4128000")], no_stamp
    assert attribute(rows, no_stamp, 150)[0][2].startswith("NO SAMPLE"), attribute(rows, no_stamp, 150)

    # The outlook is computed, not asserted from a constant: a window wide enough to cover two
    # different holders must report ambiguity, and one narrower than the gap must not.
    ol = {w: (one, amb) for w, one, amb, _n, _z in outlook(rows, [5, 400])}
    assert ol[5][0] == 80.0 and ol[5][1] == 0.0, ol   # 4 of 5: the 200s sample IS the "-" row, so it is nothing-held
    assert ol[400][1] > 0.0, ol                               # 400s spans the handover -> ambiguous

    # THE WRITER/READER CONTRACT. These are the sampler's own fixture lines, copied verbatim from
    # tools/diag/accel_sampler.sh: if that writer changes its columns, this reader must fail here
    # rather than silently parse zeros out of a shifted line.
    contract = ("1000\t111\t1\t/serve A\t/scope/a\tticket1\n"
                "1010\t222\t2\t/serve B\t/scope/b\tticket2\n")
    assert rows_from_text(contract) == [(1000, "111", "ticket1"), (1010, "222", "ticket2")], \
        rows_from_text(contract)
    assert rows_from_text("1000\t-\t-\t-\t-\t-\n") == [(1000, "-", "-")], "the '-' row shape"
    # And a line with fewer columns is skipped rather than mis-parsed into a holder.
    assert rows_from_text("1000\t111\tonly-three\n") == [], "a short row must not become a holder"

    # The disagreement set: the sampler's windowed `hits` beside this tool's per-fault verdict. The
    # fixture lines are that file's real shape (`<epoch>\t<hits|->\t<raw line>`).
    attr_text = """105\t111\t[ 105.000000] nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT address=0x1]
150\t111 222\t[ 150.000000] nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT address=0x2]
250\t-\t[ 250.000000] nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT address=0x3]
"""
    pairs = attributed_rows(attr_text)
    assert pairs == [(105, "111"), (150, "111 222"), (250, "-")], pairs
    ds = disagreements(rows, pairs, max_gap=150)
    # 105 DISAGREES, and that is the useful case: the window names one holder, but the sample AFTER the
    # fault is the '-' row, so the join says the node changed hands to nobody -- the window omitted a
    # handover it could not see. 150: window names two, join says HANDOVER-ADJACENT -> agree (same fact,
    # stated more precisely). 250: window says nobody, join says NOTHING HELD -> agree.
    assert [d[3] for d in ds] == [False, True, True], ds
    assert "next sample: nothing held the node" in ds[0][2], ds[0]
    # And a window that names ONE holder while the join sees a handover is a real disagreement, which is
    # the case worth printing rather than hiding.
    one_vs_hand = disagreements(rows, [(150, "111")], max_gap=150)
    assert one_vs_hand[0][3] is False, one_vs_hand

    # Stamps and addresses are parsed, and non-fault lines are ignored.
    text = ("[1234.500000] nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x4128000 flags=0x0007]\n"
            "[1235.000000] unrelated kernel line\n")
    assert faults(text) == [(1234.5, "0x4128000")], faults(text)
    # EVERY OUTPUT LINE IS EXERCISED. engine#2602 shipped a renderer that crashed on the first real log
    # because its self-test only asserted the analysis; a tool that cannot print is not a tool that works.
    assert fmt_attr(105, "111", "ATTRIBUTED: pid 111 x", True).endswith("\tATTRIBUTED: pid 111 x")
    assert "DISAGREE" in fmt_attr(105, "111", "v", False)
    # Field ORDER and content, not column widths: a width literal is a brittle assertion that fails on a
    # cosmetic change and says nothing about a format-string crash, which is the failure worth catching.
    row = fmt_outlook_row(5, 62.1, 23.1, 14.9, 0.0).split()
    assert row == ["5s", "62.1%", "23.1%", "14.9%", "0.0%"], row
    assert fmt_fault(None, "0x1", "v") == "-\t0x1\tv", fmt_fault(None, "0x1", "v")
    assert fmt_fault(105.4, "0x1", "v") == "105\t0x1\tv", fmt_fault(105.4, "0x1", "v")

    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", required=False, help="sampler log (tab-separated epoch/pid/…/ticket)")
    ap.add_argument("--dmesg", default="-", help="file of dmesg text, or '-' for stdin (default)")
    ap.add_argument("--boot-epoch", type=float, default=None,
                    help="epoch of boot, so kernel stamps become absolute (default: /proc/stat btime)")
    ap.add_argument("--max-gap", type=float, default=15.0,
                    help="oldest usable sample before a fault, in seconds (default 15)")
    ap.add_argument("--attributed", default=None,
                    help="sampler attributed.log: print its windowed hits beside this tool's verdict")
    ap.add_argument("--outlook", type=float, nargs="*", metavar="W",
                    help="print attributability for these +-W windows and exit (no faults needed)")
    ap.add_argument("--selftest", "--self-test", dest="self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.samples:
        ap.error("--samples is required unless --self-test")
    if args.attributed:
        rows = samples(args.samples)
        pairs = attributed_rows(open(args.attributed, encoding="utf-8", errors="replace").read())
        ds = disagreements(rows, pairs, args.max_gap)
        for epoch, hits, verdict, agree in ds:
            print(fmt_attr(epoch, hits, verdict, agree))
        n = len(ds) or 1
        print("faults: %d; agree %d, disagree %d" % (len(ds), sum(1 for d in ds if d[3]),
                                                      sum(1 for d in ds if not d[3])))
        return 0
    if args.outlook:
        rows = samples(args.samples)
        print("window  one-holder  ambiguous  nothing-held  no-sample")
        for w, one, amb, nothing, none in outlook(rows, args.outlook):
            print(fmt_outlook_row(w, one, amb, nothing, none))
        return 0

    boot = args.boot_epoch
    if boot is None:
        with open("/proc/stat", encoding="utf-8") as f:
            boot = float(next(int(l.split()[1]) for l in f if l.startswith("btime")))
    text = sys.stdin.read() if args.dmesg == "-" else open(args.dmesg, encoding="utf-8", errors="replace").read()

    rows = samples(args.samples)
    boots = [(None if s is None else boot + s, a) for s, a in faults(text)]
    res = attribute(rows, boots, args.max_gap)
    for sec, addr, verdict in res:
        when = "-" if sec is None else "%.0f" % sec
        print(fmt_fault(sec, addr, verdict))
    print(yield_line(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
