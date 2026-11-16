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


ACCEL = "amdxdna"   # the driver whose node the sampler's holders describe


def faults(text):
    """[(boot_seconds, address, is_accel)] for every IO_PAGE_FAULT line, in log order.

    `is_accel` matters because the sampler records who holds `/dev/accel`: a fault on another PCI function
    (the NVMe stream is `nvme 0000:c4:00.0 domain=0x0015`) has nothing to do with those holders, and
    attributing it to one is a category error rather than a weak answer. Measured on this box: 137 NVMe
    faults were "attributed" to a `gpu-hold` ticket that was holding the NPU.
    """
    out = []
    for line in text.splitlines():
        if "IO_PAGE_FAULT" not in line:
            continue
        m = STAMP.match(line)
        addr = re.search(r"address=(0x[0-9a-fA-F]+)", line)
        out.append((float(m.group(1)) if m else None, addr.group(1) if addr else "?", ACCEL in line))
    return out


def rows_from_text(text):
    """[(epoch, pid, ticket)] from sampler-log text; a '-' pid is its 'nothing holds it' row."""
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 6 or not parts[0].strip().isdigit():
            continue
        # A GAP row (engine#2624) carries its length in column 3 where a sample carries its fds; keep the length as
        # the note so the verdict can say how long the sampler was stalled, not just that it was.
        pid = parts[1].strip()
        rows.append((int(parts[0]), pid, (parts[2].strip() or "?") if pid == "GAP" else parts[5].strip()))
    rows.sort()
    return rows


def samples(path):
    """[(epoch, pid, ticket)] from the sampler log; a '-' pid is its 'nothing holds it' row."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return rows_from_text(f.read())


def attribute(rows, boots, max_gap, any_device=False):
    """[(boot_seconds, address, verdict)] — one verdict per fault, decided by its own timestamp."""
    out = []
    for sec, addr, is_accel in boots:
        if not is_accel and not any_device:
            # The sampler records who holds /dev/accel. A fault on another PCI function has nothing to do with
            # those holders: measured on this box, 137 NVMe faults were "attributed" to a gpu-hold ticket that was
            # holding the NPU. A category error is not a weak answer, so refuse it unless asked to proceed.
            out.append((sec, addr, "NOT THE ACCEL NODE: this fault is on another device, so the sampler's holders "
                                   "do not describe it (pass --any-device to attribute anyway)"))
            continue
        if sec is None:
            out.append((sec, addr, "NO SAMPLE: the line carries no kernel timestamp"))
            continue
        # A GAP row is written when the sampler RESUMES (accel_sampler.sh: `printf '%s\tGAP\t%s' "$NOW" "$((NOW - LAST))"`),
        # so a row at epoch E with length L describes the stall (E-L, E] -- BEFORE its own timestamp. A fault is
        # "stalled" when such a span covers it; GAP rows are never a sample to attribute to, before or after
        # (review of engine#2623: reading the row as "the last sample before t" called faults AFTER a stall stalled,
        # and named "pid GAP" as the next holder for faults INSIDE one).
        stall = next((r for r in rows if r[1] == "GAP" and _gap_len(r[2]) is not None
                      and r[0] - _gap_len(r[2]) < sec <= r[0]), None)
        if stall is not None:
            out.append((sec, addr, "NO SAMPLE: the sampler was stalled across this fault (a %ss gap ending at %d)"
                        % (stall[2], stall[0])))
            continue
        real = [r for r in rows if r[1] != "GAP"]
        before = [r for r in real if r[0] <= sec]
        after = [r for r in real if r[0] > sec]
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


def _gap_len(field):
    """Seconds of a GAP row's length field, or None when it is not a number (a malformed row covers nothing)."""
    try:
        return int(field)
    except (TypeError, ValueError):
        return None


def yield_line(res):
    """The number that decides whether a sampling run can answer the question at all."""
    # Every verdict must be counted, or the summary says "0 of everything" about a capture full of faults -- the
    # silent-drop class this file exists to avoid. "NOT THE ACCEL NODE" is a fifth outcome, not a footnote.
    k = {"ATTRIBUTED": 0, "HANDOVER-ADJACENT": 0, "NOTHING HELD": 0, "NO SAMPLE": 0, "NOT THE ACCEL NODE": 0}
    for _, _, v in res:
        for key in k:
            if v.startswith(key):
                k[key] += 1
                break
    n = len(res)
    total = sum(k.values())
    return ("faults: %d; attributable %d, ambiguous %d, nothing-held %d, no-sample %d, other-device %d"
            % (n, k["ATTRIBUTED"], k["HANDOVER-ADJACENT"], k["NOTHING HELD"], k["NO SAMPLE"],
               k["NOT THE ACCEL NODE"])) + ("" if total == n else "  !! %d fault(s) uncounted" % (n - total))


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
            holders = {p for _e, p in near if p not in ("-", "GAP")}
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
        res = attribute(rows, [(epoch, "-", True)], max_gap)[0][2]   # attributed.log holds accel faults
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
    boots = [(50.0, "0x1000", True),     # held by 111, next sample also 111 -> ATTR
             (150.0, "0x2000", True),    # held by 111, next sample is '-'    -> HANDOVER-ADJACENT
             (250.0, "0x3000", True),    # last sample before is '-'          -> NOTHING HELD
             (305.0, "0x4000", True),    # first sample of 222's tenure       -> ATTR
             (900.0, "0x5000", True)]    # 500 s after the last sample        -> NO SAMPLE
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
                      (200, "222", "gpu-hold cl-b pid 2")], [(150.0, "0x9000", True)], max_gap=150)
    assert hand[0][2].startswith("HANDOVER-ADJACENT"), hand
    assert "pid 111" in hand[0][2] and "pid 222" in hand[0][2], hand
    assert "no accel client held it" in res[2][2], res[2]      # not "I did not look"
    assert yield_line(res) == ("faults: 5; attributable 2, ambiguous 1, nothing-held 1, no-sample 1, other-device 0"), yield_line(res)

    # --max-gap decides freshness, and a smaller gap converts ATTRIBUTED into NO SAMPLE rather than
    # silently attributing across a long unobserved stretch.
    tight = [kind(v) for _, _, v in attribute(rows, [(50.0, "0x1", True)], max_gap=5)]
    assert tight == ["NO SAMPLE"], tight

    # The line shape the kernel actually writes: without a leading stamp there is no time to join on,
    # so the verdict is NO SAMPLE and says why -- it must never default to the nearest holder.
    no_stamp = faults("amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT address=0x4128000]")
    assert no_stamp == [(None, "0x4128000", True)], no_stamp   # the device check passes; only the stamp is missing
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

    # engine#2624: the sampler writes `<epoch>\tGAP\t<gap s>\t-\t-\t-` when it slept too long. Column 2 is the pid
    # column, so a reader that trusts it invents a holder called GAP -- which is what this tool did before this arm.
    gap_log = "1000\t-\t-\t-\t-\t-\n1010\tGAP\t93\t-\t-\t-\n"
    gap_rows = rows_from_text(gap_log)
    assert gap_rows == [(1000, "-", "-"), (1010, "GAP", "93")], gap_rows   # the note carries the gap length
    # The GAP row at 1010 with length 93 is the stall (917, 1010], written on RESUME: a fault inside it is stalled,
    # a fault after it is not (review of engine#2623 -- the first version read it the other way round).
    inside = attribute(gap_rows, [(1005.0, "0x1", True)], 150)[0][2]
    assert inside.startswith("NO SAMPLE") and "stalled" in inside, inside
    after_gap = attribute(gap_rows, [(1015.0, "0x1", True)], 150)[0][2]
    assert "stalled" not in after_gap, after_gap
    # and a stall between two holders never names a holder called GAP
    hand = rows_from_text("900\t111\t1\t/a\ts\tgpu-hold cl-a\n910\tGAP\t10\t-\t-\t-\n912\t222\t1\t/b\ts\tgpu-hold cl-b\n")
    for t in (899.5, 905.0, 911.0):
        v = attribute(hand, [(t, "0x1", True)], 15)[0][2]
        assert "GAP" not in v.replace("a 10s gap", ""), (t, v)
    assert "stalled" in attribute(hand, [(905.0, "0x1", True)], 15)[0][2]
    assert attribute(hand, [(911.0, "0x1", True)], 15)[0][2].startswith("HANDOVER-ADJACENT: pid 111")
    # ... and it is not a holder for the outlook either: with a real holder after it, the window still sees one.
    mixed = gap_rows + [(1020, "1234", "gpu-hold cl-a pid 9")]
    assert outlook(mixed, [5])[0][1] > 0.0, outlook(mixed, [5])   # some windows span the real holder

    # THE CATEGORY ERROR, measured: 137 NVMe faults on this box were "attributed" to a gpu-hold ticket that was
    # holding the NPU. A fault on another device must be refused, not guessed -- and --any-device must still work.
    nvme = [(105.0, "0xe1e18000", False)]
    assert attribute(rows, nvme, 150)[0][2].startswith("NOT THE ACCEL NODE"), attribute(rows, nvme, 150)
    assert attribute(rows, nvme, 150, any_device=True)[0][2].startswith("HANDOVER-ADJACENT"), attribute(rows, nvme, 150, True)   # --any-device proceeds; this fixture is a handover

    # The summary must account for every fault: a verdict missing from the tally makes "0 of everything" look like
    # an empty capture. This arm would have printed 0/0/0/0 for 137 NVMe faults before the counter was added.
    assert yield_line([(1.0, "0x1", "NOT THE ACCEL NODE: other device")]) == (
        "faults: 1; attributable 0, ambiguous 0, nothing-held 0, no-sample 0, other-device 1"), \
        yield_line([(1.0, "0x1", "NOT THE ACCEL NODE: other device")])

    # Stamps and addresses are parsed, and non-fault lines are ignored.
    text = ("[1234.500000] nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x4128000 flags=0x0007]\n"
            "[1235.000000] unrelated kernel line\n")
    assert faults(text) == [(1234.5, "0x4128000", False)], faults(text)

    # THE KERNEL PADS THE STAMP, and the unpadded form above is not what it writes: two spaces at 321 s, four below
    # 10 s (`[    0.000000]`). engine#2622 shipped a reader whose regex required a digit straight after `[`, matched
    # nothing on real captures, and reported "ordering not shown by this capture" -- a negative for input it should
    # have matched. `^\[\s*(...)` handles both widths, and this fixture is what keeps that true.
    padded = ("[  321.308031] nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x4128000 flags=0x0000]\n"
              "[    0.000000] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0003 address=0x4130000 flags=0x0007]\n")
    assert faults(padded) == [(321.308031, "0x4128000", False), (0.0, "0x4130000", True)], faults(padded)

    # A capture in the human-readable form (`dmesg -T`, journalctl default) carries NO stamp this parser can use.
    # The verdict must say it cannot read the file, not that the kernel showed no ordering.
    human = "[Sun Nov 15 08:53:39 2026] amdxdna 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT address=0x4128000]\n"
    assert faults(human) == [(None, "0x4128000", True)], faults(human)
    assert attribute(rows, faults(human), 150)[0][2].startswith("NO SAMPLE"), attribute(rows, faults(human), 150)
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
    ap.add_argument("--any-device", action="store_true",
                    help="attribute faults on devices other than the accel node (off by default: the sampler's "
                         "holders describe /dev/accel, so attributing another device's fault is a category error)")
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
    boots = [(None if s is None else boot + s, a, acc) for s, a, acc in faults(text)]
    res = attribute(rows, boots, args.max_gap, any_device=args.any_device)
    for sec, addr, verdict in res:
        when = "-" if sec is None else "%.0f" % sec
        print(fmt_fault(sec, addr, verdict))
    print(yield_line(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
