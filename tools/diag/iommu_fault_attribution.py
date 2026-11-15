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


def samples(path):
    """[(epoch, pid, ticket)] from the sampler log; a '-' pid is its 'nothing holds it' row."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6 or not parts[0].strip().isdigit():
                continue
            rows.append((int(parts[0]), parts[1].strip(), parts[5].strip()))
    rows.sort()
    return rows


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

    # Stamps and addresses are parsed, and non-fault lines are ignored.
    text = ("[1234.500000] nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x4128000 flags=0x0007]\n"
            "[1235.000000] unrelated kernel line\n")
    assert faults(text) == [(1234.5, "0x4128000")], faults(text)
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", required=False, help="sampler log (tab-separated epoch/pid/…/ticket)")
    ap.add_argument("--dmesg", default="-", help="file of dmesg text, or '-' for stdin (default)")
    ap.add_argument("--boot-epoch", type=float, default=None,
                    help="epoch of boot, so kernel stamps become absolute (default: /proc/stat btime)")
    ap.add_argument("--max-gap", type=float, default=15.0,
                    help="oldest usable sample before a fault, in seconds (default 15)")
    ap.add_argument("--selftest", "--self-test", dest="self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.samples:
        ap.error("--samples is required unless --self-test")

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
        print("%s\t%s\t%s" % (when, addr, verdict))
    print(yield_line(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
