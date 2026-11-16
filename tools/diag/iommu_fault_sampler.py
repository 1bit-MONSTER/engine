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
"""Record the IOMMU fault RATE, because `dmesg` cannot be counted.

engine#2542 quoted three different populations for one boot -- 20, 78, 41 faults -- and all three were
true of the buffer at the moment it was read, because `printk_ratelimited` drops most of the stream and
the ring buffer discards the rest. A census taken once cannot say how fast it is happening, and a
throttled log cannot say how many times it happened; only two samples in order can.

So this appends one line per interval:

    <epoch> <printed_fault_lines> <suppressed_total> <meaning>

Both counts are cumulative since the sampler started, built from the lines that are NEW in each snapshot
(kernel stamp `[ s.us ]` later than the last one seen), so the ring buffer rotating cannot change them: the
delta between two lines is exactly what the kernel printed, and dropped, in that interval. Counting the
whole buffer each time instead let a rotation hide new drops (an old suppression line leaving while new
ones arrive read as "rotated", or as a smaller rise -- review of engine#2580). That needs `dmesg`'s
default `[ s.us ]` prefix (`dmesg`, `journalctl -k -o short-monotonic`). A source with no stamps at all
(`journalctl -k`'s default format, `dmesg -T`) cannot be ordered, so it is counted whole each time as
before, and the baseline line says which mode is in use (see Tally).

    # one interval every 5s, forever (root: reading dmesg)
    sudo tools/diag/iommu_fault_sampler.py --interval 5 --out /var/log/iommu-fault-rate.log

    # a bounded sample, for a window run
    tools/diag/iommu_fault_sampler.py --interval 2 --count 10 --dmesg "cat fixture.txt"

`suppressed_total` sums `census.suppressed(new_lines)["amd_iommu_report_page_fault"]`, the IOMMU fault
reporter's own counter and no other caller's: the same journal carries `kauditd_printk_skb` and
`set_capacity_and_notify` suppression lines (267 and 2 on the 11-15 boot, beside the IOMMU's 5699), and
summing every caller counted an audit burst as dropped faults. `--caller` names another reporter. Each
suppression message reports what was dropped since the previous one, so the new lines' counts SUM to the
interval's drops. A rising
suppressed_total with a flat printed count is the normal case under load and is not a quiet system --
that is the whole point of keeping it.

Nothing here talks to a device, and `--dmesg` takes any command, so the self-test runs on fixtures.
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_census():
    """Import the sibling census module, sharing its `suppressed()` instead of re-deriving the regex."""
    spec = importlib.util.spec_from_file_location("iommu_fault_census",
                                                  os.path.join(_HERE, "iommu_fault_census.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The printk_ratelimited caller that reports AMD-Vi IO_PAGE_FAULT events. Other callers' counters (audit,
# block-device resize, ...) are in the same buffer and are NOT dropped faults (review of engine#2580).
IOMMU_CALLER = "amd_iommu_report_page_fault"


def sample(census, text, caller=IOMMU_CALLER):
    """(printed fault lines, dropped-callback total for `caller`) for one dmesg snapshot."""
    printed = sum(1 for line in text.splitlines() if "IO_PAGE_FAULT" in line)
    return printed, census.suppressed(text).get(caller, 0)


# dmesg's default prefix, `[  1234.567890] `: the only order the lines carry
STAMP = re.compile(r"^\[\s*(\d+\.\d+)\]")


class Tally:
    """Cumulative (printed, dropped) since the first snapshot.

    The first snapshot picks the mode, because the two sources on this box differ exactly here:
      * "cursor" -- the source carries dmesg's `[ s.us ]` stamp (`dmesg`, `journalctl -k -o short-monotonic`).
        A line counts once: every line in the baseline, after that only lines stamped later than the newest
        stamp seen. Lines that leave the buffer are not seen again, so a rotation can neither subtract from the
        totals nor mask what arrived meanwhile. If every stamp in a snapshot is older than the newest one seen,
        the clock restarted (a reboot under a persistent source): that snapshot is counted as a new buffer, and
        the interval says so instead of freezing at the old totals.
      * "buffer" -- no line carries a stamp (`journalctl -k` in its default format, `dmesg -T`). Nothing can be
        ordered, so each snapshot is counted whole, as before engine#2603: a rotation can hide drops, and the
        interval says `rotated` when the total falls. The mode is announced on the baseline line, because a
        cursor that never advances would read `quiet` for ever (review of engine#2603).
    """

    def __init__(self, census, caller=IOMMU_CALLER):
        self.census, self.caller = census, caller
        self.mode = None            # "cursor" or "buffer", set by the first snapshot
        self.last = None            # newest kernel stamp counted (cursor mode)
        self.printed = self.dropped = 0
        self.unstamped = 0          # cursor mode: fault/suppression lines in the last snapshot with no stamp
        self.event = ""             # what the last add() noticed, for the interval's meaning

    def add(self, text):
        lines = text.splitlines()
        stamped = [(float(m.group(1)), line) for line in lines for m in [STAMP.match(line)] if m]
        self.event = ""
        if self.mode is None:
            self.mode = "cursor" if stamped else "buffer"
        if self.mode == "buffer":
            self.printed, self.dropped = sample(self.census, text, self.caller)
            return self.printed, self.dropped
        if self.last is None:
            new = lines                                   # baseline: nothing to order against yet
        elif stamped and max(t for t, _ in stamped) < self.last:
            self.event = "stamps went backwards (reboot?): counted the new buffer"
            new = [line for _, line in stamped]
            self.last = None
        else:
            new = [line for t, line in stamped if t > self.last]
        self.unstamped = 0 if self.last is None else sum(
            1 for line in lines if not STAMP.match(line) and ("IO_PAGE_FAULT" in line or "callbacks suppressed" in line))
        if stamped:
            newest = max(t for t, _ in stamped)
            self.last = newest if self.last is None else max(self.last, newest)
        p, d = sample(self.census, "\n".join(new), self.caller)
        self.printed += p
        self.dropped += d
        return self.printed, self.dropped


def describe(prev, cur):
    """One interval's meaning, so the log does not need a human to interpret it later."""
    if prev is None:
        return "baseline"
    d_printed = cur[0] - prev[0]
    d_dropped = cur[1] - prev[1]
    if d_dropped < 0:
        # buffer mode only (no stamps to order by): the ring buffer discarded earlier suppression lines, so the
        # total fell. That is not a negative rate; it is the buffer forgetting, and drops in it are unknown.
        return f"rotated ({d_dropped} from total), re-baselined; drops in this interval unknown"
    if d_dropped > 0:
        # The dropped count is the only quantity that cannot be throttled away.
        return f"FAULTING +{d_dropped} dropped, +{d_printed} printed"
    if d_printed > 0:
        return f"faulting +{d_printed} printed, none dropped"
    return "quiet"


def step(tally, prev, cmd):
    """One sample: (the line after the epoch, the totals to carry as `prev`).

    A source that fails is not an empty buffer. `dmesg` as a non-root user on this box exits 1 with
    `read kernel buffer failed: Operation not permitted` and prints nothing, which counted as a buffer with
    no faults read "quiet" -- a healthy device from a sampler that read nothing (review of engine#2603). So a
    non-zero exit is reported as `unreadable: <its first stderr line>`, is not counted, and leaves the
    tally and `prev` as they were; the next readable sample's interval then spans the gap."""
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        why = (p.stderr.strip().splitlines() or [f"exit {p.returncode}"])[0][:120]
        return f"- - unreadable: {why} (not counted)", prev
    cur = tally.add(p.stdout)
    meaning = describe(prev, cur)
    if prev is None:
        meaning += (" (mode: cursor on kernel stamps)" if tally.mode == "cursor" else
                    " (mode: whole buffer -- no kernel stamps in this source, so a rotation can hide drops;"
                    " dmesg or journalctl -k -o short-monotonic carry them)")
    if tally.event:
        meaning += "; " + tally.event
    if tally.mode == "cursor" and tally.unstamped:
        meaning += f"; {tally.unstamped} unstamped line(s) in the buffer not counted"
    return f"{cur[0]} {cur[1]} {meaning}", cur


def self_test():
    census = _load_census()
    F = ("kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 "
         "address=0x{addr} flags=0x0000]\n")
    quiet = "kernel: amdxdna 0000:c6:00.1: nothing to report\n"
    ratelimited = "kernel: amd_iommu_report_page_fault: 5611 callbacks suppressed\n"

    # A quiet snapshot has no lines and no counter.
    assert sample(census, quiet) == (0, 0)

    # Printed lines are counted only from fault lines, and the suppressed total is the counter value.
    burst = quiet + F.format(addr="4128000") + F.format(addr="4129000") + ratelimited
    assert sample(census, burst) == (2, 5611), sample(census, burst)

    # The interval's meaning is decided by the DROPPED delta, not the printed delta: a frozen printed
    # count with a growing counter is the case dmesg-only analysis reads as silence.
    assert describe(None, (2, 5611)) == "baseline"
    assert describe((2, 5611), (2, 5611)) == "quiet"
    assert describe((2, 5611), (2, 9000)) == "FAULTING +3389 dropped, +0 printed"
    assert describe((2, 5611), (9, 5611)) == "faulting +7 printed, none dropped"

    # Two suppression lines for the same caller SUM, because each message reports the count dropped
    # since the previous one -- `census.suppressed()` adds per caller on purpose. I wrote this assertion
    # the other way round first, and the self-test caught it.
    two = ratelimited + "kernel: amd_iommu_report_page_fault: 5700 callbacks suppressed\n"
    assert sample(census, two) == (0, 11311), sample(census, two)

    # Rotation (review of engine#2580): counting the whole buffer each time, an old suppression line
    # leaving in the same interval as new ones arrive read as "rotated" (3000 new drops shown as none) or as
    # a smaller rise (6000 shown as 389). The tally counts only lines stamped after the last one seen, so
    # the interval is the new lines alone, whatever left the buffer.
    S = "[%11.6f] "
    b0 = S % 1.0 + F.format(addr="4128000") + S % 2.0 + ratelimited
    for new_drops in (3000, 6000):
        t = Tally(census)
        before = t.add(b0)
        b1 = S % 6.0 + "kernel: amd_iommu_report_page_fault: %d callbacks suppressed\n" % new_drops  # b0 rotated out
        after = t.add(b1)
        assert (before, after) == ((1, 5611), (1, 5611 + new_drops)), (before, after)
        assert describe(before, after) == "FAULTING +%d dropped, +0 printed" % new_drops, describe(before, after)
    # A snapshot that still holds the old lines does not count them twice, and a fault line that rotates
    # out never makes the printed count fall.
    t = Tally(census)
    t.add(b0)
    assert t.add(b0) == (1, 5611), t.add(b0)
    assert t.add(S % 7.0 + quiet) == (1, 5611)
    # After the baseline an unstamped fault line cannot be ordered: it is not counted, and it is reported.
    t.add(F.format(addr="4130000"))
    assert (t.mode, t.printed, t.unstamped) == ("cursor", 1, 1), (t.mode, t.printed, t.unstamped)

    # journalctl's default format carries no `[ s.us ]` on any line (0 of 6140 on 11-16). A cursor would never
    # advance and read "quiet" for ever (review of engine#2603), so a source with no stamps counts the buffer whole.
    J = "Nov 15 10:00:0%d h kernel: "
    j0 = J % 0 + F.format(addr="4128000") + J % 1 + ratelimited
    j1 = j0 + J % 3 + F.format(addr="4129000") + J % 4 + F.format(addr="4129000") + \
        J % 5 + "kernel: amd_iommu_report_page_fault: 150 callbacks suppressed\n"
    t = Tally(census)
    before, after = t.add(j0), t.add(j1)
    assert t.mode == "buffer" and (before, after) == ((1, 5611), (3, 5761)), (t.mode, before, after)
    assert describe(before, after) == "FAULTING +150 dropped, +2 printed", describe(before, after)
    # ... and in that mode a rotation is named, never a negative count or "quiet"
    assert describe((3, 5761), (2, 150)).startswith("rotated (-5611 from total)"), describe((3, 5761), (2, 150))

    # A clock restart (every stamp older than the newest seen) is counted as a new buffer and named, instead of
    # freezing the totals at their old values while the new boot faults.
    t = Tally(census)
    t.add(S % 500.0 + F.format(addr="4128000") + S % 501.0 + ratelimited)
    after = t.add(S % 3.0 + F.format(addr="4129000") + S % 4.0 + "kernel: amd_iommu_report_page_fault: 7 callbacks suppressed\n")
    assert after == (2, 5618) and t.event.startswith("stamps went backwards"), (after, t.event)

    # Only the IOMMU reporter's counter is a dropped fault. The 11-15 journal had kauditd (267) and
    # set_capacity (2) suppression lines beside it; summing every caller turned an audit burst into
    # "FAULTING +267 dropped" with no fault at all (review of engine#2580).
    others = ("kernel: kauditd_printk_skb: 267 callbacks suppressed\n"
              "kernel: set_capacity_and_notify: 2 callbacks suppressed\n")
    assert sample(census, burst + others) == (2, 5611), sample(census, burst + others)
    assert describe(sample(census, burst), sample(census, burst + others)) == "quiet"
    assert sample(census, others, caller="kauditd_printk_skb") == (0, 267)

    # A failing source is "unreadable", never "quiet", and leaves the tally alone; an empty one that succeeded is
    # a real empty buffer. Both print nothing to stdout -- only the exit status tells them apart (review of #2603).
    t = Tally(census)
    line, after = step(t, None, "echo 'dmesg: read kernel buffer failed: Operation not permitted' >&2; exit 1")
    assert line.startswith("- - unreadable: dmesg: read kernel buffer failed") and after is None, (line, after)
    assert t.mode is None, t.mode                     # a failed read must not choose the mode either
    line, after = step(t, (2, 5611), "false")
    assert line == "- - unreadable: exit 1 (not counted)" and after == (2, 5611), (line, after)
    line, after = step(t, None, "true")
    assert line.startswith("0 0 baseline") and after == (0, 0), (line, after)
    assert step(t, after, "true")[0] == "0 0 quiet", step(t, after, "true")

    # The fixture above is the shape that misled engine#2542: two devices, one log format, so the
    # printed count mixes populations. The sampler does not classify by device -- the census does --
    # it only measures rate, which is why it can stay this small.
    print("self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between samples")
    ap.add_argument("--count", type=int, default=0, help="samples to take (0 = forever)")
    ap.add_argument("--out", default="-", help="log file to append to ('-' = stdout)")
    ap.add_argument("--dmesg", default="dmesg", help="command whose stdout is the dmesg text")
    ap.add_argument("--caller", default=IOMMU_CALLER,
                    help="the printk_ratelimited caller whose suppressed count is the dropped-fault count")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    census = _load_census()
    tally = Tally(census, args.caller)
    out = sys.stdout if args.out == "-" else open(args.out, "a", buffering=1)
    prev, n = None, 0
    try:
        while not args.count or n < args.count:
            line, prev = step(tally, prev, args.dmesg)
            print(f"{int(time.time())} {line}", file=out)
            n += 1
            if args.count and n >= args.count:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
