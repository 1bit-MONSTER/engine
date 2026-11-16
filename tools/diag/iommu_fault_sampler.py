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

    <epoch> <printed_fault_lines> <suppressed_total>

    # one interval every 5s, forever (root: reading dmesg)
    sudo tools/diag/iommu_fault_sampler.py --interval 5 --out /var/log/iommu-fault-rate.log

    # a bounded sample, for a window run
    tools/diag/iommu_fault_sampler.py --interval 2 --count 10 --dmesg "cat fixture.txt"

`suppressed_total` is `census.suppressed(text)["amd_iommu_report_page_fault"]`, the IOMMU fault
reporter's own counter and no other caller's: the same journal carries `kauditd_printk_skb` and
`set_capacity_and_notify` suppression lines (267 and 2 on the 11-15 boot, beside the IOMMU's 5699), and
summing every caller counted an audit burst as dropped faults. `--caller` names another reporter. The
counters grow monotonically, so the DELTA across two lines is the number of faults the kernel dropped
in that interval. A rising
suppressed_total with a flat printed count is the normal case under load and is not a quiet system --
that is the whole point of keeping it.

Nothing here talks to a device, and `--dmesg` takes any command, so the self-test runs on fixtures.
"""

import argparse
import importlib.util
import os
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


def describe(prev, cur):
    """One interval's meaning, so the log does not need a human to interpret it later."""
    if prev is None:
        return "baseline"
    d_printed = cur[0] - prev[0]
    d_dropped = cur[1] - prev[1]
    if d_dropped < 0:
        # The ring buffer discarded earlier suppression lines, so the total fell. That is not a negative
        # rate; it is the buffer forgetting. Re-baseline instead of reporting a number that never happened.
        return f"rotated ({d_dropped} from total), re-baselined"
    if d_dropped > 0:
        # The dropped count is the only quantity that cannot be throttled away.
        return f"FAULTING +{d_dropped} dropped, +{d_printed} printed"
    if d_printed > 0:
        return f"faulting +{d_printed} printed, none dropped"
    return "quiet"


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

    # Because the total SUMS, it can go backwards when the buffer drops an old suppression line. An
    # interval must not be rendered as a negative fault count; it means the buffer forgot.
    assert describe((5, 11311), (5, 5611)) == "rotated (-5700 from total), re-baselined"

    # Only the IOMMU reporter's counter is a dropped fault. The 11-15 journal had kauditd (267) and
    # set_capacity (2) suppression lines beside it; summing every caller turned an audit burst into
    # "FAULTING +267 dropped" with no fault at all (review of engine#2580).
    others = ("kernel: kauditd_printk_skb: 267 callbacks suppressed\n"
              "kernel: set_capacity_and_notify: 2 callbacks suppressed\n")
    assert sample(census, burst + others) == (2, 5611), sample(census, burst + others)
    assert describe(sample(census, burst), sample(census, burst + others)) == "quiet"
    assert sample(census, others, caller="kauditd_printk_skb") == (0, 267)

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
    out = sys.stdout if args.out == "-" else open(args.out, "a", buffering=1)
    prev, n = None, 0
    try:
        while not args.count or n < args.count:
            text = subprocess.run(args.dmesg, shell=True, capture_output=True, text=True).stdout
            cur = sample(census, text, args.caller)
            print(f"{int(time.time())} {cur[0]} {cur[1]} {describe(prev, cur)}", file=out)
            prev, n = cur, n + 1
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
