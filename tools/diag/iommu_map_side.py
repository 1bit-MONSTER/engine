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
"""Ask the map side about each IO_PAGE_FAULT: was that IOVA ever mapped, and when.

`tools/diag/unmap_fault_order.py` asks when the page was unmapped. This asks the other half: for every fault in
the capture window, the nearest `iommu_map` and `iommu_unmap_fast` of the **same IOVA**, with how many there were
and whether the window even covers the fault. That is what separated "a stale translation" (which engine#2478
assumed for a day) from the answer: on 2026-11-16 an unfiltered 795 s capture held 252,704 map/unmap events, and
**four of eight faults were on an IOVA that was never mapped and never unmapped anywhere in the window** -- an
absent translation that nothing re-creates, not a stale entry.

The capture this reads is three files, and the recipe matters more than the parsing:

  TR=/sys/kernel/tracing
  echo 0 > $TR/tracing_on; echo > $TR/trace
  echo 'p:kprobes/capmap iommu_map iova=%si size=%cx'     >> $TR/kprobe_events   # %cx is SIZE; %dx is paddr
  echo 'p:kprobes/capunm iommu_unmap_fast iova=%si size=%dx' >> $TR/kprobe_events
  echo 1 > $TR/events/kprobes/capmap/enable; echo 1 > $TR/events/kprobes/capunm/enable
  journalctl -k -b 0 -o short-monotonic > before.txt
  echo 1 > $TR/tracing_on; sleep 300; echo 0 > $TR/tracing_on
  cat $TR/trace > capture.txt; journalctl -k -b 0 -o short-monotonic > after.txt

Two rules, both learned by losing runs: **check a positive control first** (30 s of events; if the probe records
nothing, the probe is wrong, not the device), and **never filter on an address list chosen minutes earlier** --
the faulting IOVAs move faster than that (measured: a five-address filter saw 8 events, then 0 five minutes later).

Usage:
  tools/diag/iommu_map_side.py --trace capture.txt --before before.txt --after after.txt
  tools/diag/iommu_map_side.py --self-test
"""
from __future__ import annotations

import argparse
import bisect
import collections
import re
import sys

# Anchored on the timestamp, NOT on the task name: a task name can contain a space ("next-server (v1-8386"), and
# anchoring on `^\s+\S+\s+\[` silently dropped 2,176 of 115,941 map lines on a real capture -- 1.9 %, which is
# exactly the fraction that can contain the event being looked for.
EVENT = re.compile(r"(\d+\.\d+):\s+\S*cap(map|unm):\s+\([^)]*\)\s+iova=(0x[0-9a-f]+)\s+size=(0x[0-9a-f]+)")
FAULT = re.compile(r"^\[\s*(\d+\.\d+)\].*IO_PAGE_FAULT.*address=(0x[0-9a-f]+)")


def parse_events(path):
    """([(t, iova, size)] maps, [(t, iova, size)] unmaps) -- insertion order, which is log order."""
    maps, unmaps = [], []
    try:
        fh = open(path, errors="replace")
    except OSError as e:
        raise SystemExit(f"iommu_map_side: cannot read {path}: {e}")
    with fh:
        for line in fh:
            m = EVENT.search(line)
            if not m:
                continue
            t, kind, iova, size = float(m.group(1)), m.group(2), int(m.group(3), 16), int(m.group(4), 16)
            (maps if kind == "map" else unmaps).append((t, iova, size))
    return maps, unmaps


def parse_faults(path):
    """[(boot_seconds, iova)] for every IO_PAGE_FAULT line in a `journalctl -o short-monotonic` capture."""
    out = []
    try:
        fh = open(path, errors="replace")
    except OSError as e:
        raise SystemExit(f"iommu_map_side: cannot read {path}: {e}")
    with fh:
        for line in fh:
            m = FAULT.match(line)
            if m:
                out.append((float(m.group(1)), int(m.group(2), 16)))
    return out


def _nearest(times, t):
    """(delta_seconds or None, count of events for this iova)"""
    if not times:
        return None, 0
    i = bisect.bisect_left(times, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(times):
            d = times[j] - t
            if best is None or abs(d) < abs(best):
                best = d
    return best, len(times)


def analyse(maps, unmaps, faults, before, after):
    """[(t, iova, text)] -- one line per fault inside the capture window, in the words the census reads."""
    if not maps and not unmaps:
        return [], "no capture events: the probe recorded nothing, so this file says nothing about the device"
    times = [e[0] for e in maps + unmaps]
    lo, hi = min(times), max(times)
    win = {f for f in after if f not in set(before)}
    by_map = collections.defaultdict(list)
    for t, iova, _s in sorted(maps):
        by_map[iova].append(t)
    by_unm = collections.defaultdict(list)
    for t, iova, _s in sorted(unmaps):
        by_unm[iova].append(t)
    rows = []
    for t, iova in win:
        if not (lo <= t <= hi):
            rows.append((t, iova, "OUTSIDE the buffered range: the ring held %.0f s and this fault is not in it"
                                      % (hi - lo)))
            continue
        dm, nm = _nearest(by_map.get(iova), t)
        du, nu = _nearest(by_unm.get(iova), t)
        if dm is None and du is None:
            rows.append((t, iova, "NEVER MAPPED OR UNMAPPED in %.0f s of capture: the translation is absent and "
                                  "nothing re-creates it" % (hi - lo)))
            continue
        parts = []
        for label, d, n in (("map", dm, nm), ("unmap", du, nu)):
            if d is None:
                parts.append(f"no {label} of this iova")
            else:
                parts.append(f"nearest {label} {d * 1e6:+.0f} us (of {n})")
        rows.append((t, iova, " | ".join(parts)))
    return rows, (f"buffered {lo:.1f}..{hi:.1f} s ({hi - lo:.0f} s); {len(maps)} map, {len(unmaps)} unmap; "
                  f"{len(rows)} fault(s) in the window")


def render(maps, unmaps, before, after, faults_rows, header):
    lines = [header]
    for t, iova, text in faults_rows:
        lines.append(f"  t={t:.3f} iova={iova:#x}: {text}")
    if not faults_rows:
        lines.append("  no fault inside the capture window: this capture cannot answer, whatever it contains")
    return lines


def self_test() -> int:
    import tempfile
    import pathlib

    ok = True

    def chk(cond, what):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + what)
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        # A task name WITH A SPACE, which is the shape that silently dropped 1.9 % of lines on a real capture.
        cap = (
            " next-server (v1-8386    [000] ..... 1000.000000: capmap: (iommu_map+0x4/0x80) iova=0x4128000 size=0x1000\n"
            "          <idle>-0       [001] ..s1. 1000.500000: capunm: (iommu_unmap_fast+0x4/0x10) iova=0x4128000 size=0x1000\n"
            "          <idle>-0       [001] ..s1. 1005.000000: capmap: (iommu_map+0x4/0x80) iova=0x5128000 size=0x1000\n"
        )
        (d / "cap.txt").write_text(cap)
        before = "[1000.000000] kernel: x\n"
        after = (
            before
            + "[1001.000000] strixhalo kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x4128000 flags=0x0000]\n"
            + "[1002.000000] strixhalo kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x9999000 flags=0x0000]\n"
        )
        (d / "before.txt").write_text(before)
        (d / "after.txt").write_text(after)

        maps, unmaps = parse_events(str(d / "cap.txt"))
        chk(len(maps) == 2 and len(unmaps) == 1, "a task name containing a space still parses (2 map, 1 unmap)")
        chk(all(e[2] == 0x1000 for e in maps + unmaps), "size parses as 4 KiB")

        faults = parse_faults(str(d / "after.txt"))
        chk(len(faults) == 2, "two faults parse with their boot stamps")

        # the fault on the mapped-and-unmapped page: both neighbours reported, in microseconds
        dm, nm = _nearest([1000.0], 1001.0)
        chk(dm == -1.0 and nm == 1, "nearest map is 1 s before the fault, and it is counted")
        rows, _hdr = analyse(maps, unmaps, faults, parse_faults(str(d / "before.txt")), faults)
        by_iova = {r[1]: r[2] for r in rows}
        chk("nearest map -1000000 us" in by_iova[0x4128000], "a mapped page reports its nearest map and unmap")
        chk("NEVER MAPPED OR UNMAPPED" in by_iova[0x9999000],
            "a page with no map or unmap anywhere in the capture is named as absent, not as uncorrelated")

        # an empty capture must not read as "the device is fine"
        empty, hdr2 = analyse([], [], faults, [], faults)
        chk(empty == [] and "says nothing about the device" in hdr2,
            "an empty capture is refused as evidence rather than reported as a negative")

        # a fault outside the buffered range is named as such, not silently correlated
        rows3, _ = analyse(maps, unmaps, [(9000.0, 0x4128000)], [], [(9000.0, 0x4128000)])
        chk("OUTSIDE the buffered range" in rows3[0][2], "a fault the ring does not cover says so")
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace")
    ap.add_argument("--before")
    ap.add_argument("--after")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.trace and a.before and a.after):
        ap.print_help()
        return 2
    maps, unmaps = parse_events(a.trace)
    rows, header = analyse(maps, unmaps, parse_faults(a.after), parse_faults(a.before), parse_faults(a.after))
    for line in render(maps, unmaps, a.before, a.after, rows, header):
        print(line)
    if not maps and not unmaps:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
