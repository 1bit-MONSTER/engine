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

Three rules, all learned by losing runs: **check a positive control first** (30 s of events; if the probe records
nothing, the probe is wrong, not the device); **never filter on an address list chosen minutes earlier** -- the
faulting IOVAs move faster than that (measured: a five-address filter saw 8 events, then 0 five minutes later); and
**count the probes back after cleanup, because the delete can lie**. Measured 2026-11-16 on one box, five tries:
a fresh event deletes; an event that recorded 810 or 1,148 events deletes; one that reported `echo: I/O error` was
**gone anyway**; and an event that had recorded ~40,000 events stayed armed through three deletes, a buffer clear and
a `free_buffer` kick -- `enable=0`, so it records nothing, but it holds a kprobe slot on `iommu_map` until reboot.
So: the error message is wrong in both directions, and **only `grep -c . kprobe_events` before and after tells you
whether you cleaned up**. Keep the capture as short as the question allows, and prefer the one probe the question
needs (the unmap side answers most of them) over two.

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
#
# The map/unmap distinction comes from the PROBED FUNCTION in the event text, not from the event's name: any lane
# names its probe differently (`capmap`, `unm`, `b2map`, `prs2mp`), and a tool that only reads its own naming
# cannot read the captures that already exist on the box. The function is the fact; the name is a convention.
EVENT = re.compile(r"(\d+\.\d+):\s+\S*:\s+\(([\w.]+)\+[^)]*\)\s+iova=(0x[0-9a-f]+)\s+size=(0x[0-9a-f]+)")
# Any line whose event text names a function and an iova. A function this file cannot classify is COUNTED, not
# dropped: `__iommu_map`, `iommu_map_pages` and `iommu_map.part.0` are maps the exact-name lists below never held,
# and a silent drop made a lane's map probe read as "0 map" -- which then produced the unanswerable claim in (1).
IOVA_LINE = re.compile(r"\(\s*([\w.]+)\+[^)]*\)\s+iova=")
# Documentation of what is known to exist. NOT the classifier: see classify().
MAP_FUNCS = ("iommu_map", "iommu_map_sg")
UNMAP_FUNCS = ("iommu_unmap", "iommu_unmap_fast")


def classify(fn):
    """`unmap` is tested first because every unmap name contains `map`."""
    if "unmap" in fn:
        return "unmap"
    if "map" in fn:
        return "map"
    return "other"
FAULT = re.compile(r"^\[\s*(\d+\.\d+)\].*IO_PAGE_FAULT.*address=(0x[0-9a-f]+)")
# The IOMMU's OWN tracepoints (engine#2678): `/sys/kernel/tracing/events/iommu/{map,unmap,io_page_fault}`.
# The class is the EVENT NAME, so this form needs no classifier at all -- which is why it is the better
# capture: a kprobe is a convention the text only half-satisfies, an event name is the kernel's statement.
#   wget-1394013 [012] .... 34862.494190: map: IOMMU: iova=0x3d207000 - 0x3d208000 paddr=0x187168a000 size=4096
#   irq/26-AMD-Vi-265 [003] .... 34226.857983: io_page_fault: IOMMU:nvme 0000:c4:00.0 iova=0xe5356000 flags=0x0000
TP_EVENT = re.compile(r"(\d+\.\d+):\s+(map|unmap|io_page_fault):\s+IOMMU:\s*(.*)$")
TP_IOVA = re.compile(r"iova=(0x[0-9a-f]+)")
TP_SIZE = re.compile(r"size=(0x[0-9a-f]+|\d+)")
TP_RANGE = re.compile(r"iova=(0x[0-9a-f]+)\s*-\s*(0x[0-9a-f]+)")


def _tp(event, rest):
    """(t, iova, size) from a tracepoint event body, or None when it carries no iova."""
    mi = TP_IOVA.search(rest)
    if not mi:
        return None
    iova = int(mi.group(1), 16)
    ms = TP_SIZE.search(rest)
    if ms:
        raw = ms.group(1)
        size = int(raw, 16) if raw.startswith("0x") else int(raw)
    else:
        mr = TP_RANGE.search(rest)
        size = (int(mr.group(2), 16) - iova) if mr else 0
    return iova, size


def parse_events(path):
    """([(t, iova, size)] maps, [...] unmaps, Counter unrecognised, Counter widened) -- in log order."""
    maps, unmaps, other, widened = [], [], collections.Counter(), collections.Counter()
    try:
        fh = open(path, errors="replace")
    except OSError as e:
        raise SystemExit(f"iommu_map_side: cannot read {path}: {e}")
    with fh:
        for line in fh:
            m = EVENT.search(line)
            if not m:
                tp = TP_EVENT.search(line)
                if tp:
                    got = _tp(tp.group(2), tp.group(3))
                    if got and tp.group(2) == "map":
                        maps.append((float(tp.group(1)), got[0], got[1]))
                    elif got and tp.group(2) == "unmap":
                        unmaps.append((float(tp.group(1)), got[0], got[1]))
                    continue
                loose = IOVA_LINE.search(line)
                if loose:
                    other[loose.group(1)] += 1
                continue
            t, fn, iova, size = float(m.group(1)), m.group(2), int(m.group(3), 16), int(m.group(4), 16)
            kind = classify(fn)
            if kind == "map":
                maps.append((t, iova, size))
            elif kind == "unmap":
                unmaps.append((t, iova, size))
            else:
                other[fn] += 1
            if (kind == "map" and fn not in MAP_FUNCS) or (kind == "unmap" and fn not in UNMAP_FUNCS):
                widened[fn] += 1
    return maps, unmaps, other, widened


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
                continue
            tp = TP_EVENT.search(line)
            if tp and tp.group(2) == "io_page_fault":
                got = _tp(tp.group(2), tp.group(3))
                if got:
                    out.append((float(tp.group(1)), got[0]))
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


def analyse(maps, unmaps, faults, before, after, other=(), widened=()):
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
            # Only a capture that probed BOTH sides may claim absence. A one-sided capture knows nothing about the
            # other half, and saying "never mapped" from an unmap-only trace is the instrument answering a question
            # it was never asked -- measured: 579,219 unmaps and 0 maps in scratch-i679/burst.trace read as
            # "NEVER MAPPED OR UNMAPPED" at rc 0.
            if not maps:
                rows.append((t, iova, "not in this capture's map side: no map events in %.0f s of capture, so whether "
                                      "this iova was ever mapped is not answerable here" % (hi - lo)))
            elif not unmaps:
                rows.append((t, iova, "not in this capture's unmap side: no unmap events in %.0f s of capture, so "
                                      "whether this iova was released is not answerable here" % (hi - lo)))
            else:
                rows.append((t, iova, "NEVER MAPPED OR UNMAPPED in %.0f s of capture: the translation is absent and "
                                      "nothing re-creates it" % (hi - lo)))
            continue
        parts = []
        for label, d, n in (("map", dm, nm), ("unmap", du, nu)):
            if d is None:
                # Two different facts print the same word if this is not split: a side that WAS probed and has no
                # event for this iova ("no map of this iova"), and a side that was never probed at all ("no map
                # probe in this capture"). On a one-sided capture the second is the truth, and writing the first
                # makes silence read as absence -- the same error the absence branch above now refuses.
                probed = bool(maps if label == "map" else unmaps)
                parts.append(f"no {label} of this iova" if probed
                             else f"no {label} probe in this capture, so that half is unanswerable (see exit code)")
            else:
                parts.append(f"nearest {label} {d * 1e6:+.0f} us (of {n})")
        rows.append((t, iova, " | ".join(parts)))
    hdr = (f"buffered {lo:.1f}..{hi:.1f} s ({hi - lo:.0f} s); {len(maps)} map, {len(unmaps)} unmap; "
           f"{len(rows)} fault(s) in the window")
    if other:
        hdr += "; %d events with an unrecognised function (e.g. %s)" % (sum(other.values()), next(iter(other)))
    if widened:
        hdr += "; %d events counted by substring but not in the documented names (e.g. %s)" % (
            sum(widened.values()), next(iter(widened)))
    if not maps:
        hdr += "; no map events in this capture (the map side is unanswerable)"
    elif not unmaps:
        hdr += "; no unmap events in this capture (the unmap side is unanswerable)"
    return rows, hdr


def render(faults_rows, header):
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

        maps, unmaps, _other, _w = parse_events(str(d / "cap.txt"))
        chk(len(maps) == 2 and len(unmaps) == 1, "a task name containing a space still parses (2 map, 1 unmap)")
        # 2b. the same events under ANOTHER lane's naming (`unm:`/`map:`/`prs2mp:`) parse identically, because the
        #     class comes from the probed function. A tool that only reads its own event names cannot read the
        #     captures that already exist on the box -- measured: 579,219 unmaps in ~/.cache/scratch-*/burst.trace
        #     are named `unm:`, and the first version of this file refused them.
        other = pathlib.Path(td) / "other.txt"
        other.write_text(cap.replace("capmap:", "whatever:").replace("capunm:", "unm:"))
        m2, u2, _o2, _w2 = parse_events(str(other))
        chk((m2, u2) == (maps, unmaps), "event names invented by another lane parse by the probed function, not the name")
        chk(all(e[2] == 0x1000 for e in maps + unmaps), "size parses as 4 KiB")

        # 2c. functions the exact-name lists never held are MAPS, and one this file cannot classify is COUNTED.
        #     A silent drop made a lane's probe read as "0 map", which is what produced the unanswerable claim below.
        odd = pathlib.Path(td) / "odd.txt"
        odd.write_text(
            "          <idle>-0       [001] ..s1. 1000.000000: p: (__iommu_map+0x4/0x80) iova=0x1000 size=0x1000\n"
            "          <idle>-0       [001] ..s1. 1001.000000: p: (iommu_map_pages+0x4/0x80) iova=0x2000 size=0x1000\n"
            "          <idle>-0       [001] ..s1. 1002.000000: p: (iommu_map.part.0+0x4/0x80) iova=0x3000 size=0x1000\n"
            "          <idle>-0       [001] ..s1. 1003.000000: p: (iommu_tlb_sync+0x4/0x80) iova=0x4000 size=0x1000\n")
        m4, u4, o4, w4 = parse_events(str(odd))
        chk(len(m4) == 3, "__iommu_map, iommu_map_pages and iommu_map.part.0 are all counted as maps (3)")
        chk(u4 == [] and o4 == {"iommu_tlb_sync": 1},
            "a function with neither word is counted, not dropped: %s" % dict(o4))
        chk(w4 == {"__iommu_map": 1, "iommu_map_pages": 1, "iommu_map.part.0": 1},
            "the three widened names are surfaced with their counts, not just counted as maps: %s" % dict(w4))
        # a name that contains `map` and is NOT an IOMMU map: the substring rule catches it, so the header must
        # say so. A denylist would be a guess; a number a reader can see is not.
        odd2 = pathlib.Path(td) / "odd2.txt"
        odd2.write_text("          <idle>-0       [001] ..s1. 1000.000000: p: (bitmap_alloc+0x4/0x80) iova=0x1000 size=0x1000\n")
        m6, _u6, _o6, w6 = parse_events(str(odd2))
        chk(len(m6) == 1 and w6 == {"bitmap_alloc": 1},
            "a `map`-containing non-IOMMU name is counted AND surfaced (%s), so the map side cannot be inflated silently"
            % dict(w6))
        _r4, hdr4 = analyse(m4, u4, [(1004.0, 0x9999000)], [], [(1004.0, 0x9999000)], o4, w4)
        chk("1 events with an unrecognised function (e.g. iommu_tlb_sync)" in hdr4,
            "the unrecognised event is named in the header: %s" % hdr4)
        chk("3 events counted by substring but not in the documented names" in hdr4,
            "the widened names are surfaced in the header: %s" % hdr4)

        # 2d. ONE-SIDED captures: an unmap-only trace cannot say a page was never mapped, and says so.
        rows5, hdr5 = analyse([], [(1000.0, 0x4128000, 0x1000), (1002.0, 0x5128000, 0x1000)],
                              [(1001.0, 0xdeadb000)], [], [(1001.0, 0xdeadb000)])
        chk("map side" in rows5[0][2] and "not answerable" in rows5[0][2],
            "an unmap-only capture refuses the map-side claim: %s" % rows5[0][2][:60])
        chk("NEVER MAPPED OR UNMAPPED" not in rows5[0][2], "...and does not say NEVER MAPPED")
        chk("no map events in this capture" in hdr5, "...and says it in the header too")
        chk(exit_code([], [(1, 2, 3)]) == 4, "a one-sided capture exits 4, not 0")
        chk(exit_code(maps, unmaps) == 0 and exit_code([], []) == 3, "both sides 0, nothing captured 3")

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

        # 7b. a one-sided capture: the row answers on the side it probed AND says the other half was never asked about.
        #     Measured: burst.trace has 579,219 unmaps and no map probe, and its rows used to read "no map of this
        #     iova" -- which says the iova was never mapped rather than that nobody looked.
        # two unmap events bracket the fault so it is inside the buffered range: a single event gives lo == hi and
        # the row (correctly) reports "OUTSIDE" instead, which would make this arm pass for the wrong reason.
        rows4, _h4 = analyse([], [(1000.5, 0x4128000, 0x1000), (1002.0, 0x5128000, 0x1000)],
                            [(1001.0, 0x4128000)], [], [(1001.0, 0x4128000)])
        txt = " | ".join(r[2] for r in rows4)
        chk("no map probe in this capture" in txt and "no map of this iova" not in txt,
            "a row on a one-sided capture says the other half was never probed")

        # a fault outside the buffered range is named as such, not silently correlated
        rows3, _ = analyse(maps, unmaps, [(9000.0, 0x4128000)], [], [(9000.0, 0x4128000)])
        chk("OUTSIDE the buffered range" in rows3[0][2], "a fault the ring does not cover says so")
        # 2e. the IOMMU's own tracepoints (engine#2678): the event name IS the class, and the fault carries
        #     `iova=` where journalctl's IO_PAGE_FAULT carries `address=`. Both forms must read the same way.
        tp = pathlib.Path(td) / "tp.txt"
        tp.write_text(
            "wget-1394013 [012] .... 34862.494190: map: IOMMU: iova=0x3d207000 - 0x3d208000 paddr=0x187168a000 size=4096\n"
            "wget-1394013 [012] .... 34862.494290: unmap: IOMMU: iova=0x3d207000 size=4096\n"
            "irq/26-AMD-Vi-265 [003] .... 34862.494250: io_page_fault: IOMMU:nvme 0000:c4:00.0 iova=0x3d207000 flags=0x0000\n"
            "wget-1394013 [012] .... 34862.494300: map: IOMMU: iova=0x4e000000 size=0x1000\n")
        m7, u7, o7, w7 = parse_events(str(tp))
        chk(len(m7) == 2 and len(u7) == 1 and o7 == {} and w7 == {},
            "tracepoint map/unmap parse by EVENT NAME, nothing unrecognised (2 map, 1 unmap) -- %s %s" % (o7, w7))
        chk(m7[0][2] == 4096 and m7[1][2] == 4096,
            "tracepoint size parses from size= and from size=0x1000 (%d, %d)" % (m7[0][2], m7[1][2]))
        f7 = parse_faults(str(tp))
        chk(f7 == [(34862.49425, 0x3D207000)], "tracepoint io_page_fault parses its iova (not journalctl's address=)")
        rows7, hdr7 = analyse(m7, u7, f7, [], f7)
        chk(len(rows7) == 1 and "nearest map" in rows7[0][2],
            "a tracepoint capture correlates: %s" % (rows7[0][2][:70] if rows7 else "no rows"))

        # 3. the ASSEMBLY: main() is what a lane runs, and until this arm nothing exercised it end to end --
        #    the exit code and the header shape of a one-sided capture were asserted only piecewise.
        import contextlib
        import io
        one_sided = d / "one_sided.txt"
        one_sided.write_text(
            "          <idle>-0       [001] ..s1. 1000.000000: capunm: (iommu_unmap_fast+0x4/0x10) iova=0x4128000 size=0x1000\n"
            "          <idle>-0       [001] ..s1. 1002.000000: capunm: (iommu_unmap_fast+0x4/0x10) iova=0x5128000 size=0x1000\n")
        (d / "f.txt").write_text(
            "[1001.000000] strixhalo kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0xdeadb000 flags=0x0000]\n")
        (d / "none.txt").write_text("")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc_main = main(["--trace", str(one_sided), "--before", str(d / "none.txt"), "--after", str(d / "f.txt")])
        printed = buf.getvalue()
        chk(rc_main == 4 and "no map events in this capture" in printed and "not answerable here" in printed,
            "main() on an unmap-only capture prints the map-side refusal and exits 4 (rc=%d)" % rc_main)
        chk("NEVER MAPPED OR UNMAPPED" not in printed, "...and main() never prints the absence claim for it")

    return 0 if ok else 1


def exit_code(maps, unmaps):
    """0 both sides probed; 3 nothing captured; 4 ONE side absent, so the missing half cannot be claimed."""
    if not maps and not unmaps:
        return 3
    if not maps or not unmaps:
        return 4
    return 0


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
    maps, unmaps, other, widened = parse_events(a.trace)
    rows, header = analyse(maps, unmaps, parse_faults(a.after), parse_faults(a.before), parse_faults(a.after),
                           other, widened)
    for line in render(rows, header):
        print(line)
    return exit_code(maps, unmaps)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
