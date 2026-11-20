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
"""IOMMU DMA lifecycle: which owner holds a faulting IOVA, and how the fault sits
in the map/unmap cycle. Engine#2478.

Two questions, two modes, because they are answered by different instruments:

  --lifecycle TRACE   parses a kprobe trace of the DMA teardown and reports, per
                      page: the map->unmap->map cycle count, the unmapped dwell
                      distribution, and how long before each fault the page was
                      last unmapped / when it was next mapped.

  --owners IOVA...    reverse-maps physical pages (identity IOVA, i.e.
                      CONFIG_IOMMU_DEFAULT_DMA_LAZY) to (pid, va, region) by
                      reading /proc/*/pagemap over the ranges /proc/*/maps
                      reports. Needs CAP_SYS_ADMIN for pagemap.

Both refuse to answer "nobody"/"no fault" when their coverage is partial: a
partial answer is exit 2 (cannot tell), never exit 1 (nobody). The distinction is
the whole point -- an instrument that cannot distinguish "absent" from "not
looked at" is what produced three wrong readings on this issue.

The probes that produce TRACE (verified on this box: io/iommu tracepoints exist
but never fire, so kprobes are required):

    echo 0 > tracing_on; echo > trace; echo > kprobe_events
    echo 'p:mp iommu_map         dom=$arg1 iova=$arg2:u64' >> kprobe_events
    echo 'p:uf iommu_unmap_fast  dom=$arg1 iova=$arg2:u64' >> kprobe_events
    echo 'p:du __iommu_dma_unmap dev=$arg1 iova=$arg2:u64' >> kprobe_events
    echo 'p:pf report_iommu_fault dom=$arg1 dev=$arg2 iova=$arg3:u64' >> kprobe_events
    echo 65536 > buffer_size_kb; echo 1 > events/kprobes/enable; echo 1 > tracing_on
    ... window ...
    cat trace > TRACE
"""
import argparse
import ctypes
import os
import re
import struct
import sys

PAGE = 4096
ENTRY = 8
CHUNK = 1 << 20

TRACE = re.compile(r"\s*(\S+?)-(\d+)\s+\[\d+\].*?\s(\d+\.\d+):\s(mp|uf|du|pf):.*?iova=(\d+)")


def load_trace(path):
    """-> {iova: [(t, kind)]}, sorted by time. Kinds: mp=map, uf/du=unmap, pf=fault."""
    ev = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = TRACE.search(line)
            if m:
                ev.setdefault(int(m.group(5)), []).append(
                    (float(m.group(3)), m.group(4), m.group(1)))
    for v in ev.values():
        v.sort(key=lambda e: e[0])
    return ev


def cycles(events):
    """Unmapped intervals: time from an unmap to the next map of the same page.

    du (__iommu_dma_unmap) calls uf (iommu_unmap_fast), so one teardown can appear
    twice; only the first unmap of a run opens the interval.
    """
    dwell, unmapped_at = [], None
    for t, kind, _comm in events:
        if kind in ("uf", "du"):
            if unmapped_at is None:
                unmapped_at = t
        elif kind == "mp":
            if unmapped_at is not None:
                dwell.append(t - unmapped_at)
                unmapped_at = None
    return dwell


def lifecycle(events):
    """Per page: (dwell list, [(fault_t, state_at_fault, unmapped_for, next_map_delta)]).

    state_at_fault is the property that decides the question: a fault is only
    explained by the teardown if the page was UNMAPPED when it arrived. Reporting
    the last unmap regardless of an intervening map is how a mapped page gets
    described as a teardown race -- the self-test pins this.
    """
    dwell = cycles(events)
    faults = []
    state, unmapped_at = None, None
    # With no unmap probe on this page, "no unmap seen" is absence of a PROBE, not
    # presence of a mapping: every fault must report state UNKNOWN, never MAPPED.
    saw_unmap = any(k in ("uf", "du") for _, k, _c in events)
    for i, (t, kind, comm) in enumerate(events):
        if kind == "mp":
            state, unmapped_at = "mapped", None
        elif kind in ("uf", "du"):
            if state != "unmapped":
                unmapped_at = t
            state = "unmapped"
        elif kind == "pf":
            after = [e for e in events[i + 1:] if e[1] == "mp"]
            before = [e for e in events[:i] if e[1] == "mp"]
            faults.append((t, state if saw_unmap else None,
                           t - unmapped_at if state == "unmapped" else None,
                           after[0][0] - t if after else None,
                           comm, before[-1][2] if before else None,
                           after[0][2] if after else None))
    return dwell, faults


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def report_lifecycle(path):
    ev = load_trace(path)
    if not ev:
        print(f"cannot tell: no kprobe events in {path}")
        return 2
    pages = sorted(ev, key=lambda k: -len(ev[k]))
    total_faults = sum(1 for k in ev for e in ev[k] if e[1] == "pf")
    print(f"{len(pages)} page(s), {sum(len(v) for v in ev.values())} events, "
          f"{total_faults} fault(s)")
    for page in pages:
        dwell, faults = lifecycle(ev[page])
        print(f"\n{page:#x}: {len(ev[page])} events, {len(dwell)} unmapped interval(s)")
        if dwell:
            print(f"  dwell  p50={pct(dwell, .5)*1000:.0f}ms  "
                  f"p90={pct(dwell, .9)*1000:.0f}ms  max={max(dwell)*1000:.0f}ms")
        for t, state, unm, nm, comm, own_before, own_after in faults:
            if state is None:
                # no map/unmap for this page in the trace: the question is unanswerable here
                where = "state UNKNOWN (no map/unmap of this page in the trace)"
            elif state == "unmapped":
                where = f"UNMAPPED for {unm:.2f}s at the fault"
            else:
                where = "MAPPED at the fault -- not a teardown race"
            nm_s = f"next map +{nm:.2f}s" if nm is not None else "never re-mapped"
            hands = ""
            if own_before and own_after and own_before != own_after:
                hands = f" -- CHANGED HANDS {own_before} -> {own_after}"
            print(f"  FAULT @{t:.2f}: {where}, {nm_s}{hands}")
    return 0 if total_faults else 2


def scan_proc(pid, targets, budget):
    """-> ([(pid, va, pfn)], bytes_read). Reads only mapped ranges: pagemap is
    dense there and empty elsewhere, and scanning holes costs gigabytes for nothing."""
    hits, used, ranges = [], 0, []
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                lo, hi = line.split()[0].split("-")
                ranges.append((int(lo, 16), int(hi, 16)))
    except OSError:
        return hits, used
    try:
        fd = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
    except OSError:
        return hits, used
    try:
        for lo, hi in ranges:
            if used >= budget:
                break
            off = (lo // PAGE) * ENTRY
            want = min(((hi - lo) // PAGE + 1) * ENTRY, budget - used)
            while want > 0:
                n = min(want, CHUNK)
                try:
                    buf = os.pread(fd, n, off)
                except OSError:
                    break
                if not buf:
                    break
                used += len(buf)
                want -= len(buf)
                for i in range(0, len(buf) - ENTRY + 1, ENTRY):
                    e = struct.unpack_from("<Q", buf, i)[0]
                    if not e & (1 << 63):     # not present in RAM
                        continue
                    if e & (1 << 58):         # swapped: bits 0-54 are a swap offset
                        continue
                    if e & (1 << 57):         # file page / shared anon
                        continue
                    # bit 55 is soft-dirty and may be set on any page: not a filter.
                    pfn = e & ((1 << 55) - 1)
                    if pfn in targets:
                        hits.append((pid, (off + i) // ENTRY * PAGE, pfn))
                off += len(buf)
    finally:
        os.close(fd)
    return hits, used


def region_of(pid, va):
    try:
        with open(f"/proc/{pid}/maps") as f:
            for line in f:
                lo, hi = line.split()[0].split("-")
                if int(lo, 16) <= va < int(hi, 16):
                    return line.rstrip()
    except OSError:
        return None
    return None


def pagemap_readable():
    """A positive control: a page this process knows is resident must have a nonzero PFN in its own pagemap.

    Unprivileged, every entry's PFN word reads 0 (measured 2026-11-20 on this box: same page, `pfn=0x0` as a
    normal user and `pfn=0x534af4` under sudo), and `scan_proc()` then matches no target at all -- so an owner
    search would answer "nobody" for a page that is held. That is the failure this file's own header forbids
    ("a partial answer is exit 2, never exit 1"), one file over from the kpageflags `0x0` reading.
    """
    import mmap
    page = mmap.mmap(-1, PAGE)
    try:
        page[0:1] = b"x"                       # fault it in
        va = ctypes.addressof(ctypes.c_char.from_buffer(page))
        fd = os.open("/proc/self/pagemap", os.O_RDONLY)
        try:
            e = struct.unpack("<Q", os.pread(fd, ENTRY, (va // PAGE) * ENTRY))[0]
        finally:
            os.close(fd)
        return bool(e & (1 << 63)) and (e & ((1 << 55) - 1)) != 0
    finally:
        page.close()


def report_owners(addrs, per_proc_mb, total_mb):
    targets = {int(a, 16) // PAGE: a for a in addrs}
    tf, hits, scanned, covered, skipped = set(targets), [], 0, 0, 0
    for pid in sorted((p for p in os.listdir("/proc") if p.isdigit()), key=int):
        if scanned >= total_mb << 20:
            skipped += 1
            continue
        h, used = scan_proc(int(pid), tf, per_proc_mb << 20)
        scanned += used
        covered += 1 if used else 0
        hits += h
    print(f"scanned {scanned / 2**20:.1f} MiB of pagemap; {covered} process(es) read, "
          f"{skipped} skipped")
    for pid, va, pfn in hits:
        print(f"OWNER pid={pid} {pfn:#x} ({targets[pfn]}) va={va:#x}  {region_of(pid, va)}")
    if hits:
        return 0
    if not pagemap_readable():
        # Nothing in the scan could have matched: without CAP_SYS_ADMIN every PFN reads 0. Saying "nobody"
        # here is the answer this file exists to refuse.
        print("cannot tell: pagemap PFNs are not readable as this user (needs CAP_SYS_ADMIN), "
              "so an owner cannot be found either way")
        return 2
    if skipped:
        # an unread process may be the owner: this is NOT a negative result
        print(f"cannot tell: {skipped} process(es) were not read")
        return 2
    print("no process in the read set holds any of these pages as private anon")
    return 1


def self_test():
    # A synthetic trace with a known answer: page 0x1000 is mapped, unmapped at
    # 10.0, re-mapped at 10.5 (dwell 0.5s), and a fault at 11.0 with no unmap
    # between -> the fault must be reported as UNPAIRED, not as a teardown race.
    lines = [
        "  t-1 [000] 1.000000: mp: (iommu_map+0x0/0x80) dom=0x1 iova=4096",
        "  t-1 [000] 10.000000: uf: (iommu_unmap_fast+0x0/0x80) dom=0x1 iova=4096",
        "  t-1 [000] 10.500000: mp: (iommu_map+0x0/0x80) dom=0x1 iova=4096",
        "  t-1 [000] 11.000000: pf: (report_iommu_fault+0x4/0xc0) dom=0x1 dev=0x2 iova=4096",
        "  t-1 [000] 12.000000: uf: (iommu_unmap_fast+0x0/0x80) dom=0x1 iova=8192",
        "  t-1 [000] 12.200000: pf: (report_iommu_fault+0x4/0xc0) dom=0x1 dev=0x2 iova=8192",
    ]
    import tempfile
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(lines) + "\n")
        ev = load_trace(path)
        assert set(ev) == {4096, 8192}, f"pages: {sorted(ev)}"
        dwell, faults = lifecycle(ev[4096])
        assert len(dwell) == 1 and abs(dwell[0] - 0.5) < 1e-9, f"dwell={dwell}"
        assert len(faults) == 1, f"faults={faults}"
        t, state, unm, next_map, comm, own_b, own_a = faults[0]
        assert state == "mapped", \
            f"0x1000 was re-mapped at 10.5 before the 11.0 fault; it must read MAPPED, got {state} (unmapped_for={unm})"
        assert unm is None, f"a mapped page has no unmapped-for interval, got {unm}"
        assert next_map is None, f"no map after the fault, got {next_map}"
        _, f2 = lifecycle(ev[8192])
        t2, state2, unm2, _, _c2, _ob2, _oa2 = f2[0]
        assert state2 == "unmapped" and abs(unm2 - 0.2) < 1e-9, \
            f"0x2000 is unmapped from 12.0 and faults at 12.2: {f2}"
        # a trace with map+fault but NO unmap probe must report UNKNOWN, not MAPPED
        fd3, p3 = tempfile.mkstemp()
        with os.fdopen(fd3, "w") as f:
            f.write("\n".join(lines[:1] + [
                "  t-1 [000] 10.000000: pf: (report_iommu_fault+0x4/0xc0) dom=0x1 dev=0x2 iova=4096"]) + "\n")
        _, f3 = lifecycle(load_trace(p3)[4096])
        assert f3[0][1] is None, \
            f"no unmap probe in the trace: state must be UNKNOWN, got {f3[0][1]}"
        os.unlink(p3)
        # percentiles must not crash or lie on a single sample
        assert pct([0.5], .5) == 0.5 and pct([0.5], .9) == 0.5
        # engine#2478: the owner search refuses when it cannot read a PFN at all. The control must say so as
        # a bool on every box (root: True; an ordinary user: False), and `report_owners` returns 2 in that case
        # -- the end-to-end refusal is measured with the CLI in the PR body, because a full scan is far too slow
        # for a self-test.
        assert isinstance(pagemap_readable(), bool), "the pagemap control must decide, not raise"
        # an empty trace is `cannot tell`, not `clean`
        fd2, p2 = tempfile.mkstemp()
        os.close(fd2)
        assert report_lifecycle(p2) == 2, "empty trace must exit 2"
        os.unlink(p2)
        # a NON-empty trace with no fault is also `cannot tell`: the dwell is
        # reported but the pairing question has no data. Without this arm the
        # final `return 0 if total_faults else 2` is never executed.
        fd4, p4 = tempfile.mkstemp()
        with os.fdopen(fd4, "w") as f:
            f.write(lines[0] + "\n" + lines[1] + "\n")
        assert report_lifecycle(p4) == 2, "map+unmap with no fault must exit 2"
        os.unlink(p4)
    finally:
        os.unlink(path)
    print("self-test: dwell, paired/unpaired classification and cannot-tell all behave")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lifecycle", metavar="TRACE", help="kprobe trace to analyse")
    ap.add_argument("--owners", nargs="+", metavar="IOVA", help="physical addresses to reverse-map")
    ap.add_argument("--per-proc-mb", type=int, default=256)
    ap.add_argument("--total-mb", type=int, default=4096)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.lifecycle:
        return report_lifecycle(a.lifecycle)
    if a.owners:
        return report_owners(a.owners, a.per_proc_mb, a.total_mb)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
