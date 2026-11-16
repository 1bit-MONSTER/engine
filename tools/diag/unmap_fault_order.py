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
"""Engine#2478/#2568: does an unmap of a page land while a command still references it?

The fault stream on this box is NOT driven by volume -- measured, each with its own window: 20 min of `O_DIRECT`
reads (3 TiB), 10 min of mmap/touch/munmap churn (157 TiB), a 213 s build issuing **60,459** `dma_iova_destroy`
calls, and three boots' worth of uptime. Every one of them produced zero faults (printed *and* suppressed), so what
is left is **ordering**: a fault needs the unmap of a page to land while a command that references it is still
outstanding. That is not measurable by a load; it needs the two events in one capture.

This reads two inputs and correlates them:

  * `--trace FILE`  a `tracefs` capture with a kprobe on the unmap path, e.g.
      # iommu_unmap_fast(domain, iova, size) -> %rdi/%rsi/%rdx on x86-64. Written as %di/%si the probe
      # FIRES and parses, but the iova column holds the domain POINTER and the size column holds the iova
      # (measured 2026-11-16: iova=0xffff8e83841c3a00 size=0x45ec0), so nothing can ever correlate --
      # which this tool would print as an ordinary negative. Use %si/%dx and check the values look like IOVAs.
      echo 'p:unm iommu_unmap_fast iova=%si size=%dx' >> kprobe_events
      ... (run the workload) ... cat trace > cap.txt
  * `--journal FILE` (or stdin) kernel-log lines carrying `IO_PAGE_FAULT ... address=0x...`

For every fault it reports the nearest unmap of the SAME iova, with the signed delta in microseconds:
negative = the unmap preceded the fault (the translation was gone when the device wrote), positive = the unmap
followed it (a stale mapping or a different cause). `--window US` bounds what counts as "the same event".

Exit status: 0 when both inputs parsed (the tool reports, it does not judge -- a zero-correlation capture and a
correlated one are both valid observations), 2 on an unreadable input, 3 when the trace has no unmap events at all
(an empty capture is not a negative -- engine#2531's first trap).
"""
import argparse
import re
import sys

FAULT_RE = re.compile(r"IO_PAGE_FAULT domain=\S+ address=(0x[0-9a-fA-F]+)")
UNMAP_RE = re.compile(r"unm:.*iova=(0x[0-9a-fA-F]+)(?:.*size=(0x[0-9a-fA-F]+))?")
# tracefs puts the timestamp AFTER the comm/pid/cpu/flags fields (` bash-1 [001] .... 100.000000: ev: ...`),
# so it is searched mid-line, not anchored -- my first version anchored it and parsed nothing.
EPOCH_RE = re.compile(r"(\d+\.\d+):")
# Two stamp shapes, because the two sources that carry these lines disagree: `journalctl -k` writes an ISO stamp
# and a `dmesg`/trace-style capture writes `[1234.567890]` seconds-since-boot. A fault line is only correlatable
# when its clock is the trace's, so the seconds form is preferred when both appear.
ISO_RE = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)")
BOOT_RE = re.compile(r"^\s*\[\s*(\d+\.\d+)\]")


def read_lines(path):
    if path in (None, "-"):
        return sys.stdin.read().splitlines()
    with open(path, "r", errors="replace") as f:
        return f.read().splitlines()


def parse_trace(lines):
    """[(t, iova, size)] from a tracefs capture. Lines without a leading timestamp are skipped."""
    out = []
    for line in lines:
        m = EPOCH_RE.search(line)
        if not m:
            continue
        mu = UNMAP_RE.search(line)
        if not mu:
            continue
        size = int(mu.group(2), 16) if mu.group(2) else 0x1000
        out.append((float(m.group(1)), int(mu.group(1), 16), size))
    return out


def parse_faults(lines, journal_epoch):
    """[(t, iova, text)] from kernel-log lines; `journal_epoch` maps an ISO stamp to seconds."""
    out = []
    for line in lines:
        mf = FAULT_RE.search(line)
        if not mf:
            continue
        mb = BOOT_RE.search(line)
        if mb:                          # already on the trace's boot clock
            t = float(mb.group(1))
        else:
            mi = ISO_RE.search(line)
            t = journal_epoch(mi.group(1)) if mi else None
        out.append((t, int(mf.group(1), 16), line.strip()))
    return out


def correlate(unmaps, faults, window_us):
    """[(fault, delta_us, unmap)] -- the nearest unmap of the same iova within the window, or None."""
    by_iova = {}
    for t, iova, size in unmaps:
        by_iova.setdefault(iova, []).append((t, size))
    rows = []
    for t, iova, text in faults:
        best = None
        if t is not None:
            for ut, size in by_iova.get(iova, ()):
                d = (ut - t) * 1e6   # negative = the unmap came first
                if abs(d) <= window_us and (best is None or abs(d) < abs(best[1])):
                    best = ((ut, iova, size), d)
        rows.append((t, iova, text, best))
    return rows


IOVA_MAX = 1 << 47          # x86-64 IOMMU IOVAs are below this; a canonical kernel pointer (0xffff8...) is not


def pointer_shaped(unmaps):
    """True when the capture's iova column cannot be IOVAs at all.

    A kprobe whose format string names the wrong argument (engine#2592 review: `%di` instead of `%si` puts the
    domain pointer in the iova field) yields values like 0xffff8e83841c3a00. Nothing can correlate them, and this
    tool's ordinary negative -- "no fault shares an iova with an unmap" -- would read as a real finding. Every iova
    at or above 2**47 is that mistake, not a measurement.
    """
    return bool(unmaps) and all(u[1] >= IOVA_MAX for u in unmaps)


def _span(window_us):
    """The window as a reader thinks of it: seconds once it is a second or more. `+-30000000 us` is not readable."""
    return f"{window_us / 1e6:g} s" if window_us >= 1e6 else f"{window_us:.0f} us"


def render(rows, unmaps, window_us):
    lines = [f"unmap events parsed: {len(unmaps)} over {len({u[1] for u in unmaps})} distinct iova(s)"]
    if unmaps:
        lines.append(f"  sizes: {sorted({u[2] for u in unmaps})[0]:#x} .. {sorted({u[2] for u in unmaps})[-1]:#x}")
    hit = [r for r in rows if r[3] is not None]
    lines.append(f"faults parsed: {len(rows)}; with an unmap of the same iova within +-{_span(window_us)}: {len(hit)}")
    for t, iova, _text, best in hit[:20]:
        (ut, _u, size), d = best
        verdict = "unmap BEFORE fault (translation gone)" if d < 0 else "unmap AFTER fault (stale mapping or other cause)"
        lines.append(f"  iova {iova:#x} size {size:#x}: unmap {d:+.0f} us, {verdict}")
    if rows and not hit:
        # A negative must name its own condition. The same sentence at a 5 ms window and at 30 s read alike, and the
        # 5 ms one was simply wrong, so the scale is in the line and the remedy is named with it.
        lines.append(f"  no fault shares an iova with an unmap inside +-{_span(window_us)}: ordering not shown by "
                     f"this capture AT THIS WINDOW -- the measured unmap-to-fault interval on this box is 5-25 s, "
                     f"so widen --window before reading this as 'no unmap preceded the fault'")
    return lines


def self_test():
    """Four shapes, each asserted: before, after, out-of-window, and an empty capture."""
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("ok   " if cond else "FAIL ") + msg)
        ok = ok and cond

    unm = [(100.000000, 0x4128000, 0x1000), (200.000000, 0x5128000, 0x1000)]
    # 1. unmap precedes the fault by 500 us -> reported as BEFORE
    rows = correlate(unm, [(100.0005, 0x4128000, "f1")], 5000)
    chk(rows[0][3] is not None and rows[0][3][1] < 0, "an unmap 500 us before the fault is reported as BEFORE")
    # 2. unmap follows it
    rows = correlate(unm, [(199.9995, 0x5128000, "f2")], 5000)
    chk(rows[0][3] is not None and rows[0][3][1] > 0, "an unmap 500 us after the fault is reported as AFTER")
    # 3. same iova but far outside the window -> not correlated
    rows = correlate(unm, [(100.500, 0x4128000, "f3")], 5000)
    chk(rows[0][3] is None, "an unmap 500 ms away is not counted in a 5 ms window")
    # 4. an unread fault time cannot be correlated (and is kept in the count)
    rows = correlate(unm, [(None, 0x4128000, "f4")], 5000)
    chk(rows[0][3] is None and len(rows) == 1, "a fault with no timestamp is counted but not correlated")
    # 5. the parser reads a real tracefs line shape
    got = parse_trace(["    bash-1234 [001] .... 100.000000: unm: (iommu_unmap_fast+0x0/0x20) iova=0x4128000 size=0x1000"])
    chk(len(got) == 1 and got[0][1] == 0x4128000 and got[0][2] == 0x1000, "a tracefs unmap line parses (iova, size)")
    # 6. a pointer-shaped iova column is refused, not reported as "nothing correlated". My first recipe named %di/%si
    #    for a (domain, iova, size) function, which put the domain pointer here and made every capture uncorrelatable.
    ptr = parse_trace(["    pi-1-1 [016] d.h.. 358.053875: unm: (iommu_unmap_fast+0x4/0x10) "
                       "iova=0xffff8e83841c3a00 size=0x45ec0"])
    chk(pointer_shaped(ptr), "a pointer-shaped iova column is detected, not read as uncorrelated")
    chk(not pointer_shaped(got), "a real IOVA column is not mistaken for one")
    # 7. the window reads in seconds once it is seconds, and a negative names the window it used. A default alone
    #    does not fix the failure I am fixing: 30 s finding nothing printed the same sentence 5 ms did.
    #    The fault is 100 s from any unmap: at 30 s that is still a miss, which is the shape a too-narrow window gives.
    out = " ".join(render(correlate(unm, [(300.000, 0x4128000, "f5")], 30_000_000), unm, 30_000_000))
    chk("+-30 s" in out and "30000000 us" not in out, "a 30 s window reads as '+-30 s', not '+-30000000 us'")
    chk("AT THIS WINDOW" in out and "5-25 s" in out,
        "a negative names the window it used and the measured interval, not 'nothing correlated' bare")
    chk("+-5000 us" in " ".join(render(correlate(unm, [(100.500, 0x4128000, "f3")], 5000), unm, 5000)),
        "a sub-second window still reads in microseconds")
    # 8. the REAL kernel stamp is PADDED: `[  321.308031]` (two spaces; one at >= 1000 s, four in early dmesg). Written
    #    without the inner `\s*`, BOOT_RE matched only the unpadded shape the doc example uses, so every real line parsed
    #    as untimed and nothing could correlate (#2622). #2625 rewrote the arms above and dropped this one, and with it
    #    the pre-#2622 regex passed every arm again -- so it is pinned here, at both the parse and the correlation.
    padded = ["[  321.308031] strixhalo kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged "
              "[IO_PAGE_FAULT domain=0x0015 address=0x4128000 flags=0x0000]",
              "[    0.000000] strixhalo kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged "
              "[IO_PAGE_FAULT domain=0x0015 address=0x4129000 flags=0x0000]"]
    pf = parse_faults(padded, lambda _s: None)
    chk([f[0] for f in pf] == [321.308031, 0.0], "padded boot stamps parse (two- and four-space padding)")
    near = parse_trace(["   bash-1 [002] .... 321.308000: unm: (iommu_unmap_fast+0x4/0x10) iova=0x4128000 size=0x1000"])
    hit = correlate(near, pf[:1], 30_000_000)
    chk(hit and hit[0][3] is not None and hit[0][3][1] < 0,
        "a padded fault correlates with an unmap 31 us earlier (BEFORE), not an uncorrelated negative")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", help="a tracefs capture with a kprobe on the unmap path")
    ap.add_argument("--journal", action="append", help="kernel-log lines (repeatable; default stdin)")
    # 30 s, not 5 ms: measured on this box 2026-11-16, three of four faults had an unmap of the SAME 4 KiB page
    # 5.6 s / 24.7 s / 24.7 s earlier, and a 5 ms default answered "no fault shares an iova with an unmap" on the
    # very capture that contains them. The interval is seconds here; a small default hides the finding.
    ap.add_argument("--window", type=float, default=30_000_000.0,
                    help="window in microseconds (default 30,000,000 = 30 s; measured interval on this box is 5-25 s)")
    ap.add_argument("--epoch-offset", type=float, default=0.0,
                    help="seconds to add to journal stamps to reach the trace's boot clock")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.trace:
        print("unmap_fault_order: --trace is required (or --self-test)", file=sys.stderr)
        return 2
    try:
        unmaps = parse_trace(read_lines(args.trace))
    except OSError as e:
        print(f"unmap_fault_order: {e}", file=sys.stderr)
        return 2
    if not unmaps:
        print("unmap_fault_order: no unmap events in the trace -- an empty capture is not a negative; was the "
              "probe armed before the load, and did it fire? (see --self-test for the four shapes)", file=sys.stderr)
        return 3
    if pointer_shaped(unmaps):
        print("unmap_fault_order: this capture's iova column is not IOVA-shaped (every value >= 2**47, e.g. "
              f"{unmaps[0][1]:#x}): the kprobe is naming the wrong argument. iommu_unmap_fast(domain, iova, size) "
              "is %si/%dx, not %di/%si -- re-arm and check that the values look like IOVAs before reading any "
              "result from it.", file=sys.stderr)
        return 4
    lines = []
    for j in (args.journal or [None]):
        try:
            lines += read_lines(j)
        except OSError as e:
            print(f"unmap_fault_order: {e}", file=sys.stderr)
            return 2

    def journal_epoch(iso):
        import datetime
        dt = datetime.datetime.fromisoformat(iso)
        return dt.timestamp() + args.epoch_offset

    # journal stamps are wall-clock, the trace is boot-relative: align on the FIRST fault and its capture, which the
    # caller supplies with --epoch-offset. Without it the deltas are meaningless, so say so rather than print zeros.
    faults = parse_faults(lines, journal_epoch)
    if args.epoch_offset == 0.0 and any(ISO_RE.search(l) and not BOOT_RE.search(l) for l in lines):
        print("unmap_fault_order: note: the journal is on the wall clock and the trace on the boot clock; pass "
              "--epoch-offset (or feed a `[seconds]` capture) or the deltas are not meaningful", file=sys.stderr)
    for l in render(correlate(unmaps, faults, args.window), unmaps, args.window):
        print(l)
    return 0


if __name__ == "__main__":
    sys.exit(main())
