#!/usr/bin/env python3
"""engine#315 — read [hrx-wb-fp] records and apply the pre-registered outcome table.

The instrumentation (docs/315-instrumentation.patch, enabled by
GGML_HRX_WRITEBACK_FINGERPRINT=1) logs one line per deferred host writeback:

    [hrx-wb-fp] value=<id> size=<n> barrier=<0|1> rec_hash=<hex> sync_hash=<hex> dst_hash=<hex>
                rec_class=<ZERO|NONFINITE|FINITE|EMPTY> sync_class=... dst_class=...
                rec_eq_sync=<0|1> rec_nf16=.. sync_nf16=.. dst_nf16=.. rec_nf32=.. sync_nf32=.. dst_nf32=..

`rec_class` is the state of the mapped staging buffer **when the writeback was queued**,
`sync_class` immediately **before** the host memcpy, `dst_class` the host destination **after** it.

Pre-registered reading (docs/315-instrumentation-plan.md):

    sync=NONFINITE, dst=NONFINITE           -> producer wrote non-finite values: H1 excluded, H2 stands
    sync=ZERO/stale, dst=NONFINITE          -> host read staging before the device copy landed: H1
    sync=ZERO/stale, dst=FINITE             -> same ordering defect, non-fatal this time
    sync=FINITE,    dst=FINITE              -> this writeback is clean

Usage: 315-wbfp-read.py <server.log> [<server.log>...]
Exit 0 if records were found and classified; 3 if no records (diagnostic inert / flag off).
"""
from __future__ import annotations

import re
import sys
from collections import Counter

REC = re.compile(
    r"\[hrx-wb-fp\] value=(-?\d+) size=(\d+) barrier=(\d+) rec_hash=([0-9a-f]+) sync_hash=([0-9a-f]+) "
    r"dst_hash=([0-9a-f]+) rec_class=(\w+) sync_class=(\w+) dst_class=(\w+) rec_eq_sync=(\d+) "
    r"rec_nf16=(\d+) sync_nf16=(\d+) dst_nf16=(\d+) rec_nf32=(\d+) sync_nf32=(\d+) dst_nf32=(\d+)")


def classify(r) -> str:
    s, d = r["sync_class"], r["dst_class"]
    if s == "NONFINITE" and d == "NONFINITE":
        return "H1-excluded (producer wrote non-finite: sync and dst already non-finite)"
    if s in ("ZERO", "EMPTY") and d == "NONFINITE":
        return "H1 (host read staging before the device copy landed; fatal in this event)"
    if s in ("ZERO", "EMPTY") and d == "FINITE":
        return "H1 (same ordering defect, non-fatal in this event)"
    if s == "FINITE" and d == "FINITE":
        return "clean"
    return f"unclassified (sync={s} dst={d})"


def main(paths) -> int:
    records, faces = [], Counter()
    for p in paths:
        try:
            text = open(p, errors="replace").read()
        except OSError as exc:
            print(f"  cannot read {p}: {exc}")
            continue
        for line in text.splitlines():
            m = REC.search(line)
            if m:
                g = m.groups()
                records.append(dict(value=int(g[0]), size=int(g[1]), barrier=int(g[2]),
                                    rec_class=g[6], sync_class=g[7], dst_class=g[8],
                                    rec_eq_sync=int(g[9]), rec_nf16=int(g[10]),
                                    sync_nf16=int(g[11]), dst_nf16=int(g[12]),
                                    rec_nf32=int(g[13]), sync_nf32=int(g[14]), dst_nf32=int(g[15])))
        faces["nan"] += text.count("HRX returned NaN logits")
        faces["hsa"] += sum(text.count(s) for s in
                            ("HSA_STATUS_ERROR", "Queue error", "wait for HRX graph replay commands failed"))
        faces["alloc_fail"] += len(re.findall(r"hrx_allocator_allocate_buffer.*failed", text))

    print(f"records: {len(records)}   faces in the same logs: nan={faces['nan']} hsa={faces['hsa']} "
          f"startup_alloc_failures={faces['alloc_fail']}")
    if not records:
        print("no [hrx-wb-fp] records: either the flag was off (expected for the inertness arm) or "
              "no deferred writeback was published in this run")
        return 3

    triples = Counter((r["rec_class"], r["sync_class"], r["dst_class"]) for r in records)
    print("\nrec_class -> sync_class -> dst_class (count):")
    for (a, b, c), n in triples.most_common(10):
        print(f"  {a:>9} -> {b:>9} -> {c:>9}  x{n}")
    same = sum(1 for r in records if r["rec_eq_sync"])
    print(f"\nrec_hash == sync_hash: {same}/{len(records)} "
          f"(equal means the staging bytes did not change between queueing and publishing)")

    verdicts = Counter(classify(r) for r in records)
    print("\npre-registered reading:")
    for v, n in verdicts.most_common():
        print(f"  x{n:<4} {v}")

    h1 = sum(n for v, n in verdicts.items() if v.startswith("H1 ("))
    h1x = sum(n for v, n in verdicts.items() if v.startswith("H1-excluded"))
    print()
    if h1 and not h1x:
        print("=> mechanism: the deferred/async writeback window (H1) — fix the publish ordering")
    elif h1x and not h1:
        print("=> mechanism: the producer wrote non-finite values (H2) — H1 excluded for these records")
    elif h1 and h1x:
        print("=> mixed: both rows present; report counts separately, do not pick one")
    else:
        print("=> neither row present: these writebacks are clean, so the faulting buffer is not this route")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
