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
"""Census IOMMU page faults per boot, so "same addresses or different?" is a command, not an argument.

engine#2478 measured 2361 `AMD-Vi: Event logged [IO_PAGE_FAULT ...]` lines on the root NVMe SSD in the boot that
died at 23:47:35, with a handful of addresses repeating hundreds of times. Which of two faults that is decides where
the next hour goes, and the two look identical in a `head -3`:

  * the same address set in two boots -> one deterministic allocation whose translation went stale; find the buffer;
  * disjoint sets -> something is re-allocating (or freeing and reusing) behind a live descriptor; find the race.

So the question has to be asked as a set comparison across boots, which is what this prints. Nothing here reads the
device: it takes journal text on stdin or from `--journal`, so it can run on a capture, on another host, or on a
synthetic fixture in the self-test.

    journalctl -k -b -1 | python3 tools/diag/iommu_fault_census.py --label -1
    journalctl -k -b -2 | python3 tools/diag/iommu_fault_census.py --label -2 --json
    python3 tools/diag/iommu_fault_census.py --journal a.txt --journal b.txt   # intersects the files

Exit status is 0 whenever the input was parsed, because the census reports and does not judge: 0 in both fault
shapes. Only an unreadable input is an error (2).
"""
import argparse
import json
import re
import sys

# `nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0xeb536000 flags=0x0000]`
FAULT = re.compile(
    r"(?:(?P<dev>[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]):\s+)?"
    r"AMD-Vi: Event logged \[IO_PAGE_FAULT domain=(?P<domain>0x[0-9a-fA-F]+) address=(?P<addr>0x[0-9a-fA-F]+)"
)


def parse(text, label):
    """{(label, device, domain): [int addresses]} for every fault line in text."""
    rows = {}
    for line in text.splitlines():
        m = FAULT.search(line)
        if not m:
            continue
        key = (label, m.group("dev") or "-", m.group("domain"))
        rows.setdefault(key, []).append(int(m.group("addr"), 16))
    return rows


def census(rows):
    out = []
    for (label, dev, domain), addrs in sorted(rows.items()):
        counts = {}
        for a in addrs:
            counts[a] = counts.get(a, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        out.append(
            {
                "label": label,
                "device": dev,
                "domain": domain,
                "faults": len(addrs),
                "distinct": len(counts),
                "top": [{"address": hex(a), "count": c} for a, c in top],
                "_set": set(counts),
            }
        )
    return out


def intersection(entries):
    """Distinct addresses present in EVERY label's set: the deterministic-allocation signature."""
    by_label = {}
    for e in entries:
        by_label.setdefault(e["label"], set()).update(e["_set"])
    if len(by_label) < 2:
        return None
    sets = list(by_label.values())
    return sorted(set.intersection(*sets))


def render(entries, inter, as_json):
    if as_json:
        return json.dumps(
            {
                "boots": [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries],
                "shared_across_boots": None if inter is None else [hex(a) for a in inter],
            },
            indent=2,
        )
    lines = []
    for e in entries:
        lines.append(
            "%s %s domain=%s: %d fault(s), %d distinct" % (e["label"], e["device"], e["domain"], e["faults"], e["distinct"])
        )
        for t in e["top"]:
            lines.append("    %s x%d" % (t["address"], t["count"]))
    if inter is None:
        lines.append("shared across boots: (need two or more labeled journals)")
    else:
        lines.append(
            "shared across boots: %d address(es) -> %s"
            % (
                len(inter),
                "one deterministic allocation with a stale translation: find the buffer"
                if inter
                else "disjoint sets: re-allocation or free behind a live descriptor: find the race",
            )
        )
        for a in inter[:5]:
            lines.append("    %s" % hex(a))
    return "\n".join(lines)


def self_test():
    fails = 0

    def ok(name, got, want):
        nonlocal fails
        if got != want:
            print("FAIL %s: got %r wanted %r" % (name, got, want))
            fails += 1
        else:
            print("ok   %s" % name)

    one = (
        "Nov 14 23:46:34 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged "
        "[IO_PAGE_FAULT domain=0x0015 address=0xeb536000 flags=0x0000]\n"
        "Nov 14 23:46:35 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged "
        "[IO_PAGE_FAULT domain=0x0015 address=0xeb536000 flags=0x0000]\n"
    )
    c = census(parse(one, "b1"))
    ok("the same address twice is one distinct address", (c[0]["faults"], c[0]["distinct"]), (2, 1))
    ok("the requester is the device, not the VM", c[0]["device"], "0000:c4:00.0")
    ok("the IOMMU domain is kept", c[0]["domain"], "0x0015")

    other = one.replace("0xeb536000", "0xeb5fa000")
    both = census({**parse(one, "b1"), **parse(other, "b2")})
    ok("disjoint boots share nothing", intersection(both), [])

    mixed = other + one
    both = census({**parse(one, "b1"), **parse(mixed, "b2")})
    ok("a shared address is found across boots", intersection(both), [0xEB536000])

    noise = "Nov 14 23:40:00 host kernel: nvme nvme0: I/O 12 QID 1 timeout, aborting\n"
    ok("a line that is not a page fault is ignored", parse(noise, "b1"), {})

    ok(
        "a fault line with no device prefix still counts, device '-'",
        census(parse("AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x1000 flags=0x0000]\n", "b1"))[0]["device"],
        "-",
    )

    ok("one boot cannot answer the cross-boot question", intersection(c), None)

    print("iommu_fault_census self-test: %d failure(s)" % fails)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--journal", action="append", default=[], help="file of journal text (repeatable)")
    ap.add_argument("--label", default="stdin", help="boot label for the journal on stdin")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    rows = {}
    if a.journal:
        for path in a.journal:
            try:
                with open(path, "r", errors="replace") as fh:
                    rows.update(parse(fh.read(), path))
            except OSError as e:
                print("iommu_fault_census: cannot read %s: %s" % (path, e), file=sys.stderr)
                return 2
    else:
        rows = parse(sys.stdin.read(), a.label)

    entries = census(rows)
    if not entries:
        print("no IO_PAGE_FAULT lines in the input", file=sys.stderr)
        return 0
    print(render(entries, intersection(entries), a.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
