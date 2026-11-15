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
    # the runbook that reads this output, and the two root-only tests it decides: docs/nvme-iommu-fault-triage.md
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
    r"(?: flags=(?P<flags>0x[0-9a-fA-F]+))?"
)


def parse(text, label):
    """{(label, device, domain): [int addresses]} for every fault line in text."""
    rows = {}
    for line in text.splitlines():
        m = FAULT.search(line)
        if not m:
            continue
        key = (label, m.group("dev") or "-", m.group("domain"))
        rows.setdefault(key, []).append((int(m.group("addr"), 16), (m.group("flags") or "").lower()))
    return rows


SUPPRESSED = re.compile(r"(\S+): (\d+) callbacks suppressed")


def suppressed(text):
    """{caller: N} from `printk_ratelimited`'s "N callbacks suppressed" lines.

    engine#2478: `amd_iommu_report_page_fault: 5611 callbacks suppressed` means the kernel DROPPED 5611 fault
    callbacks, so every count on this page is the printed subset. A census that reports a compressed stream
    without saying so is the same failure as a histogram that drops the lines it cannot read (engine#2562).
    The counter is per caller, not per device, so it is reported per journal rather than per device row."""
    out = {}
    for caller, n in SUPPRESSED.findall(text):
        out[caller] = out.get(caller, 0) + int(n)
    return out


def census(rows):
    out = []
    for (label, dev, domain), pairs in sorted(rows.items()):
        counts, flag_counts = {}, {}
        for a, fl in pairs:
            counts[a] = counts.get(a, 0) + 1
            # A line whose flags field is absent -- or present and unparseable, which this cannot tell apart --
            # is counted in ``faults`` either way, so it must appear in the histogram too: a bucket that only
            # holds the lines it could parse under-counts, and an under-count is indistinguishable from a quiet
            # device. ``(absent)`` keeps sum(flags) == faults for every bucket (review of engine#2559).
            key = fl or "(absent)"
            flag_counts[key] = flag_counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        out.append(
            {
                "label": label,
                "device": dev,
                "domain": domain,
                "faults": len(pairs),
                "distinct": len(counts),
                # engine#2542: the flags field is the most informative part of a fault line and the least obvious --
                # two devices on this host carry different values (0x0007 vs 0x0000, plus rare 0x0050), and a decode that
                # explains two of three is incomplete. Report the histogram so nobody has to hand-grep for it.
                "flags": dict(sorted(flag_counts.items())),
                "top": [{"address": hex(a), "count": c} for a, c in top],
                "_set": set(counts),
            }
        )
    return out


def offsets(entries):
    """{(device, domain): [address minus the label's own lowest address]}.

    engine#2478: an absolute intersection answers "does the same page fault again next boot" and came back empty across
    three boots while every boot's faults sat in the same 0.1 GiB window -- the slot recurs and the pages do not. A
    buffer re-allocated at the same place has the same *offset* from each boot's base, so comparing offsets answers the
    question the absolute set cannot."""
    groups = {}
    for e in entries:
        if not e["_set"]:
            continue
        base = min(e["_set"])
        # 0 is excluded: it is the base by construction, so every boot shares it and it says nothing. What matters is
        # whether the pages *above* the base land at the same offsets, which only holds if the object's layout recurs.
        groups.setdefault((e["device"], e["domain"]), {})[e["label"]] = {a - base for a in e["_set"]} - {0}
    out = {}
    for key, per_label in groups.items():
        if len(per_label) >= 2:
            out[key] = sorted(set.intersection(*per_label.values()))
    return out or None


def intersection(entries):
    """{(device, domain): [addresses present in every label]}. Per device: pooling them mixed an NPU address with an
    NVMe address and called the pair a shared buffer (review of engine#2491)."""
    groups = {}
    for e in entries:
        groups.setdefault((e["device"], e["domain"]), {})[e["label"]] = e["_set"]
    out = {}
    for key, per_label in groups.items():
        if len(per_label) >= 2:
            out[key] = sorted(set.intersection(*per_label.values()))
    return out or None


def shape(e):
    """Which fault shape this (boot, device, domain) is, decided by what the tool already counts: repeats per address.

    The cross-boot set cannot decide it. These addresses are IOVAs, handed out per boot by dma-iommu in probe/allocation
    order, so one deterministic buffer with a stale translation still lands on a different IOVA every boot -- "disjoint"
    is the expected answer under BOTH hypotheses, and a tool that reads it as "race" is wrong on real hardware
    (review of engine#2491, measured on the box: NVMe 3800 faults over 10 distinct, then 2361/9, then 71/4).
    """
    repeats = e["faults"] / float(max(e["distinct"], 1))
    # Both criteria, because each one alone mislabels the box's real streams: 20+ distinct pages is churn even if a few of
    # them repeat (the NPU stream is 390 faults over 321 distinct, max 6 each), and a handful of pages repeating fewer than
    # five times is not evidence of a live stale mapping either.
    if e["distinct"] < 20 and repeats >= 5:
        return (
            "few live mappings faulting over and over (%.1f faults per address) -> a stale translation: find the buffer"
            % repeats
        )
    return (
        "many distinct pages faulted about once each (%.1f faults per address) -> re-allocation, or a free behind a live "
        "descriptor: find the race" % repeats
    )


def render(entries, inter, as_json, sup=None):
    if as_json:
        return json.dumps(
            {
                "boots": [
                    dict({k: v for k, v in e.items() if not k.startswith("_")}, shape=shape(e)) for e in entries
                ],
                "shared_addresses": None
                if inter is None
                else {("%s %s" % k): [hex(a) for a in v] for k, v in inter.items()},
            },
            indent=2,
        )
    lines = []
    for e in entries:
        lines.append(
            "%s %s domain=%s: %d fault(s), %d distinct"
            % (e["label"], e["device"], e["domain"], e["faults"], e["distinct"])
        )
        for t in e["top"]:
            lines.append("    %s x%d" % (t["address"], t["count"]))
        lines.append(
            "    flags: %s" % ", ".join("%s x%d" % (f, n) for f, n in sorted(e["flags"].items(), key=lambda kv: -kv[1]))
        )
        lines.append("    -> %s" % shape(e))
    if inter is None:
        lines.append("shared across boots: (need two or more labeled journals of the same device)")
    else:
        for (dev, dom), addrs in sorted(inter.items()):
            lines.append("shared across boots, %s domain=%s: %d address(es)" % (dev, dom, len(addrs)))
            for a in addrs[:5]:
                lines.append("    %s" % hex(a))
        # engine#2478: the absolute set is empty when the same buffer moves between boots. The offset set says whether
        # it moved *to the same relative place*, which is what a re-allocation of one object looks like.
        off = offsets(entries)
        for (dev, dom), offs in sorted((off or {}).items()):
            if not offs:
                continue
            lines.append(
                "shared across boots by offset, %s domain=%s: %d offset(s) from each boot's lowest address"
                % (dev, dom, len(offs))
            )
            for a in offs[:5]:
                lines.append("    +%s" % hex(a))
    # engine#2478: the ratelimiter drops fault callbacks, so the counts above are a subset. Say so, and by how
    # much, or a reader compares two boots whose drop rates differ and reads the difference as the device's.
    for label, drops in sorted((sup or {}).items()):
        n = sum(drops.values())
        if n:
            printed = sum(e["faults"] for e in entries if e["label"] == label)
            lines.append(
                "%s suppressed: %d fault callback(s) dropped by printk -- %d printed here, so the stream is at least %d"
                % (label, n, printed, printed + n)
            )
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
    ok("disjoint boots share nothing", intersection(both), {("0000:c4:00.0", "0x0015"): []})

    mixed = other + one
    both = census({**parse(one, "b1"), **parse(mixed, "b2")})
    ok("a shared address is found across boots", intersection(both), {("0000:c4:00.0", "0x0015"): [0xEB536000]})

    # the review's arm: two boots sharing no page, each with a few pages repeating many times, must NOT be called a race
    heavy = "".join(
        "Nov 14 20:0%d host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=%s flags=0x0000]\n"
        % (i % 10, a)
        for a in ("0xe8247000", "0xe7fdd000")
        for i in range(300)
    )
    heavy2 = heavy.replace("0xe8247000", "0xebe54000").replace("0xe7fdd000", "0xeb5fa000")
    hb = census({**parse(heavy, "b-2"), **parse(heavy2, "b-1")})
    ok("repeating pages in disjoint boots are still the stale-translation shape", shape(hb[0]), shape(hb[1]))
    ok("...and that arm's own text is the buffer one", "find the buffer" in shape(hb[0]), True)
    ok(
        "...while the intersection is reported as an empty FACT, not as a race",
        intersection(hb),
        {("0000:c4:00.0", "0x0015"): []},
    )

    # engine#2478: the same buffer re-allocated at the same relative place shares an OFFSET, not an address, so the
    # absolute intersection is empty while the offset intersection is not. Both are reported because the pair, not
    # either half, is the diagnosis ("moved with the object" vs "moved with the slot").
    moved = "".join(
        "Nov 14 20:0%d host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=%s flags=0x0000]\n"
        % (i % 10, a)
        for a in ("0xeb536000", "0xeb537000", "0xeb5fa000")
        for i in range(3)
    )
    moved2 = moved.replace("0xeb536000", "0xe6000000").replace("0xeb537000", "0xe6001000").replace("0xeb5fa000", "0xe60c4000")
    mb = census({**parse(moved, "b1"), **parse(moved2, "b2")})
    ok("the same object at a different base shares no address", intersection(mb), {("0000:c4:00.0", "0x0015"): []})
    ok("...but it does share offsets, which is the point", offsets(mb),
       {("0000:c4:00.0", "0x0015"): [0x1000, 0xC4000]})
    ok("...and two objects at different offsets share nothing by offset",
       offsets(census({**parse(moved, "b1"), **parse(moved.replace("0xeb5fa000", "0xeb636000"), "b2")})),
       {("0000:c4:00.0", "0x0015"): [0x1000]})

    # the second bug the review found: intersections must not pool devices
    npu = "Nov 14 20:00 host kernel: nvme 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0002 address=0xeb536000 flags=0x0000]\n"
    pooled = census({**parse(one, "b1"), **parse(npu, "b2")})
    # one boot each, so no device has two sets to compare: nothing is reported, and in particular the NVMe/NPU address
    # that happens to be equal is never offered as a shared buffer (before the fix, pooling them made it one).
    ok("an NPU address equal to an NVMe address is not a shared buffer", intersection(pooled), None)

    ok("printk's suppressed count parses", suppressed("x: amd_iommu_report_page_fault: 5611 callbacks suppressed"), {"amd_iommu_report_page_fault": 5611})
    ok("...and a journal with no suppression line reports none", suppressed(one), {})
    ok("...and the rendered page says the stream is larger than the printed count",
       "so the stream is at least 12" in render(census(parse(one + "y: amd_iommu_report_page_fault: 10 callbacks suppressed\n", "b1")),
                                                 None, False, {"b1": {"amd_iommu_report_page_fault": 10}}), True)

    noise = "Nov 14 23:40:00 host kernel: nvme nvme0: I/O 12 QID 1 timeout, aborting\n"
    ok("a line that is not a page fault is ignored", parse(noise, "b1"), {})

    ok(
        "a fault line with no device prefix still counts, device '-'",
        census(parse("AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x1000 flags=0x0000]\n", "b1"))[0]["device"],
        "-",
    )

    ok("one boot cannot answer the cross-boot question", intersection(c), None)
    ok("one fault on one address is the churn shape, not a buffer", "find the race" in shape(c[0]), True)

    # the review's NPU numbers, which the old repeats-only rule would have called a stale translation: 390 faults spread
    # over 321 distinct pages with at most 6 on any one of them.
    npu_real = {"0x%012x" % (0x7FC64EFC6000 + i * 0x1000): (6 if i < 3 else 1) for i in range(321)}
    npu_real["0x%012x" % 0x7FC64EFC6000] = 6
    n = census(parse("".join(
        "Nov 14 20:00 host kernel: nvme 0000:c6:00.1: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0002 address=%s flags=0x0000]\n"
        % a for a, c in npu_real.items() for _ in range(c)), "b0"))[0]
    ok("the box's NPU stream (321 distinct pages, max 6 each) is the churn shape", "find the race" in shape(n), True)
    ok("...and the box's NVMe stream (10 distinct, hundreds each) is the buffer shape",
       "find the buffer" in shape(census(parse("".join(
           "Nov 14 20:00 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=0x%x000 flags=0x0000]\n"
           % i for i in range(10) for _ in range(380)), "b-2"))[0]), True)

    # The counts engine#2478 actually measured in the boot that died: five addresses carrying most of the 2361 faults.
    # Until this arm existed, nothing tested the top-five ORDER -- only that the numbers were counted somewhere.
    real = {
        "0xebe54000": 301,
        "0xeb5fa000": 299,
        "0xe54de000": 297,
        "0xeb5f1000": 278,
        "0xebc8e000": 262,
        "0x1000": 1,
        "0x2000": 1,
    }
    text = "".join(
        "Nov 14 20:00 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0015 address=%s flags=0x0000]\n"
        % a
        for a, n in real.items()
        for _ in range(n)
    )
    c = census(parse(text, "died"))[0]
    ok("the measured boot censuses to 1439 faults over 7 distinct addresses", (c["faults"], c["distinct"]), (1439, 7))
    ok(
        "the top five keep the measured order, largest first",
        [(t["address"], t["count"]) for t in c["top"]],
        [("0xebe54000", 301), ("0xeb5fa000", 299), ("0xe54de000", 297), ("0xeb5f1000", 278), ("0xebc8e000", 262)],
    )
    ok("the top five stop at five and do not list the singletons", len(c["top"]), 5)

    # engine#2542: two flag values on one device must both appear -- a histogram with one bucket hides the outlier that
    # any decode has to explain, which is exactly how a wrong decode survived review.
    two_flags = (
        "Nov 14 20:00 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged "
        "[IO_PAGE_FAULT domain=0x0015 address=0x1000 flags=0x0000]\n"
        "Nov 14 20:01 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged "
        "[IO_PAGE_FAULT domain=0x0015 address=0x1000 flags=0x0050]\n"
        "Nov 14 20:02 host kernel: nvme 0000:c4:00.0: AMD-Vi: Event logged "
        "[IO_PAGE_FAULT domain=0x0015 address=0x2000 flags=0x0000]\n"
    )
    f = census(parse(two_flags, "b"))[0]
    ok("both flags values on one device are counted", f["flags"], {"0x0000": 2, "0x0050": 1})
    ok("...and the rare one is not lost in the total", f["faults"], 3)

    no_flags = (
        "AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0001 address=0x1000]\n"
    )
    ok("a fault line without a flags field still parses", census(parse(no_flags, "b"))[0]["faults"], 1)
    ok("  ...and it is COUNTED, not dropped from the histogram",
       census(parse(no_flags, "b"))[0]["flags"], {"(absent)": 1})
    # engine#2559 review: the histogram must partition the total, or the lines it could not bucket vanish from a
    # report that still says how many faults there were -- the same loss the histogram was added to prevent.
    for _e in census(parse(two_flags + no_flags, "b")):
        ok("  flags partition the fault count (%s)" % _e["device"], sum(_e["flags"].values()), _e["faults"])

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
    sup = {}
    if a.journal:
        for path in a.journal:
            try:
                with open(path, "r", errors="replace") as fh:
                    text = fh.read()
            except OSError as e:
                print("iommu_fault_census: cannot read %s: %s" % (path, e), file=sys.stderr)
                return 2
            rows.update(parse(text, path))
            drops = suppressed(text)
            if drops:
                sup[path] = drops
    else:
        text = sys.stdin.read()
        rows = parse(text, a.label)
        drops = suppressed(text)
        if drops:
            sup[a.label] = drops

    entries = census(rows)
    if not entries:
        print("no IO_PAGE_FAULT lines in the input", file=sys.stderr)
        return 0
    print(render(entries, intersection(entries), a.json, sup))
    return 0


if __name__ == "__main__":
    sys.exit(main())
