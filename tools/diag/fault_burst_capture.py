#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
"""Engine#2542: capture WHO was submitting at the instant of an IOMMU fault burst.

The measured shape of the amdxdna fault on this box (2026-11-16 15:42:45 AST) is
**ten faults in one second, ten consecutive 4 KiB pages, one flag word**:

    0x7fa381800000 0x7fa381801000 ... 0x7fa381809000   flags=0x0077 x10
    device: amdxdna 0000:c6:00.1

That is one operation walking a contiguous range whose mapping is missing -- not a
stream and not ten clients. Two consequences this tool exists for:

  * an event that lasts one second cannot be attributed by a census taken minutes
    later: by then the client has exited and `/proc` shows nobody holding the
    device. The snapshot has to be taken WHEN the burst fires;
  * counting `IO_PAGE_FAULT` without the flags merges classes that differ in the
    field saying what kind of access failed (amdxdna `0x0077` vs nvme `0x0000`).

So: watch a log source, and on the FIRST matching line, collect the burst for
`--window` seconds, then snapshot the device holders immediately and record one
block. `--replay` runs the same path over a file (used by `--self-test`).

The parse and summary functions are pure and exercised by `--self-test`; the
snapshot half is exercised against a fake `/proc` tree, including the arms that
must NOT fabricate an answer (no holders, non-uniform stride, mixed flags).
"""

import argparse
import os
import re
import sys
import time

FAULT = re.compile(
    r"(?P<dev>[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])[: ].*?"
    r"address=(?P<addr>0x[0-9a-f]+).*?flags=(?P<flags>0x[0-9a-f]+)")
DEV_ACCEL = "/dev/accel"


def parse_fault(line, device_prefix="amdxdna"):
    """One log line -> {device, address, flags}, or None.

    The device name is required: an nvme fault parsed as an amdxdna one is a
    category error, and this lane has already measured 137 of those attributed to
    an NPU ticket. A caller that wants another device passes `device_prefix`.
    """
    if device_prefix not in line:
        return None
    m = FAULT.search(line)
    if m is None:
        return None
    return {"device": m.group("dev"), "address": int(m.group("addr"), 16),
            "flags": int(m.group("flags"), 16)}


def summarize(events):
    """The burst as one record. Absences are reported, never filled in.

    Address ORDER in the log is not address order on the device: the 12:35 burst
    today reports 0x63000, 0x68000, 0x64000, 0x69000, ... -- the hardware reports
    the walking accesses out of sequence. So "is this one ascending run" is the
    wrong question, and an earlier version of this function answered it anyway and
    called a real single walk "not a walk".

    What is order-independent, and therefore what this reports:

      * `pages` -- the distinct 4 KiB pages touched;
      * `repeats` -- accesses beyond one per PAGE (the 12:35 burst is 10 accesses
        over 7 pages: three pages fault at offset 0 and again at 0x400, so a count
        of duplicate addresses would report 0 repeats and describe the wrong thing);
      * `span` -- max minus min address, i.e. the range the submitter walked;
      * `one_40k_window` -- the whole burst inside one 40 KiB window, which is the
        shape a single command descriptor block produces.

    `devices` and `flags` stay sets: a burst on two devices or with two flag words
    is not one kind of event, and collapsing either hides the distinction the tool
    exists for. `stride` survives only for the all-distinct, evenly spaced case and
    is None otherwise -- a stride invented from two far-apart addresses would
    describe a burst that did not happen.
    """
    if not events:
        return {"count": 0, "devices": [], "flags": [], "span": 0, "stride": None,
                "pages": 0, "repeats": 0, "one_40k_window": False}
    addrs = [e["address"] for e in events]
    pages = sorted({a & ~0xFFF for a in addrs})
    steps = {b - a for a, b in zip(pages, pages[1:])}
    span = max(addrs) - min(addrs)
    return {
        "count": len(events),
        "devices": sorted({e["device"] for e in events}),
        "flags": sorted({e["flags"] for e in events}),
        "span": span,
        "pages": len(pages),
        "repeats": len(addrs) - len(pages),
        "one_40k_window": span < 40 * 1024,
        "stride": (steps.pop() if len(steps) == 1 and len(addrs) == len(set(addrs))
                   else None),
    }


def holders(proc_root, prefix=DEV_ACCEL, self_pid=None):
    """pids with an fd whose link target is under `prefix`, ascending.

    A prefix that does not name a device must name nobody: an empty prefix would
    otherwise match every fd on the box, which reads as "everyone is using the
    NPU".
    """
    if not prefix:
        return []
    out = []
    try:
        names = os.listdir(proc_root)
    except OSError:
        return []
    for name in names:
        if not name.isdigit():
            continue
        pid = int(name)
        if self_pid is not None and pid == self_pid:
            continue
        fddir = os.path.join(proc_root, name, "fd")
        try:
            fds = os.listdir(fddir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fddir, fd))
            except OSError:
                continue
            if target == prefix or target.startswith(prefix + "/"):
                out.append(pid)
                break
    return sorted(out)


def describe(pid, proc_root):
    """pid -> (comm, cmdline) or None when the process is gone."""
    base = os.path.join(proc_root, str(pid))
    try:
        with open(os.path.join(base, "comm")) as f:
            comm = f.read().strip()
    except OSError:
        return None
    try:
        with open(os.path.join(base, "cmdline"), "rb") as f:
            cmdline = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        cmdline = ""
    return (comm, cmdline)


def render(summary, snaps):
    """The block a reader gets. Every field that could not be read says so."""
    stride = ("0x%x across pages" % summary["stride"]) if summary["stride"] is not None \
        else "no single page stride"
    out = ["burst: %d access(es) on %s, flags %s, %d page(s), span 0x%x, %s"
           % (summary["count"], ",".join(summary["devices"]) or "none",
              ",".join("0x%04x" % f for f in summary["flags"]) or "none",
              summary["pages"], summary["span"], stride)]
    if summary["repeats"]:
        out.append("  %d access(es) repeat a page already faulted in this burst"
                   % summary["repeats"])
    if summary["one_40k_window"]:
        out.append("  the whole burst is inside one 40 KiB window: one descriptor block's walk")
    if len(summary["devices"]) > 1:
        out.append("  MORE THAN ONE DEVICE in this burst: not one client's walk")
    if not snaps:
        out.append("  holders: none (no process had an fd under the device during the burst)")
    for pid, comm, cmdline in snaps:
        out.append("  holder: pid %d %s %s" % (pid, comm, cmdline[:80]))
    return "\n".join(out)


def self_test():
    """Arms for the parse, the summary and the snapshot. Exit 0 iff all pass."""
    bad = 0

    def chk(cond, what):
        nonlocal bad
        if not cond:
            print("FAIL %s" % what)
            bad += 1

    line = ("Nov 16 15:42:45 strixhalo kernel: amdxdna 0000:c6:00.1: AMD-Vi: "
            "Event logged [IO_PAGE_FAULT domain=0x0002 address=0x7fa381807000 flags=0x0077]")
    chk(parse_fault(line) == {"device": "0000:c6:00.1", "address": 0x7fa381807000,
                              "flags": 0x0077}, "an amdxdna fault line parses")
    chk(parse_fault(line, device_prefix="nvme") is None,
        "the nvme stream does not parse as an amdxdna fault")

    pages = [dict(device="0000:c6:00.1", address=0x7fa381800000 + 0x1000 * i,
                  flags=0x0077) for i in range(10)]
    s = summarize(pages)
    chk(s["count"] == 10 and s["stride"] == 0x1000 and s["pages"] == 10,
        "ten consecutive pages summarise as one 0x1000 walk over ten pages")
    chk(s["one_40k_window"], "ten pages starting at 0x…000 is inside one 40 KiB window")
    chk(s["span"] == 9 * 0x1000, "span is nine strides, not ten")
    chk(summarize(pages[:1])["stride"] is None,
        "a single fault has NO stride (it must not invent one)")
    chk(summarize([])["count"] == 0, "an empty burst is countable, not an error")
    mixed_dev = pages + [dict(device="0000:c4:00.0", address=0x5000, flags=0x0000)]
    chk(len(summarize(mixed_dev)["devices"]) == 2,
        "two devices stay two devices in the summary")
    chk(len(summarize(mixed_dev)["flags"]) == 2,
        "two flag words stay two flag words in the summary")
    chk(summarize(list(reversed(pages)))["stride"] == 0x1000,
        "REVERSED log order is still one 0x1000 walk: the device reports out of sequence")
    chk(summarize(list(reversed(pages)))["count"] == 10,
        "reversing the log does not change what the burst contains")
    # the 12:35 burst measured today: 10 accesses, 7 distinct pages, three pages
    # faulting twice (offset 0 and offset 0x400), reported out of address order
    mixed = [dict(device="0000:c6:00.1", address=base, flags=0x0007) for base in
             (0x7f291bd63000, 0x7f291bd68000, 0x7f291bd64000, 0x7f291bd69000,
              0x7f291bd65000, 0x7f291bd66000, 0x7f291bd67000, 0x7f291bd63400,
              0x7f291bd64400, 0x7f291bd65400)]
    m = summarize(mixed)
    chk(m["count"] == 10 and m["pages"] == 7,
        "the 12:35 shape is 10 accesses over 7 pages")
    chk(m["repeats"] == 3, "three of those accesses repeat a page already faulted")
    chk(m["one_40k_window"], "and the whole 12:35 burst is inside one 40 KiB window")
    chk(summarize(pages + [pages[3]])["repeats"] == 1,
        "a repeated address is counted as a repeat, not hidden")
    far = [dict(device="0000:c6:00.1", address=0x100000, flags=0x0007),
           dict(device="0000:c6:00.1", address=0x200000, flags=0x0007)]
    chk(summarize(far)["one_40k_window"] is False,
        "accesses a megabyte apart are NOT one descriptor block's window")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        proc = os.path.join(tmp, "proc")
        for pid, target in (("4242", DEV_ACCEL + "/accel0"), ("4343", "/dev/null"),
                            ("4444", "/dev/accelx0")):
            os.makedirs(os.path.join(proc, pid, "fd"))
            os.symlink(target, os.path.join(proc, pid, "fd", "3"))
        chk(holders(proc) == [4242], "only the fd under the prefix is a holder")
        chk(holders(proc, self_pid=4242) == [], "the caller is never named a holder")
        chk(holders(proc, prefix="") == [], "an empty prefix names nobody")
        chk(holders(os.path.join(tmp, "nope")) == [], "a missing proc root names nobody")
        with open(os.path.join(proc, "4242", "comm"), "w") as f:
            f.write("flm-runner\n")
        chk(describe(4242, proc) == ("flm-runner", ""), "comm and an empty cmdline read")
        chk(describe(9999, proc) is None, "a vanished pid is None, not a fabricated name")

    block = render(s, [(4242, "flm-runner", "flm --serve")])
    chk("holder: pid 4242" in block, "the block names the holder")
    chk("none" in render(summarize([]), []), "an empty capture says none, not blank")

    print("%s" % ("ok" if bad == 0 else "%d FAIL" % bad))
    return 1 if bad else 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", help="log file to replay instead of watching kmsg")
    ap.add_argument("--window", type=float, default=2.0,
                    help="seconds of burst to collect after the first fault")
    ap.add_argument("--device", default="amdxdna", help="device name the line must carry")
    ap.add_argument("--dev-prefix", default=DEV_ACCEL, help="fd prefix that means 'holds the device'")
    ap.add_argument("--proc-root", default="/proc")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    events, snaps, total = [], [], 0
    src = None
    if a.replay:
        src = open(a.replay)
    for line in src if src is not None else sys.stdin:
        total += 1
        ev = parse_fault(line, a.device)
        if ev is None:
            continue
        events.append(ev)
        if len(events) == 1:
            # Taken on the FIRST fault, not after the burst: the client may be gone
            # by the time the window closes, and a snapshot of nobody is the
            # reading this tool exists to avoid making the default.
            for pid in holders(a.proc_root, a.dev_prefix, self_pid=os.getpid()):
                d = describe(pid, a.proc_root)
                snaps.append((pid, *(d if d else ("gone", ""))))
            deadline = time.time() + a.window
        elif a.replay is None and time.time() > deadline:
            break
    if src is not None:
        src.close()

    if not events:
        print("no %s fault in %d line(s) examined: nothing to capture "
              "(an empty capture is not evidence of no fault)" % (a.device, total))
        return 2
    print(render(summarize(events), snaps))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
