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
DOMAIN = re.compile(r"domain=(?P<dom>0x[0-9a-f]+)")
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
            "flags": int(m.group("flags"), 16),
            "domain": (int(d.group("dom"), 16) if (d := DOMAIN.search(line)) else None)}


TS = re.compile(r"^\w{3}\s+\d+ (\d\d):(\d\d):(\d\d)")


def seconds_of_day(line):
    """`Nov 16 12:35:26` -> 45236. None when the line carries no syslog stamp."""
    m = TS.match(line)
    if m is None:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def split_by_gap(stamped, gap):
    """[(seconds, event)] -> [[event, ...]]; a new burst starts when the gap exceeds `gap`.

    A day's log is many bursts, and treating it as one is how the first version of
    this tool reported 90 accesses in a single "burst" whose span was 800 GiB. A
    clock that goes backwards is a date rollover in a syslog without a year, so it
    counts as a large positive gap rather than a negative one.
    """
    if not stamped:
        return []
    out, cur, prev = [], [stamped[0][1]], stamped[0][0]
    for ts, ev in stamped[1:]:
        step = ts - prev
        if step < 0:
            step += 24 * 3600
        if step > gap:
            out.append(cur)
            cur = []
        cur.append(ev)
        prev = ts
    out.append(cur)
    return out


def _gaps(stamped):
    """The inter-access gaps in seconds (rollover-corrected), for showing a split choice."""
    out = []
    for (a, _), (b, _) in zip(stamped, stamped[1:]):
        step = b - a
        if step < 0:
            step += 24 * 3600
        out.append(step)
    return out


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
        return {"count": 0, "devices": [], "flags": [], "domains": [], "span": 0,
                "stride": None, "pages": 0, "repeats": 0, "one_40k_window": False}
    addrs = [e["address"] for e in events]
    pages = sorted({a & ~0xFFF for a in addrs})
    steps = {b - a for a, b in zip(pages, pages[1:])}
    span = max(addrs) - min(addrs)
    return {
        "count": len(events),
        "devices": sorted({e["device"] for e in events}),
        "flags": sorted({e["flags"] for e in events}),
        # The IOMMU DOMAIN separates a host VA (0x0002 on this box) from a device
        # aperture (0x0003): a fault in the aperture has no host page behind it, so a
        # page class read for it would describe a different device's memory (#2777).
        "domains": sorted({e["domain"] for e in events if e.get("domain") is not None}),
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
    doms = ",".join("0x%04x" % d for d in summary["domains"]) or "none"
    out = ["burst: %d access(es) on %s, domain %s, flags %s, %d page(s), span 0x%x, %s"
           % (summary["count"], ",".join(summary["devices"]) or "none", doms,
              ",".join("0x%04x" % f for f in summary["flags"]) or "none",
              summary["pages"], summary["span"], stride)]
    if len(summary["domains"]) > 1:
        out.append("  MORE THAN ONE IOMMU DOMAIN in this burst: host VA and aperture mixed")
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
                              "flags": 0x0077, "domain": 0x0002},
        "an amdxdna fault line parses, with its IOMMU domain")
    chk(parse_fault("amdxdna 0000:c6:00.1 address=0x1000 flags=0x0007")["domain"] is None,
        "a line with no domain parses with domain None, not a fabricated 0")
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
    # #2777: the domain separates a host VA (0x0002 here) from a device aperture (0x0003).
    # Mixing them in one burst is a different finding, not an average.
    a_v = dict(device="0000:c6:00.1", address=0x7f2900001000, flags=0x0007, domain=0x0002)
    ap_v = dict(device="0000:c6:00.1", address=0x7f291bd63000, flags=0x0007, domain=0x0003)
    chk(summarize([a_v, a_v])["domains"] == [0x0002], "one domain stays one domain")
    chk(summarize([a_v, ap_v])["domains"] == [0x0002, 0x0003],
        "a burst spanning host VA and aperture reports BOTH domains")

    # the split threshold is a CHOICE, and its consequence must be visible:
    # today's nine bursts are separated by >= 17 s while a burst's own accesses are
    # all in the same second, so 5 s and 10 s both separate them and 30 s does not.
    ev = dict(device="0000:c6:00.1", address=0x1000, flags=0x0007)
    two = [(0, ev), (0, ev), (17, ev), (17, ev)]
    chk(len(split_by_gap(two, 5)) == 2, "two bursts 17 s apart split at a 5 s gap")
    chk(len(split_by_gap(two, 30)) == 1, "the SAME two bursts do NOT split at a 30 s gap")
    chk([len(b) for b in split_by_gap(two, 5)] == [2, 2], "each burst keeps its own count")
    # 23:59:58 -> 00:00:03 is a +5 s gap, not a -86395 s one. Without the rollover
    # correction the second burst would join the first and this arm would pass only
    # because nothing checked it.
    # 23:59:50 -> 00:00:10 is a +20 s gap, i.e. a NEW burst. Without the rollover
    # correction the step is -86380, which reads as "same burst" and merges them.
    chk(len(split_by_gap([(86390, ev), (10, ev)], 5)) == 2,
        "a gap that crosses midnight is +20 s (a new burst), not -86380 s")
    chk(len(split_by_gap([(86398, ev), (3, ev)], 5)) == 1,
        "and 23:59:58 -> 00:00:03 is +5 s: the same burst, not two")
    chk(seconds_of_day("Nov 16 12:35:26 x") == 12 * 3600 + 35 * 60 + 26,
        "the syslog stamp parses to seconds of day (45326, computed here not copied)")
    chk(seconds_of_day("no stamp here") is None, "a line without a stamp has no time")

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

    # engine#2755 review, twice over: a REPLAY must not present today's /proc as the burst's holders, and a live
    # capture must end at its window even when nothing else arrives.
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        proc = os.path.join(tmp, "proc")
        os.makedirs(os.path.join(proc, "4242", "fd"))
        os.symlink(DEV_ACCEL + "/accel0", os.path.join(proc, "4242", "fd", "3"))
        log = os.path.join(tmp, "burst.txt")
        with open(log, "w") as fh:
            fh.write("Nov 16 15:42:45 strixhalo kernel: amdxdna 0000:c6:00.1: AMD-Vi: Event logged "
                     "[IO_PAGE_FAULT domain=0x0002 address=0x7fa381807000 flags=0x0077]\n")
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--replay", log, "--proc-root", proc],
                             capture_output=True, text=True)
        chk("pid 4242" not in out.stdout and "not captured" in out.stdout,
            "a replay does not snapshot today's /proc as the burst's holder: %s" % out.stdout[:120])

        # The window closes with no further fault line: feed one burst down a pipe that stays open.
        p2 = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--window", "0.5", "--proc-root", proc],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p2.stdin.write("Nov 16 15:42:45 strixhalo kernel: amdxdna 0000:c6:00.1: AMD-Vi: Event logged "
                       "[IO_PAGE_FAULT domain=0x0002 address=0x7fa381807000 flags=0x0077]\n")
        p2.stdin.flush()
        try:
            p2.wait(timeout=10)
            early = p2.returncode == 0
        except subprocess.TimeoutExpired:
            p2.kill()
            early = False
        chk(early, "a live capture ends at its window instead of waiting for another fault line")

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
    ap.add_argument("--split", type=float, default=0.0,
                    help="replay only: start a new burst when the log gap exceeds this many "
                         "seconds (0 = treat the whole file as one burst)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    events, snaps, total, stamped = [], [], 0, []
    replay_note = [False]   # a replay cannot snapshot holders: /proc is today's
    src = None
    if a.replay:
        src = open(a.replay)
    # engine#2755 review: the window must close even when NOTHING else arrives. Checking the clock at the top of
    # the loop does not do that -- the read itself blocks -- so a live source is read through select() with the
    # remaining time, and a quiet window ends the capture instead of sitting on stdin.
    def live_lines(stream, until):
        import select
        while True:
            left = None if until[0] is None else until[0] - time.time()
            if left is not None and left <= 0:
                return
            ready, _, _ = select.select([stream], [], [], left)
            if not ready:
                return                       # the window closed with nothing more to read
            line = stream.readline()
            if not line:
                return
            yield line

    until = [None]        # set to time()+window when the first fault is seen
    source = src if src is not None else (sys.stdin if a.replay else live_lines(sys.stdin, until))
    for line in source:
        total += 1
        ev = parse_fault(line, a.device)
        if ev is None:
            continue
        ts = seconds_of_day(line)
        if ts is not None:
            stamped.append((ts, ev))
        events.append(ev)
        if len(events) == 1:
            # Taken on the FIRST fault, not after the burst: the client may be gone
            # by the time the window closes, and a snapshot of nobody is the
            # reading this tool exists to avoid making the default.
            #
            # engine#2755 review: NOT in a replay. `/proc` is today's, and the burst is
            # whatever the file recorded, so a snapshot taken here would name the wrong
            # process with full confidence -- the exact mis-attribution this tool exists
            # to prevent. The block says which it is instead of leaving the reader to guess.
            if a.replay is None:
                for pid in holders(a.proc_root, a.dev_prefix, self_pid=os.getpid()):
                    d = describe(pid, a.proc_root)
                    snaps.append((pid, *(d if d else ("gone", ""))))
            else:
                replay_note[0] = True
            deadline = time.time() + a.window
            until[0] = deadline
        elif a.replay is None and time.time() > deadline:
            break
    if src is not None:
        src.close()

    if not events:
        print("no %s fault in %d line(s) examined: nothing to capture "
              "(an empty capture is not evidence of no fault)" % (a.device, total))
        return 2
    bursts = split_by_gap(stamped, a.split) if a.split > 0 else []
    if bursts:
        # A replay of a day is many bursts; the per-burst count is the finding, so it
        # is printed per burst rather than summarised away.
        counts = [len(b) for b in bursts]
        for i, b in enumerate(bursts, 1):
            print("burst %d/%d" % (i, len(bursts)))
            print(render(summarize(b), []))
        # The threshold is a choice, so show what it was chosen against: the DISTINCT
        # non-zero gaps. Listing all 89 gaps (mostly 0 s, inside a burst) hides the
        # one that matters -- the smallest gap between bursts.
        seen = sorted({g for g in _gaps(stamped) if g > 0})
        if seen:
            print("--split %.0fs: %d burst(s); the gaps between bursts in this log are %s"
                  % (a.split, len(bursts), ",".join("%.0fs" % g for g in seen)))
        print("%d burst(s); access counts %s%s"
              % (len(bursts), sorted(counts),
                 " -- all equal" if len(set(counts)) == 1 else " -- NOT all equal"))
        if replay_note[0]:
            print("  holders: not captured (--replay: /proc is today's, so a snapshot here would name the "
                  "wrong process for a burst this file recorded)")
        return 0
    for line in render(summarize(events), snaps).splitlines():
        print(line)
    if replay_note[0]:
        print("  holder: not captured (--replay: /proc is today's, so a snapshot here would name the "
              "wrong process for a burst this file recorded)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
