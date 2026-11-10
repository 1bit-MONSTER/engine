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
"""amdxdna_tdr_report.py -- what the amdxdna driver's TDR recovery did, read from the kernel log.

    tools/amdxdna_tdr_report.py                 this boot's kernel log (journalctl -k -b)
    tools/amdxdna_tdr_report.py --boot -1       another boot (journalctl's numbering)
    tools/amdxdna_tdr_report.py FILE [FILE...]  saved logs (dmesg or journalctl text); - is stdin
    tools/amdxdna_tdr_report.py --json ...      the same numbers as JSON
    tools/amdxdna_tdr_report.py --fail-on-failed-restart ...   exit 1 when a context restart failed
    tools/amdxdna_tdr_report.py --self-test

Read-only: it opens no device and changes nothing. Stdlib only.

Why it exists (engine#1750). A timeout (TDR) makes the driver stop every context, power-cycle the array and
restart the contexts. The question that issue asks is what happens to a context whose restart FAILS, and
the first thing anyone needs is how often that has happened on a box. The driver logs every step, but the
answer was a grep one had to know how to write. This prints it per module load:

    timeouts detected          every "TDR timeout detected"
    power-cycle recoveries     the driver reset the array and walked the contexts
    throttled                  inside the minimum interval between resets; no context was touched
    skipped contexts           being torn down by their owner (-ESRCH), not a failure
    failed restarts            "<ctx> restart failed, ret <n>", with the context, the error, and the step
                               that failed when the driver logged one
    latched                    "NPU array is unusable until the driver is reloaded"
    context creates refused    "Init hwctx failed, ret <n>", by error

For a failed restart it also says which behaviour the loaded module has: the carried patch 0013 logs
"no firmware context (teardown ret <n>), failing <k> parked job(s)" from the restart itself; a module
without it logs only the bare line from the TDR walk, and one without 0010 unpauses the context anyway.

A module load is a "Initialized amdxdna_accel_driver" line. Lines before the first one in the input are
reported as a load of their own, since a saved log often starts mid-boot.
"""
import argparse
import json
import re
import subprocess
import sys

LINE = re.compile(r"amdxdna [0-9a-f:.]+: (?:\[drm\] )?(?:\*ERROR\* )?(?P<fn>[A-Za-z0-9_]+): (?P<msg>.*)$")
LOAD = re.compile(r"Initialized amdxdna_accel_driver (?P<ver>\S+)")
FAILED = re.compile(r"^(?P<ctx>\S+) restart failed, ret (?P<ret>-?\d+)"
                    r"(?:: no firmware context \(teardown ret (?P<td>-?\d+)\), failing (?P<parked>\d+) parked job)?")
STEP = re.compile(r"^(Create hwctx failed|Map host buf failed|Add heap \d+ failed|Config cu failed),? ?ret (-?\d+)")
THROTTLED = re.compile(r"^TDR recovery throttled|^NPU power-cycle throttled")
SKIPPED = re.compile(r"^TDR: (\d+) context\(s\) skipped")
LATCHED = re.compile(r"^NPU array is unusable until the driver is reloaded")
NOT_RECOVERED = re.compile(r"^NPU power-cycle did not recover the array: (-?\d+)")
CREATE_REFUSED = re.compile(r"^Init hwctx failed, ret (-?\d+)")


def new_load(version):
    return {"driver": version, "timeouts": 0, "power_cycles": 0, "throttled": 0, "skipped_contexts": 0,
            "failed_restarts": [], "latched": 0, "not_recovered": [], "creates_refused": {}, "lines": 0}


def parse(lines):
    """-> a list of module loads, each a dict of counts. Lines that are not the driver's are ignored."""
    loads, cur, step = [], None, None
    for raw in lines:
        m = LOAD.search(raw)
        if m:
            cur = new_load(m.group("ver"))
            loads.append(cur)
            step = None
            continue
        m = LINE.search(raw)
        if not m:
            continue
        if cur is None:
            cur = new_load(None)
            loads.append(cur)
        cur["lines"] += 1
        fn, msg = m.group("fn"), m.group("msg").strip()
        if msg == "TDR timeout detected":
            cur["timeouts"] += 1
        elif "power-cycling NPU" in msg and fn == "aie2_tdr_recover_all":
            # aie2_hw_reset() logs its own "power-cycling NPU" line for the same reset: count the walk's
            cur["power_cycles"] += 1
            step = None
        elif THROTTLED.search(msg):
            cur["throttled"] += 1
        elif SKIPPED.search(msg):
            cur["skipped_contexts"] += int(SKIPPED.search(msg).group(1))
        elif LATCHED.search(msg):
            cur["latched"] += 1
        elif NOT_RECOVERED.search(msg):
            cur["not_recovered"].append(int(NOT_RECOVERED.search(msg).group(1)))
        elif CREATE_REFUSED.search(msg):
            ret = CREATE_REFUSED.search(msg).group(1)
            cur["creates_refused"][ret] = cur["creates_refused"].get(ret, 0) + 1
        elif fn == "aie2_hwctx_restart" and STEP.search(msg):
            step = STEP.search(msg).group(1)
        elif FAILED.search(msg):
            f = FAILED.search(msg)
            entry = {"context": f.group("ctx"), "ret": int(f.group("ret")), "step": step,
                     "logged_by": fn, "teardown_ret": None, "parked_jobs": None}
            if f.group("td") is not None:
                entry["teardown_ret"], entry["parked_jobs"] = int(f.group("td")), int(f.group("parked"))
            prev = cur["failed_restarts"][-1] if cur["failed_restarts"] else None
            # with 0013 the restart logs the failure and the TDR walk logs it again: one failure, not two
            if (prev and prev["context"] == entry["context"] and prev["ret"] == entry["ret"]
                    and prev["logged_by"] != fn and not prev.get("paired")):
                prev["paired"] = True
                if entry["teardown_ret"] is not None:
                    prev["teardown_ret"], prev["parked_jobs"] = entry["teardown_ret"], entry["parked_jobs"]
            else:
                cur["failed_restarts"].append(entry)
            step = None
    for load in loads:
        for f in load["failed_restarts"]:
            f.pop("paired", None)
            f.pop("logged_by", None)
            f["module_fails_parked_jobs"] = f["teardown_ret"] is not None      # the 0013 line
    return loads


def render(loads):
    out = []
    if not loads:
        return "no amdxdna lines in the input\n"
    for i, l in enumerate(loads):
        head = f"driver {l['driver']}" if l["driver"] else "before the first driver load line in the input"
        out.append(f"module load {i + 1}: {head} ({l['lines']} amdxdna lines)")
        out.append(f"  timeouts detected        {l['timeouts']}")
        out.append(f"  power-cycle recoveries   {l['power_cycles']}")
        out.append(f"  throttled                {l['throttled']}")
        out.append(f"  skipped contexts         {l['skipped_contexts']}")
        out.append(f"  failed restarts          {len(l['failed_restarts'])}")
        for f in l["failed_restarts"]:
            where = f" at '{f['step']}'" if f["step"] else ""
            if f["module_fails_parked_jobs"]:
                tail = (f"; firmware context taken away (teardown ret {f['teardown_ret']}), "
                        f"{f['parked_jobs']} parked job(s) failed with -ENODEV (patch 0013)")
            else:
                tail = "; the module does not say what happened to the parked jobs (no patch 0013)"
            out.append(f"    {f['context']}: ret {f['ret']}{where}{tail}")
        out.append(f"  latched (reload needed)  {l['latched']}")
        if l["not_recovered"]:
            out.append(f"  power-cycle did not recover the array: {len(l['not_recovered'])} "
                       f"(ret {', '.join(str(r) for r in l['not_recovered'])})")
        if l["creates_refused"]:
            by = ", ".join(f"ret {k} x{v}" for k, v in sorted(l["creates_refused"].items(), key=lambda kv: -kv[1]))
            out.append(f"  context creates refused  {sum(l['creates_refused'].values())} ({by})")
    return "\n".join(out) + "\n"


def read_inputs(args):
    if args.files:
        for p in args.files:
            f = sys.stdin if p == "-" else open(p, errors="replace")
            with f:
                yield from f
        return
    cmd = ["journalctl", "-k", "--no-pager", "-b", str(args.boot)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    except OSError as e:
        raise SystemExit(f"cannot run {' '.join(cmd)}: {e}; pass a saved log file instead")
    if r.returncode:
        raise SystemExit(f"{' '.join(cmd)} exited {r.returncode}: {r.stderr.strip()[:300]}; pass a saved log file instead")
    yield from r.stdout.splitlines()


# Lines as the driver prints them. The first block is from a box running pin + 0001-0009; the failed
# restarts are written from the format strings in the patched source (0010 alone, then 0013).
P = "Nov 09 04:46:22 host kernel: amdxdna 0000:c6:00.1: [drm] "
SELF = [
    "Nov 08 23:11:19 host kernel: [drm] Initialized amdxdna_accel_driver 0.18.0 for 0000:c6:00.1 on minor 0",
    P + "*ERROR* aie2_legacy_tdr_detect: TDR timeout detected",
    P + "aie2_tdr_work_func: Device isn't making progress",
    P + "*ERROR* aie2_set_cmd_timeout: Firmware timeout state capture:",
    P + "*ERROR* aie2_set_cmd_timeout:         Context ID: 15",
    P + "aie2_tdr_recover_all: Array stopped dispatching, power-cycling NPU to recover",
    P + "aie2_hw_reset: NPU firmware unhealthy: power-cycling NPU",
    P + "*ERROR* aie2_legacy_tdr_detect: TDR timeout detected",
    P + "aie2_tdr_recover_all: TDR recovery throttled: last power-cycle 45560 ms ago (< 60000 ms), retrying on "
        "the next watchdog tick; no context touched",
    P + "*ERROR* aie2_legacy_tdr_detect: TDR timeout detected",
    P + "aie2_tdr_recover_all: Array stopped dispatching, power-cycling NPU to recover",
    P + "aie2_tdr_recover_all: TDR: 2 context(s) skipped, being torn down",
    P + "*ERROR* aie2_hwctx_init: Alloc hw resource failed, ret -19",
    P + "*ERROR* amdxdna_drm_create_hwctx_ioctl: Init hwctx failed, ret -19",
    P + "*ERROR* amdxdna_drm_create_hwctx_ioctl: Init hwctx failed, ret -16",
    P + "*ERROR* amdxdna_drm_create_hwctx_ioctl: Init hwctx failed, ret -19",
    "Nov 09 05:00:00 host kernel: usb 1-1: new high-speed USB device number 4",          # not ours
    # a second load, a module with 0010 only: the TDR walk logs the failure, and the array latches
    "[  812.100000] [drm] Initialized amdxdna_accel_driver 0.18.0 for 0000:c6:00.1 on minor 0",
    "[  900.000000] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_legacy_tdr_detect: TDR timeout detected",
    "[  900.000100] amdxdna 0000:c6:00.1: [drm] aie2_tdr_recover_all: Firmware unresponsive, power-cycling NPU",
    "[  900.500000] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hwctx_restart: Map host buf failed, ret -22",
    "[  900.500100] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_tdr_resume_hwctx: hwctx.4242.15 restart failed, ret -22",
    "[  900.500200] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_tdr_recover_all: NPU array is unusable until the driver is "
    "reloaded: TDR recovery failed (0, a context could not be restarted); new contexts and submissions are refused "
    "with -ENODEV",
    # a third load, a module with 0013: the restart logs it (with the jobs), then the walk logs it again
    "[ 1500.000000] [drm] Initialized amdxdna_accel_driver 0.18.0 for 0000:c6:00.1 on minor 0",
    "[ 1600.000000] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_legacy_tdr_detect: TDR timeout detected",
    "[ 1600.000100] amdxdna 0000:c6:00.1: [drm] aie2_tdr_recover_all: Array stopped dispatching, power-cycling NPU to recover",
    "[ 1600.400000] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hwctx_restart: Create hwctx failed, ret -5",
    "[ 1600.400100] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hwctx_restart: hwctx.77.14 restart failed, ret -5: no firmware "
    "context (teardown ret 0), failing 3 parked job(s) and later submissions with -ENODEV",
    "[ 1600.400200] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_tdr_resume_hwctx: hwctx.77.14 restart failed, ret -5",
    "[ 1600.400300] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hwctx_restart: Config cu failed, ret -110",
    "[ 1600.400400] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hwctx_restart: hwctx.78.13 restart failed, ret -110: no firmware "
    "context (teardown ret 0), failing 0 parked job(s) and later submissions with -ENODEV",
    "[ 1600.400500] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_tdr_resume_hwctx: hwctx.78.13 restart failed, ret -110",
    "[ 1600.400600] amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_tdr_recover_all: NPU power-cycle did not recover the array: -110",
]


def self_test():
    fails = []

    def check(good, what):
        if not good:
            fails.append(what)

    loads = parse(SELF)
    check(len(loads) == 3, f"three module loads, got {len(loads)}")
    a, b, c = (loads + [new_load(None)] * 3)[:3]
    check(a["driver"] == "0.18.0", "the driver version is read from the load line")
    check(a["timeouts"] == 3, f"load 1: 3 timeouts, got {a['timeouts']}")
    check(a["power_cycles"] == 2, f"load 1: 2 power-cycles (aie2_hw_reset's own line is not a second one), got {a['power_cycles']}")
    check(a["throttled"] == 1 and a["skipped_contexts"] == 2, "load 1: 1 throttled, 2 skipped contexts")
    check(a["failed_restarts"] == [] and a["latched"] == 0, "load 1: no failed restart, not latched")
    check(a["creates_refused"] == {"-19": 2, "-16": 1}, f"load 1: creates refused by error, got {a['creates_refused']}")
    check(a["lines"] == 15, f"load 1: 15 driver lines (the usb line is not one), got {a['lines']}")
    fb = b["failed_restarts"]
    check(len(fb) == 1 and fb[0]["context"] == "hwctx.4242.15" and fb[0]["ret"] == -22, "load 2: one failed restart, named")
    check(fb and fb[0]["step"] == "Map host buf failed", "load 2: the failing step is the driver's own line before it")
    check(fb and fb[0]["module_fails_parked_jobs"] is False, "load 2: a bare line means no patch 0013")
    check(b["latched"] == 1 and b["power_cycles"] == 1, "load 2: latched, and 'Firmware unresponsive' is a power-cycle too")
    fc = c["failed_restarts"]
    check(len(fc) == 2, f"load 3: the restart's line and the walk's line are one failure each, got {len(fc)}")
    check(len(fc) == 2 and fc[0]["parked_jobs"] == 3 and fc[0]["teardown_ret"] == 0 and fc[0]["step"] == "Create hwctx failed",
          "load 3: the 0013 line gives the parked jobs and the teardown result")
    check(len(fc) == 2 and fc[1]["context"] == "hwctx.78.13" and fc[1]["ret"] == -110 and fc[1]["step"] == "Config cu failed"
          and fc[1]["parked_jobs"] == 0, "load 3: the second context keeps its own step and count")
    check(all(f["module_fails_parked_jobs"] for f in fc), "load 3: the module is one with patch 0013")
    check(c["not_recovered"] == [-110], "load 3: a power-cycle that did not recover is reported with its error")
    text = render(loads)
    check("failed restarts          0" in text and "context creates refused  3 (ret -19 x2, ret -16 x1)" in text, "the text for load 1")
    check("hwctx.77.14: ret -5 at 'Create hwctx failed'; firmware context taken away (teardown ret 0), 3 parked job(s) "
          "failed with -ENODEV (patch 0013)" in text, "the text for a 0013 failure")
    check("hwctx.4242.15: ret -22 at 'Map host buf failed'; the module does not say" in text, "the text for a bare failure")
    check(parse(["Nov 09 host kernel: usb 1-1: hello"]) == [] and "no amdxdna lines" in render([]), "an input with none of ours")
    mid = parse(SELF[1:6])
    check(len(mid) == 1 and mid[0]["driver"] is None and mid[0]["power_cycles"] == 1, "a log that starts mid-boot is still counted")
    json.dumps(loads)
    for f in fails:
        print("FAIL " + f)
    print("FAILED" if fails else "amdxdna_tdr_report: ok")
    return 1 if fails else 0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("files", nargs="*", help="saved kernel logs (- is stdin); default: journalctl -k -b")
    p.add_argument("--boot", default="0", help="journalctl boot to read when no file is given (default 0, this boot)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--fail-on-failed-restart", action="store_true", help="exit 1 when any context restart failed")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    loads = parse(read_inputs(a))
    sys.stdout.write(json.dumps(loads, indent=1) + "\n" if a.json else render(loads))
    if a.fail_on_failed_restart and any(l["failed_restarts"] for l in loads):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
