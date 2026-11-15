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
"""driver_refusal_census.py: name the driver's refusals instead of leaving a bare number.

`amdxdna` says two different things about a bad command, and they mean opposite things (engine#2542):

    [drm] Unsupported op %d                                  the dispatch switch has NO handler; nothing ran
    [drm] *ERROR* %s: command opcode 0x%x failed, status 0x%x  a handler ran and the firmware rejected it

The first prints an **ERT** command op in **decimal**; the second prints a **MSG_OP_*** mailbox opcode in **hex**. Two
namespaces, one log. `Unsupported op 18` is `ERT_START_DPU` (XRT `xrt/detail/ert.h`: `ERT_ACCESS_TEST = 17`,
`ERT_START_DPU = 18`, `ERT_CMD_CHAIN = 19`, `ERT_START_NPU = 20`) -- and 0x12 in the mailbox namespace is a *different*
op (`MSG_OP_CHAIN_EXEC_BUFFER_CF`), which is the coincidence that makes a lookup table worth having rather than a guess.

    driver_refusal_census.py [--journal FILE]... [--json] [--self-test]

Exit: 0 no refusal lines, 1 refusals found (they are findings, not noise), 2 unreadable input.
"""
import argparse
import json
import re
import sys

# The ERT command opcodes, as XRT defines them (xrt/detail/ert.h, the enum at the end of the file). Only the ones a
# driver can plausibly be asked for are listed; `unsupported` names every value the switch has no case for, which is
# the interesting set, but a table of all 23 would suggest this file is the authority. It is not: ert.h is.
ERT = {
    0: "ERT_START_CU",
    2: "ERT_CONFIGURE",
    5: "ERT_EXEC_WRITE",
    7: "ERT_START_COPYBO",
    11: "ERT_INIT_CU",
    12: "ERT_START_FA",
    17: "ERT_ACCESS_TEST",
    18: "ERT_START_DPU",
    19: "ERT_CMD_CHAIN",
    20: "ERT_START_NPU",
    21: "ERT_START_NPU_PREEMPT",
    22: "ERT_START_NPU_PREEMPT_ELF",
}

UNSUPPORTED = re.compile(r"\[drm\] Unsupported op (\d+)")
FAILED = re.compile(r"command opcode 0x([0-9a-fA-F]+) failed, status 0x([0-9a-fA-F]+)")


def scan(lines):
    """{kind: [(count, detail)]}, with the ERT name resolved when the table has it."""
    out = {"unsupported": {}, "failed": {}}
    for line in lines:
        m = UNSUPPORTED.search(line)
        if m:
            op = int(m.group(1))
            key = "op %d (%s)" % (op, ERT.get(op, "not in the ERT table"))
            out["unsupported"][key] = out["unsupported"].get(key, 0) + 1
            continue
        m = FAILED.search(line)
        if m:
            # the mailbox namespace: reported as the driver prints it, never mapped onto the ERT table
            key = "MSG_OP 0x%s, status 0x%s" % (m.group(1).lower(), m.group(2).lower())
            out["failed"][key] = out["failed"].get(key, 0) + 1
    return out


def report(found, name):
    lines = []
    for key, n in sorted(found["unsupported"].items(), key=lambda kv: -kv[1]):
        lines.append("%s %s: %d x driver has no handler -- nothing ran; this is a protocol/version disagreement, "
                     "not a fault" % (name, key, n))
    for key, n in sorted(found["failed"].items(), key=lambda kv: -kv[1]):
        lines.append("%s %s: %d x a handler ran and the firmware rejected it (mailbox namespace, hex as printed)"
                     % (name, key, n))
    return lines


def self_test():
    fails = 0

    def chk(ok, what):
        nonlocal fails
        if not ok:
            print("FAIL " + what)
            fails += 1

    found = scan([
        "amdxdna 0000:c6:00.1: [drm] Unsupported op 18",
        "amdxdna 0000:c6:00.1: [drm] Unsupported op 18",
        "amdxdna 0000:c6:00.1: [drm] *ERROR* x: command opcode 0x12 failed, status 0x1",
        "amdxdna 0000:c6:00.1: [drm] Unsupported op 99",
    ])
    chk(found["unsupported"].get("op 18 (ERT_START_DPU)") == 2,
        "op 18 is named ERT_START_DPU and counted twice: %r" % (found["unsupported"],))
    chk(found["unsupported"].get("op 99 (not in the ERT table)") == 1,
        "an unknown op says so instead of guessing a name: %r" % (found["unsupported"],))
    chk(found["failed"].get("MSG_OP 0x12, status 0x1") == 1,
        "a failed command is reported in the mailbox namespace, hex as printed: %r" % (found["failed"],))
    # the namespace trap, as an arm: 0x12 in MODE 2 must not become ERT_START_DPU, and 18 must not become MSG_OP 0x12
    chk("ERT_START_DPU" not in json.dumps(found["failed"]),
        "the mailbox line never carries an ERT name")
    chk(scan(["IO_PAGE_FAULT domain=0x0003 address=0x1 flags=0x7"]) == {"unsupported": {}, "failed": {}},
        "a page-fault line is not a refusal")
    if fails:
        print("driver_refusal_census self-test: %d failure(s)" % fails)
        return 1
    print("driver_refusal_census self-test: ok")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description="Name the amdxdna driver's refusals instead of leaving a bare number.")
    ap.add_argument("--journal", action="append", default=[], help="kernel log text (repeatable)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.journal:
        a.journal = ["-"]
    found = {"unsupported": {}, "failed": {}}
    for path in a.journal:
        try:
            text = sys.stdin.read() if path == "-" else open(path, errors="replace").read()
        except OSError as e:
            print("driver_refusal_census: %s: %s" % (path, e), file=sys.stderr)
            return 2
        part = scan(text.splitlines())
        for kind in found:
            for k, n in part[kind].items():
                found[kind][k] = found[kind].get(k, 0) + n
    if a.json:
        print(json.dumps(found, indent=1, sort_keys=True))
    else:
        name = ", ".join(a.journal)
        lines = report(found, name)
        for line in lines:
            print("  " + line)
        if not lines:
            print("  %s: no driver refusal lines" % name)
    return 1 if (found["unsupported"] or found["failed"]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
