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
"""Classify the producer probe's output per the pre-registered table.

Why this exists: dev/315/315-campaign/evidence/315-producer-probe-preregistration.md pre-declares six verdict
branches for the `[hrx-producer]` lines, of which the decisive pair separates "the device never wrote the
region" from "the device wrote a wrong value". Scoring that by eye invites the exact post-hoc reading the
pre-registration exists to prevent, so the table is implemented here and the verdict is printed, not
chosen. This script decides nothing the pre-registration did not already fix.

The payload constants are the whole point: 0x7FC0DEAD is the probe's OWN payload (so finding it means the
region was never overwritten), 0x7FC00000 is an ordinary quiet NaN from arithmetic.

Usage: producer-probe-readout.py <arm-dir-or-glob> [...]
"""
from __future__ import annotations

import glob
import os
import re
import sys

SENTINEL_PAYLOAD = 0x7FC0DEAD          # the probe's own fill; still present => never written
QUIET_NAN = 0x7FC00000                 # ordinary arithmetic NaN

ARMED = re.compile(r"\[hrx-producer\] armed op=(?P<op>\S+) name=(?P<name>\S+) bytes=(?P<bytes>\d+) words=(?P<words>\d+)")
ANOM = re.compile(
    r"\[hrx-producer\] anomaly op=(?P<op>\S+) name=(?P<name>\S+) words=(?P<words>\d+) "
    r"sentinel=(?P<sentinel>\d+) nan=(?P<nan>\d+) finite=(?P<finite>\d+) first_nan=0x(?P<first>[0-9a-fA-F]+)")


def classify(armed_present: bool, a: dict | None) -> str:
    """The pre-registered table (as amended), in order. Function so it can be unit-tested.

    Read at source, `sentinel + nan + finite == words` exactly, and the probe's own payload is counted in
    `sentinel` ALONE - it is the first branch of the classification if/else, so a surviving payload word
    never increments `nan` and never becomes `first_nan`. `first_nan` is therefore always the first
    genuine, non-sentinel NaN. The earlier version of this table assumed otherwise and was impossible to
    satisfy; see the amendment in the pre-registration.
    """
    if not armed_present:
        return "SELECTOR_ARMED_NOTHING (INVALID for the question - widen the size band, do not read as clean)"
    if a is None:
        return "WRITTEN_CLEAN (armed and no anomaly line: the hooked graph wrote the region without NaN)"
    sent, nan, words = int(a["sentinel"]), int(a["nan"]), int(a["words"])
    first = int(a["first"], 16)
    if sent and nan:
        return (f"PARTIAL_UNWRITTEN - {sent} of {words} words still hold the probe payload AND {nan} are "
                "genuine NaNs: the region was partly written, so size the gap against `words`")
    if sent and not nan:
        return (f"NEVER_WRITTEN - all {sent} of {words} payload words survived and there is no genuine NaN: "
                "the device did not write this region. Consumer reads pre-fill content => visibility/"
                "staleness defect, not compute")
    if nan and first == QUIET_NAN:
        return ("WRITTEN_QUIET_NAN - no payload survived and the first genuine NaN is an ordinary quiet NaN "
                "from arithmetic => the never-written/restored-copy family is excluded")
    if nan:
        return f"WRITTEN_OTHER_PAYLOAD - written, first genuine NaN payload 0x{first:08x} (identify producer)"
    return "WRITTEN_FINITE - no payload survived and no NaN: the region was written cleanly"


def scan(paths: list[str]) -> int:
    dirs = sorted({os.path.dirname(p) for p in paths
                   for d in [p] for p in (glob.glob(d) if not os.path.isdir(d) else [d])})
    if not dirs:
        print("no arm directories matched")
        return 2
    for d in dirs:
        log = os.path.join(d, "server.log")
        if not os.path.exists(log):
            continue
        armed, anomalies = [], []
        with open(log, errors="replace") as fh:
            for line in fh:
                if "hrx-producer" not in line:
                    continue
                m = ARMED.search(line)
                if m:
                    armed.append(m.groupdict())
                m = ANOM.search(line)
                if m:
                    anomalies.append(m.groupdict())
        arm = os.path.basename(d.rstrip("/"))
        print(f"=== {arm}")
        print(f"    armed_lines={len(armed)}  anomaly_lines={len(anomalies)}")
        for a in armed[:4]:
            print(f"    ARMED   {a['name']} op={a['op']} bytes={a['bytes']} words={a['words']}")
        if not anomalies:
            print(f"    VERDICT {classify(bool(armed), None)}")
        for a in anomalies[:6]:
            tot = int(a["sentinel"]) + int(a["nan"]) + int(a["finite"])
            flag = "" if tot == int(a["words"]) else f"  !! partition {tot} != words {a['words']}"
            print(f"    ANOMALY {a['name']} words={a['words']} sentinel={a['sentinel']} nan={a['nan']} "
                  f"finite={a['finite']} first_nan=0x{int(a['first'], 16):08x}{flag}")
            print(f"    VERDICT {classify(True, a)}")
    return 0


def _selftest() -> int:
    cases = [
        (False, None, "SELECTOR_ARMED_NOTHING"),
        (True, None, "WRITTEN_CLEAN"),
        # entire region unwritten: payload words only, no genuine NaN (first_nan unset => 0)
        (True, {"first": "00000000", "sentinel": "154880", "nan": "0", "finite": "0", "words": "154880"}, "NEVER_WRITTEN"),
        # unwritten tail plus genuine NaNs
        (True, {"first": "7fc00000", "sentinel": "17", "nan": "3", "finite": "154860", "words": "154880"}, "PARTIAL_UNWRITTEN"),
        (True, {"first": "7fc00000", "sentinel": "0", "nan": "2048", "finite": "152832", "words": "154880"}, "WRITTEN_QUIET_NAN"),
        (True, {"first": "ffc00000", "sentinel": "0", "nan": "2048", "finite": "152832", "words": "154880"}, "WRITTEN_OTHER_PAYLOAD"),
        (True, {"first": "00000000", "sentinel": "0", "nan": "0", "finite": "154880", "words": "154880"}, "WRITTEN_FINITE"),
    ]
    bad = 0
    for armed, a, want in cases:
        got = classify(armed, a)
        if not got.startswith(want):
            print(f"SELFTEST FAIL: armed={armed} a={a} want={want} got={got}")
            bad += 1
    # The identity the source guarantees: sentinel + nan + finite == words. Assert it on every case so a
    # future edit that breaks the reading is caught here rather than in a result.
    for _, a, _ in cases:
        if a is None:
            continue
        tot = int(a["sentinel"]) + int(a["nan"]) + int(a.get("finite", "0"))
        if tot != int(a["words"]):
            print(f"SELFTEST FAIL: partition {tot} != words {a['words']} for {a}")
            bad += 1
    print(f"selftest: {len(cases) - bad}/{len(cases)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "--selftest":
        sys.exit(_selftest())
    sys.exit(scan(argv))
