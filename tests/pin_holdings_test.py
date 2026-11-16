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
"""tools/pin_holdings.py on synthetic pins, holdings and box files (hermetic, ~0.5 s).

The decisive cases are the ones the pin watch was green through: a box copy with the pinned size but other
bytes (only a hash sees it), a cached hash that must not survive a rewrite of the file, and a re-host whose pin
records a different file than its upstream's (Qwen3.8-27B-UD-Q4_K_XL, 2026-11-12..16).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "pin_holdings.py")

fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails += 1


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def gitblob(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def run(tmp, *extra, holdings="holdings.json"):
    r = subprocess.run([sys.executable, TOOL, "--pins", os.path.join(tmp, "pins.json"),
                        "--holdings", os.path.join(tmp, holdings), "--cache", os.path.join(tmp, "cache.json"),
                        "--json", os.path.join(tmp, "out.json"), *extra], capture_output=True, text=True)
    out = json.load(open(os.path.join(tmp, "out.json"))) if os.path.exists(os.path.join(tmp, "out.json")) else {}
    if os.path.exists(os.path.join(tmp, "out.json")):
        os.remove(os.path.join(tmp, "out.json"))
    return r, out


def state(out, path):
    return next((h["state"] for h in out.get("holdings") or [] if h["path"] == path), None)


with tempfile.TemporaryDirectory() as tmp:
    good, text = b"G" * 4096, b'{"chat_template": "x"}\n'
    old27, new27 = b"o" * 1000, b"n" * 1100
    pins = {"pins": {
        "up/Model-GGUF": {"commit": "a", "pinned": "d", "used_for": ["upstream of us/Model-Rehost"],
                          "files": {"m.gguf": {"oid": sha256(new27), "size": len(new27), "lfs": True},
                                    "sub/head.gguf": {"oid": sha256(good), "size": len(good), "lfs": True},
                                    "renamed.gguf": {"oid": sha256(b"r"), "size": 1, "lfs": True}}},
        "us/Model-Rehost": {"commit": "b", "pinned": "d", "used_for": ["published: re-host"],
                            "files": {"m.gguf": {"oid": sha256(old27), "size": len(old27), "lfs": True},
                                      "head.gguf": {"oid": sha256(good), "size": len(good), "lfs": True},
                                      "other-name.gguf": {"oid": sha256(b"z"), "size": 1, "lfs": True}}},
        "x/Text": {"commit": "c", "pinned": "d", "used_for": [],
                   "files": {"tokenizer_config.json": {"oid": gitblob(text), "size": len(text), "lfs": False}}},
    }}
    json.dump(pins, open(os.path.join(tmp, "pins.json"), "w"))
    box = os.path.join(tmp, "box")
    os.makedirs(box)
    paths = {}
    for name, data in {"head.gguf": good, "tok.json": text, "m-old.gguf": old27, "m-new.gguf": new27,
                       "same-size.gguf": b"X" * len(good)}.items():
        paths[name] = os.path.join(box, name)
        open(paths[name], "wb").write(data)

    def holdings(rows):
        json.dump({"holdings": rows}, open(os.path.join(tmp, "holdings.json"), "w"))

    base = [{"pin": "up/Model-GGUF", "file": "sub/head.gguf", "path": paths["head.gguf"]},
            {"pin": "x/Text", "file": "tokenizer_config.json", "path": paths["tok.json"]},
            {"pin": "up/Model-GGUF", "file": "m.gguf", "path": paths["m-new.gguf"]}]

    # 1. every holding matches (LFS sha256 and a non-LFS git blob id); the re-host drift still fails the run.
    holdings(base)
    r, out = run(tmp)
    check([h["state"] for h in out["holdings"]] == ["ok", "ok", "ok"], f"matching copies are ok ({out.get('holdings')})")
    check(r.returncode == 1 and len(out["rehosts"]) == 1, f"the re-host behind its upstream fails the run (rc {r.returncode})")
    rh = out["rehosts"][0] if out.get("rehosts") else {}
    check(rh.get("file") == "m.gguf" and rh.get("rehost_oid") == sha256(old27),
          "the drifted re-host row names the file and the oid the re-host holds")
    check(not any(x.get("file") in ("sub/head.gguf", "renamed.gguf") for x in out.get("rehosts", [])),
          "a same-named file with the same oid (in another folder) is not drift; a renamed one is not compared")
    check("needs a re-upload" in r.stdout, "the report says the re-host needs a re-upload")

    # 1b. a clean-looking run says how much it did not look at: 3 of the 7 pinned files have a holding.
    check(out.get("pinned_files") == 7 and out.get("unlisted_pins") == 4,
          f"coverage counts the pinned files with no holding ({out.get('pinned_files')}, {out.get('unlisted_pins')})")
    check("3 of 7 pinned files checked; 4 not checked" in r.stdout and "unknown without --scan" in r.stdout,
          "the header says how many were not checked, and that absent vs unmapped is unknown without --scan")

    # 1c. --scan tells ABSENT (not on the box) from UNMAPPED (on the box, missing from the map): a box file with a
    # pinned name and size that no holding lists is the blindness the tool exists for, and fails the run.
    scan = os.path.join(tmp, "scan")
    os.makedirs(os.path.join(scan, "deep"))
    open(os.path.join(scan, "deep", "renamed.gguf"), "wb").write(b"r")          # pin up renamed.gguf: size 1, unmapped
    open(os.path.join(scan, "deep", "other-name.gguf"), "wb").write(b"zz")      # pinned name, wrong size: not a match
    open(os.path.join(scan, "head.gguf"), "wb").write(good)                     # us/Model-Rehost head.gguf: unmapped
    r2, out2 = run(tmp, "--scan", scan)
    um = sorted((u["pin"], u["file"]) for u in out2.get("unmapped_files", []))
    check(um == [("up/Model-GGUF", "renamed.gguf"), ("us/Model-Rehost", "head.gguf")] and out2.get("absent") == 2,
          f"--scan finds the unmapped copies by name and size, not by name alone ({um}, absent {out2.get('absent')})")
    check("UNMAPPED:" in r2.stdout and "2 held and missing from models/holdings.json" in r2.stdout,
          "the report names each unmapped file")


    # 2. the re-host fixed: everything clean, rc 0.
    pins["pins"]["us/Model-Rehost"]["files"]["m.gguf"] = dict(pins["pins"]["up/Model-GGUF"]["files"]["m.gguf"])
    json.dump(pins, open(os.path.join(tmp, "pins.json"), "w"))
    r, out = run(tmp)
    check(r.returncode == 0 and out["rehosts"] == [], f"all copies and re-hosts match -> rc 0 (rc {r.returncode})")
    check("0 hashed" in r.stdout, "the second run takes every hash from the cache")
    r2, _ = run(tmp, "--scan", scan)
    check(r2.returncode == 1, f"on an otherwise clean run, an unmapped copy alone fails it (rc {r2.returncode})")

    # 3. the box holds the OLD file under the pinned name: a size mismatch, reported without reading it.
    holdings(base[:2] + [{"pin": "up/Model-GGUF", "file": "m.gguf", "path": paths["m-old.gguf"]}])
    r, out = run(tmp)
    check(r.returncode == 1 and state(out, paths["m-old.gguf"]) == "size differs",
          f"the old file at the pinned path is drift (rc {r.returncode}, {state(out, paths['m-old.gguf'])})")
    check("have" not in out["holdings"][2], "a size mismatch is decided without hashing the file")

    # 4. same size, other bytes: only the hash can see it.
    holdings([{"pin": "up/Model-GGUF", "file": "sub/head.gguf", "path": paths["same-size.gguf"]}])
    r, out = run(tmp)
    check(r.returncode == 1 and state(out, paths["same-size.gguf"]) == "oid differs",
          f"same size, different bytes -> oid differs (rc {r.returncode})")

    # 5. a cached hash does not outlive a rewrite of the file (same size, new mtime).
    holdings(base[:1])
    r, out = run(tmp)
    check(r.returncode == 0, "the head copy is ok before the rewrite")
    time.sleep(0.01)
    open(paths["head.gguf"], "wb").write(b"H" * len(good))
    r, out = run(tmp)
    check(r.returncode == 1 and state(out, paths["head.gguf"]) == "oid differs",
          f"rewritten in place at the same size -> re-hashed and caught (rc {r.returncode})")

    # 6. a missing copy is drift, not a skip.
    holdings([{"pin": "x/Text", "file": "tokenizer_config.json", "path": os.path.join(box, "gone.json")}])
    r, out = run(tmp)
    check(r.returncode == 1 and out["holdings"][0]["state"] == "missing", f"a missing copy fails (rc {r.returncode})")

    # 7. a map entry pins.json cannot resolve is rc 2, never clean.
    for bad in ({"pin": "no/Such", "file": "m.gguf", "path": paths["m-new.gguf"]},
                {"pin": "up/Model-GGUF", "file": "nope.gguf", "path": paths["m-new.gguf"]}):
        holdings([bad])
        r, out = run(tmp)
        check(r.returncode == 2 and out.get("errors"), f"unresolvable holding {bad['pin']}:{bad['file']} -> rc 2 "
                                                     f"(rc {r.returncode})")

    # 7b. --log appends one line per run with the outcome, so a failing timer is visible after the journal rotates.
    lg = os.path.join(tmp, "PIN-HOLDINGS.log")
    holdings(base[1:])  # head.gguf was rewritten in 5; the other two still match
    run(tmp, "--log", lg)
    holdings([{"pin": "no/Such", "file": "m.gguf", "path": paths["m-new.gguf"]}])
    run(tmp, "--log", lg)
    rows = open(lg).read().splitlines()
    check(len(rows) == 2 and " ok (2/7 checked) mapped=2 checked=2 drifted=0 rehost_drifted=0 map_errors=0 absent=- unmapped=-" in rows[0]
          and " error " in rows[1], f"--log: one line per run, outcome first ({rows})")

    # 8. --rehost-only needs no holdings file and reads no box file.
    r, out = run(tmp, "--rehost-only", holdings="absent.json")
    check(r.returncode == 0 and out["holdings"] is None, f"--rehost-only runs without a holdings map (rc {r.returncode})")
    pins["pins"]["up/Model-GGUF"]["used_for"].append("upstream of us/Not-Pinned")
    json.dump(pins, open(os.path.join(tmp, "pins.json"), "w"))
    r, out = run(tmp, "--rehost-only", holdings="absent.json")
    check(r.returncode == 1 and any(x["state"] == "re-host not pinned" for x in out["rehosts"]),
          "an 'upstream of' naming an unpinned re-host is reported")

print("pin_holdings_test: " + ("PASS" if fails == 0 else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
