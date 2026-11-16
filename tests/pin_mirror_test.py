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
"""tools/pin_mirror.py against a fake ModelScope (no network, ~0.1 s).

The decisive cases: a mirror file under the pinned name with other bytes must be refused as a source, a fetched file
that is not the pinned one must never be moved into place, a non-LFS pin (a git blob id) is compared by its blob id
and not waved through on a sha256, and no request carries a token.
"""
import hashlib
import io
import json
import os
import sys
import tempfile
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import pin_mirror as M  # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails += 1


def sha(b):
    return hashlib.sha256(b).hexdigest()


GOOD, OTHER, TEXT = b"W" * 5000, b"V" * 5000, b'{"chat_template": "t"}\n'
SERVED = {}      # (repo, path) -> bytes the fake mirror serves
LISTED = {}      # repo -> listing rows
SEEN = []        # every request's headers


class Resp(io.BytesIO):
    def __init__(self, b, status=200):
        super().__init__(b)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def fake_urlopen(req, timeout=0):
    # patched under the tool's own _open, so SEEN is what would really go on the wire
    url, headers = req.full_url, {k.lower(): v for k, v in req.header_items()}
    SEEN.append(headers)
    if "/api/v1/models/" in url:
        repo = url.split("/api/v1/models/")[1].split("/repo/files")[0]
        if repo not in LISTED:
            raise urllib.error.HTTPError(url, 404, "nf", {}, None)
        return Resp(json.dumps({"Code": 200, "Data": {"Files": LISTED[repo]}}).encode())
    repo, path = url.split("/models/")[1].split("/resolve/master/")
    b = SERVED[(repo, path)]
    rng = headers.get("range")
    if rng:
        start = int(rng.split("=")[1].rstrip("-"))
        return Resp(b[start:], 206)
    return Resp(b)


M.urllib.request.urlopen = fake_urlopen


def files(repo, rows):
    LISTED[repo] = [{"Path": p, "Size": len(b), "Sha256": sha(b), "IsLFS": lfs, "Type": "blob"}
                    for p, b, lfs in rows]
    for p, b, _ in rows:
        SERVED[(repo, p)] = b


with tempfile.TemporaryDirectory() as tmp:
    pins = {"pins": {
        "up/Good-GGUF": {"files": {"m.gguf": {"oid": sha(GOOD), "size": len(GOOD), "lfs": True},
                                   "tokenizer_config.json": {"oid": M.gitblob(TEXT), "size": len(TEXT), "lfs": False}}},
        "up/Moved-GGUF": {"files": {"m.gguf": {"oid": sha(GOOD), "size": len(GOOD), "lfs": True}}},
        "up/Partial-GGUF": {"files": {"a.gguf": {"oid": sha(GOOD), "size": len(GOOD), "lfs": True},
                                      "b.gguf": {"oid": sha(GOOD), "size": len(GOOD), "lfs": True}}},
        "up/Absent-GGUF": {"files": {"m.gguf": {"oid": sha(GOOD), "size": len(GOOD), "lfs": True}}},
        "1bit-MONSTER/Ours": {"files": {"m.gguf": {"oid": sha(GOOD), "size": len(GOOD), "lfs": True}}},
    }}
    pf = os.path.join(tmp, "pins.json")
    json.dump(pins, open(pf, "w"))
    files("up/Good-GGUF", [("m.gguf", GOOD, True), ("tokenizer_config.json", TEXT, False)])
    files("up/Moved-GGUF", [("m.gguf", OTHER, True)])
    files("up/Partial-GGUF", [("a.gguf", GOOD, True)])
    out = os.path.join(tmp, "r.json")

    # 1. one mirror serves other bytes -> rc 1, and that file is named.
    rc = M.main(["--pins", pf, "--check", "--json", out])
    r = json.load(open(out))
    st = {(x["repo"], x["path"]): x["state"] for x in r["rows"]}
    check(rc == 1, f"a mirror with other bytes under the pinned name fails the check (rc {rc})")
    check(st.get(("up/Moved-GGUF", "m.gguf")) == "DIFFERS", "the differing file is reported DIFFERS")
    check(st.get(("up/Good-GGUF", "m.gguf")) == "match", "a matching LFS file is a match")
    check(st.get(("up/Good-GGUF", "tokenizer_config.json")) == "match",
          "a non-LFS pin is compared by its git blob id (downloaded), and matches")
    check(st.get(("up/Partial-GGUF", "b.gguf")) == "missing", "a pinned file the mirror lacks is missing")
    check(r["unmirrored"] == ["up/Absent-GGUF"], f"a repo with no mirror is listed, not failed ({r['unmirrored']})")
    check(not any(x["repo"].startswith("1bit-MONSTER/") for x in r["rows"]) and "1bit-MONSTER/Ours" not in r["unmirrored"],
          "our own orgs are not asked")

    # 2. the non-LFS check really is the blob id: a sha256 of the text equal to nothing in the pin must not pass.
    SERVED[("up/Good-GGUF", "tokenizer_config.json")] = TEXT + b" "
    rc = M.main(["--pins", pf, "--check", "--repo", "up/Good-GGUF", "--json", out])
    st = {x["path"]: x["state"] for x in json.load(open(out))["rows"]}
    check(rc == 1 and st["tokenizer_config.json"] == "DIFFERS", f"a changed non-LFS file is DIFFERS (rc {rc})")
    SERVED[("up/Good-GGUF", "tokenizer_config.json")] = TEXT

    # 3. --repo limited to clean repos -> rc 0.
    rc = M.main(["--pins", pf, "--check", "--repo", "up/Good-GGUF", "up/Absent-GGUF"])
    check(rc == 0, f"every compared file matches -> rc 0 (rc {rc})")

    # 4. fetch: the pinned bytes land in place; other bytes are deleted, never moved into place.
    dest = os.path.join(tmp, "dl")
    rc = M.main(["--pins", pf, "--fetch", "up/Good-GGUF", "m.gguf", "--dest", dest])
    check(rc == 0 and open(os.path.join(dest, "m.gguf"), "rb").read() == GOOD, f"a verified fetch lands (rc {rc})")
    rc = M.main(["--pins", pf, "--fetch", "up/Moved-GGUF", "m.gguf", "--dest", os.path.join(tmp, "dl2")])
    left = os.listdir(os.path.join(tmp, "dl2"))
    check(rc == 1 and left == [], f"a fetch of other bytes is refused and leaves nothing (rc {rc}, left {left})")

    # 5. a partial download resumes and the hash covers the bytes already on disk.
    d3 = os.path.join(tmp, "dl3")
    os.makedirs(d3)
    open(os.path.join(d3, "m.gguf.part"), "wb").write(GOOD[:1234])
    rc = M.main(["--pins", pf, "--fetch", "up/Good-GGUF", "m.gguf", "--dest", d3])
    check(rc == 0 and open(os.path.join(d3, "m.gguf"), "rb").read() == GOOD, f"a resumed fetch verifies (rc {rc})")
    check(any(h.get("range") == "bytes=1234-" for h in SEEN), "the resume asked for the remaining range only")
    open(os.path.join(d3, "m.gguf.part"), "wb").write(OTHER[:1234])  # a corrupt partial must not pass as resumed
    os.remove(os.path.join(d3, "m.gguf"))
    rc = M.main(["--pins", pf, "--fetch", "up/Good-GGUF", "m.gguf", "--dest", d3])
    check(rc == 1 and not os.path.exists(os.path.join(d3, "m.gguf")), f"a corrupt partial is caught by the hash (rc {rc})")

    # 6. unknown inputs are rc 2; no request ever carries a token.
    check(M.main(["--pins", pf, "--fetch", "up/Good-GGUF", "nope.gguf", "--dest", dest]) == 2, "an unpinned file is rc 2")
    check(M.main(["--pins", pf, "--check", "--repo", "no/Such"]) == 2, "an unpinned repo is rc 2")
    check(SEEN and not any("authorization" in h for h in SEEN) and all(h.get("user-agent") == M.UA for h in SEEN),
          f"no request carries a token ({len(SEEN)} requests)")

print("pin_mirror_test: " + ("PASS" if fails == 0 else f"{fails} FAILED"))
sys.exit(1 if fails else 0)
