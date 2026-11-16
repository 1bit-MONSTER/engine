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
"""Do the files we hold match the pins? The box half of the pin watch (docs/model-pins.md).

usage: tools/pin_holdings.py [--pins FILE] [--holdings FILE] [--cache FILE] [--out FILE] [--json FILE]
                             [--rehost-only]

tools/pin_watch.py --check compares each pin with what Hugging Face serves. It cannot see the box, so a pin
can be bumped to a new upstream file while the box keeps serving the old one, and every daily run stays
green. That happened: unsloth/Qwen3.8-27B-GGUF was pinned at the UD3.1 Qwen3.8-27B-UD-Q4_K_XL.gguf on
2026-11-12 (oid 2a5f3565..., 17,659,407,264 B) while ~/models/qwen38/UD-Q4_K_XL/ and our re-host
1bit-MONSTER/Qwen3.8-27B-UD-Q4_K_XL-MTP still held the earlier file (3f227079..., 17,559,178,144 B).

Two checks, one report:

  holdings  each entry of models/holdings.json names a pin, a file of that pin, and the path of our copy on
            the box. The copy's sha256 (the git blob id for a non-LFS file, which is what the pin records
            for those) is compared with the pin's oid. Hashes are cached by path, size, mtime and inode
            (--cache), so a daily run re-reads only a file that changed; a size mismatch is reported
            without reading the file at all.
  re-hosts  a pin whose used_for says "upstream of <repo>" is compared, file by file of the same name,
            with <repo>'s own pin. Both oids come from models/pins.json, so this half needs neither the
            box nor the network, and --rehost-only runs it alone (CI can).

Exit codes: 0 every holding and every re-host matches, 1 a drift (a holding differs from its pin or is
missing, or a re-host holds a different file than its upstream pin), 2 usage, or a holding names a pin or a
file pins.json does not have (the map is wrong, which must not read as clean).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = os.path.join(ROOT, "models", "pins.json")
HOLDINGS = os.path.join(ROOT, "models", "holdings.json")
CACHE = os.path.expanduser("~/.cache/1bit/pin-holdings-sha256.json")
CHUNK = 16 << 20


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_id(path: str, lfs: bool) -> str:
    """sha256 of the bytes (an LFS oid), or the git blob id (sha1 of "blob <size>\\0" + bytes)."""
    h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % os.path.getsize(path))
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Cache:
    """Hash cache keyed by real path; an entry counts only while size, mtime and inode are what was hashed."""

    def __init__(self, path: str | None):
        self.path = path
        self.rows = {}
        if path and os.path.exists(path):
            try:
                self.rows = load(path)
            except (OSError, ValueError):
                self.rows = {}  # an unreadable cache costs a re-hash, never a wrong answer
        self.hashed = 0

    def get(self, path: str, lfs: bool) -> str:
        real = os.path.realpath(path)
        st = os.stat(real)
        key = f"{real}|{'sha256' if lfs else 'gitblob'}"
        row = self.rows.get(key)
        stamp = [st.st_size, st.st_mtime_ns, st.st_ino]
        if row and row.get("stat") == stamp:
            return row["id"]
        got = file_id(real, lfs)
        self.hashed += 1
        self.rows[key] = {"stat": stamp, "id": got}
        return got

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)


def check_holdings(pins: dict, holdings: list[dict], cache: Cache) -> tuple[list[dict], list[str]]:
    """(rows, errors): one row per holding with its state; errors are map entries pins.json cannot resolve."""
    rows, errors = [], []
    for h in holdings:
        repo, name = h.get("pin"), h.get("file")
        path = os.path.expanduser(h.get("path") or "")
        pin = pins.get(repo)
        if pin is None:
            errors.append(f"holding {path}: pin {repo!r} is not in pins.json")
            continue
        want = pin["files"].get(name)
        if want is None:
            errors.append(f"holding {path}: pin {repo} has no file {name!r}")
            continue
        row = {"pin": repo, "file": name, "path": path, "want": want["oid"], "want_size": want.get("size")}
        if not os.path.exists(path):
            row["state"] = "missing"
        else:
            size = os.path.getsize(os.path.realpath(path))
            row["size"] = size
            if want.get("size") is not None and size != want["size"]:
                row["state"] = "size differs"  # no read needed: different bytes for certain
            else:
                row["have"] = cache.get(path, bool(want.get("lfs", True)))
                row["state"] = "ok" if row["have"] == want["oid"] else "oid differs"
        rows.append(row)
    return rows, errors


def check_rehosts(pins: dict) -> list[dict]:
    """Every same-named file where a re-host's pin and its upstream pin record different bytes."""
    out = []
    for up, pin in sorted(pins.items()):
        for use in pin.get("used_for", []):
            if not use.startswith("upstream of "):
                continue
            rehost = use[len("upstream of "):].strip()
            mine = pins.get(rehost)
            if mine is None:
                out.append({"upstream": up, "rehost": rehost, "file": None, "state": "re-host not pinned"})
                continue
            theirs = {os.path.basename(k): (k, v) for k, v in mine["files"].items()}
            for path, want in sorted(pin["files"].items()):
                hit = theirs.get(os.path.basename(path))
                if hit is None:
                    continue  # a renamed or different quant in the re-host is not the same file to compare
                if hit[1]["oid"] != want["oid"]:
                    out.append({"upstream": up, "rehost": rehost, "file": path, "rehost_file": hit[0],
                                "state": "re-host holds a different file",
                                "upstream_oid": want["oid"], "upstream_size": want.get("size"),
                                "rehost_oid": hit[1]["oid"], "rehost_size": hit[1].get("size")})
    return out


def gb(n) -> str:
    return "?" if n is None else f"{n / 1e9:.2f} GB"


def report(hrows: list[dict] | None, rrows: list[dict], errors: list[str], hashed: int) -> str:
    day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()  # UTC, as the CI run dates its report
    bad = [r for r in hrows or [] if r["state"] != "ok"]
    lines = [f"# Pin holdings, {day}", ""]
    if hrows is None:
        lines.append("Holdings: not checked (--rehost-only).")
    else:
        lines.append(f"Holdings: {len(hrows)} checked, {len(bad)} differ from their pin "
                     f"({hashed} hashed this run, the rest from the cache).")
    lines.append(f"Re-hosts: {len(rrows)} differ from their upstream pin.")
    if errors:
        lines.append(f"Map errors: {len(errors)} (exit 2).")
    lines.append("")
    if bad:
        lines += ["## Box copies that are not the pinned file", ""]
        for r in bad:
            have = r.get("have", "")[:12] or "-"
            lines.append(f"- `{r['path']}` ({r['state']}): pin {r['pin']} `{r['file']}` is {r['want'][:12]}… "
                         f"{gb(r['want_size'])}; the box holds {have} {gb(r.get('size'))}")
        lines.append("")
    if rrows:
        lines += ["## Re-hosts behind their upstream pin", ""]
        for r in rrows:
            if r["file"] is None:
                lines.append(f"- {r['rehost']} (upstream {r['upstream']}): the re-host has no pin")
                continue
            lines.append(f"- {r['rehost']} `{r['rehost_file']}` is {r['rehost_oid'][:12]}… {gb(r['rehost_size'])}; "
                         f"upstream {r['upstream']} `{r['file']}` is pinned at {r['upstream_oid'][:12]}… "
                         f"{gb(r['upstream_size'])} — the re-host needs a re-upload (owner-only)")
        lines.append("")
    if errors:
        lines += ["## Holdings map errors", ""] + [f"- {e}" for e in errors] + [""]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pins", default=PINS)
    ap.add_argument("--holdings", default=HOLDINGS)
    ap.add_argument("--cache", default=CACHE, help="hash cache ('' disables it)")
    ap.add_argument("--out", help="also write the Markdown report here")
    ap.add_argument("--json", help="also write the rows as JSON here")
    ap.add_argument("--rehost-only", action="store_true", help="only the pins.json half (no box files read)")
    a = ap.parse_args(argv)
    try:
        pins = load(a.pins)["pins"]
    except (OSError, ValueError, KeyError) as e:
        print(f"pin_holdings: cannot read {a.pins}: {e}", file=sys.stderr)
        return 2
    cache = Cache(a.cache or None)
    hrows, errors = None, []
    if not a.rehost_only:
        try:
            holdings = load(a.holdings)["holdings"]
        except (OSError, ValueError, KeyError) as e:
            print(f"pin_holdings: cannot read {a.holdings}: {e}", file=sys.stderr)
            return 2
        hrows, errors = check_holdings(pins, holdings, cache)
        cache.save()
    rrows = check_rehosts(pins)
    text = report(hrows, rrows, errors, cache.hashed)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"holdings": hrows, "rehosts": rrows, "errors": errors}, f, indent=1)
    if errors:
        return 2
    if rrows or any(r["state"] != "ok" for r in hrows or []):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
