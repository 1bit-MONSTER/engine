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
"""ModelScope as a fallback download source for pinned models, checked against the pin (docs/model-pins.md).

usage: tools/pin_mirror.py --check [--repo REPO...] [--pins FILE] [--out FILE] [--json FILE]
       tools/pin_mirror.py --fetch REPO FILE --dest DIR [--pins FILE]

A pin names Hugging Face bytes by their sha256 (`lfs.oid`). ModelScope (modelscope.cn) mirrors many of the same
repos under the same id, and its public file listing reports each file's sha256, so a mirror can stand in for
Hugging Face exactly when its file carries the pinned oid -- and must not otherwise, whatever its name says.
Everything here is anonymous: the public listing and download need no token, and this tool never sends one.

  --check   for every pin outside our own orgs (or --repo ...): list the same id on ModelScope and compare each
            pinned file. States: match (the mirror may serve it), DIFFERS (it serves other bytes: do not use it),
            missing (the mirror has no file at that path), no mirror (no repo under that id). A non-LFS pin
            records a git blob id, which a sha256 cannot be compared with, so a small text file is downloaded
            (<= 1 MiB) and its blob id computed.
  --fetch   download one pinned file from the mirror into DIR, hashing while it streams; the file is moved into
            place only if its hash is the pin's oid (sha256 for an LFS file, the git blob id for the rest), and
            deleted otherwise. A partial download resumes.

Exit codes: 0 every compared file matches (--check) or the file was fetched and verified (--fetch); 1 a mirror
serves different bytes for a pinned file (--check), or the fetched file is not the pinned one (--fetch); 2 usage,
an unknown repo or file, or no mirror listing could be read at all.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = os.path.join(ROOT, "models", "pins.json")
ENDPOINT = os.environ.get("MODELSCOPE_ENDPOINT", "https://modelscope.cn")
OWN = ("1bit-MONSTER", "bong-water-water-bong")  # we publish these; a mirror of them is not a source
UA = "1bit-engine-pin-mirror"
SMALL = 1 << 20
CHUNK = 16 << 20


def _open(url: str, headers: dict | None = None, timeout: int = 120):
    # no Authorization header, ever: the fallback must work for anyone, and no token leaves this box
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})}),
                                  timeout=timeout)


def listing(repo: str) -> dict[str, dict] | None:
    """{path: {"size", "sha256", "lfs"}} of the mirror's master, or None when ModelScope has no such repo."""
    q = urllib.parse.quote(repo, safe="/")
    url = f"{ENDPOINT}/api/v1/models/{q}/repo/files?Revision=master&Recursive=true"
    last = None
    for attempt in range(4):
        try:
            with _open(url) as r:
                d = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # transient: back off, then give up
            last = e
        time.sleep(1.5 * (attempt + 1))
    else:
        raise last
    if d.get("Code") != 200:
        return None
    return {f["Path"]: {"size": f.get("Size"), "sha256": f.get("Sha256") or "", "lfs": bool(f.get("IsLFS"))}
            for f in (d.get("Data") or {}).get("Files") or [] if f.get("Type") != "tree"}


def resolve_url(repo: str, path: str) -> str:
    return f"{ENDPOINT}/models/{urllib.parse.quote(repo, safe='/')}/resolve/master/{urllib.parse.quote(path)}"


def gitblob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def compare(repo: str, pin: dict, files: dict[str, dict]) -> list[dict]:
    rows = []
    for path, want in sorted(pin["files"].items()):
        f = files.get(path)
        row = {"repo": repo, "path": path, "want": want["oid"], "size": want.get("size")}
        if f is None:
            row["state"] = "missing"
        elif want.get("lfs", True):
            row["have"] = f["sha256"]
            row["state"] = ("no sha256 listed" if not f["sha256"] else
                            "match" if f["sha256"] == want["oid"] and f["size"] in (None, want.get("size")) else
                            "DIFFERS")
        elif (f["size"] or 0) > SMALL:
            row["state"] = "not compared (non-LFS file over 1 MiB)"
        else:
            with _open(resolve_url(repo, path)) as r:
                blob = r.read(SMALL + 1)
            row["have"] = gitblob(blob)
            row["state"] = "match" if row["have"] == want["oid"] else "DIFFERS"
        rows.append(row)
    return rows


def check(pins: dict, repos: list[str]) -> tuple[list[dict], list[str], list[str]]:
    rows, unmirrored, unread = [], [], []
    for repo in repos:
        try:
            files = listing(repo)
        except Exception as e:
            unread.append(f"{repo}: {e}")
            continue
        if files is None:
            unmirrored.append(repo)
            continue
        try:
            rows += compare(repo, pins[repo], files)
        except Exception as e:
            unread.append(f"{repo}: {e}")
    return rows, unmirrored, unread


def report(rows, unmirrored, unread, total) -> str:
    day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    by = {}
    for r in rows:
        by.setdefault(r["state"], []).append(r)
    mirrored = total - len(unmirrored) - len(unread)
    repos_ok = mirrored - len({r["repo"] for r in rows if r["state"] != "match"})
    lines = [f"# ModelScope mirrors of our pins, {day}", "",
             f"{total} repos asked: {mirrored} mirrored ({repos_ok} with every pinned "
             f"file matching), {len(unmirrored)} not on ModelScope under the same id, {len(unread)} not read.",
             "Files: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(by.items())), ""]
    bad = [r for r in rows if r["state"] == "DIFFERS"]
    if bad:
        lines += ["## Mirrors serving other bytes than the pin (not a fallback for these files)", ""]
        lines += [f"- {r['repo']} `{r['path']}`: pin {r['want'][:12]}…, mirror {r.get('have', '')[:12]}…" for r in bad]
        lines.append("")
    other = [r for r in rows if r["state"] not in ("match", "DIFFERS")]
    if other:
        lines += ["## Not usable as a fallback", ""]
        lines += [f"- {r['repo']} `{r['path']}`: {r['state']}" for r in other]
        lines.append("")
    if unmirrored:
        lines += ["## No ModelScope repo under the same id", ""] + [f"- {r}" for r in unmirrored] + [""]
    if unread:
        lines += ["## Could not be read", ""] + [f"- {u}" for u in unread] + [""]
    return "\n".join(lines)


def fetch(pins: dict, repo: str, path: str, dest: str) -> int:
    want = pins[repo]["files"][path]
    os.makedirs(dest, exist_ok=True)
    final = os.path.join(dest, os.path.basename(path))
    part = final + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    h = hashlib.sha256() if want.get("lfs", True) else None
    if have and h is not None:
        with open(part, "rb") as f:  # resume: the hash has to cover the bytes already on disk
            for b in iter(lambda: f.read(CHUNK), b""):
                h.update(b)
    hdr = {"Range": f"bytes={have}-"} if have else {}
    with _open(resolve_url(repo, path), hdr, timeout=300) as r:
        if have and r.status != 206:  # the server ignored the range: start over rather than append a second copy
            have, h = 0, (hashlib.sha256() if h is not None else None)
        with open(part, "ab" if have else "wb") as out:
            for b in iter(lambda: r.read(CHUNK), b""):
                out.write(b)
                if h is not None:
                    h.update(b)
    size = os.path.getsize(part)
    if h is not None:
        got = h.hexdigest()
    else:  # a git blob id needs the size first, so it is a second streamed pass, never the whole file in memory
        g = hashlib.sha1(b"blob %d\0" % size)
        with open(part, "rb") as f:
            for b in iter(lambda: f.read(CHUNK), b""):
                g.update(b)
        got = g.hexdigest()
    if got != want["oid"] or (want.get("size") is not None and size != want["size"]):
        os.remove(part)
        print(f"pin_mirror: {repo} {path} from ModelScope is {got[:12]}… {size} B, the pin is "
              f"{want['oid'][:12]}… {want.get('size')} B: deleted, not used", file=sys.stderr)
        return 1
    os.replace(part, final)
    print(f"{final}: {size} B, {got} = pin {repo} {path}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pins", default=PINS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--repo", nargs="+", metavar="REPO")
    ap.add_argument("--fetch", nargs=2, metavar=("REPO", "FILE"))
    ap.add_argument("--dest")
    ap.add_argument("--out")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        pins = json.load(open(a.pins, encoding="utf-8"))["pins"]
    except (OSError, ValueError, KeyError) as e:
        print(f"pin_mirror: cannot read {a.pins}: {e}", file=sys.stderr)
        return 2
    if a.fetch:
        repo, path = a.fetch
        if not a.dest:
            ap.error("--fetch needs --dest")
        if repo not in pins or path not in pins[repo]["files"]:
            print(f"pin_mirror: {repo} {path} is not pinned", file=sys.stderr)
            return 2
        return fetch(pins, repo, path, a.dest)
    if not a.check:
        ap.error("give --check or --fetch")
    repos = a.repo or sorted(r for r in pins if r.split("/")[0] not in OWN)
    unknown = [r for r in repos if r not in pins]
    if unknown:
        print(f"pin_mirror: not pinned: {', '.join(unknown)}", file=sys.stderr)
        return 2
    rows, unmirrored, unread = check(pins, repos)
    text = report(rows, unmirrored, unread, len(repos))
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text + "\n")
    if a.json:
        json.dump({"rows": rows, "unmirrored": unmirrored, "unread": unread}, open(a.json, "w"), indent=1)
    if unread and len(unread) == len(repos):
        return 2
    return 1 if any(r["state"] == "DIFFERS" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
