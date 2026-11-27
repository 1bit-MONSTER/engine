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
"""loom_card_meta.py -- a metadata-only tree of every public Loom card of some Hugging Face orgs (engine#3156).

tools/ci/loom_coverage_check.py reads a card tree (one directory per card holding model.json and model.json.lock), and
the real cards are GB downloads, so CI could only answer "cannot tell". The coverage check needs exactly two small files
per card: model.json names the checkpoint files, and model.json.lock binds their sha256. This writes those two files
(and hf.json: repo id + revision) for every card of the given orgs, each at the revision the listing reports.

    loom_card_meta.py --org loom-fun --org loom-draft --out cards/      then
    tools/ci/loom_coverage_check.py --cards cards/loom-fun --cards cards/loom-draft --advisory

Nothing here runs card code: it reads JSON over HTTPS. Exit 0 on success (a card without either file is skipped and
counted), 1 when the listing of an org or any card fetch failed (the map would be incomplete). Only PUBLIC cards are
seen: the listing is anonymous, so a model covered by a private card reads UNCOVERED here.
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/models"
FILES = ("model.json", "model.json.lock")
MAX_BYTES = 4 << 20          # a model.json or lock is a few kB; anything this large is not one
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
SHA = re.compile(r"^[0-9a-f]{40}$")


class Missing(Exception):
    """HTTP 404: the file does not exist at that revision (a card without a lock), as opposed to a failed fetch."""


def http_get(url, limit=MAX_BYTES, want_next=False):
    """The body (or (body, next page url) with want_next). Raises Missing on 404, other errors as they come."""
    req = urllib.request.Request(url, headers={"User-Agent": "loom-card-meta"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(limit + 1)
            link = r.headers.get("Link", "") if want_next else ""
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise Missing(url) from e
        raise
    if len(data) > limit:
        raise ValueError(f"{url}: larger than {limit} bytes")
    if not want_next:
        return data
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return data, (m.group(1) if m else None)


def list_cards(org, get=http_get):
    """[(repo id, revision)] of the org's public models tagged `loom` (the card layout), every page of the listing API
    (it pages through a Link: rel="next" header; one page alone would silently drop cards past the first 1000)."""
    url = f"{API}?author={urllib.parse.quote(org)}&filter=loom&limit=1000&full=true"   # without `full` the listing has no sha
    rows = []
    for _ in range(100):
        data, url = get(url, limit=64 << 20, want_next=True)
        rows += json.loads(data)
        if not url:
            break
    else:
        raise RuntimeError(f"{org}: more than 100 listing pages")
    out = []
    for m in rows:
        rid, sha = m.get("id", ""), m.get("sha", "")
        owner, _, name = rid.partition("/")
        if owner != org or not NAME.match(name) or not SHA.match(sha or ""):
            continue   # never a path we did not validate
        out.append((rid, sha))
    return out


def fetch_card(rid, sha, root, get=http_get):
    """"ok" (root/<name>/{model.json, model.json.lock, hf.json} written), "missing" (a file is 404 at that revision: not
    a card the map can read) or "failed" (any other error). A card directory appears only complete: both files are read
    first and the directory is renamed into place, so a failed fetch never leaves a half card the map would misread
    (a model.json without its lock reads NO-LOCK, a finding, when it was a transient error)."""
    name = rid.split("/", 1)[1]
    got = {}
    for f in FILES:
        try:
            data = get(f"https://huggingface.co/{rid}/resolve/{sha}/{f}")
            json.loads(data)   # both are JSON; anything else is not a card file
        except Missing:
            return "missing"
        except Exception:
            return "failed"
        got[f] = data
    d, part = root / name, root / (name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    for f, data in got.items():
        (part / f).write_bytes(data)
    (part / "hf.json").write_text(json.dumps({"id": rid, "revision": sha}) + "\n")
    shutil.rmtree(d, ignore_errors=True)
    part.rename(d)
    return "ok"


def fetch_org(org, out, get=http_get, workers=6):
    """{"listed", "ok", "missing", "failed"} for one org."""
    root = pathlib.Path(out) / org
    root.mkdir(parents=True, exist_ok=True)
    cards = list_cards(org, get)
    with cf.ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: fetch_card(c[0], c[1], root, get), cards))
    return {"listed": len(cards), "ok": res.count("ok"), "missing": res.count("missing"), "failed": res.count("failed")}


def self_test():
    """Hermetic: a fake HTTP layer with two listing pages, a good card, one without a lock (404), one whose lock fetch
    fails (500), and names that must never become paths."""
    import tempfile
    fails = 0

    def check(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        fails += 0 if cond else 1

    good, nolock, flaky = "a" * 40, "b" * 40, "c" * 40
    page1 = [{"id": "org/good-loom", "sha": good}, {"id": "org/nolock-loom", "sha": nolock},
             {"id": "org/../escape", "sha": good}, {"id": "other/foreign-loom", "sha": good}]
    page2 = [{"id": "org/flaky-loom", "sha": flaky}, {"id": "org/bad-sha-loom", "sha": "xyz"}]
    files = {f"https://huggingface.co/org/good-loom/resolve/{good}/model.json": b'{"model": {"id": "m"}}',
             f"https://huggingface.co/org/good-loom/resolve/{good}/model.json.lock": b'{"files": []}',
             f"https://huggingface.co/org/nolock-loom/resolve/{nolock}/model.json": b'{"model": {"id": "n"}}',
             f"https://huggingface.co/org/flaky-loom/resolve/{flaky}/model.json": b'{"model": {"id": "f"}}'}

    def fake(url, limit=MAX_BYTES, want_next=False):
        if url.startswith(API):
            return (json.dumps(page2).encode(), None) if "page=2" in url else (json.dumps(page1).encode(), API + "?page=2")
        if url.endswith("flaky-loom/resolve/" + flaky + "/model.json.lock"):
            raise OSError("HTTP 500")
        if url not in files:
            raise Missing(url)
        return files[url]

    with tempfile.TemporaryDirectory() as td:
        r = fetch_org("org", td, get=fake, workers=1)
        root = pathlib.Path(td) / "org"
        check(r["listed"] == 3, f"both listing pages are read; only the org's validated names count (3; got {r['listed']})")
        check((r["ok"], r["missing"], r["failed"]) == (1, 1, 1),
              f"one card written, a 404 lock is 'missing', a 500 lock is 'failed' (got {r})")
        check((root / "good-loom" / "model.json.lock").is_file() and (root / "good-loom" / "hf.json").is_file(),
              "the good card has model.json, model.json.lock and hf.json")
        check(not (root / "flaky-loom").exists() and not (root / "flaky-loom.part").exists(),
              "a failed lock fetch leaves no card directory (the map must not read a half card as NO-LOCK)")
        check(not (root / "nolock-loom").exists(), "a card without a lock is not written either")
        check(not (pathlib.Path(td) / "escape").exists() and not (root / ".." / "escape").exists(),
              "a '..' repo name never becomes a path")
        check(not (root / "foreign-loom").exists(), "another org's card in the listing is not written")
        hf = json.loads((root / "good-loom" / "hf.json").read_text())
        check(hf == {"id": "org/good-loom", "revision": good}, "hf.json records the repo id and the revision fetched")
    big = b"x" * (MAX_BYTES + 1)

    class R:
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n): return big[:n]
    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: R()
    try:
        http_get("https://example.invalid/x")
        check(False, "an oversized response is refused")
    except ValueError:
        check(True, "an oversized response is refused")
    finally:
        urllib.request.urlopen = orig
    print(f"loom_card_meta self-test: {'0 failed' if not fails else f'{fails} failed'}")
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--org", action="append", default=[], help="a Hugging Face org whose Loom cards to read (repeatable)")
    ap.add_argument("--out", default="loom-card-meta", help="the tree to write: OUT/<org>/<card>/")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.org:
        ap.error("give at least one --org")
    rc = 0
    for org in a.org:
        try:
            r = fetch_org(org, a.out)
        except Exception as e:
            print(f"loom_card_meta: {org}: listing failed: {e}", file=sys.stderr)
            rc = 1
            continue
        # one line per org, so a card that drops out (no lock, or a failed fetch) is visible in the job log rather than
        # only as an UNCOVERED row in the map; a failed fetch fails the run (the map would be incomplete)
        print(f"loom_card_meta: {org}: {r['listed']} card(s) tagged loom, {r['ok']} written, {r['missing']} without "
              f"model.json + lock, {r['failed']} failed to fetch")
        if r["failed"]:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
