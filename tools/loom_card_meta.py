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
counted), 1 when the listing of an org failed.
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/models"
FILES = ("model.json", "model.json.lock")
MAX_BYTES = 4 << 20          # a model.json or lock is a few kB; anything this large is not one
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
SHA = re.compile(r"^[0-9a-f]{40}$")


def http_get(url, limit=MAX_BYTES):
    req = urllib.request.Request(url, headers={"User-Agent": "loom-card-meta"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{url}: larger than {limit} bytes")
    return data


def list_cards(org, get=http_get):
    """[(repo id, revision)] of the org's public models tagged `loom` (the card layout), from the listing API."""
    url = f"{API}?author={urllib.parse.quote(org)}&filter=loom&limit=1000&full=true"   # without `full` the listing has no sha
    out = []
    for m in json.loads(get(url, limit=64 << 20)):
        rid, sha = m.get("id", ""), m.get("sha", "")
        owner, _, name = rid.partition("/")
        if owner != org or not NAME.match(name) or not SHA.match(sha or ""):
            continue   # never a path we did not validate
        out.append((rid, sha))
    return out


def fetch_card(rid, sha, root, get=http_get):
    """Write root/<name>/{model.json, model.json.lock, hf.json}; True when both files were fetched."""
    name = rid.split("/", 1)[1]
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        try:
            data = get(f"https://huggingface.co/{rid}/resolve/{sha}/{f}")
            json.loads(data)   # both are JSON; anything else is not a card file
        except Exception:
            return False
        (d / f).write_bytes(data)
    (d / "hf.json").write_text(json.dumps({"id": rid, "revision": sha}) + "\n")
    return True


def fetch_org(org, out, get=http_get, workers=6):
    root = pathlib.Path(out) / org
    root.mkdir(parents=True, exist_ok=True)
    cards = list_cards(org, get)
    with cf.ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(lambda c: fetch_card(c[0], c[1], root, get), cards))
    return len(cards), sum(ok)


def self_test():
    """Hermetic: a fake HTTP layer with one good card, one without a lock, and names that must never become paths."""
    import tempfile
    fails = 0

    def check(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        fails += 0 if cond else 1

    good, nolock = "a" * 40, "b" * 40
    listing = [{"id": "org/good-loom", "sha": good}, {"id": "org/nolock-loom", "sha": nolock},
               {"id": "org/../escape", "sha": good}, {"id": "other/foreign-loom", "sha": good},
               {"id": "org/bad-sha-loom", "sha": "xyz"}]
    files = {f"https://huggingface.co/org/good-loom/resolve/{good}/model.json": b'{"model": {"id": "m"}}',
             f"https://huggingface.co/org/good-loom/resolve/{good}/model.json.lock": b'{"files": []}',
             f"https://huggingface.co/org/nolock-loom/resolve/{nolock}/model.json": b'{"model": {"id": "n"}}'}

    def fake(url, limit=MAX_BYTES):
        if url.startswith(API):
            return json.dumps(listing).encode()
        if url not in files:
            raise OSError("404")
        return files[url]

    with tempfile.TemporaryDirectory() as td:
        n, ok = fetch_org("org", td, get=fake, workers=1)
        root = pathlib.Path(td) / "org"
        check(n == 2, f"only validated names of the org are listed (2 of 5; got {n})")
        check(ok == 1, f"a card is written only when both files exist (got {ok})")
        check((root / "good-loom" / "model.json.lock").is_file() and (root / "good-loom" / "hf.json").is_file(),
              "the good card has model.json, model.json.lock and hf.json")
        check(not (pathlib.Path(td) / "escape").exists() and not (root / ".." / "escape").exists(),
              "a '..' repo name never becomes a path")
        check(not (root / "foreign-loom").exists(), "another org's card in the listing is not written")
        hf = json.loads((root / "good-loom" / "hf.json").read_text())
        check(hf == {"id": "org/good-loom", "revision": good}, "hf.json records the repo id and the revision fetched")
    try:
        big = b"x" * (MAX_BYTES + 1)

        class R:
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
    except Exception as e:   # pragma: no cover
        check(False, f"size cap arm raised {e!r}")
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
            n, ok = fetch_org(org, a.out)
            print(f"loom_card_meta: {org}: {ok} of {n} card(s) written ({n - ok} without model.json + lock)")
        except Exception as e:
            print(f"loom_card_meta: {org}: listing failed: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
