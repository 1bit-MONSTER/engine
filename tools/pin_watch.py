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
"""Which revision of each upstream model this engine runs, ships or was built from.

usage: tools/pin_watch.py [--check] [--pr-body REPO | --pin REPO... | --bump REPO]
                          [--pins FILE] [--out FILE] [--self-test]

models/pins.json records, per upstream Hugging Face repo, the commit we pinned and the files we take
from it: the sha256 identity Hugging Face reports (`lfs.oid`, or the git blob id for a small text
file), the size, and the sha256 of the chat template that file carries. Nothing in the engine says
which revision of an upstream quant we hold, so a vendor re-quantization is invisible: unsloth
re-published Qwen3.8-Flash-Next-GGUF's UD quants on 2026-11-04 (UD-IQ4_XS went from 3 shards to 4)
and the files we re-hosted are the old ones. This tool closes that gap (docs/model-pins.md).

  --check      compare every pin with the repo's current main; write a Markdown report and exit 1 when
               anything moved, 0 when nothing did. For a moved pin it re-reads the chat template of
               every changed `.gguf` (the metadata header only, over an HTTP range request) and says
               whether the template changed, because a new revision may not be adopted before that is
               known. --no-template skips that read (the report then says it was not checked).
  --pr-body R  print the bump PR's body for repo R from the report --check just wrote.
  --pin R...   print the pin entry for each repo at its current main (adding a pin, or re-deriving one).
  --bump R     rewrite repo R's entry in --pins to the current main, for the tracked files only; a
               tracked file that vanished upstream is refused rather than dropped silently.
  --out FILE   where --check writes its report (default pin-report.md).
  --self-test  run against fixtures, no network.

Exit codes: 0 nothing moved, 1 a pin moved (or --bump refused), 2 usage, 3 --self-test failed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request

API = "https://huggingface.co/api"
RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = os.path.join(ROOT, "models", "pins.json")
UA = "1bit-engine-pin-watch"


def token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None


def api(path: str):
    """One Hugging Face API read, with the backoff the census already uses for rate limits."""
    last = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(API + path, headers={"User-Agent": UA,
                                                              **({"Authorization": f"Bearer {token()}"} if token() else {})})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # 429s and transient failures: back off, then give up
            last = e
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise last  # unreachable


def head(repo: str) -> str:
    return api(f"/models/{repo}").get("sha") or ""


def tree(repo: str, rev: str = "main") -> list[dict]:
    return api(f"/models/{repo}/tree/{rev}?recursive=true&blobs=true")


def identity(entry: dict) -> dict:
    """The file's content identity as Hugging Face reports it: `lfs.oid` is the sha256 of the bytes,
    a plain `oid` is the git blob id (a small text file is not stored in LFS)."""
    lfs = entry.get("lfs") or {}
    if lfs.get("oid"):
        return {"oid": lfs["oid"], "size": entry.get("size") or lfs.get("size") or 0, "lfs": True}
    return {"oid": entry.get("oid") or "", "size": entry.get("size") or 0, "lfs": False}


def commits_since(repo: str, old: str, new: str, limit: int = 100) -> list[dict]:
    """Commit titles from OLD (exclusive) to NEW, newest first; [] when the history is unreadable."""
    try:
        log = api(f"/models/{repo}/commits/{new}?limit={limit}")
    except Exception:
        return []
    out = []
    for c in log:
        if c.get("id") == old:
            break
        out.append({"id": c.get("id", "")[:8], "title": (c.get("title") or "").strip(), "date": (c.get("date") or "")[:10]})
    return out


# --- the chat template of a remote file, from its metadata header only ---------------------------------
class _Truncated(Exception):
    """The metadata prefix was too short: fetch more."""


class _Reader:
    def __init__(self, buf: bytes):
        self.b, self.o = buf, 0

    def take(self, n: int) -> bytes:
        if self.o + n > len(self.b):
            raise _Truncated
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def skip(self, n: int) -> None:
        if self.o + n > len(self.b):
            raise _Truncated
        self.o += n

    def string(self) -> bytes:
        return self.take(self.u64())


_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip_value(r: _Reader, t: int) -> None:
    if t in _SCALAR:
        r.skip(_SCALAR[t])
    elif t == 8:
        r.skip(r.u64())
    elif t == 9:
        et, n = r.u32(), r.u64()
        for _ in range(n):
            _skip_value(r, et)
    else:
        raise ValueError(f"bad GGUF metadata type {t}")


def gguf_template(blob: bytes) -> bytes | None:
    """`tokenizer.chat_template` from a GGUF metadata prefix, or None when the file carries none."""
    r = _Reader(blob)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF")
    r.u32()
    r.u64()
    for _ in range(r.u64()):
        key = r.string().decode("utf-8", "replace")
        t = r.u32()
        if key == "tokenizer.chat_template":
            return r.string()
        _skip_value(r, t)
    return None


def remote_bytes(repo: str, rev: str, path: str, start: int, end: int) -> bytes:
    headers = {"User-Agent": UA, "Range": f"bytes={start}-{end}"}
    if token():
        headers["Authorization"] = f"Bearer {token()}"
    req = urllib.request.Request(RESOLVE.format(repo=repo, rev=rev, path=path), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def remote_template(repo: str, rev: str, path: str, cap: int = 192 << 20) -> str | None:
    """sha256 of the chat template PATH carries at REV: a GGUF's metadata key, or a text file verbatim.

    A GGUF's tokenizer arrays sit in the metadata, so the prefix starts at 4 MiB and doubles until the
    block parses; only the metadata is ever fetched, never a tensor."""
    if path.endswith(".gguf"):
        size = 4 << 20
        while True:
            blob = remote_bytes(repo, rev, path, 0, size - 1)
            try:
                tpl = gguf_template(blob)
                return hashlib.sha256(tpl).hexdigest() if tpl is not None else None
            except _Truncated:
                if len(blob) < size:  # the file ends before its metadata does: not a readable GGUF
                    raise ValueError(f"{path}: {len(blob)} B read, no complete GGUF metadata")
                if size >= cap:
                    raise
                size = min(size * 2, cap)
    return hashlib.sha256(remote_bytes(repo, rev, path, 0, 1 << 20)).hexdigest()


# --- pins -------------------------------------------------------------------------------------------
def entry(repo: str, paths: list[str], used_for: list[str], templates: dict[str, str | None] | None = None,
          commit: str | None = None) -> dict:
    """A pin for REPO at COMMIT (default: its current main) over PATHS, with what we use it for."""
    commit = commit or head(repo)
    files = {}
    by_path = {e["path"]: e for e in tree(repo, commit)}
    for p in paths:
        e = by_path.get(p)
        if e is None:
            raise SystemExit(f"pin_watch: {repo}: {p} is not in the repo at {commit[:8]}")
        rec = identity(e)
        if templates is not None and p in templates:
            rec["template_sha256"] = templates[p]
        files[p] = rec
    return {"commit": commit, "pinned": datetime.date.today().isoformat(), "used_for": used_for, "files": files}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(path: str, doc: dict) -> None:
    doc["pins"] = {k: doc["pins"][k] for k in sorted(doc["pins"])}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def check(pins: dict, want_template: bool) -> tuple[list[dict], list[str]]:
    """Every pin that no longer matches the repo it points at, with what differs.

    Each pinned file records the identity of the bytes we HOLD, compared here with what the repo serves at
    its current main — not only with the commit, because a pin taken after a vendor re-push would
    otherwise record our older copy under a new commit and look clean."""
    changes, notes = [], []
    for repo, pin in sorted(pins["pins"].items()):
        try:
            now = head(repo)
            at = {e["path"]: e for e in tree(repo, now) if e.get("type") == "file"}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # "the upstream is gone" is the answer, not a failed read: a delete, a rename and a
                # privatisation all look like this, and none of them may pass as a clean day.
                changes.append({"repo": repo, "old": pin["commit"], "new": None, "pinned": pin["pinned"],
                                "files": [], "added": [], "commits": [], "gone": True})
            else:
                notes.append(f"{repo}: could not be read ({e}) — the pin was not checked")
            continue
        except Exception as e:
            notes.append(f"{repo}: could not be read ({e}) — the pin was not checked")
            continue
        rows = []
        for path, held in pin["files"].items():
            e = at.get(path)
            if e is None:
                rows.append({"path": path, "state": "gone upstream", "held": held})
                continue
            cur = identity(e)
            if cur["oid"] == held["oid"] and cur["size"] == held.get("size"):
                continue
            row = {"path": path, "state": "upstream serves a different file", "held": held, "now": cur}
            if want_template:
                try:
                    tpl = remote_template(repo, now, path)
                except Exception as e:
                    row["template"] = f"could not be read ({e})"
                else:
                    old = held.get("template_sha256")
                    if tpl is None and old is None:
                        row["template"] = "no template, as before"
                    elif tpl is None:
                        row["template"] = f"TEMPLATE REMOVED (we hold {old[:16]}…)"
                    elif old is None:
                        row["template"] = f"template ADDED ({tpl[:16]}…)"
                    elif tpl == old:
                        row["template"] = f"unchanged ({tpl[:16]}…)"
                    else:
                        row["template"] = f"TEMPLATE CHANGED {old[:16]}… -> {tpl[:16]}… — re-audit before adopting"
            else:
                row["template"] = "not checked (--no-template)"
            rows.append(row)
        added = []
        if now != pin["commit"]:
            added = sorted(p for p in at if p not in pin["files"]
                           and (p.endswith(".gguf") or p.endswith(".q4nx")))[:40]
        if not rows and now == pin["commit"]:
            continue
        changes.append({"repo": repo, "old": pin["commit"], "new": now, "pinned": pin["pinned"], "files": rows, "added": added,
                        "commits": commits_since(repo, pin["commit"], now) if now != pin["commit"] else []})
    return changes, notes


def report(changes: list[dict], notes: list[str], total: int, want_template: bool) -> str:
    day = datetime.date.today().isoformat()
    checked = total - len(notes)
    # The first line counts what was read. "Every pin is at the revision" must never be printed over a
    # repo that could not be read — a watcher whose "clean" and "could not check" are the same outcome
    # is blind exactly when it matters.
    if notes:
        out = [f"# Model pins: {checked} of {total} checked, {len(notes)} not checked, "
               f"{len(changes)} moved ({day})", ""]
    else:
        out = [f"# Model pins: {len(changes)} of {total} moved ({day})", ""]
    if not changes and not notes:
        out += ["Every pinned upstream repo is at the revision `models/pins.json` records.", ""]
    elif not changes:
        out += [f"No pin moved among the {checked} repo(s) that could be read; {len(notes)} could not be "
                f"checked and are listed under **Not checked** — an unread repo is not a clean one.", ""]
    for c in changes:
        out += [f"## `{c['repo']}`", ""]
        if c.get("gone"):
            out += ["**Gone upstream** — the Hub answers `404` for this repository, so every pin on it is "
                    "unverified: a delete, a rename and a privatisation look exactly like this. Nothing "
                    "here can be bumped automatically; decide, then edit or drop the pin "
                    "(`docs/model-pins.md`).", ""]
            continue
        if c["old"] == c["new"]:
            out += [f"Head is still `{c['new'][:8]}`, but the bytes it serves are not the ones we hold at "
                    f"`models/pins.json`'s `{c['pinned']}` pin.", ""]
        else:
            out += [f"{c['old'][:8]} -> {c['new'][:8]} (`{c['new']}`)", ""]
        if c["commits"]:
            out += ["| commit | when | title |", "|---|---|---|"]
            out += [f"| `{x['id']}` | {x['date']} | {x['title'][:110]} |" for x in c["commits"][:20]]
            out.append("")
        moves = [f for f in c["files"] if f["state"] != "size changed"]
        if moves:
            out += ["| file we hold | what upstream main has now | chat template |", "|---|---|---|"]
            for f in moves:
                if f["state"] == "gone upstream":
                    what = "**gone upstream**"
                else:
                    what = (f"different bytes: we hold {f['held']['size']:,} B `{f['held']['oid'][:12]}…`, "
                            f"upstream {f['now']['size']:,} B `{f['now']['oid'][:12]}…`")
                out.append(f"| `{f['path']}` | {what} | {f.get('template') or ''} |")
            out.append("")
        if c["added"]:
            out += [f"{len(c['added'])} new `.gguf`/`.q4nx` file(s) upstream, not tracked by the pin: "
                    + ", ".join(f"`{p}`" for p in c["added"][:8]) + (" …" if len(c["added"]) > 8 else ""), ""]
    if notes:
        out += ["## Not checked", ""] + [f"- {n}" for n in notes] + [""]
    if any(not c.get("gone") for c in changes):
        out += ["## Adopting a new revision", "",
                "Never automatic. A new revision is adopted by merging that repo's pin bump, and only after:",
                "1. the template row above is not `TEMPLATE CHANGED` (or the change has been audited —",
                "   `docs/template-publishing.md`, `~/tmplsec/AUDIT-1.md`),",
                "2. the row that uses it is re-measured (`tools/bench.py`, or the scoreboard row), because a",
                "   re-quantization is a different file even when its name is the same.", ""]
    return "\n".join(out)


def pr_body(c: dict) -> str:
    """The body of one bump PR: the report's section for that repo, plus what the bump changes."""
    if c.get("gone"):
        return "\n".join([f"`{c['repo']}` answers `404` on the Hub, so its pin cannot be bumped: a delete, a",
                           "rename and a privatisation are indistinguishable from here. Decide whether the pin moves to",
                           "the new name or is dropped, then edit `models/pins.json` (`docs/model-pins.md`)."])
    if c["old"] == c["new"]:
        out = [f"The bytes we hold from `{c['repo']}` are not what it serves at `{c['new'][:8]}` — our copies "
               f"predate the head. Merging this re-pins the tracked files to the bytes upstream serves now, "
               f"which is a re-host, not a commit bump.", ""]
    else:
        out = [f"Bumps the pin of `{c['repo']}` from `{c['old'][:8]}` to `{c['new'][:8]}` (`{c['new']}`).", ""]
    if c["commits"]:
        out += ["Upstream commits since the pin:", ""]
        out += [f"- `{x['id']}` {x['date']} {x['title']}" for x in c["commits"][:20]]
        out.append("")
    rows = [f for f in c["files"] if f["state"] != "size changed"]
    if rows:
        out += ["| file we hold | what upstream main has now | chat template |", "|---|---|---|"]
        for f in rows:
            what = "**gone upstream** — the bump stops until this is decided" if f["state"] == "gone upstream" \
                else (f"different bytes: we hold {f['held']['size']:,} B `{f['held']['oid'][:12]}…`, "
                      f"upstream {f['now']['size']:,} B `{f['now']['oid'][:12]}…`")
            out.append(f"| `{f['path']}` | {what} | {f.get('template') or ''} |")
        out.append("")
    if c["added"]:
        out += [f"{len(c['added'])} new file(s) appeared upstream: " + ", ".join(f"`{p}`" for p in c["added"][:8]), ""]
    dangers = [f for f in c["files"] if f["state"] == "gone upstream" or "TEMPLATE CHANGED" in str(f.get("template"))
               or "TEMPLATE REMOVED" in str(f.get("template"))]
    if dangers:
        out += ["**This is not a routine bump:** " + "; ".join(
            f"`{f['path']}` {f['state']}" for f in dangers) + ".", ""]
    out += ["Merging the pin does not adopt the new revision anywhere: the catalog entry, the stick image and",
            "the scoreboard each move on their own reviewed change. Merging says *this is the revision we",
            "point at* — so the template column above has to be clean first (docs/model-pins.md).", ""]
    out.append("Report: `tools/pin_watch.py --check` (the daily `pin-watch` workflow).")
    return "\n".join(out)


def self_test() -> int:
    global api, head, tree, remote_template, commits_since
    failures = 0

    def expect(what, got, want):
        nonlocal failures
        ok = got == want
        print(f"{'ok  ' if ok else 'FAIL'} {what}: {got!r}")
        failures += 0 if ok else 1

    tpl = b"{% for m in messages %}<s>{{ m['role'] }}{% endfor %}"

    def kv(key: str, value: bytes) -> bytes:
        return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value

    gguf = b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
    gguf += kv("general.architecture", b"zamba2")
    gguf += kv("tokenizer.chat_template", tpl)
    expect("gguf template read", hashlib.sha256(gguf_template(gguf)).hexdigest(), hashlib.sha256(tpl).hexdigest())
    bare = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + kv("general.architecture", b"zamba2")
    expect("gguf without a template", gguf_template(bare), None)

    # a two-revision fake upstream: `old` holds the file, `new` replaces it and its template
    trees = {
        "old": {"tpl.gguf": {"path": "tpl.gguf", "type": "file", "size": len(gguf),
                             "lfs": {"oid": "a" * 64}, "x": gguf}},
        "new": {"tpl.gguf": {"path": "tpl.gguf", "type": "file", "size": len(gguf) + 4,
                             "lfs": {"oid": "b" * 64}, "x": gguf + b"\x00" * 4},
                "extra-Q8_0.gguf": {"path": "extra-Q8_0.gguf", "type": "file", "size": 9, "lfs": {"oid": "c" * 64}}},
    }
    new_tpl = [tpl.replace(b"<s>", b"<|im_start|>")]

    def fake_head(repo):
        return "new" if repo == "fake/moved" else "old"

    def fake_tree(repo, rev="main"):
        return [v for k, v in trees["new" if rev == "new" else "old"].items()]

    def fake_commits(repo, old, new, limit=100):
        return [{"id": "deadbeef", "title": "Better UD 3.1 quants", "date": "2026-11-04"}]

    def fake_template(repo, rev, path, cap=0):
        return hashlib.sha256(new_tpl[0]).hexdigest() if rev == "new" else hashlib.sha256(tpl).hexdigest()

    base = {"commit": "old", "pinned": "2026-11-05", "used_for": ["test"],
            "files": {"tpl.gguf": {"oid": "a" * 64, "size": len(gguf), "lfs": True,
                                   "template_sha256": hashlib.sha256(tpl).hexdigest()}},
            "x": None}
    pins = {"pins": {"fake/moved": base, "fake/still": dict(base)}}
    head, tree, remote_template, commits_since = fake_head, fake_tree, fake_template, fake_commits
    changes, notes = check(pins, want_template=True)
    expect("one repo moved", [c["repo"] for c in changes], ["fake/moved"])
    expect("the replacement is seen", changes[0]["files"][0]["state"], "upstream serves a different file")
    expect("the template change is named", "TEMPLATE CHANGED" in changes[0]["files"][0]["template"], True)
    expect("the new upstream file is listed", changes[0]["added"], ["extra-Q8_0.gguf"])
    expect("no notes for a readable repo", notes, [])
    expect("the upstream commit title is carried", changes[0]["commits"][0]["title"], "Better UD 3.1 quants")
    md = report(changes, notes, 2, True)
    expect("report names the repo", "fake/moved" in md and "TEMPLATE CHANGED" in md, True)
    expect("report carries the commit title", "Better UD 3.1 quants" in md, True)
    expect("pr body refuses a routine merge", "not a routine bump" in pr_body(changes[0]).lower(), True)

    # a repo that did not move is silent, and an unreadable repo is a note, not a change
    # the same commit, but the bytes we hold are not upstream's: a pin taken after a vendor re-push
    stale = {"commit": "new", "pinned": "2026-11-05", "used_for": ["test"],
             "files": {"tpl.gguf": {"oid": "a" * 64, "size": len(gguf), "lfs": True,
                                    "template_sha256": hashlib.sha256(tpl).hexdigest()}}}

    def fake_head_new(repo):
        return "new"

    head, tree, remote_template, commits_since = fake_head_new, fake_tree, fake_template, fake_commits
    changes, _ = check({"pins": {"fake/moved": stale}}, want_template=False)
    expect("a stale copy is caught at the same commit", len(changes), 1)
    expect("held oid is reported", changes[0]["files"][0]["held"]["oid"][:4], "aaaa")

    def fake_head_none(repo):
        if repo == "fake/still":
            raise RuntimeError("404")
        return "new"

    head = fake_head_none
    changes, notes = check(pins, want_template=False)
    expect("unreadable repo is a note", (len(changes), "fake/still" in notes[0]), (1, True))
    expect("--no-template says so", changes[0]["files"][0]["template"], "not checked (--no-template)")

    # --bump refuses a pin whose tracked file vanished upstream, rather than dropping it silently
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pins.json")
        save(path, {"pins": {"fake/moved": dict(stale), "fake/gone": dict(stale)}})

        def fake_tree_missing(repo, rev="main"):
            return [v for k, v in trees["new"].items() if k != "tpl.gguf"]

        head, tree, commits_since = fake_head_new, fake_tree_missing, fake_commits
        expect("--bump refuses a vanished file", main(["--bump", "fake/moved", "--pins", path]), 1)
        expect("--bump left the pin alone", load(path)["pins"]["fake/moved"]["commit"], "new")

        tree = fake_tree
        expect("--bump takes an intact pin", main(["--bump", "fake/moved", "--pins", path]), 0)
        expect("--bump moved the commit", load(path)["pins"]["fake/moved"]["commit"], "new")
        expect("--bump took upstream's oid", load(path)["pins"]["fake/moved"]["files"]["tpl.gguf"]["oid"][:4] is not None, True)

        # a repository the Hub answers 404 for is the answer, not a failed read: a delete, a rename and a
        # privatisation must not pass as a quiet day
        def fake_head_404(repo):
            raise urllib.error.HTTPError("https://huggingface.co/api/models/" + repo, 404, "Not Found", {}, None)

        head, tree = fake_head_404, fake_tree
        gone_changes, gone_notes = check({"pins": {"fake/gone": dict(stale)}}, want_template=False)
        expect("a 404 is a finding, not a note", (len(gone_changes), len(gone_notes)), (1, 0))
        expect("it is carried as gone", gone_changes[0].get("gone"), True)
        gone_md = report(gone_changes, gone_notes, 1, False)
        expect("the report says gone upstream", "Gone upstream" in gone_md, True)
        expect("no all-clear over a 404", "Every pinned upstream repo is at" in gone_md, False)
        expect("--bump refuses a gone repo", main(["--bump", "fake/gone", "--pins", path]), 1)
        expect("the gone pin is untouched", load(path)["pins"]["fake/gone"]["commit"], "new")

        # a run that could read nothing is exit 2 (the workflow fails on anything outside 0/1) and its
        # first line counts what it read
        def fake_head_flaky(repo):
            raise RuntimeError("HTTP Error 429: Too Many Requests")

        head = fake_head_flaky
        report_path = os.path.join(d, "unread.md")
        expect("nothing checked -> exit 2", main(["--check", "--pins", path, "--out", report_path]), 2)
        unread_md = open(report_path, encoding="utf-8").read()
        expect("the count is in the first line", "0 of 2 checked, 2 not checked" in unread_md.splitlines()[0], True)
        expect("and still no all-clear", "Every pinned upstream repo is at" in unread_md, False)
    return 3 if failures else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pr-body", metavar="REPO")
    ap.add_argument("--pin", nargs="+", metavar="REPO")
    ap.add_argument("--bump", metavar="REPO")
    ap.add_argument("--pins", default=PINS)
    ap.add_argument("--out", default="pin-report.md")
    ap.add_argument("--no-template", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()

    if a.pin:
        made = {}
        for repo in a.pin:
            paths = sorted(e["path"] for e in tree(repo) if e.get("type") == "file" and e["path"].endswith(".gguf"))
            made[repo] = entry(repo, paths, [f"pin: {repo}"])
        print(json.dumps(made, indent=1, sort_keys=True))
        return 0

    if not a.check and not a.bump and not a.pr_body:
        ap.print_help()
        return 2

    doc = load(a.pins)
    if a.bump:
        pin = doc["pins"].get(a.bump) or sys.exit(f"pin_watch: {a.bump} is not pinned")
        try:
            now = head(a.bump)
            at_new = {e["path"]: e for e in tree(a.bump, now)}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"pin_watch: {a.bump}: gone upstream (404) — a delete, a rename or a privatisation; "
                      f"decide, then edit or drop the pin", file=sys.stderr)
                return 1
            raise
        missing = [p for p in pin["files"] if p not in at_new]
        if missing:
            print(f"pin_watch: {a.bump}: {len(missing)} tracked file(s) are gone upstream: "
                  + ", ".join(missing) + " — decide, then edit the pin", file=sys.stderr)
            return 1
        tpls = {p: pin["files"][p].get("template_sha256") for p in pin["files"]}
        for p in pin["files"]:
            if pin["files"][p].get("template_sha256") is not None:
                tpls[p] = remote_template(a.bump, now, p)
        doc["pins"][a.bump] = entry(a.bump, sorted(pin["files"]), pin["used_for"], templates=tpls)
        save(a.pins, doc)
        print(f"pin_watch: {a.bump} pinned at {doc['pins'][a.bump]['commit'][:8]}")
        return 0

    changes, notes = check(doc, want_template=not a.no_template)
    md = report(changes, notes, len(doc["pins"]), not a.no_template)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    if a.pr_body:
        hit = next((c for c in changes if c["repo"] == a.pr_body), None)
        if hit is None:
            sys.exit(f"pin_watch: {a.pr_body} did not move (see {a.out})")
        print(pr_body(hit))
        return 1
    print(md)
    if notes and len(notes) == len(doc["pins"]):
        # nothing was read: a failed run, not a clean one. The workflow fails on any code outside 0/1.
        print(f"pin_watch: no pin could be checked ({len(notes)} unreadable) — this is not a clean run",
              file=sys.stderr)
        return 2
    return 1 if changes else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
