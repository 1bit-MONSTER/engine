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
"""loom_coverage_check.py -- which shipped GPU models have a Loom card, and on which quant (engine#3156).

The Loom switch moves one model at a time, so the coverage map is the plan
(docs/loom.md, engine#3156). It was a hand-written table until this tool, and a
hand-written table is how the row that matters gets missed: the trap this encodes
is that a card matching on FAMILY AND SIZE is not a card for the model.

    ggml-org-Qwen2.5-VL-7B-Instruct-loom     a VL model, not Qwen2.5-7B-Instruct
    unsloth-Qwen3-30B-A3B-Instruct-2507-loom a later model, not Qwen3-Coder-30B-A3B

So a match is reported only when the card's `model.json` -> model.id names the same
size class as the registry row, and the row's quantisation is then looked for in the
same file. A card that names the model but not our quant is COVERED-BUT-REQUANT,
which is a different answer from covered: the gate has to A/B the same GGUF where the
card allows, or say why it cannot.

Exit: 0 every shipped row is covered on its own quant, 1 a row is not (named), 2 cannot
tell -- no card tree, which is CI's normal state because the cards are ~GB downloads.
"""
import argparse
import json
import pathlib
import re
import sys

QUANTS = re.compile(r"UD[-_]?Q\d+_K_[A-Z]+|Q\d+_K_[A-Z]+|Q\d+_\d|IQ\d+_[A-Z]+|BF16|F16|MXFP4|AWQ", re.I)


def read_rows(tsv):
    """(device, model) rows of registry/check_models.tsv."""
    out = []
    for line in pathlib.Path(tsv).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out


def read_manifest(path):
    """(device, model id) rows of the manifest `os/v3/puppy/catalogue-manifest.py` writes for the stick:
    `mode=live`, then one `hrx <id>` / `npu <id>` line per shipped catalogue entry."""
    out = []
    for line in pathlib.Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.split("=")[0] == "mode":
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0] in ("hrx", "npu"):
            out.append((parts[0], parts[1].strip()))
    return out


def _is_quant_spec(rest):
    """True when `rest` is a quant tag and nothing else: `q4_k_m`, `ud-q4_k_xl`, `mxfp4`.

    The quant is matched first and whatever it leaves behind has to be a size qualifier, so a model that merely
    starts with the same text (`instruct-2507-q4_k_m`) is not the same model.
    """
    m = QUANTS.search(rest)
    if not m:
        return False
    leftover = rest[: m.start()] + rest[m.end() :]
    return all(t in ("ud", "xl", "xs", "xxs") for t in re.split(r"[-_]", leftover) if t)


def _named_by(basename, key):
    """True when a pinned FILE belongs to catalogue id `key`: the stem is the id, or the id plus a quant spec."""
    if basename == key + ".gguf":
        return True
    if not basename.startswith(key + "-"):
        return False
    rest = basename[len(key) + 1 : -len(".gguf")] if basename.endswith(".gguf") else basename[len(key) + 1 :]
    return _is_quant_spec(rest)


def rows_from_manifest(manifest_rows, ours):
    """(rows, unresolved): the shipped catalogue as (device, model FILE) rows.

    A catalogue id names a MODEL, and every verdict this tool makes is about a FILE (the bytes a card binds),
    so an id is resolved to the files we pin for it. The pin records basenames; an id is either a file stem
    (`Qwen3-4B-Q4_K_M`) or a repo name (`unsloth/Qwen3-4B-GGUF`), so the last path segment is the key, with a
    trailing `-gguf` dropped. An id that resolves to nothing is reported as unresolved rather than guessed at:
    a coverage claim that rests on a name match is the trap this tool exists to avoid.
    """
    names = (ours or {}).get("names") or set()
    rows, unresolved = [], []
    for device, model_id in manifest_rows:
        if device != "hrx":
            continue  # the NPU route is a different map (docs/npu.md); this tool reports HRX rows only
        key = model_id.split("/")[-1].lower()
        # A catalogue id carries the distribution's own suffix (`Qwen3-4B-1bit` in the stick's Lemonade ids,
        # `unsloth/Qwen3-4B-GGUF` in a repo id); neither is part of the model name.
        for suffix in ("-1bit", "-npu", "-gguf"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        if key.endswith(".gguf"):
            rows.append((device, key))
            continue
        hits = sorted(n for n in names if _named_by(n, key))
        if not hits:
            unresolved.append(model_id)
        rows.extend((device, n) for n in hits)
    return rows, unresolved


def identity(card_dir):
    """(model id, quants of the CHECKPOINT files) from a card's model.json, or (None, set()) when unreadable.

    The checkpoint is what the card serves, and it is the only thing a coverage claim may rest on. A card may also
    name files that are NOT the checkpoint -- Qwen3.6-35B-A3B ships a DFlash drafter whose files carry their own
    quantisation -- and reading `model.json` as text counted those too, which made that row read COVERED on Q8_0
    when its checkpoint is Q4_K_M (engine#3156 review). So walk the variant/service/checkpoint-file paths only.
    """
    f = card_dir / "model.json"
    try:
        doc = json.loads(f.read_text())
    except Exception:
        return None, set()
    model = doc.get("model") if isinstance(doc, dict) else None
    ident = model.get("id") if isinstance(model, dict) else None
    quants = set()
    for variant in ((doc.get("variants") or {}) if isinstance(doc, dict) else {}).values():
        for svc in (variant.get("services") or {}).values():
            for entry in (((svc.get("parameters") or {}).get("model") or {}).get("files") or []):
                quants |= {q.upper() for q in QUANTS.findall(entry.get("path", ""))}
    return ident, quants



def pin_bytes(pins_path):
    """({sha256 we ship}, {basenames we pin}) from models/pins.json, or None when it cannot be read.

    The pin is the only record of the BYTES we ship (`files.<name>.oid` is the LFS sha256), and a card binds its
    checkpoint by sha256 in `model.json.lock`. Two cards can name the same model on the same quant and bind
    different files -- `unsloth/flint`'s UD-Q4_K_XL is 25.05 GB where the card's Q4_K_M is 23.39 GB, and our own
    pin is sometimes a RE-HOST (the Flash-Next card binds the 1bit-MONSTER shards while the upstream pin is an
    older revision, and `used_for` says so in as many words). So a coverage claim that stops at the quant is an
    upper bound, and this is the arm that turns it into a fact.
    """
    try:
        doc = json.loads(pathlib.Path(pins_path).expanduser().read_text())
    except Exception:
        return None
    pins = doc.get("pins") if isinstance(doc, dict) else None
    if not isinstance(pins, dict):
        return None
    shas, names, by_name = set(), set(), {}
    for entry in pins.values():
        for fn, meta in ((entry or {}).get("files") or {}).items():
            base = fn.split("/")[-1].lower()
            names.add(base)
            sha = (meta or {}).get("oid")
            if isinstance(sha, str) and len(sha) == 64:
                shas.add(sha.lower())
                by_name.setdefault(base, set()).add(sha.lower())
    return {"shas": shas, "names": names, "by_name": by_name}


CHECKPOINT_KEYS = ("model", "checkpoint")


def checkpoint_shas(card_dir, lock):
    """The sha256 of the card's CHECKPOINT files (not its drafter's), from the lock, or an empty set."""
    try:
        doc = json.loads((card_dir / "model.json").read_text())
    except Exception:
        return set()
    names, shas = set(), set()
    for variant in ((doc.get("variants") or {}) if isinstance(doc, dict) else {}).values():
        for svc in ((variant or {}).get("services") or {}).values():
            params = (svc or {}).get("parameters") or {}
            for key in CHECKPOINT_KEYS:
                for f in ((params.get(key) or {}).get("files") or []):
                    n = (f or {}).get("path", "").split("/")[-1].lower()
                    if n in lock:
                        names.add(n)
    for n in names:
        shas |= lock[n]
    # engine#3156 review: the old fallback returned EVERY file the lock binds when the checkpoint file names could
    # not be read, which compares our bytes against the DRAFTER's (three files for Qwen3-Coder-30B-A3B). An
    # unreadable checkpoint is cannot-tell for the bytes arm, so return nothing and let the caller report
    # COVERED-UNPINNED rather than a fabricated DIFFERENT-UPLOAD.
    return shas


def lock_bytes(card_dir):
    """basename -> {sha256 the card binds} from model.json.lock, or None when the card has no lock."""
    try:
        doc = json.loads((card_dir / "model.json.lock").read_text())
    except Exception:
        return None
    out = {}
    for f in (doc.get("files") or []):
        origin, sha = (f or {}).get("origin"), (f or {}).get("sha256")
        if isinstance(origin, str) and isinstance(sha, str) and len(sha) == 64:
            out.setdefault(origin.split("/")[-1].lower(), set()).add(sha.lower())
    return out

NON_DISTINCTIVE = {"loom", "gguf", "merged"}
# Publisher/organisation tokens that appear in card directory names (and HF ids) but are not model identity.
VENDOR = {"ggml", "org", "bartowski", "unsloth", "lmstudio", "lmstudio-community", "model", "ai", "the",
          "someone", "acme", "liquidai", "qwen", "meta-llama", "google", "microsoft",
          # our own org's cards (`1bit-MONSTER/<model>-loom`, and the `1bit-<model>` trees the lanes build): without
          # these, same_model() rejects every card we publish ourselves and the row that matters most reads
          # UNCOVERED — GLM-4.7-Flash's published card did, with its lock binding our pinned sha256.
          "1bit", "monster"}


def _quant_token(t):
    # `xl`/`xs`/`xxs` are the K-quant size suffix of an Unsloth Dynamic tag (Q4_K_XL) and `ud` its prefix:
    # they qualify the QUANT, so they must not distinguish a card from a row (engine#3156).
    return bool(re.fullmatch(r"i?q\d+([_-][a-z0-9]+)*|f16|bf16|mxfp4|awq|xl|xs|xxs|ud", t))


def _tokens(name):
    """Distinctive tokens of a model name: family, size, and qualifiers, with quant tokens removed.

    Vendor and organisation tokens (`bartowski`, `ggml`, `org`, …) are filtered by VENDOR rather than by
    position, because a card directory may carry one or two of them (`ggml-org-Foo-loom`).
    """
    stem = re.sub(r"\.gguf$", "", name).split("/")[-1].lower()
    # single letters are quant suffixes split apart (`Q4_K_M` -> q4, k, m); they are never identity
    return [t for t in re.split(r"[^a-z0-9.]+", stem) if len(t) > 1 and not _quant_token(t)]


def same_model(row_name, card_dir, card_id):
    """True when a card is for THIS model: every row token appears, and the card adds nothing but `loom`.

    This is the trap the map has to survive: `ggml-org-Qwen2.5-VL-7B-Instruct-loom` and
    `unsloth-Qwen3-30B-A3B-Instruct-2507-loom` match their rows on family and size and are different
    models. A size-class match alone is a candidate, never a cover.
    """
    row = set(_tokens(row_name)) - NON_DISTINCTIVE
    card = set(_tokens(card_dir)) - NON_DISTINCTIVE
    if card_id:
        card |= set(_tokens(card_id)) - NON_DISTINCTIVE
    # every token of the row must be present, and nothing beyond the vendor/`loom` vocabulary may be added:
    # `Qwen2.5-VL-7B-Instruct` and `Qwen3-30B-A3B-Instruct-2507` both add a token and are different models.
    return row <= card and (card - row) <= (NON_DISTINCTIVE | VENDOR)


def canon_quant(q):
    """One spelling for a quant tag: `UD_Q4_K_XL` from a file name and `UD-Q4_K_XL` from a card must compare
    equal, and the UD (Unsloth Dynamic) prefix is part of the quant, not a model-name token."""
    return q.upper().replace("_", "-") if q else q


def quant_of(name):
    m = QUANTS.search(name.replace("-", "_"))
    return m.group(0).upper() if m else None


def classify(row, cards, ours=None):
    """(verdict, detail) for one shipped model against the card list."""
    stem = re.sub(r"\.gguf$", "", row)
    want_q = quant_of(stem)
    # Bytes first: a card whose lock binds the sha256 we pin for THIS row's file is a card for it, whatever its
    # directory is called. The name rule below needs every publisher token in VENDOR, and a corpus from a new
    # uploader (loom-draft's 401 cards: `openbmb-MiniCPM5-1B-loom` binds our pinned MiniCPM5 bytes) read UNCOVERED
    # on a word list, not on evidence (engine#3156). The sha is the identity; the name rule stays for the rows we
    # cannot compare.
    fn = stem.split("/")[-1].lower() + ".gguf"
    row_shas = ((ours or {}).get("by_name") or {}).get(fn) or set()
    # A registry row is a bare file name: when two pins carry that name with different bytes, the row does not say
    # which bytes are ours, so a card on either would be a false cover (#3525 review). Then the name rule decides.
    if len(row_shas) == 1:
        for d, _ident, _quants, lock in cards:
            if lock is None:
                continue
            both = checkpoint_shas(d, lock) & row_shas
            if both:
                return "COVERED", f"{d.name} ({want_q or 'bytes'}) sha256 {sorted(both)[0][:12]} (bytes match)"
    hits = []
    for d, ident, quants, _lock in cards:
        if not same_model(row, d.name, ident):
            continue
        hits.append((d, ident, quants, _lock))
    if not hits:
        return "UNCOVERED", "no card names this model"
    same_quant = [h for h in hits if want_q and canon_quant(want_q) in {canon_quant(q) for q in h[2]}]
    if same_quant:
        d, _ident, _quants, lock = same_quant[0]
        # name and quant agree; the question left is whether the card binds BYTES we ship. Compared as a set of
        # sha256, not by filename: our pin is sometimes a re-host, so the same weights can arrive named differently
        # (`1bit-MONSTER/Qwen3.8-Flash-Next-…-MTP` vs the card's `@quantized` shards), and a filename match is not
        # evidence of anything. A card with no lock cannot answer at all.
        if ours is None:
            return "COVERED-NOBYTES", f"{d.name} ({want_q}); no pins to compare"
        if lock is None:
            return "NO-LOCK", f"{d.name} ({want_q}) has no model.json.lock: its checkpoint is unverified here"
        fn = re.sub(r"\.gguf$", "", row).split("/")[-1].lower() + ".gguf"
        if fn not in ours["names"]:
            # no pin for this row: covered on name+quant, and the report says the bytes were not checked rather
            # than failing a PR for a bookkeeping gap in models/pins.json
            return "COVERED-UNPINNED", f"{d.name} ({want_q}); we pin no file named {fn}"
        theirs = checkpoint_shas(d, lock)
        if not theirs:
            return "COVERED-UNPINNED", f"{d.name} ({want_q}); its lock names no checkpoint file we can read"
        both = theirs & ours["shas"]
        if both:
            return "COVERED", f"{d.name} ({want_q}) sha256 {sorted(both)[0][:12]}"
        return "DIFFERENT-UPLOAD", (f"{d.name} binds {sorted(theirs)[0][:12]}, we ship none of "
                                    f"{len(theirs)} file(s) it names")
    return "REQUANT", f"{hits[0][0].name} has {sorted(hits[0][2]) or '(no quant named)'}, we ship {want_q}"


def card_roots(paths):
    out = []
    for p in paths:
        d = pathlib.Path(p).expanduser()
        if d.is_dir():
            out.append(d)
    return out


def cards_of(roots):
    cards = []
    for root in roots:
        for d in sorted(root.iterdir()):
            if d.is_dir():
                ident, quants = identity(d)
                cards.append((d, ident, quants, lock_bytes(d)))
    return cards


def check(tsv, roots, ours=None, rows=None):
    """(rc, lines): rc as documented, lines the report. Returned rather than printed so the self-test can assert
    the VERDICT a row got -- COVERED, UNCOVERED, REQUANT are three different answers and two of them share an rc."""
    cards = cards_of(roots)
    if not cards:
        return 2, ["loom_coverage: no card tree found (pass --cards); the cards are ~GB downloads, so this is "
                   "cannot tell, not a pass"]
    lines, bad, unpinned_bytes = [], 0, 0
    compared = unpinned_rows = 0
    for device, model in (rows if rows is not None else read_rows(tsv)):
        if device != "hrx":
            continue
        verdict, detail = classify(model, cards, ours)
        lines.append(f"{verdict:9s} {model:46s} {detail}")
        # COVERED-UNPINNED is covered on name+quant with no pin of ours to compare, which is a bookkeeping gap
        # rather than a finding; DIFFERENT-UPLOAD and NO-LOCK are findings and fail the row.
        # engine#3156 review: COVERED-NOBYTES means we have no pins file at all, so the bytes arm cannot run --
        # that is cannot tell (rc 2), not a finding against whichever PR happens to run the gate. It is counted
        # separately and reported in the summary so the label and the exit code agree.
        if verdict == "COVERED-NOBYTES":
            unpinned_bytes += 1
        elif verdict == "COVERED":
            compared += 1
        elif verdict == "COVERED-UNPINNED":
            unpinned_rows += 1
        elif verdict not in ("COVERED", "COVERED-UNPINNED"):
            bad += 1
    # The count of rows whose BYTES were actually compared, always printed (engine#3156 review: `COVERED-UNPINNED`
    # exits 0 -- the row IS covered on name+quant and failing it would be a bookkeeping red -- but a map where every
    # row is unpinned then reads as "every shipped row covered" with zero byte comparisons behind it. The number is
    # what makes the label and the exit code honest, so it is emitted even when it is 0 of N.)
    if compared + unpinned_rows + unpinned_bytes:
        # Not printed for a tree with no `hrx` row at all: "nothing is reported for the rows this map does not
        # cover" is its own arm, and a summary line would be a report about nothing.
        lines.append(f"loom_coverage: {compared} of {compared + unpinned_rows + unpinned_bytes} row(s) compared on "
                     f"BYTES; {unpinned_rows} covered on name+quant only (we pin no file named for them)")
    if unpinned_bytes:
        lines.append(f"loom_coverage: {unpinned_bytes} row(s) could not be compared: no pins file "
                     f"(pass --pins); the name+quant arm ran, the bytes arm did not -- cannot tell, not a pass")
    if bad:
        return 1, lines
    return (2 if unpinned_bytes else 0), lines


def self_test():
    """Hermetic: a synthetic registry and card tree, with the two traps the map has to get right."""
    import tempfile
    fails = 0
    total = 0   # engine#3156 review: count the arms instead of a hand-moved literal that drifted 7 -> 11 for 10 arms

    def make_card(root, cards):
        """Write the card tree `card_roots()` expects: a dir per card holding a model.json that names its
        checkpoint files and a model.json.lock that binds their sha256. Element layout:
        (name, model id, checkpoint quants, drafter quants, lock rows); the last two are optional."""
        for card in cards:
            cname, ident, quants = card[0], card[1], card[2]
            extra = card[3] if len(card) > 3 else ()
            drafter = extra if isinstance(extra, tuple) else ()
            lock = card[4] if len(card) > 4 else None
            d = root / cname
            d.mkdir()
            # the shape a real card has: the checkpoint is a file path under a variant's service parameters
            files = [{"path": f"parameter/M-{q}.gguf"} for q in quants]
            doc = {"model": {"id": ident},
                   "variants": {"gguf": {"services": {"chat": {"parameters": {"model": {"files": files}}}}}}}
            # ...and a card may carry a DRAFTER beside the checkpoint, under the SAME service's `draft`
            # parameter -- which is where the real Qwen3.6-35B-A3B card keeps its Q8_0 DFlash file
            # (`variants.gguf.services.chat.parameters.draft.files[0].path`). The coverage claim must not read
            # those, which is the defect this arm exists for (engine#3156 review).
            if drafter:
                chat = doc["variants"]["gguf"]["services"]["chat"]
                chat["parameters"]["draft"] = {"files": [{"path": f"parameter/D-{q}.gguf"} for q in drafter]}
            (d / "model.json").write_text(json.dumps(doc))
            # the lock is what binds the BYTES; None means "this card has no lock", which is its own arm
            if lock is not None:
                (d / "model.json.lock").write_text(json.dumps(
                    {"files": [{"origin": f"@checkpoint/M-{q}.gguf", "sha256": sha}
                               for q, sha in lock]}))

    def arm(name, want, rows, cards, expect_rc, pins=None, says=None):
        """Assert the VERDICT, not only the exit code: COVERED (0), UNCOVERED and REQUANT (both 1) are three
        answers and two of them share an rc, so an rc-only arm cannot tell them apart -- which is the one
        distinction the tool exists to make.

        `says` additionally requires a substring in some report line, for the arms whose subject IS a sentence: a
        summary that reports a count has to be asserted by its count, or the arm passes on any line that starts
        with the tool's name."""
        nonlocal fails, total
        total += 1
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "cards"
            root.mkdir()
            make_card(root, cards)
            tsv = pathlib.Path(td) / "models.tsv"
            tsv.write_text("\n".join(f"{dev}\t{m}" for dev, m in rows) + "\n")
            pins_path = pathlib.Path(td) / "pins.json"
            pins_path.write_text(json.dumps({"pins": pins or {}}))
            rc, lines = check(tsv, [root], pin_bytes(pins_path) if pins is not None else None)
            verdicts = [line.split()[0] for line in lines] if rc != 2 else ["cannot-tell"]
            # "no-verdict-at-all" is the arm for a row class that must not be reported (`npu`): it passes only
            # when nothing was said, so a tool that skipped nothing cannot satisfy it.
            ok = rc == expect_rc and ((want == "no-verdict-at-all" and not lines) or want in verdicts)
            if ok and says is not None:
                ok = any(says in line for line in lines)
            detail = f"verdicts {verdicts}, want {want!r}"
            print(f"  {'ok  ' if ok else 'FAIL'} {name} (rc {rc} want {expect_rc}; {detail})")
            fails += 0 if ok else 1

    def manifest_arm(name, manifest, pins, want_rows, want_unresolved=(), cards=None, want_verdict=None,
                     expect_rc=0):
        """Assert what a MANIFEST resolves to, then (optionally) the verdict those rows earn.

        This is the arm --manifest was missing, and its absence is why the wrong input shape shipped: the shipped
        catalogue names MODELS (`hrx GLM-4.7-Flash-1bit`) and every verdict this tool makes is about a FILE, so
        the manifest path's whole job is the id -> file resolution in between. An arm that asserted only the
        verdict would have passed on a resolution that produced zero rows (there is simply nothing to classify),
        so these assert the resolved (device, model) rows and the unresolved list, and the verdict only on top.
        """
        nonlocal fails, total
        total += 1
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "cards"
            root.mkdir()
            make_card(root, cards or [])
            mpath = pathlib.Path(td) / "onebit-models.txt"
            mpath.write_text(manifest)
            pins_path = pathlib.Path(td) / "pins.json"
            pins_path.write_text(json.dumps({"pins": pins or {}}))
            ours = pin_bytes(pins_path)
            rows, unresolved = rows_from_manifest(read_manifest(mpath), ours)
            ok = sorted(rows) == sorted(want_rows) and sorted(unresolved) == sorted(want_unresolved)
            got = f"rows {rows}, unresolved {unresolved}"
            if ok and want_verdict is not None:
                rc, lines = check(pathlib.Path(td) / "models.tsv", [root], ours, rows)
                verdicts = [line.split()[0] for line in lines]
                ok = rc == expect_rc and want_verdict in verdicts
                got += f" -> verdicts {verdicts} (rc {rc} want {expect_rc})"
            print(f"  {'ok  ' if ok else 'FAIL'} {name} ({got})")
            fails += 0 if ok else 1

    # a card for the model on the quant we ship, and its lock binds the file our pin records
    SHA_A = "a" * 64
    arm("covered on our quant, same bytes", "COVERED",
        [("hrx", "Qwen3-0.6B-Q4_K_M.gguf")],
        [("unsloth-Qwen3-0.6B-loom", "Qwen/Qwen3-0.6B", ["Q4_K_M"], (),
          [("Q4_K_M", SHA_A)])], 0,
        pins={"Qwen/Qwen3-0.6B-GGUF": {"files": {"Qwen3-0.6B-Q4_K_M.gguf": {"oid": SHA_A}}}})
    # the arm this patch exists for: same model, same quant, and the card binds a DIFFERENT FILE
    arm("same model and quant, different bytes", "DIFFERENT-UPLOAD",
        [("hrx", "Qwen3-0.6B-Q4_K_M.gguf")],
        [("unsloth-Qwen3-0.6B-loom", "Qwen/Qwen3-0.6B", ["Q4_K_M"], (),
          [("Q4_K_M", "b" * 64)])], 1,
        pins={"Qwen/Qwen3-0.6B-GGUF": {"files": {"Qwen3-0.6B-Q4_K_M.gguf": {"oid": SHA_A}}}})
    # a card that binds nothing cannot support a coverage claim: no lock is an answer, not a pass
    arm("a card with no lock is not covered", "NO-LOCK",
        [("hrx", "Qwen3-0.6B-Q4_K_M.gguf")],
        [("unsloth-Qwen3-0.6B-loom", "Qwen/Qwen3-0.6B", ["Q4_K_M"], (), None)], 1,
        pins={"Qwen/Qwen3-0.6B-GGUF": {"files": {"Qwen3-0.6B-Q4_K_M.gguf": {"oid": SHA_A}}}})
    # a row we hold no pin for is covered on name+quant and says so, rather than failing for bookkeeping
    arm("a row we hold no pin for is covered-unpinned", "COVERED-UNPINNED",
        [("hrx", "qwen2.5-7b-instruct-q4_k_m-merged.gguf")],
        [("bartowski-Qwen2.5-7B-Instruct-loom", "Qwen/Qwen2.5-7B-Instruct", ["Q4_K_M"], (),
          [("Q4_K_M", SHA_A)])], 0, pins={})
    # engine#3156 review: the map must state how many rows had their BYTES compared, because COVERED-UNPINNED exits 0
    # by design and a map where every row is unpinned otherwise reads as "every shipped row covered". Two rows, no
    # pins at all -> rc 0 still (the rows are covered on name+quant), but the summary must say 0 of 2.
    arm("every row unpinned still prints how many were compared on bytes", "loom_coverage:",
        [("hrx", "qwen2.5-7b-instruct-q4_k_m-merged.gguf"), ("hrx", "Qwen3-0.6B-Q4_K_M.gguf")],
        [("bartowski-Qwen2.5-7B-Instruct-loom", "Qwen/Qwen2.5-7B-Instruct", ["Q4_K_M"], (),
          [("Q4_K_M", SHA_A)]),
         ("unsloth-Qwen3-0.6B-loom", "Qwen/Qwen3-0.6B", ["Q4_K_M"], (), [("Q4_K_M", SHA_A)])], 0,
        pins={}, says="0 of 2 row(s) compared on BYTES")
    # a card for the model on ANOTHER quant is a finding, not covered
    arm("card on another quant is not covered", "REQUANT",
        [("hrx", "Qwen3.6-35B-A3B-Q8_0.gguf")],
        [("ggml-org-Qwen3.6-35B-A3B-loom", "Qwen/Qwen3.6-35B-A3B", ["Q4_K_M"])], 1)
    # the trap: a same-size card for a DIFFERENT model must not count
    arm("a same-size different model does not cover", "UNCOVERED",
        [("hrx", "MiniCPM4-8B-Q4_K_M.gguf")],
        [("someone-MiniCPM5-8B-loom", "Acme/MiniCPM5-8B", ["Q4_K_M"])], 1)
    # and a VL/derived model at the same size does not cover the base row
    arm("a VL sibling does not cover", "UNCOVERED",
        [("hrx", "qwen2.5-7b-instruct-q4_k_m-merged.gguf")],
        [("ggml-org-Qwen2.5-VL-7B-Instruct-loom", "ggml-org/Qwen2.5-VL-7B-Instruct", ["Q4_K_M"])], 1)
    # a card whose DRAFTER carries our quant while the checkpoint does not is REQUANT, not covered: this is exactly
    # Qwen3.6-35B-A3B, whose checkpoint is Q4_K_M and whose Q8_0 is the DFlash drafter (engine#3156 review).
    arm("a drafter's quant does not cover the checkpoint", "REQUANT",
        [("hrx", "Qwen3.6-35B-A3B-Q8_0.gguf")],
        [("ggml-org-Qwen3.6-35B-A3B-loom", "Qwen/Qwen3.6-35B-A3B", ["Q4_K_M"], ["Q8_0"])], 1)
    # no card tree at all is cannot-tell
    arm("no cards is cannot tell", "cannot-tell",
        [("hrx", "Qwen3-0.6B-Q4_K_M.gguf")], [], 2)
    # non-GPU rows are not this map's business
    arm("npu rows are ignored (nothing is reported for them)", "no-verdict-at-all",
        [("npu", "Qwen3-0.6B")], [("unsloth-Qwen3-0.6B-loom", "Qwen/Qwen3-0.6B", ["Q4_K_M"])], 0)
    # ---- the manifest path (--manifest) ----------------------------------------------------------------
    # The shipped catalogue names MODELS (`hrx GLM-4.7-Flash-1bit`) while every verdict this tool makes is about
    # a FILE, so the resolution in between is the only thing that can be wrong invisibly: a resolver that matched
    # nothing produced an empty map, and an empty map reads exactly like a clean one (#3510). These arms assert
    # the resolved rows, not just the verdict, because "no rows" is a state a verdict-only arm cannot see.
    def pins_of(*files, sha=SHA_A):
        return {"1bit-MONSTER/pins": {"files": {f: {"oid": sha} for f in files}}}

    STICK = "mode=live\n# written by os/v3/puppy/catalogue-manifest.py\nhrx GLM-4.7-Flash-1bit\n"
    manifest_arm("manifest: the stick's -1bit suffix is not part of the model name", STICK,
                 pins_of("GLM-4.7-Flash-Q4_K_M.gguf"), [("hrx", "glm-4.7-flash-q4_k_m.gguf")])
    # ...and the resolution has to feed the verdict path, not merely produce rows
    manifest_arm("manifest: a resolved id earns a real verdict (COVERED on bytes)", STICK,
                 pins_of("GLM-4.7-Flash-Q4_K_M.gguf"), [("hrx", "glm-4.7-flash-q4_k_m.gguf")],
                 cards=[("1bit-GLM-4.7-Flash", "1bit-MONSTER/GLM-4.7-Flash", ["Q4_K_M"], (), [("Q4_K_M", SHA_A)])],
                 want_verdict="COVERED", expect_rc=0)
    # same model, same quant, and the card binds a file we do not ship: the manifest path must reach this verdict
    manifest_arm("manifest: a resolved id can still be DIFFERENT-UPLOAD", STICK,
                 pins_of("GLM-4.7-Flash-Q4_K_M.gguf"), [("hrx", "glm-4.7-flash-q4_k_m.gguf")],
                 cards=[("1bit-GLM-4.7-Flash", "1bit-MONSTER/GLM-4.7-Flash", ["Q4_K_M"], (), [("Q4_K_M", "b" * 64)])],
                 want_verdict="DIFFERENT-UPLOAD", expect_rc=1)
    # the OTHER half of #3510, isolated from the suffix: `qwen3-4b` is a PREFIX of `qwen3-4b-instruct-2507`, and a
    # prefix match claims the instruct model's bytes for the base model. The suffix has to be a quant spec, so a
    # bare id resolves to exactly one file here (the suffix arms above cover the `-1bit` half).
    manifest_arm("manifest: an id resolves on stem + quant spec, not on prefix", "hrx Qwen3-4B\n",
                 pins_of("Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
                 [("hrx", "qwen3-4b-q4_k_m.gguf")])
    manifest_arm("manifest: the instruct model resolves to its own file", "hrx Qwen3-4B-Instruct-2507-1bit\n",
                 pins_of("Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
                 [("hrx", "qwen3-4b-instruct-2507-q4_k_m.gguf")])
    # a repo id ends in -GGUF for the same reason: the distribution suffix, not the model name
    manifest_arm("manifest: a repo-style id ending -gguf resolves", "hrx unsloth/Qwen3-4B-GGUF\n",
                 pins_of("Qwen3-4B-Q4_K_M.gguf"), [("hrx", "qwen3-4b-q4_k_m.gguf")])
    # an id we pin nothing for is UNRESOLVED, never a name-matched guess: a coverage claim resting on a name
    # match is the trap the tool exists to avoid, and a silent skip would report a shipped model as absent
    manifest_arm("manifest: an id with no pinned file is reported, not guessed", "hrx ZAYA1-8B-1bit\n",
                 pins_of("Qwen3-4B-Q4_K_M.gguf"), [], want_unresolved=["ZAYA1-8B-1bit"])
    # the NPU route is a different map (docs/npu.md): skipped here, and not even counted as unresolved
    manifest_arm("manifest: npu rows are not this map's business", "npu Qwen3-0.6B-1bit\n",
                 pins_of("Qwen3-0.6B-Q4_K_M.gguf"), [])
    # one catalogue id can own SEVERAL pinned files, and the map answers per FILE -- this is why a shipped row is
    # not one verdict, and why the REQUANT lines in the posted map are the mechanism working
    manifest_arm("manifest: one id owning several pins yields one row per file",
                 # two FILES are two sha256s: pins_of() gives every file the same sha, which a real pin never does and
                 # which the bytes-first rule would rightly read as the card's bytes (engine#3156)
                 "hrx Qwen3-0.6B\n", {"1bit-MONSTER/pins": {"files": {"Qwen3-0.6B-Q4_K_M.gguf": {"oid": SHA_A},
                                                                      "Qwen3-0.6B-Q8_0.gguf": {"oid": "c" * 64}}}},
                 [("hrx", "qwen3-0.6b-q4_k_m.gguf"), ("hrx", "qwen3-0.6b-q8_0.gguf")],
                 cards=[("unsloth-Qwen3-0.6B-loom", "Qwen/Qwen3-0.6B", ["Q4_K_M"], (), [("Q4_K_M", SHA_A)])],
                 want_verdict="REQUANT", expect_rc=1)
    # bytes first (engine#3156, loom-draft): a card from an uploader the VENDOR list does not know still covers the
    # row when its lock binds the sha256 we pin for that row's file, even under a different file name...
    arm("bytes: an unknown uploader's card on our pinned bytes is COVERED", "COVERED",
        [("hrx", "MiniCPM5-1B-Q4_K_M.gguf")],
        [("openbmb-MiniCPM5-1B-loom", "minicpm5-1b", ["Q4_K_M"], (), [("Q4_K_M", SHA_A)])], 0,
        pins=pins_of("MiniCPM5-1B-Q4_K_M.gguf"), says="bytes match")
    # ...but only when the row's bytes are unambiguous: two pins with the same file name and different sha256 do not
    # say which bytes the row ships, so a card on the OTHER pin's bytes must not cover it (#3525 review, cl-flm)
    arm("bytes: a file name pinned twice with different bytes is not a bytes cover", "UNCOVERED",
        [("hrx", "MiniCPM5-1B-Q4_K_M.gguf")],
        [("openbmb-MiniCPM5-1B-loom", "minicpm5-1b", ["Q4_K_M"], (), [("Q4_K_M", "d" * 64)])], 1,
        pins={"vendor/MiniCPM5-1B-GGUF": {"files": {"MiniCPM5-1B-Q4_K_M.gguf": {"oid": SHA_A}}},
              "other/MiniCPM5-1B-GGUF": {"files": {"MiniCPM5-1B-Q4_K_M.gguf": {"oid": "d" * 64}}}})
    # ...and the name rule is unchanged when the bytes differ: the same card on other bytes is not a cover
    arm("bytes: the same unknown-uploader card on OTHER bytes stays UNCOVERED", "UNCOVERED",
        [("hrx", "MiniCPM5-1B-Q4_K_M.gguf")],
        [("openbmb-MiniCPM5-1B-loom", "minicpm5-1b", ["Q4_K_M"], (), [("Q4_K_M", "b" * 64)])], 1,
        pins=pins_of("MiniCPM5-1B-Q4_K_M.gguf"))
    # a manifest line that already names a file is taken as one
    manifest_arm("manifest: a bare .gguf id is used as the file name", "hrx Qwen3-0.6B-Q4_K_M.gguf\n",
                 pins_of("Qwen3-0.6B-Q4_K_M.gguf"), [("hrx", "qwen3-0.6b-q4_k_m.gguf")])
    # parser: mode=, comments, blank lines and junk are not rows (the real file starts with `mode=live`)
    manifest_arm("manifest: mode=, comments and junk lines are not rows",
                 "mode=live\n\n# a comment\n  hrx Qwen3-0.6B  \ngarbage\nhrx\nnpu\n",
                 pins_of("Qwen3-0.6B-Q4_K_M.gguf"), [("hrx", "qwen3-0.6b-q4_k_m.gguf")])
    print(f"loom_coverage_check self-test: {total - fails} ok, {fails} failed")
    return 0 if fails == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default="registry/check_models.tsv")
    ap.add_argument("--cards", action="append", default=None,
                    help="a card tree (repeatable); default ~/amd-oss/loom-fun/{cards,cards-staging}")
    ap.add_argument("--check", action="store_true", help="run the live check (the default)")
    ap.add_argument("--advisory", action="store_true",
                    help="report the map but never fail for it: which models have a card is the STATE of the world, "
                         "not a defect in the PR that happens to be running the gate")
    ap.add_argument("--pins", default="models/pins.json",
                    help="the pins that record the BYTES we ship (models/pins.json); without it the check still "
                         "reports name+quant coverage and says so")
    ap.add_argument("--manifest", default=None,
                    help="a shipped catalogue manifest (os/v3/puppy/catalogue-manifest.py's output: `hrx <id>` "
                         "lines). Rows come from it instead of --registry, and each id is resolved to the files we "
                         "pin. This is what makes the map cover the catalogue the stick ships rather than one "
                         "model per architecture")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    roots = card_roots(a.cards or ["~/amd-oss/loom-fun/cards", "~/amd-oss/loom-fun/cards-staging"])
    ours = pin_bytes(a.pins)
    rows = None
    if a.manifest:
        rows, unresolved = rows_from_manifest(read_manifest(a.manifest), ours)
        if unresolved:
            print(f"loom_coverage: {len(unresolved)} shipped id(s) resolve to no pinned file, so no row could be "
                  f"classified for them: {', '.join(sorted(unresolved)[:5])}"
                  + (" …" if len(unresolved) > 5 else ""))
        if not rows:
            return 2  # nothing resolvable is cannot tell, not a pass
    rc, lines = check(a.registry, roots, ours, rows)
    for line in lines:
        print(line, file=sys.stderr if rc == 2 else sys.stdout)
    if a.advisory and rc == 1:
        # the map is reported, the gate stays green: an uncovered row is a plan item (the issue's second half), and
        # a PR gate that reds for it would block every lane for something no PR can change.
        print("loom_coverage: advisory row -- the map is above, and being uncovered is not this PR's defect")
        return 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
