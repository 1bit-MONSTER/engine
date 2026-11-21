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

QUANTS = re.compile(r"Q\d+_K_[MSL]|Q\d+_\d|IQ\d+_[A-Z]+|BF16|F16|MXFP4|AWQ", re.I)


def read_rows(tsv):
    """(device, model) rows of registry/check_models.tsv."""
    out = []
    for line in pathlib.Path(tsv).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append((parts[0], parts[1]))
    return out


def identity(card_dir):
    """(model id, quants named) from a card's model.json, or (None, set()) when unreadable."""
    f = card_dir / "model.json"
    try:
        text = f.read_text()
        doc = json.loads(text)
    except Exception:
        return None, set()
    model = doc.get("model") if isinstance(doc, dict) else None
    ident = model.get("id") if isinstance(model, dict) else None
    return ident, {q.upper() for q in QUANTS.findall(text)}


NON_DISTINCTIVE = {"loom", "gguf", "merged"}
# Publisher/organisation tokens that appear in card directory names (and HF ids) but are not model identity.
VENDOR = {"ggml", "org", "bartowski", "unsloth", "lmstudio", "lmstudio-community", "model", "ai", "the",
          "someone", "acme", "liquidai", "qwen", "meta-llama", "google", "microsoft"}


def _quant_token(t):
    return bool(re.fullmatch(r"i?q\d+([_-][a-z0-9]+)*|f16|bf16|mxfp4|awq", t))


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


def quant_of(name):
    m = QUANTS.search(name.replace("-", "_"))
    return m.group(0).upper() if m else None


def classify(row, cards):
    """(verdict, detail) for one shipped model against the card list."""
    stem = re.sub(r"\.gguf$", "", row)
    want_q = quant_of(stem)
    hits = []
    for d, ident, quants in cards:
        if not same_model(row, d.name, ident):
            continue
        hits.append((d, ident, quants))
    if not hits:
        return "UNCOVERED", "no card names this model"
    same_quant = [h for h in hits if want_q and want_q in h[2]]
    if same_quant:
        return "COVERED", f"{same_quant[0][0].name} ({want_q})"
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
                cards.append((d, ident, quants))
    return cards


def check(tsv, roots):
    """(rc, lines): rc as documented, lines the report. Returned rather than printed so the self-test can assert
    the VERDICT a row got -- COVERED, UNCOVERED, REQUANT are three different answers and two of them share an rc."""
    cards = cards_of(roots)
    if not cards:
        return 2, ["loom_coverage: no card tree found (pass --cards); the cards are ~GB downloads, so this is "
                   "cannot tell, not a pass"]
    lines, bad = [], 0
    for device, model in read_rows(tsv):
        if device != "hrx":
            continue
        verdict, detail = classify(model, cards)
        lines.append(f"{verdict:9s} {model:46s} {detail}")
        if verdict != "COVERED":
            bad += 1
    return (0 if bad == 0 else 1), lines


def self_test():
    """Hermetic: a synthetic registry and card tree, with the two traps the map has to get right."""
    import tempfile
    fails = 0

    def arm(name, want, rows, cards, expect_rc):
        """Assert the VERDICT, not only the exit code: COVERED (0), UNCOVERED and REQUANT (both 1) are three
        answers and two of them share an rc, so an rc-only arm cannot tell them apart -- which is the one
        distinction the tool exists to make."""
        nonlocal fails
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td) / "cards"
            root.mkdir()
            for cname, ident, quants in cards:
                d = root / cname
                d.mkdir()
                (d / "model.json").write_text(json.dumps({"model": {"id": ident}, "quants": quants}))
            tsv = pathlib.Path(td) / "models.tsv"
            tsv.write_text("\n".join(f"{dev}\t{m}" for dev, m in rows) + "\n")
            rc, lines = check(tsv, [root])
            verdicts = [line.split()[0] for line in lines] if rc != 2 else ["cannot-tell"]
            # "no-verdict-at-all" is the arm for a row class that must not be reported (`npu`): it passes only
            # when nothing was said, so a tool that skipped nothing cannot satisfy it.
            ok = rc == expect_rc and ((want == "no-verdict-at-all" and not lines) or want in verdicts)
            detail = f"verdicts {verdicts}, want {want!r}"
            print(f"  {'ok  ' if ok else 'FAIL'} {name} (rc {rc} want {expect_rc}; {detail})")
            fails += 0 if ok else 1

    # a card for the model on the quant we ship
    arm("covered on our quant", "COVERED",
        [("hrx", "Qwen3-0.6B-Q4_K_M.gguf")],
        [("unsloth-Qwen3-0.6B-loom", "Qwen/Qwen3-0.6B", ["Q4_K_M"])], 0)
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
    # no card tree at all is cannot-tell
    arm("no cards is cannot tell", "cannot-tell",
        [("hrx", "Qwen3-0.6B-Q4_K_M.gguf")], [], 2)
    # non-GPU rows are not this map's business
    arm("npu rows are ignored (nothing is reported for them)", "no-verdict-at-all",
        [("npu", "Qwen3-0.6B")], [("unsloth-Qwen3-0.6B-loom", "Qwen/Qwen3-0.6B", ["Q4_K_M"])], 0)
    print(f"loom_coverage_check self-test: {6 - fails} ok, {fails} failed")
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
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    roots = card_roots(a.cards or ["~/amd-oss/loom-fun/cards", "~/amd-oss/loom-fun/cards-staging"])
    rc, lines = check(a.registry, roots)
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
