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
"""config/loom-cards.json: every onebit model in the shipped catalogue -> its loom-fun Loom card, or none.

The map is DATA, not a route: config/loom-routes.json still decides what is served by loom-serve, and an entry
lands there only after the model passed the gate (engine#3158). This file records which loom-fun card
(https://huggingface.co/loom-fun, Jeremy Fowers and Daniel Holanda) each catalogue model would use, pinned at the
card repository's commit, with the card's licence and the weights' licence kept apart.

How a card is matched (`match`):
  exact       the card's checkpoint file (model.json.lock: repo, revision, sha256) is byte-identical to the file our
              catalogue serves (models/pins.json oid). The card runs our weights unchanged.
  same-model  the card is for the same upstream model, but from another GGUF (other packager or other quant), so
              the bytes differ: a gate row measures the card's bytes, and serving our own file needs the card's
              programs re-pointed at our checkpoint (a config change) and the gate re-run.
  none        no loom-fun card for this model; `nearest` / `gap` say what a port needs (engine#3156).
NPU catalogue entries are listed with match "npu" (they never take the GPU route).

Usage:
  loom_cards.py --build --catalogue server_models.json --cards-meta DIR --hf-list hf-loomfun.json
                --catalogue-hf catalogue-hf.json [--pins models/pins.json] [-o config/loom-cards.json]
      DIR holds <card>.model.json and <card>.lock (model.json.lock) fetched at each card's pinned commit;
      hf-list is the HF API listing of author=loom-fun (id, sha, cardData); catalogue-hf has the weights repos'
      cardData at our pinned commit. Network fetching is done by the caller (see docs/loom-route.md).
  loom_cards.py --check [--map config/loom-cards.json] [--routes config/loom-routes.json] [--pins models/pins.json]
      Offline: shape, pins, licences, and every `exact` claim re-checked against models/pins.json; every
      config/loom-routes.json entry must name a pinned file (sha256 + size).
  loom_cards.py --self-test
"""
import argparse
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MATCHES = ("exact", "same-model", "none", "npu")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")

# Hand-written per-model notes for models with no card: the arch string our registry decodes, the nearest card
# and what a port needs. Kept here (not guessed by the build) so each line is a reviewed claim.
NOTES = {
    "MiniCPM5-1B-1bit": ("llama", "unsloth-Llama-3.2-3B-Instruct-loom",
                         "llama geometry; the chat block is hand-written (BOS <s> + thinking toggle; without them it decodes garbage). Needs no new code: the private v2 package passes the accuracy gate on 24 prompts but fails the speed gate's decode rule (engine#3320: prefill 12.2k vs 6.7k tok/s, TTFT 196 vs 321 ms, but decode 222 vs 240 tok/s at 2k context and a tie at 32 tokens), so a published card (engine#3385) that decodes at least as fast is what a card entry needs"),
    "MiniCPM-V-4.6-1bit": ("minicpmv (qwen3 text + vision tower)", "unsloth-Qwen3-VL-4B-Instruct-loom",
                           "text half = qwen3; needs the MiniCPM-V vision encoder + resampler programs"),
    "LFM2-1.2B-1bit": ("lfm2", "bartowski-LFM3-1.7B-loom",
                       "lfm2 short-conv + GQA blocks; check LFM3 card programs against lfm2 layer schedule"),
    "Qwen3-Embedding-0.6B-1bit": ("qwen3 (embedding)", "unsloth-Qwen3-0.6B-loom",
                                  "same blocks as Qwen3-0.6B; needs an embedding service (last-token pool + norm)"),
    "bge-reranker-v2-m3-1bit": ("bert (xlm-roberta)", None,
                                "encoder-only; no loom-fun encoder card; needs bert blocks + rerank head service"),
    "Qwen3-Reranker-0.6B-1bit": ("qwen3 (reranker)", "unsloth-Qwen3-0.6B-loom",
                                 "same blocks as Qwen3-0.6B; needs a rerank service (yes/no logit score)"),
    "GLM-4.7-Flash-1bit": ("deepseek2 (MLA + MoE)", None,
                           "absorbed-MLA attention + MoE kernels in progress (engine#3221)"),
    "nomic-embed-text-v2-moe-1bit": ("nomic-bert-moe", None, "encoder-only MoE; no encoder card"),
    "jina-reranker-v1-tiny-en-1bit": ("jina-bert", None, "encoder-only (ALiBi); no encoder card"),
    "ZAYA1-8B-1bit": ("zaya", None, "new arch (CCA attention, router, residual scales): engine#3322"),
    "ZAYA1-8B-PM-1bit": ("zaya", None, "same weights as ZAYA1-8B-1bit: engine#3322"),
    "Qwen2.5-Coder-7B-Instruct-1bit": ("qwen2", "bartowski-Qwen2.5-7B-Instruct-loom",
                                       "same qwen2 geometry as Qwen2.5-7B: programs cover it with config only"),
    "Qwen2.5-Coder-1.5B-Instruct-1bit": ("qwen2", "bartowski-Qwen2.5-1.5B-Instruct-loom",
                                         "qwen2 like Qwen2.5-1.5B, but NOT config only: our GGUF has an untied Q6_K output.weight + Q4_K token_embd and the card's programs are generated for tied embeddings; rerun the generator"),
}


def norm_name(repo):
    """'unsloth/Qwen3-8B-GGUF' -> 'qwen3-8b'."""
    n = repo.split("/")[-1].lower()
    n = re.sub(r"-q[0-9].*$", "", n)  # rippertnt/...-Q4_K_M-GGUF
    for suf in ("-gguf",):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def catalogue_models(cat):
    for k, v in cat.items():
        if isinstance(v, dict) and v.get("recipe") == "onebit":
            yield k, v


def our_file(checkpoint, pins):
    """Resolve 'repo:file.gguf' / 'repo:Q4_K_M' / 'repo' to (repo, file, oid) through models/pins.json."""
    repo, _, sel = checkpoint.partition(":")
    files = (pins.get(repo) or {}).get("files") or {}
    ggufs = [f for f in files if f.endswith(".gguf") and "mmproj" not in f and not f.startswith("MTP/")]
    pick = None
    if sel.endswith(".gguf"):
        pick = sel if sel in files else None
    elif sel:
        c = [f for f in ggufs if sel.lower() in f.lower()]
        pick = c[0] if len(c) == 1 else None
    elif len(ggufs) == 1:
        pick = ggufs[0]
    oid = (files.get(pick) or {}).get("oid") if pick else None
    return repo, pick or (sel or None), oid


def card_checkpoint(lock):
    """Main (non-mmproj, non-drafter) checkpoint file of a card lock: (repo, revision, file, sha256)."""
    origins = lock.get("origins") or {}
    for f in lock.get("files") or []:
        o = f.get("origin", "")
        if not o.startswith("@") or not o.endswith(".gguf") or "mmproj" in o:
            continue
        name, _, path = o[1:].partition("/")
        if name in ("drafter", "vision") or os.path.basename(path).startswith(("mtp-", "dspark-")):
            continue
        org = origins.get(name) or {}
        rev = (org.get("resolved") or {}).get("revision") or (org.get("requested") or {}).get("revision")
        return (org.get("requested") or {}).get("repository"), rev, path, f.get("sha256")
    return None, None, None, None


def licence_of(cd):
    if not cd:
        return None
    lic = cd.get("license")
    if lic == "other" and cd.get("license_name"):
        return "other:" + cd["license_name"]
    return lic


def build(cat, cards_meta, hf_list, catalogue_hf, pins):
    cards = {}
    for m in hf_list:
        n = m["id"].split("/", 1)[1]
        lp = os.path.join(cards_meta, n + ".lock")
        if not os.path.exists(lp):
            continue
        lock = json.load(open(lp))
        repo, rev, path, sha = card_checkpoint(lock)
        cards[n] = {"repo": m["id"], "revision": m["sha"], "licence": licence_of(m.get("cardData")),
                    "ckpt_repo": repo, "ckpt_revision": rev, "ckpt_file": path, "ckpt_sha256": sha}
    out = []
    for mid, v in catalogue_models(cat):
        repo, f, oid = our_file(v.get("checkpoint", ""), pins)
        whf = catalogue_hf.get(repo) or {}
        e = {"id": mid, "checkpoint": v.get("checkpoint"), "weights_revision": (pins.get(repo) or {}).get("commit"),
             "weights_file": f, "weights_sha256": oid, "weights_licence": licence_of(whf)}
        if (v.get("recipe_options") or {}).get("onebit_backend") == "npu":
            e["match"] = "npu"
            out.append(e)
            continue
        hit = None
        if oid:
            hit = next((n for n, c in sorted(cards.items()) if c["ckpt_sha256"] == oid), None)
        kind = "exact" if hit else None
        if not hit:
            want = norm_name(repo)
            hits = sorted(n for n, c in cards.items() if c["ckpt_repo"] and norm_name(c["ckpt_repo"]) == want)
            if hits:
                hit, kind = hits[0], "same-model"
        if hit:
            c = cards[hit]
            e.update(match=kind, card={"repo": c["repo"], "revision": c["revision"], "licence": c["licence"],
                                       "checkpoint": {"repo": c["ckpt_repo"], "revision": c["ckpt_revision"],
                                                      "file": c["ckpt_file"], "sha256": c["ckpt_sha256"]}})
        else:
            e["match"] = "none"
            note = NOTES.get(mid)
            if note:
                e["arch"], near, e["gap"] = note
                if near:
                    c = cards.get(near)
                    e["nearest"] = {"repo": c["repo"], "revision": c["revision"]} if c else {"repo": "loom-fun/" + near}
        out.append(e)
    return {
        "about": ("Each onebit model in the shipped Lemonade catalogue -> its loom-fun Loom card (or none), pinned at "
                  "the card repo's commit. Data only: config/loom-routes.json decides routing, after the gate "
                  "(engine#3158). Cards: https://huggingface.co/loom-fun by Jeremy Fowers and Daniel Holanda; card "
                  "licence (the Loom programs) and weights licence are separate fields. Built and checked by "
                  "tools/loom_cards.py."),
        "models": out,
    }


def check(doc, pins):
    errs = []
    if not isinstance(doc.get("models"), list):
        return ["models: not a list"]
    seen = set()
    for e in doc["models"]:
        mid = e.get("id")
        where = "model %r" % mid
        if not mid or mid in seen:
            errs.append(where + ": missing or duplicate id")
        seen.add(mid)
        m = e.get("match")
        if m not in MATCHES:
            errs.append(where + ": match %r not one of %s" % (m, MATCHES))
            continue
        if m in ("exact", "same-model"):
            c = e.get("card") or {}
            if not str(c.get("repo", "")).startswith("loom-fun/"):
                errs.append(where + ": card.repo must be a loom-fun repo")
            if not SHA40.match(str(c.get("revision", ""))):
                errs.append(where + ": card.revision must be a 40-hex commit")
            if not c.get("licence"):
                errs.append(where + ": card.licence missing")
            ck = c.get("checkpoint") or {}
            if not SHA64.match(str(ck.get("sha256", ""))):
                errs.append(where + ": card.checkpoint.sha256 must be 64-hex")
            if m == "exact":
                repo = (e.get("checkpoint") or "").split(":")[0]
                oid = (((pins.get(repo) or {}).get("files") or {}).get(e.get("weights_file") or "") or {}).get("oid")
                if not oid:
                    errs.append(where + ": exact but our weights file is not pinned in models/pins.json")
                elif oid != ck.get("sha256"):
                    errs.append(where + ": exact but card checkpoint sha256 != our pinned oid")
                if e.get("weights_sha256") != oid:
                    errs.append(where + ": weights_sha256 disagrees with models/pins.json")
        if m == "none" and "card" in e:
            errs.append(where + ": match none but a card is given")
        if m != "npu" and not e.get("weights_licence"):
            errs.append(where + ": weights_licence missing (record it per model, never from a sibling)")
    return errs


def check_routes(routes_doc, pins):
    """config/loom-routes.json: every entry's gguf_sha256 + gguf_size must be a file models/pins.json pins (a route
    names only bytes we ship; a gate row on another upload measures a model we do not serve, engine#3158)."""
    errs = []
    pinned = {}
    for repo, p in pins.items():
        for fname, f in (p.get("files") or {}).items():
            if f.get("oid"):
                pinned[f["oid"]] = (repo, fname, f.get("size"))
    for i, r in enumerate(routes_doc.get("routes", [])):
        m = r.get("match") or {}
        sha, size = m.get("gguf_sha256", ""), m.get("gguf_size")
        where = "route %r" % r.get("id", i)
        if not sha:
            errs.append(where + ": no match.gguf_sha256 (a shipped route must name the exact checkpoint)")
        elif sha.lower() not in pinned:
            errs.append(where + ": gguf_sha256 is not a file models/pins.json pins")
        elif pinned[sha.lower()][2] is not None and pinned[sha.lower()][2] != size:
            errs.append(where + ": gguf_size %s != the pinned size %s" % (size, pinned[sha.lower()][2]))
    return errs


def self_test():
    sha_a, sha_b = "a" * 64, "b" * 64
    pins = {"o/M-GGUF": {"commit": "1" * 40, "files": {"M-Q4_K_M.gguf": {"oid": sha_a}}},
            "o/N-GGUF": {"commit": "2" * 40, "files": {"N-Q4_K_M.gguf": {"oid": sha_b}}}}
    cat = {"M-1bit": {"recipe": "onebit", "checkpoint": "o/M-GGUF:M-Q4_K_M.gguf"},
           "N-1bit": {"recipe": "onebit", "checkpoint": "o/N-GGUF:Q4_K_M"},
           "Z-1bit": {"recipe": "onebit", "checkpoint": "o/Z-GGUF"},
           "P-1bit": {"recipe": "onebit", "checkpoint": "o/P", "recipe_options": {"onebit_backend": "npu"}},
           "other": {"recipe": "llamacpp", "checkpoint": "x/y"}}
    hf = [{"id": "loom-fun/o-M-loom", "sha": "c" * 40, "cardData": {"license": "mit"}},
          {"id": "loom-fun/p-N-loom", "sha": "d" * 40, "cardData": {"license": "mit"}}]
    locks = {"o-M-loom": {"origins": {"checkpoint": {"requested": {"repository": "o/M-GGUF", "revision": "1" * 40}}},
                          "files": [{"origin": "@checkpoint/mmproj-F16.gguf", "sha256": "e" * 64},
                                    {"origin": "@checkpoint/M-Q4_K_M.gguf", "sha256": sha_a},
                                    {"origin": "@drafter/D.gguf", "sha256": "f" * 64}]},
             "p-N-loom": {"origins": {"checkpoint": {"requested": {"repository": "p/N-GGUF", "revision": "3" * 40}}},
                          "files": [{"origin": "@checkpoint/N-Q4_K_M.gguf", "sha256": "9" * 64}]}}
    chf = {r: {"license": "apache-2.0"} for r in ("o/M-GGUF", "o/N-GGUF", "o/Z-GGUF")}
    fails = []

    def ok(cond, what):
        if not cond:
            fails.append(what)

    with tempfile.TemporaryDirectory() as d:
        for n, l in locks.items():
            json.dump(l, open(os.path.join(d, n + ".lock"), "w"))
        doc = build(cat, d, hf, chf, pins)
    by = {e["id"]: e for e in doc["models"]}
    ok(set(by) == {"M-1bit", "N-1bit", "Z-1bit", "P-1bit"}, "only onebit entries are mapped")
    ok(by["M-1bit"]["match"] == "exact", "byte-identical checkpoint -> exact (mmproj/drafter skipped)")
    ok(by["N-1bit"]["match"] == "same-model", "same model, other packager, other bytes -> same-model")
    ok(by["N-1bit"]["weights_file"] == "N-Q4_K_M.gguf", "quant tag resolves through pins")
    ok(by["Z-1bit"]["match"] == "none" and "card" not in by["Z-1bit"], "no card -> none")
    ok(by["P-1bit"]["match"] == "npu", "npu backend -> npu")
    ok(check(doc, pins) == [], "built doc passes check: %s" % check(doc, pins))
    # mutations the check must catch
    import copy
    bad = copy.deepcopy(doc)
    [e for e in bad["models"] if e["id"] == "M-1bit"][0]["card"]["checkpoint"]["sha256"] = sha_b
    ok(any("!= our pinned oid" in x for x in check(bad, pins)), "exact with other bytes is caught")
    bad = copy.deepcopy(doc)
    [e for e in bad["models"] if e["id"] == "N-1bit"][0]["card"]["revision"] = "main"
    ok(any("40-hex" in x for x in check(bad, pins)), "moving card revision is caught")
    bad = copy.deepcopy(doc)
    [e for e in bad["models"] if e["id"] == "Z-1bit"][0]["weights_licence"] = None
    ok(any("weights_licence" in x for x in check(bad, pins)), "missing weights licence is caught")
    bad = copy.deepcopy(doc)
    bad["models"].append(copy.deepcopy(bad["models"][0]))
    ok(any("duplicate" in x for x in check(bad, pins)), "duplicate id is caught")
    bad = copy.deepcopy(doc)
    bad["models"][0]["match"] = "maybe"
    ok(any("not one of" in x for x in check(bad, pins)), "unknown match kind is caught")
    ok(norm_name("rippertnt/HyperCLOVAX-SEED-Text-Instruct-1.5B-Q4_K_M-GGUF") == "hyperclovax-seed-text-instruct-1.5b",
       "norm_name drops quant suffix")
    rt = {"routes": [{"id": "ok", "match": {"gguf_sha256": sha_a, "gguf_size": 7}}]}
    pins_sz = copy.deepcopy(pins)
    pins_sz["o/M-GGUF"]["files"]["M-Q4_K_M.gguf"]["size"] = 7
    ok(check_routes(rt, pins_sz) == [], "a route on a pinned file (sha + size) passes")
    ok(any("not a file" in x for x in check_routes({"routes": [{"id": "x", "match": {"gguf_sha256": "c" * 64, "gguf_size": 7}}]}, pins_sz)),
       "a route on an unpinned upload is caught")
    ok(any("gguf_size" in x for x in check_routes({"routes": [{"id": "x", "match": {"gguf_sha256": sha_a, "gguf_size": 8}}]}, pins_sz)),
       "a route whose size disagrees with the pin is caught")
    ok(any("no match.gguf_sha256" in x for x in check_routes({"routes": [{"id": "x", "match": {"model": ["m"]}}]}, pins_sz)),
       "a route without a sha256 is caught")
    for f in fails:
        print("FAIL:", f)
    print("loom_cards self-test: %d checks, %d failed" % (21, len(fails)))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--catalogue")
    ap.add_argument("--cards-meta")
    ap.add_argument("--hf-list")
    ap.add_argument("--catalogue-hf")
    ap.add_argument("--pins", default=os.path.join(ROOT, "models", "pins.json"))
    ap.add_argument("--map", default=os.path.join(ROOT, "config", "loom-cards.json"))
    ap.add_argument("--routes", default=os.path.join(ROOT, "config", "loom-routes.json"))
    ap.add_argument("-o", "--output")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    pins = json.load(open(a.pins))["pins"]
    if a.build:
        for k in ("catalogue", "cards_meta", "hf_list", "catalogue_hf"):
            if not getattr(a, k):
                ap.error("--build needs --" + k.replace("_", "-"))
        doc = build(json.load(open(a.catalogue)), a.cards_meta, json.load(open(a.hf_list)),
                    json.load(open(a.catalogue_hf)), pins)
        errs = check(doc, pins)
        out = a.output or a.map
        with open(out, "w") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        for x in errs:
            print("check:", x, file=sys.stderr)
        print("wrote %s (%d models, %d check errors)" % (out, len(doc["models"]), len(errs)))
        return 1 if errs else 0
    if a.check:
        errs = check(json.load(open(a.map)), pins)
        errs += ["loom-routes: " + x for x in check_routes(json.load(open(a.routes)), pins)]
        for x in errs:
            print("loom-cards:", x, file=sys.stderr)
        print("loom-cards check: %d errors" % len(errs))
        return 1 if errs else 0
    ap.error("one of --build, --check, --self-test")


if __name__ == "__main__":
    sys.exit(main())
