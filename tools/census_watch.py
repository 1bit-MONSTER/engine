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
"""Watch HF for new text-generation models the registry does not map.

`tools/census.py` is a snapshot: it sweeps every page and reports coverage. New models
land on HF every day, and a new architecture class is exactly what moves the mapped
share without anyone noticing. This watcher polls the newest text-generation models,
reads each one's architecture class from its config (inline in the listing, so there is
no per-model fetch), and compares it against `registry/architectures.json` — the same
read `tools/census.py`'s coverage() makes, so "uncovered" here means "not in the number
the census reports".

The port drops the old C++ probe (1bit-MONSTER's `Testing/census_coverage.py` compiled
`rcpp_arch_from_string` with g++): this repository's registry is generated data
(`tools/registry_build.py`), so the class is a plain lookup. There is no `strip_arch`
either — `architectures.json` is keyed by the HF architecture class, which is the key
the census counts.

usage: tools/census_watch.py [--limit N] [--state PATH] [--full]

Exit codes (the alert contract):
  0  no uncovered class among the newest models that warrants an alert, and no new
     FastFlowLM repo whose family this engine cannot serve
  1  an uncovered class arrived (the alert), or a new FastFlowLM repo arrived whose
     architecture `registry/architectures.json` does not map, or a new one arrived that
     publishes a Q4NX pack for a family the NPU route does not serve, or a pinned
     supporting upstream (`registry/support_pins.json`) moved, or a d3 weight repo
     appeared — any one is enough
  2  runtime failure (no network, unreadable registry, an unreadable FastFlowLM
     listing or releases page, ...)

One watch never silences the other. Both run in the same process, the architecture walk
is reported and remembered before the FastFlowLM watch starts, and the exit code is the
OR of the two alerts: an uncovered class is not turned green by a quiet FastFlowLM day,
and a brand-new FastFlowLM repo is not turned green by a covered architecture walk. A
FastFlowLM fetch that fails is a runtime failure (2), because "the listing did not load"
must never be printed as "nothing new" — but when an alert is also present the alert wins
(1), since that is the code a human has to act on, and the failure is still printed as
`!! FLM WATCH FAILED` for the log grep.

A class whose models all load through custom modeling code (`auto_map` in the config, a
runtime library that is not Transformers, or a repo with no library at all that ships its own
`modeling_*.py`; in each case Transformers itself has no implementation of the class) and
that fewer than REMOTE_CODE_SPREAD uploaders publish is reported but does not alert: research one-offs like that arrive every day and are not something a backend
could map. Once REMOTE_CODE_SPREAD uploaders use it, or it is a reviewed class
(registry/significant.json), it alerts like any other. The exception is a reviewed class
whose entry says `"watch": "report"`: a research architecture reviewed as not an alias that
nothing upstream implements, so a new upload of it needs no action. It is reported, not alerted.

The FastFlowLM watch (engine#588) reads their HF org and their runtime releases, because
that is where a model family arrives that this engine may have to serve on the NPU. One
listing request covers the whole org: `expand[]` puts `lastModified`, the tags (license,
`base_model`) and the config (architecture / model_type) in the listing itself, so a repo
is classified without a per-model fetch — the same inline-config design as the walk above.
The list route carries sibling *names* only (there is no size field in any `expand[]`
option), so each `*.q4nx` size is read from the repo detail (`?blobs=true`) once, and only
for a repo that is new or changed. `--full` reprints every repo and release instead of the
delta (and so re-reads every size); a daily run prints only what moved.

The npu-route column is answered from the registry, not from the repo name: a family is
served by the NPU route when its `architectures.json` entry lists `npu` in `backends`, and
the line then prints that entry's `npu_model_type`. There is no feature flag to read off a
repo — `lax` and `layerP` name things inside this repository, and neither appears in the
FastFlowLM org listing, in the ten newest release bodies, or in any asset name (checked
2026-10-26), so a marker guessed from a repo name would be invention. The report names
FastFlowLM files, sizes, licenses and tags only — nothing about our NPU kernels or their
speed.

A gap that has been answered stays answered (engine#2171). CI keeps no state between runs, so
every scheduled run sees the whole org as new, and each `!! NPU FLM` / `!! NEW FLM` family that
docs/npu.md had already answered alerted again every day, re-filing the census issue as soon as
the last one closed. `registry/flm_answers.json` is the machine-readable form of those answers:
a pack whose declared architecture (or, for a config-less repo, whose id) has an entry prints
`. NPU FLM` / `. FLM` with the issue that tracks it and does not alert. An answer only quiets a
gap: once the registry serves the family on the NPU route the line is `+ FLM` again, and the
test fails on a committed answer for a family that is no longer a gap, so a stale answer is
removed rather than kept. A family with no entry still alerts on its first sighting.

State (the model ids already classified) lives outside the checkout, at
$CENSUS_WATCH_STATE (default ~/.1bit/census-watch-state.json), so a run never dirties
the tree it reads. A model whose config is missing (a gated repo) is reported the first
time it is seen and then remembered, so a routine gated upload never fails a run twice —
only genuinely uncovered classes are the alert.

The daily systemd timer on the development box (scripts/1bit-census-watch.{service,timer})
runs scripts/census-watch.sh, which wraps this.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "registry")
API = "https://huggingface.co/api/models"
DEFAULT_LIMIT = 120
DEFAULT_STATE = os.path.join(os.path.expanduser("~"), ".1bit", "census-watch-state.json")
MAX_SEEN = 5000  # cap state growth; oldest dropped
REMOTE_CODE_SPREAD = 3  # uploaders before a custom-code class alerts
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')

# FastFlowLM's HF org and runtime releases (engine#588). The org is small (about 60 repos),
# so one page of 100 covers it; the releases page is read newest-first.
FLM_AUTHOR = "FastFlowLM"
FLM_LIMIT = 100
GH_RELEASES = "https://api.github.com/repos/FastFlowLM/FastFlowLM/releases?per_page={limit}"

# Supporting upstreams we depend on but do not vendor (registry/support_pins.json): the Q4NX
# converter the repack path runs, and the d3 source behind the decision packs. Pinned by
# commit, because CI carries no state - "the commit differs from the reviewed one" is a stable
# fact, while "a new commit appeared" would be true on every run.
SUPPORT_PINS_PATH = os.path.join(ROOT, "registry", "support_pins.json")
GH_REPO_HEAD = "https://api.github.com/repos/{repo}/commits?per_page=1"
# A pin may name the paths the NPU route depends on (`paths`). Then "moved" means a commit that
# touches one of them, not any commit: vllm-project/semantic-router lands dozens of router-side
# commits a day, so the whole-repo head moved on every run and re-filed the census issue daily
# while the d3 contract it is pinned for stayed byte-identical (engine#2171). The last commit
# touching a path is read at the pin and at the head; a difference is a move.
GH_PATH_HEAD = "https://api.github.com/repos/{repo}/commits?sha={sha}&path={path}&per_page=1"
# Recorded answers for FastFlowLM packs this engine does not serve (engine#2171).
FLM_ANSWERS_PATH = os.path.join(ROOT, "registry", "flm_answers.json")
HF_ORG_URL = "https://huggingface.co/api/models?author={org}&sort=createdAt&direction=-1&limit=100"
GH_RELEASES_LIMIT = 20
# Words that mean a release notes body or an asset name talks about the Q4NX container:
# npu/q4nx.h pins that layout behind a chunk table, so a release that mentions any of
# these is flagged for a human to read before the pinned version moves.
KERNEL_FORMAT_MARKERS = ("q4nx", "chunk", "layout")
# A published `model_type` names its modality (`gemma3_text`, `qwen3_5_text`); the registry
# class does not, so these trailing tokens are dropped before the name is compared.
MODALITY_TOKENS = frozenset(("text", "vl", "vision", "audio", "image", "video", "it",
                             "instruct", "chat", "base", "mm", "multimodal"))
FLM_ORG_URL = (f"{API}?author={FLM_AUTHOR}&sort=lastModified&direction=-1&limit={FLM_LIMIT}"
               "&expand[]=lastModified&expand[]=tags&expand[]=config&expand[]=siblings")
# Gated (auto-access) repos answer the listing with no `config`, so their architecture stays
# unknowable and the class is never seen - the blind spot behind engine#619's five
# UNVERIFIABLE rows. Both names are what huggingface_hub and the `hf` CLI read; with no token
# in the environment the request is byte-identical to what this watcher always sent, and an
# invalid one only costs a 401 the retry loop already tolerates.
TOKEN_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def hf_token():
    """The token to send, or '' - read per call so a test can set it."""
    for var in TOKEN_VARS:
        token = os.environ.get(var, "").strip()
        if token:
            return token
    return ""


_AUTH_REJECTED = False  # a 401 on the token turns it off for the rest of the run


def request_headers():
    """Headers for every HF request: the UA, plus Authorization when a token is present."""
    headers = {"User-Agent": "1bit-census-watch"}
    token = hf_token()
    if token and not _AUTH_REJECTED:
        headers["Authorization"] = "Bearer " + token
    return headers

# A derivative carries no config.json of its own by design (a quantized GGUF, a LoRA
# adapter); the raw release it derives from carries the config and is what gets checked.
DERIVATIVE_TAGS = ("gguf", "lora", "peft")

# Library names that are not Transformers. A model published through one of these (a raw
# PyTorch checkpoint, an MLX or ONNX export) has no Transformers implementation either, so
# its `architectures[0]` is the same kind of unmappable one-off as an `auto_map` class.
# engine#296: Ihatetomatoes/nanoBeard-sloop-14M is `library_name: pytorch`, declares
# `model_type: nanobeard-gpt` and `architectures: ["GPT"]`, and no backend has a "GPT" arch.
NON_TRANSFORMERS_LIBRARIES = frozenset((
    "pytorch", "onnx", "onnxruntime-genai", "mlx", "mlx-serve", "litert-lm", "llama.cpp",
    "mlc-llm", "tensorflow", "tf-keras", "keras", "jax", "flax", "openvino", "coreml",
))

# A repo's own modeling code, by the name Transformers gives such files (`modeling_<name>.py`),
# at the top level of the repo. engine#1922: Climb-Ai/Climb-1-Toki declares
# `architectures: ["MicroLlamaForCausalLM"]` and `model_type: llama` with no `auto_map` and no
# library, and ships `modeling_micro_llama.py`, where that class is a bare `torch.nn.Module`
# with its own tensor names (`layers.0.attn.q_proj`, `token_embd`). It alerted as a new class.
_OWN_MODELING = re.compile(r"^modeling_[A-Za-z0-9_]+\.py$")

COVERED, UNCOVERED, UNVERIFIABLE, DERIVATIVE = "covered", "uncovered", "unverifiable", "derivative"


def hf_get_page(url, tries=4):
    """GET one listing page: (models, the rel="next" cursor or "").

    HF ignores `p=` when the listing is sorted (every page returns the first one), so
    pagination follows the Link header's cursor — the same way tools/census.py walks the
    full sweep. A page fetched with `p=` overlaps the previous one 100%, which silently
    capped this watcher at 100 models however large --limit was.
    """
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=request_headers())
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r), r.headers.get("Link", "")
        except urllib.error.HTTPError as e:
            # A rejected token must not cost the whole run: fall back to anonymous and say so.
            # A gated repo then answers with no config again, which is what UNVERIFIABLE reports.
            if e.code == 401 and "Authorization" in req.headers:
                global _AUTH_REJECTED
                _AUTH_REJECTED = True
                print(f"hf_get_page {url}: the token was rejected (401); continuing without it",
                      file=sys.stderr)
                continue
            if t == tries - 1:
                print(f"hf_get_page {url}: {e}", file=sys.stderr)
                return [], ""
            time.sleep(2 * (t + 1))
        except Exception as e:
            if t == tries - 1:
                print(f"hf_get_page {url}: {e}", file=sys.stderr)
                return [], ""
            time.sleep(2 * (t + 1))
    return [], ""


def load_registry():
    """The committed registry and the reviewed not-an-alias list (docs/arch-gaps.md)."""
    archs = json.load(open(os.path.join(REG, "architectures.json")))["architectures"]
    sig = json.load(open(os.path.join(REG, "significant.json")))["classes"]
    return archs, sig


def architectures_of(model):
    """The config's architecture class, or '' when the listing carried no config."""
    cfg = model.get("config") or {}
    archs = cfg.get("architectures") or []
    return archs[0] if archs and isinstance(archs[0], str) else ""


def is_remote_code(model):
    """True when Transformers has no implementation: `auto_map`, or not a Transformers model.

    A raw checkpoint through a non-Transformers library carries an architecture class no
    backend can map (the library's own code runs it), exactly like an `auto_map` model.
    A missing `library_name` is unknown, not custom, so it alerts - unless the repo ships its
    own `modeling_*.py` and nothing calls it a Transformers model: then the class is whatever
    that file defines, and without `auto_map` Transformers cannot even be told to load it.
    The Hub puts `library_name: transformers` and the `transformers` tag on a repo it can
    load with the library, so a new class Transformers really has keeps alerting.
    """
    cfg = model.get("config") or {}
    if cfg.get("auto_map"):
        return True
    transformers = "transformers" in (model.get("tags") or [])
    if model.get("library_name") in NON_TRANSFORMERS_LIBRARIES:
        return not transformers
    if not model.get("library_name") and not transformers:
        return any(_OWN_MODELING.match((sib or {}).get("rfilename") or "")
                   for sib in model.get("siblings") or ())
    return False


def is_derivative(model):
    tags = model.get("tags") or []
    return (model.get("library_name") == "peft"
            or any(t in tags for t in DERIVATIVE_TAGS))


def upstream_classes(text):
    """The HF architectures a NEWER llama.cpp would map, from its `convert_hf_to_gguf.py`.

    engine#1922: the digest says `!! UNCOVERED` for any class our registry does not carry, but the registry is GENERATED from the
    pinned sources, so that label alone cannot tell "our pin is behind" from "nobody upstream knows this class either". The first
    needs a pin bump (the registry regenerates for free); the second needs nothing from us. Both looked identical to every reader,
    and settling one by hand took two commands against a raw.githubusercontent URL.

    `text` is the converter's source; each `@Model.register("XForCausalLM")` names a class llama.cpp can convert. A class the
    converter names but the registry lacks is pin-behind; anything else is upstream-only. Pure, so the classification is testable
    without a network call -- which matters, because the wrong answer here sends a lane to bump a pin for nothing, or to ignore a
    real gap.
    """
    out = set()
    src = text or ""
    for m in re.finditer(r"@Model\.register\(", src):
        # Scan to the matching close paren rather than matching one quoted name: the converter registers SEVERAL classes
        # per decorator (`@Model.register("AForCausalLM", "B")`), and a regex that takes only the first would label the
        # second as upstream-only -- the wrong answer in the direction that tells a lane to ignore a real gap.
        depth, i, end = 0, m.end() - 1, len(src)
        while i < end:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        for name in re.findall(r'"([A-Za-z0-9_]+)"', src[m.end():i]):
            out.add(name)
    return out


def classify_uncovered(classes, upstream):
    """Which uncovered classes a pin bump would fix, and which nobody has (engine#1922).

    `upstream` is a set from upstream_classes(), or None when the fetch failed -- in which case every class is reported as
    `unknown`, not guessed as either. A digest that cannot tell must say so rather than pick the flattering label.
    """
    verdict = {}
    for cls in classes:
        if upstream is None:
            verdict[cls] = "unknown"
        elif cls in upstream:
            verdict[cls] = "pin-behind"
        else:
            verdict[cls] = "upstream"
    return verdict


def classify(models, archs, sig, previously_seen=()):
    """Classify a batch of HF model records against the registry.

    Pure — no network, no disk — so tests/census_watch_test.py can drive it. A model is
    covered when its architecture class is a registry key with at least one backend;
    uncovered when it is not, which is the alert. A model with no config is unverifiable
    (gated repos need a token) unless it is a derivative, which is expected and silent.

    `previously_seen` suppresses repeat reporting of unverifiable models.
    """
    out = {COVERED: [], UNCOVERED: [], UNVERIFIABLE: [], DERIVATIVE: []}
    reasons = {}  # model id -> why it could not be classified
    classes = {}  # uncovered class -> [model ids]
    native = set()  # uncovered classes with at least one model that needs no custom code
    for m in models:
        mid = m.get("id", "")
        arch = architectures_of(m)
        if not arch:
            if is_derivative(m):
                out[DERIVATIVE].append(mid)
                continue
            if mid not in previously_seen:
                out[UNVERIFIABLE].append(mid)
                reasons[mid] = "no config / no architecture"
            continue
        entry = archs.get(arch)
        if entry and entry.get("backends"):
            out[COVERED].append(mid)
            continue
        out[UNCOVERED].append(mid)
        classes.setdefault(arch, []).append(mid)
        if not is_remote_code(m):
            native.add(arch)
    out["classes"] = classes
    out["reasons"] = reasons
    # reviewed classes marked report-only (`"watch": "report"`): reported, no alert
    out["reviewed_report"] = sorted(c for c in classes if c in sig and sig[c].get("watch") == "report")
    out["significant"] = sorted(c for c in classes if c in sig and c not in out["reviewed_report"])
    # custom-code classes from fewer than REMOTE_CODE_SPREAD uploaders: reported, no alert
    out["remote_code"] = sorted(
        c for c, ids in classes.items()
        if c not in sig and c not in native and len({i.split("/")[0] for i in ids}) < REMOTE_CODE_SPREAD)
    return out


def sweep_newest(limit, seen, tag="text-generation", pages_guard=20, fetch=None):
    """The newest `limit` models whose ids are not in `seen`, newest first.

    Walks the createdAt-descending listing by its Link cursor until it has `limit` unseen
    models, the cursor runs out, or a whole page is already known (the horizon: everything
    older has been classified). `fetch(url) -> (models, next_cursor)` is injectable so
    tests can drive the walk offline.
    """
    fetch = fetch or hf_get_page
    fresh, batch_ids = [], set()
    # full=true: the listing then carries each repo's file names (`siblings`), which is how
    # is_remote_code() sees a repo's own modeling code; config=true alone leaves them out
    url = f"{API}?pipeline_tag={tag}&sort=createdAt&direction=-1&limit=100&config=true&full=true"
    page = 0
    while url and len(fresh) < limit and page < pages_guard:
        batch, link = fetch(url)
        if not batch:
            break
        for m in batch:
            mid = m.get("id")
            if mid and mid not in seen and mid not in batch_ids:
                batch_ids.add(mid)
                fresh.append(m)
        if all(m.get("id") in seen for m in batch):
            break
        nxt = _NEXT.search(link or "")
        url = nxt.group(1) if nxt else None
        page += 1
        time.sleep(0.3)
    return fresh[:limit]


def load_state(path):
    try:
        with open(path) as f:
            state = json.load(f)
    except Exception:
        return {"last_run": None, "seen": {}, "flm_models": {}, "flm_tags": []}
    state.setdefault("seen", {})
    state.setdefault("flm_models", {})
    state.setdefault("flm_tags", [])
    return state


def save_state(path, state):
    seen = state["seen"]
    if len(seen) > MAX_SEEN:
        state["seen"] = dict(list(seen.items())[-MAX_SEEN:])
    flm_models = state.get("flm_models") or {}
    if len(flm_models) > MAX_SEEN:
        state["flm_models"] = dict(list(flm_models.items())[-MAX_SEEN:])
    tags = state.get("flm_tags") or []
    if len(tags) > MAX_SEEN:
        state["flm_tags"] = tags[-MAX_SEEN:]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)
        f.write("\n")


def remember(state, res):
    """Record what was classified, so the next walk moves past it."""
    for group in (COVERED, UNCOVERED, UNVERIFIABLE, DERIVATIVE):
        for mid in res[group]:
            state["seen"][mid] = group
    return state


UPSTREAM_CONVERTER_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py"


def get_text(url, tries=3):
    """The raw body of `url`, or raises. Separate from get_json because the converter is Python source, not JSON."""
    last = None
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=request_headers()), timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:      # noqa: BLE001 - the caller decides; this helper only reports
            last = e
            time.sleep(1)
    raise last


def get_json(url, tries=3):
    """GET one JSON document: an org listing, a repo detail, a releases page.

    Raises RuntimeError when it cannot be read, so the caller can tell "nothing new" from
    "the watch did not run". A failed FastFlowLM fetch must never be printed as a quiet
    day: that is how a new family would slip past the alert.
    """
    last = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "1bit-census-watch"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            last = e
            if t + 1 < tries:
                time.sleep(2 * (t + 1))
    raise RuntimeError(f"{url}: {last}")


def license_of(tags):
    """The `license:` tag HF derives from the model card (e.g. `license:apache-2.0`)."""
    for tag in tags or ():
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return ""


def base_models_of(tags):
    """The `base_model:` tags, minus the relation prefix (`quantized:`, `finetune:`, ...)."""
    out = []
    for tag in tags or ():
        if not tag.startswith("base_model:"):
            continue
        name = tag.split(":", 1)[1]
        for rel in ("finetune:", "quantized:", "adapter:", "merge:"):
            if name.startswith(rel):
                name = name[len(rel):]
        if name.strip() and name.strip() not in out:
            out.append(name.strip())
    return out


def q4nx_files(siblings):
    """The `*.q4nx` siblings of a repo, with the byte size we could read (0 = unknown).

    The listing route carries `rfilename` only, the repo detail (`?blobs=true`) carries
    `size`/`lfs.size`; both shapes end up here.
    """
    out = []
    for sib in siblings or ():
        sib = sib or {}
        name = sib.get("rfilename") or sib.get("name") or ""
        if not name.lower().endswith(".q4nx"):
            continue
        size = sib.get("size") or (sib.get("lfs") or {}).get("size") or 0
        out.append({"name": name, "size": int(size or 0)})
    return out


def flm_architecture(record):
    """The architecture class the registry is keyed by, else the config's model_type.

    The D3-* repos publish no `architectures` and only a `model_type` (`qwen3_5_text`),
    so the second half is not decoration: without it every D3 repo reads as unmapped.
    """
    config = (record or {}).get("config") or {}
    archs = config.get("architectures") or []
    if archs:
        return str(archs[0])
    return str(config.get("model_type") or "")


def normalize_family(name):
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def family_stem(name):
    """A published family name with its modality suffix removed (`qwen3_5_text` -> `qwen3_5`).

    HF `model_type` strings carry the modality (`gemma3_text`, `qwen3_5_text`) while the
    registry is keyed by the class (`Gemma3ForCausalLM`), so the suffix has to go before
    the two can be compared. Only whole `_`-separated tokens are dropped.
    """
    parts = str(name or "").split("_")
    while len(parts) > 1 and parts[-1].lower() in MODALITY_TOKENS:
        parts.pop()
    return "_".join(parts)


def family_tokens(name):
    """`Qwen3_5ForCausalLM` / `qwen3_5_text` -> `[qwen3, 5, for, causal, lm]` / `[qwen3, 5, text]`.

    A camel-case bump and any punctuation both start a new token, so a family name can be
    matched token by token against a class instead of character by character (`deepseek_v3`
    must not match `DeepseekV32ForCausalLM` through the shared `deepseekv3` prefix).
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name or ""))
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def resolve_family(name, archs):
    """The `architectures.json` entry a published family name refers to, or None.

    An exact architecture class is the answer; otherwise the name (modality suffix
    stripped, punctuation dropped) is compared with each entry's `gguf` /
    `npu_model_type` (`qwen3_5_text` -> `qwen3_5` -> `qwen35`, which is the entry's gguf
    string), and only then token by token with the class name. Ties prefer a class whose
    own name starts with the published one, then a `ForCausalLM` one (the org publishes
    text models), then the shortest class, then an `npu` backend, then alphabetical, so
    the pick is stable across runs and machines.
    """
    if not name:
        return None
    if name in archs:
        return name
    stem = family_stem(name)
    want = normalize_family(stem)
    tokens = family_tokens(stem)
    if not want or not tokens:
        return None
    exact, token = [], []
    for cls, entry in archs.items():
        entry = entry or {}
        have = normalize_family(cls)
        rank = (not have.startswith(want), not cls.endswith("ForCausalLM"), len(have),
                "npu" not in (entry.get("backends") or []), cls)
        if want in {normalize_family(entry.get("gguf")),
                    normalize_family(entry.get("npu_model_type"))}:
            exact.append(rank)
            continue
        have_tokens = family_tokens(cls)
        if have_tokens[:len(tokens)] == tokens and len(want) >= 4:
            token.append((-len(tokens),) + rank)
    pool = sorted(exact) or sorted(token)
    return pool[0][-1] if pool else None


def flm_repo(record, archs):
    """One listing record -> the row the report prints (pure; sizes are filled later)."""
    arch = flm_architecture(record)
    cls = resolve_family(arch, archs)
    entry = (archs.get(cls) or {}) if cls else {}
    backends = entry.get("backends") or []
    # A repo with no config is a quantized/adapter drop of somebody else's model (the
    # gguf repos in the org), the same derivative the architecture walk recognises, so it
    # is reported but is not a new family — there is no architecture in the listing to map.
    return {
        "id": record.get("id") or "",
        "lastModified": str(record.get("lastModified") or ""),
        "q4nx": q4nx_files(record.get("siblings")),
        "license": license_of(record.get("tags")),
        "base_models": base_models_of(record.get("tags")),
        "architecture": arch,
        "derivative": bool(is_derivative(record)),
        "registry_class": cls or "",
        "registry_gguf": entry.get("gguf") or "",
        "npu_model_type": entry.get("npu_model_type") or entry.get("gguf") or "",
        "mapped": bool(cls and backends),
        "npu": "npu" in backends,
        "hrx": "hrx" in backends,
    }


_REVIEWED_CACHE = {}


def reviewed_report_classes():
    """Classes the registry has reviewed and recorded as NOT an alias (`watch: report`).

    The FLM half of the census ignored `registry/significant.json`, so a pack whose family this engine has
    explicitly declined (`Lfm3ForCausalLM`, `DFlash2DraftModel`) printed `!! NEW FLM ... registry: none` and
    kept the daily issue open forever, while the same class in the model census printed `. REVIEWED`. Same
    rule now applies to both halves: a reviewed class is listed, not alerted (engine#1451).
    """
    if "v" not in _REVIEWED_CACHE:
        sig = json.load(open(os.path.join(REG, "significant.json")))["classes"]
        _REVIEWED_CACHE["v"] = frozenset(c for c, e in sig.items() if e.get("watch") == "report")
    return _REVIEWED_CACHE["v"]


_ANSWERS_CACHE = {}


def flm_answers(path=None):
    """`registry/flm_answers.json` as {"architectures": {...}, "repos": {...}} (empty when absent)."""
    path = path or FLM_ANSWERS_PATH
    if path not in _ANSWERS_CACHE:
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        _ANSWERS_CACHE[path] = {"architectures": data.get("architectures") or {},
                                "repos": data.get("repos") or {}}
    return _ANSWERS_CACHE[path]


def answer_for(repo, answers=None):
    """The recorded answer for a repo that would alert, else None (engine#2171).

    Only a gap can be answered: a repo the NPU route serves, or a derivative drop, gets None
    whatever the file says, so an answer cannot hide a family that starts to load. The repo id
    is looked up first (a config-less repo has no architecture to key on), then the
    architecture the repo declares.
    """
    if not (npu_gap(repo) or class_unmapped(repo)):
        return None
    answers = answers if answers is not None else flm_answers()
    hit = (answers.get("repos") or {}).get(repo["id"])
    if hit is None and repo["architecture"]:
        hit = (answers.get("architectures") or {}).get(repo["architecture"])
    return hit


def answer_note(answer):
    """The short form printed on the line: the tracking issue, else the doc that records it."""
    issue = answer.get("issue")
    return f"answered: engine#{issue}" if issue else f"answered: {answer.get('doc') or 'registry/flm_answers.json'}"


def class_unmapped(repo):
    """True when `registry/architectures.json` does not map the repo's architecture class."""
    return not repo["derivative"] and not repo["mapped"]


def unmapped(repo):
    """True when a repo publishes a family this engine cannot serve (the FLM alert).

    Two ways that happens, and the second is the one the org's `*-NPU2` packs hit: the
    class is not in the registry at all, or it is there only for a backend that cannot run
    a Q4NX pack. FastFlowLM publishes NPU packs, so `qwen35`/`gemma4`/`dflash` — mapped to
    `hrx` alone — read as mapped and served while the NPU route cannot load them at all
    (the 2026-10 D3-* family, Nemotron-3.5-Bolt-E4B and its DFlash drafter arrived that
    way, printed as `+ FLM ... npu route: no` and never alerted).
    """
    return class_unmapped(repo) or npu_gap(repo)


def npu_gap(repo):
    """True when the repo is a Q4NX (NPU) pack whose family the NPU route does not serve.

    A repo with no `*.q4nx` file is not an NPU pack (a README-only or audio/vision-only
    drop), and a config-less drop is a derivative of somebody else's model, so neither is
    a coverage gap of ours.
    """
    return bool(repo["q4nx"]) and repo["mapped"] and not repo["npu"] and not repo["derivative"]


def repo_detail_url(repo_id):
    return f"{API}/{repo_id}?blobs=true"


def fill_sizes(repo, fetch=None):
    """Read the `*.q4nx` byte sizes from the repo detail (the listing has names only).

    One extra request, and only for a repo whose sizes are still unknown (new or changed,
    or every repo under --full). A failure here leaves the size at 0 and prints a note:
    the repo's arrival is still reported, which is the part that alerts.
    """
    if not repo["q4nx"] or all(item["size"] for item in repo["q4nx"]):
        return repo
    try:
        record = (fetch or get_json)(repo_detail_url(repo["id"]))
    except Exception as e:
        print(f"census-watch: flm sizes {repo['id']}: {e}", file=sys.stderr)
        return repo
    sizes = {item["name"]: item["size"] for item in q4nx_files((record or {}).get("siblings"))}
    for item in repo["q4nx"]:
        item["size"] = item["size"] or sizes.get(item["name"], 0)
    return repo


def sweep_org(archs, limit=FLM_LIMIT, fetch=None):
    """FastFlowLM's HF repos, newest first, from one listing request."""
    listing = (fetch or get_json)(FLM_ORG_URL)
    if not isinstance(listing, list):
        raise RuntimeError(f"unexpected org listing: {type(listing).__name__}")
    rows = [flm_repo(record, archs) for record in listing if (record or {}).get("id")]
    return rows[:limit]


def kernel_format_flags(release):
    """The KERNEL_FORMAT_MARKERS named by a release's notes or asset names (sorted)."""
    release = release or {}
    notes = " ".join(str(release.get(k) or "") for k in ("tag_name", "name", "body"))
    assets = " ".join(str((a or {}).get("name") or "") for a in release.get("assets") or ())
    haystack = f"{notes} {assets}".lower()
    return sorted({m for m in KERNEL_FORMAT_MARKERS if m in haystack})


def release_summary(release):
    release = release or {}
    return {
        "tag": release.get("tag_name") or "",
        "date": str(release.get("published_at") or release.get("created_at") or ""),
        "assets": [str((a or {}).get("name") or "") for a in release.get("assets") or ()],
        "kernel_format": kernel_format_flags(release),
        "url": release.get("html_url") or "",
    }


def sweep_releases(limit=GH_RELEASES_LIMIT, fetch=None):
    """FastFlowLM's runtime releases, newest first (tag, date, asset names, format flags)."""
    releases = (fetch or get_json)(GH_RELEASES.format(limit=limit))
    if not isinstance(releases, list):
        raise RuntimeError(f"unexpected releases page: {type(releases).__name__}")
    return [release_summary(r) for r in releases][:limit]


def flm_deltas(state, repos, releases):
    """(new-or-changed repos, new releases) — everything the daily report prints."""
    known_models = state.get("flm_models") or {}
    known_tags = set(state.get("flm_tags") or ())
    fresh_repos = [r for r in repos if known_models.get(r["id"]) != r["lastModified"]]
    fresh_releases = [r for r in releases if r["tag"] not in known_tags]
    return fresh_repos, fresh_releases


def remember_flm(state, repos, releases):
    """Record the seen ids and tags, so the next run prints only what moved."""
    models = state.setdefault("flm_models", {})
    for repo in repos:
        models[repo["id"]] = repo["lastModified"]
    tags = state.setdefault("flm_tags", [])
    known = set(tags)
    for release in releases:
        if release["tag"] and release["tag"] not in known:
            tags.append(release["tag"])
            known.add(release["tag"])
    return state


def gib(size):
    return f"{size / float(1024 ** 3):.2f} GiB"


def flm_line(repo, reviewed=None, answers=None):
    """One repo, one line: files, license, base model, architecture, both mapping answers.

    `reviewed` is `reviewed_report_classes()`: a class recorded there as not-an-alias is printed as
    `. FLM ... registry: none (reviewed, not an alias)` instead of `!! NEW FLM`, so an answered question
    stops being asked (engine#1451)."""
    parts = []
    size = sum(item["size"] for item in repo["q4nx"])
    if repo["q4nx"]:
        names = ", ".join(item["name"] for item in repo["q4nx"])
        parts.append(f"q4nx {len(repo['q4nx'])} file(s)"
                     + (f" {gib(size)}" if size else "") + f" [{names}]")
    else:
        parts.append("no q4nx file")
    if repo["license"]:
        parts.append(f"license {repo['license']}")
    if repo["base_models"]:
        parts.append("base " + ", ".join(repo["base_models"]))
    parts.append(f"arch {repo['architecture'] or 'none (no config in the repo)'}")
    if repo["mapped"]:
        parts.append(f"registry {repo['registry_class']} -> {repo['registry_gguf']}")
        parts.append(f"npu route: {repo['npu_model_type']}" if repo["npu"] else "npu route: no")
    else:
        seen = (reviewed if reviewed is not None else reviewed_report_classes())
        parts.append("registry: none (reviewed, not an alias - registry/significant.json)"
                     if repo["architecture"] in seen else "registry: none")
    if repo["derivative"]:
        parts.append("derivative drop (no config by design)")
    seen = (reviewed if reviewed is not None else reviewed_report_classes())
    decl = class_unmapped(repo) and repo["architecture"] in seen
    answer = answer_for(repo, answers)
    if answer is not None:
        parts.append(answer_note(answer))
    if npu_gap(repo):
        marker = ". NPU FLM" if answer is not None else "!! NPU FLM"
    elif decl or (answer is not None and unmapped(repo)):
        marker = ". FLM"
    else:
        marker = "!! NEW FLM" if unmapped(repo) else "+ FLM"
    return f"{marker} {repo['id']}  ({repo['lastModified'][:10]})  " + "  ".join(parts)


def release_line(release):
    flags = ", ".join(release["kernel_format"])
    return (f"+ FLM RELEASE {release['tag']}  ({release['date'][:10]})  "
            + (f"kernel-format: {flags}  " if flags else "")
            + f"assets {len(release['assets'])}")


def report_flm(repos, releases, fresh_repos, fresh_releases, full=False, answers=None):
    """Print the FastFlowLM delta and return its exit code (1 on an unserved, unanswered family)."""
    shown_repos = repos if full else fresh_repos
    shown_releases = releases if full else fresh_releases
    reviewed = reviewed_report_classes()
    answers = answers if answers is not None else flm_answers()
    answered = [repo["id"] for repo in shown_repos if answer_for(repo, answers) is not None]
    unmapped_repos = [repo["id"] for repo in shown_repos
                      if class_unmapped(repo) and repo["architecture"] not in reviewed
                      and repo["id"] not in answered]
    # Split the two reasons in the summary: "map the class" and "give the NPU route this
    # family" are different pieces of work, and a Q4NX pack named for an HRX-only family
    # is the second one.
    gaps = {}
    for repo in shown_repos:
        if npu_gap(repo) and repo["id"] not in answered:
            family = repo["registry_gguf"] or repo["registry_class"] or repo["architecture"]
            gaps.setdefault(family, []).append(repo["id"])
    flagged = [r["tag"] for r in shown_releases if r["kernel_format"]]
    print(f"census-watch: FLM watch ({FLM_AUTHOR}): {len(repos)} repo(s), "
          f"{len(releases)} release(s) checked; {len(fresh_repos)} new/changed repo(s), "
          f"{len(fresh_releases)} new release(s)" + ("; full listing" if full else ""))
    for repo in shown_repos:
        print("  " + flm_line(repo, reviewed, answers))
    for release in shown_releases:
        print("  " + release_line(release))
    if not shown_repos and not shown_releases:
        print("census-watch: FLM: nothing new since the last run")
    if unmapped_repos:
        print(f"  -> {len(unmapped_repos)} FastFlowLM repo(s) publish a family this engine does "
              f"not serve: {', '.join(unmapped_repos)}")
    if gaps:
        detail = "; ".join(f"{family} ({len(ids)} pack(s))" for family, ids in sorted(gaps.items()))
        print(f"  -> {sum(len(i) for i in gaps.values())} Q4NX (NPU) pack(s) for a family the NPU "
              f"route does not serve: {detail}")
        print("     mapped for another backend only, so the census counts the family as covered: "
              "the NPU route (or a recorded decision that it will not serve it) is what is missing")
    if answered:
        print(f"  -> {len(answered)} pack(s) already answered in registry/flm_answers.json (listed, not "
              "alerted; the issue on each line tracks the work)")
    if flagged:
        print(f"  -> release(s) {', '.join(flagged)} name the Q4NX container format "
              f"({', '.join(KERNEL_FORMAT_MARKERS)}): read the notes before the pinned "
              f"npu/q4nx decoder moves")
    return 1 if (unmapped_repos or gaps) else 0


def watch_flm(archs, state, limit=FLM_LIMIT, full=False, fetch=None, detail=None, answers=None):
    """Sweep the org + releases, print the delta, remember it: (exit code, did it run?)"""
    try:
        repos = sweep_org(archs, limit, fetch)
        releases = sweep_releases(GH_RELEASES_LIMIT, fetch)
    except Exception as e:
        print(f"census-watch: !! FLM WATCH FAILED: {e}", file=sys.stderr)
        return 0, False
    fresh_repos, fresh_releases = flm_deltas(state, repos, releases)
    for repo in (repos if full else fresh_repos):
        fill_sizes(repo, detail)
    rc = report_flm(repos, releases, fresh_repos, fresh_releases, full, answers)
    remember_flm(state, repos, releases)
    return rc, True


def exit_code(arch_rc, flm_rc, flm_ok, support_rc=0):
    """Combine the watches into the documented exit code (see the module docstring)."""
    if arch_rc or flm_rc or support_rc:
        return 1
    return 0 if flm_ok else 2


def report(models, res, previously_seen):
    """Print the run's summary and return the exit code (1 when a class is uncovered)."""
    in_scope = len(res[COVERED]) + len(res[UNCOVERED])
    print(f"census-watch: {len(models)} newest text-generation models checked, "
          f"{in_scope} with an architecture, {len(res[COVERED])} mapped, "
          f"{len(res['classes'])} uncovered class(es), "
          f"{len(res[UNVERIFIABLE])} unverifiable (gated/no-config), "
          f"{len(res[DERIVATIVE])} derivative (no config by design)")
    if not models:
        print(f"census-watch: nothing new since the last run "
              f"({len(previously_seen)} models already classified)")
    reviewed_report = set(res.get("reviewed_report", ()))
    quiet = set(res.get("remote_code", ())) | reviewed_report
    for cls, ids in sorted(res["classes"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if cls in reviewed_report:
            print(f"  . REVIEWED {cls}: {len(ids)} model(s), e.g. {ids[0]}")
            print(f"     -> reviewed, not an alias, report only (registry/significant.json, "
                  f"docs/arch-gaps.md)")
        elif cls in quiet:
            owners = len({i.split("/")[0] for i in ids})
            print(f"  . REMOTE-CODE {cls}: {len(ids)} model(s) from {owners} uploader(s), e.g. {ids[0]}")
            print(f"     -> custom modeling code (auto_map, a non-Transformers runtime, or the repo's own modeling_*.py); "
                  f"alerts once {REMOTE_CODE_SPREAD} uploaders use it")
        elif cls in res["significant"]:
            print(f"  !! UNCOVERED {cls} (reviewed, not an alias — docs/arch-gaps.md): "
                  f"{len(ids)} model(s), e.g. {ids[0]}")
            print(f"     -> needs real engine support, not a mapping; if a backend starts "
                  f"accepting it, record why in registry/significant.json ('mapped_ok')")
        else:
            print(f"  !! UNCOVERED {cls} (new class): {len(ids)} model(s), e.g. {ids[0]}")
            # engine#1922: the registry is GENERATED from the pinned sources, so "not in the registry" cannot by itself
            # say whether a pin bump would fix this or whether nobody upstream has it. Say which, when it is knowable.
            verdict = res.get("upstream_verdict", {}).get(cls, "unknown")
            if verdict == "pin-behind":
                print(f"     -> UPSTREAM HAS IT: llama.cpp's converter registers this class, our pin does not -- "
                      f"a pin bump regenerates the registry")
            elif verdict == "upstream":
                print(f"     -> UPSTREAM DOES NOT HAVE IT: no llama.cpp converter maps this class, so there is nothing "
                      f"to map here until one does")
            else:
                print(f"     -> could not tell whether upstream has it (converter unavailable); check before mapping")
            print(f"     -> map it in registry/architectures.json (a pinned backend's code "
                  f"must accept the architecture)")
    # engine#1922: ask upstream ONCE whether it knows the uncovered classes, and only when there is something to ask about.
    # A failed fetch is recorded as None so the report says "could not tell" rather than guessing either way.
    if res["classes"]:
        upstream = None
        try:
            upstream = upstream_classes(get_text(UPSTREAM_CONVERTER_URL))
        except Exception:
            upstream = None
        res["upstream_verdict"] = classify_uncovered(res["classes"], upstream)
    # say whether a token was already sent: "gated repos need a token" is misleading advice on a
    # run that sent one and still got no config back
    hint = " (HF_TOKEN sent, still no config)" if hf_token() else ""
    for mid in sorted(res[UNVERIFIABLE]):
        print(f"  ? UNVERIFIABLE {mid} ({res['reasons'][mid]}) — gated repos need a token{hint}; "
              f"retried next run")
    return 1 if any(c not in quiet for c in res["classes"]) else 0


def load_support_pins(path=SUPPORT_PINS_PATH):
    """The pinned supporting upstreams (registry/support_pins.json)."""
    with open(path) as f:
        return json.load(f)


def sweep_support(pins, fetch=None):
    """(per-repo head vs pin, d3 weight repos that are not in the pinned list)."""
    fetch = fetch or get_json
    heads = {}
    for repo, entry in sorted((pins.get("repos") or {}).items()):
        entry = entry or {}
        commit = (fetch(GH_REPO_HEAD.format(repo=repo)) or [{}])[0] or {}
        sha = commit.get("sha") or ""
        pinned = entry.get("commit") or ""
        paths = list(entry.get("paths") or ())
        moved_paths, dead_paths = [], []
        if paths and sha:
            for path in paths:
                at_pin = (fetch(GH_PATH_HEAD.format(repo=repo, sha=pinned, path=path)) or [{}])[0] or {}
                if not at_pin.get("sha"):
                    # no commit at the pin touches it (a typo, or renamed before the pin): it could never move
                    dead_paths.append(path)
                    continue
                if sha == pinned:
                    continue
                at_head = (fetch(GH_PATH_HEAD.format(repo=repo, sha=sha, path=path)) or [{}])[0] or {}
                if at_pin["sha"] != (at_head.get("sha") or ""):
                    moved_paths.append(path)
        heads[repo] = {
            "repo": repo,
            "commit": sha,
            "date": ((commit.get("commit") or {}).get("committer") or {}).get("date") or "",
            "pinned": pinned,
            "reviewed": entry.get("reviewed") or "",
            "paths": paths,
            "moved_paths": moved_paths,
            "dead_paths": dead_paths,
            "moved": (bool(moved_paths or dead_paths) if paths else sha != pinned),
        }
    sources = pins.get("d3_sources") or {}
    org, prefix = sources.get("org") or "", sources.get("prefix") or ""
    known = set(sources.get("known") or ())
    new_d3 = []
    if org:
        for record in fetch(HF_ORG_URL.format(org=org)) or []:
            rid = (record or {}).get("id") or ""
            if rid and rid.startswith(org + "/" + prefix) and rid not in known:
                new_d3.append(rid)
    return heads, new_d3


def report_support(heads, new_d3, org="", prefix=""):
    """Print the supporting pins; return 1 when one moved or an unpinned d3 repo appeared."""
    moved = [h for h in heads.values() if h["moved"]]
    for h in sorted(heads.values(), key=lambda x: x["repo"]):
        if h["moved"]:
            print(f"  !! SUPPORT MOVED {h['repo']}  head {h['commit'][:12]} ({h['date'][:10]})  "
                  f"pinned {h['pinned'][:12]} reviewed {h['reviewed']}"
                  + (f"  watched paths changed: {', '.join(h['moved_paths'])}" if h.get("moved_paths") else "")
                  + (f"  watched paths with no commit at the pin (fix `paths`): {', '.join(h['dead_paths'])}"
                     if h.get("dead_paths") else ""))
        elif h.get("paths") and h["commit"] != h["pinned"]:
            print(f"  + SUPPORT PIN {h['repo']}  head {h['commit'][:12]} ({h['date'][:10]})  pinned "
                  f"{h['pinned'][:12]} reviewed {h['reviewed']}: no commit since the pin touches the "
                  f"{len(h['paths'])} watched path(s)")
        else:
            print(f"  + SUPPORT PIN {h['repo']}  {h['commit'][:12]} ({h['date'][:10]}) reviewed {h['reviewed']}")
    if moved:
        print(f"  -> {len(moved)} pinned supporting upstream(s) moved: "
              + ", ".join(h["repo"] for h in moved)
              + " - read the change, then bump the commit and `reviewed` in registry/support_pins.json")
    if new_d3:
        print(f"  -> {len(new_d3)} {org}/{prefix}* repo(s) not in registry/support_pins.json: "
              + ", ".join(new_d3[:10])
              + " - a d3 pack for it is on the way; add it to the list once looked at")
    return 1 if (moved or new_d3) else 0


def watch_support(pins, fetch=None):
    """Sweep the pinned upstreams and the d3 org: (exit code, did it run?)"""
    try:
        heads, new_d3 = sweep_support(pins, fetch)
    except Exception as e:
        print(f"census-watch: !! SUPPORT WATCH FAILED: {e}", file=sys.stderr)
        return 0, False
    sources = pins.get("d3_sources") or {}
    return report_support(heads, new_d3, sources.get("org") or "", sources.get("prefix") or ""), True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"newest models to check (default {DEFAULT_LIMIT})")
    ap.add_argument("--state", default=os.environ.get("CENSUS_WATCH_STATE", DEFAULT_STATE),
                    help=f"state file (default {DEFAULT_STATE})")
    ap.add_argument("--full", action="store_true",
                    help="list every FastFlowLM repo and release, not just the delta")
    ap.add_argument("--support-pins", default=SUPPORT_PINS_PATH,
                    help="pinned supporting upstreams (default registry/support_pins.json)")
    a = ap.parse_args()

    archs, sig = load_registry()
    state = load_state(a.state)
    previously_seen = set(state["seen"])
    state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    models = sweep_newest(a.limit, previously_seen)
    res = classify(models, archs, sig, previously_seen)
    rc = report(models, res, previously_seen)

    # Remember what was classified, so the walk moves past it. An unverifiable model is
    # remembered too but reported only the first time it is seen.
    remember(state, res)
    if models and rc == 0:
        print("census-watch: OK — no uncovered class in the newest models")

    # The FastFlowLM watch runs after the architecture walk and remembers its own delta, so
    # a failure here cannot lose what the walk just classified; the exit code is the OR of
    # the two alerts (engine#588).
    flm_rc, flm_ok = watch_flm(archs, state, full=a.full)

    # The supporting-upstream pins run last: they are the only watch whose "fix" is a reviewed
    # edit to registry/support_pins.json rather than a registry mapping.
    support_rc, support_ok = (0, True)
    try:
        support_rc, support_ok = watch_support(load_support_pins(a.support_pins))
    except FileNotFoundError as e:
        print(f"census-watch: !! SUPPORT PINS MISSING: {e}", file=sys.stderr)
        support_ok = False
    except Exception as e:
        print(f"census-watch: !! SUPPORT WATCH FAILED: {e}", file=sys.stderr)
        support_ok = False
    rc = exit_code(rc, flm_rc, flm_ok, support_rc)

    save_state(a.state, state)
    print(f"census-watch: state {a.state}")
    if rc == 1:
        print("census-watch: EXIT 1 (an alert is above: uncovered class, unmapped FastFlowLM "
              "family, moved supporting pin and/or a new d3 repo)")
    elif rc == 2:
        print("census-watch: EXIT 2 (a watch did not run — see the failure above)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
