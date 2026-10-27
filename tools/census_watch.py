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
     architecture `registry/architectures.json` does not map — either one is enough
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

A class whose models all load through custom modeling code (`auto_map` in the config, or a
runtime library that is not Transformers, so Transformers itself has no implementation of
it either) and that fewer than REMOTE_CODE_SPREAD uploaders publish is reported but does
not alert: research one-offs like that arrive every day and are not something a backend
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
            req = urllib.request.Request(url, headers={"User-Agent": "1bit-census-watch"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r), r.headers.get("Link", "")
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
    A missing `library_name` is unknown, not custom, so it alerts.
    """
    cfg = model.get("config") or {}
    if cfg.get("auto_map"):
        return True
    if model.get("library_name") in NON_TRANSFORMERS_LIBRARIES:
        return "transformers" not in (model.get("tags") or [])
    return False


def is_derivative(model):
    tags = model.get("tags") or []
    return (model.get("library_name") == "peft"
            or any(t in tags for t in DERIVATIVE_TAGS))


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
    url = f"{API}?pipeline_tag={tag}&sort=createdAt&direction=-1&limit=100&config=true"
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


def unmapped(repo):
    """True when a repo publishes a family this engine cannot serve (the FLM alert)."""
    return not repo["mapped"] and not repo["derivative"]


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


def flm_line(repo):
    """One repo, one line: files, license, base model, architecture, both mapping answers."""
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
        parts.append("registry: none")
    if repo["derivative"]:
        parts.append("derivative drop (no config by design)")
    marker = "!! NEW FLM" if unmapped(repo) else "+ FLM"
    return f"{marker} {repo['id']}  ({repo['lastModified'][:10]})  " + "  ".join(parts)


def release_line(release):
    flags = ", ".join(release["kernel_format"])
    return (f"+ FLM RELEASE {release['tag']}  ({release['date'][:10]})  "
            + (f"kernel-format: {flags}  " if flags else "")
            + f"assets {len(release['assets'])}")


def report_flm(repos, releases, fresh_repos, fresh_releases, full=False):
    """Print the FastFlowLM delta and return its exit code (1 on an unmapped family)."""
    shown_repos = repos if full else fresh_repos
    shown_releases = releases if full else fresh_releases
    unmapped_repos = [repo["id"] for repo in shown_repos if unmapped(repo)]
    flagged = [r["tag"] for r in shown_releases if r["kernel_format"]]
    print(f"census-watch: FLM watch ({FLM_AUTHOR}): {len(repos)} repo(s), "
          f"{len(releases)} release(s) checked; {len(fresh_repos)} new/changed repo(s), "
          f"{len(fresh_releases)} new release(s)" + ("; full listing" if full else ""))
    for repo in shown_repos:
        print("  " + flm_line(repo))
    for release in shown_releases:
        print("  " + release_line(release))
    if not shown_repos and not shown_releases:
        print("census-watch: FLM: nothing new since the last run")
    if unmapped_repos:
        print(f"  -> {len(unmapped_repos)} FastFlowLM repo(s) publish a family this engine does "
              f"not map: {', '.join(unmapped_repos)}")
    if flagged:
        print(f"  -> release(s) {', '.join(flagged)} name the Q4NX container format "
              f"({', '.join(KERNEL_FORMAT_MARKERS)}): read the notes before the pinned "
              f"npu/q4nx decoder moves")
    return 1 if unmapped_repos else 0


def watch_flm(archs, state, limit=FLM_LIMIT, full=False, fetch=None, detail=None):
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
    rc = report_flm(repos, releases, fresh_repos, fresh_releases, full)
    remember_flm(state, repos, releases)
    return rc, True


def exit_code(arch_rc, flm_rc, flm_ok):
    """Combine the two watches into the documented exit code (see the module docstring)."""
    if arch_rc or flm_rc:
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
            print(f"     -> custom modeling code (auto_map, or a non-Transformers runtime); "
                  f"alerts once {REMOTE_CODE_SPREAD} uploaders use it")
        elif cls in res["significant"]:
            print(f"  !! UNCOVERED {cls} (reviewed, not an alias — docs/arch-gaps.md): "
                  f"{len(ids)} model(s), e.g. {ids[0]}")
            print(f"     -> needs real engine support, not a mapping; if a backend starts "
                  f"accepting it, record why in registry/significant.json ('mapped_ok')")
        else:
            print(f"  !! UNCOVERED {cls} (new class): {len(ids)} model(s), e.g. {ids[0]}")
            print(f"     -> map it in registry/architectures.json (a pinned backend's code "
                  f"must accept the architecture)")
    for mid in sorted(res[UNVERIFIABLE]):
        print(f"  ? UNVERIFIABLE {mid} ({res['reasons'][mid]}) — gated repos need a token; "
              f"retried next run")
    return 1 if any(c not in quiet for c in res["classes"]) else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"newest models to check (default {DEFAULT_LIMIT})")
    ap.add_argument("--state", default=os.environ.get("CENSUS_WATCH_STATE", DEFAULT_STATE),
                    help=f"state file (default {DEFAULT_STATE})")
    ap.add_argument("--full", action="store_true",
                    help="list every FastFlowLM repo and release, not just the delta")
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
    rc = exit_code(rc, flm_rc, flm_ok)

    save_state(a.state, state)
    print(f"census-watch: state {a.state}")
    if rc == 1:
        print("census-watch: EXIT 1 (an alert is above: uncovered class and/or unmapped "
              "FastFlowLM family)")
    elif rc == 2:
        print("census-watch: EXIT 2 (a watch did not run — see the failure above)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
