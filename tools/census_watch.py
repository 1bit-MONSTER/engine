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

usage: tools/census_watch.py [--limit N] [--state PATH]

Exit codes (the alert contract):
  0  no uncovered classes among the newest models
  1  an uncovered class arrived (the alert)
  *  runtime failure (no network, unreadable registry, ...)

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
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')

# A derivative carries no config.json of its own by design (a quantized GGUF, a LoRA
# adapter); the raw release it derives from carries the config and is what gets checked.
DERIVATIVE_TAGS = ("gguf", "lora", "peft")

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
    out["classes"] = classes
    out["reasons"] = reasons
    out["significant"] = sorted(c for c in classes if c in sig)
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
        return {"last_run": None, "seen": {}}
    state.setdefault("seen", {})
    return state


def save_state(path, state):
    seen = state["seen"]
    if len(seen) > MAX_SEEN:
        state["seen"] = dict(list(seen.items())[-MAX_SEEN:])
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
    for cls, ids in sorted(res["classes"].items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if cls in res["significant"]:
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
    return 1 if res["classes"] else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                    help=f"newest models to check (default {DEFAULT_LIMIT})")
    ap.add_argument("--state", default=os.environ.get("CENSUS_WATCH_STATE", DEFAULT_STATE),
                    help=f"state file (default {DEFAULT_STATE})")
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
    save_state(a.state, state)
    print(f"census-watch: state {a.state}")
    if rc != 0:
        print(f"census-watch: EXIT {rc} (an uncovered class arrived in the newest models)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
