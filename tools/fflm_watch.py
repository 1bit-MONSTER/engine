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
"""Watch FastFlowLM's model drops and its runtime releases (engine #588).

tools/census_watch.py watches HF *architecture classes*, which is the wrong shape for FastFlowLM: its drops are
files, not architectures. `FastFlowLM/<something>-NPU2` repos carry a `model.q4nx` and land without any registry
change, and the runtime itself ships tags whose notes say when the q4nx container or its chunk layout moved — which
is what `npu/q4nx.cpp` decodes. Both went unnoticed until a person happened to look, so this watcher reports them
in the same daily run:

1. the HF org listing (`author=FastFlowLM`), one line per new or changed repo: `lastModified`, the size of its
   `*.q4nx` files, its license tag, its base model, the architecture class from the repo config, whether
   `registry/architectures.json` maps it and whether that entry lists the `npu` backend (our NPU route has the
   family), and the FastFlowLM feature names the repo itself carries;
2. the `FastFlowLM/FastFlowLM` GitHub releases, one line per new tag: date, asset count, and whether the notes or
   asset names mention the container/format words (`q4nx`, `chunk`, `layout`) that mean our Q4NX decoder pins may
   need a look;
3. a state file of last-seen repos (by `lastModified`) and tags, so a quiet day says so; `--full` lists everything.

Exit codes (the alert contract, same shape as tools/census_watch.py):
  0  nothing new, or everything new is a family our registry maps
  1  a new/changed repo whose architecture our registry does not map (`!! NEW FLM`)
  2  runtime failure (no network, unreadable state)

The NPU route is judged from the generated registry, never by hand: an `architectures.json` entry whose `backends`
lists `npu` is a family the NPU route serves (`npu_model_type`), and `hrx` is the GPU route. So this file cannot
drift from what the engine actually serves.

Private-IP rule (engine #588): this report names FastFlowLM files, sizes and versions. It says nothing about our
NPU kernels, their names or their speed; the coverage column says only whether the registry has the family.
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

HF_ORG_URL = "https://huggingface.co/api/models?author=FastFlowLM&sort=lastModified&limit={limit}"
HF_REPO_URL = "https://huggingface.co/api/models/{repo_id}?blobs=true"
GH_RELEASES_URL = "https://api.github.com/repos/FastFlowLM/FastFlowLM/releases?per_page={limit}"

# The words that mean the q4nx container or its chunking moved (engine #588 item 2).
KERNEL_FORMAT_MARKERS = ("q4nx", "chunk", "layout")

# FastFlowLM names the feature flags in its repo id and tags (`-layerP`, `lax`); we surface them as given and do
# not guess at their meaning, because they are FastFlowLM's names and not ours.
FLM_FEATURE_MARKERS = ("layerp", "lax", "ple")

DEFAULT_STATE = os.path.expanduser("~/.1bit/fflm-watch-state.json")


def http_json(url, tries=3):
    """GET one JSON document. Raises RuntimeError when it cannot be read."""
    last = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "1bit-fflm-watch"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:  # noqa: PERF203 - retried
            last = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def load_registry():
    """The generated registry: architecture class -> {'gguf', 'backends', 'npu_model_type', ...}."""
    with open(os.path.join(REG, "architectures.json"), encoding="utf-8") as f:
        return json.load(f)["architectures"]


def load_state(path):
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        state = {}
    state.setdefault("repos", {})
    state.setdefault("releases", [])
    return state


def save_state(path, state):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, sort_keys=True)
        f.write("\n")


def license_of(tags):
    for tag in tags or ():
        if tag.startswith("license:"):
            return tag.split(":", 1)[1]
    return ""


def base_models_of(tags):
    """`base_model:org/name` and `base_model:finetune:org/name` tags, deduplicated, finetune marker stripped."""
    bases = []
    for tag in tags or ():
        if not tag.startswith("base_model:"):
            continue
        name = tag.split(":", 1)[1]
        if name.startswith(("finetune:", "quantized:", "adapter:")):
            name = name.split(":", 1)[1]
        if name not in bases:
            bases.append(name)
    return bases


def flm_features(repo_id, tags):
    """FastFlowLM's own feature naming, from the id and tags (`layerP`, `lax`, ...)."""
    haystack = " ".join([repo_id.split("/", 1)[-1]] + list(tags or ())).lower()
    return sorted({marker for marker in FLM_FEATURE_MARKERS if marker in haystack})


def q4nx_files(siblings):
    """The `*.q4nx` files of a repo with their sizes (LFS size when the pointer is what the listing reports)."""
    files = []
    for sibling in siblings or ():
        name = sibling.get("rfilename") or ""
        if not name.endswith(".q4nx"):
            continue
        size = sibling.get("size")
        lfs = sibling.get("lfs") or {}
        if lfs.get("size"):
            size = lfs["size"]
        files.append({"name": name, "size": int(size or 0)})
    return sorted(files, key=lambda item: item["name"])


def architecture_of(config):
    """The class census/registry speak: `architectures[0]`, else the config `model_type`, else ''."""
    config = config or {}
    archs = config.get("architectures") or []
    if archs and isinstance(archs[0], str):
        return archs[0]
    model_type = config.get("model_type")
    return model_type if isinstance(model_type, str) else ""


def normalize_family(name):
    """A family name with the punctuation dropped and in lower case (`qwen3_5_text` -> `qwen35text`)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def resolve_family(name, archs):
    """The registry entry for a class name, a gguf arch or a FastFlowLM model_type.

    A FastFlowLM repo config names a text tower (`model_type: qwen3_5_text`) and has no `architectures` list, while
    the registry is keyed by the HF class and carries the gguf/npu names it was generated with. So: try the class,
    then an exact `npu_model_type` / `gguf` match, then a normalized prefix match (`qwen35text` against the registry
    name `qwen35`) of at least four characters - preferring an entry the NPU route serves when several qualify.
    A name that matches nothing stays unmapped, which is the alert.
    """
    if not name:
        return ""
    if name in archs:
        return name
    for cls, entry in archs.items():
        if name in (entry.get("npu_model_type"), entry.get("gguf")):
            return cls
    wanted = normalize_family(name)
    if len(wanted) < 4:
        return ""
    candidates = []
    for cls, entry in archs.items():
        for candidate in (cls, entry.get("gguf"), entry.get("npu_model_type")):
            stem = normalize_family(candidate)
            if len(stem) >= 4 and wanted.startswith(stem):
                candidates.append((len(stem), cls))
                break
    if not candidates:
        return ""
    best = max(length for length, _ in candidates)
    tied = [cls for length, cls in candidates if length == best]
    # A multimodal wrapper (MiniCPM-V and friends) is registered on the same text tower's gguf arch, so a
    # repo whose model_type is the plain text tower must resolve to the plain family: prefer a causal-LM class,
    # then one the NPU route serves, then alphabetical so the answer never depends on dict order.
    tied.sort(key=lambda cls: (0 if "ForCausalLM" in cls else 1,
                               0 if "npu" in (archs[cls].get("backends") or ()) else 1,
                               cls))
    return tied[0]


def repo_summary(repo, archs):
    """One HF repo as the report sees it: files, license, base family, and how our routes cover it."""
    repo_id = repo.get("id") or repo.get("modelId") or ""
    tags = repo.get("tags") or []
    architecture = architecture_of(repo.get("config"))
    registry_class = resolve_family(architecture, archs)
    entry = archs.get(registry_class, {}) if registry_class else {}
    backends = entry.get("backends") or []
    return {
        "id": repo_id,
        "lastModified": repo.get("lastModified") or repo.get("createdAt") or "",
        "q4nx": q4nx_files(repo.get("siblings")),
        "license": license_of(tags),
        "base_models": base_models_of(tags),
        "architecture": architecture,
        "registry_class": registry_class,
        "registry_gguf": entry.get("gguf", ""),
        "npu_model_type": entry.get("npu_model_type", ""),
        "mapped": bool(backends),
        "npu": "npu" in backends,
        "hrx": "hrx" in backends,
        "flm_features": flm_features(repo_id, tags),
    }


def kernel_format_flags(release):
    """The container/format words a release's notes or asset names carry (engine #588 item 2)."""
    text = " ".join([release.get("name") or "", release.get("body") or ""]
                    + [asset.get("name") or "" for asset in release.get("assets") or ()]).lower()
    return sorted({marker for marker in KERNEL_FORMAT_MARKERS if marker in text})


def release_summary(release):
    return {
        "tag": release.get("tag_name") or "",
        "name": release.get("name") or "",
        "published_at": release.get("published_at") or "",
        "prerelease": bool(release.get("prerelease")),
        "assets": [asset.get("name") or "" for asset in release.get("assets") or ()],
        "kernel_format": kernel_format_flags(release),
    }


def gib(size):
    return f"{size / (1024 ** 3):.2f} GiB"


def new_repos(state, repos):
    """Repos whose `lastModified` is unknown or moved; unchanged ones are not reported."""
    fresh = []
    for repo in repos:
        seen = state["repos"].get(repo["id"])
        if seen != repo["lastModified"]:
            fresh.append(repo)
    return fresh


def new_releases(state, releases):
    seen = set(state["releases"])
    return [release for release in releases if release["tag"] not in seen]


def report(repos, releases, fresh_repos, fresh_releases, full=False):
    """Print the day's lines and return the exit code (1 when an unmapped family arrived)."""
    models_line(repos, releases, fresh_repos, fresh_releases)
    shown_repos = repos if full else fresh_repos
    shown_releases = releases if full else fresh_releases
    unknown = 0
    for repo in shown_repos:
        size = sum(item["size"] for item in repo["q4nx"])
        detail = [f"q4nx {len(repo['q4nx'])} file(s) {gib(size)}" if repo["q4nx"] else "no q4nx file"]
        if repo["license"]:
            detail.append(f"license {repo['license']}")
        if repo["base_models"]:
            detail.append(f"base {', '.join(repo['base_models'])}")
        if repo["architecture"]:
            detail.append(f"arch {repo['architecture']}")
            routes = []
            if repo["mapped"]:
                routes.append(f"registry {repo['registry_class']} -> {repo['registry_gguf']}")
            if repo["npu"]:
                routes.append(f"npu {repo['npu_model_type']}")
            if repo["hrx"]:
                routes.append("hrx")
            detail.append("routes " + (" + ".join(routes) if routes else "none"))
        if repo["flm_features"]:
            detail.append("flm " + ",".join(repo["flm_features"]))
        marker = "+ REPO"
        if not repo["mapped"]:
            marker = "!! NEW FLM"
            unknown += 1
        print(f"  {marker} {repo['id']}  ({repo['lastModified'][:10]})")
        print(f"     -> {'  '.join(detail)}")
    for release in shown_releases:
        flags = ", ".join(release["kernel_format"]) if release["kernel_format"] else "none"
        print(f"  + RELEASE {release['tag']}  ({release['published_at'][:10]})  "
              f"kernel-format: {flags}  assets {len(release['assets'])}")
    if not shown_repos and not shown_releases:
        print("fflm-watch: nothing new since the last run")
    return 1 if unknown else 0


def models_line(repos, releases, fresh_repos, fresh_releases):
    """The one summary line the daily report carries."""
    unknown = sum(1 for repo in fresh_repos if not repo["mapped"])
    format_moved = sum(1 for release in fresh_releases if release["kernel_format"])
    print(f"fflm-watch: {len(repos)} FastFlowLM repo(s), {len(releases)} release(s) checked, "
          f"{len(fresh_repos)} new/changed repo(s), {len(fresh_releases)} new release(s)"
          + (f", {unknown} unknown famil{'y' if unknown == 1 else 'ies'}" if unknown else "")
          + (f", {format_moved} release(s) naming the container format" if format_moved else ""))
    return True


def remember(state, repos, releases):
    for repo in repos:
        state["repos"][repo["id"]] = repo["lastModified"]
    seen = set(state["releases"])
    for release in releases:
        if release["tag"] not in seen:
            state["releases"].append(release["tag"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--limit", type=int, default=50, help="newest repos and releases to check (default 50)")
    ap.add_argument("--state", default=os.environ.get("FFLM_WATCH_STATE", DEFAULT_STATE),
                    help=f"state file (default {DEFAULT_STATE})")
    ap.add_argument("--full", action="store_true", help="list every repo and release, not only the deltas")
    a = ap.parse_args()

    try:
        archs = load_registry()
        listing = http_json(HF_ORG_URL.format(limit=a.limit))
        repos = [repo_summary(http_json(HF_REPO_URL.format(repo_id=repo["id"])), archs)
                 for repo in listing if repo.get("id")]
        releases = [release_summary(release)
                    for release in http_json(GH_RELEASES_URL.format(limit=a.limit))][:a.limit]
    except (RuntimeError, OSError, json.JSONDecodeError) as error:
        print(f"fflm-watch: {error}", file=sys.stderr)
        return 2

    state = load_state(a.state)
    fresh_repos = new_repos(state, repos)
    fresh_releases = new_releases(state, releases)
    rc = report(repos, releases, fresh_repos, fresh_releases, a.full)
    remember(state, repos, releases)
    save_state(a.state, state)
    if rc != 0:
        print("fflm-watch: EXIT 1 (a FastFlowLM repo arrived whose family the registry does not map)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
