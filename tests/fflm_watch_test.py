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
"""Pin the FastFlowLM watcher (tools/fflm_watch.py, ctest fflm_watch).

Engine #588: FastFlowLM publishes files, not architectures, so its drops are invisible to the HF
architecture census. The watcher's job is to notice a new `-NPU2` repo or a runtime tag that moves the
q4nx container, and to say whether the registry maps the family. A regression that turned that into a
silent non-report (or a permanent `!! NEW FLM`) is what this test exists to catch.

Everything here is offline: the fixtures under tests/fixtures/fflm/ are recorded API payloads (HF model
detail, the org listing, the GitHub releases list), trimmed to the fields the watcher reads.
"""
import io
import json
import os
import sys
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import fflm_watch as fw  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests", "fixtures", "fflm")

# A registry shaped like the generated one: a class the NPU route serves, and one only the GPU route does.
ARCHS = {
    "Qwen3_5MoeForCausalLM": {"gguf": "qwen35moe", "backends": ["hrx", "npu"], "npu_model_type": "qwen3_5_moe"},
    "Qwen35ForCausalLM": {"gguf": "qwen35", "backends": ["hrx"]},
}

failures = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        failures.append(name)


def load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def report_of(repos, releases, fresh_repos, fresh_releases, full=False):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fw.report(repos, releases, fresh_repos, fresh_releases, full)
    return rc, buf.getvalue()


mapped_repo = load("repo-mapped.json")
unknown_repo = load("repo-unknown.json")
org_listing = load("org.json")
releases = load("releases.json")

# 1. The HF repo payload is read the way the report needs it: files, license, base family, routes.
mapped = fw.repo_summary(mapped_repo, ARCHS)
check("repo id", mapped["id"] == "FastFlowLM/D3-Mini-NPU2")
check("q4nx files with sizes",
      [item["name"] for item in mapped["q4nx"]] == ["model.q4nx", "vision_weight.q4nx"]
      and all(item["size"] > 0 for item in mapped["q4nx"]))
check("license tag", mapped["license"] == "apache-2.0")
check("base model from the tag, finetune marker stripped", mapped["base_models"] == ["vllm-sr/d3-mini"])
check("architecture falls back to the config model_type", mapped["architecture"] == "qwen3_5_text")
check("a FLM text-tower model_type resolves to the class the registry keys that family by",
      mapped["registry_class"] == "Qwen35ForCausalLM" and mapped["registry_gguf"] == "qwen35")
check("the resolved entry's backends decide the routes (qwen35 is GPU-only in this registry)",
      mapped["mapped"] and mapped["hrx"] and not mapped["npu"])
check("a name the registry does not know stays unmapped", fw.resolve_family("ornith_text", ARCHS) == "")

# 2. A family the registry does not map is the alert, and it is the only thing that alerts.
unknown = fw.repo_summary(unknown_repo, ARCHS)
check("unknown family is unmapped", not unknown["mapped"] and unknown["registry_class"] == "")
check("unknown family keeps its own name", unknown["architecture"] == "ornith_text")
rc, text = report_of([unknown], [], [unknown], [])
check("unknown family -> exit 1 and !! NEW FLM", rc == 1 and "!! NEW FLM FastFlowLM/Ornith-1.5-35B-NPU2" in text)
rc, text = report_of([mapped], [], [mapped], [])
check("mapped family -> exit 0", rc == 0 and "+ REPO FastFlowLM/D3-Mini-NPU2" in text)
check("mapped family prints its routes and file sizes",
      "registry Qwen35ForCausalLM -> qwen35" in text and "npu " not in text and "GiB" in text)

# 3. FastFlowLM's own feature naming is surfaced, not interpreted.
featured = dict(mapped_repo, id="FastFlowLM/Qwen3.8-27B-layerP-NPU2")
featured["tags"] = ["qwen3_5_text", "q4nx", "license:apache-2.0", "lax"]
check("FLM feature markers come from the id and tags",
      fw.repo_summary(featured, ARCHS)["flm_features"] == ["lax", "layerp"])
check("an NPU-served family still reports its npu route when the registry lists one",
      fw.repo_summary(dict(mapped_repo, config={"model_type": "qwen3_5_moe"}), ARCHS)["npu"])
check("no marker, no feature", mapped["flm_features"] == [])

# 4. GitHub releases: tag/date/assets, and the container-format flag.
summaries = [fw.release_summary(release) for release in releases]
check("release tag and date", summaries[0]["tag"] == "v1.0.10" and summaries[0]["published_at"].startswith("2026-"))
check("assets are listed", len(summaries[0]["assets"]) > 0)
check("a release whose notes name q4nx/chunk is flagged",
      "q4nx" in summaries[1]["kernel_format"] and "chunk" in summaries[1]["kernel_format"])
check("a release that only fixes features is not flagged", summaries[0]["kernel_format"] == [])

# 5. State keeps deltas: a second run with the same payloads reports nothing new.
state = {"repos": {}, "releases": []}
fresh_repos = fw.new_repos(state, [mapped, unknown])
check("first run sees both repos", len(fresh_repos) == 2)
fw.remember(state, [mapped, unknown], summaries)
check("state records the repo stamps and tags",
      state["repos"]["FastFlowLM/D3-Mini-NPU2"] == mapped["lastModified"]
      and state["releases"] == [summary["tag"] for summary in summaries])
check("second run is quiet", not fw.new_repos(state, [mapped, unknown]) and not fw.new_releases(state, summaries))
rc, text = report_of([mapped, unknown], summaries, [], [])
check("a quiet run says so and stays green", rc == 0 and "nothing new since the last run" in text)
bumped = dict(unknown, lastModified="2026-10-26T00:00:00.000Z")
check("a moved lastModified is a delta again", [repo["id"] for repo in fw.new_repos(state, [mapped, bumped])]
      == ["FastFlowLM/Ornith-1.5-35B-NPU2"])

# 6. --full lists everything even when nothing is new, and the summary line carries the counts.
rc, text = report_of([mapped, unknown], summaries, [], [], full=True)
check("--full lists unchanged repos and releases",
      "+ REPO FastFlowLM/D3-Mini-NPU2" in text and "+ RELEASE v1.0.10" in text and rc == 1)
rc, text = report_of([mapped], summaries, [mapped], [summaries[1]])
check("the summary line names the counts and the format move",
      "1 new/changed repo(s)" in text and "1 new release(s)" in text
      and "1 release(s) naming the container format" in text)

# 7. The org listing shape the tool walks, and the committed registry it reads.
listing = [fw.repo_summary(repo, ARCHS) for repo in [mapped_repo, unknown_repo]]
check("every listed repo becomes a summary",
      [repo["id"] for repo in listing] == [repo["id"] for repo in org_listing])
archs = fw.load_registry()
check("committed registry maps a Qwen3.5 family the NPU route serves",
      any("npu" in (entry.get("backends") or ()) for cls, entry in archs.items()
          if cls.startswith("Qwen3_5")))
check("committed registry is keyed by HF class", any("ForCausalLM" in cls for cls in archs))

# 8. State round-trips through the file the daily run uses.
state_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "fflm-watch-test-state.json")
if os.path.exists(state_path):
    os.unlink(state_path)
fw.save_state(state_path, state)
reloaded = fw.load_state(state_path)
check("state round-trips", reloaded["repos"] == state["repos"] and reloaded["releases"] == state["releases"])
check("a missing state file starts empty", fw.load_state(state_path + ".absent")["repos"] == {})
os.unlink(state_path)

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
