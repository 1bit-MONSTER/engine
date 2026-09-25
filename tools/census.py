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
"""The HF census: how many text-generation models this engine maps, and how many it has checked.

usage: tools/census.py [--max-pages N] [--out registry/census.json]
       tools/census.py --report    # recompute coverage from the saved counts, no network
       tools/census.py --pr-body   # print the census PR's markdown from the saved file

Walks every page of huggingface.co/api/models?pipeline_tag=text-generation&config=true
(the config comes inline, so there is no per-model fetch) and counts models by their
config's first `architectures` entry. Coverage is then read against
registry/architectures.json (mapped: a backend's code accepts the architecture) and
registry/checked.json (checked: a model of that architecture ran and passed serve_e2e here).
The two counts are reported separately and are never added together.

The unmapped architectures with the most models are listed, so the next mapping to add is
the one that covers the most models.
"""
import argparse
import datetime
import json
import os
import re
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://huggingface.co/api/models?pipeline_tag=text-generation&config=true&limit=1000"
REG = os.path.join(ROOT, "registry")
_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


def sweep(max_pages):
    counts, total, no_arch, url, pages = {}, 0, 0, API, 0
    while url and (max_pages is None or pages < max_pages):
        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "1bit-engine-census"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    batch = json.load(r)
                    link = r.headers.get("Link", "")
                break
            except Exception as e:  # rate limits and transient errors: back off and retry
                if attempt == 5:
                    raise
                print(f"page {pages + 1}: {e}; retrying", file=sys.stderr)
                time.sleep(30 * (attempt + 1))
        for m in batch:
            total += 1
            archs = (m.get("config") or {}).get("architectures") or []
            a = archs[0] if archs and isinstance(archs[0], str) else ""
            if not a:
                no_arch += 1
                continue
            counts[a] = counts.get(a, 0) + 1
        pages += 1
        m = _NEXT.search(link)
        url = m.group(1) if m else None
        if pages % 50 == 0:
            print(f"{pages} pages, {total} models", file=sys.stderr)
    return {"total": total, "no_arch": no_arch, "pages": pages, "complete": url is None,
            "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))}


def coverage(raw):
    archs = json.load(open(os.path.join(REG, "architectures.json")))["architectures"]
    checked_path = os.path.join(REG, "checked.json")
    results = json.load(open(checked_path))["results"] if os.path.exists(checked_path) else []
    passed = {(r["backend"], r["arch"]) for r in results if r["passed"]}

    def checked_on(hf):
        a = archs.get(hf)
        if not a:
            return []
        out = []
        for b in a["backends"]:
            key = a.get("npu_model_type", "") if b == "npu" else a["gguf"]
            if (b, key) in passed:
                out.append(b)
        return out

    with_arch = raw["total"] - raw["no_arch"]
    mapped = checked = 0
    per_backend = {b: {"mapped": 0, "checked": 0} for b in ("vulkan", "hrx", "zinc", "npu")}
    unmapped = []
    for hf, n in raw["counts"].items():
        a = archs.get(hf)
        if a and a["backends"]:
            mapped += n
            for b in a["backends"]:
                per_backend[b]["mapped"] += n
        else:
            unmapped.append((hf, n))
        c = checked_on(hf)
        if c:
            checked += n
            for b in c:
                per_backend[b]["checked"] += n
    pct = lambda x: round(100.0 * x / with_arch, 2) if with_arch else 0.0
    return {
        "models": raw["total"], "with_architecture": with_arch,
        "architectures_seen": len(raw["counts"]),
        "mapped_models": mapped, "mapped_pct": pct(mapped),
        "checked_models": checked, "checked_pct": pct(checked),
        "per_backend": {b: v | {"mapped_pct": pct(v["mapped"]), "checked_pct": pct(v["checked"])}
                        for b, v in per_backend.items()},
        "top_unmapped": [{"architecture": h, "models": n} for h, n in unmapped[:25]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--pr-body", action="store_true")
    ap.add_argument("--out", default=os.path.join(REG, "census.json"))
    a = ap.parse_args()
    if a.pr_body:
        c = json.load(open(a.out))["coverage"]
        print(f"{c['models']} text-generation models on HF, {c['with_architecture']} with an architecture "
              f"({c['architectures_seen']} architectures).\n")
        print("| | models | share |\n|---|---|---|")
        print(f"| mapped (a backend's code accepts the architecture) | {c['mapped_models']} | {c['mapped_pct']}% |")
        print(f"| checked (passed serve_e2e on Strix Halo) | {c['checked_models']} | {c['checked_pct']}% |")
        print("\n| backend | mapped | checked |\n|---|---|---|")
        for b, v in c["per_backend"].items():
            print(f"| {b} | {v['mapped_pct']}% | {v['checked_pct']}% |")
        print("\nUnmapped architectures with the most models:\n")
        for u in c["top_unmapped"][:10]:
            print(f"- `{u['architecture']}`: {u['models']}")
        return 0
    if a.report:
        data = json.load(open(a.out))
        raw = data["raw"]
    else:
        raw = sweep(a.max_pages)
        data = {"date": datetime.date.today().isoformat()}
    data["coverage"] = coverage(raw)
    data["raw"] = raw
    with open(a.out, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    c = data["coverage"]
    print(f"{c['models']} models, {c['with_architecture']} with an architecture "
          f"({c['architectures_seen']} architectures): mapped {c['mapped_models']} ({c['mapped_pct']}%), "
          f"checked {c['checked_models']} ({c['checked_pct']}%)"
          + ("" if raw["complete"] else f" [partial: {raw['pages']} pages]"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
