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
"""Drop the onebit NPU catalogue entries whose kernel set the payload does not carry (engine#3287).

    npu_catalogue_filter.py <server_models.json> <npu kernels dir> <config/npu-catalogue-sets.json>
    npu_catalogue_filter.py --self-test

The Lemonade fork's catalogue lists every onebit NPU model the project has worked on; a release ships kernel sets for
some of them. v2026.47 offered Qwen3-1.7B-NPU-1bit with no set in the payload, and loading it ended in a 500 that blamed
the NPU array ("another client may still be releasing the array"), not the missing set. This keeps only the NPU
entries whose set (by geometry, config/npu-catalogue-sets.json) is in <npu kernels dir> with a manifest, and drops an
NPU entry the map does not know: a catalogue row nobody mapped is a row nobody proved. HRX entries and every
non-onebit entry are left exactly as they are. The file is rewritten in place only when something was dropped, and
each drop is printed with its reason.
"""
import json
import os
import sys
import tempfile


def is_npu(entry):
    return (isinstance(entry, dict) and entry.get("recipe") == "onebit"
            and isinstance(entry.get("recipe_options"), dict)
            and entry["recipe_options"].get("onebit_backend") == "npu")


def has_set(kernels, name):
    d = os.path.join(kernels, name)
    return bool(name) and (os.path.isfile(os.path.join(d, "manifest")) or os.path.isfile(os.path.join(d, "manifest.json")))


def plan(catalogue, kernels, sets):
    """-> (kept catalogue, [(model, reason)] dropped)"""
    kept, dropped = {}, []
    for model, entry in catalogue.items():
        if not is_npu(entry):
            kept[model] = entry
            continue
        if model not in sets:
            dropped.append((model, "not in config/npu-catalogue-sets.json (an unmapped NPU entry is not offered)"))
        elif sets[model] is None:
            dropped.append((model, "needs a route this payload does not ship (mapped to null)"))
        elif not has_set(kernels, sets[model]):
            dropped.append((model, f"no kernel set {sets[model]} in the payload"))
        else:
            kept[model] = entry
    return kept, dropped


def run(cat_path, kernels, map_path):
    with open(cat_path) as f:
        catalogue = json.load(f)
    with open(map_path) as f:
        sets = json.load(f)["sets"]
    kept, dropped = plan(catalogue, kernels, sets)
    for model, why in dropped:
        print(f"npu_catalogue_filter: dropped {model}: {why}")
    npu_kept = [m for m, e in kept.items() if is_npu(e)]
    print(f"npu_catalogue_filter: {len(npu_kept)} NPU entr(y/ies) offered: {' '.join(npu_kept) or '<none>'}")
    if dropped:
        tmp = cat_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(kept, f, indent=2)
            f.write("\n")
        os.replace(tmp, cat_path)
    return 0


def self_test():
    bad = 0

    def check(cond, what):
        nonlocal bad
        print(("ok   " if cond else "FAIL ") + what)
        bad += 0 if cond else 1

    npu = {"recipe": "onebit", "recipe_options": {"onebit_backend": "npu"}}
    hrx = {"recipe": "onebit", "recipe_options": {}}
    other = {"recipe": "llamacpp"}
    with tempfile.TemporaryDirectory() as t:
        k = os.path.join(t, "kernels")
        os.makedirs(os.path.join(k, "1024x3072x16x8"))
        open(os.path.join(k, "1024x3072x16x8", "manifest"), "w").write("npumanifest 2\n")
        os.makedirs(os.path.join(k, "2560x9728x32x8"))  # a directory with no manifest is not a set
        sets = {"A-NPU": "1024x3072x16x8", "B-NPU": "2048x6144x16x8", "C-NPU": "2560x9728x32x8", "D-NPU": None}
        cat = {"A-NPU": npu, "B-NPU": npu, "C-NPU": npu, "D-NPU": npu, "E-NPU": npu, "H": hrx, "L": other}
        kept, dropped = plan(cat, k, sets)
        names = dict(dropped)
        check("A-NPU" in kept, "an NPU entry whose set ships (with a manifest) is kept")
        check("B-NPU" in names and "no kernel set" in names["B-NPU"], "an NPU entry with no set shipped is dropped")
        check("C-NPU" in names, "a set directory without a manifest does not count as shipped")
        check("D-NPU" in names and "null" in names["D-NPU"], "an entry mapped to null (a route not shipped) is dropped")
        check("E-NPU" in names and "not in" in names["E-NPU"], "an NPU entry the map does not know is dropped")
        check(kept.get("H") is hrx and kept.get("L") is other, "HRX and non-onebit entries are untouched")
        check(list(kept) == ["A-NPU", "H", "L"], "the kept catalogue keeps its order")
        os.makedirs(os.path.join(k, "legacy"))
        open(os.path.join(k, "legacy", "manifest.json"), "w").write("{}")
        check(has_set(k, "legacy"), "a format-1 set (manifest.json) counts too")
        p = os.path.join(t, "server_models.json")
        json.dump(cat, open(p, "w"))
        m = os.path.join(t, "map.json")
        json.dump({"about": "x", "sets": sets}, open(m, "w"))
        run(p, k, m)
        check(sorted(json.load(open(p))) == ["A-NPU", "H", "L"], "run rewrites the catalogue without the dropped rows")
        before = open(p).read()
        run(p, k, m)
        check(open(p).read() == before, "a second run changes nothing")
    print("npu_catalogue_filter self-test:", "FAIL" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        sys.exit(self_test())
    if len(sys.argv) != 4:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        sys.exit(1)
    sys.exit(run(*sys.argv[1:]))
