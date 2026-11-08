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
#
# catalogue-manifest.py <server_models.json> <out.txt>
#
# Writes the onebit subset of the payload's Lemonade catalogue as the manifest release-check.sh parses:
#
#     mode=live
#     hrx <model id>          (recipe onebit, no NPU backend: these run on the GPU through HRX)
#     npu <model id>          (recipe_options.onebit_backend == npu)
#
# Same format and same walk as os/mini/mkimage.sh writes for the os/mini image, so the models row of
# 1bit-release-check is answered the same way on both artifacts. The walk handles both catalogue shapes
# (a dict keyed by model id - what the release vendors - and a list of records with an "id" field): the
# first version of that block only looked for an "id" field, so it matched nothing in the keyed catalogue
# and wrote an empty manifest, which made the row pass vacuously.
#
# The NPU marker lives INSIDE recipe_options ({"onebit_backend": "npu"}); reading it from the record's top
# level missed all four NPU entries and labelled them hrx. Both places are checked.
import json
import sys


def models(node):
    if isinstance(node, list):
        for x in node:
            yield from models(x)
    elif isinstance(node, dict):
        if "id" in node and isinstance(node.get("id"), str):
            yield node
        else:
            for k, v in node.items():
                if isinstance(v, dict) and ("recipe" in v or "onebit_backend" in v):
                    rec = dict(v)
                    rec.setdefault("id", k)
                    if isinstance(rec.get("id"), str):
                        yield rec
                else:
                    yield from models(v)


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: catalogue-manifest.py <server_models.json> <out.txt>")
    cat, out = argv[1], argv[2]
    try:
        data = json.load(open(cat))
    except Exception as e:  # noqa: BLE001 - any read/parse failure is the same refusal
        sys.exit("catalogue-manifest: %s is not readable JSON (%s)" % (cat, e))
    rows, seen = [], set()
    for m in models(data):
        opts = m.get("recipe_options")
        backend = m.get("onebit_backend")
        if isinstance(opts, dict) and opts.get("onebit_backend"):
            backend = opts.get("onebit_backend")
        if m.get("recipe") == "onebit":
            kind = "npu" if backend == "npu" else "hrx"
        elif backend == "npu":
            kind = "npu"
        else:
            continue
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        rows.append("%s %s" % (kind, m["id"]))
    rows.sort(key=lambda r: (r.split()[0] != "hrx", r))
    with open(out, "w") as f:
        f.write("mode=live\n")
        for r in rows:
            f.write(r + "\n")
    print("onebit-models: %d hrx + %d npu from %s"
          % (sum(r.startswith("hrx ") for r in rows), sum(r.startswith("npu ") for r in rows), cat))


if __name__ == "__main__":
    main(sys.argv)
