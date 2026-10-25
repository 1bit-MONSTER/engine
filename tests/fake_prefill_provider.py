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
# fake_prefill_provider.py: a prefill provider for `1bit serve --device hybrid` (docs/hybrid.md) that
# has no NPU. It honours the provider contract (app/serve_hybrid.h):
#
#   --tokens <ids file> --out <state file> --model <model> [--elfs DIR]
#   stdout: one JSON line {"n_tokens": N, "argmax": id, "prefill_ms": x}; exit 0
#
# Two ways to make the state file:
#   FAKE_PREFILL_COPY=1   copy the ids file (tests/fake_backend.py's restore counts its lines;
#                         tests/hybrid_route.sh, no model needed)
#   default               ask the decoding llama-server itself (ONEBIT_HYBRID_BACKEND_URL, which serve
#                         sets for every provider) to prefill the ids into slot 0 and save the slot: the
#                         state file is then a real sequence-state file for a real model, made by the
#                         same model that will decode from it (tests/hybrid_e2e.sh). The file lands in
#                         the backend's --slot-save-path, which is where --out points.
#
# A real provider computes the KV itself (the NPU engine) and writes the same file format.
import json, os, shutil, sys, time, urllib.request

args = sys.argv[1:]
opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
tokens_path, out_path = opt["--tokens"], opt["--out"]
with open(tokens_path) as f:
    ids = [int(line) for line in f if line.strip()]
t0 = time.time()

if os.environ.get("FAKE_PREFILL_COPY"):
    shutil.copyfile(tokens_path, out_path)
    print(json.dumps({"n_tokens": len(ids), "argmax": ids[-1], "prefill_ms": round((time.time() - t0) * 1000, 1)}), flush=True)
    sys.exit(0)

url = os.environ["ONEBIT_HYBRID_BACKEND_URL"]


def post(path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


# the prompt through the backend, no generation: the slot then holds exactly the prompt's cells
reply = post("/completion", {"prompt": ids, "n_predict": 0, "id_slot": 0, "cache_prompt": False, "temperature": 0})
# the slot's state, by bare name into the backend's --slot-save-path (where --out points)
saved = post("/slots/0?action=save", {"filename": os.path.basename(out_path)})
n = int(saved.get("n_saved", 0))
# the slot is emptied again, so the file is the only way the KV reaches the decode
post("/slots/0?action=erase", {})
if n != len(ids):
    print(f"fake_prefill_provider: saved {n} tokens for {len(ids)} ids", file=sys.stderr, flush=True)
    sys.exit(1)
print(json.dumps({"n_tokens": n, "argmax": -1, "prefill_ms": round((time.time() - t0) * 1000, 1),
                  "prompt_n": reply.get("timings", {}).get("prompt_n")}), flush=True)
