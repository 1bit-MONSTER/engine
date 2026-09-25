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
"""Run and check models per backend; record the results in registry/checked.json.

usage: tools/registry_check.py path/to/1bit --models DIR [--npu-models DIR] [--only BACKEND]

For every row of registry/check_models.tsv (backend, model path under DIR or --npu-models),
runs tests/serve_e2e.sh: the model must load, answer "Paris" to a fixed question under
temperature 0, and stream. The architecture recorded is the one the backend loads: the GGUF
`general.architecture` for vulkan, hrx and zinc, and config.json's model_type for npu.

Each result replaces the previous one for the same backend and model, so re-running a
subset keeps the rest. Failures are recorded too, and only passes count as checked.
"""
import argparse
import datetime
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "registry/checked.json")
LIST = os.path.join(ROOT, "registry/check_models.tsv")


def gguf_arch(path):
    """general.architecture from a GGUF header (v2/v3)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return ""
        version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        if version < 2:
            return ""

        def s():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        for _ in range(n_kv):
            key = s()
            (t,) = struct.unpack("<I", f.read(4))
            if t == 8:
                val = s()
                if key == "general.architecture":
                    return val
            elif t == 9:
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if et == 8:
                    for _ in range(n):
                        s()
                else:
                    f.seek(sizes[et] * n, 1)
            else:
                f.seek(sizes[t], 1)
    return ""


def npu_model_type(path):
    try:
        return json.load(open(os.path.join(path, "config.json"))).get("model_type", "")
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("--models", required=True)
    ap.add_argument("--npu-models", help="NPU model directories (default: DIR/npu)")
    ap.add_argument("--only", help="run only this backend")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()

    a.npu_models = a.npu_models or os.path.join(a.models, "npu")
    rows = []
    for line in open(LIST):
        line = line.strip()
        if line and not line.startswith("#"):
            backend, model, *extra = line.split("\t")
            rows.append((backend, model, extra))
    data = json.load(open(OUT)) if os.path.exists(OUT) else {"results": []}
    results = {(r["backend"], r["model"]): r for r in data["results"]}
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short=12", "HEAD"],
                            capture_output=True, text=True).stdout.strip()

    for backend, model, extra in rows:
        if a.only and backend != a.only:
            continue
        path = os.path.join(a.npu_models if backend == "npu" else a.models, model)
        if not os.path.exists(path):
            print(f"skip {backend} {model}: not found")
            continue
        arch = npu_model_type(path) if backend == "npu" else gguf_arch(path)
        cmd = [os.path.join(ROOT, "tests/serve_e2e.sh"), a.bin, path, backend] + extra
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            out, ok = p.stdout, p.returncode == 0
        except subprocess.TimeoutExpired as e:
            out, ok = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), False
            out += "\nFAIL timeout"
        said = next((l.split("said:", 1)[1].strip() for l in out.splitlines() if "said:" in l), "")
        failed = [l[5:].strip() for l in out.splitlines() if l.startswith("FAIL ")]
        results[(backend, model)] = {
            "backend": backend, "model": model, "arch": arch, "passed": ok,
            "said": said[:120], "failed": failed,
            "date": datetime.date.today().isoformat(), "engine": commit,
        }
        print(f"{'PASS' if ok else 'FAIL'} {backend:6} {arch:10} {model}  said: {said[:60]!r}")
        data["results"] = sorted(results.values(), key=lambda r: (r["arch"], r["backend"], r["model"]))
        with open(OUT, "w") as f:  # after every run, so an interrupted sweep keeps what finished
            json.dump(data, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
