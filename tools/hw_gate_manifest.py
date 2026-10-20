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
"""The KLD half of the HRX hardware gate (scripts/hw-gate.sh, docs/hw-gate.md).

The base logits are too big for git (hundreds of MB per model), so they stay on the gate box
under $HW_GATE_STORE/kld and tests/golden/hw-gate/manifest.json records what they are: their
sha256, a fingerprint of the model file and of the text they were computed from, the pins, and
the thresholds a run is judged by.

  hw_gate_manifest.py set    MANIFEST NAME BASE MODEL TEXT   after --record
  hw_gate_manifest.py verify MANIFEST NAME BASE MODEL TEXT   exit 1 if any of the three changed
  hw_gate_manifest.py judge  MANIFEST NAME LOG               parse llama-perplexity, exit 1 over a threshold
  hw_gate_manifest.py memcheck MODEL|- LLAMA_CPP_DIR         tools/mem_budget.py pre-check before a GPU step
"""
import hashlib
import json
import os
import re
import subprocess
import sys

# Defaults for a newly recorded model. A rerun at the recording pin gives KLD 0 (HRX is
# deterministic for a fixed graph); the thresholds leave room for a kernel that legitimately
# changes rounding and still catch a wrong kernel by orders of magnitude (docs/hw-gate.md).
DEFAULT_LIMITS = {"max_mean_kld": 0.01, "max_kld_p99": 0.2, "min_same_top_p": 95.0}


def sha256(path, limit=None):
    h = hashlib.sha256()
    left = limit
    with open(path, "rb") as f:
        while True:
            n = 1 << 20 if left is None else min(1 << 20, left)
            if n == 0:
                break
            b = f.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def model_fingerprint(path):
    # the first 64 MiB holds the GGUF header, metadata and the first tensors; hashing 17 GB per run is too slow
    return {"file": os.path.basename(path), "size": os.path.getsize(path), "sha256_first_64MiB": sha256(path, 64 << 20)}


def pins(root):
    out = {}
    for sub in ("llama.cpp", "hrx-system"):
        try:
            line = subprocess.run(["git", "-C", root, "ls-tree", "HEAD", "third_party/" + sub],
                                  capture_output=True, text=True, check=True).stdout.split()
            out[sub] = line[2]
        except (subprocess.CalledProcessError, IndexError):
            out[sub] = "unknown"
    return out


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"about": "HRX hardware gate KLD bases (docs/hw-gate.md). The .kld files live on the gate box, "
                     "not in git; this file pins what they are.", "kld": {}}


def cmd_set(manifest, name, base, model, text):
    m = load(manifest)
    old = m["kld"].get(name, {})
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m["kld"][name] = {
        "base_sha256": sha256(base), "base_bytes": os.path.getsize(base),
        "model": model_fingerprint(model), "text": {"file": os.path.basename(text), "sha256": sha256(text)},
        "args": "-c 512 --chunks 8 -b 512 -ub 512 --device HRX0 -ngl 99",
        "recorded_at": pins(root), "limits": old.get("limits", dict(DEFAULT_LIMITS)),
    }
    with open(manifest, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
        f.write("\n")
    return 0


def cmd_verify(manifest, name, base, model, text):
    e = load(manifest)["kld"].get(name)
    if e is None:
        print(f"no KLD golden for {name} in {manifest}: record one with scripts/hw-gate.sh --record")
        return 1
    if not os.path.exists(base):
        print(f"base logits missing: {base} (record them at the release pin, docs/hw-gate.md)")
        return 1
    if os.path.getsize(base) != e["base_bytes"] or sha256(base) != e["base_sha256"]:
        print(f"base logits {base} do not match the manifest sha256")
        return 1
    if model_fingerprint(model) != e["model"]:
        print(f"model file {model} is not the one the base was recorded with")
        return 1
    if sha256(text) != e["text"]["sha256"]:
        print(f"text {text} is not the one the base was recorded with")
        return 1
    return 0


def stat(log, label):
    m = re.search(r"^" + re.escape(label) + r"\s*:\s*([-+0-9.eE]+)", log, re.M)
    return float(m.group(1)) if m else None


def cmd_judge(manifest, name, logpath):
    limits = load(manifest)["kld"].get(name, {}).get("limits", DEFAULT_LIMITS)
    with open(logpath, errors="replace") as f:
        log = f.read()
    mean = stat(log, "Mean    KLD")
    p99 = stat(log, "99.0%   KLD")
    top = stat(log, "Same top p")
    if mean is None or top is None:
        print("no KL divergence statistics in the llama-perplexity log")
        return 1
    bad = []
    if mean != mean or mean > limits["max_mean_kld"]:
        bad.append(f"mean KLD {mean:.6f} > {limits['max_mean_kld']}")
    if p99 is not None and (p99 != p99 or p99 > limits["max_kld_p99"]):
        bad.append(f"KLD p99 {p99:.5f} > {limits['max_kld_p99']}")
    if top < limits["min_same_top_p"]:
        bad.append(f"same top token {top:.2f}% < {limits['min_same_top_p']}%")
    msg = f"mean KLD {mean:.6f}, p99 {p99 if p99 is not None else float('nan'):.5f}, same top {top:.2f}%"
    print(msg + ("; " + "; ".join(bad) if bad else ""))
    return 1 if bad else 0


FAULT_ZONE_GIB = 95.0   # engine#315: faults only seen at GTT >= 95 GiB
OPS_HEADROOM_GIB = 6.0  # test-backend-ops allocates a few GiB at most


def cmd_memcheck(model, llama_dir):
    """Exit 1 when the step would take the box into the #315 zone or past MemAvailable.
    The estimator failing is not a refusal: scripts/mem-guard.sh still guards the step."""
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mem_budget.py")
    g = float(1 << 30)
    if model == "-":
        p = subprocess.run([sys.executable, tool, "box", "--json"], capture_output=True, text=True)
        try:
            led = json.loads(p.stdout)
        except ValueError:
            return 0
        used = led.get("gtt_used", 0) / g
        avail = led["mem_available"] / g
        if used + OPS_HEADROOM_GIB >= FAULT_ZONE_GIB or avail < 16:
            print(f"memory pre-check: GTT used {used:.1f} GiB, MemAvailable {avail:.1f} GiB: too close to "
                  f"the engine#315 zone, not now", file=sys.stderr)
            return 1
        print(f"memory pre-check: GTT used {used:.1f} GiB, MemAvailable {avail:.1f} GiB: ok", file=sys.stderr)
        return 0
    p = subprocess.run([sys.executable, tool, "model", model, "-c", "8192", "--no-fit-params",
                        "--llama-cpp", llama_dir, "--json"], capture_output=True, text=True)
    try:
        r = json.loads(p.stdout)
    except ValueError:
        return 0
    msg = (f"memory pre-check: {os.path.basename(model)} needs {r['need'] / g:.1f} GiB, GTT after load "
           f"{r['gtt_used_after'] / g:.1f} GiB, MemAvailable {r['mem_available'] / g:.1f} GiB")
    if not r["fits"] or r["fault_zone_after"]:
        print(msg + ": not now", file=sys.stderr)
        return 1
    print(msg + ": ok", file=sys.stderr)
    return 0


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd, args = argv[1], argv[2:]
    if cmd == "set" and len(args) == 5:
        return cmd_set(*args)
    if cmd == "verify" and len(args) == 5:
        return cmd_verify(*args)
    if cmd == "memcheck" and len(args) == 2:
        return cmd_memcheck(*args)
    if cmd == "judge" and len(args) == 3:
        return cmd_judge(*args)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
