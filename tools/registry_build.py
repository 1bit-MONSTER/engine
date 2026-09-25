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
"""Build registry/architectures.json: which backends of this engine map each HF architecture.

usage: tools/registry_build.py [--check] [--out registry/architectures.json]

Every entry is read from a pinned source, never typed in by hand:
- HF architecture -> GGUF architecture: the `@ModelBase.register(...)` classes in llama.cpp's
  converter (convert_hf_to_gguf.py and conversion/*.py), in the upstream pin and in our HRX
  fork; the upstream pin wins where both name one.
- vulkan: the GGUF architecture is in the upstream pin's src/llama-arch.cpp (LLM_ARCH_NAMES).
- hrx: the same, in our HRX fork (third_party/llama.cpp).
- zinc: the GGUF architecture is one ZINC's parseArchitecture (src/model/config.zig) accepts.
- npu: the fast lane's model types (NPU_MODEL_TYPES below, matching npu/). The NPU runs Q4NX
  model directories, so it matches on HF model_type, not on a GGUF architecture.

"Mapped" means the backend's own code accepts the architecture. Whether a model of that
architecture was run and checked here is recorded separately, in registry/checked.json.

--check exits 1 when the file on disk differs from what the pinned sources give.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = "third_party/llama.cpp-vulkan"
HRX = "third_party/llama.cpp"
ZINC = "third_party/zinc"

# The NPU fast lane serves Qwen3 dense Q4NX directories (docs/npu.md). HF architecture ->
# model_type, as the directory's config.json names them.
NPU_ARCHS = {"Qwen3ForCausalLM": "qwen3"}


def pin(path):
    out = subprocess.run(["git", "-C", ROOT, "ls-tree", "HEAD", path], capture_output=True, text=True)
    parts = out.stdout.split()
    return parts[2] if len(parts) >= 3 else ""


def converter_files(tree):
    files = [os.path.join(tree, "convert_hf_to_gguf.py")]
    conv = os.path.join(tree, "conversion")
    if os.path.isdir(conv):
        files += sorted(os.path.join(conv, f) for f in os.listdir(conv) if f.endswith(".py"))
    return [f for f in files if os.path.isfile(f)]


def model_arch_names(tree):
    """MODEL_ARCH.X -> "name" from gguf-py/gguf/constants.py."""
    src = open(os.path.join(tree, "gguf-py/gguf/constants.py")).read()
    body = src[src.index("MODEL_ARCH_NAMES"):]
    body = body[:body.index("}")]
    return dict(re.findall(r"MODEL_ARCH\.([A-Z0-9_]+)\s*:\s*\"([^\"]+)\"", body))


def hf_to_gguf(tree):
    """{HF architecture: GGUF architecture} from the converter's registered classes."""
    names = model_arch_names(tree)
    classes = {}  # class name -> (bases, model_arch or None, registered HF names)
    for path in converter_files(tree):
        mod = ast.parse(open(path).read(), path)
        for node in ast.walk(mod):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else ""
                     for b in node.bases]
            arch = None
            for st in node.body:
                if (isinstance(st, (ast.Assign, ast.AnnAssign)) and isinstance(st.value, ast.Attribute)
                        and isinstance(st.value.value, ast.Attribute) and st.value.value.attr == "MODEL_ARCH"):
                    targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                    if any(isinstance(t, ast.Name) and t.id == "model_arch" for t in targets):
                        arch = st.value.attr
            hf = []
            for d in node.decorator_list:
                if (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "register"):
                    hf += [a.value for a in d.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            classes[node.name] = (bases, arch, hf)

    def resolve(name, seen=()):
        if name not in classes or name in seen:
            return None
        bases, arch, _ = classes[name]
        if arch:
            return arch
        for b in bases:
            a = resolve(b, seen + (name,))
            if a:
                return a
        return None

    out = {}
    for cname, (_, _, hf) in classes.items():
        arch = resolve(cname)
        if not arch or arch == "MMPROJ" or arch not in names:
            continue  # vision/audio projector classes carry no text architecture
        for h in hf:
            out.setdefault(h, names[arch])
    return out


def runtime_archs(tree):
    src = open(os.path.join(tree, "src/llama-arch.cpp")).read()
    body = src[src.index("LLM_ARCH_NAMES"):]
    body = body[:body.index("};")]
    return set(re.findall(r"\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*\"([^\"]+)\"\s*\}", body)) - {"clip"}


def zinc_archs(tree):
    src = open(os.path.join(tree, "src/model/config.zig")).read()
    body = src[src.index("pub fn parseArchitecture"):]
    body = body[:body.index("return .unknown")]
    return set(re.findall(r"std\.mem\.eql\(u8,\s*arch_str,\s*\"([^\"]+)\"\)", body))


def build():
    for t in (UPSTREAM, HRX, ZINC):
        if not os.path.isdir(os.path.join(ROOT, t, "src")):
            sys.exit(f"registry_build: {t} is not checked out (git submodule update --init {t})")
    up, hrx = os.path.join(ROOT, UPSTREAM), os.path.join(ROOT, HRX)
    mapping = hf_to_gguf(hrx)
    mapping.update(hf_to_gguf(up))
    run_up, run_hrx, run_zinc = runtime_archs(up), runtime_archs(hrx), zinc_archs(os.path.join(ROOT, ZINC))

    archs = {}
    for hf in sorted(set(mapping) | set(NPU_ARCHS)):
        g = mapping.get(hf, "")
        backends = []
        if g in run_hrx:
            backends.append("hrx")
        if hf in NPU_ARCHS:
            backends.append("npu")
        if g in run_up:
            backends.append("vulkan")
        if g in run_zinc:
            backends.append("zinc")
        entry = {"gguf": g, "backends": backends}
        if hf in NPU_ARCHS:
            entry["npu_model_type"] = NPU_ARCHS[hf]
        archs[hf] = entry

    return {
        "about": "HF architecture -> GGUF architecture and the backends whose code accepts it. "
                 "Generated by tools/registry_build.py from the pinned sources; do not edit.",
        "sources": {"llama.cpp (vulkan)": pin(UPSTREAM), "llama.cpp (hrx)": pin(HRX), "zinc": pin(ZINC)},
        "counts": {b: sum(b in a["backends"] for a in archs.values()) for b in ("hrx", "npu", "vulkan", "zinc")}
                  | {"architectures": len(archs), "mapped": sum(bool(a["backends"]) for a in archs.values())},
        "architectures": archs,
    }


def gap_violations(registry):
    """Classes reviewed as not aliases (registry/significant.json, docs/arch-gaps.md) that the
    registry maps without a recorded reason ('mapped_ok'): each needs a person's look."""
    sig = json.load(open(os.path.join(ROOT, "registry/significant.json")))["classes"]
    mapped = registry["architectures"]
    return [(cls, mapped[cls]) for cls, e in sig.items() if cls in mapped and not e.get("mapped_ok")]


def report_gaps(registry):
    bad = gap_violations(registry)
    for cls, m in bad:
        print(f"{cls} is mapped (gguf {m['gguf']}, {', '.join(m['backends'])}) but docs/arch-gaps.md reviewed it as "
              f"not an alias: check that the pinned backend really implements it, then record why in "
              f"registry/significant.json ('mapped_ok')")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "registry/architectures.json"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-gaps", action="store_true",
                    help="only check the committed registry against registry/significant.json (no pins needed)")
    a = ap.parse_args()
    if a.check_gaps:
        bad = report_gaps(json.load(open(a.out)))
        if not bad:
            n = len(json.load(open(os.path.join(ROOT, "registry/significant.json")))["classes"])
            print(f"none of the {n} reviewed significant classes is mapped without a reason")
        return 1 if bad else 0
    text = json.dumps(build(), indent=1, sort_keys=False) + "\n"
    report_gaps(json.loads(text))
    if a.check:
        old = open(a.out).read() if os.path.exists(a.out) else ""
        if old != text:
            print(f"{a.out} is stale: run tools/registry_build.py")
            return 1
        print(f"{a.out} matches the pinned sources")
        return 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(text)
    c = json.loads(text)["counts"]
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
