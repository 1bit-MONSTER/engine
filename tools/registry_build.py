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

usage: tools/registry_build.py [--check | --check-gaps | --check-pins] [--out registry/architectures.json]

Every entry is read from a pinned source, never typed in by hand:
- HF architecture -> GGUF architecture: the `@ModelBase.register(...)` classes in llama.cpp's
  converter (convert_hf_to_gguf.py and conversion/*.py) in our HRX fork (third_party/llama.cpp).
- hrx: the GGUF architecture is in the fork's src/llama-arch.cpp (LLM_ARCH_NAMES) and given a
  model class. The same llama-server runs it on the CPU (`1bit serve --device cpu`), so hrx also
  means "runs on the CPU route".
  (The vulkan column, the upstream llama.cpp release the engine built for Vulkan, left with that
  build in RFC #213 stage 3; the zinc column left with ZINC in the core strip, 2026-10-04.)
- gguf_architectures.hrx: every GGUF architecture the fork's runtime builds a model for, whether
  or not a converter class names it. CMakeLists.txt compiles it into `1bit serve`, which refuses
  any other architecture on hrx and cpu (app/serve.cpp, refuse_missing_arch).
- npu: the fast lane's model types (NPU_MODEL_TYPES below, matching npu/). The NPU runs Q4NX
  model directories, so it matches on HF model_type, not on a GGUF architecture.

"Mapped" means the backend's own code accepts the architecture. Whether a model of that
architecture was run and checked here is recorded separately, in registry/checked.json.

--check exits 1 when the file on disk differs from what the pinned sources give, and 77 (CTest's skip) when the
fork is not checked out: it is the only form that reads the CONTENT, so it is ctest `registry_content`, and a
skip there must not read as a pass. --check-pins only compares the pins the file records with the gitlinks, so
it needs no checkouts (CI and ctest `registry_pins` run it): a pin bump must regenerate the registry. "The
gitlinks" means the index where it has one, so a bump that is staged but not yet committed is checked too,
which is when a lane actually runs it.
"""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRX = "third_party/llama.cpp"

# The NPU serves these HF architectures (docs/npu.md):
#  - the fast lane serves Qwen3 and Qwen3.5 dense Q4NX directories (model_type from
#    config.json): Qwen3.8-27B-NPU2 is a `qwen3_5` directory and is measured on the NPU route
#    (bench/npu-leaderboard/27b-*.json), so the dense family is served, not only the MoE;
#  - the GGUF-on-NPU route repacks a .gguf and runs the model-generic forward, for six
#    arches each verified by a captured `--device npu` run (goal muhz0s39-8c9l9o). Text
#    decoders only; vision-language variants of these arches are not claimed.
# HF architecture -> model_type the forward's config.json names.
NPU_ARCHS = {
    "Qwen3ForCausalLM": "qwen3",            # fast lane (Q4NX directory)
    # fast lane: Qwen3.8-27B-NPU2 (qwen3_5) — the text container of a VLM family; the vision
    # tower is a separate `vision_weight.q4nx` that this route does not load, so the two text
    # LM classes are claimed and `Qwen3_5Model` (the bare vision wrapper) is not.
    "Qwen3_5ForCausalLM": "qwen3_5",
    "Qwen3_5ForConditionalGeneration": "qwen3_5",
    "Qwen2ForCausalLM": "qwen2",            # GGUF-on-NPU: Qwen2.5-7B-Instruct
    "Qwen3MoeForCausalLM": "qwen3moe",      # GGUF-on-NPU: Qwen3-Coder-30B-A3B
    "Qwen3_5MoeForCausalLM": "qwen3_5_moe",  # GGUF-on-NPU: Qwen3.6-35B-A3B (gguf qwen35moe)
    "DeepseekV2ForCausalLM": "deepseek2",   # GGUF-on-NPU: GLM-4.7-Flash
    "MiniCPMForCausalLM": "minicpm",        # GGUF-on-NPU: MiniCPM4-8B
    "LlamaForCausalLM": "llama",            # GGUF-on-NPU: MiniCPM5-1B
}


def pin(path):
    # The index first, HEAD only as a fallback: a pin bump is *staged* before it is committed, and every check here
    # runs before that commit. Reading only HEAD reported the pin being replaced, so `--check` said the registry
    # matched while the bump sat in the index, and the stale `sources` pin surfaced only in ctest `registry_pins`
    # after the commit -- ten minutes of CI to learn what the same command could have said locally (engine#1738).
    staged = subprocess.run(["git", "-C", ROOT, "ls-files", "-s", "--", path], capture_output=True, text=True)
    # `ls-files -s` prints "<mode> <sha> <stage>\t<path>", one line per STAGE. Stages 1/2/3 mean an unmerged index --
    # a submodule conflict -- and the first line is then the merge BASE, so matching on the mode alone would return a
    # third pin that nobody chose: silent, and the very staleness this function exists to stop (engine#1745 review).
    rows = [l.split() for l in staged.stdout.splitlines() if l.split()]
    if any(len(r) >= 3 and r[2] != "0" for r in rows):
        raise SystemExit("registry_build: the index has an unmerged entry for %s (stages %s) -- resolving that conflict "
                         "is what decides the pin, so this cannot check it" %
                         (path, ", ".join(sorted({r[2] for r in rows if len(r) >= 3}))))
    if rows and len(rows[0]) >= 3 and rows[0][0] == "160000":  # the gitlink as staged
        return rows[0][1]
    out = subprocess.run(["git", "-C", ROOT, "ls-tree", "HEAD", "--", path], capture_output=True, text=True)
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
    """GGUF architecture names the runtime can build a model for: named in LLM_ARCH_NAMES and
    given a class in llama_model_mapping (a name alone, as upstream has for gptj, loads nothing)."""
    src = open(os.path.join(tree, "src/llama-arch.cpp")).read()
    body = src[src.index("LLM_ARCH_NAMES"):]
    body = body[:body.index("};")]
    names = dict(re.findall(r"\{\s*(LLM_ARCH_[A-Z0-9_]+)\s*,\s*\"([^\"]+)\"\s*\}", body))
    model = open(os.path.join(tree, "src/llama-model.cpp")).read()
    start = model.index("llama_model_mapping(")
    mapping = model[start:model.index("\n}\n", start)]
    built = set(re.findall(r"case\s+(LLM_ARCH_[A-Z0-9_]+)\s*:", mapping))
    return {n for e, n in names.items() if e in built} - {"clip"}


TARBALL_CACHE = os.environ.get("ONEBIT_REGISTRY_TARBALL_CACHE") or os.path.join(
    os.path.expanduser("~"), ".cache", "1bit-registry-tarballs")


def fork_slug(root=None):
    """owner/repo for the fork, from `.gitmodules` — the slug the pin names, not a guessed default."""
    mods = os.path.join(root or ROOT, ".gitmodules")
    if not os.path.isfile(mods):
        return None
    path = url = None
    for line in open(mods).read().splitlines():
        line = line.strip()
        if line.startswith("path ="):
            path = line.split("=", 1)[1].strip()
        elif line.startswith("url ="):
            url = line.split("=", 1)[1].strip()
            m = re.search(r"github\.com[:/]+([^/]+/[^/]+?)(?:\.git)?$", url)
            if path == HRX and m:
                return m.group(1)
            path = url = None
    return None


def tarball_tree(sha, say, root=None, cache=None):
    """Extracted source tree for `sha`, fetching once into a cache; None (after printing why) when it cannot tell.

    engine#3212: a failed fetch is cannot-tell, never clean — the same rule as the row's 77. A tarball the fork does
    not carry (a 404 for a sha that was force-pushed away) must not read as "the registry matches".
    """
    import tarfile, urllib.error, urllib.request
    if not sha:
        say("the index pins no commit for %s, so there is nothing to fetch" % HRX)
        return None
    cache = cache or TARBALL_CACHE
    def prune(cache, sha, say):
        """Keep this run's pin and drop the previous one's, unless ONEBIT_REGISTRY_TARBALL_KEEP=all.

        engine#3212: an extracted tree is ~200 MB beside its 39 MB tarball, so an unbounded cache grows by that on every
        pin. Called on BOTH paths -- a cache HIT can also be the run that should prune the sibling, which my first
        version missed because the early return skipped it.
        """
        if os.environ.get("ONEBIT_REGISTRY_TARBALL_KEEP", "1") == "all":
            return
        for name in sorted(os.listdir(cache)):
            stale = os.path.join(cache, name)
            if name.startswith(sha):
                continue
            if os.path.isdir(stale):
                shutil.rmtree(stale, ignore_errors=True)
            elif name.endswith((".tar.gz", ".part")):
                os.unlink(stale)
            else:
                continue
            say(f"pruned the previous pin's {name} (ONEBIT_REGISTRY_TARBALL_KEEP=all keeps them)")

    dest = os.path.join(cache, sha)
    if os.path.isdir(os.path.join(dest, "src")):
        say(f"tarball already extracted at {dest}")
        prune(cache, sha, say)
        return dest
    slug = fork_slug(root)
    if not slug:
        say(f"no github owner/repo for {HRX} in {os.path.join(root or ROOT, '.gitmodules')}, so the tarball URL is unknown")
        return None
    url = f"https://codeload.github.com/{slug}/tar.gz/{sha}"
    os.makedirs(cache, exist_ok=True)
    tar_path = os.path.join(cache, f"{sha}.tar.gz")
    if not os.path.isfile(tar_path):
        say(f"fetching {url}")
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tar_path + ".part", "wb") as f:
                f.write(r.read())
            os.replace(tar_path + ".part", tar_path)
        except (urllib.error.URLError, OSError) as e:
            if os.path.exists(tar_path + ".part"):
                os.unlink(tar_path + ".part")
            say(f"could not fetch {url}: {e}")
            return None
    say(f"extracting {tar_path} ({os.path.getsize(tar_path) // 1048576} MB) into {dest}")
    try:
        with tarfile.open(tar_path) as tf:
            top = tf.getnames()[0].split("/")[0]
            tmp = dest + ".part"
            os.makedirs(tmp, exist_ok=True)
            tf.extractall(tmp)
            os.replace(os.path.join(tmp, top), dest) if not os.path.isdir(dest) else None
            shutil.rmtree(tmp, ignore_errors=True)
    except (tarfile.TarError, OSError) as e:
        say(f"could not extract {tar_path}: {e}")
        return None
    if not os.path.isdir(os.path.join(dest, "src")):
        say(f"extracted {sha} but {dest}/src is missing, so this is not the fork's tree")
        return None
    prune(cache, sha, say)
    return dest


def build(tree=None):
    """Derive the registry from the fork's sources at `tree`, or from the checked-out submodule.

    `tree` exists for `--from-tarball` (engine#3212): the same derivation, from a source tarball at the pinned sha,
    because most of the fleet's checkouts have no submodule and the content question is asked on every pin bump.
    """
    if tree is None:
        if not os.path.isdir(os.path.join(ROOT, HRX, "src")):
            # 77, not 1: the content comes from a submodule, so a checkout without it cannot tell, and "cannot tell"
            # must never read as clean. CTest reports 77 as a skip with this message (engine#2698).
            print(f"registry_build: {HRX} is not checked out (git submodule update --init {HRX}): the content "
                  f"checks skip here, they do not pass")
            raise SystemExit(77)
        hrx = os.path.join(ROOT, HRX)
    else:
        hrx = tree
    mapping = hf_to_gguf(hrx)
    run_hrx = runtime_archs(hrx)

    archs = {}
    for hf in sorted(set(mapping) | set(NPU_ARCHS)):
        g = mapping.get(hf, "")
        backends = []
        if g in run_hrx:
            backends.append("hrx")
        if hf in NPU_ARCHS:
            backends.append("npu")
        entry = {"gguf": g, "backends": backends}
        if hf in NPU_ARCHS:
            entry["npu_model_type"] = NPU_ARCHS[hf]
        archs[hf] = entry

    return {
        "about": "HF architecture -> GGUF architecture and the backends whose code accepts it. "
                 "Generated by tools/registry_build.py from the pinned sources; do not edit.",
        "sources": {"llama.cpp (hrx)": pin(HRX)},
        "counts": {b: sum(b in a["backends"] for a in archs.values()) for b in ("hrx", "npu")}
                  | {"architectures": len(archs), "mapped": sum(bool(a["backends"]) for a in archs.values())},
        "architectures": archs,
        "gguf_architectures": {"hrx": sorted(run_hrx)},
    }


def _mapped_ok(e):
    """A reviewed class is exempt only when 'mapped_ok' is a non-empty, non-whitespace string:
    a missing, empty or vacuous value (e.g. true or "   ") is still a violation."""
    r = e.get("mapped_ok")
    return isinstance(r, str) and bool(r.strip())


def gap_violations(registry, sig=None):
    """Classes reviewed as not aliases (registry/significant.json, docs/arch-gaps.md) that the
    registry maps without a recorded reason ('mapped_ok'): each needs a person's look."""
    if sig is None:
        sig = json.load(open(os.path.join(ROOT, "registry/significant.json")))["classes"]
    mapped = registry["architectures"]
    return [(cls, mapped[cls]) for cls, e in sig.items() if cls in mapped and not _mapped_ok(e)]


def report_gaps(registry):
    bad = gap_violations(registry)
    for cls, m in bad:
        print(f"{cls} is mapped (gguf {m['gguf']}, {', '.join(m['backends'])}) but docs/arch-gaps.md reviewed it as "
              f"not an alias: check that the pinned backend really implements it, then record why in "
              f"registry/significant.json ('mapped_ok')")
    return bad


def self_test():
    """The paths that must answer cannot-tell, none of which needs the network (engine#3212).

    The fetch itself is covered by the known-answer run recorded in the PR (three pins, 350/9/350/350 each); what an
    arm can pin offline is that every way this can fail to produce a tree says so instead of looking clean.
    """
    import tempfile
    fails = 0

    def chk(cond, what):
        nonlocal fails
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            fails += 1

    said = []
    with tempfile.TemporaryDirectory() as d:
        chk(tarball_tree("", said.append, root=d, cache=d) is None, "no pinned sha -> no tree")
        chk(tarball_tree("deadbeef", said.append, root=d, cache=d) is None,
            "no .gitmodules in the tree -> the slug is unknown, not guessed")
        # a cache entry that exists but is not the fork's tree
        os.makedirs(os.path.join(d, "cafebabe", "docs"), exist_ok=True)
        slugs = os.path.join(d, ".gitmodules")
        open(slugs, "w").write('[submodule "third_party/llama.cpp"]\n\tpath = %s\n\turl = https://github.com/x/y.git\n' % HRX)
        chk(tarball_tree("cafebabe", said.append, root=d, cache=d) is None,
            "a cache dir without src/ is not the fork's tree, so it is still cannot-tell")
    chk(all(isinstance(s, str) and s for s in said), "every failure said why")
    print(f"registry_build self-test: {4 - fails} ok, {fails} failed")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "registry/architectures.json"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-gaps", action="store_true",
                    help="only check the committed registry against registry/significant.json (no pins needed)")
    ap.add_argument("--check-pins", action="store_true",
                    help="only check that the registry records the committed pins (no checkouts needed)")
    ap.add_argument("--from-tarball", action="store_true",
                    help="when the fork is not checked out, fetch the pinned sha's source tarball (cached) and derive "
                         "from that tree -- one command for the content question on every pin bump (engine#3212)")
    ap.add_argument("--self-test", action="store_true",
                    help="the --from-tarball cannot-tell paths, offline (engine#3212)")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    # engine#3212: OPT-IN and default-unchanged. `registry_content`'s 77 is deliberate and offline by design; a
    # networked row would make a deterministic gate depend on the network, so nothing below runs without the flag.
    tree = None
    if a.from_tarball and not os.path.isdir(os.path.join(ROOT, HRX, "src")):
        tree = tarball_tree(pin(HRX), lambda m: print(f"registry_build: --from-tarball: {m}"))
        if tree is None:
            print("registry_build: --from-tarball could not produce the pinned tree -- cannot tell, not a pass")
            return 2
    if a.check_pins:
        recorded = json.load(open(a.out))["sources"]
        stale = [(name, recorded.get(name, ""), pin(path)) for name, path in
                 (("llama.cpp (hrx)", HRX),)
                 if recorded.get(name) != pin(path)]
        for name, old, new in stale:
            print(f"{a.out}: {name} recorded at {old[:9] or 'nothing'}, pinned at {new[:9]}")
        if stale:
            print("a pin moved without regenerating the registry: check out the submodules at their "
                  "pins (git submodule update --init) and run tools/registry_build.py")
            return 1
        print(f"{a.out} records the committed pins")
        return 0
    if a.check_gaps:
        bad = report_gaps(json.load(open(a.out)))
        if not bad:
            n = len(json.load(open(os.path.join(ROOT, "registry/significant.json")))["classes"])
            print(f"none of the {n} reviewed significant classes is mapped without a reason")
        return 1 if bad else 0
    text = json.dumps(build(tree), indent=1, sort_keys=False) + "\n"
    report_gaps(json.loads(text))
    if a.check:
        # The registry's content comes from the checked-out sources, so those must BE the pinned ones: with the pin
        # read from the index (above) a bump staged against a submodule left at the old commit would record the new
        # pin beside the old architecture list and pass.
        head = "" if tree else subprocess.run(["git", "-C", os.path.join(ROOT, HRX), "rev-parse", "HEAD"],
                                             capture_output=True, text=True).stdout.strip()
        if head and head != pin(HRX):
            print(f"{HRX} is checked out at {head[:9]}, the pin is {pin(HRX)[:9] or 'unset'}: "
                  "run git submodule update --init before regenerating")
            return 1
        old = open(a.out).read() if os.path.exists(a.out) else ""
        if old != text:
            print(f"{a.out} is stale: run tools/registry_build.py")
            return 1
        print(f"{a.out} matches the pinned sources" + (f" (derived from the {pin(HRX)[:9]} tarball, not a checkout)"
                                                      if tree else ""))
        return 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(text)
    c = json.loads(text)["counts"]
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
