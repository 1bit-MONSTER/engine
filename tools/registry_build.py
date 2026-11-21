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


def submodule_url(path, root=ROOT):
    """The submodule's clone url from .gitmodules, or '' when it names none.

    engine#3212: the content axis asks "did the fork's architecture tables move with this pin?", and the inputs that
    answer it are all already known to this tool -- the sha from `pin(path)` and the repo slug from here. Reading the
    slug from .gitmodules rather than a constant is what keeps this honest when a submodule is re-homed.
    """
    import configparser
    cfg = configparser.ConfigParser()
    try:
        cfg.read(os.path.join(root, ".gitmodules"))
    except (OSError, configparser.Error):
        return ""
    return (cfg.get(f'submodule "{path}"', 'url', fallback='') or "").strip()


def tarball_url(url, sha):
    """codeload tarball for one commit of a github clone url, or '' for anything that is not github https.

    Pure, so the arm can assert it without a network: a wrong slug or a short sha is the failure mode that would
    otherwise surface as a 404 thirty seconds into a fetch.
    """
    m = re.match(r'^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$', (url or '').strip())
    if not m or not re.fullmatch(r'[0-9a-f]{40}', sha or ''):
        return ''
    return f'https://codeload.github.com/{m.group(1)}/{m.group(2)}/tar.gz/{sha}'


def fetch_tree(url, sha, cache=None):
    """(tree dir, note) for a source tarball at `sha`, or (None, why). Cached by sha, extracted once.

    engine#3212: opt-in only -- the ctest row stays offline, because a deterministic gate must not depend on the
    network. A failure returns why, and every caller treats that as cannot-tell rather than as agreement.
    """
    import tarfile
    import tempfile
    import urllib.request
    root = cache or os.path.join(os.path.expanduser("~"), ".cache", "registry_build")
    dest = os.path.join(root, sha)
    marker = os.path.join(dest, ".complete")

    def prune(note):
        """Drop the previous pin's extracted tree, unless ONEBIT_REGISTRY_TARBALL_KEEP=all (engine#3212).

        An extracted tree is ~200 MB and nothing here is shared between pins, so an unbounded cache grows by that on
        every bump -- and a bump is exactly when this runs. Called on BOTH paths: a cache HIT can be the run that
        should prune the sibling, which a version that only pruned after extracting would walk straight past.
        """
        if os.environ.get("ONEBIT_REGISTRY_TARBALL_KEEP", "1") == "all":
            return note
        dropped = []
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if name == sha or not os.path.isdir(path):
                continue
            shutil.rmtree(path, ignore_errors=True)
            dropped.append(name[:9])
        if dropped:
            note += (f"; pruned the previous pin(s) {', '.join(dropped)} (ONEBIT_REGISTRY_TARBALL_KEEP=all keeps them)")
        return note

    if os.path.isfile(marker):
        trees = [d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d))]
        if len(trees) == 1:
            return os.path.join(dest, trees[0]), prune(f"cached at {dest}")
    os.makedirs(dest, exist_ok=True)
    tgz = os.path.join(dest, "src.tgz")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tgz, "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        with tarfile.open(tgz) as tf:
            tf.extractall(dest, filter="data")  # noqa: S202 -- a pinned public sha, into our own cache
    except Exception as e:  # noqa: BLE001 -- any failure here is cannot-tell, never a pass
        return None, f"could not fetch {url}: {type(e).__name__}: {e}"
    finally:
        if os.path.exists(tgz):
            os.remove(tgz)
    trees = [d for d in os.listdir(dest) if os.path.isdir(os.path.join(dest, d))]
    if len(trees) != 1:
        return None, f"{url} did not extract to exactly one directory ({len(trees)} found)"
    open(marker, "w").close()
    return os.path.join(dest, trees[0]), prune(f"fetched {url} into {dest}")


def build(tree=None):
    if tree is None and not os.path.isdir(os.path.join(ROOT, HRX, "src")):
        # 77, not 1: the content comes from a submodule, so a checkout without it cannot tell, and "cannot tell"
        # must never read as clean. CTest reports 77 as a skip with this message (engine#2698).
        print(f"registry_build: {HRX} is not checked out (git submodule update --init {HRX}): the content "
              f"checks skip here, they do not pass")
        raise SystemExit(77)
    hrx = tree or os.path.join(ROOT, HRX)
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
    """The tarball path's pure parts (engine#3212). No network: `fetch_tree` is exercised only on failure."""
    fail = 0

    def ok(cond, what):
        nonlocal fail
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            fail += 1

    url = submodule_url(HRX)
    ok(url.startswith('https://github.com/') and url.endswith('.git'),
       f"the pinned submodule's clone url is read from .gitmodules ({url})")
    ok(submodule_url('third_party/not-a-submodule') == '', "a path that is not a submodule yields no url")

    sha = 'a' * 40
    ok(tarball_url(url, sha) == f'https://codeload.github.com/1bit-MONSTER/llama.cpp/tar.gz/{sha}',
       "a github clone url and a full sha give that sha's codeload tarball")
    ok(tarball_url('git@github.com:1bit-MONSTER/llama.cpp.git', sha) == '',
       "an ssh remote is not a tarball url, rather than a guess")
    ok(tarball_url(url, sha[:9]) == '', "a short sha is refused here, not thirty seconds into a 404")
    ok(tarball_url('https://gitlab.com/a/b', sha) == '', "a non-github host is refused")

    # The rule that matters (engine#3212): a fetch that fails is CANNOT TELL. Nothing is extracted and no marker is
    # written, so the next run tries again instead of trusting a partial tree.
    import tempfile
    cache = tempfile.mkdtemp()
    tree, why = fetch_tree('https://127.0.0.1:9/none.tgz', 'b' * 40, cache=cache)
    ok(tree is None and 'could not fetch' in why, "an unreachable tarball is cannot-tell, not an empty tree")
    ok(not os.path.exists(os.path.join(cache, 'b' * 40, '.complete')), "...and it leaves no completion marker behind")

    print(f"registry_build --self-test: {'ok' if not fail else f'{fail} failure(s)'}")
    return 0 if not fail else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "registry/architectures.json"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-gaps", action="store_true",
                    help="only check the committed registry against registry/significant.json (no pins needed)")
    ap.add_argument("--check-pins", action="store_true",
                    help="only check that the registry records the committed pins (no checkouts needed)")
    ap.add_argument("--from-tarball", action="store_true",
                    help="derive from the PINNED sha's source tarball when the submodule is not checked out "
                         "(opt-in: the ctest row stays offline, engine#3212)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
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
    tree = None
    if a.from_tarball and not os.path.isdir(os.path.join(ROOT, HRX, "src")):
        # engine#3212: the content question is asked on every pin bump, and answering it by hand cost three tarball
        # fetches in one session. The pin still comes from the INDEX (`pin(HRX)`), not from the tree we fetch, so this
        # cannot launder a bump that is staged but not committed.
        sha = pin(HRX)
        url = tarball_url(submodule_url(HRX), sha)
        if not url:
            print(f"registry_build: cannot tell -- no usable github url in .gitmodules for {HRX} "
                  f"(got {submodule_url(HRX)!r}) or the pin is not a full sha ({sha!r})")
            return 2
        tree, note = fetch_tree(url, sha)
        if not tree:
            # cannot-tell, not clean: the same rule the ctest row's 77 follows.
            print(f"registry_build: cannot tell -- {note}")
            return 2
        print(f"registry_build: deriving from {sha[:12]} ({note})")
    text = json.dumps(build(tree=tree), indent=1, sort_keys=False) + "\n"
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
        print(f"{a.out} matches the pinned sources")
        return 0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write(text)
    c = json.loads(text)["counts"]
    print(f"wrote {a.out}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
