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
"""Pin the census watcher's alert contract (tools/census_watch.py, ctest census_watch).

The daily watch exits 0 when every newest model's architecture is mapped and 1 when one
is not, so a regression that made classify() stop flagging an unmapped class would turn
the alert permanently green — the failure mode this whole tool exists to prevent. This
test drives classify() and report() against synthetic batches and a synthetic registry
(the same approach tests/registry_gaps_test.py takes for the gap guard), then checks the
committed registry still parses and still maps a known class.

No network: the listing walk (sweep_newest) is exercised by the daily
timer, not here.
"""
import io
import json
import os
import sys
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import census_watch as cw  # noqa: E402

ARCHS = {
    "Qwen3ForCausalLM": {"gguf": "qwen3", "backends": ["hrx"]},
    "EmptyForCausalLM": {"gguf": "empty", "backends": []},
}
SIG = {"Qwen3_5MVLAAbsorbedForCausalLM": {"model_type": "qwen3_5_text", "family": "Qwen3.5 MVLA"}}

failures = []


def check(name, cond):
    if cond:
        print(f"ok   {name}")
    else:
        print(f"FAIL {name}")
        failures.append(name)


def model(mid, arch=None, tags=(), library=None, remote=False):
    m = {"id": mid, "tags": list(tags)}
    if library:
        m["library_name"] = library
    if arch:
        m["config"] = {"architectures": [arch]}
        if remote:
            m["config"]["auto_map"] = {"AutoModelForCausalLM": "modeling_x.XForCausalLM"}
    return m


def rc_of(models, archs=ARCHS, sig=SIG, seen=()):
    """The exit code the run would return, with report()'s output captured."""
    res = cw.classify(models, archs, sig, seen)
    with contextlib.redirect_stdout(io.StringIO()):
        return cw.report(models, res, seen), res


def text_of(models, archs=ARCHS, sig=SIG, seen=()):
    """The exit code and the report's text, so a misleading summary can be pinned."""
    res = cw.classify(models, archs, sig, seen)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cw.report(models, res, seen)
    return rc, buf.getvalue()


# 1. A mapped class is covered and the run is green.
rc, res = rc_of([model("a/one", "Qwen3ForCausalLM")])
check("mapped class -> covered, exit 0", rc == 0 and res[cw.COVERED] == ["a/one"])

# 2. An unmapped class is the alert.
rc, res = rc_of([model("b/two", "NimbusMindForCausalLM")])
check("unmapped class -> uncovered, exit 1", rc == 1 and res[cw.UNCOVERED] == ["b/two"])
check("unmapped class is grouped under its class", list(res["classes"]) == ["NimbusMindForCausalLM"])

# 3. A registry entry with no backend is not covered (it maps nothing to run on).
rc, res = rc_of([model("c/three", "EmptyForCausalLM")])
check("empty backends -> uncovered", rc == 1 and res[cw.UNCOVERED] == ["c/three"])

# 4. A reviewed not-an-alias class is still the alert, and is labeled as such.
rc, res = rc_of([model("d/four", "Qwen3_5MVLAAbsorbedForCausalLM")])
check("significant class -> uncovered, exit 1", rc == 1)
check("significant class is labeled", res["significant"] == ["Qwen3_5MVLAAbsorbedForCausalLM"])

# 4b. A custom-code class (auto_map) from fewer than REMOTE_CODE_SPREAD uploaders is reported,
#     not an alert: research one-offs arrive daily and no backend could map them.
rc, text = text_of([model("lab/hyper-a", "HyperLlamaForCausalLM", remote=True),
                    model("lab/hyper-b", "HyperLlamaForCausalLM", remote=True)])
check("remote-code class, one uploader -> exit 0", rc == 0)
check("remote-code class is still reported", "REMOTE-CODE HyperLlamaForCausalLM" in text)
rc, res = rc_of([model(f"u{i}/sol", "Sol2ForCausalLM", remote=True) for i in range(cw.REMOTE_CODE_SPREAD)])
check("remote-code class from REMOTE_CODE_SPREAD uploaders -> exit 1", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([model("v/one", "MixForCausalLM", remote=True), model("v/two", "MixForCausalLM")])
check("a class with one native (no auto_map) model alerts", rc == 1)
rc, res = rc_of([model("w/one", "Qwen3_5MVLAAbsorbedForCausalLM", remote=True)])
check("a reviewed class alerts even with custom code", rc == 1)
rc, res = rc_of([model("x/one", "OneOffForCausalLM", remote=True), model("x/two", "NimbusMindForCausalLM")])
check("a remote-code one-off does not hide a native arrival", rc == 1 and res["remote_code"] == ["OneOffForCausalLM"])

# 4c. A model that is not a Transformers model at all (engine#296: a raw `pytorch` nanoGPT
#     checkpoint with `model_type: nanobeard-gpt` and `architectures: ["GPT"]`) is the same
#     kind of one-off: no backend can map a class only its own library code implements.
rc, text = text_of([model("y/nano", "GPT", library="pytorch")])
check("non-Transformers library -> exit 0, reported", rc == 0 and "REMOTE-CODE GPT" in text)
rc, res = rc_of([model("y/nano", "GPT", library="pytorch")])
check("non-Transformers one-off is grouped as remote_code", res["remote_code"] == ["GPT"])
rc, res = rc_of([model(f"y{i}/nano", "GPT", library="pytorch") for i in range(cw.REMOTE_CODE_SPREAD)])
check("non-Transformers class from REMOTE_CODE_SPREAD uploaders -> exit 1", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([model("y/nano", "Qwen3ForCausalLM", library="pytorch")])
check("a mapped class stays covered whatever its library", rc == 0 and res[cw.COVERED] == ["y/nano"])
rc, res = rc_of([model("z/nano", "NimbusMindForCausalLM", library="pytorch"),
                 model("z/other", "ArgonneModel")])
check("a non-Transformers one-off does not hide a native arrival",
      rc == 1 and res["remote_code"] == ["NimbusMindForCausalLM"])
rc, res = rc_of([model("z/nano", "NimbusMindForCausalLM")])
check("an unknown library is not treated as custom", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([model("z/nano", "NimbusMindForCausalLM", library="pytorch", tags=["transformers"])])
check("a pytorch library with the transformers tag still alerts", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([model("z/nano", "Qwen3_5MVLAAbsorbedForCausalLM", library="mlx")])
check("a reviewed class alerts even when not Transformers", rc == 1)

# 5. A derivative with no config is expected, not an alert.
rc, res = rc_of([model("e/five", tags=["gguf", "text-generation"])])
check("derivative -> no alert, not unverifiable",
      rc == 0 and res[cw.DERIVATIVE] == ["e/five"] and not res[cw.UNVERIFIABLE])
rc, res = rc_of([model("e/six", library="peft")])
check("peft library -> derivative", rc == 0 and res[cw.DERIVATIVE] == ["e/six"])

# 6. A gated repo (no config, no derivative tag) is unverifiable: reported, not an alert.
rc, res = rc_of([model("f/seven", tags=["text-generation"])])
check("no config -> unverifiable, exit 0",
      rc == 0 and res[cw.UNVERIFIABLE] == ["f/seven"]
      and res["reasons"]["f/seven"] == "no config / no architecture")

# 7. An unverifiable model is reported once, then remembered.
rc, res = rc_of([model("f/seven", tags=["text-generation"])], seen={"f/seven"})
check("unverifiable remembered -> quiet", rc == 0 and res[cw.UNVERIFIABLE] == [])

# 8. A mixed batch: the alert fires, and the summary counts each group.
rc, res = rc_of([
    model("g/eight", "Qwen3ForCausalLM"),
    model("g/nine", "ArgonneModel"),
    model("g/ten", tags=["gguf"]),
    model("g/eleven", tags=["text-generation"]),
])
check("mixed batch -> exit 1, one uncovered class",
      rc == 1 and list(res["classes"]) == ["ArgonneModel"]
      and len(res[cw.COVERED]) == 1 and len(res[cw.DERIVATIVE]) == 1 and len(res[cw.UNVERIFIABLE]) == 1)

# 9. A run with nothing to check says so, instead of reporting a green "no uncovered
#    class" over zero models (the first production run after a stale state did exactly
#    that: the horizon break stopped the walk and the summary read as all-clear).
rc, text = text_of([])
check("no new models -> exit 0 and says nothing new", rc == 0 and "nothing new" in text)
rc, text = text_of([model("i/sixteen", "Qwen3ForCausalLM")])
check("a run that checked models does not claim nothing new",
      rc == 0 and "nothing new" not in text)

# 10. remember() produces a state that serializes, every group included. (A tuple key here
#    broke json.dump in the first live run.)
import tempfile  # noqa: E402
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "nested", "state.json")
    cw.save_state(path, {"last_run": "t", "seen": {"x": "covered"}})
    check("state round-trips", cw.load_state(path)["seen"] == {"x": "covered"})
    cw.save_state(path, {"last_run": "t", "seen": {f"m{i}": "covered" for i in range(cw.MAX_SEEN + 10)}})
    check("state is capped at MAX_SEEN", len(cw.load_state(path)["seen"]) == cw.MAX_SEEN)
    check("missing state file loads empty", cw.load_state(os.path.join(td, "absent.json"))["seen"] == {})

    _, mixed = rc_of([
        model("h/twelve", "Qwen3ForCausalLM"),
        model("h/thirteen", "NimbusMindForCausalLM"),
        model("h/fourteen", tags=["gguf"]),
        model("h/fifteen", tags=["text-generation"]),
    ])
    state = cw.remember({"last_run": "t", "seen": {}}, mixed)
    check("remember() records every group",
          state["seen"] == {"h/twelve": cw.COVERED, "h/thirteen": cw.UNCOVERED,
                            "h/fourteen": cw.DERIVATIVE, "h/fifteen": cw.UNVERIFIABLE})
    cw.save_state(path, state)
    check("remembered state serializes and reloads", cw.load_state(path)["seen"] == state["seen"])
    rc, res = rc_of([model("h/fifteen", tags=["text-generation"])], seen=set(cw.load_state(path)["seen"]))
    check("a reloaded state quiets the next run's unverifiable", rc == 0 and res[cw.UNVERIFIABLE] == [])

# 11. The listing walk follows the Link cursor, not `p=`. HF ignores `p=` on a sorted
#     listing (every page returns the first one), which capped the first live run at 101
#     models however large --limit was.
def pages(*batches):
    """A sweep_newest fetch serving synthetic pages; records the URLs requested."""
    urls = []

    def fetch(url):
        urls.append(url)
        i = len(urls) - 1
        if i >= len(batches):
            return [], ""
        batch = [model(mid) for mid in batches[i]]
        if i + 1 < len(batches):
            return batch, f'<https://huggingface.co/api/models?cursor=PAGE{i + 1}>; rel="next"'
        return batch, ""

    return fetch, urls


fetch, urls = pages([f"a/{i}" for i in range(100)], [f"b/{i}" for i in range(100)])
out = cw.sweep_newest(120, seen=set(), fetch=fetch)
check("sweep reaches the limit across two pages",
      len(out) == 120 and out[0]["id"] == "a/0" and out[119]["id"] == "b/19")
check("sweep uses the cursor, never p=",
      len(urls) == 2 and "p=" not in urls[0] and "cursor=PAGE1" in urls[1])

fetch, urls = pages([f"a/{i}" for i in range(100)])
out = cw.sweep_newest(5, seen={f"a/{i}" for i in range(95)}, fetch=fetch)
check("sweep skips already-classified models",
      [m["id"] for m in out] == [f"a/{i}" for i in range(95, 100)])

fetch, urls = pages([f"a/{i}" for i in range(3)])
out = cw.sweep_newest(10, seen={"a/0", "a/1", "a/2"}, fetch=fetch)
check("sweep stops at the horizon after one page", out == [] and len(urls) == 1)

fetch, urls = pages([])
check("sweep tolerates an empty first page", cw.sweep_newest(10, seen=set(), fetch=fetch) == [])

# 12. The committed registry parses and still maps a class the engine runs.
archs, sig = cw.load_registry()
check("committed registry has entries", len(archs) > 100)
check("committed registry maps Qwen3ForCausalLM",
      bool(archs.get("Qwen3ForCausalLM", {}).get("backends")))
check("committed significant list parses", isinstance(sig, dict) and len(sig) > 0)
check("no committed registry entry has a class with no backend field",
      all("backends" in e and "gguf" in e for e in archs.values()))

# 13. Newly reviewed non-alias classes stay out of the runtime registry and are documented.
reviewed_gaps = {
    "Needle3ForCausalLM": ("needle3", "Sakura Needle-3 (ONNX export)"),
    "SepiaCharMLP": ("sepia-char-mlp", "SEPIA character-level MLP"),
    "Qwen3_5MLAForConditionalGeneration": (
        "qwen3_5_mla", "Qwen3.5 MLA attention variant"),
}
for cls, (model_type, family) in reviewed_gaps.items():
    entry = sig.get(cls, {})
    check(f"{cls} is recorded as a significant gap",
          entry.get("model_type") == model_type and entry.get("family") == family
          and cls not in archs)
    rc, res = rc_of([model(f"reviewed/{model_type}", cls)], archs, sig)
    check(f"{cls} remains reported as reviewed and uncovered",
          rc == 1 and res["significant"] == [cls] and res[cw.UNCOVERED] == [f"reviewed/{model_type}"])
with open(os.path.join(ROOT, "docs", "arch-gaps.md")) as f:
    gap_docs = f.read()
check("reviewed non-alias classes are documented",
      all(cls in gap_docs for cls in reviewed_gaps)
      and "not an alias" in gap_docs)

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
