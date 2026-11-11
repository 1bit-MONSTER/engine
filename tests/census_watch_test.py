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
timer, not here; the FastFlowLM org and release sweeps (engine#588) are driven from the
recorded payloads in tests/fixtures/fflm/, captured from the live listing on 2026-10-26
with only the fields this watcher reads kept.
"""
import io
import json
import os
import sys
import shutil
import tempfile
import urllib.error
import urllib.request
import contextlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import census_watch as cw  # noqa: E402

ARCHS = {
    "Qwen3ForCausalLM": {"gguf": "qwen3", "backends": ["hrx"]},
    "EmptyForCausalLM": {"gguf": "empty", "backends": []},
}
SIG = {"Qwen3_5MVLAAbsorbedForCausalLM": {"model_type": "qwen3_5_text", "family": "Qwen3.5 MVLA"}}
FLM_LINE_PREFIXES = ("  + FLM FastFlowLM/", "  !! NEW FLM ", "  !! NPU FLM ", "  . FLM ", "  . NPU FLM ")
# Sections 17-22 pin the alert rule itself, on a fixture org whose MiniCPM-V pack is answered in the
# committed registry/flm_answers.json; they pass no answers so the rule is tested on an unanswered gap.
# The answers file has its own section (27).
NO_ANSWERS = {"architectures": {}, "repos": {}}

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

# 4d. engine#1922: a repo with no library at all that ships its own modeling_*.py. The record is
#     Climb-Ai/Climb-1-Toki's listing row of 2026-11-10 (the fields this watcher reads): class
#     MicroLlamaForCausalLM, `model_type: llama`, no auto_map, and modeling_micro_llama.py where
#     that class is a bare torch.nn.Module. It alerted as a new class; it is the same one-off.
def own_code(mid, arch, files=("config.json", "model.safetensors", "modeling_micro_llama.py"), **kw):
    m = model(mid, arch, **kw)
    m["siblings"] = [{"rfilename": f} for f in files]
    return m


TOKI_TAGS = ["pytorch", "safetensors", "llama", "toki-pona", "micro-llama", "text-generation",
             "conversational", "license:apache-2.0", "region:us"]
rc, text = text_of([own_code("Climb-Ai/Climb-1-Toki", "MicroLlamaForCausalLM", tags=TOKI_TAGS)])
check("own modeling_*.py, no library -> exit 0, reported",
      rc == 0 and "REMOTE-CODE MicroLlamaForCausalLM" in text and "UNCOVERED" not in text)
rc, res = rc_of([own_code(f"u{i}/toki", "MicroLlamaForCausalLM") for i in range(cw.REMOTE_CODE_SPREAD)])
check("own-code class from REMOTE_CODE_SPREAD uploaders -> exit 1", rc == 1 and res["remote_code"] == [])
# the rule must not quiet a class Transformers has: the Hub marks such a repo as a Transformers model
rc, res = rc_of([own_code("o/new", "NimbusMindForCausalLM", tags=["transformers"])])
check("own modeling_*.py with the transformers tag still alerts", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([own_code("o/new", "NimbusMindForCausalLM", library="transformers")])
check("own modeling_*.py with library transformers still alerts", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([own_code("o/new", "NimbusMindForCausalLM", library="sentence-transformers")])
check("own modeling_*.py under an unknown library still alerts", rc == 1 and res["remote_code"] == [])
# only a top-level modeling_<name>.py counts: not a training script, not a file in a folder
rc, res = rc_of([own_code("o/new", "NimbusMindForCausalLM", files=("config.json", "train.py", "model.py"))])
check("other python files are not modeling code", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([own_code("o/new", "NimbusMindForCausalLM", files=("config.json", "src/modeling_x.py"))])
check("a modeling file in a subfolder is not the repo's model class", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([own_code("o/a", "MixForCausalLM"), model("o/b", "MixForCausalLM")])
check("a class with one model that ships no code alerts", rc == 1 and res["remote_code"] == [])
rc, res = rc_of([own_code("o/a", "Qwen3_5MVLAAbsorbedForCausalLM")])
check("a reviewed class alerts even with its own modeling file", rc == 1)
rc, res = rc_of([own_code("o/a", "Qwen3ForCausalLM")])
check("a mapped class stays covered whatever files it ships", rc == 0 and res[cw.COVERED] == ["o/a"])

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
# engine#1922: the file names arrive only with full=true; without them 4d's rule sees nothing
check("sweep asks the listing for file names (full=true) and the config",
      "full=true" in urls[0] and "config=true" in urls[0])

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
# engine#2014 review: the list above is hand-written and names three classes, so it keeps covering those three
# while the file it is about grows (37 entries today) -- a fourth, fifth and thirty-seventh class can be added
# with no row in docs/arch-gaps.md and this file stays green. significant.json's own `about` says its classes
# are documented there, so derive the list from the file: a missing row then fails here instead of resting on
# three names that nobody will remember to extend. The floor stops the loop passing vacuously if the file were
# ever loaded empty -- a check that iterates nothing must not report success.
undocumented = sorted(cls for cls in sig if cls not in gap_docs)
check(f"every one of the {len(sig)} reviewed classes is cited in docs/arch-gaps.md"
      + (f" -- missing: {', '.join(undocumented)}" if undocumented else ""),
      len(sig) >= 30 and not undocumented)

# 14. A reviewed class marked `"watch": "report"` is listed but does not alert (engine#381).
check("GrugMoeForCausalLM is reviewed as report-only",
      sig.get("GrugMoeForCausalLM", {}).get("watch") == "report"
      and sig["GrugMoeForCausalLM"].get("model_type") == "grug_moe"
      and "GrugMoeForCausalLM" not in archs and "GrugMoeForCausalLM" in gap_docs)
grug = model("open-athena/Grug-67B-A2B-Merge", "GrugMoeForCausalLM")
rc, res = rc_of([grug], archs, sig)
check("report-only reviewed class -> exit 0",
      rc == 0 and res["reviewed_report"] == ["GrugMoeForCausalLM"] and res["significant"] == []
      and res[cw.UNCOVERED] == ["open-athena/Grug-67B-A2B-Merge"])
_, out = text_of([grug], archs, sig)
check("report-only reviewed class is printed as REVIEWED, not UNCOVERED",
      ". REVIEWED GrugMoeForCausalLM" in out and "UNCOVERED" not in out)
rc, _ = rc_of([grug, model("x/new", "BrandNewForCausalLM")], archs, sig)
check("a new class next to a report-only one still alerts", rc == 1)
sig_alert = dict(sig, GrugMoeForCausalLM={k: v for k, v in sig["GrugMoeForCausalLM"].items() if k != "watch"})
rc, res = rc_of([grug], archs, sig_alert)
check("without the watch field the reviewed class alerts", rc == 1 and res["significant"] == ["GrugMoeForCausalLM"])

# 15. The classes the 2026-10-14..16 streak flagged (issue #427 and the two un-filed runs
#     before it) are reviewed, documented and non-alerting.
for cls, mid in (
    ("DFlash2DraftModel", "nchapman/Qwen3.5-4B-DFlash2"),
    ("DiacritizerModel", "ahmadsy/tashkeel-nano-1p1m"),
    ("MBartForConditionalGeneration", "nhatho1999/orthomni-codeco"),
    ("BlindForCausalLM", "Vi-LAB/KenT_0.5B_Base"),
    ("NanochronoForCausalLM", "tensin320/round14_5_114_fitleech_sn38-2023-01"),
    # issue #437 (2026-10-17): upstream-only architecture, after the pin.
    ("MuseSparkForConditionalGeneration", "Interchained/Muse-Spark-Open-816B-A42B-GGUF"),
    # engine #1667 (2026-11-09): the two UNCOVERED classes the daily census raised that day, declined
    # with their reasons in significant.json. report-only, because nothing a backend can read defines
    # Liwet's router and nanoGPT's GPT is not GPT-2 under another name.
    ("IndonesianChatLM", "fadhilahkhairogi/Liwet-400M-A188M-id-chat"),
    ("GPT", "learner-subodh/Delivery-Rider-Assistant-SLM"),
    # DeepseekV41ForCausalLM used to be here (issue #522, reviewed report-only, "needs real
    # support: engram + candidate blocks"). Since pin 5e6388a5a it is *mapped* - the fork ships
    # the deepseek41 architecture - so it is asserted below as covered, not as report-only.
):
    check(f"{cls} is reviewed report-only and documented",
          sig.get(cls, {}).get("watch") == "report" and cls not in archs and cls in gap_docs)
    rc, res = rc_of([model(mid, cls)], archs, sig)
    check(f"{cls} report-only -> exit 0, no alert",
          rc == 0 and res["reviewed_report"] == [cls] and res["significant"] == [])

# --- a reviewed class the pin starts to implement is covered, not alerted (2026-10-30) ------------
#
# DeepSeek-V4.1-Flash was reviewed report-only at #522 because the pinned deepseek4 path ran
# neither the engram nor the candidate-block machinery. Pin 5e6388a5a (fork #156, the port of
# upstream #31086 + #31300) implements it as its own architecture, so the same model must now be
# COVERED and silent: no "significant" alert, and no reviewed-report line either - alerting on a
# class the engine supports would be the false alarm this watcher exists to avoid.
rc, res = rc_of([model("overhand/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored", "DeepseekV41ForCausalLM")],
                archs, sig)
check("DeepseekV41ForCausalLM mapped at this pin -> covered, exit 0",
      rc == 0 and res[cw.COVERED] == ["overhand/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored"])
check("DeepseekV41ForCausalLM mapped -> no significant alert and no reviewed_report line",
      res["significant"] == [] and res["reviewed_report"] == [])

# --- FastFlowLM org + releases watch (engine#588) ----------------------------------
#
# The recorded payloads are what huggingface.co/api/models?author=FastFlowLM and
# api.github.com/repos/FastFlowLM/FastFlowLM/releases returned on 2026-10-26, trimmed to
# the fields the watcher reads. A fetch factory keyed on the URL keeps the test offline;
# the repo detail (`?blobs=true`, the only route that carries file sizes) is served from
# two recorded details so `fill_sizes` is exercised without the network.
FLM_FIX = os.path.join(ROOT, "tests", "fixtures", "fflm")


def fixture(name):
    with open(os.path.join(FLM_FIX, name)) as f:
        return json.load(f)


def flm_fetch(url):
    """Serve the recorded org listing or releases page for the URL the watcher asks for."""
    if url == cw.FLM_ORG_URL:
        return fixture("org.json")
    if url.startswith(cw.GH_RELEASES.split("?")[0]):
        return fixture("releases.json")
    raise AssertionError(f"unexpected URL: {url}")


def flm_detail(url):
    """Serve a recorded repo detail (file sizes live only on that route), else no files."""
    if "D3-Mini-NPU2" in url:
        return fixture("repo-mapped.json")
    if "MiniCPM-V-4.7-1B-NPU2" in url:
        return fixture("repo-unknown.json")
    return {"id": url, "siblings": []}


STATE_DIR = tempfile.mkdtemp(prefix="census-watch-test-")


def flm_state():
    """A state file that does not exist yet: the first run's defaults."""
    return cw.load_state(os.path.join(STATE_DIR, "state.json"))


# A listing entry for the NPU-gap case: an FLM Q4NX pack whose family the live registry maps
# for HRX only. It used to be D3-Mini (`qwen3_5_text` -> qwen35), until the dense qwen3_5 family
# was registered as served on the NPU route (engine#587: Qwen3.8-27B-NPU2 is that family and is
# measured there, bench/npu-leaderboard/27b-*.json), so the example moved to a family that is
# still HRX-only - gemma3, which FLM also publishes Q4NX packs for. Reading the live registry is
# the point: if gemma3 is ever registered for the NPU this check fails and the example moves again.
LFM3_REPO = {
    "id": "FastFlowLM/LFM3-1.7B-NPU2",
    "lastModified": "2026-10-28T00:00:00.000Z",
    "tags": ["base_model:LiquidAI/LFM3-1.7B", "license:other"],
    # reviewed, not an alias: registry/significant.json, watch: report -> the pack is listed, not alerted
    "config": {"architectures": ["Lfm3ForCausalLM"], "model_type": "lfm3"},
    "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.q4nx"}],
}

HRX_ONLY_REPO = {
    "id": "FastFlowLM/Gemma3-12B-NPU2",
    "lastModified": "2026-10-20T00:00:00.000Z",
    "tags": ["base_model:google/gemma-3-12b-it", "license:gemma"],
    "config": {"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3"},
    "siblings": [{"rfilename": "config.json"}, {"rfilename": "model.q4nx"},
                 {"rfilename": "vision_weight.q4nx"}],
}


# 16. The org listing is classified inline: tags, architecture, registry mapping and the
#     npu route all come from the one listing response, no per-model fetch.
live_archs = cw.load_registry()[0]
repos = cw.sweep_org(live_archs, fetch=flm_fetch)
by_id = {r["id"]: r for r in repos}
check("flm org: every recorded repo is read", len(repos) == 5)
d3 = by_id["FastFlowLM/D3-Mini-NPU2"]
check("flm org: a model_type-only repo still resolves (qwen3_5_text -> qwen35)",
      d3["registry_class"] == "Qwen3_5ForCausalLM" and d3["mapped"])
check("flm org: the dense qwen3_5 family is served on the NPU route (engine#587)",
      d3["npu"] and d3["npu_model_type"] == "qwen3_5" and not cw.npu_gap(d3))
check("flm org: license tag read", d3["license"] == "apache-2.0")
check("flm org: base model tag read, relation prefix dropped",
      d3["base_models"] == ["vllm-sr/d3-mini"])
check("flm org: q4nx file names come from the listing",
      [f["name"] for f in d3["q4nx"]] == ["model.q4nx", "vision_weight.q4nx"])
check("flm org: the listing carries no sizes", all(f["size"] == 0 for f in d3["q4nx"]))
qp = by_id["FastFlowLM/Qwen3-8B-NPU2"]
check("flm org: the npu route is reported for a family it serves",
      qp["npu"] and qp["npu_model_type"] == "qwen3")
llama = by_id["FastFlowLM/Llama-3.2-1B-NPU2"]
check("flm org: a second npu family is reported", llama["npu_model_type"] == "llama")
mini = by_id["FastFlowLM/MiniCPM-V-4.7-1B-NPU2"]
check("flm org: an unmapped architecture is unmapped, not a guess",
      mini["architecture"] == "MiniCPMV4_7ForConditionalGeneration"
      and not mini["mapped"] and cw.unmapped(mini) and not mini["npu"])
gguf = by_id["FastFlowLM/Qwen3.5-0.8B-gguf"]
check("flm org: a config-less gguf drop is a derivative, not a new family",
      gguf["derivative"] and not cw.unmapped(gguf) and not gguf["mapped"])
# The NPU-pack case (2026-10): a Q4NX pack whose family IS mapped, but only for a backend that
# cannot run a Q4NX directory. gemma3 in the live registry is exactly that, and FLM ships packs
# for it, so the guard's firing path keeps a real example now that qwen3_5 is NPU-served.
gap_repos = cw.sweep_org(live_archs, fetch=lambda url: [HRX_ONLY_REPO])
gap = gap_repos[0]
check("flm org: a Q4NX pack for an HRX-only family is an NPU gap",
      gap["mapped"] and gap["q4nx"] and not gap["npu"] and cw.npu_gap(gap) and cw.unmapped(gap))
check("flm org: a family the NPU route serves is not a gap",
      not cw.npu_gap(qp) and not cw.unmapped(qp) and not cw.npu_gap(llama))
check("flm org: a derivative drop is not an NPU gap", not cw.npu_gap(gguf))
check("flm org: a repo with no q4nx file is not an NPU gap",
      not cw.npu_gap({"id": "x/y", "q4nx": [], "mapped": True, "npu": False, "derivative": False}))

# 17. File sizes come from the repo detail, and only where they are still unknown.
calls = []

def counting_detail(url):
    calls.append(url)
    return flm_detail(url)

cw.fill_sizes(d3, fetch=counting_detail)
check("flm sizes: model.q4nx size read from the detail",
      [f["size"] for f in d3["q4nx"]] == [2750560152, 667061544])
cw.fill_sizes(d3, fetch=counting_detail)
check("flm sizes: a filled row is not refetched", len(calls) == 1)
broken = [{"name": "model.q4nx", "size": 0}]
with contextlib.redirect_stderr(io.StringIO()):
    kept = cw.fill_sizes({"id": "x/y", "q4nx": broken},
                          fetch=lambda url: (_ for _ in ()).throw(RuntimeError("x")))
check("flm sizes: a failed detail fetch keeps the repo (size 0, no crash)",
      [f["size"] for f in kept["q4nx"]] == [0])

# engine#1451: `Lfm3ForCausalLM` and `DFlash2DraftModel` are recorded in registry/significant.json as
# reviewed-not-an-alias. The class census already printed them `. REVIEWED`; the FLM half printed
# `!! NEW FLM ... registry: none` forever, which is what kept a daily issue open on an answered question.
_declined_rows = cw.sweep_org(live_archs, fetch=lambda url: fixture("org.json") + [LFM3_REPO])
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    cw.report_flm(_declined_rows, cw.sweep_releases(fetch=flm_fetch),
                  [r for r in _declined_rows if r["id"] == "FastFlowLM/LFM3-1.7B-NPU2"], [], answers=NO_ANSWERS)
_declined_listing = _buf.getvalue()

# 18. One line per repo, `!! NEW FLM` only for a family we cannot serve, and the exit code
#     is the OR of both watches (an unmapped class is not turned green by a quiet day).
state = flm_state()
buf, err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
    flm_rc, flm_ok = cw.watch_flm(live_archs, state, fetch=flm_fetch, detail=flm_detail, answers=NO_ANSWERS)
first = buf.getvalue()
check("flm first run: both watches ran", flm_ok)
check("flm first run: an unmapped family alerts (exit 1)", flm_rc == 1)
check("flm report: one line per repo",
      len([ln for ln in first.splitlines()
           if ln.startswith(FLM_LINE_PREFIXES)]) == 5)
check("flm report: the unknown family is marked !! NEW FLM",
      "  !! NEW FLM FastFlowLM/MiniCPM-V-4.7-1B-NPU2" in first)
check("flm report: a reviewed-and-declined class is listed, not alerted (engine#1451)",
      "  . FLM FastFlowLM/LFM3-1.7B-NPU2" in _declined_listing
      and "!! NEW FLM FastFlowLM/LFM3-1.7B-NPU2" not in _declined_listing
      and "registry: none (reviewed, not an alias" in _declined_listing)
check("flm report: a reviewed-and-declined class is not counted as an unmapped family",
      "LFM3-1.7B-NPU2" not in _declined_listing.split("publish a family this engine does not serve")[-1]
      if "publish a family this engine does not serve" in _declined_listing else True)
check("flm report: a mapped family is not marked",
      "  + FLM FastFlowLM/Qwen3-8B-NPU2" in first and "!! NEW FLM FastFlowLM/Qwen3-8B-NPU2" not in first)
check("flm report: the derivative drop does not alert",
      "!! NEW FLM FastFlowLM/Qwen3.5-0.8B-gguf" not in first)
check("flm report: an NPU-served family is not marked !! NPU FLM (engine#587)",
      "  + FLM FastFlowLM/D3-Mini-NPU2" in first
      and "!! NPU FLM FastFlowLM/D3-Mini-NPU2" not in first)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    gap_all = cw.sweep_org(live_archs, fetch=lambda url: fixture("org.json") + [HRX_ONLY_REPO])
    cw.report_flm(gap_all, cw.sweep_releases(fetch=flm_fetch),
                  [r for r in gap_all if r["id"] == HRX_ONLY_REPO["id"]], [], answers=NO_ANSWERS)
gapfirst = buf.getvalue()
check("flm report: an NPU pack for an HRX-only family is marked !! NPU FLM",
      "  !! NPU FLM FastFlowLM/Gemma3-12B-NPU2" in gapfirst)
check("flm report: the NPU gap is summarised by family",
      "Q4NX (NPU) pack(s) for a family the NPU route does not serve" in gapfirst
      and "gemma3 (1 pack(s))" in gapfirst)
check("flm report: sizes are named in GiB, one line each",
      "GiB" in first and "npu route: qwen3" in first and "registry: none" in first)
check("flm report: releases are listed with their tags",
      "  + FLM RELEASE v1.0.10" in first and len(first.splitlines()) < 40)
check("flm report: the container-format release is flagged",
      "kernel-format: chunk" in first and "q4nx" in first.split("  -> release(s) ")[-1])

# 19. The delta: the second run prints nothing, a moved repo prints one line.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    quiet_rc, quiet_ok = cw.watch_flm(live_archs, state, fetch=flm_fetch, detail=flm_detail, answers=NO_ANSWERS)
second = buf.getvalue()
check("flm delta: a second run is quiet and green",
      quiet_ok and quiet_rc == 0 and "nothing new since the last run" in second)
check("flm delta: no repo line on a quiet day",
      not any(p[:7] in second for p in FLM_LINE_PREFIXES))
gap_moved = [HRX_ONLY_REPO]
gap_repos2 = cw.sweep_org(live_archs, fetch=lambda url: gap_moved)
fresh, fresh_rel = cw.flm_deltas(state, gap_repos2, cw.sweep_releases(fetch=flm_fetch))
check("flm delta: only the moved repo is new", [r["id"] for r in fresh] == [gap_moved[0]["id"]])
check("flm delta: known releases are not new", fresh_rel == [])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    moved_rc = cw.report_flm(gap_repos2, cw.sweep_releases(fetch=flm_fetch), fresh, fresh_rel, answers=NO_ANSWERS)
check("flm delta: a moved NPU-gap pack alerts (an HRX-only family with a Q4NX pack)",
      moved_rc == 1 and "  !! NPU FLM FastFlowLM/Gemma3-12B-NPU2" in buf.getvalue())
# D3-Mini moving used to be that alert; it is the same move on a family the NPU route now serves,
# so the delta reports it and does not alert on it (engine#587).
moved = [dict(r) for r in fixture("org.json")]
moved[0]["lastModified"] = "2026-11-02T00:00:00.000Z"   # D3-Mini: qwen35, Q4NX, npu route qwen3_5
moved_repos = cw.sweep_org(live_archs, fetch=lambda url: moved)
fresh1, _ = cw.flm_deltas(state, moved_repos, cw.sweep_releases(fetch=flm_fetch))
check("flm delta: the moved repo is reported", [r["id"] for r in fresh1] == [moved[0]["id"]])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    served_rc = cw.report_flm(moved_repos, cw.sweep_releases(fetch=flm_fetch), fresh1, [], answers=NO_ANSWERS)
check("flm delta: a moved NPU-served pack does not alert",
      "  !! NPU FLM FastFlowLM/D3-Mini-NPU2" not in buf.getvalue())
served = cw.sweep_org(live_archs, fetch=lambda url: fixture("org.json"))
fresh2 = [r for r in served if r["id"] == "FastFlowLM/Qwen3-8B-NPU2"]   # qwen3: served on the NPU
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    served_rc = cw.report_flm(served, cw.sweep_releases(fetch=flm_fetch), fresh2, [], answers=NO_ANSWERS)
check("flm delta: a moved family the NPU route serves stays green",
      served_rc == 0 and buf.getvalue().count("  + FLM ") == 1
      and "!! NPU FLM" not in buf.getvalue())

# 20. `--full` prints every repo and release on demand.
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    full_rc = cw.report_flm(moved_repos, cw.sweep_releases(fetch=flm_fetch), [], [], full=True, answers=NO_ANSWERS)
full = buf.getvalue()
check("flm --full: every repo listed", sum(full.count(p) for p in FLM_LINE_PREFIXES) == 5)
check("flm --full: every release listed", full.count("  + FLM RELEASE ") == 3)
check("flm --full: an unmapped repo in the listing still alerts", full_rc == 1)
# 21. The supporting-upstream pins (registry/support_pins.json): a moved commit and a new d3
#     repo both alert, an unchanged pin and a listed d3 repo do not, and a failed fetch is a
#     runtime failure rather than a quiet day.
PINS = {
    "repos": {
        "ROCm/FLM_Q4NX_Converter": {"commit": "aaa111", "reviewed": "2026-10-30", "why": "converter"},
        "vllm-project/semantic-router": {"commit": "bbb222", "reviewed": "2026-10-30", "why": "d3"},
    },
    "d3_sources": {"org": "vllm-sr", "prefix": "d3", "known": ["vllm-sr/d3-lite", "vllm-sr/d3-turbo"]},
}


def support_fetch(converter="aaa111", router="bbb222", repos=("vllm-sr/d3-lite",)):
    def fetch(url):
        if "/commits" in url:
            sha = converter if "Q4NX" in url else router
            return [{"sha": sha, "commit": {"committer": {"date": "2026-10-30T00:00:00Z"}}}]
        return [{"id": rid} for rid in repos]
    return fetch


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sup_rc, sup_ok = cw.watch_support(PINS, fetch=support_fetch())
steady = buf.getvalue()
check("support pins: unchanged pins and listed d3 repos are green", sup_ok and sup_rc == 0)
check("support pins: an unchanged pin is printed with its commit",
      "+ SUPPORT PIN ROCm/FLM_Q4NX_Converter  aaa111" in steady)
check("support pins: nothing is flagged moved", "!! SUPPORT MOVED" not in steady)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sup_rc, _ = cw.watch_support(PINS, fetch=support_fetch(converter="ccc333"))
moved_text = buf.getvalue()
check("support pins: a moved converter commit alerts", sup_rc == 1)
check("support pins: the moved repo is named with both commits",
      "!! SUPPORT MOVED ROCm/FLM_Q4NX_Converter" in moved_text and "ccc333" in moved_text
      and "aaa111" in moved_text)
check("support pins: the fix is in the pin file, and says so",
      "bump the commit and `reviewed` in registry/support_pins.json" in moved_text)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sup_rc, _ = cw.watch_support(PINS, fetch=support_fetch(repos=("vllm-sr/d3-lite", "vllm-sr/d3-quasar")))
check("support pins: a d3 repo not in the list alerts", sup_rc == 1)
check("support pins: the new d3 repo is named", "vllm-sr/d3-quasar" in buf.getvalue())
check("support pins: a non-d3 repo in the org is not this watch's business",
      cw.watch_support(PINS, fetch=support_fetch(repos=("vllm-sr/mmbert-base-32k",)))[0] == 0)

with contextlib.redirect_stderr(io.StringIO()):
    sup_rc, sup_ok = cw.watch_support(PINS, fetch=lambda url: (_ for _ in ()).throw(RuntimeError("no network")))
check("support pins: a failed fetch is a failure, not a quiet day", sup_rc == 0 and not sup_ok)
check("support pins: a failure does not turn an alert green",
      cw.exit_code(0, 0, True, 0) == 0 and cw.exit_code(0, 0, True, 1) == 1
      and cw.exit_code(0, 0, False, 0) == 2 and cw.exit_code(0, 0, False, 1) == 1)

committed = json.load(open(os.path.join(ROOT, "registry", "support_pins.json")))
check("support pins: the committed file parses and pins both upstreams",
      set(committed["repos"]) >= {"ROCm/FLM_Q4NX_Converter", "vllm-project/semantic-router"}
      and all(len(e["commit"]) == 40 and e.get("reviewed") for e in committed["repos"].values()))
check("support pins: every d3 source in the file is prefixed like the watch expects",
      all(r.startswith(committed["d3_sources"]["org"] + "/" + committed["d3_sources"]["prefix"])
          for r in committed["d3_sources"]["known"]))

state_path = os.path.join(STATE_DIR, "round-trip.json")
cw.save_state(state_path, state)
reloaded = cw.load_state(state_path)
check("flm state: ids and lastModified round-trip through the state file",
      reloaded["flm_models"] == state["flm_models"] and len(reloaded["flm_models"]) == 5)
check("flm state: release tags round-trip through the state file",
      reloaded["flm_tags"] == state["flm_tags"] and len(reloaded["flm_tags"]) == 3)

# 21. Releases: flags come from the notes and the asset names.
releases = cw.sweep_releases(fetch=flm_fetch)
by_tag = {r["tag"]: r for r in releases}
check("flm releases: tag, date and asset names read",
      by_tag["v1.0.10"]["date"].startswith("2026-10-22") and len(by_tag["v1.0.10"]["assets"]) == 9
      and "HRX_fastflowlm_1.0.10_linux.tar.gz" in by_tag["v1.0.10"]["assets"])
check("flm releases: a q4nx mention in the notes is flagged",
      "q4nx" in by_tag["v0.9.40"]["kernel_format"])
check("flm releases: a clean release is not flagged", by_tag["v1.0.9"]["kernel_format"] == [])
check("flm releases: an asset name alone flags the format",
      cw.kernel_format_flags({"tag_name": "v9", "assets": [{"name": "flm.q4nx"}]}) == ["q4nx"])
check("flm releases: layout/chunk markers are what the flag reports",
      cw.kernel_format_flags({"tag_name": "v9", "body": "new chunk table layout"})
      == ["chunk", "layout"])

# 22. The alert contract: neither watch can silence the other, and a failed FLM fetch is a
#     runtime failure (2), never a quiet day.
check("exit code: an uncovered class alone -> 1", cw.exit_code(1, 0, True) == 1)
check("exit code: an unmapped FLM family alone -> 1", cw.exit_code(0, 1, True) == 1)
check("exit code: both -> 1", cw.exit_code(1, 1, True) == 1)
check("exit code: an alert beats a failed FLM fetch", cw.exit_code(1, 0, False) == 1)
check("exit code: nothing new and both watches ran -> 0", cw.exit_code(0, 0, True) == 0)
check("exit code: a failed FLM fetch alone -> 2 (runtime failure)", cw.exit_code(0, 0, False) == 2)


def fail_fetch(url):
    raise RuntimeError(f"no network: {url}")


buf, err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
    dead_rc, dead_ok = cw.watch_flm(live_archs, flm_state(), fetch=fail_fetch, answers=NO_ANSWERS)
check("flm failure: reported as a failure, not as nothing new",
      dead_ok is False and dead_rc == 0 and "FLM WATCH FAILED" in err.getvalue())
check("flm failure: combined exit code is 2, so the daily job notices",
      cw.exit_code(0, dead_rc, dead_ok) == 2)

# 23. The family resolver's contract, on a synthetic registry shaped like architectures.json.
FLM_ARCHS = {
    "Qwen3_5ForCausalLM": {"gguf": "qwen35", "backends": ["hrx", "npu"], "npu_model_type": "qwen3_5"},
    "MiniCPMV4_6ForConditionalGeneration": {"gguf": "qwen35", "backends": ["hrx"]},
    "GptOssForCausalLM": {"gguf": "gpt-oss", "backends": ["hrx"]},
}
check("family: the modality suffix is stripped before comparing",
      cw.resolve_family("qwen3_5_text", FLM_ARCHS) == "Qwen3_5ForCausalLM")
check("family: the gguf string is matched with punctuation dropped",
      cw.resolve_family("gpt_oss", FLM_ARCHS) == "GptOssForCausalLM")
check("family: an exact class wins", cw.resolve_family("GptOssForCausalLM", FLM_ARCHS) == "GptOssForCausalLM")
check("family: a version bump is not a prefix match", cw.resolve_family("qwen3_6_text", FLM_ARCHS) is None)
check("family: an unmapped family stays unmapped", cw.resolve_family("minicpmv4_7", FLM_ARCHS) is None)
check("family: an empty name maps to nothing", cw.resolve_family("", FLM_ARCHS) is None)

# 24. The 2026-10-27 census batch (engine#619) stays covered. Four classes were reported; each is
#     now either reviewed report-only and documented, or recorded as a variant/remote-code class
#     that is documented and deliberately left out of significant.json, so a *native* upload of
#     it still alerts. GraniteSwitchForCausalLM was the batch's only live alert: the mirror the
#     run named is gated, but the class is IBM's own and its configs are public, so it was
#     reviewable after all (docs/arch-gaps.md, `granite_switch`).
batch_619 = {
    "LiveMemForCausalLM": ("livemem", "Popbones/LiveMem-RL-mlx-q4-backbone"),
    "CDMMoELMPoincare": ("cdm_moe_poincare", "DuoNeural/Lattice-7-Hermes-CDM-Poincare-Instruct"),
    "Senryu2ForCausalLM": ("senryu2", "KerzgrAI/Senryu-2.2-1B"),
    "GraniteSwitchForCausalLM": ("granite_switch", "ibm-granite/granite-switch-4.1-8b"),
}
for cls, (model_type, mid) in batch_619.items():
    entry = sig.get(cls) or {}
    check(f"{cls} is reviewed report-only and documented",
          entry.get("model_type") == model_type and entry.get("watch") == "report"
          and cls not in archs and cls in gap_docs)
    rc, res = rc_of([model(mid, cls)], archs, sig)
    check(f"{cls} report-only -> exit 0, no alert",
          rc == 0 and res["reviewed_report"] == [cls] and res["significant"] == []
          and res[cw.UNCOVERED] == [mid])
_, out = text_of([model("ibm-granite/granite-switch-4.1-8b", "GraniteSwitchForCausalLM")], archs, sig)
check("granite switch is printed as REVIEWED, not UNCOVERED",
      ". REVIEWED GraniteSwitchForCausalLM" in out and "UNCOVERED" not in out)

variant_619 = {
    "LNewqForConditionalGeneration": "crussella0129/LNewq",
    "Newq1ForConditionalGeneration": "crussella0129/Newq1",
    "NewqForConditionalGeneration": "crussella0129/Newq",
    "KeuralMoECausalLM": "hossain0909/keural-dpo2-final",
}
for cls, mid in variant_619.items():
    check(f"{cls} is documented and not silenced in significant.json",
          cls in gap_docs and cls not in sig and cls not in archs)
    rc, res = rc_of([model(mid, cls, remote=True)], archs, sig)
    check(f"{cls} custom-code upload -> reported, exit 0",
          rc == 0 and res["remote_code"] == [cls])
    rc, res = rc_of([model(f"native/{cls}", cls)], archs, sig)
    check(f"{cls} without auto_map still alerts",
          rc == 1 and res["significant"] == [] and res["remote_code"] == [])

# The counts quoted in the docs come from the committed census, not from a live walk.
with open(os.path.join(ROOT, "registry", "census.json")) as f:
    counts = json.load(f)["raw"]["counts"]
check("census.json records every engine#619 class",
      all(counts.get(cls) for cls in list(batch_619) + list(variant_619)))
check("census.json shows granite switch is not a one-off",
      counts["GraniteSwitchForCausalLM"] > 1)


# 25. The 2026-11-05 census batch (engine#619): two reviewed non-alias classes are recorded and
#     report-only, and FastFlowLM's NPU2 families are answered in docs/npu.md instead of being
#     mapped as GGUF aliases the pinned backends do not accept.
batch_1105 = {
    "Ernie5_5_MoeForCausalLM": ("ernie5_5_moe", "ml-resources/ERNIE-5.5-Flash-Base"),
    "Factorial": ("Factorial", "bull220202/Factorial_Base_Tiny_300M_Ver_LM"),
}
for cls, (model_type, mid) in batch_1105.items():
    entry = sig.get(cls) or {}
    check(f"{cls} is reviewed report-only and documented",
          entry.get("model_type") == model_type and entry.get("watch") == "report"
          and cls not in archs and cls in gap_docs)
    rc, res = rc_of([model(mid, cls)], archs, sig)
    check(f"{cls} report-only -> exit 0, no alert",
          rc == 0 and res["reviewed_report"] == [cls] and res["significant"] == []
          and res[cw.UNCOVERED] == [mid])
check("census.json records the 2026-11-05 classes",
      counts.get("Ernie5_5_MoeForCausalLM", 0) > 0 and counts.get("Factorial", 0) > 0)

with open(os.path.join(ROOT, "docs", "npu.md")) as f:
    npu_docs = f.read()
# [repo id, the architecture class its config declares ("" for a config-less drop)]
flm_1105 = [
    ("FastFlowLM/MiniCPM-OCR-1.0-NPU2", "MiniCPMOcrForConditionalGeneration"),
    ("FastFlowLM/MiniCPM-V-4.7-35B-A3B-NPU2", "MiniCPMV4_7ForConditionalGeneration"),
    ("FastFlowLM/MiniCPM-V-4.7-1B-NPU2", "MiniCPMV4_7ForConditionalGeneration"),
    ("FastFlowLM/Qwen3.8-27B-DFlash2-NPU2", "DFlash2DraftModel"),
    ("FastFlowLM/LFM3-1.7B-NPU2", "Lfm3ForCausalLM"),
    ("FastFlowLM/Hy-Memory-30B-A3B-NPU2", "HYV3ForCausalLM"),
    ("FastFlowLM/smolvla-NPU2", ""),
]
for repo_id, cls in flm_1105:
    check(f"{repo_id} is recorded in docs/npu.md", repo_id in npu_docs)
    if not cls:
        continue
    if cls == "HYV3ForCausalLM":
        check("Hy-Memory's hy_v3 is mapped for hrx only, so its pack is an NPU gap",
              archs[cls]["backends"] == ["hrx"] and "npu" not in archs[cls]["backends"]
              and cls in npu_docs)
        hy = cw.flm_repo({"id": repo_id, "config": {"architectures": [cls]},
                          "siblings": [{"rfilename": "model.q4nx"}]}, archs)
        check("Hy-Memory pack resolves to the recorded npu gap, not a mapping",
              hy["mapped"] and cw.npu_gap(hy) and cw.unmapped(hy) and not hy["npu"])
    else:
        check(f"{cls} stays out of the registry (no GGUF alias claimed)",
              cls not in archs and cls in npu_docs)
        repo = cw.flm_repo({"id": repo_id, "config": {"architectures": [cls]},
                            "siblings": [{"rfilename": "model.q4nx"}]}, archs)
        check(f"{repo_id} still reads as an unserved family",
              not repo["mapped"] and cw.unmapped(repo))


# 26. The 2026-11-06 census batch (engine#1253): the two new UNCOVERED classes are recorded,
#     report-only, and the FastFlowLM NPU rows are answered family by family in docs/npu.md
#     instead of being mapped as GGUF aliases or given an `npu` backend they do not have.
batch_1106 = {
    "Lfm3ForCausalLM": ("lfm3", "D0zy/essay-model-merged"),
    "NeedleForToolCalling": ("needle", "Amdkn7/needle3"),
}
for cls, (model_type, mid) in batch_1106.items():
    entry = sig.get(cls) or {}
    check(f"{cls} is reviewed report-only and documented",
          entry.get("model_type") == model_type and entry.get("watch") == "report"
          and cls not in archs and cls in gap_docs)
    rc, res = rc_of([model(mid, cls)], archs, sig)
    check(f"{cls} report-only -> exit 0, no alert",
          rc == 0 and res["reviewed_report"] == [cls] and res["significant"] == []
          and res[cw.UNCOVERED] == [mid])
check("census.json shows LFM3 is a family, not a one-off",
      counts.get("Lfm3ForCausalLM", 0) > 1)
check("census.json records the NeedleForToolCalling family",
      counts.get("NeedleForToolCalling", 0) > 1)

# The 2026-11-06 body's `!! NPU FLM` rows: each family is mapped for hrx only, so its Q4NX pack is
# an NPU gap and docs/npu.md has to carry the per-family answer.
# qwen35 is deliberately absent: engine#587 measured the route on that family, so it is registered
# for `npu` now and its pack is served instead of being a gap - pinned by the two checks below.
npu_flm_families = {
    "dflash": ("DFlashDraftModel", "Nemotron-3.5-Bolt-E4B-DFlash-NPU2"),
    "gemma4": ("Gemma4ForConditionalGeneration", "Gemma4-12B-IT-NPU2"),
    "hunyuan-dense": ("HunYuanDenseV1ForCausalLM", "Hy-MT2-1.8B-NPU2"),
    "gemma3": ("Gemma3ForCausalLM", "medgemma-1.5-4b-it-NPU2"),
    "lfm2": ("Lfm2ForCausalLM", "LFM2-2.6B-NPU2"),
    "gpt-oss": ("GptOssForCausalLM", "GPT-OSS-20B-NPU2"),
    "hy_v3": ("HYV3ForCausalLM", "FastFlowLM/Hy-Memory-30B-A3B-NPU2"),
}
for gguf, (cls, repo_id) in npu_flm_families.items():
    check(f"{gguf}: {cls} is mapped for hrx only",
          archs[cls]["gguf"] == gguf and archs[cls]["backends"] == ["hrx"])
    check(f"{gguf}: the NPU-route answer for {repo_id} is in docs/npu.md",
          f"`{gguf}`" in npu_docs and repo_id in npu_docs)
check("qwen35: the dense family is registered for the NPU route (engine#587)",
      archs["Qwen3_5ForConditionalGeneration"]["gguf"] == "qwen35"
      and archs["Qwen3_5ForConditionalGeneration"]["backends"] == ["hrx", "npu"]
      and archs["Qwen3_5ForConditionalGeneration"]["npu_model_type"] == "qwen3_5")
check("qwen35: the bare vision wrapper is not claimed",
      archs["Qwen3_5Model"]["backends"] == ["hrx"])
check("qwen35: the NPU-route answer for Qwen3.8-27B-NPU2 is in docs/npu.md",
      "`qwen35`" in npu_docs and "Qwen3.8-27B-NPU2" in npu_docs)
pack = cw.flm_repo({"id": "FastFlowLM/Qwen3.8-27B-NPU2",
                    "config": {"architectures": ["Qwen3_5ForConditionalGeneration"]},
                    "siblings": [{"rfilename": "model.q4nx"}]}, archs)
check("a qwen35 NPU pack is served, not the recorded gap",
      pack["mapped"] and pack["npu"] and pack["npu_model_type"] == "qwen3_5"
      and not cw.npu_gap(pack) and not cw.unmapped(pack))

shutil.rmtree(STATE_DIR, ignore_errors=True)
# 16. Gated repos (engine#619): the request carries a token when one is in the environment, and
#     the run says so instead of repeating "gated repos need a token" at a run that sent one.
#     No network here; this pins the header construction and the report text, not HF's answer.
for var in cw.TOKEN_VARS:
    os.environ.pop(var, None)
check("no token in the environment -> UA only",
      cw.request_headers() == {"User-Agent": "1bit-census-watch"} and cw.hf_token() == "")
os.environ[cw.TOKEN_VARS[0]] = "hf_example"
check("HF_TOKEN -> Bearer header",
      cw.request_headers().get("Authorization") == "Bearer hf_example")
os.environ.pop(cw.TOKEN_VARS[0])
os.environ[cw.TOKEN_VARS[1]] = "hf_second"
check("HUGGING_FACE_HUB_TOKEN is the fallback",
      cw.request_headers().get("Authorization") == "Bearer hf_second")
check("the unverifiable line names the token it already sent",
      "(HF_TOKEN sent, still no config)" in text_of([model("gated/one")])[1])
os.environ.pop(cw.TOKEN_VARS[1], None)
check("with no token the line stays the plain advice",
      "HF_TOKEN sent" not in text_of([model("gated/one")])[1])

# 17. A rejected token (401) must not cost the whole run: the page comes back anonymously.
#     Pins the fallback that keeps engine#619's watcher useful on a stale HF_TOKEN.
seen_auth = []


class _Stub(io.BytesIO):
    headers = {"Link": ""}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(req, timeout=0):
    seen_auth.append(req.headers.get("Authorization"))
    if req.headers.get("Authorization"):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
    return _Stub(b'[{"id": "x/y"}]')


os.environ[cw.TOKEN_VARS[0]] = "hf_stale"
cw._AUTH_REJECTED = False
_real = urllib.request.urlopen
urllib.request.urlopen = _fake_urlopen
try:
    page, _ = cw.hf_get_page("https://example.invalid/api/models")
finally:
    urllib.request.urlopen = _real
    os.environ.pop(cw.TOKEN_VARS[0], None)
    cw._AUTH_REJECTED = False
check("a 401 falls back to anonymous instead of failing the run",
      page == [{"id": "x/y"}] and seen_auth[0] == "Bearer hf_stale" and seen_auth[1] is None)


# 17. engine#1922: DeepSeek V4.1 Flash SuperExpert is reviewed and report-only -- listed, not alerted.
# The class is remote code (its own modeling_ module) AND ships AW-SECD-compressed weights, so nothing at
# the pin decodes it: mapping it in architectures.json would name a backend that cannot read the file.
_DS = "DeepseekV4FlashSuperExpertForCausalLM"
check("engine#1922 class is recorded as report-only",
      sig.get(_DS, {}).get("watch") == "report")
check("engine#1922 entry states WHY (the compressed weights), not just that it is declined",
      "AW-SECD" in sig.get(_DS, {}).get("family", ""))
check("engine#1922 class stays out of the runtime registry", _DS not in archs)
check("engine#1922 entry records its model_type", sig.get(_DS, {}).get("model_type") == "deepseek_v4_flash_superexpert")


# 27. engine#2171: an answered gap stays answered. CI keeps no state, so every scheduled run sees the whole
# FastFlowLM org as new; before registry/flm_answers.json each family docs/npu.md had already answered
# printed `!! NPU FLM` / `!! NEW FLM` again and re-filed the census issue the day after it closed.
# The rows are the 2026-11-11 run's alerting packs (run 43016580831: id and the architecture each declares).
RUN_2171 = [
    ("FastFlowLM/MiniCPM-V-4.7-1B-NPU2", "MiniCPMV4_7ForConditionalGeneration"),
    ("FastFlowLM/Hy-Memory-30B-A3B-NPU2", "HYV3ForCausalLM"),
    ("FastFlowLM/MiniCPM-OCR-1.0-NPU2", "MiniCPMOcrForConditionalGeneration"),
    ("FastFlowLM/MiniCPM-V-4.7-35B-A3B-NPU2", "MiniCPMV4_7ForConditionalGeneration"),
    ("FastFlowLM/Nemotron-3.5-Bolt-E4B-DFlash-NPU2", "DFlashDraftModel"),
    ("FastFlowLM/Nemotron-3.5-Bolt-E4B-NPU2", "Gemma4ForConditionalGeneration"),
    ("FastFlowLM/Hy-MT2-1.8B-NPU2", "HunYuanDenseV1ForCausalLM"),
    ("FastFlowLM/Gemma4-12B-IT-NPU2", "Gemma4UnifiedForConditionalGeneration"),
    ("FastFlowLM/smolvla-NPU2", None),
    ("FastFlowLM/medgemma-1.5-4b-it-NPU2", "gemma3_text"),
    ("FastFlowLM/LFM2-2.6B-NPU2", "Lfm2ForCausalLM"),
    ("FastFlowLM/LFM2-1.2B-NPU2", "Lfm2ForCausalLM"),
    ("FastFlowLM/GPT-OSS-Safeguard-20b-NPU2", "GptOssForCausalLM"),
    ("FastFlowLM/GPT-OSS-20B-NPU2", "GptOssForCausalLM"),
    ("FastFlowLM/Gemma3-4B-NPU2", "gemma3_text"),
    ("FastFlowLM/Gemma3-1B-NPU2", "Gemma3ForCausalLM"),
    ("FastFlowLM/Gemma4-E4B-IT-NPU2", "Gemma4ForConditionalGeneration"),
    ("FastFlowLM/Gemma4-E2B-IT-NPU2", "Gemma4ForConditionalGeneration"),
    ("FastFlowLM/Qwen3.6-35B-A3B-NPU2", "Qwen3_5MoeForConditionalGeneration"),
    ("FastFlowLM/medgemma-4b-it-NPU2", "gemma3_text"),
    ("FastFlowLM/Translategemma-4B-Instruct-NPU2", "gemma3_text"),
    ("FastFlowLM/Qwen2.5-VL-3B-Instruct-NPU2", "Qwen2_5_VLForConditionalGeneration"),
    ("FastFlowLM/LFM2.5-1.2B-Thinking-NPU2", "Lfm2ForCausalLM"),
    ("FastFlowLM/Phi4-mini-Instruct-NPU2", "Phi3ForCausalLM"),
    ("FastFlowLM/LFM2-2.6B-Transcript-NPU2", "Lfm2ForCausalLM"),
    ("FastFlowLM/LFM2.5-1.2B-NPU2", "Lfm2ForCausalLM"),
    ("FastFlowLM/LFM2-1.2B-Tool-NPU2", "Lfm2ForCausalLM"),
    ("FastFlowLM/Whisper-V3-Turbo-NPU2", "WhisperForConditionalGeneration"),
    ("FastFlowLM/Embedding-Gemma-300M-NPU2", "Gemma3TextModel"),
    ("FastFlowLM/Gemma3-270M-NPU2", "Gemma3ForCausalLM"),
    ("FastFlowLM/Gemma3-4B-Text-NPU2", "gemma3_text"),
]


def run_row(rid, arch, q4nx=True):
    rec = {"id": rid, "lastModified": "2026-11-11T00:00:00.000Z", "tags": [],
           "siblings": ([{"rfilename": "model.q4nx"}] if q4nx else [{"rfilename": "README.md"}])}
    if arch:
        rec["config"] = ({"architectures": [arch]} if arch[0].isupper() else {"model_type": arch})
    return cw.flm_repo(rec, live_archs)


_rows2171 = [run_row(rid, arch, q4nx=(arch is not None)) for rid, arch in RUN_2171]
for _answers, _tag in ((NO_ANSWERS, "before"), (None, "after")):
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        _rc = cw.report_flm(_rows2171, [], _rows2171, [], answers=_answers)
    if _tag == "before":
        _before_rc, _before = _rc, _buf.getvalue()
    else:
        _after_rc, _after = _rc, _buf.getvalue()
check("engine#2171 before: the 2026-11-11 rows alert with no answers (the rule is unchanged)",
      _before_rc == 1 and _before.count("  !! NPU FLM ") + _before.count("  !! NEW FLM ") == len(RUN_2171))
check("engine#2171 after: every one of the 31 rows is answered, the run is quiet",
      _after_rc == 0 and "  !! " not in _after
      and _after.count("  . NPU FLM ") + _after.count("  . FLM ") == len(RUN_2171))
check("engine#2171 after: each answered line names the issue that tracks it",
      "FastFlowLM/MiniCPM-V-4.7-1B-NPU2" in _after and "answered: engine#1788" in _after
      and "answered: engine#1783" in _after and "answered: engine#1786" in _after)
check("engine#2171 after: the summary says how many were answered, not that a family is unserved",
      "31 pack(s) already answered" in _after and "does not serve" not in _after)

_fresh_family = run_row("FastFlowLM/Brandnew-3B-NPU2", "BrandnewForCausalLM")
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    _rc = cw.report_flm(_rows2171 + [_fresh_family], [], _rows2171 + [_fresh_family], [])
check("engine#2171: a family with no answer still alerts beside the answered ones",
      _rc == 1 and "  !! NEW FLM FastFlowLM/Brandnew-3B-NPU2" in _buf.getvalue())
_served = run_row("FastFlowLM/Qwen3-8B-NPU2", "Qwen3ForCausalLM")
_fake = {"architectures": {"Qwen3ForCausalLM": {"answer": "x", "issue": 1}}, "repos": {}}
check("engine#2171: an answer cannot touch a family the NPU route serves",
      cw.answer_for(_served, _fake) is None and cw.flm_line(_served, answers=_fake).startswith("+ FLM "))
_deriv = cw.flm_repo({"id": "FastFlowLM/Qwen3-8B-gguf", "tags": ["gguf"], "siblings": []}, live_archs)
check("engine#2171: a derivative drop is never answered (it is not a gap)",
      cw.answer_for(_deriv, {"architectures": {}, "repos": {"FastFlowLM/Qwen3-8B-gguf": {"answer": "x"}}}) is None)

# The committed file: every answer names its doc, is for a family that is still a gap, and has a row in docs/npu.md.
_committed = json.load(open(os.path.join(ROOT, "registry", "flm_answers.json")))
_npu_md = open(os.path.join(ROOT, "docs", "npu.md")).read()
_arch_answers = _committed.get("architectures") or {}
check("flm answers: the committed file parses and is not empty", len(_arch_answers) >= 10)
check("flm answers: each entry says what the answer is and where it is recorded",
      all(e.get("answer") and (e.get("doc") or "").startswith("docs/npu.md")
          for e in list(_arch_answers.values()) + list((_committed.get("repos") or {}).values())))
check("flm answers: a tracking issue is a number",
      all(isinstance(e["issue"], int) for e in _arch_answers.values() if "issue" in e))
_stale = [a for a in _arch_answers
          if "npu" in ((live_archs.get(cw.resolve_family(a, live_archs) or "") or {}).get("backends") or [])]
check(f"flm answers: no answer is for a family the registry serves on the NPU route (stale: {_stale})", not _stale)
_undocumented = [a for a in list(_arch_answers) + list(_committed.get("repos") or {}) if f"`{a}`" not in _npu_md]
check(f"flm answers: every answered architecture has a row in docs/npu.md (missing: {_undocumented})",
      not _undocumented)

# 21b. engine#2171: a pin that names its `paths` moves only when a commit touches one of them.
# vllm-project/semantic-router lands dozens of router-side commits a day; the d3 contract it is pinned for
# did not change between a9219240e and 15b4fb5ad, yet the whole-repo head alerted on every run.
PATH_PINS = {"repos": {"vllm-project/semantic-router": {
    "commit": "pin000", "reviewed": "2026-11-11", "why": "d3",
    "paths": ["src/model-runtime/vllm_srun/families/vela3_d3", "config/schemas/systemone"]}}}


def path_fetch(head="head99", touched=()):
    """Head `head`; the last commit touching each path is the same at pin and head unless the path is in `touched`."""
    def fetch(url):
        if "&path=" in url:
            sha = url.split("sha=")[1].split("&")[0]
            path = urllib.parse.unquote(url.split("path=")[1].split("&")[0])
            return [{"sha": ("new-" + path) if (sha == head and path in touched) else ("old-" + path)}]
        if "/commits" in url:
            return [{"sha": head, "commit": {"committer": {"date": "2026-11-11T00:00:00Z"}}}]
        return []
    return fetch


import urllib.parse  # noqa: E402
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    p_rc, p_ok = cw.watch_support(PATH_PINS, fetch=path_fetch())
check("support paths: a head that moved without touching a watched path is green", p_ok and p_rc == 0)
check("support paths: ...and the line says why it is green",
      "no commit since the pin touches the 2 watched path(s)" in buf.getvalue()
      and "!! SUPPORT MOVED" not in buf.getvalue())
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    p_rc, _ = cw.watch_support(PATH_PINS, fetch=path_fetch(touched=("config/schemas/systemone",)))
check("support paths: a commit touching a watched path alerts and names the path",
      p_rc == 1 and "!! SUPPORT MOVED vllm-project/semantic-router" in buf.getvalue()
      and "watched paths changed: config/schemas/systemone" in buf.getvalue())
with contextlib.redirect_stdout(io.StringIO()):
    p_rc, _ = cw.watch_support(PATH_PINS, fetch=path_fetch(head="pin000"))
check("support paths: a head equal to the pin reads no paths and is green", p_rc == 0)
check("support paths: a pin without `paths` still moves on any commit (the converter's rule is unchanged)",
      cw.watch_support(PINS, fetch=support_fetch(router="zzz999"))[0] == 1)
check("support paths: the committed router pin watches the d3 family and the System One contract",
      any("vela3_d3" in x for x in committed["repos"]["vllm-project/semantic-router"].get("paths", []))
      and any("systemone" in x for x in committed["repos"]["vllm-project/semantic-router"].get("paths", [])))

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")