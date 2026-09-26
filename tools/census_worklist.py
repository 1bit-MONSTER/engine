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
"""Build registry/unmapped_worklist.json: every HF architecture the census saw that the
registry does not map, with its model count and a bucket that says what closing it takes.

usage: tools/census_worklist.py [--out registry/unmapped_worklist.json] [--top N]

The census (tools/census.py) reports mapped coverage as a single percentage. That number
hides the fact that the unmapped tail is made of very different kinds of work: a name the
runtime already accepts (a bridge fix), a same-shape sibling that needs a shape check, a
genuine decoder the runtime has no arch for, and non-text modalities (TTS, vision, audio,
diffusion) that no text decoder can honestly claim.

Buckets:
  bridge-blind-spot  the backend runtime already accepts the architecture's GGUF arch, so
                     only the HF -> GGUF bridge (converter registration / registry_build) is
                     missing. No kernels.
  rename-alias       a same-shape sibling/rename of a supported family. Registering the name
                     may be enough, but it needs a shape/parity check first.
  new-LM-family      a genuine text decoder the runtime has no arch for. Needs implementation.
  non-LM-modality    TTS/vision/audio/diffusion/action. Needs modality support, never an alias.

The high-count entries are reviewed by hand (REVIEWED below, the record of that review). The
long tail is bucketed by rules over the pinned converter/runtime and a name heuristic, and
carries needs_review=true; tasks 2-5 re-bucket those as they are worked.

Every input is read from the pins or the saved census; nothing is typed in except the review.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import registry_build as rb  # noqa: E402

BUCKETS = ("bridge-blind-spot", "rename-alias", "new-LM-family", "non-LM-modality")

# Model families reviewed by hand because they carry the most models (or because a rule
# misfiled them). arch -> (bucket, note). This is the review record; keep it in step with
# docs/arch-gaps.md and registry/significant.json.
REVIEWED = {
    # --- the runtime has the arch; only the HF -> GGUF bridge is missing ---
    "GPTJForCausalLM": ("bridge-blind-spot", "gptj is in both runtimes; the pin's converter no longer registers the HF name"),
    "BertLMHeadModel": ("bridge-blind-spot", "bert is in both runtimes; the converter registers BertModel/BertForMaskedLM, not the LM head"),
    "MptForCausalLM": ("bridge-blind-spot", "mpt is in both runtimes; MPTForCausalLM is registered, the mixed-case spelling is not"),
    "GPTNeoXModel": ("bridge-blind-spot", "gptneox is in both runtimes; only GPTNeoXForCausalLM is registered"),
    "GPT2Model": ("bridge-blind-spot", "gpt2 is in both runtimes; only GPT2LMHeadModel is registered"),
    "MistralModel": ("bridge-blind-spot", "mistral is served as llama; MistralForCausalLM is registered, the bare Model is not"),
    # --- same-shape siblings/renames; register the name after a shape check ---
    "RobertaForCausalLM": ("rename-alias", "bert-shaped, causal head; roberta maps to bert, the causal variant needs a check"),
    "XLMRobertaForCausalLM": ("rename-alias", "bert-shaped, causal head; only the encoder is registered"),
    "CamembertForCausalLM": ("rename-alias", "bert-shaped, causal head; only the encoder is registered"),
    "GPT2LMHeadCustomModel": ("rename-alias", "gpt2 shape under a custom head name"),
    "GPJTGPT2ModelForCausalLM": ("rename-alias", "gpt2 shape"),
    "OpenAIGPTLMHeadModel": ("rename-alias", "gpt2-shaped (GPT-1 has learned positions); shape check"),
    "InternLMForCausalLM": ("rename-alias", "internlm2 is in the runtimes; v1 layout differs, needs a check"),
    "YiForCausalLM": ("rename-alias", "llama-shaped"),
    "MinistralForCausalLM": ("rename-alias", "llama/mistral-shaped"),
    "MobileLLMForCausalLM": ("rename-alias", "llama-shaped"),
    "AquilaForCausalLM": ("rename-alias", "llama-shaped with qk-norm; check"),
    "SkyworkForCausalLM": ("rename-alias", "llama-shaped"),
    "ZhinaoForCausalLM": ("rename-alias", "llama-shaped"),
    "HyperLlamaForCausalLM": ("rename-alias", "llama shape under a custom name"),
    "SparseLlamaForCausalLM": ("rename-alias", "llama shape under a custom name"),
    "SparseMistralforCausalLM": ("rename-alias", "llama/mistral shape under a custom name"),
    "ConstrainedLlamaForCausalLM": ("rename-alias", "llama shape under a custom name"),
    "CustomLlamaForCausalLM": ("rename-alias", "llama shape under a custom name"),
    "BitLlamaForCausalLM": ("rename-alias", "llama shape under a custom name"),
    "MobilintLlamaForCausalLM": ("rename-alias", "llama shape under a vendor name"),
    "RECAST8b_LlamaForCausalLM": ("rename-alias", "llama shape under a custom name"),
    "RECAST1B_LlamaForCausalLM": ("rename-alias", "llama shape under a custom name"),
    "ProgressiveYocoLlamaForCausalLM": ("rename-alias", "llama shape under a custom name"),
    "CambrianQwenForCausalLM": ("rename-alias", "qwen2 shape"),
    "Qwen2ReasoningForCausalLM": ("rename-alias", "qwen2 shape"),
    "FP8Qwen3ForCausalLM": ("rename-alias", "qwen3 shape, fp8 checkpoint variant"),
    "PawQwen3ForCausalLM": ("rename-alias", "qwen3 shape"),
    "Qwen3RecoveredForCausalLM": ("rename-alias", "qwen3 shape"),
    "MobilintQwen3ForCausalLM": ("rename-alias", "qwen3 shape under a vendor name"),
    "DogeForCausalLM": ("rename-alias", "llama shape"),
    "NanochatGPTForCausalLM": ("rename-alias", "gpt2-shaped"),
    "NanoChatForCausalLM": ("rename-alias", "gpt2-shaped"),
    "NanoGPTForCausalLM": ("rename-alias", "gpt2-shaped"),
    "GPT2ALMHeadModel": ("rename-alias", "gpt2 shape with an audio head"),
    "PoptorchPipelinedGPT2LMHeadModel": ("rename-alias", "gpt2 shape, pipelined wrapper"),
    "SelfDebiasingGPT2LMHeadModel": ("rename-alias", "gpt2 shape"),
    "MixFormerSequentialForCausalLM": ("rename-alias", "mpt shape (the MosaicML name used by several Phi-1 checkpoints)"),
    "BailingMoeV2_5ForCausalLM": ("rename-alias", "bailingmoe2 shape, v2.5 revision"),
    "BailingMoeLinearV2ForCausalLM": ("rename-alias", "bailingmoe shape, linear-attention variant"),
    "SolarOpen2ForCausalLM": ("rename-alias", "glm4moe shape (Solar-Open is a GLM-4-MoE derivative)"),
    "FlexOlmoForCausalLM": ("rename-alias", "olmo shape"),
    "NCPOlmo3ForCausalLM": ("rename-alias", "olmo2 shape"),
    "Phi3SmallForCausalLM": ("rename-alias", "phi3 shape, small variant"),
    "KPhi3ForCausalLM": ("rename-alias", "phi3 shape under a custom name"),
    "Eagle3DeepseekV2ForCausalLM": ("rename-alias", "deepseek2 shape, Eagle3 draft head"),
    # --- genuine decoders the runtime has no arch for ---
    "Step1MoEForCausalLM": ("new-LM-family", "Step-1 MoE, no runtime arch"),
    "OPTForCausalLM": ("new-LM-family", "OPT; the arch was dropped from the runtime"),
    "GPTNeoForCausalLM": ("new-LM-family", "GPT-Neo (non-X); no runtime arch"),
    "ChessForCausalLM": ("new-LM-family", "chess model; no runtime arch"),
    "CodeGenForCausalLM": ("new-LM-family", "CodeGen; no runtime arch"),
    "PicoDecoderHF": ("new-LM-family", "no runtime arch"),
    "DynamicAlibiForCausalLM": ("new-LM-family", "alibi attention variant; not llama's"),
    "DynamicSlidingWindowForCausalLM": ("new-LM-family", "dynamic sliding-window attention; not llama's"),
    "SlidingWindowForCausalLM": ("new-LM-family", "sliding-window attention variant"),
    "AlibiForCausalLM": ("new-LM-family", "alibi attention variant"),
    "OpenLMForCausalLM": ("new-LM-family", "OpenLM; no runtime arch"),
    "DynamicForgettingForCausalLM": ("new-LM-family", "dynamic-forgetting attention; research arch"),
    "BartForConditionalGeneration": ("new-LM-family", "BART encoder-decoder; no runtime arch"),
    "BartForCausalLM": ("new-LM-family", "BART decoder; no runtime arch"),
    "MBartForConditionalGeneration": ("new-LM-family", "mBART encoder-decoder; no runtime arch"),
    "MBartForCausalLM": ("new-LM-family", "mBART decoder; no runtime arch"),
    "MarianForCausalLM": ("new-LM-family", "Marian MT; no runtime arch"),
    "MarianMTModel": ("new-LM-family", "Marian MT; no runtime arch"),
    "BioGptForCausalLM": ("new-LM-family", "BioGPT; no runtime arch"),
    "RwkvForCausalLM": ("new-LM-family", "RWKV-4; the runtimes have rwkv6/rwkv7, not rwkv"),
    "XGLMForCausalLM": ("new-LM-family", "XGLM; no runtime arch"),
    "XLNetLMHeadModel": ("new-LM-family", "XLNet; no runtime arch"),
    "CTRLLMHeadModel": ("new-LM-family", "CTRL control codes; no runtime arch"),
    "M2M100ForConditionalGeneration": ("new-LM-family", "M2M-100 MT; no runtime arch"),
    "NARBartForConditionalGeneration": ("new-LM-family", "non-autoregressive BART; no runtime arch"),
    "T5GemmaForConditionalGeneration": ("new-LM-family", "T5 encoder + Gemma decoder hybrid"),
    "MorphT5AutoForConditionalGeneration": ("new-LM-family", "T5-derived; encoder-decoder"),
    "MorphT5ConcatForConditionalGeneration": ("new-LM-family", "T5-derived; encoder-decoder"),
    "MorphT5SumForConditionalGeneration": ("new-LM-family", "T5-derived; encoder-decoder"),
    "Glm5NextForConditionalGeneration": ("new-LM-family", "GLM-5-Next; newer than the pinned glm arch"),
    "LLaDA2MoeModelLM": ("new-LM-family", "LLaDA MoE diffusion LM"),
    "HopfieldForCausalLM": ("new-LM-family", "no runtime arch"),
    "NotaGenLMHeadModel": ("new-LM-family", "music LM; no runtime arch"),
    "EncoderDecoderModel": ("new-LM-family", "generic encoder-decoder wrapper"),
    "AutoModelForCausalLM": ("new-LM-family", "config names the generic wrapper, not the arch"),
    "GPTForCausalLM": ("new-LM-family", "config names the generic GPT wrapper"),
    "transformerModel": ("new-LM-family", "generic transformer wrapper"),
    # --- non-text modalities ---
    "ParlerTTSForConditionalGeneration": ("non-LM-modality", "TTS (audio output)"),
    "WhisperForCausalLM": ("non-LM-modality", "speech (Whisper encoder + LM head)"),
    "MoshiForConditionalGeneration": ("non-LM-modality", "full-duplex speech"),
    "Qwen2_5OmniForConditionalGeneration": ("non-LM-modality", "omni (audio + vision + text)"),
    "NemotronLabsDiffusionModel": ("non-LM-modality", "diffusion text generation"),
    "DiffusionGemmaForBlockDiffusion": ("non-LM-modality", "block diffusion"),
    "DiscreteDiffusionModel": ("non-LM-modality", "discrete diffusion"),
    "RecurrentGemmaForCausalLM": ("new-LM-family", "recurrent (Griffin) decoder, not a modality"),
    "SarvamMLAForCausalLM": ("new-LM-family", "MLA attention; no runtime arch"),
    "LlavaLlamaForCausalLM": ("non-LM-modality", "vision-language"),
    "LlavaQwen2ForCausalLM": ("non-LM-modality", "vision-language"),
    "LlavaQwenForCausalLM": ("non-LM-modality", "vision-language"),
    "LlavaMistralForCausalLM": ("non-LM-modality", "vision-language"),
    "LlavaPhiForCausalLM": ("non-LM-modality", "vision-language"),
    "LlavaLlamaModel": ("non-LM-modality", "vision-language"),
    "LlavaNextForConditionalGeneration": ("non-LM-modality", "vision-language"),
    "TinyLlavaForConditionalGeneration": ("non-LM-modality", "vision-language"),
    "NMMaskMoELLaVAQwen3ForCausalLM": ("non-LM-modality", "vision-language"),
    "AdapterMoELLaVAQwen3ForCausalLM": ("non-LM-modality", "vision-language"),
    "MPLUGOwl2LlamaForCausalLM": ("non-LM-modality", "vision-language"),
    "Moondream": ("non-LM-modality", "vision-language"),
    "Phi3VForCausalLM": ("non-LM-modality", "vision-language"),
    "Phi4MMForCausalLM": ("non-LM-modality", "vision + audio"),
    "MllamaForCausalLM": ("non-LM-modality", "vision-language"),
    "Ovis": ("non-LM-modality", "vision-language"),
    "InternVLChatModel": ("non-LM-modality", "vision-language"),
    "MFuyuForCausalLM": ("non-LM-modality", "vision-language"),
    "ReVisionForConditionalGeneration": ("non-LM-modality", "vision-language"),
    "DetikzifyForConditionalGeneration": ("non-LM-modality", "image -> TikZ"),
    "Prot2TextModel": ("non-LM-modality", "protein -> text"),
    "AV2TextForConditionalGeneration": ("non-LM-modality", "audio-visual -> text"),
}

# Non-text modalities by name. Tokens are matched as whole CamelCase words so "Gemma"
# does not trip "MM" and "Sarvam" does not trip "Sam".
MODALITY = re.compile(
    r"(?<![A-Za-z])("
    r"TTS|Speech|Audio|Voice|Whisper|Moshi|Omni|ASR|Vision|VL|LLaVA|Llava|Phi3V|Phi4MM|"
    r"Mllama|Ovis|Moondream|InternVL|CogVLM|Janus|Chameleon|HunyuanVL|Image|Video|OCR|"
    r"Depth|Action|Drive|Molmo|Diffusion|AV2Text|Prot2Text|Grounding|DETR|ReVision|"
    r"Detikzify|MPLUG|Fuyu|Pixel|Bark|MusicGen|Parler|Cosmos|WorldModel|SpeechT5"
    r")(?![a-z])")

SUFFIXES = ("ForCausalLMEagle3", "ForCausalLMV", "ForConditionalGeneration", "ForCausalLM",
            "ForMaskedLM", "ForSequenceClassification", "LMHeadModel", "ForBlockDiffusion",
            "ForPlanning", "LMHead", "Model")


def stem(name):
    for s in SUFFIXES:
        if name.endswith(s) and len(name) > len(s):
            return name[:-len(s)]
    return name


def families():
    """(converter stems -> gguf, runtime ggufs) read from the pins."""
    up, hrx = os.path.join(rb.ROOT, rb.UPSTREAM), os.path.join(rb.ROOT, rb.HRX)
    conv = {}
    conv.update(rb.hf_to_gguf(hrx))
    conv.update(rb.hf_to_gguf(up))
    runtime = rb.runtime_archs(up) | rb.runtime_archs(hrx)
    stems = {}
    for hf, gguf in conv.items():
        stems.setdefault(stem(hf).lower(), gguf)
    return stems, runtime


def classify(arch, stems, runtime, significant):
    if arch in REVIEWED:
        bucket, note = REVIEWED[arch]
        return bucket, note, "reviewed", False
    if arch in significant:
        e = significant[arch]
        mod = MODALITY.search(e.get("family", "") + " " + arch)
        bucket = "non-LM-modality" if mod else "new-LM-family"
        return bucket, f"registry/significant.json: {e.get('family', arch)}", "significant.json", False
    if MODALITY.search(arch):
        return "non-LM-modality", f"name matches modality token {MODALITY.search(arch).group(1)!r}", "rule:modality", True
    s = stem(arch).lower()
    gguf_ci = {g.lower(): g for g in runtime}
    if s in stems:
        g = stems[s]
        b = "bridge-blind-spot" if g.lower() in gguf_ci else "rename-alias"
        return b, f"stem {stem(arch)!r} is registered in the converter -> {g}", f"rule:converter-stem:{g}", False
    if s in gguf_ci:
        g = gguf_ci[s]
        return "bridge-blind-spot", f"runtime arch {g!r} exists", f"rule:runtime-arch:{g}", False
    # longest converter-stem / runtime-arch substring, min 4 chars, as an alias hint
    best = ""
    for cand in list(stems) + list(gguf_ci):
        if len(cand) >= 4 and cand in s and len(cand) > len(best):
            best = cand
    if best:
        g = stems.get(best, gguf_ci.get(best, ""))
        return "rename-alias", f"contains supported family {best!r} -> {g}", f"rule:family-substring:{g}", True
    return "new-LM-family", "no runtime arch and no supported family match", "rule:default", True


def build():
    census = json.load(open(os.path.join(rb.ROOT, "registry/census.json")))
    counts = census["raw"]["counts"]
    registry = json.load(open(os.path.join(rb.ROOT, "registry/architectures.json")))["architectures"]
    significant = json.load(open(os.path.join(rb.ROOT, "registry/significant.json")))["classes"]
    stems, runtime = families()
    mapped = {a for a, e in registry.items() if e.get("backends")}

    entries = []
    for arch, n in counts.items():
        if arch in mapped:
            continue
        bucket, note, method, needs_review = classify(arch, stems, runtime, significant)
        entries.append({"architecture": arch, "models": n, "bucket": bucket,
                        "method": method, "note": note, "needs_review": needs_review})
    entries.sort(key=lambda e: (-e["models"], e["architecture"]))

    per_bucket = {b: {"architectures": 0, "models": 0} for b in BUCKETS}
    for e in entries:
        per_bucket[e["bucket"]]["architectures"] += 1
        per_bucket[e["bucket"]]["models"] += e["models"]
    return {
        "about": "Every HF architecture the census saw that registry/architectures.json does not map, "
                 "with its model count and what closing it takes. Generated by tools/census_worklist.py; "
                 "the high-count bucket is the hand review in that file (REVIEWED), the tail is rules.",
        "census_date": census.get("date"),
        "census_models": census["coverage"]["models"],
        "mapped_pct": census["coverage"]["mapped_pct"],
        "checked_pct": census["coverage"]["checked_pct"],
        "unmapped": {"architectures": len(entries), "models": sum(e["models"] for e in entries)},
        "per_bucket": {b: v | {"share_of_unmapped_models": round(100.0 * v["models"] / max(1, sum(e["models"] for e in entries)), 2)}
                       for b, v in per_bucket.items()},
        "needs_review": sum(e["needs_review"] for e in entries),
        "entries": entries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(rb.ROOT, "registry/unmapped_worklist.json"))
    ap.add_argument("--top", type=int, default=0, help="also print the top N entries")
    a = ap.parse_args()
    data = build()
    with open(a.out, "w") as f:
        json.dump(data, f, indent=1)
        f.write("\n")
    u = data["unmapped"]
    print(f"wrote {a.out}: {u['architectures']} unmapped architectures, {u['models']} models "
          f"({data['needs_review']} need review)")
    for b in BUCKETS:
        v = data["per_bucket"][b]
        print(f"  {b:18s} {v['architectures']:5d} archs  {v['models']:6d} models  {v['share_of_unmapped_models']:5.1f}%")
    for e in data["entries"][:a.top]:
        print(f"  {e['models']:6d}  {e['architecture']:52s} {e['bucket']:18s} {e['method']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
