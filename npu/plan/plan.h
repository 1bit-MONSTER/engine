// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// plan.json: the model's stream plan (engine #799 P1), read instead of probing the filesystem.
//
// Today the forward decides three things by looking at which files happen to exist in --kernels:
// whether gate/up is fused (`full_i8_GU_...` present wins over split G/U), whether the kernel family
// is the `_b8`/`_pb16i` one, and where the M=16 decode twins live (`kernels-m16`, `m16`, then the
// kernel dir itself).  That is the trap the serve switch hit: the same model dir served with a
// different --kernels silently changed which kernels ran.  A plan.json puts those decisions in a file
// generated from the model (tools/npu_plan.py), so the runtime reads a description and a mismatch is
// a named refusal instead of a different path.
//
// The plan is generated, target-independent, and checked in beside the model.  Everything here is a
// *description*: nothing in this file touches a device, and a plan that disagrees with the model dir
// is refused by name rather than repaired.
//
// Schema, version 1 (`"format": "1bit-npu-plan/1"`):
//
//   {
//     "format": "1bit-npu-plan/1",
//     "generated_by": "tools/npu_plan.py 1.0",
//     "model":   {"hidden": 1024, "layers": 28, "arch": "qwen3"},
//     "container": {"layout": "rm"|"b8"|"pb16i"|"t2s", "chunk_kind": "mode0|mode1|mode2|pb16i", "pb16i": false},
//     "fusion":  {"gate_up": "fused"|"split"},
//     "prefill": {"M": 128},
//     "decode":  {"M": 16, "enabled": true, "sealed_elf": "",
//                 "dirs": [{"root": "model", "path": "kernels-m16"},
//                          {"root": "kernels", "path": "m16"},
//                          {"root": "kernels", "path": "."}]},
//     "kernels": {"tag": ""},                                  // "" | "_b8" | "_pb16i" | "_t2s"
//     "attention": {"mode": "causal"|"noncausal_full_attention"},   // optional, default causal
//     "moe":     {"batched_expert_gu": {"enabled": false, "K": 0, "N": 0}},
//     "layers":  [{"index": 0, "role": "attn"|"gdn"|"moe"|"mla",
//                  "qkv": {"K": 1024, "N": 1536, "fused": true},
//                  "gate_up": {"K": 1024, "N": 6144},
//                  "down": {"K": 3072, "N": 1024},
//                  "attn_gate": {"N": 0}}]
//   }
//
// `dirs` name where the decode family is looked for, in order.  Each says which of the two paths the
// serve was already given it hangs off (`model` = the -m directory, `kernels` = --kernels) and a
// relative path under it -- never an absolute one and never "..", because a plan file must not be able
// to point the loader at a directory the command line did not name.
#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::npu::plan {

// The only format this build reads.  A newer/older file is refused by name, never interpreted.
inline constexpr std::string_view kFormat = "1bit-npu-plan/1";

// Which layer math a layer runs.  Everything except `attn` is a recurrent/GDN family; the engine maps
// these onto its own roles.
enum class LayerRole { Attn, Gdn, Moe, Mla };

inline const char* role_name(LayerRole r) {
    switch (r) {
        case LayerRole::Attn: return "attn";
        case LayerRole::Gdn:  return "gdn";
        case LayerRole::Moe:  return "moe";
        case LayerRole::Mla:  return "mla";
    }
    return "?";
}

struct Projection {
    int K = 0, N = 0;
    bool present() const { return K > 0 && N > 0; }
};

struct LayerPlan {
    int index = -1;
    LayerRole role = LayerRole::Attn;
    Projection qkv;        // self-attn q|k|v (fused: `fused` says whether q/k/v are one kernel)
    bool qkv_fused = true;
    Projection attn_gate;  // the output gate (GDN's z / gated attention)
    Projection gate_up;    // fused gate+up when `fused_gu`, else gate only
    Projection down;
};

// A named refusal: the plan loader and the forward both answer with these, so a caller can say what
// is wrong without reading a log.
struct PlanError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Plan {
    std::string format;      // kFormat when loaded
    std::string generated_by;
    // model
    int hidden = 0, layers = 0;
    std::string arch;
    // container
    std::string layout = "rm";     // "rm" | "b8" | "pb16i" | "t2s"
    std::string chunk_kind;        // informational: mode0|mode1|mode2|pb16i
    bool pb16i = false;
    // kernel decisions that used to be filename probes
    bool fused_gu = true;          // "fused" vs "split" gate/up
    std::string tag;               // "" | "_b8" | "_pb16i" | "_t2s" | "_t2s2": the artifact-name suffix
    int prefill_m = 0;             // 0 = not stated (the engine keeps its own default)
    int decode_m = 0;
    bool decode_enabled = true;
    // The attention mask the container's own pack declares (`decision_config.json`'s attention_mode).
    // "causal" is the default and the only mode that can generate; "noncausal_full_attention" is a
    // prompt-encoding mode (every full-attention layer sees the whole prompt, which is what a d3
    // readout pack is trained with), so the forward refuses to generate with it by name.
    // The q4_1 tile ORDER the container's payload is written in (engine#1110).  A container from another
    // converter can be a different order, and a wrong order decodes as *noise with no error at all* --
    // measured: one FLM converter checkout's containers answer a repeated single token (five times the same
    // character) where ours answers " Paris.", and every structural check passes either way (same tensor
    // names, dtypes, byte counts and chunk kinds).  "engine-1" is what tools/npu_plan.py writes and
    // the only order this decoder implements; absent means the same thing, so every container written
    // before this field existed keeps loading, and present-but-unknown is a refusal naming the id --
    // the only point at which a mis-ordered container can be caught before it answers confidently and
    // wrongly.
    std::string tile_order = "engine-1";
    std::string attention_mode = "causal";
    bool noncausal() const { return attention_mode == "noncausal_full_attention"; }
    // The rotary the decoder applies (engine#799, the hybrid half).  `plain` is one position index, and is
    // what every plan written before this field existed means.  `mrope` is the vision-conditioned form: a
    // 3 x N block of position ids whose planes (t, h, w) advance differently across an image, so a text row
    // with all three planes equal is bit-identical to plain RoPE while an image row is not.  `sections` is
    // the pack's own split of the rotary width over the planes (Qwen3-VL: [24, 20, 20]), carried so the block
    // a caller builds can be checked against the pack instead of guessed.
    //
    // A description, like `kv`: a caller that cannot build the planes must refuse `mrope` by name rather than
    // fall back to one index.  The fallback is not a small error -- text after a 448x448 image resumes at
    // position 19, where a token-count advance would give 201 (engine#799), and both predict fluently.
    std::string rope_kind = "plain";   // "plain" | "mrope"
    std::vector<int> rope_sections;    // mrope only: one positive width per plane
    int rope_planes = 1;               // 1 for plain, sections.size() for mrope
    bool mrope() const { return rope_kind == "mrope"; }
    // What the decoder is fed (engine#799).  `tokens` is an id per row, which the decoder looks up itself, and
    // is what every plan meant before this field.  `embeds` is rows x hidden of precomputed embeddings, already
    // merged with any vision features -- AMD's reference splits a VL pack into embedding.onnx + text.onnx for
    // exactly this.  `vision.merge` says who merged: "external" (a companion graph or the caller, with
    // `feature_dim`-wide features) or "none".  Descriptive only, as above: a plan that says `embeds` and a
    // caller that feeds ids is the caller's refusal to make, not a default to interpolate here.
    std::string inputs_kind = "tokens";   // "tokens" | "embeds"
    std::string vision_merge = "none";    // "external" | "none"
    int vision_feature_dim = 0;
    bool takes_embeds() const { return inputs_kind == "embeds"; }
    // The drafter the plan may name (engine#799 §1 `drafter {head path, k, tau, verify_rows, tree}`).  Absent means
    // no drafter, which is what every plan written before this field means.  Descriptive, like `rope` and `kv`: the
    // speculator (P6) reads it, and a plan that names a drafter this engine cannot drive is refused at the point that
    // tries to drive it -- never silently served as plain decode, because a speculator with the wrong tau or k answers
    // with different text for the same prompt and no error at all.
    //
    //   head        the draft head artifact, a BARE FILE NAME under the model dir (the same rule kv.hadamard.signs
    //               follows: a plan must not be able to aim the loader at a file the command line did not name)
    //   k           draft tokens per pass, >= 1
    //   tau         the acceptance gate, 0 < tau <= 1
    //   verify_rows rows the verifier checks per pass, >= 1 (4 in P6's design: one draft plus a leaf fill)
    //   tree        "leaf" (P6's tree) or "none" (a plain chain)
    std::string drafter_head;
    int drafter_k = 0, drafter_verify_rows = 0;
    double drafter_tau = 0.0;
    std::string drafter_tree = "none";   // "leaf" | "none"
    bool has_drafter() const { return !drafter_head.empty(); }
    // When a GDN layer's recurrent state is committed (engine#799 §1 `state {gdn commit}`): "immediate" is every plan
    // written before this field and is what the step path does today; "deferred" is the speculator's contract, where a
    // rejected draft row must not leave its recurrence behind.  A target that cannot defer must refuse it by name:
    // committing early and reporting the accepted tokens gives the right-looking text with a state that no future token
    // agrees with.
    std::string gdn_commit = "immediate";   // "immediate" | "deferred"
    bool gdn_deferred() const { return gdn_commit == "deferred"; }
    // P7 (#806) part A: what the KV cache is stored as.  `f16` is every plan written before this field
    // existed, and stays the default for that reason.  `int8` stores that side as int8 with one fp32
    // scale per (head, token), taken AFTER a per-head Hadamard rotation -- the rotation is what makes one
    // scale affordable, because it collapses the per-channel magnitude spread the quantizer would
    // otherwise have to cover (docs/npu-plan.md).  `key` is the side that pays no un-rotation, since
    // (QH)(KH)^T = QK^T; `value` is a separate choice because an int8 V needs its own inverse transform
    // on the output, so K-only is the cheapest variant to prove the path with.
    //
    // This is a description, not an implementation: a target that cannot honour `int8` must refuse it by
    // name.  Falling back to f16 would answer with the right text and a different cache contract, which
    // is the failure mode the plan layer exists to remove.
    std::string kv_key = "f16", kv_value = "f16";
    int kv_hadamard_block = 0;       // 0 = no rotation
    std::string kv_hadamard_signs;   // optional per-channel sign file, bare name under the model dir
    bool kv_int8() const { return kv_key == "int8" || kv_value == "int8"; }
    // Where the decode family is looked for, in order: which given path it hangs off, and the
    // relative directory under it ("." = that directory itself).
    struct DecodeDir { bool under_model = false; std::string path; };
    std::vector<DecodeDir> decode_dirs;
    std::string sealed_elf;                // pb16i: one sealed ELF, per-role instance names
    struct { bool enabled = false; int K = 0, N = 0; } moe_batched_gu;
    std::vector<LayerPlan> layer_plans;

    // One kernel the plan expects: the role it serves, the geometry it launches at and the artifact
    // that implements it.  `role` is the forward's own role tag ("QKV", "O", "GU", "G", "U", "D",
    // "GB", "UB", the Gemma4 second attention type, the MLA and GDN roles), so a reader can match a
    // design to the code that launches it without a second name table.
    //
    // It is the unique name (q4nx_design_role_name(): "QKV", "QKVLIN", "QA", "GB", ...), NOT the
    // forward's short log tag -- those differ exactly where a model carries two roles that read alike,
    // since a qwen3_5 pack derives an attention QKV and a GDN QKVLIN and both log as "QKV".
    // designs_agree() still compares a multiset of (role, M, K, N, elf) rather than a role-keyed
    // lookup: that is what makes a duplicated or mispaired set a refusal instead of a match, and it
    // keeps the check honest if a future role name is ever reused.
    //
    // Why the plan carries this at all (engine#803 A2): the design set is what a target opens, and
    // while it was derived in the forward, "which ELF does this role run" was answered by
    // q4nx_derive_designs() plus filename probes inside one file.  A plan that names the set makes
    // the kernel identity part of the model's description -- and the engine then refuses a plan whose
    // designs are not the ones the container derives, instead of silently running whatever the probes
    // find.  An absent section is the old behaviour (the probes), so every plan written before this
    // field existed keeps loading.
    struct Design {
        std::string role;
        int M = 0, K = 0, N = 0;
        std::string elf;
    };
    std::vector<Design> designs;

    // True when the plan's `designs` and a derived set name the same kernels: same roles, same M/K/N,
    // same artifact, both directions (a role the plan omits is a kernel the model needs and the plan
    // would not open).  An empty section agrees with anything.  On disagreement `err` names the role
    // and both sides -- never a default, because a plan that names a different ELF than the weights
    // need is exactly the silent divergence this layer exists to remove.
    bool designs_agree(const std::vector<Design>& derived, std::string& err) const;

    // Import-time completeness (engine#799, rule (c)): every op the plan names must have its kernel on
    // disk, in one of `dirs` (the same list the forward would search, first hit wins).  A plan is a
    // description, so it can be complete as JSON and unservable in fact -- that gap is what this closes,
    // and it closes it BEFORE anything is opened, naming the design and the artifact it wanted.
    //
    // It cannot check the tensors: whether a weight carries its quantization is the container's question
    // (npu/q4nx.h refuses a tensor it cannot decode), and this layer only sees the plan.  A plan with no
    // `designs` section passes here -- there is nothing named to be missing, and that is what every plan
    // written before the section means.
    bool check_artifacts(const std::vector<std::string>& dirs, std::string& err) const;

    bool loaded() const { return !format.empty(); }

    // Load and validate.  Throws PlanError naming the field for anything this build cannot run --
    // never a default, because a default here is a different model.
    static Plan load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw PlanError("plan: cannot read " + path);
        nlohmann::json j;
        try {
            f >> j;
        } catch (const std::exception& ex) {
            throw PlanError("plan: " + path + " is not JSON (" + ex.what() + ")");
        }
        if (!j.is_object()) throw PlanError("plan: " + path + " is not a JSON object");
        Plan p;
        const std::string got = j.value("format", "");
        if (got != kFormat)
            throw PlanError("plan: " + path + " format is \"" + got + "\", this engine reads \"" +
                            std::string(kFormat) + "\"");
        p.format = got;
        p.generated_by = j.value("generated_by", "");

        const auto model = obj(j, "model", path);
        p.hidden = positive(model, "hidden", path);
        p.layers = positive(model, "layers", path);
        p.arch = model.value("arch", "");

        if (j.contains("container")) {
            const auto& c = obj(j, "container", path);
            p.layout = c.value("layout", "rm");
            p.chunk_kind = c.value("chunk_kind", "");
            p.pb16i = c.value("pb16i", false);
            if (p.layout != "rm" && p.layout != "b8" && p.layout != "pb16i" && p.layout != "t2s")
                throw PlanError("plan: " + path + " container.layout \"" + p.layout +
                                "\" is not rm, b8, pb16i or t2s");
            // Optional, defaulted on purpose: a container written before this field existed is in our
            // order, so this is not a format bump.  An id we do not implement is refused rather than
            // decoded at a guess (engine#1110).
            p.tile_order = c.value("tile_order", "engine-1");
            if (p.tile_order != "engine-1")
                throw PlanError("plan: " + path + " container.tile_order \"" + p.tile_order +
                                "\" is not \"engine-1\", the only q4_1 tile order this engine decodes: "
                                "re-convert the model with this repo's tools/npu_plan.py rather than serving tiles "
                                "whose order we would have to guess (engine#1110)");
        }
        if (j.contains("fusion")) {
            const auto& fz = obj(j, "fusion", path);
            const std::string gu = fz.value("gate_up", "fused");
            if (gu == "split") p.fused_gu = false;
            else if (gu != "fused")
                throw PlanError("plan: " + path + " fusion.gate_up \"" + gu +
                                "\" is not fused or split");
        }
        if (j.contains("prefill")) p.prefill_m = positive_or_zero(obj(j, "prefill", path), "M", path);
        // Optional and defaulted on purpose: a plan written before this field existed must load and
        // mean exactly what it meant then (causal), so this is not a format bump.
        if (j.contains("attention")) {
            const auto& a = obj(j, "attention", path);
            if (!a.contains("mode") || !a.at("mode").is_string())
                throw PlanError("plan: " + path + " attention has no \"mode\" string");
            p.attention_mode = a.at("mode").get<std::string>();
            if (p.attention_mode != "causal" && p.attention_mode != "noncausal_full_attention")
                throw PlanError("plan: " + path + " attention.mode \"" + p.attention_mode +
                                "\" is not causal or noncausal_full_attention");
        }
        // Rotary and the decoder's input kind (engine#799, the hybrid half).  Both are optional and defaulted
        // on purpose: `plain` + `tokens` is what every plan written before these fields means, so this is not a
        // format bump.  What is NOT defaulted is a half-stated section -- `mrope` without the planes' split, or
        // `embeds` with no feature width, states a contract nothing can check later.
        if (j.contains("rope")) {
            const auto& r = obj(j, "rope", path);
            if (r.contains("kind")) {
                p.rope_kind = typed<std::string>(r, "kind", "rope", path);
                if (p.rope_kind != "plain" && p.rope_kind != "mrope")
                    throw PlanError("plan: " + path + " rope.kind \"" + p.rope_kind +
                                    "\" is not plain or mrope");
            }
            if (r.contains("planes")) {
                p.rope_planes = int_field(r, "planes", "rope", path);
                if (p.rope_planes < 1)
                    throw PlanError("plan: " + path + " rope.planes " + std::to_string(p.rope_planes) +
                                    " must be at least 1");
            }
            if (r.contains("sections")) {
                if (!r.at("sections").is_array())
                    throw PlanError("plan: " + path + " rope.sections must be an array");
                for (const auto& one : r.at("sections")) {
                    if (!one.is_number_integer() || one.get<int>() < 1)
                        throw PlanError("plan: " + path +
                                        " rope.sections holds a value that is not a positive integer (each is a "
                                        "plane's share of the rotary width)");
                    p.rope_sections.push_back(one.get<int>());
                }
            }
            if (p.rope_kind == "mrope") {
                // The split is not optional here: without it the planes have no declared width, and a caller
                // building them from a guess would rotate a different model than the pack was trained as.
                if (p.rope_sections.empty())
                    throw PlanError("plan: " + path +
                                    " rope.kind is mrope but rope.sections is empty (one positive width per "
                                    "plane is what makes the position block checkable against the pack)");
                // The split is the authority on how many planes there are: `planes` is only there to be able
                // to disagree with it out loud.
                if (!r.contains("planes")) p.rope_planes = (int)p.rope_sections.size();
                if (p.rope_planes < 2)
                    throw PlanError("plan: " + path + " rope.kind is mrope but rope.planes is " +
                                    std::to_string(p.rope_planes) + " (mrope has a plane per axis: t, h, w)");
                if ((int)p.rope_sections.size() != p.rope_planes)
                    throw PlanError("plan: " + path + " rope.sections has " +
                                    std::to_string(p.rope_sections.size()) + " widths but rope.planes is " +
                                    std::to_string(p.rope_planes) + " (one width per plane)");
            } else {
                // A plain rope states nothing about planes: a section here would be a contract with no reader
                // and would read as if the planes existed.
                if (!p.rope_sections.empty())
                    throw PlanError("plan: " + path +
                                    " rope.sections is set but rope.kind is plain (sections describe the mrope "
                                    "split; a plain rope has one index)");
                if (p.rope_planes != 1)
                    throw PlanError("plan: " + path + " rope.planes is " + std::to_string(p.rope_planes) +
                                    " but rope.kind is plain (a plain rope has one plane)");
            }
        }
        if (j.contains("inputs")) {
            const auto& in = obj(j, "inputs", path);
            if (in.contains("kind")) {
                p.inputs_kind = typed<std::string>(in, "kind", "inputs", path);
                if (p.inputs_kind != "tokens" && p.inputs_kind != "embeds")
                    throw PlanError("plan: " + path + " inputs.kind \"" + p.inputs_kind +
                                    "\" is not tokens or embeds");
            }
            if (in.contains("vision")) {
                // Vision features arrive inside the embeddings, so a plan that feeds ids has nowhere to put
                // them: refused rather than ignored, because ignoring it would serve the text half of a vision
                // model and answer confidently about an image it never saw.
                if (p.inputs_kind != "embeds")
                    throw PlanError("plan: " + path +
                                    " inputs.vision is set but inputs.kind is tokens (vision features are merged "
                                    "into embeddings, so inputs.kind must be embeds)");
                const auto& v = obj(in, "vision", path);
                p.vision_merge = typed<std::string>(v, "merge", "inputs.vision", path);
                p.vision_feature_dim = int_field(v, "feature_dim", "inputs.vision", path);
                if (p.vision_merge != "external" && p.vision_merge != "none")
                    throw PlanError("plan: " + path + " inputs.vision.merge \"" + p.vision_merge +
                                    "\" is not external or none");
                if (p.vision_feature_dim < 0)
                    throw PlanError("plan: " + path + " inputs.vision.feature_dim must not be negative");
                if (p.vision_merge == "external" && p.vision_feature_dim == 0)
                    throw PlanError("plan: " + path +
                                    " inputs.vision.merge is external but feature_dim is 0 (an external merge "
                                    "has to say how wide the features it merged are)");
                if (p.vision_merge == "none" && p.vision_feature_dim != 0)
                    throw PlanError("plan: " + path + " inputs.vision.feature_dim is " +
                                    std::to_string(p.vision_feature_dim) +
                                    " but inputs.vision.merge is none (no merge, no features)");
            }
        }
        // The drafter (engine#799 §1).  Absent means no drafter; every stated field is checked, because a speculator
        // with a plausible-but-wrong k/tau/rows answers different text for the same prompt with no error at all.
        if (j.contains("drafter")) {
            const auto& dr = obj(j, "drafter", path);
            p.drafter_head = typed<std::string>(dr, "head", "drafter", path);
            if (!p.drafter_head.empty() &&
                (p.drafter_head.find('/') != std::string::npos || p.drafter_head.find("..") != std::string::npos))
                throw PlanError("plan: " + path + " drafter.head \"" + p.drafter_head +
                                "\" must be a bare file name (no directory part)");
            if (dr.contains("k")) {
                p.drafter_k = int_field(dr, "k", "drafter", path);
                if (p.drafter_k < 1)
                    throw PlanError("plan: " + path + " drafter.k " + std::to_string(p.drafter_k) +
                                    " must be at least 1");
            }
            if (dr.contains("verify_rows")) {
                p.drafter_verify_rows = int_field(dr, "verify_rows", "drafter", path);
                if (p.drafter_verify_rows < 1)
                    throw PlanError("plan: " + path + " drafter.verify_rows " +
                                    std::to_string(p.drafter_verify_rows) + " must be at least 1");
            }
            if (dr.contains("tau")) {
                if (!dr.at("tau").is_number())
                    throw PlanError("plan: " + path + " drafter.tau must be a number");
                p.drafter_tau = dr.at("tau").get<double>();
                if (!(p.drafter_tau > 0.0 && p.drafter_tau <= 1.0))
                    throw PlanError("plan: " + path + " drafter.tau " + std::to_string(p.drafter_tau) +
                                    " is not in (0, 1] (it is an acceptance gate, not a temperature)");
            }
            if (dr.contains("tree")) {
                p.drafter_tree = typed<std::string>(dr, "tree", "drafter", path);
                if (p.drafter_tree != "leaf" && p.drafter_tree != "none")
                    throw PlanError("plan: " + path + " drafter.tree \"" + p.drafter_tree +
                                    "\" is not leaf or none");
            }
            // A named drafter with no k/tau/rows is a half-stated contract: the values exist to be checked against what
            // the speculator actually does, so a plan that omits them cannot be honoured either.
            if (p.has_drafter() && (p.drafter_k == 0 || p.drafter_verify_rows == 0 || p.drafter_tau == 0.0))
                throw PlanError("plan: " + path +
                                " names drafter.head but not k, verify_rows and tau (a drafter's numbers are part of "
                                "its contract: the same prompt answers different text at a different tau)");
        }
        // When GDN state is committed (engine#799 §1).  Absent means immediate, which is what the step path does today.
        if (j.contains("state")) {
            const auto& st = obj(j, "state", path);
            if (st.contains("gdn_commit")) {
                p.gdn_commit = typed<std::string>(st, "gdn_commit", "state", path);
                if (p.gdn_commit != "immediate" && p.gdn_commit != "deferred")
                    throw PlanError("plan: " + path + " state.gdn_commit \"" + p.gdn_commit +
                                    "\" is not immediate or deferred");
            }
        }
        // A second graph (a vision encoder, a projector) is deliberately NOT a plan field: one plan is one
        // decoder stream over one token sequence.  A hybrid is two model directories and a merge above them,
        // which is the serving layer's business -- a companion list would put two graphs and their scheduling in
        // a file whose only consumer cannot open or order them.  Refused by name rather than ignored, so a pack
        // that tries says so at load (engine#799).
        if (j.contains("companion"))
            throw PlanError("plan: " + path +
                            " companion is not part of the decoder plan: a plan describes ONE decoder stream, "
                            "and a second graph (vision, projector) is its own model directory that the "
                            "serving layer merges into inputs.embeds (engine#799)");
        // Optional and defaulted on purpose, for the same reason as attention.mode: a plan written
        // before P7 existed describes f16 KV, which is what every plan meant then.
        if (j.contains("kv")) {
            const auto& kv = obj(j, "kv", path);
            // Type before value: a number where a dtype belongs is a refusal that names the field, not a
            // JSON exception escaping the loader.
            for (const char* side : {"key", "value"})
                if (kv.contains(side) && !kv.at(side).is_string())
                    throw PlanError("plan: " + path + " kv." + side + " is not a string");
            p.kv_key = kv.value("key", "f16");
            p.kv_value = kv.value("value", "f16");
            if (p.kv_key != "f16" && p.kv_key != "int8")
                throw PlanError("plan: " + path + " kv.key \"" + p.kv_key + "\" is not f16 or int8");
            if (p.kv_value != "f16" && p.kv_value != "int8")
                throw PlanError("plan: " + path + " kv.value \"" + p.kv_value + "\" is not f16 or int8");
            if (kv.contains("hadamard")) {
                const auto& h = obj(kv, "hadamard", path);
                p.kv_hadamard_block = positive_or_zero(h, "block", path);
                const int b = p.kv_hadamard_block;
                if (b != 0 && (b < 2 || b > 65536 || (b & (b - 1)) != 0))
                    throw PlanError("plan: " + path + " kv.hadamard.block " + std::to_string(b) +
                                    " is not 0 or a power of two in [2, 65536]");
                p.kv_hadamard_signs = h.value("signs", "");
                // A bare file name, never a path: a plan must not be able to point the loader at a file
                // the command line did not name (the same rule decode.dirs follows).
                if (p.kv_hadamard_signs.find('/') != std::string::npos ||
                    p.kv_hadamard_signs.find("..") != std::string::npos)
                    throw PlanError("plan: " + path + " kv.hadamard.signs \"" + p.kv_hadamard_signs +
                                    "\" must be a bare file name (no directory part)");
                if (b == 0 && !p.kv_hadamard_signs.empty())
                    throw PlanError("plan: " + path +
                                    " kv.hadamard.signs names a file but kv.hadamard.block is 0 (no rotation)");
            }
        }
        if (j.contains("kernels")) {
            const auto& k = obj(j, "kernels", path);
            p.tag = k.value("tag", "");
            if (p.tag != "" && p.tag != "_b8" && p.tag != "_pb16i" && p.tag != "_t2s" && p.tag != "_t2s2")
                throw PlanError("plan: " + path + " kernels.tag \"" + p.tag +
                                "\" is not \"\", \"_b8\", \"_pb16i\", \"_t2s\" or \"_t2s2\"");
        }
        if (j.contains("decode")) {
            const auto& d = obj(j, "decode", path);
            p.decode_m = positive_or_zero(d, "M", path);
            p.decode_enabled = d.value("enabled", true);
            p.sealed_elf = d.value("sealed_elf", "");
            if (d.contains("dirs")) {
                if (!d.at("dirs").is_array()) throw PlanError("plan: " + path + " decode.dirs must be an array");
                for (const auto& one : d.at("dirs")) {
                    if (!one.is_object()) throw PlanError("plan: " + path + " decode.dirs holds a non-object");
                    const std::string root = one.value("root", "");
                    if (root != "model" && root != "kernels")
                        throw PlanError("plan: " + path + " decode.dirs root \"" + root +
                                        "\" is not model or kernels");
                    DecodeDir dir;
                    dir.under_model = (root == "model");
                    dir.path = relative_dir(one.value("path", ""), path);
                    p.decode_dirs.push_back(dir);
                }
                if (p.decode_dirs.empty())
                    throw PlanError("plan: " + path + " decode.dirs is empty (the kernel dir is always last)");
            }
        }
        if (j.contains("moe")) {
            const auto& m = obj(j, "moe", path);
            if (m.contains("batched_expert_gu")) {
                const auto& b = obj(m, "batched_expert_gu", path);
                p.moe_batched_gu.enabled = b.value("enabled", false);
                p.moe_batched_gu.K = positive_or_zero(b, "K", path);
                p.moe_batched_gu.N = positive_or_zero(b, "N", path);
                if (p.moe_batched_gu.enabled && (p.moe_batched_gu.K <= 0 || p.moe_batched_gu.N <= 0))
                    throw PlanError("plan: " + path +
                                    " moe.batched_expert_gu is enabled but does not name K and N");
            }
        }
        if (j.contains("layers")) {
            if (!j.at("layers").is_array()) throw PlanError("plan: " + path + " layers must be an array");
            int expect = 0;
            for (const auto& one : j.at("layers")) {
                if (!one.is_object()) throw PlanError("plan: " + path + " layers holds a non-object");
                LayerPlan lp;
                lp.index = one.value("index", expect);
                if (lp.index != expect)
                    throw PlanError("plan: " + path + " layers are out of order at index " +
                                    std::to_string(expect));
                expect++;
                const std::string r = one.value("role", "");
                if (r == "attn") lp.role = LayerRole::Attn;
                else if (r == "gdn") lp.role = LayerRole::Gdn;
                else if (r == "moe") lp.role = LayerRole::Moe;
                else if (r == "mla") lp.role = LayerRole::Mla;
                else
                    throw PlanError("plan: " + path + " layer " + std::to_string(lp.index) + " role \"" +
                                    r + "\" is not attn, gdn, moe or mla");
                lp.qkv = projection(one, "qkv", path);
                lp.qkv_fused = one.contains("qkv") && one.at("qkv").is_object()
                                   ? one.at("qkv").value("fused", true)
                                   : true;
                lp.attn_gate = projection(one, "attn_gate", path);
                lp.gate_up = projection(one, "gate_up", path);
                lp.down = projection(one, "down", path);
                p.layer_plans.push_back(lp);
            }
            if (p.layers != int(p.layer_plans.size()))
                throw PlanError("plan: " + path + " model.layers is " + std::to_string(p.layers) +
                                " but layers[] holds " + std::to_string(p.layer_plans.size()));
        }
        // A pb16i plan whose decode family is the sealed ELF must name it, and a plan that names one
        // must be pb16i: the two fields describe the same decision.
        if (p.pb16i && p.decode_enabled && p.sealed_elf.empty() && p.tag == "_pb16i")
            throw PlanError("plan: " + path +
                            " is the pb16i family but decode.sealed_elf is empty (the sealed ELF is what "
                            "the decode roles run)");
        if (!p.pb16i && !p.sealed_elf.empty())
            throw PlanError("plan: " + path + " names decode.sealed_elf but container.pb16i is false");
        if (p.tag == "_b8" && p.layout != "b8")
            throw PlanError("plan: " + path + " kernels.tag is _b8 but container.layout is " + p.layout);
        if ((p.tag == "_t2s" || p.tag == "_t2s2") && p.layout != "t2s")
            throw PlanError("plan: " + path + " kernels.tag is " + p.tag + " but container.layout is " + p.layout);
        // _pb16i and _t2s are both the integer-exact ternary family (container.pb16i), and a plan
        // that claims one without the other would pick artifacts the images cannot feed.
        const bool ternary_tag = p.tag == "_pb16i" || p.tag == "_t2s" || p.tag == "_t2s2";
        if (ternary_tag != p.pb16i)
            throw PlanError("plan: " + path + " kernels.tag \"" + p.tag + "\" and container.pb16i " +
                            (p.pb16i ? "true" : "false") + " disagree");
        // The per-role kernel set (engine#803 A2).  Optional: absent means the engine keeps deriving it
        // from the container's dims, which is what every plan written before this field means.
        if (j.contains("designs")) {
            const auto& ds = j.at("designs");
            if (!ds.is_array()) throw PlanError("plan: " + path + " designs is not an array");
            for (size_t i = 0; i < ds.size(); i++) {
                const auto& d = ds.at(i);
                const std::string where = "designs[" + std::to_string(i) + "]";
                if (!d.is_object()) throw PlanError("plan: " + path + " " + where + " is not an object");
                Design des;
                // Every field is read through typed()/int_field() so a wrongly typed value ("M": "128")
                // is a refusal that names plans[i].M, like every other field in this loader, instead of
                // nlohmann's type_error escaping to the caller with no position in it (engine#1330 review).
                des.role = typed<std::string>(d, "role", where, path);
                if (des.role.empty()) throw PlanError("plan: " + path + " " + where + ".role is empty");
                // Named with the index, not just the field: a plan can carry a dozen of these and
                // "M must be positive" without a position is a hunt.
                des.M = int_field(d, "M", where, path);
                if (des.M <= 0) throw PlanError("plan: " + path + " " + where + ".M must be positive");
                des.K = int_field(d, "K", where, path);
                if (des.K <= 0) throw PlanError("plan: " + path + " " + where + ".K must be positive");
                des.N = int_field(d, "N", where, path);
                if (des.N <= 0) throw PlanError("plan: " + path + " " + where + ".N must be positive");
                des.elf = typed<std::string>(d, "elf", where, path);
                if (des.elf.empty())
                    throw PlanError("plan: " + path + " " + where + ".elf is empty (the artifact the role runs)");
                // An exact duplicate is a plan mistake, and saying which entry it repeats reads better than
                // the geometry comparison's "elf a.elf, the container derives a.elf" (engine#1330 review).
                for (size_t k = 0; k < p.designs.size(); k++) {
                    const Design& prev = p.designs[k];
                    if (prev.role == des.role && prev.M == des.M && prev.K == des.K && prev.N == des.N &&
                        prev.elf == des.elf)
                        throw PlanError("plan: " + path + " " + where + " repeats designs[" +
                                        std::to_string(k) + "] (" + des.role + " " + std::to_string(des.M) +
                                        "x" + std::to_string(des.K) + "x" + std::to_string(des.N) + " " +
                                        des.elf + ")");
                }
                p.designs.push_back(des);
            }
        }
        return p;
    }

private:
    // A typed read that names the field: nlohmann's own type_error says nothing about which entry of
    // which plan was wrong, and a plan can carry a dozen designs.
    template <typename T>
    static T typed(const nlohmann::json& j, const char* key, const std::string& where,
                   const std::string& path) {
        if (!j.contains(key)) return T{};
        if (!j.at(key).is_string() && !j.at(key).is_number_integer() && !j.at(key).is_number_float() &&
            !j.at(key).is_boolean())
            throw PlanError("plan: " + path + " " + where + "." + key + " has the wrong type");
        try {
            return j.at(key).get<T>();
        } catch (const std::exception&) {
            throw PlanError("plan: " + path + " " + where + "." + key + " has the wrong type");
        }
    }
    static int int_field(const nlohmann::json& j, const char* key, const std::string& where,
                         const std::string& path) {
        if (!j.contains(key)) return 0;
        if (!j.at(key).is_number_integer())
            throw PlanError("plan: " + path + " " + where + "." + key + " must be an integer");
        return j.at(key).get<int>();
    }

    static const nlohmann::json& obj(const nlohmann::json& j, const char* key, const std::string& path) {
        if (!j.contains(key) || !j.at(key).is_object())
            throw PlanError("plan: " + path + " has no \"" + key + "\" object");
        return j.at(key);
    }
    static int positive(const nlohmann::json& j, const char* key, const std::string& path) {
        const int v = positive_or_zero(j, key, path);
        if (v <= 0)
            throw PlanError("plan: " + path + " " + key + " must be a positive integer");
        return v;
    }
    static int positive_or_zero(const nlohmann::json& j, const char* key, const std::string& path) {
        if (!j.contains(key)) return 0;
        const auto& v = j.at(key);
        if (!v.is_number_integer() || v.get<long long>() < 0 || v.get<long long>() > 1 << 20)
            throw PlanError("plan: " + path + " " + key + " is not a sane non-negative integer");
        return int(v.get<long long>());
    }
    static Projection projection(const nlohmann::json& one, const char* key, const std::string& path) {
        Projection out;
        if (!one.contains(key)) return out;
        if (!one.at(key).is_object())
            throw PlanError("plan: " + path + " layer." + key + " must be an object");
        out.K = positive_or_zero(one.at(key), "K", path);
        out.N = positive_or_zero(one.at(key), "N", path);
        if ((out.K == 0) != (out.N == 0))
            throw PlanError("plan: " + path + " layer." + key + " names only one of K and N");
        return out;
    }
    // A decode dir is a path under the model directory: relative, no "..", no absolute.
    static std::string relative_dir(const std::string& in, const std::string& path) {
        if (in.empty()) throw PlanError("plan: " + path + " decode.dirs holds an empty string");
        if (in.front() == '/' || in.find("..") != std::string::npos || in.find('\\') != std::string::npos)
            throw PlanError("plan: " + path + " decode.dirs entry \"" + in +
                            "\" must be relative to the model directory and must not contain \"..\"");
        std::string out;
        for (const char c : in)
            out += (std::isprint(static_cast<unsigned char>(c)) ? c : '?');
        return out;
    }
};

// Out of line for readability: the comparison is the whole point of the section.  `derived` is what
// the container's own geometry produces (q4nx_derive_designs() plus the MoE pair), so this is the
// check that keeps a plan and the artifacts its weights need from drifting apart silently.
inline bool Plan::designs_agree(const std::vector<Design>& derived, std::string& err) const {
    if (designs.empty()) return true;   // no section: the engine derives, as it always did
    // Compared as a MULTISET of (role, M, K, N, elf), not by role alone: the role tags are not unique --
    // a qwen3_5 pack derives both an attention QKV and a GDN QKVLIN design, and both are logged as
    // "QKV" -- so a role-keyed lookup would compare the wrong pair and, worse, report agreement.
    auto same = [](const Design& a, const Design& b) {
        return a.role == b.role && a.M == b.M && a.K == b.K && a.N == b.N && a.elf == b.elf;
    };
    std::vector<bool> used(derived.size(), false);
    for (const auto& mine : designs) {
        bool found = false;
        for (size_t i = 0; i < derived.size(); i++) {
            if (used[i] || !same(mine, derived[i])) continue;
            used[i] = found = true;
            break;
        }
        if (found) continue;
        // Not found: is it a role the container does not derive at all, or the same role at a different
        // geometry / artifact?  The second is the dangerous one, so it says both sides.
        const Design* near = nullptr;
        for (const auto& d : derived)
            if (d.role == mine.role) near = &d;
        if (!near) {
            err = "plan designs: role \"" + mine.role + "\" is not a design this container derives";
            return false;
        }
        if (near->M != mine.M || near->K != mine.K || near->N != mine.N) {
            err = "plan designs(" + mine.role + "): M=" + std::to_string(mine.M) + " K=" +
                  std::to_string(mine.K) + " N=" + std::to_string(mine.N) + ", the container derives M=" +
                  std::to_string(near->M) + " K=" + std::to_string(near->K) + " N=" +
                  std::to_string(near->N);
            return false;
        }
        err = "plan designs(" + mine.role + "): elf \"" + mine.elf + "\", the container derives \"" +
              near->elf + "\"";
        return false;
    }
    // Both directions: a derived design the plan does not name is a kernel the model needs and the plan
    // would not open once the target takes its set from here.
    for (size_t i = 0; i < derived.size(); i++) {
        if (!used[i]) {
            err = "plan designs: the container derives " + derived[i].role + " (" + derived[i].elf +
                  ") but the plan does not name it";
            return false;
        }
    }
    return true;
}

inline bool Plan::check_artifacts(const std::vector<std::string>& dirs, std::string& err) const {
    if (designs.empty()) return true;   // nothing named, nothing to be missing
    if (dirs.empty()) {
        err = "plan designs: there is no kernel directory to look the artifacts up in";
        return false;
    }
    for (size_t i = 0; i < designs.size(); i++) {
        const Design& d = designs[i];
        bool found = false;
        for (const std::string& root : dirs) {
            if (root.empty()) continue;
            std::ifstream f(root + "/" + d.elf, std::ios::binary);
            if (!f) continue;
            f.seekg(0, std::ios::end);
            if (f.tellg() >= 1) { found = true; break; }   // present and non-empty, like the forward's probe
        }
        if (found) continue;
        err = "plan designs[" + std::to_string(i) + "] (" + d.role + "): the artifact \"" + d.elf +
              "\" is in none of " + std::to_string(dirs.size()) + " kernel dir(s) -- this op has no kernel here";
        return false;
    }
    return true;
}

}  // namespace onebit::npu::plan
