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
//     "container": {"layout": "rm"|"b8", "chunk_kind": "mode0|mode1|mode2|pb16i", "pb16i": false},
//     "fusion":  {"gate_up": "fused"|"split"},
//     "prefill": {"M": 128},
//     "decode":  {"M": 16, "enabled": true, "sealed_elf": "",
//                 "dirs": [{"root": "model", "path": "kernels-m16"},
//                          {"root": "kernels", "path": "m16"},
//                          {"root": "kernels", "path": "."}]},
//     "kernels": {"tag": ""},                                  // "" | "_b8" | "_pb16i"
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
    std::string layout = "rm";     // "rm" | "b8"
    std::string chunk_kind;        // informational: mode0|mode1|mode2|pb16i
    bool pb16i = false;
    // kernel decisions that used to be filename probes
    bool fused_gu = true;          // "fused" vs "split" gate/up
    std::string tag;               // "" | "_b8" | "_pb16i": the artifact-name suffix
    int prefill_m = 0;             // 0 = not stated (the engine keeps its own default)
    int decode_m = 0;
    bool decode_enabled = true;
    // Where the decode family is looked for, in order: which given path it hangs off, and the
    // relative directory under it ("." = that directory itself).
    struct DecodeDir { bool under_model = false; std::string path; };
    std::vector<DecodeDir> decode_dirs;
    std::string sealed_elf;                // pb16i: one sealed ELF, per-role instance names
    struct { bool enabled = false; int K = 0, N = 0; } moe_batched_gu;
    std::vector<LayerPlan> layer_plans;

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
            if (p.layout != "rm" && p.layout != "b8" && p.layout != "pb16i")
                throw PlanError("plan: " + path + " container.layout \"" + p.layout +
                                "\" is not rm, b8 or pb16i");
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
        if (j.contains("kernels")) {
            const auto& k = obj(j, "kernels", path);
            p.tag = k.value("tag", "");
            if (p.tag != "" && p.tag != "_b8" && p.tag != "_pb16i")
                throw PlanError("plan: " + path + " kernels.tag \"" + p.tag +
                                "\" is not \"\", \"_b8\" or \"_pb16i\"");
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
        if ((p.tag == "_pb16i") != p.pb16i)
            throw PlanError("plan: " + path + " kernels.tag \"" + p.tag + "\" and container.pb16i " +
                            (p.pb16i ? "true" : "false") + " disagree");
        return p;
    }

private:
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

}  // namespace onebit::npu::plan
