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

// npu_plan_test: the plan.json loader (npu/plan/plan.h, engine #799 P1) with no model, no device and
// no kernel directory -- every fixture is written into a temporary directory here.
//
// What this pins down is the part that replaced the filename probes: a plan states the decisions, and
// anything this build cannot honour is a refusal that NAMES the field.  A default here would be a
// different model, which is exactly the failure mode the plan exists to remove.
#include "plan/plan.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace fs = std::filesystem;
using onebit::npu::plan::Plan;
using onebit::npu::plan::PlanError;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

std::string write_plan(const std::string& name, const nlohmann::json& body) {
    const fs::path p = fs::temp_directory_path() / ("npu_plan_test_" + name + ".json");
    std::ofstream f(p);
    f << body.dump(2) << "\n";
    f.close();
    return p.string();
}

// A minimal valid plan: one attn layer, rm family, fused gate/up.
nlohmann::json base_plan() {
    return nlohmann::json{
        {"format", "1bit-npu-plan/1"},
        {"generated_by", "npu_plan_test"},
        {"model", {{"hidden", 1024}, {"layers", 2}, {"arch", "qwen3"}}},
        {"container", {{"layout", "rm"}, {"chunk_kind", "mode0"}, {"pb16i", false}}},
        {"fusion", {{"gate_up", "fused"}}},
        {"prefill", {{"M", 128}}},
        {"decode", {{"M", 16}, {"enabled", true}, {"sealed_elf", ""},
                    {"dirs", nlohmann::json::array({nlohmann::json{{"root", "kernels"}, {"path", "."}}})}}},
        {"kernels", {{"tag", ""}}},
        {"moe", {{"batched_expert_gu", {{"enabled", false}, {"K", 0}, {"N", 0}}}}},
        {"layers", nlohmann::json::array({
            nlohmann::json{{"index", 0}, {"role", "attn"},
                           {"qkv", {{"K", 1024}, {"N", 4096}, {"fused", true}}},
                           {"gate_up", {{"K", 1024}, {"N", 6144}}},
                           {"down", {{"K", 3072}, {"N", 1024}}}},
            nlohmann::json{{"index", 1}, {"role", "gdn"},
                           {"qkv", {{"K", 1024}, {"N", 6144}, {"fused", true}}},
                           {"attn_gate", {{"K", 1024}, {"N", 2048}}},
                           {"gate_up", {{"K", 1024}, {"N", 6144}}},
                           {"down", {{"K", 3072}, {"N", 1024}}}},
        })},
    };
}

// "" when the plan loads, else the refusal text.
std::string refusal(const std::string& path) {
    try {
        Plan::load(path);
        return "";
    } catch (const PlanError& ex) {
        return ex.what();
    }
}

bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

void test_valid() {
    const std::string path = write_plan("valid", base_plan());
    const Plan p = Plan::load(path);
    check(p.loaded() && p.format == "1bit-npu-plan/1", "a valid plan loads and reports its format");
    check(p.hidden == 1024 && p.layers == 2, "model dimensions are read");
    check(p.fused_gu, "fused gate/up is read");
    check(p.layer_plans.size() == 2 && p.layer_plans[1].role == onebit::npu::plan::LayerRole::Gdn,
          "layer roles are read in order");
    check(p.layer_plans[0].qkv.present() && p.layer_plans[0].qkv.N == 4096,
          "a projection's K and N are read");
    check(!p.moe_batched_gu.enabled, "the batched-expert kernel is off unless the plan enables it");
    check(p.decode_dirs.size() == 1 && !p.decode_dirs[0].under_model && p.decode_dirs[0].path == ".",
          "a decode dir records which given path it hangs off");

    // split gate/up is a decision the plan can state, not a probe result
    auto split = base_plan();
    split["fusion"]["gate_up"] = "split";
    const Plan sp = Plan::load(write_plan("split", split));
    check(!sp.fused_gu, "gate_up split is read as split");

    // the pb16i family: tag + sealed ELF, and decode dirs under the model dir
    auto pb = base_plan();
    pb["container"] = {{"layout", "pb16i"}, {"chunk_kind", "pb16i"}, {"pb16i", true}};
    pb["kernels"]["tag"] = "_pb16i";
    pb["decode"]["sealed_elf"] = "pb16i_sealed.elf";
    pb["decode"]["dirs"] = nlohmann::json::array(
        {nlohmann::json{{"root", "model"}, {"path", "kernels-m16"}},
         nlohmann::json{{"root", "kernels"}, {"path", "."}}});
    const Plan pp = Plan::load(write_plan("pb16i", pb));
    check(pp.pb16i && pp.tag == "_pb16i" && pp.sealed_elf == "pb16i_sealed.elf",
          "the pb16i family is read with its sealed ELF");
    check(pp.decode_dirs.size() == 2 && pp.decode_dirs[0].under_model, "a model-relative decode dir is kept");

    // attention.mode is optional and defaults to causal, so a plan written before the field loads and
    // still means what it meant then.
    check(p.attention_mode == "causal" && !p.noncausal(), "a plan with no attention block is causal");
    auto nc = base_plan();
    nc["attention"] = {{"mode", "noncausal_full_attention"}};
    const Plan np = Plan::load(write_plan("noncausal", nc));
    check(np.noncausal() && np.attention_mode == "noncausal_full_attention",
          "a noncausal pack's attention mode is read");
    auto ca = base_plan();
    ca["attention"] = {{"mode", "causal"}};
    check(!Plan::load(write_plan("causal", ca)).noncausal(), "an explicit causal mode is causal");

    // the t2s family: the same integer-exact ternary container, its own artifact tag, and no sealed
    // ELF (its per-role `_t2s` kernels are the decode family, so the field stays empty by design)
    auto t2 = base_plan();
    t2["container"] = {{"layout", "t2s"}, {"chunk_kind", "pb16i"}, {"pb16i", true}};
    t2["kernels"]["tag"] = "_t2s";
    t2["decode"]["sealed_elf"] = "";
    const Plan tp = Plan::load(write_plan("t2s", t2));
    check(tp.pb16i && tp.tag == "_t2s" && tp.layout == "t2s" && tp.sealed_elf.empty(),
          "the t2s family is read as ternary with its own tag and no sealed ELF");

    // t2s wire format v2: the same weights in a different byte order, so it is a different artifact family
    // (_t2s2) and a plan that names one cannot silently drive kernels built for the other
    auto t22 = t2;
    t22["kernels"]["tag"] = "_t2s2";
    const Plan t2p = Plan::load(write_plan("t2s2", t22));
    check(t2p.pb16i && t2p.tag == "_t2s2" && t2p.layout == "t2s",
          "the t2s v2 family is read with its own tag");
    auto bad2 = t2;
    bad2["kernels"]["tag"] = "_t2s2";                      // but container.layout stays "pb16i"
    bad2["container"] = {{"layout", "pb16i"}, {"chunk_kind", "pb16i"}, {"pb16i", true}};
    check(has(refusal(write_plan("t2sv2wronglayout", bad2)), "kernels.tag is _t2s2 but container.layout is pb16i"),
          "a _t2s2 tag on a pb16i container is refused by name");
}

void test_tile_order() {
    // engine#1110: a container declares the q4_1 tile order its payload is in.  Absent means our order
    // (every plan written before the field existed keeps loading); an id this decoder does not implement
    // is refused by name, because a wrong order decodes as noise with no error anywhere else.
    {
        auto p = base_plan();
        const Plan got = Plan::load(write_plan("tileorder_absent", p));
        check(got.tile_order == "engine-1", "a plan without container.tile_order means engine-1");
    }
    {
        auto p = base_plan();
        p["container"]["tile_order"] = "engine-1";
        check(Plan::load(write_plan("tileorder_ours", p)).tile_order == "engine-1",
              "container.tile_order engine-1 loads");
    }
    {
        auto p = base_plan();
        p["container"]["tile_order"] = "flm-2026-10";
        const std::string why = refusal(write_plan("tileorder_unknown", p));
        check(has(why, "container.tile_order \"flm-2026-10\"") && has(why, "engine-1"),
              "an unknown tile order is refused by name: " + why);
        check(has(why, "engine#1110"), "the refusal points at the reason it exists");
    }
}

void test_refusals() {
    // The whole point: nothing is defaulted, every refusal names the field.
    {
        auto p = base_plan();
        p["format"] = "1bit-npu-plan/2";
        const std::string why = refusal(write_plan("badformat", p));
        check(has(why, "format is \"1bit-npu-plan/2\"") && has(why, "1bit-npu-plan/1"),
              "a newer format is refused by name: " + why);
    }
    {
        auto p = base_plan();
        p.erase("model");
        check(has(refusal(write_plan("nomodel", p)), "has no \"model\" object"), "a plan without model is refused");
    }
    {
        auto p = base_plan();
        p["model"]["layers"] = 3;  // layers[] holds 2
        check(has(refusal(write_plan("layercount", p)), "but layers[] holds 2"),
              "a layer count that disagrees with layers[] is refused");
    }
    {
        auto p = base_plan();
        p["layers"][1]["index"] = 5;
        check(has(refusal(write_plan("order", p)), "out of order"), "layers out of order are refused");
    }
    {
        auto p = base_plan();
        p["layers"][0]["role"] = "conv";
        check(has(refusal(write_plan("role", p)), "role \"conv\" is not attn, gdn, moe or mla"),
              "an unknown layer role is refused by name");
    }
    {
        // The three probes the plan replaces cannot be silently re-decided:
        auto p = base_plan();
        p["kernels"]["tag"] = "_b8";  // while container.layout stays "rm"
        check(has(refusal(write_plan("b8tag", p)), "kernels.tag is _b8 but container.layout is rm"),
              "a _b8 kernel tag on a row-major container is refused");
    }
    {
        auto p = base_plan();
        p["kernels"]["tag"] = "_t2s";  // while container.layout stays "rm"
        check(has(refusal(write_plan("t2stag", p)), "kernels.tag is _t2s but container.layout is rm"),
              "a _t2s kernel tag on a row-major container is refused");
    }
    {
        auto p = base_plan();
        p["container"] = {{"layout", "t2s"}, {"chunk_kind", "pb16i"}, {"pb16i", false}};
        p["kernels"]["tag"] = "_t2s";
        check(has(refusal(write_plan("t2snopb16i", p)), "and container.pb16i false disagree"),
              "a t2s tag on a container that does not claim the ternary family is refused");
    }
    {
        auto p = base_plan();
        p["decode"]["sealed_elf"] = "sealed.elf";  // but pb16i is false
        check(has(refusal(write_plan("sealedrm", p)), "names decode.sealed_elf but container.pb16i is false"),
              "a sealed ELF named outside the pb16i family is refused");
    }
    {
        auto p = base_plan();
        p["container"] = {{"layout", "pb16i"}, {"chunk_kind", "pb16i"}, {"pb16i", true}};
        p["kernels"]["tag"] = "_pb16i";
        p["decode"]["sealed_elf"] = "";  // pb16i with no sealed ELF
        check(has(refusal(write_plan("nosealed", p)), "decode.sealed_elf is empty"),
              "the pb16i family without its sealed ELF is refused");
    }
    {
        auto p = base_plan();
        p["decode"]["dirs"] = nlohmann::json::array({nlohmann::json{{"root", "kernels"}, {"path", "../escape"}}});
        check(has(refusal(write_plan("dotdot", p)), "must be relative"),
              "a decode dir that escapes the given path is refused");
        auto q = base_plan();
        q["decode"]["dirs"] = nlohmann::json::array({nlohmann::json{{"root", "kernels"}, {"path", "/etc"}}});
        check(has(refusal(write_plan("absolute", q)), "must be relative"),
              "an absolute decode dir is refused");
        auto r = base_plan();
        r["decode"]["dirs"] = nlohmann::json::array({nlohmann::json{{"root", "elsewhere"}, {"path", "."}}});
        check(has(refusal(write_plan("root", r)), "root \"elsewhere\" is not model or kernels"),
              "a decode dir with an unknown root is refused");
    }
    {
        auto p = base_plan();
        p["moe"]["batched_expert_gu"]["enabled"] = true;  // but no K/N
        check(has(refusal(write_plan("moe", p)), "does not name K and N"),
              "an enabled batched-expert kernel without its geometry is refused");
    }
    {
        auto p = base_plan();
        p["layers"][0]["qkv"] = {{"K", 1024}};  // N missing
        check(has(refusal(write_plan("halfproj", p)), "names only one of K and N"),
              "half a projection is refused rather than defaulted");
    }
    {
        // The two modes give different numbers, so a third string is never guessed at.
        auto p = base_plan();
        p["attention"] = {{"mode", "bidirectional"}};
        check(has(refusal(write_plan("attnmode", p)),
                  "attention.mode \"bidirectional\" is not causal or noncausal_full_attention"),
              "an unknown attention mode is refused by name");
        auto q = base_plan();
        q["attention"] = {{"mode", 1}};
        check(has(refusal(write_plan("attntype", q)), "attention has no \"mode\" string"),
              "a non-string attention mode is refused");
    }
    check(has(refusal("/nonexistent/plan.json"), "cannot read"), "a missing plan file is refused by path");
}

}  // namespace


// P7 (#806) part A: the KV contract.  Absent means f16 with no rotation -- which is what every plan written
// before this field meant -- and anything this build cannot honour is a refusal that names the field, so a
// plan asking for an int8 cache on a target without one cannot be served as f16 by accident.
void test_kv() {
    {
        const Plan p = Plan::load(write_plan("kv_absent", base_plan()));
        check(p.kv_key == "f16" && p.kv_value == "f16" && p.kv_hadamard_block == 0,
              "a plan with no kv block means f16 KV with no rotation");
        check(!p.kv_int8(), "and reports itself as not int8");
    }
    {
        auto j = base_plan();
        j["kv"] = {{"key", "int8"}, {"value", "f16"}, {"hadamard", {{"block", 256}, {"signs", "had.signs"}}}};
        const Plan p = Plan::load(write_plan("kv_variant4", j));
        check(p.kv_key == "int8" && p.kv_value == "f16", "variant 4 (int8 K, f16 V) loads as written");
        check(p.kv_int8(), "and reports itself as int8");
        check(p.kv_hadamard_block == 256 && p.kv_hadamard_signs == "had.signs",
              "the per-head rotation block and its sign file are read");
    }
    {
        auto j = base_plan();
        j["kv"] = {{"key", "f16"}, {"value", "f16"}, {"hadamard", {{"block", 0}}}};
        const Plan p = Plan::load(write_plan("kv_plain", j));
        check(!p.kv_int8() && p.kv_hadamard_block == 0, "an explicit f16/no-rotation block loads");
    }
    {
        auto j = base_plan();
        j["kv"] = {{"key", "int4"}};
        const std::string why = refusal(write_plan("kv_bad_dtype", j));
        check(has(why, "kv.key") && has(why, "int4"), "an unknown kv.key is refused, naming the field and value");
    }
    {
        auto j = base_plan();
        j["kv"] = {{"key", 8}};
        const std::string why = refusal(write_plan("kv_key_not_string", j));
        check(has(why, "kv.key") && has(why, "not a string"),
              "a non-string kv.key is a named refusal, not a JSON exception escaping the loader");
    }
    {
        auto j = base_plan();
        j["kv"] = {{"key", "int8"}, {"value", "q8_0"}};
        const std::string why = refusal(write_plan("kv_bad_value", j));
        check(has(why, "kv.value"), "an unknown kv.value is refused, naming the field");
    }
    {
        auto j = base_plan();
        j["kv"] = {{"key", "int8"}, {"hadamard", {{"block", 250}}}};
        const std::string why = refusal(write_plan("kv_bad_block", j));
        check(has(why, "kv.hadamard.block"), "a rotation block that is not a power of two is refused");
    }
    {
        auto j = base_plan();
        j["kv"] = {{"key", "int8"}, {"hadamard", {{"block", 256}, {"signs", "../elsewhere.signs"}}}};
        const std::string why = refusal(write_plan("kv_bad_signs", j));
        check(has(why, "kv.hadamard.signs"),
              "a sign file that is a path rather than a bare name is refused (a plan cannot point the "
              "loader outside the dirs the command line named)");
    }
    {
        auto j = base_plan();
        j["kv"] = {{"key", "int8"}, {"hadamard", {{"block", 0}, {"signs", "had.signs"}}}};
        const std::string why = refusal(write_plan("kv_signs_no_block", j));
        check(has(why, "kv.hadamard.signs"),
              "a sign file with no rotation block is refused as contradictory rather than ignored");
    }
}

// ---- designs: the per-role kernel set the plan may name (engine#803 A2) ----
//
// Two halves, both device-free: the loader accepts and rejects the section by field, and
// designs_agree() is the comparison the forward runs against what the container derives.
void test_designs() {
    auto with_designs = [](bool second) {
        auto p = base_plan();
        p["designs"] = nlohmann::json::array();
        p["designs"].push_back({{"role", "QKV"}, {"M", 128}, {"K", 1024}, {"N", 2048},
                                {"elf", "full_i8_QKV_K1024_N2048.elf"}});
        if (second)
            p["designs"].push_back({{"role", "O"}, {"M", 128}, {"K", 2048}, {"N", 1024},
                                    {"elf", "full_i8_O_K2048_N1024.elf"}});
        return p;
    };
    const Plan p = Plan::load(write_plan("designs2", with_designs(true)));
    check(p.designs.size() == 2, "designs: two entries parsed");
    check(p.designs[0].role == "QKV" && p.designs[0].M == 128 && p.designs[0].K == 1024 &&
              p.designs[0].N == 2048 && p.designs[0].elf == "full_i8_QKV_K1024_N2048.elf",
          "designs: role/M/K/N/elf are what the plan says");

    // A fresh derived set per case, so a case cannot inherit an earlier mutation.
    auto derived = [] {
        return std::vector<Plan::Design>{
            {"QKV", 128, 1024, 2048, "full_i8_QKV_K1024_N2048.elf"},
            {"O", 128, 2048, 1024, "full_i8_O_K2048_N1024.elf"},
        };
    };
    std::string why;
    check(p.designs_agree(derived(), why), "designs: the same set agrees");
    {
        auto swapped = derived();
        std::swap(swapped[0], swapped[1]);
        check(p.designs_agree(swapped, why), "designs: the comparison is by role, not by position");
    }
    // The refusal that matters: same shape, wrong artifact -- the ELF the weights were not written for.
    {
        auto d = derived();
        d[1].elf = "full_i8_O_K2048_N1024_b8.elf";
        why.clear();
        check(!p.designs_agree(d, why), "designs: a different artifact is refused");
        check(has(why, "O") && has(why, "_b8"),
              "designs: the artifact refusal names the role and the file: " + why);
    }
    {
        auto d = derived();
        d[1].N = 512;
        why.clear();
        check(!p.designs_agree(d, why), "designs: a different geometry is refused");
        check(has(why, "N=512") && has(why, "N=1024"),
              "designs: the geometry refusal carries both sides: " + why);
    }
    // Both directions: a role the container derives and the plan omits is a kernel the plan would not open.
    {
        auto d = derived();
        d.push_back({"D", 128, 3072, 1024, "full_i8_D_K3072_N1024.elf"});
        why.clear();
        check(!p.designs_agree(d, why), "designs: a derived role the plan omits is refused");
        check(has(why, "D"), "designs: the omission names the role: " + why);
    }

    // The role tags are NOT unique: a qwen3_5 pack derives an attention QKV and a GDN QKVLIN design,
    // and both are logged as "QKV".  A role-keyed comparison would compare the wrong pair, so the
    // agreement is a multiset of (role, M, K, N, elf) and this case is the one that proves it.
    {
        auto two_qkv = with_designs(true);
        two_qkv["designs"] = nlohmann::json::array({
            {{"role", "QKV"}, {"M", 128}, {"K", 1024}, {"N", 4096}, {"elf", "full_i8_QKV_K1024_N4096.elf"}},
            {{"role", "QKV"}, {"M", 128}, {"K", 1024}, {"N", 2048}, {"elf", "full_i8_QKVLIN_K1024_N2048.elf"}},
        });
        const Plan q = Plan::load(write_plan("twosame", two_qkv));
        std::string w;
        check(q.designs_agree({{"QKV", 128, 1024, 4096, "full_i8_QKV_K1024_N4096.elf"},
                               {"QKV", 128, 1024, 2048, "full_i8_QKVLIN_K1024_N2048.elf"}},
                              w),
              "designs: two roles that share a tag are matched as a multiset");
        // Swapping the geometries must be refused: same role, wrong one-to-one pairing.
        w.clear();
        check(!q.designs_agree({{"QKV", 128, 1024, 2048, "full_i8_QKVLIN_K1024_N2048.elf"},
                                {"QKV", 128, 1024, 4096, "full_i8_QKV_K1024_N4096.elf"},
                                {"QKV", 128, 1024, 4096, "full_i8_QKV_K1024_N4096.elf"}},
                               w),
              "designs: a duplicated geometry cannot stand in for an omitted one");
    }
    // A wrong TYPE is a refusal naming the field, not nlohmann's type_error with no position in it.
    {
        auto bad = with_designs(true);
        bad["designs"][0]["M"] = "128";
        const std::string why = refusal(write_plan("dtype", bad));
        check(has(why, "designs[0].M") && has(why, "must be an integer"),
              "designs: a wrongly typed M is refused by field: " + why);
    }
    // ... and an exact duplicate says which entry it repeats, rather than naming the same elf on both sides.
    {
        auto dup = with_designs(true);
        dup["designs"].push_back(dup["designs"][0]);
        const std::string why = refusal(write_plan("ddup2", dup));
        check(has(why, "designs[2] repeats designs[0]"),
              "designs: an exact duplicate names the entry it repeats: " + why);
    }
    // No section: the probes' behaviour, and every plan written before the field keeps loading.
    const Plan plain = Plan::load(write_plan("nodesigns", base_plan()));
    check(plain.designs.empty(), "designs: an absent section parses to empty");
    why.clear();
    check(plain.designs_agree({{"QKV", 128, 1024, 2048, "x.elf"}}, why),
          "designs: an absent section agrees with anything");

    // Refusals, each naming the field.
    {
        auto bad = with_designs(false);
        bad["designs"][0]["elf"] = "";
        check(has(refusal(write_plan("dnoelf", bad)), ".elf is empty"), "designs: an empty elf is refused");
    }
    {
        auto bad = with_designs(false);
        bad["designs"][0]["M"] = 0;
        check(has(refusal(write_plan("dnom", bad)), "designs[0]"), "designs: a non-positive M is refused by index");
    }
    {
        auto bad = with_designs(false);
        bad["designs"] = 7;
        check(has(refusal(write_plan("dnotarray", bad)), "designs is not an array"),
              "designs: a non-array section is refused");
    }
}

int main() {
    test_valid();
    test_kv();
    test_tile_order();
    test_refusals();
    test_designs();
    std::printf("%s\n", failures ? "FAILED" : "npu_plan: ok");
    return failures ? 1 : 0;
}
