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
#include "tmp_root.h"   // engine#2029: scratch goes on disk, not in tmpfs

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
    const fs::path p = ::onebit::test::tmp_root() / ("npu_plan_test_" + name + ".json");
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

// engine#2208: a b8 container may name a sealed int8 decode ELF; an rm container still may not.
void test_b8_sealed() {
    auto b8 = base_plan();
    b8["container"]["layout"] = "b8";
    b8["kernels"]["tag"] = "_b8";
    b8["decode"]["sealed_elf"] = "SEALED-I8.elf";
    const std::string r = refusal(write_plan("b8_sealed", b8));
    check(r.empty(), "a b8 plan naming decode.sealed_elf loads (" + r + ")");
    if (r.empty()) check(Plan::load(write_plan("b8_sealed", b8)).sealed_elf == "SEALED-I8.elf", "b8 decode.sealed_elf is read");
    auto rm = base_plan();
    rm["decode"]["sealed_elf"] = "SEALED-I8.elf";
    check(has(refusal(write_plan("rm_sealed", rm)), "container.layout is not b8"),
          "an rm plan naming decode.sealed_elf is still refused");
}

// engine#1990: prefill.sealed_elf / prefill.sealed_rows, the wide family.
void test_wide_family() {
    auto pb = base_plan();
    pb["container"]["layout"] = "pb16i";
    pb["container"]["chunk_kind"] = "pb16i";
    pb["container"]["pb16i"] = true;
    pb["kernels"]["tag"] = "_pb16i";
    pb["decode"]["sealed_elf"] = "S16.elf";
    const std::string base_refusal = refusal(write_plan("wide_base", pb));
    check(base_refusal.empty(), "the pb16i base plan of the wide arms loads (" + base_refusal + ")");
    const Plan none = Plan::load(write_plan("wide_none", pb));
    check(none.sealed_wide_elf.empty() && none.sealed_wide_rows == 0, "no wide family unless the plan names one");

    auto w = pb;
    w["prefill"]["sealed_elf"] = "S32.elf";
    w["prefill"]["sealed_rows"] = 32;
    const Plan wp = Plan::load(write_plan("wide_ok", w));
    check(wp.sealed_wide_elf == "S32.elf" && wp.sealed_wide_rows == 32 && wp.prefill_m == 128,
          "prefill.sealed_elf and sealed_rows are read, prefill.M is untouched");

    auto norows = pb;
    norows["prefill"]["sealed_elf"] = "S32.elf";
    check(has(refusal(write_plan("wide_norows", norows)), "17..128"), "a wide family without a row count is refused");
    for (int bad : {16, 129}) {
        auto b = w;
        b["prefill"]["sealed_rows"] = bad;
        check(has(refusal(write_plan("wide_rows", b)), "17..128"),
              "sealed_rows " + std::to_string(bad) + " is refused");
    }
    auto b17 = w;
    b17["prefill"]["sealed_rows"] = 17;
    check(refusal(write_plan("wide_17", b17)).empty(), "sealed_rows 17 loads");
    auto orphan = pb;
    orphan["prefill"]["sealed_rows"] = 32;
    check(has(refusal(write_plan("wide_orphan", orphan)), "without prefill.sealed_elf"),
          "sealed_rows without sealed_elf is refused");
    auto nodecode = base_plan();
    nodecode["prefill"]["sealed_elf"] = "S32.elf";
    nodecode["prefill"]["sealed_rows"] = 32;
    check(has(refusal(write_plan("wide_nodecode", nodecode)), "decode.sealed_elf is empty"),
          "a wide family without the sealed decode set is refused");
    for (const char* name : {"sub/S32.elf", "../S32.elf", "/abs/S32.elf"}) {
        auto b = w;
        b["prefill"]["sealed_elf"] = name;
        check(has(refusal(write_plan("wide_path", b)), "bare file name"), std::string("path ") + name + " is refused");
    }
}

// engine#2073: decode.narrow_elf / decode.narrow_rows, the narrow family.
void test_narrow_family() {
    auto pb = base_plan();
    pb["container"]["layout"] = "pb16i";
    pb["container"]["chunk_kind"] = "pb16i";
    pb["container"]["pb16i"] = true;
    pb["kernels"]["tag"] = "_pb16i";
    pb["decode"]["sealed_elf"] = "S16.elf";
    const Plan none = Plan::load(write_plan("narrow_none", pb));
    check(none.sealed_narrow_elf.empty() && none.sealed_narrow_rows == 0, "no narrow family unless the plan names one");
    auto n = pb;
    n["decode"]["narrow_elf"] = "S4.elf";
    n["decode"]["narrow_rows"] = 4;
    const Plan np = Plan::load(write_plan("narrow_ok", n));
    check(np.sealed_narrow_elf == "S4.elf" && np.sealed_narrow_rows == 4 && np.sealed_elf == "S16.elf",
          "decode.narrow_elf and narrow_rows are read beside the sealed set");
    auto norows = pb;
    norows["decode"]["narrow_elf"] = "S4.elf";
    check(has(refusal(write_plan("narrow_norows", norows)), "1..15"), "a narrow family without a row count is refused");
    auto b16 = n;
    b16["decode"]["narrow_rows"] = 16;
    check(has(refusal(write_plan("narrow_16", b16)), "1..15"), "narrow_rows 16 is refused");
    for (int ok : {1, 15}) {
        auto b = n;
        b["decode"]["narrow_rows"] = ok;
        check(refusal(write_plan("narrow_edge", b)).empty(), "narrow_rows " + std::to_string(ok) + " loads");
    }
    auto orphan = pb;
    orphan["decode"]["narrow_rows"] = 4;
    check(has(refusal(write_plan("narrow_orphan", orphan)), "without decode.narrow_elf"),
          "narrow_rows without narrow_elf is refused");
    auto nosealed = base_plan();
    nosealed["decode"]["narrow_elf"] = "S4.elf";
    nosealed["decode"]["narrow_rows"] = 4;
    check(has(refusal(write_plan("narrow_nosealed", nosealed)), "decode.sealed_elf is empty"),
          "a narrow family without the sealed decode set is refused");
    auto path = n;
    path["decode"]["narrow_elf"] = "sub/S4.elf";
    check(has(refusal(write_plan("narrow_path", path)), "bare file name"), "a path as narrow_elf is refused");
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

    // The search list a target is configured with: the plan's entries, each under the directory it names, in the
    // plan's order (engine#1984 review: nothing resolved them, so a plan naming any place but the built-in three
    // lost its decode twins without an error).
    {
        const std::vector<std::string> got = pp.resolved_decode_dirs("/m", "/k");
        check(got.size() == 2 && got[0] == "/m/kernels-m16" && got[1] == "/k/.",
              "decode dirs resolve under the model dir or the kernel dir, in the plan's order");
        auto seq = base_plan();
        seq["decode"]["dirs"] = nlohmann::json::array({nlohmann::json{{"root", "model"}, {"path", "kernels-m16-seq"}}});
        const std::vector<std::string> only = Plan::load(write_plan("seq", seq)).resolved_decode_dirs("/m", "/k");
        check(only.size() == 1 && only[0] == "/m/kernels-m16-seq",
              "a plan naming one directory outside the built-in places resolves to exactly that directory");
        check(Plan().resolved_decode_dirs("/m", "/k").empty(), "a plan that is not loaded resolves to no directories");
    }

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

// ---- import-time completeness: every named op has its kernel on disk (engine#799 rule (c)) ----
//
// The plan can be complete as JSON and unservable in fact. These cases pin that the load-time check says
// so, naming the design and the artifact, before anything is opened.
void test_artifacts() {
    auto with_designs = [] {
        auto p = base_plan();
        p["designs"] = nlohmann::json::array();
        p["designs"].push_back({{"role", "QKV"}, {"M", 128}, {"K", 1024}, {"N", 4096},
                                {"elf", "full_i8_QKV_K1024_N4096.elf"}});
        p["designs"].push_back({{"role", "O"}, {"M", 128}, {"K", 2048}, {"N", 1024},
                                {"elf", "full_i8_O_K2048_N1024.elf"}});
        return p;
    };
    const Plan p = Plan::load(write_plan("artifacts", with_designs()));
    const fs::path root = ::onebit::test::tmp_root() / "npu_plan_test_artifacts";
    const fs::path kdir = root / "kernels";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(kdir, ec);
    auto make_file = [](const std::string& path, size_t bytes) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        for (size_t i = 0; i < bytes; i++) f.put(' ');
    };

    // A plan with no designs has nothing named, so nothing can be missing.
    {
        const Plan plain = Plan::load(write_plan("artifacts_plain", base_plan()));
        std::string why;
        check(plain.check_artifacts({kdir.string()}, why), "a plan with no designs section is complete");
    }
    // Both artifacts present -> complete.
    {
        for (const char* n : {"full_i8_QKV_K1024_N4096.elf", "full_i8_O_K2048_N1024.elf"})
            make_file((kdir / n).string(), 8);
        std::string why;
        check(p.check_artifacts({kdir.string()}, why), "both artifacts on disk is complete: " + why);
    }
    // One missing -> refused, naming the design and the artifact.
    {
        fs::remove(kdir / "full_i8_O_K2048_N1024.elf", ec);
        std::string why;
        check(!p.check_artifacts({kdir.string()}, why), "a missing artifact is refused");
        check(has(why, "designs[1]") && has(why, "O") && has(why, "full_i8_O_K2048_N1024.elf"),
              "the refusal names the design and the artifact: " + why);
    }
    // A second directory is searched too (the forward's own first-hit order).
    const fs::path k2 = root / "kernels-m16";
    {
        fs::create_directories(k2, ec);
        make_file((k2 / "full_i8_O_K2048_N1024.elf").string(), 8);
        std::string why;
        check(p.check_artifacts({kdir.string(), k2.string()}, why),
              "the second directory is searched as well: " + why);
    }
    // An empty file is not a kernel -- same >= 1 byte rule the forward's probe uses.
    {
        std::ofstream((k2 / "full_i8_O_K2048_N1024.elf").string(), std::ios::binary | std::ios::trunc);
        fs::remove(kdir / "full_i8_O_K2048_N1024.elf", ec);
        std::string why;
        check(!p.check_artifacts({kdir.string(), k2.string()}, why), "a zero-byte artifact is refused");
    }
    // No directory at all is a refusal that says so, not a crash.
    {
        std::string why;
        check(!p.check_artifacts({}, why), "no kernel directory is refused");
        check(has(why, "no kernel directory"), "the refusal says there was nowhere to look: " + why);
    }
}

// engine#799, the hybrid half: the rotary kind and the decoder's input kind.  Both are absent-by-default
// (`plain`, `tokens`), and both refuse a half-stated contract by name -- an mrope plan with no split, or a
// vision block on a plan that feeds token ids, is a model that would answer confidently and wrongly.
void test_rope_and_inputs() {
    {
        const Plan p = Plan::load(write_plan("hybrid_absent", base_plan()));
        check(p.rope_kind == "plain" && !p.mrope() && p.rope_planes == 1 && p.rope_sections.empty(),
              "a plan with no rope block means one plain position index");
        check(p.inputs_kind == "tokens" && !p.takes_embeds() && p.vision_merge == "none" && p.vision_feature_dim == 0,
              "and one token per row, with no vision features");
    }
    {
        // Exactly what tools/npu_plan.py writes for a model that declares no mrope: stated, not implied.
        auto j = base_plan();
        j["rope"] = {{"kind", "plain"}, {"planes", 1}};
        const Plan p = Plan::load(write_plan("hybrid_plain_stated", j));
        check(!p.mrope() && p.rope_planes == 1, "a stated plain rope (the generator's own shape) loads");
    }
    {
        // and what it writes for a VL pack, through the same loader.
        auto j = base_plan();
        j["rope"] = {{"kind", "mrope"}, {"planes", 3}, {"sections", {24, 20, 20}}};
        const Plan p = Plan::load(write_plan("hybrid_mrope", j));
        check(p.mrope() && p.rope_planes == 3 && p.rope_sections == std::vector<int>({24, 20, 20}),
              "mrope with its planes and the pack's own split loads as written");
    }
    {
        auto j = base_plan();
        j["rope"] = {{"kind", "mrope"}, {"sections", {16, 16}}};
        const Plan p = Plan::load(write_plan("hybrid_mrope_2plane", j));
        check(p.mrope() && p.rope_planes == 2, "planes defaults to the number of sections stated");
    }
    {
        auto j = base_plan();
        j["inputs"] = {{"kind", "embeds"}, {"vision", {{"merge", "external"}, {"feature_dim", 2048}}}};
        const Plan p = Plan::load(write_plan("hybrid_embeds", j));
        check(p.takes_embeds() && p.vision_merge == "external" && p.vision_feature_dim == 2048,
              "an embeddings plan that names its external merge and feature width loads");
    }
    {
        auto j = base_plan();
        j["rope"] = {{"kind", "mrope"}, {"planes", 3}};
        const std::string why = refusal(write_plan("hybrid_mrope_no_sections", j));
        check(has(why, "rope.sections") && has(why, "mrope"),
              "mrope without the split is refused, naming rope.sections: " + why);
    }
    {
        auto j = base_plan();
        j["rope"] = {{"kind", "mrope"}, {"planes", 3}, {"sections", {24, 20}}};
        const std::string why = refusal(write_plan("hybrid_mrope_arity", j));
        check(has(why, "rope.sections") && has(why, "rope.planes"),
              "one width per plane is enforced, naming both fields: " + why);
    }
    {
        auto j = base_plan();
        j["rope"] = {{"kind", "plain"}, {"sections", {24, 20, 20}}};
        const std::string why = refusal(write_plan("hybrid_plain_sections", j));
        check(has(why, "rope.sections") && has(why, "plain"),
              "a plain rope with a split is refused rather than reading as if the planes existed: " + why);
    }
    {
        auto j = base_plan();
        j["inputs"] = {{"vision", {{"merge", "external"}, {"feature_dim", 2048}}}};
        const std::string why = refusal(write_plan("hybrid_vision_no_embeds", j));
        check(has(why, "inputs.vision") && has(why, "embeds"),
              "vision features on a token-id plan are refused, naming both: " + why);
    }
    {
        auto j = base_plan();
        j["inputs"] = {{"kind", "embeds"}, {"vision", {{"merge", "external"}, {"feature_dim", 0}}}};
        const std::string why = refusal(write_plan("hybrid_external_no_width", j));
        check(has(why, "feature_dim"),
              "an external merge has to state how wide its features are: " + why);
    }
    {
        auto j = base_plan();
        j["companion"] = nlohmann::json::array({nlohmann::json{{"role", "vision"}, {"device", "igpu"}}});
        const std::string why = refusal(write_plan("hybrid_companion", j));
        check(has(why, "companion") && has(why, "ONE decoder stream"),
              "a companion graph is refused by name, with the reason a plan cannot hold one: " + why);
    }
}

// engine#799 §1: the two sections the spec names and the loader could not express -- `drafter {head, k, tau,
// verify_rows, tree}` and `state {gdn_commit}`.  Both are absent-by-default (no drafter, immediate commit), and a
// half-stated contract is refused by name, because a speculator with different numbers answers different text for the
// same prompt and raises nothing.
void test_drafter_and_state() {
    {
        const Plan p = Plan::load(write_plan("drafter_absent", base_plan()));
        check(!p.has_drafter() && !p.gdn_deferred(), "no drafter and immediate GDN commit are the defaults");
    }
    {
        auto j = base_plan();
        j["drafter"] = {{"head", "draft_head.q4nx"}, {"k", 4}, {"tau", 0.6}, {"verify_rows", 4}, {"tree", "leaf"}};
        j["state"] = {{"gdn_commit", "deferred"}};
        const Plan p = Plan::load(write_plan("drafter_full", j));
        check(p.has_drafter() && p.drafter_head == "draft_head.q4nx" && p.drafter_k == 4 &&
                  p.drafter_verify_rows == 4 && p.drafter_tau == 0.6 && p.drafter_tree == "leaf",
              "a fully stated drafter loads as written");
        check(p.gdn_deferred(), "and a deferred GDN commit is reported");
    }
    {
        auto j = base_plan();
        j["drafter"] = {{"head", "draft.q4nx"}, {"k", 2}, {"tau", 1.0}, {"verify_rows", 2}};
        const Plan p = Plan::load(write_plan("drafter_edge_tau", j));
        check(p.drafter_tau == 1.0 && p.drafter_tree == "none", "tau 1.0 is allowed and tree defaults to none");
    }
    {
        auto j = base_plan();
        j["drafter"] = {{"head", "draft.q4nx"}, {"k", 2}, {"tau", 1.5}, {"verify_rows", 2}};
        const std::string why = refusal(write_plan("drafter_tau", j));
        check(has(why, "drafter.tau") && has(why, "(0, 1]"), "a tau outside (0,1] is refused, naming the range: " + why);
    }
    {
        auto j = base_plan();
        j["drafter"] = {{"head", "sub/draft.q4nx"}, {"k", 2}, {"tau", 0.5}, {"verify_rows", 2}};
        const std::string why = refusal(write_plan("drafter_path", j));
        check(has(why, "drafter.head") && has(why, "bare file name"),
              "a drafter head with a directory part is refused (a plan cannot aim the loader): " + why);
    }
    {
        auto j = base_plan();
        j["drafter"] = {{"head", "draft.q4nx"}, {"k", 2}, {"verify_rows", 2}};
        const std::string why = refusal(write_plan("drafter_half", j));
        check(has(why, "drafter.head") && has(why, "tau"),
              "a drafter with no tau is refused as a half-stated contract: " + why);
    }
    {
        auto j = base_plan();
        j["state"] = {{"gdn_commit", "promptly"}};
        const std::string why = refusal(write_plan("state_bad", j));
        check(has(why, "state.gdn_commit"), "an unknown gdn_commit is refused by name: " + why);
    }
}

int main() {
    test_valid();
    test_wide_family();
    test_b8_sealed();
    test_narrow_family();
    test_rope_and_inputs();
    test_drafter_and_state();
    test_kv();
    test_tile_order();
    test_refusals();
    test_designs();
    test_artifacts();
    std::printf("%s\n", failures ? "FAILED" : "npu_plan: ok");
    return failures ? 1 : 0;
}
