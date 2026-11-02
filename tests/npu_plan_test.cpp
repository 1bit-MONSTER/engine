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

int main() {
    test_valid();
    test_refusals();
    std::printf("%s\n", failures ? "FAILED" : "npu_plan: ok");
    return failures ? 1 : 0;
}
