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

// npu_target_caps_test: the rule that a plan asking for a cache its target cannot store is refused BY NAME
// (npu/target/target.h, check_kv_caps) -- engine#806 P7a's promise, enforced instead of documented.
//
// Why this is its own test rather than an arm of a device run: the refusal has to happen before any kernel
// or ELF is touched, and it has to be the same answer on every target, including the ones that do not exist
// yet (P5's Hexagon).  A rule only a device can exercise is a rule that regresses silently, so this runs on
// any host with no model and no NPU.
//
// The two ways it can be wrong are both checked here: refusing too little (an int8 plan loads and is served
// an f16 cache -- a different model answering the same text, which is the trap the plan layer exists to
// remove) and refusing too much (a shipped f16 plan fails to load, which is a worse regression than the bug
// the check prevents).
#include "plan/plan.h"
#include "target/target.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace fs = std::filesystem;
using onebit::npu::plan::Plan;
using namespace npu::target;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

// The smallest plan that loads, so the kv block is the only variable.
nlohmann::json base_plan() {
    return nlohmann::json{
        {"format", "1bit-npu-plan/1"},
        {"model", {{"hidden", 1024}, {"layers", 1}, {"arch", "qwen3"}}},
        {"container", {{"layout", "rm"}, {"chunk_kind", "mode0"}, {"pb16i", false}}},
        {"fusion", {{"gate_up", "fused"}}},
        {"kernels", {{"tag", ""}}},
        {"layers", nlohmann::json::array({nlohmann::json{
                       {"index", 0}, {"role", "attn"},
                       {"qkv", {{"K", 1024}, {"N", 4096}, {"fused", true}}},
                       {"gate_up", {{"K", 1024}, {"N", 6144}}},
                       {"down", {{"K", 3072}, {"N", 1024}}}}})},
    };
}

std::string write_plan(const std::string& name, const nlohmann::json& body) {
    const fs::path p = fs::temp_directory_path() / ("npu_target_caps_test_" + name + ".json");
    std::ofstream f(p);
    f << body.dump(2) << "\n";
    f.close();
    return p.string();
}

// What today's targets report: the xdna target sets this false (P4), and nothing in the tree can store an
// int8 cache yet, which is precisely why every int8 plan must be refused today.
Caps no_int8_kv() {
    Caps c;
    c.max_rows = 16;
    c.prefill_rows = 128;
    c.t2_expansion = false;
    c.int8_kv = false;
    return c;
}

void test_refuses_int8_without_caps() {
    auto j = base_plan();
    j["kv"] = {{"key", "int8"}, {"value", "f16"}, {"hadamard", {{"block", 256}}}};
    const Plan p = Plan::load(write_plan("i8k", j));
    check(p.kv_int8(), "the plan reports itself as int8 before the target is asked");

    std::string err;
    const bool ok = check_kv_caps(p.kv_key, p.kv_value, no_int8_kv(), err);
    check(!ok, "an int8 K plan is refused by a target with no int8 KV path");
    check(has(err, "int8") && has(err, "int8_kv"),
          "the refusal names what was asked for and what is missing: " + err);
    check(has(err, "kv.key") && has(err, "kv.value"),
          "the refusal names the plan fields, so a user can find the line to edit");
    check(has(err, "refusing") && has(err, "f16"),
          "the refusal says it is refusing rather than serving f16 (no silent downgrade)");
    // Printed, not just asserted: what a user actually reads is part of the contract, and a wording change that
    // keeps every substring above intact should still be visible to whoever runs this by hand.
    std::printf("  a refusal reads: %s\n", err.c_str());
}

void test_refuses_int8_value_without_caps() {
    // K-only is the cheap variant; V is a separate choice, and asking for V int8 must be caught the same way
    // rather than only being spotted when the writer lands.
    std::string err;
    const bool ok = check_kv_caps("f16", "int8", no_int8_kv(), err);
    check(!ok && has(err, "kv.value"),
          "an int8 V plan is refused by a target with no int8 KV path");
}

void test_accepts_when_caps_allow() {
    Caps c = no_int8_kv();
    c.int8_kv = true;
    for (const auto& kinds : {std::pair<std::string, std::string>{"int8", "f16"},
                              std::pair<std::string, std::string>{"f16", "int8"},
                              std::pair<std::string, std::string>{"int8", "int8"}}) {
        std::string err;
        check(check_kv_caps(kinds.first, kinds.second, c, err),
              "a target that reports int8 KV accepts " + kinds.first + "/" + kinds.second);
        check(err.empty(), "and sets no error text when it accepts");
    }
}

void test_does_not_refuse_shipped_plans() {
    // The regression this guards: every container shipped so far is f16 KV, and refusing one of those would
    // be worse than the bug the check prevents.  Built from a real generator-shaped plan, not a stub.
    const Plan p = Plan::load(write_plan("f16", base_plan()));
    std::string err;
    check(check_kv_caps(p.kv_key, p.kv_value, no_int8_kv(), err),
          "a plan with no kv block (f16, what every shipped container means) is accepted, with no device");
    check(err.empty(), "and sets no error text");
}

}  // namespace

int main() {
    test_refuses_int8_without_caps();
    test_refuses_int8_value_without_caps();
    test_accepts_when_caps_allow();
    test_does_not_refuse_shipped_plans();
    std::printf("%s\n", failures ? "FAILED" : "npu_target_caps: ok");
    return failures ? 1 : 0;
}
