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

// The model table of `1bit serve --laya-models` (laya/models.h): parsing, the Laya question it
// builds, the pick with its fallback, aliases and shared backends, and the refusals by name.
// Needs no checkpoint: argv[1] is the shipped config/laya-models-game.json, checked as well.
#include "models.h"

#include <cstdio>
#include <string>

namespace {
int failures = 0;
void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) failures++;
}

const char* kTable = R"({
 "question": {"instructions": "Which?", "classes": {"reflex": "quick", "plan": "slow", "chat": "talk"}},
 "default": "plan", "min_confidence": 0.2,
 "size_gate": {"chars": 100, "class": "plan"},
 "classes": {
  "reflex": {"model": "small.gguf", "device": "npu", "alias": "small"},
  "plan": {"model": "/abs/big.gguf", "device": "hrx", "args": ["--ctx-size", "4096"]},
  "chat": {"model": "/abs/big.gguf", "device": "hrx", "args": ["--ctx-size", "4096"]}
 }})";

bool refuses(const std::string& text, const std::string& needle) {
    onebit::laya::ModelTable t;
    std::string err;
    const bool ok = t.parse(text, "/base", err);
    return !ok && err.find(needle) != std::string::npos;
}
}  // namespace

int main(int argc, char** argv) {
    using namespace onebit::laya;
    ModelTable t;
    std::string err;
    check("parses (" + err + ")", t.parse(kTable, "/base", err));
    check("question keeps the file's class order", t.classes.size() == 3 && t.classes[0].first == "reflex" &&
                                                       t.classes[1].first == "plan" && t.classes[2].first == "chat");
    const Question q = t.laya_question();
    check("question is one choice with every class", q.type == "choice" && q.instructions == "Which?" && q.criteria.size() == 3);
    check("relative model resolves against the table dir", t.pick("reflex", 0.9f).model == "/base/small.gguf");
    check("alias defaults to the file name", t.pick("plan", 0.9f).alias == "big.gguf");
    std::string used;
    check("confident reflex -> npu", t.pick("reflex", 0.5f, &used).device == "npu" && used == "reflex");
    check("unsure reflex -> fallback plan", t.pick("reflex", 0.1f, &used).device == "hrx" && used == "plan");
    check("empty class -> fallback", t.pick("", 0.9f, &used).device == "hrx" && used == "plan");
    check("by_alias finds a row", t.by_alias("small") && t.by_alias("small")->device == "npu");
    check("by_alias misses an unknown name", t.by_alias("auto") == nullptr);
    check("two classes share one backend", t.backends().size() == 2 &&
                                               t.backend_index(t.pick("plan", 1.f)) == t.backend_index(t.pick("chat", 1.f)));
    check("row_for is the class's own row", t.row_for("reflex") && t.row_for("reflex")->device == "npu" && !t.row_for("x"));
    check("size_gated by length", t.size_gated(std::string(100, 'a')) && !t.size_gated(std::string(99, 'a')));
    check("size gate read", t.size_gate_chars == 100 && t.size_gate_class == "plan");

    check("refuses one class", refuses(R"({"question":{"instructions":"x","classes":{"a":"d"}},"classes":{"a":{"model":"m","device":"hrx"}}})",
                                       "at least two"));
    check("refuses a class without a row",
          refuses(R"({"question":{"instructions":"x","classes":{"a":"d","b":"e"}},"classes":{"a":{"model":"m","device":"hrx"}}})",
                  "\"b\" has no row"));
    check("refuses an unknown device",
          refuses(R"({"question":{"instructions":"x","classes":{"a":"d","b":"e"}},"classes":{"a":{"model":"m","device":"vulkan"},"b":{"model":"m","device":"hrx"}}})",
                  "unknown device"));
    check("refuses a row for no class",
          refuses(R"({"question":{"instructions":"x","classes":{"a":"d","b":"e"}},"classes":{"a":{"model":"m","device":"hrx"},"b":{"model":"m","device":"hrx"},"c":{"model":"m","device":"hrx"}}})",
                  "not one of the question's classes"));
    check("refuses an unknown default",
          refuses(R"({"question":{"instructions":"x","classes":{"a":"d","b":"e"}},"default":"z","classes":{"a":{"model":"m","device":"hrx"},"b":{"model":"m","device":"hrx"}}})",
                  "\"default\""));
    check("refuses one alias for two backends",
          refuses(R"({"question":{"instructions":"x","classes":{"a":"d","b":"e"}},"classes":{"a":{"model":"m","device":"hrx","alias":"s"},"b":{"model":"n","device":"npu","alias":"s"}}})",
                  "one alias"));

    if (argc > 1) {
        try {
            const ModelTable g = load_model_table(argv[1]);
            check("shipped game table: reflex on npu, plan on hrx",
                  g.pick("reflex", 1.f).device == "npu" && g.pick("plan", 1.f).device == "hrx");
            check("shipped game table: default plan", g.fallback == "plan");
        } catch (const std::exception& e) {
            check(std::string("shipped game table loads: ") + e.what(), false);
        }
    }
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
