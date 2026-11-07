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


// laya/eval.cpp, the arithmetic of `1bit route --laya-models FILE --eval`: hand-computed cases for
// accuracy, recall, precision, the confusion matrix, ECE, log loss, the size-gate count and the refusal.
#include "eval.h"

#include <cmath>
#include <cstdio>
#include <string>

#include <nlohmann/json.hpp>

namespace {
int failures = 0;
void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) failures++;
}
bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

onebit::laya::EvalCase c(const std::string& label, const std::string& got, float p_reflex, double ms, bool gated = false) {
    onebit::laya::EvalCase e;
    e.label = label;
    e.got = got;
    e.probabilities = {{"reflex", p_reflex}, {"plan", 1.0f - p_reflex}};
    e.confidence = 0.5f;
    e.ms = ms;
    e.gated = gated;
    return e;
}
}  // namespace

int main() {
    using namespace onebit::laya;
    // 4 reflex (3 right), 2 plan (1 right, one by the size gate)
    const std::vector<EvalCase> cases = {
        c("reflex", "reflex", 0.9f, 10), c("reflex", "reflex", 0.8f, 20), c("reflex", "reflex", 0.6f, 30),
        c("reflex", "plan", 0.3f, 40),   c("plan", "plan", 0.0f, 0, true), c("plan", "reflex", 0.55f, 50),
    };
    const EvalReport r = evaluate(cases, {"reflex", "plan"});
    check("n and correct", r.n == 6 && r.correct == 4);
    check("accuracy 4/6", near(r.accuracy, 4.0 / 6));
    check("gated counted", r.gated == 1);
    check("support", r.support[0] == 4 && r.support[1] == 2);
    check("confusion", r.confusion[0][0] == 3 && r.confusion[0][1] == 1 && r.confusion[1][0] == 1 && r.confusion[1][1] == 1);
    check("recall", near(r.recall[0], 0.75) && near(r.recall[1], 0.5));
    check("precision", near(r.precision[0], 0.75) && near(r.precision[1], 0.5));
    // -log p(true): 0.9, 0.8, 0.6, 0.3 (reflex), 1.0, 0.45 (plan)
    const double ll = -(std::log(0.9) + std::log(0.8) + std::log(0.6) + std::log(0.3) + std::log(1.0) + std::log(0.45)) / 6;
    check("log loss", std::fabs(r.log_loss - ll) < 1e-5);
    // top p: .9 .8 .6 .7 1.0 .55; bins 9:{.9 ok} 8:{.8 ok} 6:{.6 ok} 7:{.7 wrong} 10->9:{1.0 ok} 5:{.55 wrong}
    // bin 9: conf (.9+1)/2=.95, acc 1 -> .05*2/6; bin 8: .2*1/6; bin 6: .4/6; bin 7: .7/6; bin 5: .55/6
    const double ece = (0.05 * 2 + 0.2 + 0.4 + 0.7 + 0.55) / 6;
    check("ECE", std::fabs(r.ece - ece) < 1e-5);
    check("decision ms p50 / p90", near(r.ms_p50, 25.0) && near(r.ms_p90, 45.0));
    const auto j = nlohmann::json::parse(report_json(r));
    check("json carries the per-class numbers", j["classes"]["plan"]["recall"] == 0.5 && j["n"] == 6);
    check("text report names the accuracy", report_text(r).find("accuracy 4/6") != std::string::npos);
    bool threw = false;
    try {
        evaluate({c("chat", "reflex", 0.5f, 1)}, {"reflex", "plan"});
    } catch (const std::exception&) {
        threw = true;
    }
    check("refuses a label that is not a class", threw);
    check("empty set is all zeros", evaluate({}, {"reflex", "plan"}).n == 0);
    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
