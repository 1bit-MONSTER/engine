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


// Routing accuracy of a model table over labelled states (`1bit route --laya-models FILE --eval CASES`):
// accuracy, per-class recall and precision, the confusion matrix, Laya's confidence, calibration (ECE over
// the top probability) and log loss. Pure arithmetic over the decisions, so it is tested without a checkpoint.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace onebit::laya {

struct EvalCase {
    std::string label;                                     // the true class
    std::string got;                                       // the class the table's question chose
    float confidence = 0.0f;                               // serve's confidence, 1 - H(p)/log k
    std::vector<std::pair<std::string, float>> probabilities;  // class -> p
    double ms = 0.0;                                       // the decision's time
    bool gated = false;                                    // decided by the size gate, not by Laya
};

struct EvalReport {
    std::vector<std::string> classes;
    int n = 0, correct = 0, gated = 0;
    double accuracy = 0.0;
    std::vector<int> support;                 // per true class
    std::vector<double> recall, precision;    // per class
    std::vector<std::vector<int>> confusion;  // [true][chosen]
    double mean_confidence = 0.0;             // serve's confidence
    double mean_top_p = 0.0;
    double ece = 0.0;                         // 10 equal-width bins over the top probability
    double log_loss = 0.0;                    // mean -log p(true class), p floored at 1e-9
    double ms_p50 = 0.0, ms_p90 = 0.0;
};

// classes: the table's classes in its order. A case whose label is not a class is refused (throws).
EvalReport evaluate(const std::vector<EvalCase>& cases, const std::vector<std::string>& classes);

std::string report_json(const EvalReport& r);
std::string report_text(const EvalReport& r);

}  // namespace onebit::laya
