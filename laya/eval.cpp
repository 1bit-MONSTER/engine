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


#include "eval.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace onebit::laya {

namespace {

int index_of(const std::vector<std::string>& v, const std::string& s) {
    const auto it = std::find(v.begin(), v.end(), s);
    return it == v.end() ? -1 : (int)(it - v.begin());
}

double quantile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double pos = q * (double)(v.size() - 1);
    const size_t lo = (size_t)std::floor(pos), hi = (size_t)std::ceil(pos);
    return v[lo] + (v[hi] - v[lo]) * (pos - (double)lo);
}

}  // namespace

EvalReport evaluate(const std::vector<EvalCase>& cases, const std::vector<std::string>& classes) {
    EvalReport r;
    r.classes = classes;
    const size_t k = classes.size();
    r.support.assign(k, 0);
    r.recall.assign(k, 0.0);
    r.precision.assign(k, 0.0);
    r.confusion.assign(k, std::vector<int>(k, 0));
    std::vector<double> ms;
    double bin_conf[10] = {0}, bin_acc[10] = {0};
    int bin_n[10] = {0};
    for (const EvalCase& c : cases) {
        const int t = index_of(classes, c.label);
        if (t < 0) throw std::runtime_error("eval: label \"" + c.label + "\" is not one of the table's classes");
        const int g = index_of(classes, c.got);
        r.n++;
        r.support[t]++;
        if (c.gated) r.gated++;
        if (g >= 0) r.confusion[t][g]++;
        const bool ok = g == t;
        if (ok) r.correct++;
        r.mean_confidence += c.confidence;
        float top = 0.0f, pt = 0.0f;
        for (const auto& [cls, p] : c.probabilities) {
            top = std::max(top, p);
            if (cls == c.label) pt = p;
        }
        r.mean_top_p += top;
        r.log_loss += -std::log(std::max(1e-9, (double)pt));
        const int b = std::min(9, (int)(top * 10.0f));
        bin_conf[b] += top;
        bin_acc[b] += ok ? 1.0 : 0.0;
        bin_n[b]++;
        ms.push_back(c.ms);
    }
    if (r.n == 0) return r;
    r.accuracy = (double)r.correct / r.n;
    r.mean_confidence /= r.n;
    r.mean_top_p /= r.n;
    r.log_loss /= r.n;
    for (int b = 0; b < 10; ++b)
        if (bin_n[b]) r.ece += std::fabs(bin_acc[b] / bin_n[b] - bin_conf[b] / bin_n[b]) * bin_n[b] / r.n;
    for (size_t c = 0; c < k; ++c) {
        int chosen = 0;
        for (size_t t = 0; t < k; ++t) chosen += r.confusion[t][c];
        r.recall[c] = r.support[c] ? (double)r.confusion[c][c] / r.support[c] : 0.0;
        r.precision[c] = chosen ? (double)r.confusion[c][c] / chosen : 0.0;
    }
    r.ms_p50 = quantile(ms, 0.5);
    r.ms_p90 = quantile(ms, 0.9);
    return r;
}

std::string report_json(const EvalReport& r) {
    nlohmann::ordered_json j;
    j["n"] = r.n;
    j["correct"] = r.correct;
    j["accuracy"] = r.accuracy;
    j["gated"] = r.gated;
    nlohmann::ordered_json per = nlohmann::ordered_json::object();
    for (size_t c = 0; c < r.classes.size(); ++c)
        per[r.classes[c]] = {{"support", r.support[c]}, {"recall", r.recall[c]}, {"precision", r.precision[c]}};
    j["classes"] = per;
    j["confusion"] = {{"rows_true_cols_chosen", r.classes}, {"matrix", r.confusion}};
    j["mean_confidence"] = r.mean_confidence;
    j["mean_top_p"] = r.mean_top_p;
    j["ece"] = r.ece;
    j["log_loss"] = r.log_loss;
    j["decision_ms_p50"] = r.ms_p50;
    j["decision_ms_p90"] = r.ms_p90;
    return j.dump();
}

std::string report_text(const EvalReport& r) {
    std::string out;
    char buf[256];
    std::snprintf(buf, sizeof buf, "accuracy %d/%d = %.3f (%d by the size gate)\n", r.correct, r.n, r.accuracy, r.gated);
    out += buf;
    for (size_t c = 0; c < r.classes.size(); ++c) {
        std::snprintf(buf, sizeof buf, "  %-16s support %4d  recall %.3f  precision %.3f\n", r.classes[c].c_str(),
                      r.support[c], r.recall[c], r.precision[c]);
        out += buf;
    }
    out += "confusion (rows true, columns chosen):\n";
    for (size_t t = 0; t < r.classes.size(); ++t) {
        std::snprintf(buf, sizeof buf, "  %-16s", r.classes[t].c_str());
        out += buf;
        for (const int v : r.confusion[t]) {
            std::snprintf(buf, sizeof buf, " %5d", v);
            out += buf;
        }
        out += "\n";
    }
    std::snprintf(buf, sizeof buf,
                  "confidence mean %.3f, top p mean %.3f, ECE %.3f, log loss %.3f, decision p50 %.0f ms, p90 %.0f ms\n",
                  r.mean_confidence, r.mean_top_p, r.ece, r.log_loss, r.ms_p50, r.ms_p90);
    out += buf;
    return out;
}

}  // namespace onebit::laya
