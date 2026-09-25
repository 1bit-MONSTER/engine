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

// laya_gate <model_dir> <golden.json>
//
// Runs the Laya scorer on the questions + state recorded in golden.json (made
// with the Python reference) and compares the raw pre-softmax logits and
// act logits, the argmax decision, and the temperature-1 softmax probabilities
// against the reference. Exit 0 when every tolerance holds.
#include "scorer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using ordered_json = nlohmann::ordered_json;

namespace {

// Python json.dumps with default separators (", " and ": ") and ensure_ascii,
// so the reconstructed state string tokenizes identically to the reference.
std::string py_escape(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '"') { out += "\\\""; ++i; }
        else if (c == '\\') { out += "\\\\"; ++i; }
        else if (c == '\b') { out += "\\b"; ++i; }
        else if (c == '\f') { out += "\\f"; ++i; }
        else if (c == '\n') { out += "\\n"; ++i; }
        else if (c == '\r') { out += "\\r"; ++i; }
        else if (c == '\t') { out += "\\t"; ++i; }
        else if (c < 0x20) {
            char b[8];
            std::snprintf(b, sizeof b, "\\u%04x", c);
            out += b;
            ++i;
        } else if (c < 0x80) {
            out += char(c);
            ++i;
        } else {
            const int len = (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
            uint32_t cp = c & (len == 2 ? 0x1Fu : len == 3 ? 0x0Fu : 0x07u);
            for (int j = 1; j < len && i + j < s.size(); ++j)
                cp = (cp << 6) | ((unsigned char)s[i + j] & 0x3Fu);
            char b[16];
            if (cp <= 0xFFFF) std::snprintf(b, sizeof b, "\\u%04x", cp);
            else {
                cp -= 0x10000;
                std::snprintf(b, sizeof b, "\\u%04x\\u%04x", 0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF));
            }
            out += b;
            i += len;
        }
    }
    return out;
}

std::string py_dumps(const ordered_json& v) {
    if (v.is_string()) return "\"" + py_escape(v.get<std::string>()) + "\"";
    if (v.is_null()) return "null";
    if (v.is_boolean()) return v.get<bool>() ? "true" : "false";
    if (v.is_number_integer()) return std::to_string(v.get<long long>());
    if (v.is_number_unsigned()) return std::to_string(v.get<unsigned long long>());
    if (v.is_number_float()) {
        char b[32];
        std::snprintf(b, sizeof b, "%.17g", v.get<double>());
        return b;
    }
    if (v.is_array()) {
        std::string out = "[";
        bool first = true;
        for (const auto& e : v) {
            if (!first) out += ", ";
            first = false;
            out += py_dumps(e);
        }
        return out + "]";
    }
    if (v.is_object()) {
        std::string out = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            out += "\"" + py_escape(it.key()) + "\": " + py_dumps(it.value());
        }
        return out + "}";
    }
    return "null";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: laya_gate <model_dir> <golden.json>\n");
        return 2;
    }
    const std::string model_dir = argv[1];

    std::ifstream f(argv[2]);
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 2;
    }
    const ordered_json golden = ordered_json::parse(f);

    // Reconstruct the state string exactly as the Python reference serialized it.
    const std::string state = py_dumps(golden.at("state"));

    // Reconstruct questions in golden.json order (ordered_json keeps it).
    std::vector<onebit::laya::Question> questions;
    for (auto it = golden.at("questions").begin(); it != golden.at("questions").end(); ++it) {
        const auto& q = it.value();
        onebit::laya::Question Q;
        Q.type = q.at("type").get<std::string>();
        Q.instructions = q.value("instructions", "");
        if (Q.type == "choice") {
            for (auto c = q.at("criteria").begin(); c != q.at("criteria").end(); ++c)
                Q.criteria.emplace_back(c.key(), c.value().get<std::string>());
        } else if (Q.type == "score") {
            for (const auto& s : q.at("criteria"))
                Q.criteria.emplace_back("", s.get<std::string>());
        }  // noul: no criteria
        questions.push_back(std::move(Q));
    }

    onebit::laya::Scorer scorer;
    if (!scorer.load(model_dir)) {
        std::fprintf(stderr, "load failed: %s\n", scorer.error().c_str());
        return 1;
    }
    std::vector<onebit::laya::Answer> answers;
    onebit::laya::RawOutput raw;
    if (!scorer.score(state, questions, answers, &raw)) {
        std::fprintf(stderr, "score failed: %s\n", scorer.error().c_str());
        return 1;
    }

    const auto& g_logits = golden.at("logits");
    const auto& g_act = golden.at("act_logits");
    const auto& marker_mask = golden.at("marker_mask");

    const int N = (int)questions.size();
    double max_logit_diff = 0, max_act_rel = 0, max_prob_diff = 0;
    int argmax_mismatches = 0;

    for (int b = 0; b < N; ++b) {
        const auto& gl = g_logits[b];
        for (size_t k = 0; k < gl.size(); ++k)
            max_logit_diff = std::max(max_logit_diff, std::fabs((double)raw.logits[b][k] - gl[k].get<double>()));
        // The act logits are ~1e3-1e4 in magnitude, so compare them relatively.
        for (size_t k = 0; k < 2; ++k) {
            const double g = g_act[b][k].get<double>();
            max_act_rel = std::max(max_act_rel, std::fabs((double)raw.act_logits[b][k] - g) / std::max(1.0, std::fabs(g)));
        }

        // argmax over valid markers (raw logits; temperature > 0 preserves it)
        int gbi = -1, rbi = -1;
        double gmx = -INFINITY;
        float rmx = -INFINITY;
        for (size_t k = 0; k < gl.size(); ++k) {
            if (!marker_mask[b][k].get<bool>()) continue;
            const double gv = gl[k].get<double>();
            const float rv = raw.logits[b][k];
            if (gbi < 0 || gv > gmx) { gmx = gv; gbi = (int)k; }
            if (rbi < 0 || rv > rmx) { rmx = rv; rbi = (int)k; }
        }
        if (gbi != rbi) ++argmax_mismatches;

        // temperature-1 softmax probabilities over valid markers
        auto softmax = [&](const auto& row, auto at) {
            double mx = -INFINITY;
            for (size_t k = 0; k < row.size(); ++k)
                if (marker_mask[b][k].get<bool>()) mx = std::max(mx, at(row, k));
            double sum = 0;
            std::vector<double> e(row.size(), 0.0);
            for (size_t k = 0; k < row.size(); ++k)
                if (marker_mask[b][k].get<bool>()) { e[k] = std::exp(at(row, k) - mx); sum += e[k]; }
            for (auto& x : e) x /= sum;
            return e;
        };
        const auto gp = softmax(gl, [](const auto& row, size_t k) { return row[k].template get<double>(); });
        const auto rp = softmax(raw.logits[b], [](const auto& row, size_t k) { return (double)row[k]; });
        for (size_t k = 0; k < gl.size(); ++k)
            if (marker_mask[b][k].get<bool>())
                max_prob_diff = std::max(max_prob_diff, std::fabs(rp[k] - gp[k]));
    }

    const double logit_tol = 1e-3, act_rel_tol = 1e-3, prob_tol = 1e-4;
    const bool ok = max_logit_diff <= logit_tol && max_act_rel <= act_rel_tol &&
                    argmax_mismatches == 0 && max_prob_diff <= prob_tol;

    std::printf("laya_gate %s\n", argv[2]);
    std::printf("  questions: %d\n", N);
    std::printf("  max |logit| diff:      %.3e  (tol %.0e)\n", max_logit_diff, logit_tol);
    std::printf("  max act_logit rel:     %.3e  (tol %.0e)\n", max_act_rel, act_rel_tol);
    std::printf("  argmax mismatches:    %d\n", argmax_mismatches);
    std::printf("  max |prob| diff:      %.3e  (tol %.0e)\n", max_prob_diff, prob_tol);
    std::printf("  %s\n", ok ? "PASS" : "FAIL");
    for (const auto& a : answers) {
        std::printf("  [%s] choice=%s score=%.4f noul=%.4f conf=%.4f actp=%.4f\n", a.type.c_str(),
                    a.choice.c_str(), a.score, a.noul, a.confidence, a.act_probability);
    }
    return ok ? 0 : 1;
}
