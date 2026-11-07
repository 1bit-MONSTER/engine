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


#include "route_eval.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "eval.h"
#include "laya_checkpoint.h"
#include "models.h"
#include "route.h"
#include "scorer.h"

namespace onebit {
namespace {

struct Case {
    std::string state;
    std::string label;  // empty: unlabelled (dump only)
    std::string src;
};

// One JSON object per line: {"state": TEXT, "label": CLASS, "src": optional id}. The harness's
// decisions.jsonl lines carry the same "state" and "label" keys, so they are cases as they are.
std::vector<Case> read_cases(const std::string& path, bool need_label) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::vector<Case> out;
    std::string line;
    int n = 0;
    while (std::getline(in, line)) {
        ++n;
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(line);
        } catch (const std::exception& e) {
            throw std::runtime_error(path + ":" + std::to_string(n) + ": " + e.what());
        }
        if (!j.is_object() || !j.contains("state") || !j["state"].is_string())
            throw std::runtime_error(path + ":" + std::to_string(n) + ": needs \"state\": TEXT");
        Case c;
        c.state = j["state"].get<std::string>();
        if (j.contains("label") && j["label"].is_string()) c.label = j["label"].get<std::string>();
        if (need_label && c.label.empty()) {
            // an unlabelled decision (no outcome yet) is not a test case
            continue;
        }
        c.src = j.value("src", path + ":" + std::to_string(n));
        out.push_back(std::move(c));
    }
    return out;
}

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int index_of(const std::vector<std::pair<std::string, std::string>>& v, const std::string& k) {
    for (size_t i = 0; i < v.size(); ++i)
        if (v[i].first == k) return (int)i;
    return -1;
}

int run_eval(laya::Scorer& scorer, const laya::ModelTable& table, const RouteEvalOptions& o) {
    const std::vector<Case> cases = read_cases(o.eval_cases, true);
    if (cases.empty()) throw std::runtime_error(o.eval_cases + ": no labelled cases");
    std::vector<std::string> classes;
    for (const auto& [k, d] : table.classes) {
        (void)d;
        classes.push_back(k);
    }
    std::ofstream per;
    if (!o.per_case.empty()) {
        per.open(o.per_case);
        if (!per) throw std::runtime_error("cannot write " + o.per_case);
    }
    std::vector<laya::EvalCase> got;
    for (const Case& c : cases) {
        if (index_of(table.classes, c.label) < 0)
            throw std::runtime_error(c.src + ": label \"" + c.label + "\" is not one of the table's classes");
        const auto t = std::chrono::steady_clock::now();
        // the path serve takes: the size gate, then the table's question
        const laya::RequestClass rc = laya::classify_with_table(scorer, table, c.state);
        laya::EvalCase e;
        e.ms = ms_since(t);
        if (rc.label.empty()) throw std::runtime_error(c.src + ": " + scorer.error());
        e.label = c.label;
        e.got = rc.label;
        e.confidence = rc.confidence;
        e.probabilities = rc.probabilities;
        e.gated = table.size_gated(c.state);
        if (per.is_open()) {
            nlohmann::ordered_json j{{"src", c.src}, {"label", e.label}, {"chosen", e.got},
                                     {"confidence", e.confidence}, {"gated", e.gated}, {"decision_ms", e.ms}};
            nlohmann::ordered_json p = nlohmann::ordered_json::object();
            for (const auto& [k, v] : e.probabilities) p[k] = v;
            j["probabilities"] = p;
            per << j.dump() << '\n';
        }
        got.push_back(std::move(e));
    }
    const laya::EvalReport r = laya::evaluate(got, classes);
    if (o.json) std::printf("%s\n", laya::report_json(r).c_str());
    else std::printf("%s", laya::report_text(r).c_str());
    return 0;
}

// DIR/h.f32: every case's encoder rows (float32, little endian, `d` per row) back to back.
// DIR/cases.jsonl: one line per (case, rotation): src, label index (-1 unlabelled), the options in the
// order the sequence carried them, qtype, row offset and length, marker positions, and the engine's own
// logits and probabilities, so a trainer can check that its head reproduces the engine before training.
// DIR/dump.json: d, rows, cases, skipped (size-gated) and the table's question.
int run_dump(laya::Scorer& scorer, const laya::ModelTable& table, const RouteEvalOptions& o) {
    namespace fs = std::filesystem;
    if (o.out_dir.empty()) throw std::runtime_error("--dump-encoder needs --out DIR");
    const std::vector<Case> cases = read_cases(o.dump_cases, false);
    fs::create_directories(o.out_dir);
    std::ofstream h(o.out_dir + "/h.f32", std::ios::binary);
    std::ofstream meta(o.out_dir + "/cases.jsonl");
    if (!h || !meta) throw std::runtime_error("cannot write in " + o.out_dir);
    const laya::Question base = table.laya_question();
    const size_t k = base.criteria.size();
    long long rows = 0;
    int written = 0, skipped = 0, d = 0;
    for (const Case& c : cases) {
        if (table.size_gated(c.state)) {  // serve never asks Laya about these, so the head never sees them
            ++skipped;
            continue;
        }
        const int y = c.label.empty() ? -1 : index_of(base.criteria, c.label);
        if (!c.label.empty() && y < 0)
            throw std::runtime_error(c.src + ": label \"" + c.label + "\" is not one of the table's classes");
        for (size_t rot = 0; rot < (o.rotate ? k : 1); ++rot) {
            laya::Question q = base;
            for (size_t i = 0; i < k; ++i) q.criteria[i] = base.criteria[(i + rot) % k];
            laya::RawOutput raw;
            raw.keep_encoder = true;
            std::vector<laya::Answer> answers;
            if (!scorer.score(c.state, {q}, answers, &raw) || answers.empty())
                throw std::runtime_error(c.src + ": " + scorer.error());
            d = raw.d;
            const std::vector<float>& enc = raw.encoder.front();
            const long long len = (long long)(enc.size() / (size_t)d);
            h.write(reinterpret_cast<const char*>(enc.data()), (std::streamsize)(enc.size() * sizeof(float)));
            std::vector<std::string> order;
            for (const auto& [key, desc] : q.criteria) {
                (void)desc;
                order.push_back(key);
            }
            nlohmann::ordered_json probs = nlohmann::ordered_json::array();
            for (const auto& key : order) {
                float p = 0.0f;
                for (const auto& [lk, lp] : answers.front().probabilities)
                    if (lk == key) p = lp;
                probs.push_back(p);
            }
            std::vector<float> logits(raw.logits.front().begin(), raw.logits.front().begin() + (long)k);
            nlohmann::ordered_json j{{"src", c.src},
                                     {"rotation", rot},
                                     {"label", y < 0 ? -1 : (int)((y + k - rot) % k)},
                                     {"options", order},
                                     {"qtype", raw.qtype.front()},
                                     {"offset", rows},
                                     {"len", len},
                                     {"markers", raw.markers.front()},
                                     {"logits", logits},
                                     {"probabilities", probs}};
            meta << j.dump() << '\n';
            rows += len;
            ++written;
        }
    }
    nlohmann::ordered_json info{{"d", d},
                                {"rows", rows},
                                {"cases", written},
                                {"skipped_size_gated", skipped},
                                {"rotate", o.rotate},
                                {"question", table.question},
                                {"laya_model", o.laya_model}};
    std::ofstream(o.out_dir + "/dump.json") << info.dump(1) << '\n';
    std::fprintf(stderr, "1bit route: %d sequences, %lld rows of %d floats in %s (%d size-gated cases skipped)\n",
                 written, rows, d, o.out_dir.c_str(), skipped);
    return 0;
}

}  // namespace

int run_route_eval(const RouteEvalOptions& in) {
    RouteEvalOptions o = in;
    if (o.eval_cases.empty() == o.dump_cases.empty())
        throw std::runtime_error("--laya-models takes exactly one of --eval CASES and --dump-encoder CASES");
    const laya::ModelTable table = laya::load_model_table(o.table);
    o.laya_model = laya_checkpoint(o.laya_model);
    laya::Scorer scorer;
    const auto t = std::chrono::steady_clock::now();
    if (!scorer.load(o.laya_model)) throw std::runtime_error(scorer.error());
    std::fprintf(stderr, "1bit route: scorer %s loaded in %.0f ms\n", o.laya_model.c_str(), ms_since(t));
    return o.eval_cases.empty() ? run_dump(scorer, table, o) : run_eval(scorer, table, o);
}

}  // namespace onebit
