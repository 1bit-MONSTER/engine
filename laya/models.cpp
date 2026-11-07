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

#include "models.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "route.h"

namespace onebit::laya {

namespace {

std::string resolve_path(const std::string& p, const std::string& base_dir) {
    namespace fs = std::filesystem;
    if (p.empty()) return p;
    if (p[0] == '~' && (p.size() == 1 || p[1] == '/')) {
        const char* home = std::getenv("HOME");
        return std::string(home ? home : "") + p.substr(1);
    }
    if (fs::path(p).is_absolute() || base_dir.empty()) return p;
    return (fs::path(base_dir) / p).lexically_normal().string();
}

bool known_device(const std::string& d) {
    return d == "npu" || d == "hrx" || d == "cpu" || d == "hybrid";
}

}  // namespace

bool ModelTable::parse(const std::string& json_text, const std::string& base_dir, std::string& err) {
    // ordered: the question's options keep the file's order (Laya's argmax index order)
    nlohmann::ordered_json j;
    try {
        j = nlohmann::ordered_json::parse(json_text);
    } catch (const std::exception& e) {
        err = std::string("model table: ") + e.what();
        return false;
    }
    if (!j.is_object()) { err = "model table: not a JSON object"; return false; }
    const auto& q = j.contains("question") ? j["question"] : nlohmann::ordered_json();
    if (!q.is_object() || !q.contains("instructions") || !q["instructions"].is_string() || !q.contains("classes") ||
        !q["classes"].is_object()) {
        err = "model table: needs \"question\": {\"instructions\": ..., \"classes\": {class: description}}";
        return false;
    }
    question = q["instructions"].get<std::string>();
    classes.clear();
    for (const auto& [cls, desc] : q["classes"].items()) {
        if (!desc.is_string() || desc.get<std::string>().empty()) {
            err = "model table: class \"" + cls + "\" needs a description";
            return false;
        }
        classes.emplace_back(cls, desc.get<std::string>());
    }
    if (classes.size() < 2) { err = "model table: the question needs at least two classes"; return false; }
    auto is_class = [&](const std::string& c) {
        return std::any_of(classes.begin(), classes.end(), [&](const auto& kv) { return kv.first == c; });
    };
    if (!j.contains("classes") || !j["classes"].is_object()) { err = "model table: needs a \"classes\" object"; return false; }
    rows.clear();
    for (const auto& [cls, r] : j["classes"].items()) {
        if (!is_class(cls)) { err = "model table: row \"" + cls + "\" is not one of the question's classes"; return false; }
        if (!r.is_object() || !r.contains("model") || !r["model"].is_string() || !r.contains("device") ||
            !r["device"].is_string()) {
            err = "model table: class \"" + cls + "\" needs \"model\" and \"device\"";
            return false;
        }
        ModelRow row;
        row.model = resolve_path(r["model"].get<std::string>(), base_dir);
        row.device = r["device"].get<std::string>();
        if (!known_device(row.device)) {
            err = "model table: class \"" + cls + "\": unknown device \"" + row.device + "\" (npu, hrx, cpu, hybrid)";
            return false;
        }
        row.alias = r.value("alias", std::filesystem::path(row.model).filename().string());
        if (r.contains("args")) {
            if (!r["args"].is_array()) { err = "model table: class \"" + cls + "\": \"args\" is a list of strings"; return false; }
            for (const auto& a : r["args"]) {
                if (!a.is_string()) { err = "model table: class \"" + cls + "\": \"args\" is a list of strings"; return false; }
                row.args.push_back(a.get<std::string>());
            }
        }
        if (r.contains("bin")) {
            if (!r["bin"].is_string() || r["bin"].get<std::string>().empty()) {
                err = "model table: class \"" + cls + "\": \"bin\" is the path of a 1bit binary";
                return false;
            }
            row.bin = resolve_path(r["bin"].get<std::string>(), base_dir);
        }
        rows.emplace_back(cls, std::move(row));
    }
    for (const auto& [cls, desc] : classes) {
        (void)desc;
        if (std::none_of(rows.begin(), rows.end(), [&](const auto& kv) { return kv.first == cls; })) {
            err = "model table: class \"" + cls + "\" has no row";
            return false;
        }
    }
    // two rows with one alias would make a pinned request ambiguous
    for (size_t a = 0; a < rows.size(); ++a)
        for (size_t b = a + 1; b < rows.size(); ++b)
            if (rows[a].second.alias == rows[b].second.alias &&
                (rows[a].second.model != rows[b].second.model || rows[a].second.device != rows[b].second.device ||
                 rows[a].second.args != rows[b].second.args || rows[a].second.bin != rows[b].second.bin)) {
                err = "model table: classes \"" + rows[a].first + "\" and \"" + rows[b].first +
                      "\" name different backends with one alias \"" + rows[a].second.alias + "\"";
                return false;
            }
    fallback = j.value("default", classes.front().first);
    if (!is_class(fallback)) { err = "model table: \"default\" \"" + fallback + "\" is not a class"; return false; }
    min_confidence = j.value("min_confidence", 0.0f);
    size_gate_chars = 0;
    size_gate_class.clear();
    if (j.contains("size_gate")) {
        const auto& g = j["size_gate"];
        if (!g.is_object() || !g.contains("chars") || !g["chars"].is_number_unsigned() || !g.contains("class") ||
            !g["class"].is_string() || !is_class(g["class"].get<std::string>())) {
            err = "model table: \"size_gate\" is {\"chars\": N, \"class\": <a class>}";
            return false;
        }
        size_gate_chars = g["chars"].get<size_t>();
        size_gate_class = g["class"].get<std::string>();
    }
    return true;
}

Question ModelTable::laya_question() const {
    Question q;
    q.type = "choice";
    q.instructions = question;
    q.criteria = classes;
    return q;
}

const ModelRow& ModelTable::pick(const std::string& cls, float confidence, std::string* used) const {
    const std::string c = (confidence >= min_confidence && !cls.empty()) ? cls : fallback;
    for (const auto& [k, row] : rows)
        if (k == c) {
            if (used) *used = c;
            return row;
        }
    for (const auto& [k, row] : rows)
        if (k == fallback) {
            if (used) *used = fallback;
            return row;
        }
    throw std::logic_error("model table: no row for the fallback class");  // parse() rules this out
}

const ModelRow* ModelTable::row_for(const std::string& cls) const {
    for (const auto& [k, row] : rows)
        if (k == cls) return &row;
    return nullptr;
}

const ModelRow* ModelTable::by_alias(const std::string& name) const {
    for (const auto& [k, row] : rows) {
        (void)k;
        if (row.alias == name) return &row;
    }
    return nullptr;
}

static bool same_backend(const ModelRow& a, const ModelRow& b) {
    return a.model == b.model && a.device == b.device && a.args == b.args && a.bin == b.bin;
}

std::vector<ModelRow> ModelTable::backends() const {
    std::vector<ModelRow> out;
    for (const auto& [k, row] : rows) {
        (void)k;
        if (std::none_of(out.begin(), out.end(), [&](const ModelRow& o) { return same_backend(o, row); })) out.push_back(row);
    }
    return out;
}

size_t ModelTable::backend_index(const ModelRow& row) const {
    const auto b = backends();
    for (size_t i = 0; i < b.size(); ++i)
        if (same_backend(b[i], row)) return i;
    throw std::logic_error("model table: row is not a backend");
}

ModelTable load_model_table(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("model table: cannot read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    ModelTable t;
    std::string err;
    const std::string base = std::filesystem::absolute(path).parent_path().string();
    if (!t.parse(ss.str(), base, err)) throw std::runtime_error(err + " in " + path);
    return t;
}

RequestClass classify_with_table(Scorer& scorer, const ModelTable& table, const std::string& state) {
    RequestClass out;
    if (table.size_gated(state)) {
        out.label = table.size_gate_class;
        out.confidence = 1.0f;
        out.probabilities = {{table.size_gate_class, 1.0f}};
        return out;
    }
    std::vector<Answer> answers;
    if (!scorer.score(state, {table.laya_question()}, answers) || answers.empty()) return out;
    out.label = answers.front().choice;
    out.confidence = answers.front().confidence;
    out.probabilities = answers.front().probabilities;
    return out;
}

}  // namespace onebit::laya
