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

// laya/models.h — a model table for `1bit serve --laya-models FILE` (docs/laya-models.md).
// Where --laya picks a *device* for one model, a model table picks a *model and its device*
// for every request: the table names its own classes and the Laya question that tells them
// apart, and each class names the model that answers it and the device it runs on.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "scorer.h"

namespace onebit::laya {

struct RequestClass;

struct ModelRow {
    std::string model;               // a .gguf file or an NPU model directory (~ and the table's directory resolved)
    std::string device;              // npu, hrx, cpu or hybrid: passed to the row's `1bit serve --device`
    std::string alias;               // the name replies carry; a request naming it skips Laya
    std::vector<std::string> args;   // extra `1bit serve` arguments for this row
    std::string bin;                 // the `1bit` binary that serves this row ("" = this one), e.g. a release build
};

struct ModelTable {
    std::string question;                                  // Laya's instructions
    std::vector<std::pair<std::string, std::string>> classes;  // class -> description, the question's options in order
    std::vector<std::pair<std::string, ModelRow>> rows;    // class -> row (every class has one)
    std::string fallback;                                  // the class a low-confidence request takes
    float min_confidence = 0.0f;
    size_t size_gate_chars = 0;                            // 0: off; else a request this long or longer is `size_gate_class`
    std::string size_gate_class;

    // Parses the table; `base_dir` resolves relative model paths. False with the reason in `err`.
    bool parse(const std::string& json_text, const std::string& base_dir, std::string& err);

    // The question asked of Laya: one choice among the classes.
    Question laya_question() const;

    // The row for class `cls` at Laya confidence `confidence`: the class's own row, or the
    // fallback class's row when the confidence is below min_confidence. Returns the class
    // that was used (cls or fallback) through `used`.
    const ModelRow& pick(const std::string& cls, float confidence, std::string* used = nullptr) const;

    // Whether `state` is decided by the size gate (no model call, and no min_confidence).
    bool size_gated(const std::string& state) const { return size_gate_chars > 0 && state.size() >= size_gate_chars; }

    // The class's own row (nullptr for an unknown class).
    const ModelRow* row_for(const std::string& cls) const;

    // The row whose alias is `name`, or nullptr.
    const ModelRow* by_alias(const std::string& name) const;

    // The distinct backends (model, device, args): two classes may share one.
    std::vector<ModelRow> backends() const;
    size_t backend_index(const ModelRow& row) const;
};

// Loads a table file (the table's own directory resolves relative model paths).
ModelTable load_model_table(const std::string& path);

// Classifies `state` with `table`'s question on the C++ scorer, applying the size gate first.
RequestClass classify_with_table(Scorer& scorer, const ModelTable& table, const std::string& state);

}  // namespace onebit::laya
