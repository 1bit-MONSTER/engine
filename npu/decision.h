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

// A Decision-3.0 (d3) readout head (engine #653): the decision metadata plus the readout matrix,
// as one loader tensor class.
//
// A d3 pack is a Qwen3.5 GDN backbone plus two pieces the model-generic loader ignores:
//
//   decision_config.json   the option `codes` ("A".."JT", at most 255), the same options as
//                          backbone `token_ids`, the family's `prompt` ("d3"), `format_id`,
//                          `base_model`, and the readout `temperature` (Decision 3.0 uses 1.0)
//   readout.safetensors    one F32 matrix [codes, hidden] -- the per-option logits for a pooled
//                          backbone state; tools/safetensors_to_q4nx.py decision turns it into
//                          `decision.q4nx` in the Q4NX container so one reader serves both
//
// So a decision is: run the backbone over the state and question, pool the last hidden state
// (attention_mode "noncausal_full_attention", pooling "last"), multiply by this matrix, and take a
// temperature softmax over the options the question offers.  Nothing here touches a device or a
// model: the class loads two files and does the arithmetic on a hidden vector, which is what makes
// it testable without hardware.
#pragma once

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace onebit::npu {

// The pack's own metadata, read from decision_config.json.  A field the endpoint depends on but the
// pack does not declare is refused at load, never guessed at request time.
struct DecisionConfig {
    std::string format_id;
    std::string prompt;
    std::string base_model;
    std::string attention_mode;
    std::string pooling;
    std::vector<std::string> codes;  // one option name per readout row, in row order
    std::vector<int> token_ids;      // the same options as backbone vocabulary ids
    float temperature = 1.0f;

    size_t options() const { return codes.size(); }
};

class DecisionHead {
public:
    // Loads `dir/decision_config.json` and `dir/decision.q4nx`.  Returns false and sets error()
    // rather than throwing, so an endpoint can answer 503 with the reason.
    bool load(const std::string& dir) {
        error_.clear();
        ok_ = false;
        try {
            read_config(dir + "/decision_config.json");
            read_readout(dir + "/decision.q4nx");
        } catch (const std::exception& e) {
            error_ = e.what();
            return false;
        }
        ok_ = true;
        return true;
    }

    bool ok() const { return ok_; }
    const std::string& error() const { return error_; }
    const DecisionConfig& config() const { return cfg_; }
    size_t options() const { return cfg_.codes.size(); }
    size_t hidden() const { return hidden_; }

    // W * h: one logit per option code, in the pack's row order.
    std::vector<float> logits(const std::vector<float>& hidden) const {
        if (hidden.size() != hidden_) throw std::runtime_error("decision: hidden state is " +
                                                              std::to_string(hidden.size()) +
                                                              " wide, the readout is " +
                                                              std::to_string(hidden_));
        std::vector<float> out(options(), 0.0f);
        for (size_t r = 0; r < options(); r++) {
            const float* row = &w_[r * hidden_];
            double acc = 0.0;
            for (size_t i = 0; i < hidden_; i++) acc += double(row[i]) * double(hidden[i]);
            out[r] = float(acc);
        }
        return out;
    }

    // Temperature softmax over the options a question offers (`rows` indexes the readout rows).
    // This is the d3 contract: a probability for every option of THAT question, summing to 1.
    std::vector<float> probabilities_over(const std::vector<float>& all_logits,
                                         const std::vector<int>& rows) const {
        if (rows.empty()) throw std::runtime_error("decision: a question needs at least one option");
        std::vector<float> picked;
        picked.reserve(rows.size());
        for (const int r : rows) {
            if (r < 0 || size_t(r) >= all_logits.size())
                throw std::runtime_error("decision: option row " + std::to_string(r) + " is outside the readout");
            picked.push_back(all_logits[size_t(r)]);
        }
        const float t = cfg_.temperature > 0 ? cfg_.temperature : 1.0f;
        float mx = picked.front();
        for (const float v : picked) mx = std::max(mx, v);
        double sum = 0.0;
        std::vector<float> out(picked.size());
        for (size_t i = 0; i < picked.size(); i++) {
            out[i] = float(std::exp(double(picked[i] - mx) / double(t)));
            sum += out[i];
        }
        if (!(sum > 0.0)) {  // every option at -inf: an even split is the only honest answer
            for (auto& v : out) v = 1.0f / float(out.size());
            return out;
        }
        for (auto& v : out) v = float(double(v) / sum);
        return out;
    }

    // Convenience: the probabilities of every option in the pack's order.
    std::vector<float> probabilities(const std::vector<float>& hidden) const {
        std::vector<int> all(options());
        for (size_t i = 0; i < all.size(); i++) all[i] = int(i);
        return probabilities_over(logits(hidden), all);
    }

    const std::vector<float>& weight() const { return w_; }

private:
    void read_config(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("decision: cannot read " + path);
        const auto c = nlohmann::json::parse(f);
        cfg_ = DecisionConfig{};
        cfg_.format_id = c.value("format_id", "");
        cfg_.prompt = c.value("prompt", "");
        cfg_.base_model = c.value("base_model", "");
        cfg_.attention_mode = c.value("attention_mode", "");
        cfg_.pooling = c.value("pooling", "");
        cfg_.temperature = c.value("temperature", 1.0f);
        if (!c.contains("codes") || !c.at("codes").is_array())
            throw std::runtime_error("decision: " + path + " has no codes array");
        for (const auto& code : c.at("codes")) {
            if (!code.is_string()) throw std::runtime_error("decision: codes must be strings");
            cfg_.codes.push_back(code.get<std::string>());
        }
        if (cfg_.codes.empty()) throw std::runtime_error("decision: codes is empty");
        if (cfg_.codes.size() > kMaxOptions)
            throw std::runtime_error("decision: " + std::to_string(cfg_.codes.size()) + " codes, the d3 " +
                                     "contract allows at most " + std::to_string(kMaxOptions));
        if (c.contains("token_ids")) {
            if (!c.at("token_ids").is_array() || c.at("token_ids").size() != cfg_.codes.size())
                throw std::runtime_error("decision: token_ids must match codes one for one");
            for (const auto& id : c.at("token_ids")) cfg_.token_ids.push_back(id.get<int>());
        }
    }

    void read_readout(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("decision: cannot read " + path +
                                         " (tools/safetensors_to_q4nx.py decision writes it)");
        uint64_t header = 0;
        f.read(reinterpret_cast<char*>(&header), 8);
        if (!f) throw std::runtime_error("decision: " + path + " is truncated");
        if (header > (1u << 26)) throw std::runtime_error("decision: " + path + " header is implausible");
        std::string htext(size_t(header), '\0');
        f.read(htext.data(), std::streamsize(header));
        if (!f) throw std::runtime_error("decision: " + path + " header is truncated");
        const auto table = nlohmann::json::parse(htext);
        const nlohmann::json* found = nullptr;
        std::string name;
        for (const auto& [k, v] : table.items()) {
            if (!v.is_object() || !v.contains("data_offsets")) continue;  // __metadata__
            found = &v;
            name = k;
        }
        if (!found) throw std::runtime_error("decision: " + path + " holds no tensor");
        if (found->value("dtype", "") != "F32")
            throw std::runtime_error("decision: readout dtype " + found->value("dtype", "?") +
                                     " is not F32; the head is the model's decision, not a quantized weight");
        const auto shape = found->at("shape").get<std::vector<long long>>();
        if (shape.size() != 2) throw std::runtime_error("decision: readout is not a matrix");
        if (size_t(shape[0]) != cfg_.codes.size())
            throw std::runtime_error("decision: readout has " + std::to_string(shape[0]) + " rows but the pack " +
                                     "declares " + std::to_string(cfg_.codes.size()) + " codes");
        hidden_ = size_t(shape[1]);
        const auto off = found->at("data_offsets").get<std::vector<uint64_t>>();
        if (off.size() != 2 || off[1] < off[0] || off[1] - off[0] != shape[0] * shape[1] * 4)
            throw std::runtime_error("decision: readout data_offsets do not match its shape");
        f.seekg(std::streamoff(8 + header + off[0]));
        w_.resize(size_t(shape[0]) * size_t(shape[1]));
        f.read(reinterpret_cast<char*>(w_.data()), std::streamsize(w_.size() * 4));
        if (!f) throw std::runtime_error("decision: readout data is truncated");
        if (w_.size() != options() * hidden_)
            throw std::runtime_error("decision: readout is not codes x hidden");
        (void)name;
    }

    static constexpr size_t kMaxOptions = 255;  // d3: "choice picks one of up to 255 options"
    DecisionConfig cfg_;
    std::vector<float> w_;  // [codes][hidden], row-major
    size_t hidden_ = 0;
    bool ok_ = false;
    std::string error_;
};

}  // namespace onebit::npu
