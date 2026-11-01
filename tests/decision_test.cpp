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

// decision_test: the d3 readout head loader (npu/decision.h) without a device, a backbone or a
// downloaded pack -- every fixture is written to a temporary directory here.
#include "decision.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using onebit::npu::DecisionHead;

namespace {

int failures = 0;
void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

// One Q4NX container with a single F32 tensor, exactly what `safetensors_to_q4nx.py decision` writes.
void write_readout(const fs::path& path, const std::vector<float>& w, int rows, int cols,
                   const std::string& dtype = "F32") {
    nlohmann::json t;
    t["weight"] = {{"dtype", dtype}, {"shape", {rows, cols}},
                   {"data_offsets", {0, w.size() * sizeof(float)}}};
    t["__metadata__"] = {{"kind", "d3-readout"}};
    std::string h = t.dump();
    h.append((8 - h.size() % 8) % 8, ' ');
    std::ofstream f(path, std::ios::binary);
    const uint64_t n = h.size();
    f.write(reinterpret_cast<const char*>(&n), 8);
    f.write(h.data(), std::streamsize(h.size()));
    f.write(reinterpret_cast<const char*>(w.data()), std::streamsize(w.size() * sizeof(float)));
}

void write_config(const fs::path& path, const nlohmann::json& cfg) {
    std::ofstream f(path);
    f << cfg.dump(2) << "\n";
}

nlohmann::json base_config() {
    return {{"format_version", 1},
            {"format_id", "d3-code-readout-v1"},
            {"prompt", "d3"},
            {"base_model", "Qwen/Qwen3.5-0.8B"},
            {"attention_mode", "noncausal_full_attention"},
            {"pooling", "last"},
            {"temperature", 1.0},
            {"codes", {"A", "B", "C"}},
            {"token_ids", {32, 33, 34}}};
}

bool close(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) <= eps; }

void test_load_and_logits() {
    const fs::path dir = fs::temp_directory_path() / "onebit-decision-test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    // W = [[1,0,0],[0,1,0],[0,0,2]] rows A,B,C over a 3-wide hidden state.
    const std::vector<float> w = {1, 0, 0, 0, 1, 0, 0, 0, 2};
    write_config(dir / "decision_config.json", base_config());
    write_readout(dir / "decision.q4nx", w, 3, 3);

    DecisionHead head;
    check(head.load(dir.string()), "the head loads: " + head.error());
    check(head.ok(), "ok() agrees with load()");
    check(head.options() == 3 && head.hidden() == 3, "codes and hidden width come from the pack");
    check(head.config().codes[0] == "A" && head.config().codes[2] == "C", "codes keep their order");
    check(head.config().token_ids.size() == 3 && head.config().token_ids[2] == 34,
          "token_ids are the options as vocabulary ids");
    check(head.config().format_id == "d3-code-readout-v1", "the format id is read");
    check(close(head.config().temperature, 1.0f), "the readout temperature is read");

    const auto logits = head.logits({2.0f, 3.0f, 0.5f});
    check(logits.size() == 3 && close(logits[0], 2.0f) && close(logits[1], 3.0f) && close(logits[2], 1.0f),
          "logits are W * h, one per option");

    const auto p = head.probabilities({2.0f, 3.0f, 0.5f});
    check(p.size() == 3, "one probability per option");
    double sum = 0.0;
    for (float v : p) sum += v;
    check(close(float(sum), 1.0f, 1e-4f), "the probabilities sum to 1");
    check(p[1] > p[0] && p[0] > p[2], "the largest logit gets the largest probability");

    // A question that offers only two of the three options: softmax is over THOSE options.
    const auto two = head.probabilities_over(logits, {0, 2});
    check(two.size() == 2 && close(two[0] + two[1], 1.0f, 1e-4f), "a subset of options still sums to 1");
    const float expected = 1.0f / (1.0f + std::exp(1.0f - 2.0f));  // logits 2 and 1, T = 1
    check(close(two[0], expected, 1e-4f), "the subset softmax matches the hand-computed value");

    fs::remove_all(dir);
}

void test_temperature_and_shape_errors() {
    const fs::path dir = fs::temp_directory_path() / "onebit-decision-test2";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::vector<float> w = {2, 0, 0, 2};  // two options over a 2-wide state
    auto cfg = base_config();
    cfg["codes"] = {"A", "B"};
    cfg["token_ids"] = {32, 33};
    cfg["temperature"] = 0.5;
    write_config(dir / "decision_config.json", cfg);
    write_readout(dir / "decision.q4nx", w, 2, 2);

    DecisionHead head;
    check(head.load(dir.string()), "a two-option head loads");
    const auto p = head.probabilities({1.0f, 0.0f});
    // logits 2 and 0, T = 0.5 -> exp(4) vs exp(0)
    const float expected = std::exp(4.0f) / (std::exp(4.0f) + 1.0f);
    check(close(p[0], expected, 1e-4f), "the readout temperature is applied");

    auto bad = cfg;
    bad["codes"] = {"A", "B", "C"};
    bad["token_ids"] = {32, 33, 34};
    write_config(dir / "decision_config.json", bad);
    DecisionHead wrong;
    check(!wrong.load(dir.string()), "a head with fewer rows than codes is refused");
    check(wrong.error().find("codes") != std::string::npos, "and the error names the mismatch");

    write_config(dir / "decision_config.json", cfg);
    write_readout(dir / "decision.q4nx", w, 2, 2, "I8");
    DecisionHead quantized;
    check(!quantized.load(dir.string()), "a quantized readout is refused (it must be F32)");
    check(quantized.error().find("F32") != std::string::npos, "and says so");

    fs::remove_all(dir);
}

void test_missing_pieces() {
    const fs::path dir = fs::temp_directory_path() / "onebit-decision-test3";
    fs::remove_all(dir);
    fs::create_directories(dir);
    DecisionHead head;
    check(!head.load(dir.string()), "a directory without a pack is refused");
    check(head.error().find("decision_config.json") != std::string::npos, "the error names the missing config");

    write_config(dir / "decision_config.json", base_config());
    DecisionHead no_readout;
    check(!no_readout.load(dir.string()), "a pack without its readout is refused");
    check(no_readout.error().find("decision.q4nx") != std::string::npos, "the error names the missing readout");

    // 300 codes is past the d3 contract's 255 options.
    auto too_many = base_config();
    std::vector<std::string> codes;
    for (int i = 0; i < 300; i++) codes.push_back("C" + std::to_string(i));
    too_many["codes"] = codes;
    too_many.erase("token_ids");
    write_config(dir / "decision_config.json", too_many);
    write_readout(dir / "decision.q4nx", std::vector<float>(300 * 2, 0.0f), 300, 2);
    DecisionHead big;
    check(!big.load(dir.string()), "more than 255 options is refused");
    check(big.error().find("255") != std::string::npos, "and the error says the limit");

    fs::remove_all(dir);
}

}  // namespace

int main() {
    test_load_and_logits();
    test_temperature_and_shape_errors();
    test_missing_pieces();
    std::printf("%s\n", failures ? "FAILED" : "decision: ok");
    return failures ? 1 : 0;
}
