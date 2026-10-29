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

// Decode-time controls for `1bit forward-serve`: the sampling knobs, the stop
// sequences, and the request parsing that turns an OpenAI-shaped body into them.
//
// The sampling is llama.cpp's, mirrored rather than linked: the NPU route holds raw
// logits and an npu::Tokenizer, not a llama_context, and llama.cpp is only built under
// ONEBIT_HRX.  The order of operations is llama.cpp's sampler chain
// (third_party/llama.cpp/common/sampling.cpp:346-391): top_k and top_p filter the raw
// logits, temperature scales the survivors, and the draw is a softmax over them.
// Temperature divides logits and never probabilities: the distribution top_p reads is a
// scratch view of the logits, not a place to write them back to.
//
// Greedy stays the default: temperature 0 (the value a body that names no sampling
// field gets) is the argmax this route has always used, so existing callers decode
// exactly as before.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace onebit::npu {

// Sentinel for "no seed given": the run picks one, like llama.cpp's LLAMA_DEFAULT_SEED.
inline constexpr uint64_t kRandomSeed = ~uint64_t(0);

// At most this many stop sequences, as in the OpenAI API.
inline constexpr size_t kMaxStop = 4;

struct SampleParams {
    float temperature = 0.0f;  // 0 = greedy (argmax), this route's historical decode
    int top_k = 0;             // 0 = off
    float top_p = 1.0f;        // 1 = off
    int min_keep = 1;          // top_p keeps at least this many candidates
    uint64_t seed = kRandomSeed;

    bool greedy() const { return temperature <= 0.0f; }
};

// A seed for one request: the caller's when it named one, otherwise a fresh one, so a
// sampled run can be reproduced by sending the seed back.
inline uint64_t resolve_seed(uint64_t seed) {
    if (seed != kRandomSeed) return seed;
    std::random_device rd;
    uint64_t s = (uint64_t(rd()) << 32) ^ rd();
    s ^= uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    return s;
}

struct Candidate {
    float logit = 0.0f;
    int id = 0;
};

// Draw one token from `logits`.  `allowed`, when given, is a per-token mask (non-zero =
// the grammar accepts the token); a mask with nothing set is refused with -1 and the
// caller reports it rather than decoding an unconstrained token.
inline int sample_token(const std::vector<float>& logits, const SampleParams& p, std::mt19937_64& rng,
                        const std::vector<char>* allowed = nullptr) {
    std::vector<Candidate> c;
    c.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        if (allowed && !(*allowed)[i]) continue;
        c.push_back({logits[i], int(i)});
    }
    if (c.empty()) return -1;
    // A higher logit wins; equal logits keep the lower id so the same seed always draws
    // the same token.
    auto better = [](const Candidate& a, const Candidate& b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    };

    if (p.greedy()) {
        // Exactly the argmax the route used before sampling existed.
        return std::min_element(c.begin(), c.end(), better)->id;
    }

    // top_k: keep the k highest logits (llama.cpp llama_sampler_top_k_impl).
    if (p.top_k > 0 && size_t(p.top_k) < c.size()) {
        std::partial_sort(c.begin(), c.begin() + p.top_k, c.end(), better);
        c.resize(size_t(p.top_k));
    }

    // top_p: softmax the survivors into a distribution, then keep the shortest prefix
    // whose mass reaches p, but never fewer than min_keep (llama.cpp
    // llama_sampler_top_p_apply).  The probabilities live in `prob`, not in the
    // candidates: c[i].logit stays a logit, because the next stage divides logits by the
    // temperature (writing probabilities back here would make it divide probabilities,
    // and exp() of a probability in [0,1] is within e^-1 of flat).
    if (p.top_p < 1.0f) {
        std::sort(c.begin(), c.end(), better);
        std::vector<double> prob(c.size());
        double total = 0.0;
        for (size_t i = 0; i < c.size(); i++) {
            prob[i] = std::exp(double(c[i].logit - c[0].logit));
            total += prob[i];
        }
        double run = 0.0;
        size_t keep = c.size();
        for (size_t i = 0; i < c.size(); i++) {
            run += prob[i] / total;
            if (run >= p.top_p && int(i + 1) >= p.min_keep) {
                keep = i + 1;
                break;
            }
        }
        c.resize(keep);
    }

    // temperature: divide the surviving logits, then draw from the cumulative sum of the
    // softmax over logit/temperature (llama.cpp llama_sampler_temp_apply, then
    // llama_sampler_dist_apply).
    float max_l = c.front().logit;
    for (const auto& x : c) max_l = std::max(max_l, x.logit);
    double sum = 0.0;
    for (auto& x : c) {
        x.logit = float(std::exp(double(x.logit - max_l) / double(p.temperature)));
        sum += x.logit;
    }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double target = sum * dist(rng);
    double run = 0.0;
    for (const auto& x : c) {
        run += x.logit;
        if (run >= target) return x.id;
    }
    return c.back().id;
}

// The bytes a decoder must hold back before it can be sure no stop sequence straddles
// the boundary: the longest stop, minus one.
inline size_t stop_holdback(const std::vector<std::string>& stops) {
    size_t longest = 0;
    for (const auto& s : stops) longest = std::max(longest, s.size());
    return longest ? longest - 1 : 0;
}

// Where the earliest complete stop sequence starts in `text`, or npos.  `len` receives
// the matched sequence length.  OpenAI semantics: generation ends at the first stop that
// appears and the text before it is kept.
inline size_t find_stop(const std::string& text, const std::vector<std::string>& stops, size_t* len = nullptr) {
    size_t best = std::string::npos;
    size_t best_len = 0;
    for (const auto& s : stops) {
        if (s.empty()) continue;
        const size_t at = text.find(s);
        if (at == std::string::npos) continue;
        if (best == std::string::npos || at < best || (at == best && s.size() > best_len)) {
            best = at;
            best_len = s.size();
        }
    }
    if (len) *len = best == std::string::npos ? 0 : best_len;
    return best;
}

// What one request asked for, once validated.
struct Options {
    SampleParams sample;
    std::vector<std::string> stop;
    std::string grammar;      // raw GBNF text; empty = unconstrained
    bool json_object = false; // response_format {"type": "json_object"}
};

// Read the decode options out of a request body.  A body that names none of them gets
// the greedy defaults, so the parser stays backward compatible.  Every refusal throws
// std::runtime_error("<field>: <what is wrong>"), which the caller turns into a 400 --
// so all of this is testable without a device.
inline Options options_from_body(const nlohmann::json& body) {
    Options o;
    auto number = [&](const char* key) {
        const auto& v = body.at(key);
        if (!v.is_number() || v.is_boolean()) throw std::runtime_error(std::string(key) + ": must be a number");
        return v.get<double>();
    };
    auto integer = [&](const char* key) {
        const auto& v = body.at(key);
        if (!v.is_number_integer() || v.is_boolean())
            throw std::runtime_error(std::string(key) + ": must be an integer");
        return v.get<long long>();
    };

    if (body.contains("temperature")) {
        const double t = number("temperature");
        if (t < 0.0) throw std::runtime_error("temperature: must be zero or more (zero decodes greedily)");
        o.sample.temperature = float(t);
    }
    if (body.contains("top_k")) {
        const long long k = integer("top_k");
        if (k < 0) throw std::runtime_error("top_k: must be zero or more (zero turns it off)");
        o.sample.top_k = k > (1 << 20) ? (1 << 20) : int(k);
    }
    if (body.contains("top_p")) {
        const double v = number("top_p");
        if (!(v > 0.0 && v <= 1.0)) throw std::runtime_error("top_p: must be above zero and at most 1 (1 turns it off)");
        o.sample.top_p = float(v);
    }
    if (body.contains("seed")) {
        const long long s = integer("seed");
        if (s < 0) throw std::runtime_error("seed: must be zero or more");
        o.sample.seed = uint64_t(s);
    }
    if (body.contains("stop")) {
        const auto& v = body.at("stop");
        std::vector<std::string> stop;
        if (v.is_string()) {
            stop.push_back(v.get<std::string>());
        } else if (v.is_array()) {
            for (const auto& s : v) {
                if (!s.is_string()) throw std::runtime_error("stop: must be a string or an array of strings");
                stop.push_back(s.get<std::string>());
            }
        } else {
            throw std::runtime_error("stop: must be a string or an array of strings");
        }
        for (const auto& s : stop)
            if (s.empty()) throw std::runtime_error("stop: sequences must not be empty");
        if (stop.size() > kMaxStop)
            throw std::runtime_error("stop: at most " + std::to_string(kMaxStop) + " sequences");
        o.stop = std::move(stop);
    }
    if (body.contains("grammar")) {
        const auto& v = body.at("grammar");
        if (!v.is_string()) throw std::runtime_error("grammar: must be a GBNF string");
        o.grammar = v.get<std::string>();
        if (o.grammar.empty()) throw std::runtime_error("grammar: must not be empty");
    }
    if (body.contains("response_format")) {
        const auto& v = body.at("response_format");
        if (!v.is_object()) throw std::runtime_error("response_format: must be an object");
        const std::string type = v.value("type", "text");
        if (type == "json_object") {
            o.json_object = true;
        } else if (type == "text") {
            // unconstrained, the default
        } else if (type == "json_schema") {
            throw std::runtime_error(
                "response_format: json_schema needs a schema-to-GBNF converter this route does not have; "
                "send grammar (GBNF) or type json_object");
        } else {
            throw std::runtime_error("response_format: unknown type '" + type + "'");
        }
    }
    if (o.json_object && !o.grammar.empty())
        throw std::runtime_error("response_format: json_object conflicts with grammar; send one of them");
    return o;
}

}  // namespace onebit::npu
