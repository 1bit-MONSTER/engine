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

#include "npu_json_schema.h"
#include "npu_tool_grammar.h"

namespace onebit::npu {

// Sentinel for "no seed given": the run picks one, like llama.cpp's LLAMA_DEFAULT_SEED.
inline constexpr uint64_t kRandomSeed = ~uint64_t(0);

// At most this many stop sequences, as in the OpenAI API.
inline constexpr size_t kMaxStop = 4;

// OpenAI's cap on `top_logprobs` is 20; FLM serves 0..100, and a decoder with the logits in hand
// pays nothing for the wider range, so this route takes FLM's number.
inline constexpr int kMaxTopLogprobs = 100;

struct SampleParams {
    float temperature = 0.0f;  // 0 = greedy (argmax), this route's historical decode
    int top_k = 0;             // 0 = off
    float top_p = 1.0f;        // 1 = off
    int min_keep = 1;          // top_p keeps at least this many candidates
    uint64_t seed = kRandomSeed;
    bool logprobs = false;     // report each token's log-probability (chat completions only)
    int top_logprobs = 0;      // how many alternatives per token, 0 = none, at most kMaxTopLogprobs

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

// How many leading ids of a logits row are tokens: the tokenizer's vocabulary when it is the
// smaller of the two.  The head's width is the container's `vocab_size`, which may exceed the
// tokenizer's (Qwen3-0.6B: 151,936 for 151,669 ids); the ids in between are padding positions with
// no text, and a pick that lands there emits nothing and never ends the reply (engine#1182,
// #2017).  `vocab` <= 0 means "no tokenizer to bound by" and keeps the whole row.
inline size_t token_bound(size_t width, int vocab) {
    return (vocab > 0 && size_t(vocab) < width) ? size_t(vocab) : width;
}

// Draw one token from `logits`.  `allowed`, when given, is a per-token mask (non-zero =
// the grammar accepts the token); a mask with nothing set is refused with -1 and the
// caller reports it rather than decoding an unconstrained token.  `vocab` is the
// tokenizer's size (token_bound): ids at or past it are never drawn, with or without a mask.
inline int sample_token(const std::vector<float>& logits, const SampleParams& p, std::mt19937_64& rng,
                        const std::vector<char>* allowed = nullptr, int vocab = 0) {
    const size_t n = token_bound(logits.size(), vocab);
    std::vector<Candidate> c;
    c.reserve(n);
    for (size_t i = 0; i < n; i++) {
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

// The greedy id of one logits row of `width` floats, bounded like sample_token(): a higher
// logit wins and a tie keeps the lower id.  `1bit npu-run` prints this per row.
inline int argmax_token(const float* row, size_t width, int vocab = 0) {
    const size_t n = token_bound(width, vocab);
    int best = 0;
    for (size_t v = 1; v < n; v++)
        if (row[v] > row[size_t(best)]) best = int(v);
    return best;
}

// One candidate of the raw distribution, as the client reads it: the id and its log-probability.
struct Logprob {
    int id = 0;
    float logprob = 0.0f;
};

// What one generated token's entry carries: the id that was drawn, its log-probability, and the
// `top_logprobs` most likely ids.  `top` need not contain `id`: the draw happens inside the
// grammar's mask while these values describe the raw distribution (see logprobs_of).
struct TokenLogprob {
    int id = 0;
    float logprob = 0.0f;
    std::vector<Logprob> top;
};

// The log-probabilities of `logits` over the whole vocabulary, as OpenAI and vLLM report them:
// the values of the distribution *before* top_k, top_p, temperature and any grammar mask, which
// is why this is computed from the untouched logits rather than inside the sampler's filtered
// candidate set.  Top entries are the highest raw logits, with the sampler's tie rule (lower id
// first), so the same seed and request always report the same list.
// `vocab` is the tokenizer's bound and MUST be passed whenever `logits` may be wider than it: a
// padded head stores rows past the vocabulary (engine#2026 -- Qwen3: 151,936 rows against 151,669
// ids), and those rows are not tokens. Normalising over them shifts every reported logprob by the
// padding's mass, which is model-dependent rather than a constant, and invisible to any comparison
// between two paths that share this helper (engine#2272). The bound also keeps padding ids out of
// the `top_logprobs` list, where an id the tokenizer does not have would otherwise appear.
inline TokenLogprob logprobs_of(const std::vector<float>& logits, int vocab, int chosen, int top_n) {
    TokenLogprob out;
    out.id = chosen;
    if (logits.empty()) return out;
    const size_t n = token_bound(logits.size(), vocab);
    if (n == 0) return out;
    // log-sum-exp in double: exp() of a float logit and then log() of the sum loses the tail that
    // top_logprobs is read for.
    float max_l = logits[0];
    for (size_t i = 0; i < n; i++) max_l = std::max(max_l, logits[i]);
    if (!std::isfinite(max_l)) return out;  // a vocabulary of infinities reports nothing meaningful
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) sum += std::exp(double(logits[i]) - double(max_l));
    const double log_z = double(max_l) + std::log(sum);
    if (!std::isfinite(log_z)) return out;
    if (chosen >= 0 && size_t(chosen) < n) out.logprob = float(double(logits[size_t(chosen)]) - log_z);
    if (top_n <= 0) return out;
    std::vector<Candidate> c;
    c.reserve(n);
    for (size_t i = 0; i < n; i++) c.push_back({logits[i], int(i)});
    auto better = [](const Candidate& a, const Candidate& b) {
        return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
    };
    const size_t k = std::min(size_t(top_n), c.size());
    std::partial_sort(c.begin(), c.begin() + k, c.end(), better);
    out.top.reserve(k);
    for (size_t i = 0; i < k; i++) out.top.push_back({c[i].id, float(double(c[i].logit) - log_z)});
    return out;
}

// For a row that is exactly the vocabulary (no padding ids). A padded row must use the overload
// above with the tokenizer's bound; this form is what a test or a non-padded model wants.
inline TokenLogprob logprobs_of(const std::vector<float>& logits, int chosen, int top_n) {
    return logprobs_of(logits, int(logits.size()), chosen, top_n);
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
    // (issue #652 item 3) Whether the request carried `tools` decides whether <tool_call> markup in
    // the answer is a call or text the caller asked for, so the answer parser has to know.
    bool tools_sent = false;
    // tools / tool_choice (#652 item 2): the grammar for tool_choice "required" and the
    // {"type": "function"} form; empty for "auto", for "none", and when no tools were sent.
    bool tool_grammar = false;  // the grammar above came from tools[], not from the client
};

// A top-level integer field: absent (or null) is the default; anything that is not a number is a 400 naming
// the field.  These two exist because `body.value(key, def)` on a wrong type throws nlohmann::json::type_error,
// which is not the route's refusal type: read outside the handler's guard it escaped and answered **500 with an
// empty body** (measured on engine#1106: `{"max_tokens": "8"}` -> 500, `{"stream": "yes"}` -> 500), i.e. a
// client fault that looks like a server fault.  Inside the guard it is an ordinary 400.
inline int int_field(const nlohmann::json& body, const char* key, int def) {
    if (!body.contains(key) || body.at(key).is_null()) return def;
    const auto& v = body.at(key);
    if (v.is_number_float()) return int(v.get<double>());  // a fractional cap is truncated, as body.value did
    if (!v.is_number_integer()) throw std::runtime_error(std::string(key) + ": must be an integer");
    return v.get<int>();
}

inline bool bool_field(const nlohmann::json& body, const char* key, bool def = false) {
    if (!body.contains(key) || body.at(key).is_null()) return def;
    const auto& v = body.at(key);
    if (!v.is_boolean()) throw std::runtime_error(std::string(key) + ": must be true or false");
    return v.get<bool>();
}

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
    if (body.contains("logprobs")) {
        const auto& v = body.at("logprobs");
        if (!v.is_boolean()) throw std::runtime_error("logprobs: must be true or false");
        o.sample.logprobs = v.get<bool>();
    }
    if (body.contains("top_logprobs")) {
        const long long n = integer("top_logprobs");
        if (n < 0 || n > kMaxTopLogprobs)
            throw std::runtime_error("top_logprobs: must be 0 to " + std::to_string(kMaxTopLogprobs));
        o.sample.top_logprobs = int(n);
    }
    // OpenAI refuses this too, and the alternative is a field that silently does nothing.
    if (o.sample.top_logprobs > 0 && !o.sample.logprobs)
        throw std::runtime_error("top_logprobs: needs logprobs: true");
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
    bool client_grammar = false;
    // (issue #652 item 3) recorded before the grammar branches, because the answer parser needs it
    // whether or not a grammar was installed.
    if (body.contains("grammar")) {
        const auto& v = body.at("grammar");
        if (!v.is_string()) throw std::runtime_error("grammar: must be a GBNF string");
        o.grammar = v.get<std::string>();
        if (o.grammar.empty()) throw std::runtime_error("grammar: must not be empty");
        client_grammar = true;
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
            // OpenAI's structured-outputs envelope: the schema is the JSON Schema to enforce, `name`
            // and `strict` describe the caller's intent and are implied by generating inside the
            // grammar.  A schema this converter cannot honour is a 400 naming what it refused.
            if (client_grammar)
                throw std::runtime_error("response_format: json_schema conflicts with grammar; send one of them");
            if (!v.contains("json_schema")) throw std::runtime_error("response_format: json_schema needs a json_schema object");
            const auto& envelope = v.at("json_schema");
            if (!envelope.is_object()) throw std::runtime_error("response_format: json_schema must be an object");
            if (!envelope.contains("schema"))
                throw std::runtime_error("response_format: json_schema needs a json_schema.schema");
            o.grammar = json_schema_to_gbnf(envelope.at("schema"));
        } else {
            throw std::runtime_error("response_format: unknown type '" + type + "'");
        }
    }
    if (body.contains("tools")) {
        o.tools_sent = true;  // the answer parser's rule (issue #652 item 3) needs this either way
        if (client_grammar)
            throw std::runtime_error("tools: conflicts with grammar; send one of them");
        if (body.contains("response_format"))
            throw std::runtime_error("tools: conflicts with response_format; send one of them");
        bool parallel = true;
        if (body.contains("parallel_tool_calls")) {
            const auto& v = body.at("parallel_tool_calls");
            if (!v.is_boolean()) throw std::runtime_error("parallel_tool_calls: must be true or false");
            parallel = v.get<bool>();
        }
        const nlohmann::json choice = body.contains("tool_choice") ? body.at("tool_choice") : nlohmann::json("auto");
        std::string g = tool_call_gbnf(body.at("tools"), choice, parallel);
        if (!g.empty()) {
            o.grammar = std::move(g);
            o.tool_grammar = true;
        }
        // "auto" and "none" install no grammar on purpose (app/npu_tool_grammar.h says why), so an
        // empty result is not an error: the answer is unconstrained and the caller validates it.
    } else if (body.contains("tool_choice") || body.contains("parallel_tool_calls")) {
        throw std::runtime_error("tool_choice: needs tools; send the functions the answer may call");
    }

    if (o.json_object && client_grammar)
        throw std::runtime_error("response_format: json_object conflicts with grammar; send one of them");
    return o;
}

}  // namespace onebit::npu
