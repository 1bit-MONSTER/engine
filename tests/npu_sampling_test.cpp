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

// npu_sampling_test: the decode controls `1bit forward-serve` gained in #625 --
// the sampler (app/npu_sampling.h) and the GBNF compiler (app/npu_grammar.h).
// Header only, no device: runs in CI.
#include "npu_grammar.h"
#include "npu_sampling.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using onebit::npu::find_stop;
using onebit::npu::Grammar;
using onebit::npu::json_grammar;
using onebit::npu::kRandomSeed;
using onebit::npu::options_from_body;
using onebit::npu::SampleParams;
using onebit::npu::sample_token;
using onebit::npu::stop_holdback;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (ok) return;
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
}

int sample(const std::vector<float>& logits, const SampleParams& p, uint64_t seed,
           const std::vector<char>* mask = nullptr) {
    std::mt19937_64 rng(seed);
    return sample_token(logits, p, rng, mask);
}

struct Feed {
    bool alive = false;
    bool can_stop = false;
};

// Feed the whole of `text` to a fresh parse of `g`.
Feed run(const Grammar& g, const std::string& text) {
    Grammar::State st = g.start();
    Feed r;
    r.alive = g.feed(st, text);
    r.can_stop = Grammar::can_stop(st);
    return r;
}

std::string error_of(const nlohmann::json& body) {
    try {
        options_from_body(body);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

void test_sampler() {
    const std::vector<float> logits = {0.5f, 2.0f, 1.0f, 2.0f};
    // A body that names no sampling field must decode exactly as the route always did.
    check(sample(logits, SampleParams(), 7) == 1, "greedy picks the highest logit");
    check(sample(logits, SampleParams(), 7) == 1, "greedy keeps the lowest id on a tie");
    SampleParams hot;
    hot.temperature = 1.0f;
    std::mt19937_64 one(3);
    check(sample_token(logits, hot, one) != -1, "sampling returns a token id");
    int seen = 0;
    for (uint64_t s = 0; s < 64; s++)
        if (sample(logits, hot, s) == 3) ++seen;
    check(seen > 0, "temperature 1 can draw the second-highest logit");

    // top_k 1 is the greedy token, whatever the seed.
    SampleParams top1 = hot;
    top1.top_k = 1;
    bool all_argmax = true;
    for (uint64_t s = 0; s < 64; s++) all_argmax = all_argmax && sample(logits, top1, s) == 1;
    check(all_argmax, "top_k 1 decodes greedily");

    // top_k 2 leaves only ids 1 and 3.
    SampleParams top2 = hot;
    top2.top_k = 2;
    bool in_set = true;
    for (uint64_t s = 0; s < 64; s++) {
        const int id = sample(logits, top2, s);
        in_set = in_set && (id == 1 || id == 3);
    }
    check(in_set, "top_k 2 keeps only the two highest logits");

    // top_p 0.5 on a peaked distribution keeps just the head.
    const std::vector<float> peaked = {6.0f, 0.0f, 0.0f};
    SampleParams narrow = hot;
    narrow.top_p = 0.5f;
    bool head_only = true;
    for (uint64_t s = 0; s < 64; s++) head_only = head_only && sample(peaked, narrow, s) == 0;
    check(head_only, "top_p 0.5 keeps only the head of a peaked distribution");

    // Equal logits spread over the whole set with no filtering.
    const std::vector<float> flat = {0.0f, 0.0f, 0.0f};
    int hits[3] = {0, 0, 0};
    std::mt19937_64 rng(1234);
    for (int i = 0; i < 60; i++) hits[sample_token(flat, hot, rng)]++;
    check(hits[0] > 0 && hits[1] > 0 && hits[2] > 0, "temperature 1 spreads over equal logits");

    // The same seed draws the same sequence: that is what makes a run reproducible.
    auto sequence = [&](uint64_t seed) {
        std::mt19937_64 r(seed);
        std::string out;
        for (int i = 0; i < 12; i++) out += char('a' + sample_token(flat, hot, r));
        return out;
    };
    check(sequence(7) == sequence(7), "the same seed draws the same tokens");
    check(sequence(7) != sequence(8), "a different seed draws a different sequence");

    // A grammar mask is the only thing that may be drawn from.
    const std::vector<float> three = {0.0f, 5.0f, 0.0f};
    std::vector<char> mask(3, 0);
    mask[2] = 1;
    bool masked = true;
    for (uint64_t s = 0; s < 32; s++) masked = masked && sample(three, hot, s, &mask) == 2;
    check(masked, "a masked token is the only candidate");
    std::vector<char> none(3, 0);
    check(sample(three, hot, 1, &none) == -1, "an empty mask is refused");
    check(sample(three, SampleParams(), 1, &mask) == 2, "greedy still honors the mask");
}

// The draw has to match the distribution the logits define.  Temperature divides logits
// and never probabilities: when the top_p stage wrote probabilities back over the logits,
// the draw ran on exp(probability), which is flat to within e^-1, so a token llama.cpp
// gives 4.6e-5 was drawn about one time in five instead of about once in 22000.
// Every number below is hand-computed softmax(logit / temperature).
void test_sampler_distribution() {
    // softmax({10, 0, 0}) = {0.99990857, 0.00004572, 0.00004572}: the tail's 45.7 ppm is
    // the reviewer's token.  20000 draws expect 0.91 of each tail token, so a correct
    // draw sees a handful at most; the old chain's ~0.21 saw ~4200.
    {
        const std::vector<float> peaked = {10.0f, 0.0f, 0.0f};
        SampleParams p;
        p.temperature = 1.0f;
        p.top_p = 0.5f;   // keeps the head alone...
        p.min_keep = 3;   // ...except that min_keep keeps all three, so the tail is drawable
        std::mt19937_64 rng(20261029);
        int hits[3] = {0, 0, 0};
        for (int i = 0; i < 20000; i++) hits[sample_token(peaked, p, rng)]++;
        check(hits[0] >= 19990, "a 0.99991 token is drawn almost every time");
        check(hits[1] <= 10 && hits[2] <= 10, "a 4.6e-5 token is drawn at 4.6e-5, not at 0.21");
    }

    // softmax({ln 3, 0, 0}) = {0.6, 0.2, 0.2}: 12000 and 4000 out of 20000, +-300
    // (~5.3 sigma of a 0.2 binomial over 20000 draws).
    {
        const std::vector<float> three = {1.0986123f, 0.0f, 0.0f};  // ln 3
        SampleParams p;
        p.temperature = 1.0f;
        p.top_p = 0.95f;
        p.min_keep = 3;
        std::mt19937_64 rng(625);
        int hits[3] = {0, 0, 0};
        for (int i = 0; i < 20000; i++) hits[sample_token(three, p, rng)]++;
        check(std::abs(hits[0] - 12000) <= 300, "a 0.6 token is drawn 0.6 of the time");
        check(std::abs(hits[1] - 4000) <= 300 && std::abs(hits[2] - 4000) <= 300,
              "a 0.2 token is drawn 0.2 of the time");
    }

    // Temperature scales the logits: at temperature 0.5, {ln 3, 0, 0} is
    // softmax({2 ln 3, 0, 0}) = {9/11, 1/11, 1/11} = {0.818, 0.0909, 0.0909}.
    {
        const std::vector<float> three = {1.0986123f, 0.0f, 0.0f};
        SampleParams p;
        p.temperature = 0.5f;  // top_p stays off: this case is the temperature stage alone
        std::mt19937_64 rng(42);
        int hits[3] = {0, 0, 0};
        for (int i = 0; i < 20000; i++) hits[sample_token(three, p, rng)]++;
        check(std::abs(hits[0] - 16364) <= 300, "temperature 0.5 sharpens 0.6 into 9/11");
        check(std::abs(hits[1] - 1818) <= 250 && std::abs(hits[2] - 1818) <= 250,
              "temperature 0.5 thins 0.2 into 1/11");
    }
}

void test_stop() {
    check(stop_holdback({}) == 0, "no stop sequences hold nothing back");
    check(stop_holdback({"abc"}) == 2, "a stop sequence holds back all but its first byte");
    check(stop_holdback({"a", "abcd"}) == 3, "the longest stop decides the holdback");
    size_t len = 0;
    check(find_stop("hello<|im_end|>world", {"<|im_end|>"}, &len) == 5 && len == 10, "find_stop locates the stop");
    check(find_stop("abc", {"zz"}) == std::string::npos, "find_stop finds nothing when there is no stop");
    check(find_stop("", {"a"}) == std::string::npos, "find_stop on empty text");
    check(find_stop("xSTOPy", {}) == std::string::npos, "no stop sequences means no stop");
    // The earliest stop wins, and at the same place the longer one is reported.
    len = 0;
    check(find_stop("aQQb", {"QQb", "QQ"}, &len) == 1 && len == 3, "the earliest stop wins, longest at that spot");
}

void test_json_grammar() {
    const Grammar g(json_grammar());
    check(run(g, "").alive && !run(g, "").can_stop, "the JSON grammar wants an object, not nothing");

    const std::string good = R"({"a": [1, true, "x"]})";
    // Every proper prefix of valid JSON stays alive, and only the whole thing may stop.
    bool prefixes_alive = true;
    for (size_t i = 1; i < good.size(); i++) prefixes_alive = prefixes_alive && run(g, good.substr(0, i)).alive;
    check(prefixes_alive, "every prefix of valid JSON stays inside the grammar");
    for (size_t i = 1; i + 1 < good.size(); i++)
        if (run(g, good.substr(0, i)).can_stop) {
            check(false, "no proper prefix of " + good + " may end the answer");
            break;
        }
    const Feed full = run(g, good);
    check(full.alive && full.can_stop, "valid JSON is accepted and may stop");

    check(run(g, R"({"a": 1} )").can_stop, "trailing space is part of the grammar");
    check(!run(g, "[]").alive, "an array is rejected: root is an object");
    check(run(g, R"({"a": })").alive == false, "a missing value is rejected");
    check(run(g, R"({"a" 1})").alive == false, "a missing colon is rejected");
    // A '}' inside a JSON string is just a byte of it: this is a prefix, not an error.
    check(run(g, R"({"a": "unterminated})").alive && !run(g, R"({"a": "unterminated})").can_stop,
          "an unterminated string is a prefix, not an accept");
    check(run(g, R"({"n": -1.5e3})").can_stop, "numbers carry sign, fraction and exponent");
    check(run(g, R"({"u": "\u00e9"})").can_stop, "string escapes are part of the grammar");

    // A token that would leave the grammar is refused, and the state after a good token
    // is the one the next token is judged against.
    Grammar::State st = g.start();
    check(!g.feed(st, "x"), "a token outside the grammar is refused");
    check(g.feed(st, "{"), "an opening brace is accepted");
    check(g.feed(st, R"("a")"), "a key is accepted");
    Grammar::State copy = st;
    check(!g.feed(copy, "x"), "a colon is required after a key");
    check(st.stacks.size() > 0, "the refused token left the state where it was");
}

void test_gbnf() {
    {
        const Grammar g(R"(root ::= "yes" | "no")");
        check(run(g, "yes").can_stop && run(g, "no").can_stop, "an alternation accepts both branches");
        check(run(g, "y").alive && !run(g, "y").can_stop, "a partial branch is alive but may not stop");
        check(!run(g, "yesx").alive, "text past the end of the grammar is refused");
        check(!run(g, "").can_stop, "an empty answer may not stop before any alternative matched");
    }
    {
        const Grammar g(R"(root ::= [a-z]+ "!")");
        check(run(g, "abc!").can_stop, "a char class with a repetition accepts its language");
        check(run(g, "ab").alive && !run(g, "ab").can_stop, "a repetition in progress is alive");
        check(!run(g, "abc").alive || !run(g, "abc").can_stop, "the trailing literal is required");
        check(!run(g, "123!").alive, "a char class refuses what it does not list");
        check(!run(g, "!").alive, "'+' needs at least one match");
    }
    {
        const Grammar g(R"(root ::= "a"{2,3})");
        check(run(g, "aa").can_stop && run(g, "aaa").can_stop, "a bounded repetition accepts the range");
        check(!run(g, "a").can_stop, "a bounded repetition needs its minimum");
        check(!run(g, "aaaa").alive, "a bounded repetition stops at its maximum");
    }
    {
        const Grammar g(R"(root ::= [^a-c] "z")");
        check(run(g, "dz").can_stop, "a negated char class accepts what it does not list");
        check(!run(g, "bz").alive, "a negated char class refuses what it lists");
    }
    {
        const Grammar g("# a comment\nroot ::= \"a\" |\t\"b\"  # trailing\n");
        check(run(g, "b").can_stop, "comments and whitespace are skipped");
    }
    {
        const Grammar g("root ::= \"é\"");
        check(run(g, "\xc3\xa9").can_stop, "a non-ASCII literal is matched as its UTF-8 bytes");
        check(run(g, "\xc3").alive && !run(g, "\xc3").can_stop,
              "half of a two-byte character is a prefix, not an accept");
        check(!run(g, "\xc3\x28").alive, "a wrong continuation byte kills the match");
    }
    {
        // Greedy repetition nests: a stack that grows with the input is fine.
        const Grammar g(R"(root ::= "a"* "b")");
        check(run(g, std::string(200, 'a') + "b").can_stop, "a long repetition still terminates");
    }

    // Malformed grammars are client errors with a line number, not crashes.
    auto bad = [](const std::string& text) {
        try {
            Grammar g(text);
        } catch (const std::runtime_error& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string quote = bad(R"(root ::= 'x')");
    check(!quote.empty() && quote.rfind("gbnf:", 0) == 0 && quote.find("line ") != std::string::npos,
          "a single-quoted literal is reported as gbnf: ... at line N");
    check(!bad(R"(nope ::= "x")").empty(), "a grammar without root is refused");
    check(!bad(R"(root ::= other)").empty(), "an undefined rule is refused");
    check(!bad(R"(root ::= "abc)").empty(), "an unterminated literal is refused");
    check(!bad(R"(root ::= [abc)").empty(), "an unterminated char class is refused");
    check(!bad(R"(root ::= "a"{3000})").empty(), "an absurd repetition count is refused");
}

void test_request_parsing() {
    const auto plain = options_from_body(nlohmann::json::parse(R"({"prompt": "hi", "max_tokens": 8})"));
    check(plain.sample.greedy(), "a body without sampling fields stays greedy");
    check(plain.sample.top_k == 0 && plain.sample.top_p == 1.0f, "no top_k and no top_p by default");
    check(plain.sample.seed == kRandomSeed, "no seed by default");
    check(plain.stop.empty() && plain.grammar.empty() && !plain.json_object, "no stop, grammar or JSON mode by default");

    const auto full = options_from_body(nlohmann::json::parse(
        R"({"prompt": "hi", "temperature": 0.8, "top_p": 0.9, "top_k": 40, "seed": 1234,
            "stop": ["\n\n", "<|im_end|>"], "grammar": "root ::= [0-9]+"})"));
    check(full.sample.temperature == 0.8f && full.sample.top_p == 0.9f && full.sample.top_k == 40, "sampling fields");
    check(full.sample.seed == 1234, "the seed is the one the client sent");
    check(full.stop.size() == 2 && full.stop[0] == "\n\n" && full.stop[1] == "<|im_end|>", "stop sequences");
    check(full.grammar == "root ::= [0-9]+", "the GBNF text is passed through");

    check(options_from_body(nlohmann::json::parse(R"({"stop": "END"})")).stop.size() == 1, "a bare string stop");
    check(options_from_body(nlohmann::json::parse(R"({"temperature": 0, "top_k": 5})")).sample.greedy(),
          "temperature 0 wins over top_k, as on the 1bit serve route");
    check(options_from_body(nlohmann::json::parse(R"({"response_format": {"type": "json_object"}})")).json_object,
          "response_format json_object is honoured");
    check(options_from_body(nlohmann::json::parse(R"({"response_format": {"type": "text"}})")).grammar.empty(),
          "response_format text is unconstrained");

    auto starts_with = [](const std::string& s, const std::string& prefix) { return s.rfind(prefix, 0) == 0; };
    check(starts_with(error_of(nlohmann::json::parse(R"({"temperature": -1})")), "temperature:"), "a negative temperature is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"top_k": -2})")), "top_k:"), "a negative top_k is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"top_p": 0})")), "top_p:"), "top_p 0 is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"top_p": 1.5})")), "top_p:"), "top_p above 1 is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"seed": "x"})")), "seed:"), "a non-integer seed is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"seed": -1})")), "seed:"), "a negative seed is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"stop": ""})")), "stop:"), "an empty stop sequence is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"stop": ["a","b","c","d","e"]})")), "stop:"),
          "more stop sequences than the limit is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"grammar": ""})")), "grammar:"), "an empty grammar is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"grammar": 7})")), "grammar:"), "a non-string grammar is refused");
    check(error_of(nlohmann::json::parse(R"({"response_format": {"type": "json_schema"}})")).find("grammar") !=
              std::string::npos,
          "json_schema points at grammar (GBNF), the converter it does not have");
    check(error_of(nlohmann::json::parse(R"({"response_format": {"type": "nope"}})")).find("nope") != std::string::npos,
          "an unknown response_format type is refused");
    check(!error_of(nlohmann::json::parse(
                        R"({"grammar": "root ::= \"a\"", "response_format": {"type": "json_object"}})"))
               .empty(),
          "grammar and json_object together are refused");
    // Nothing else in the body is inspected: the parser stays backward compatible.
    check(error_of(nlohmann::json::parse(R"({"prompt": "hi", "n": 3, "logprobs": true})")).empty(),
          "fields this route does not implement are still ignored");
}

}  // namespace

int main() {
    test_sampler();
    test_sampler_distribution();
    test_stop();
    test_json_grammar();
    test_gbnf();
    test_request_parsing();
    std::printf("%s\n", failures ? "FAILED" : "npu_sampling: ok");
    return failures ? 1 : 0;
}
