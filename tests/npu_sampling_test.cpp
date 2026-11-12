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
#include "npu_answer_parser.h"
#include "npu_grammar.h"
#include "npu_prefix_cache.h"
#include "npu_sampling.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using onebit::npu::find_stop;
using onebit::npu::allow_end_tokens;
using onebit::npu::Grammar;
using onebit::npu::AnswerFamily;
using onebit::npu::AnswerStream;
using onebit::npu::content_mask;
using onebit::npu::json_grammar;
using onebit::npu::parse_answer;
using onebit::npu::tool_call_gbnf;
using onebit::npu::kRandomSeed;
using onebit::npu::PrefixOptions;
using onebit::npu::PrefixCache;
using onebit::npu::json_schema_to_gbnf;
using onebit::npu::int_field;      // engine#1106: typed top-level fields
using onebit::npu::bool_field;
using onebit::npu::logprobs_of;
using onebit::npu::options_from_body;
using onebit::npu::SampleParams;
using onebit::npu::sample_token;
using onebit::npu::argmax_token;
using onebit::npu::token_bound;
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

// engine#1106: a typed top-level field is a named refusal, not a json::type_error that escapes the route's guard.
template <typename F>
std::string throws_of(F f) {
    try {
        f();
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

std::string error_of(const nlohmann::json& body) {
    try {
        options_from_body(body);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

// The converter's refusal for a schema, or "" when it compiles.
std::string schema_error(const std::string& schema_json) {
    try {
        json_schema_to_gbnf(nlohmann::json::parse(schema_json));
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

// sample_probs() is what sample_token() draws from (engine#1781: a speculative round checks drafts against it, so a
// mismatch would bias a sampled reply): for each set of parameters the draw's frequencies over 200000 seeded draws
// sit within 0.01 total variation of sample_probs(), every id it zeroes is never drawn, the row sums to 1, ids past
// the tokenizer's bound get no entry, and greedy parameters give the argmax's one-hot.
void test_sample_probs() {
    const std::vector<float> lg = {1.0f, 2.5f, 0.3f, 2.4f, -1.0f, 0.9f, 1.7f, 0.0f, 2.0f, -0.5f, 3.0f, 2.9f};
    struct Case { const char* name; float t; int k; float tp; };
    const Case cases[] = {{"temperature 0.8", 0.8f, 0, 1.0f},
                          {"temperature 1.3, top_k 4", 1.3f, 4, 1.0f},
                          {"temperature 0.7, top_p 0.8", 0.7f, 0, 0.8f},
                          {"temperature 1, top_k 6, top_p 0.9", 1.0f, 6, 0.9f}};
    for (const Case& cs : cases) {
        SampleParams p;
        p.temperature = cs.t;
        p.top_k = cs.k;
        p.top_p = cs.tp;
        const int bound = 10;   // the last two ids (the two largest logits but one) are padding
        const std::vector<float> q = onebit::npu::sample_probs(lg.data(), lg.size(), p, bound);
        double total = 0;
        for (float v : q) total += v;
        std::vector<double> f(lg.size(), 0.0);
        std::mt19937_64 rng(1781);
        const int N = 200000;
        for (int i = 0; i < N; i++) f[size_t(sample_token(lg, p, rng, nullptr, bound))] += 1.0 / N;
        double tv = 0;
        bool zero_never = true;
        for (size_t i = 0; i < q.size(); i++) {
            tv += std::fabs(f[i] - q[i]);
            if (q[i] == 0.0f && f[i] > 0.0) zero_never = false;
        }
        tv /= 2;
        std::printf("    sample_probs %-36s tv %.4f\n", cs.name, tv);
        check(q.size() == size_t(bound) && std::fabs(total - 1.0) < 1e-5,
              std::string("sample_probs covers the bounded row and sums to 1: ") + cs.name);
        check(tv < 0.01 && zero_never, std::string("sample_token draws what sample_probs says: ") + cs.name);
    }
    const std::vector<float> g = onebit::npu::sample_probs(lg.data(), lg.size(), SampleParams(), 10);
    check(g.size() == 10 && g[1] == 1.0f, "sample_probs at greedy is the bounded argmax's one-hot");
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

// engine#1182: a mask built only from grammar-legal tokens drops the end token, so a constrained
// answer could never stop and padded to max_tokens with finish_reason "length".
void test_grammar_end_tokens() {
    const std::vector<int> ends = {151645, 151643, -1};  // Qwen's <|im_end|>, <|endoftext|>, and a miss
    {
        // `root ::= "ab"`: mid-grammar there is no stopping point, so every end token stays masked.
        const Grammar g(R"(root ::= "ab")");
        Grammar::State st = g.start();
        std::vector<char> allowed(200000, 0);
        allow_end_tokens(allowed, st, ends);
        check(allowed[151645] == 0 && allowed[151643] == 0,
              "EOS is masked before the grammar can end, so a constrained answer cannot be truncated");
        check(g.feed(st, "ab"), "the grammar's own text is accepted");
        allow_end_tokens(allowed, st, ends);
        check(allowed[151645] == 1 && allowed[151643] == 1,
              "EOS is offered once the grammar is at an accepting state");
        check(allowed[0] == 0 && allowed[151644] == 0, "only the end tokens are added, nothing else");
    }
    {
        // The real case: the house JSON grammar after a complete object, where the reply must be
        // able to stop even though more properties would still be legal.
        const Grammar g(json_grammar());
        Grammar::State st = g.start();
        check(g.feed(st, R"({"a": 1})"), "a complete object is accepted");
        check(Grammar::can_stop(st), "a complete object is an accepting state");
        std::vector<char> allowed(200000, 0);
        allow_end_tokens(allowed, st, ends);
        check(allowed[151645] == 1, "json_schema/json_object replies may stop on EOS when the JSON is whole");
    }
    {
        // An id outside the mask (a vocabulary smaller than the end ids) is ignored, not written.
        const Grammar g(R"(root ::= "ab")");
        Grammar::State st = g.start();
        g.feed(st, "ab");
        std::vector<char> small(8, 0);
        allow_end_tokens(small, st, ends);
        int set = 0;
        for (char c : small) set += c ? 1 : 0;
        check(set == 0, "an end token outside the mask vector is ignored rather than written out of bounds");
    }
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

void test_logprobs() {
    // Two candidates, logits 0 and ln 2: probabilities 1/3 and 2/3.
    const double ln2 = std::log(2.0);
    const std::vector<float> logits = {0.0f, float(ln2)};
    const auto lp = logprobs_of(logits, 1, 2);
    check(lp.id == 1, "logprobs report the id that was drawn");
    check(std::fabs(lp.logprob - float(std::log(2.0 / 3.0))) < 1e-6f,
          "the drawn token's log-probability is log 2/3");
    check(lp.top.size() == 2 && lp.top[0].id == 1 && lp.top[1].id == 0,
          "top_logprobs are ordered by raw logit, highest first");
    check(std::fabs(lp.top[0].logprob - float(std::log(2.0 / 3.0))) < 1e-6f, "the top entry carries log 2/3");
    check(std::fabs(lp.top[1].logprob - float(std::log(1.0 / 3.0))) < 1e-6f, "the second entry carries log 1/3");

    // A flat vocabulary: every token is -log(N), and top_logprobs 0 asks for no alternatives.
    const auto flat = logprobs_of(std::vector<float>(8, 1.5f), 3, 0);
    check(std::fabs(flat.logprob - float(std::log(1.0 / 8.0))) < 1e-6f, "a flat vocabulary gives -log N");
    check(flat.top.empty(), "top_logprobs 0 reports no alternatives");

    // Ties go to the lower id, the rule the sampler draws by.
    const auto tied = logprobs_of(std::vector<float>{1.0f, 1.0f, 1.0f}, 0, 2);
    check(tied.top.size() == 2 && tied.top[0].id == 0 && tied.top[1].id == 1, "ties keep the lower id first");

    // engine#2272: a head WIDER than the tokenizer (Qwen3: 151,669 ids against 151,936 rows). The row carries logits for
    // ids the vocabulary does not have; they must not join the normaliser, and they must not be nameable in `top`.
    {
        const std::vector<float> padded = {1.0f, 2.0f, 0.5f, /* padding: */ 9.0f, 9.0f, 9.0f};
        const int vocab = 3;
        const auto bounded = logprobs_of(padded.data(), token_bound(padded.size(), vocab), 1, 2);
        const auto unbounded = logprobs_of(padded.data(), padded.size(), 1, 2);
        const double m = 2.0, lz = m + std::log(std::exp(1.0 - m) + 1.0 + std::exp(0.5 - m));
        check(std::fabs(double(bounded.logprob) - (2.0 - lz)) < 1e-6,
              "a padded row's logprob is the softmax over the VOCABULARY, not over the padding");
        check(std::fabs(double(unbounded.logprob) - double(bounded.logprob)) > 1e-3,
              "...which differs from the unbounded computation, so this was visible, not cosmetic");
        check(bounded.top.size() == 2 && bounded.top[0].id == 1 && bounded.top[1].id == 0,
              "...and `top` names only real ids, in the sampler's tie order");
        bool padding_in_top = false;
        for (const auto& e : bounded.top) padding_in_top = padding_in_top || e.id >= vocab;
        check(!padding_in_top, "...no padding id (3..5) appears in top_logprobs");
        // The vector overload forwards with logits.size(), so it cannot bound: a caller that reaches for the convenient
        // form replays engine#2272 exactly. Pinned here so the trap is visible in the tests rather than discovered again.
        check(std::fabs(double(logprobs_of(padded, 1, 2).logprob) - double(unbounded.logprob)) < 1e-9,
              "the vector overload is UNBOUNDED by construction -- callers with a padded head must pass the bound");
    }

    // More alternatives than the vocabulary reports what exists.
    check(logprobs_of(logits, 1, 100).top.size() == 2, "top_n past the vocabulary reports all of it");

    // The grammar mask can put the drawn token outside the top list, and it still carries its own
    // value: this is the "before penalties and constraints" convention, measured.
    const auto masked = logprobs_of(std::vector<float>{5.0f, 0.0f}, 1, 1);
    check(masked.top.size() == 1 && masked.top[0].id == 0, "top_logprobs is the raw distribution, mask-free");
    check(masked.logprob < masked.top[0].logprob, "a masked-in token reports its own, lower value");

    // Degenerate inputs are reported, never fatal.
    check(logprobs_of({}, 0, 4).top.empty(), "an empty vocabulary reports nothing");
    check(logprobs_of(logits, -1, 1).logprob == 0.0f, "an id outside the vocabulary reports zero");
}

void test_logprob_options() {
    auto starts_with = [](const std::string& s, const std::string& prefix) { return s.rfind(prefix, 0) == 0; };

    const auto off = options_from_body(nlohmann::json::parse(R"({"prompt": "hi"})"));
    check(!off.sample.logprobs && off.sample.top_logprobs == 0, "logprobs is off by default");

    const auto on = options_from_body(nlohmann::json::parse(R"({"logprobs": true, "top_logprobs": 5})"));
    check(on.sample.logprobs && on.sample.top_logprobs == 5, "logprobs and top_logprobs are read");

    check(options_from_body(nlohmann::json::parse(R"({"logprobs": true})")).sample.top_logprobs == 0,
          "logprobs without top_logprobs asks for no alternatives");
    check(options_from_body(nlohmann::json::parse(R"({"logprobs": true, "top_logprobs": 100})")).sample.top_logprobs == 100,
          "top_logprobs 100 is the top of the range and is accepted");

    check(starts_with(error_of(nlohmann::json::parse(R"({"logprobs": 1})")), "logprobs:"),
          "logprobs must be true or false, not a number");
    check(starts_with(error_of(nlohmann::json::parse(R"({"logprobs": true, "top_logprobs": 101})")), "top_logprobs:"),
          "top_logprobs above 100 is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"logprobs": true, "top_logprobs": -1})")), "top_logprobs:"),
          "a negative top_logprobs is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"logprobs": true, "top_logprobs": 2.5})")), "top_logprobs:"),
          "a fractional top_logprobs is refused");
    check(starts_with(error_of(nlohmann::json::parse(R"({"top_logprobs": 3})")), "top_logprobs:"),
          "top_logprobs without logprobs: true is refused rather than ignored");
}

// The pinned shared prefix (engine#1785): single-turn requests that share a system prompt and differ in the
// user turn. `sys(n)` + `turn(k)` stand for the two parts; the table sees token ids only.
void test_prefix_pin() {
    using onebit::npu::shared_prefix_len;
    auto prompt = [](int sys, int turn_start, int turn) {
        std::vector<int> v;
        for (int i = 0; i < sys; i++) v.push_back(1 + i);
        for (int i = 0; i < turn; i++) v.push_back(turn_start + i);
        return v;
    };
    PrefixOptions o;
    o.slots = 4;
    o.cache_mb = 1;
    o.min_gain = 128;
    const std::vector<int> q1 = prompt(900, 5000, 40), q2 = prompt(900, 6000, 30), q3 = prompt(900, 7000, 50);

    {   // the gap, on the table as it was: a whole saved prompt is no use to the next single-turn request
        PrefixCache c(o);
        check(c.insert(q1, 1000) >= 0, "the first single-turn prompt is saved whole");
        const auto d = c.lookup(q2);
        check(d.slot == -1 && d.cached_tokens == 0,
              "a second question under the same 900-token system prompt reuses nothing from the first");
    }
    {   // the pin: request 2 finds the shared part, request 3 resumes from it
        PrefixCache c(o);
        check(c.pin_point({}, q1) == 0, "the first request has nothing to compare with");
        c.insert(q1, 1000);
        check(shared_prefix_len(q1, q2) == 900, "the two prompts share exactly the system part");
        const size_t at = c.pin_point(q1, q2);
        check(at == 900, "the second request's pin point is the end of what it shares with the first");
        std::string why;
        const int pin = c.pin(std::vector<int>(q2.begin(), q2.begin() + long(at)), 2000, &why);
        check(pin >= 0 && why == "pinned" && c.pinned_slot() == pin, "the shared part is pinned");
        const auto d = c.lookup(q3);
        check(d.slot == pin && d.cached_tokens == 900 && d.pinned && d.reason == "hit",
              "a third question resumes from the pin: 900 of its 950 tokens are already there");
        check(onebit::npu::resume_point(d, q3.size()) == 899, "and resumes one token short of it, as every hit does");
        check(c.pin_point(q2, q3) == 0, "a prefix the pin already covers is not pinned again");
        check(!c.lookup(q1).pinned && c.lookup(q1).cached_tokens == 940,
              "a whole-prompt slot is still preferred when it is the longer match, and says it is not the pin");
    }
    {   // what is not a pin point
        PrefixCache c(o);
        check(c.pin_point(q1, q1) == 0, "a repeated prompt is insert()'s business, not a pin");
        std::vector<int> longer = q1;
        for (int i = 0; i < 200; i++) longer.push_back(9000 + i);
        check(c.pin_point(q1, longer) == 0, "a continued conversation is insert()'s business too");
        check(c.pin_point(longer, q1) == 0, "and so is a prompt the previous one continued");
        check(c.pin_point(prompt(100, 5000, 40), prompt(100, 6000, 40)) == 0,
              "100 shared tokens do not beat prefix-min-gain 128");
        check(c.pin_point(prompt(900, 5000, 40), prompt(0, 6000, 40)) == 0, "prompts that share nothing have no pin point");
        {   // only the leading tokens count: the same 900 tokens at the end of both prompts are not a prefix
            std::vector<int> a1 = prompt(0, 5000, 200), a2 = prompt(0, 6000, 200);
            const std::vector<int> tail = prompt(900, 0, 0);
            a1.insert(a1.end(), tail.begin(), tail.end());
            a2.insert(a2.end(), tail.begin(), tail.end());
            check(shared_prefix_len(a1, a2) == 0 && c.pin_point(a1, a2) == 0,
                  "a shared ending under different openings is not a pin point");
            a2[0] = a1[0];  // one shared leading token, then they part: still nothing worth a slot
            check(shared_prefix_len(a1, a2) == 1 && c.pin_point(a1, a2) == 0,
                  "and one shared leading token before that ending is not one either");
        }
        // the contract the forward relies on: a pin point is a real common prefix, never longer than either
        for (int sys : {0, 1, 127, 128, 129, 500, 900}) {
            for (int turn : {1, 40}) {
                const std::vector<int> x = prompt(sys, 5000, turn), y = prompt(sys, 6000, turn + 7);
                const size_t n = c.pin_point(x, y);
                check(n <= x.size() && n <= y.size() && std::equal(x.begin(), x.begin() + long(n), y.begin()),
                      "a pin point never reaches past the tokens both prompts start with");
                check(n == (sys >= 128 ? size_t(sys) : 0u),
                      "it is the whole shared opening when that is worth a slot, and 0 (no pin) when it is not");
            }
        }
        PrefixOptions off = o;
        off.slots = 0;
        check(PrefixCache(off).pin_point(q1, q2) == 0, "an off cache pins nothing");
        check(PrefixCache(off).pin(q1, 10) == -1, "and refuses a pin");
    }
    {   // pin_at: the end of the system turn is known on the FIRST request, so the second one is already warm
        PrefixCache c(o);
        check(c.pin_at(q1, 900) == 900, "the first request under a 900-token system prompt pins its 900 tokens");
        check(c.pin_point({}, q1) == 0, "where comparing with a previous request has nothing to offer yet");
        std::string why;
        const int pin = c.pin(std::vector<int>(q1.begin(), q1.begin() + 900), 2000, &why);
        c.insert(q1, 1000);
        const auto d = c.lookup(q2);
        check(pin >= 0 && d.slot == pin && d.cached_tokens == 900 && d.pinned,
              "so the SECOND question resumes from the pin: 900 of its 930 tokens");
        check(c.pin_at(q2, 900) == 0 && c.pin_at(q3, 900) == 0, "and later requests do not pin the same point again");
        // what is not worth a pin, or is not one
        check(c.pin_at(q1, 0) == 0, "no system turn, no pin");
        check(c.pin_at(q1, q1.size()) == 0 && c.pin_at(q1, q1.size() + 5) == 0, "a point at or past the end of the prompt is insert()'s");
        PrefixCache fresh(o);
        check(fresh.pin_at(q1, 127) == 0 && fresh.pin_at(q1, 128) == 128, "a system turn shorter than prefix-min-gain is not pinned");
        PrefixOptions off = o;
        off.slots = 0;
        check(PrefixCache(off).pin_at(q1, 900) == 0, "an off cache pins nothing");
        // a pin that ran into the question (two questions opening alike) does not cover a question that opens
        // differently, and the system turn's own end then replaces it
        PrefixCache over(o);
        std::vector<int> a1 = prompt(900, 5000, 40), a2 = prompt(900, 5000, 40), b1 = prompt(900, 8000, 40);
        a2[905] = 4242;                                            // same first five question tokens, then different
        const size_t ran_over = over.pin_point(a1, a2);
        check(ran_over == 905, "comparing two alike-opening questions pins five tokens into the question");
        over.pin(std::vector<int>(a2.begin(), a2.begin() + long(ran_over)), 2000);
        check(over.lookup(b1).cached_tokens == 0, "which a differently opening question cannot use");
        check(over.pin_at(b1, 900) == 900, "the system turn's end is still worth pinning for it");
        const int repin = over.pin(std::vector<int>(b1.begin(), b1.begin() + 900), 2000);
        check(repin >= 0 && over.lookup(b1).cached_tokens == 900 && over.lookup(a1).cached_tokens == 900,
              "and once pinned there, every question under that system prompt resumes from it");
    }
    {   // the pin outlives what evicts and replaces everything else
        PrefixOptions b = o;
        b.slots = 4;  // one more than the three snapshots the budget holds, so a slot is always free
        b.min_gain = 1;
        PrefixCache c(b);  // 1 MB
        const std::vector<int> sys(q1.begin(), q1.begin() + 900);
        const int pin = c.pin(sys, 300000);
        const int a = c.insert(prompt(0, 100, 10), 300000);
        const int bb = c.insert(prompt(0, 200, 10), 300000);
        check(pin >= 0 && a >= 0 && bb >= 0 && c.used_bytes() == 900000, "a pin and two prompts fit under 1 MB");
        // The pin is the least recently used of the three, and 300000 more does not fit: something goes.
        const int cc = c.insert(prompt(0, 300, 10), 300000);
        check(cc == a, "room is made from the oldest whole prompt");
        check(c.pinned_slot() == pin && c.lookup(q2).slot == pin, "not from the pin, though it is the oldest slot");
        // A prompt that extends the pin does not replace it.
        std::string why;
        const int whole = c.insert(q2, 1000, pin, &why);
        check(whole >= 0 && whole != pin && c.pinned_slot() == pin,
              "a prompt that resumed from the pin is saved beside it, not over it");
        check(c.lookup(q3).slot == pin && c.lookup(q3).cached_tokens == 900, "so the next question still finds the pin");
        // A new pin replaces the old one: one shared prefix at a time.
        const std::vector<int> other = prompt(0, 40000, 500);
        const int pin2 = c.pin(other, 1000);
        check(pin2 >= 0 && c.pinned_slot() == pin2, "a new pin takes over");
        check(c.lookup(q3).slot != pin || !c.lookup(q3).pinned, "and the old one is no longer pinned");
        check(c.pin(other, 2u * 1024u * 1024u, &why) == -1 && why == "larger than prefix-cache-mb" &&
                  c.pinned_slot() == pin2,
              "a pin over the budget is refused and the standing pin is kept");
    }
    {   // every slot holds a whole prompt: the pin takes the oldest
        PrefixOptions b = o;
        b.slots = 2;
        b.min_gain = 1;
        PrefixCache c(b);
        const int s0 = c.insert(prompt(0, 100, 10), 100);
        c.insert(prompt(0, 200, 10), 100);
        const int pin = c.pin(prompt(900, 0, 0), 100);
        check(pin == s0, "with no free slot the pin takes the least recently used prompt's");
        check(c.insert(prompt(0, 300, 10), 100) != pin, "and a later prompt cannot take it back");
        check(c.pinned_slot() == pin, "the pin is still there");
    }
}

void test_prefix_cache() {
    auto ids = [](int n, int start = 1) {
        std::vector<int> v;
        for (int i = 0; i < n; i++) v.push_back(start + i);
        return v;
    };
    PrefixOptions o;
    o.slots = 4;
    o.cache_mb = 1;
    o.min_gain = 1;   // these rows keep small prompts; the 128 default has its own row below

    {   // off
        PrefixOptions off;
        off.slots = 0;
        PrefixCache c(off);
        check(!c.enabled(), "prefix-slots 0 turns the cache off");
        check(c.lookup(ids(10)).reason == "off", "an off cache reports why it did nothing");
        check(c.insert(ids(10), 1024) == -1, "an off cache saves nothing");
    }
    {
        PrefixCache c(o);
        check(c.lookup(ids(10)).slot == -1, "an empty cache shares no prefix");
        check(c.lookup(ids(10)).reason == "no slot shares a prefix", "the miss says what it looked for");
        const int slot = c.insert(ids(10), 4096);
        check(slot == 0, "the first prompt lands in slot 0");
        const auto hit = c.lookup(ids(10));
        check(hit.slot == slot && hit.cached_tokens == 10 && hit.reason == "hit", "the same prompt hits");
        check(c.lookup(ids(10, 1)).slot == slot, "a longer prompt that starts the same way hits");
        check(c.lookup(ids(10, 2)).slot == -1, "a prompt whose start differs misses");
        check(c.lookup(ids(5)).slot == -1, "a shorter prompt does not hit a longer slot");
    }
    {   // the LONGEST matching slot wins, not the newest and not the immediately preceding one
        PrefixCache c(o);
        c.insert(ids(20), 1024);
        const int b = c.insert(ids(60, 1), 1024);
        check(b >= 0, "a second, longer prompt is saved");
        check(c.lookup(ids(80)).cached_tokens == 60, "the longest saved prefix is the one reused");
    }
    {   // min-gain admission
        PrefixOptions g;
        g.slots = 4;
        g.cache_mb = 1;
        g.min_gain = 128;
        PrefixCache c(g);
        check(c.insert(ids(100), 1024) == -1, "a prompt under prefix-min-gain is not saved");
        check(c.lookup(ids(100)).slot == -1, "and it is not there to hit");
        const int big = c.insert(ids(200), 1024);
        check(big >= 0, "a prompt over the gain is saved");
        check(c.insert(ids(250), 1024, big) == -1, "50 more tokens do not beat min-gain 128");
        check(c.insert(ids(400), 1024, big) == big, "200 more tokens replace the slot they extend");
        check(c.slot_count() == 4, "min-gain does not consume slots");
    }
    {   // budget: refuse what can never fit, evict the least recently used otherwise
        PrefixOptions b;
        b.slots = 4;
        b.cache_mb = 0;
        b.min_gain = 1;
        PrefixCache c(b);
        check(c.insert(ids(10), 4096) == -1, "a snapshot over prefix-cache-mb is refused");
        PrefixOptions b2 = b;
        b2.cache_mb = 1;
        PrefixCache c2(b2);
        const int s0 = c2.insert(ids(10), 300000);
        const int s1 = c2.insert(ids(11, 100), 300000);
        check(s0 == 0 && s1 == 1 && c2.used_bytes() == 600000, "two snapshots fit under 1 MB");
        c2.lookup(ids(10));  // s0 is now the most recently used
        // 600000 + 500000 = 1.1 MB > 1 MB, so exactly one slot has to go: the older one.
        const int s2 = c2.insert(ids(12, 200), 500000);
        check(s2 >= 0, "a snapshot that needs room evicts");
        check(c2.lookup(ids(10)).slot == s0, "the recently used slot survived the eviction");
        check(c2.lookup(ids(11, 100)).slot == -1, "the least recently used slot was dropped");
    }
    {   // an invalid option is refused where it is set, not silently clamped
        auto throws = [](const PrefixOptions& bad) {
            try {
                PrefixCache c(bad);
            } catch (const std::runtime_error&) {
                return true;
            }
            return false;
        };
        PrefixOptions bad;
        bad.slots = 33;
        check(throws(bad), "prefix-slots above 32 is refused");
        bad.slots = -1;
        check(throws(bad), "a negative prefix-slots is refused");
        bad.slots = 8;
        bad.min_gain = -1;
        check(throws(bad), "a negative prefix-min-gain is refused");
    }
    {   // a request abandoned before it saves anything leaves the others' prompts alone
        PrefixCache c(o);
        const int a = c.insert(ids(10), 1024);
        c.lookup(ids(10));       // request 1 hits
        c.lookup(ids(10, 500));  // request 2, abandoned
        check(c.lookup(ids(10)).slot == a, "an abandoned request does not evict anyone's prefix");
    }
}

void test_json_schema() {
    // The structured-outputs subset from #652 item 1: a required string, an optional integer and an
    // optional bounded array, with additionalProperties: false.
    const Grammar g(json_schema_to_gbnf(nlohmann::json::parse(R"({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age":  {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2}
        },
        "required": ["name"],
        "additionalProperties": false})")));

    check(run(g, R"({"name": "x"})").can_stop, "a schema object accepts its required property alone");
    check(run(g, R"({"name": "x", "age": 3})").can_stop, "an optional integer is accepted");
    check(run(g, R"({"name": "x", "tags": ["a", "b"]})").can_stop, "an optional bounded array is accepted");
    check(run(g, R"({"name": "x", "age": 3, "tags": ["a"]})").can_stop, "everything at once is accepted");
    check(run(g, R"({"name": "x")").alive && !run(g, R"({"name": "x")").can_stop,
          "a half-written object stays alive but may not end the answer");

    check(!run(g, R"({"name": 1})").alive, "a string slot refuses an integer");
    check(!run(g, R"({"name": "x", "age": "3"})").alive, "an integer slot refuses a string");
    check(!run(g, R"({"age": 3})").alive, "a missing required property is refused");
    check(!run(g, R"({"name": "x", "extra": 1})").alive, "a property the schema does not declare is refused");
    check(!run(g, R"({"name": "x", "tags": []})").alive, "minItems 1 refuses an empty array");
    check(!run(g, R"({"name": "x", "tags": ["a","b","c"]})").alive, "maxItems 2 refuses three items");
    check(!run(g, "```json\n{\"name\": \"x\"}\n```").alive, "a Markdown fence is outside the grammar");
    check(run(g, R"({"name": "x"} )").can_stop, "trailing space is still inside the grammar");

    // enum and const are alternations of their literals, strings quoted as JSON writes them.
    const Grammar e(json_schema_to_gbnf(nlohmann::json::parse(R"({"type": "string", "enum": ["a", "b"]})")));
    check(run(e, R"("a")").can_stop && run(e, R"("b")").can_stop, "enum accepts each of its literals");
    check(!run(e, R"("c")").alive, "enum refuses a value it does not list");
    const Grammar c(json_schema_to_gbnf(nlohmann::json::parse(R"({"const": 42})")));
    check(run(c, "42").can_stop && !run(c, "43").alive, "const accepts exactly its value");

    // A type list is the nullable form: ["string", "null"].
    const Grammar n(json_schema_to_gbnf(nlohmann::json::parse(R"({"type": ["string", "null"]})")));
    check(run(n, R"("x")").can_stop && run(n, "null").can_stop, "a type list accepts either branch");
    check(!run(n, "1").alive, "a type list refuses a type it does not list");

    // Nested containers.
    const Grammar nest(json_schema_to_gbnf(nlohmann::json::parse(R"({
        "type": "object",
        "properties": {"outer": {"type": "object", "properties": {"inner": {"type": "boolean"}},
                                  "required": ["inner"]}},
        "required": ["outer"]})")));
    check(run(nest, R"({"outer": {"inner": true}})").can_stop, "a nested object is accepted");
    check(!run(nest, R"({"outer": {"inner": 1}})").alive, "a nested wrong type is refused");

    // oneOf is alternation.
    const Grammar one(json_schema_to_gbnf(nlohmann::json::parse(
        R"({"oneOf": [{"type": "string"}, {"type": "integer"}]})")));
    check(run(one, R"("s")").can_stop && run(one, "7").can_stop, "oneOf accepts every branch");
    check(!run(one, "true").alive, "oneOf refuses a value outside every branch");

    // An empty schema is any JSON value, as in JSON Schema.
    const Grammar any(json_schema_to_gbnf(nlohmann::json::parse("{}")));
    check(run(any, R"({"a": [1, "x", null]})").can_stop, "an empty schema accepts any JSON");
}

void test_json_schema_refusals() {
    // A constraint this converter cannot enforce is a 400 naming the keyword, never a silent drop.
    for (const char* keyword : {"pattern", "$ref", "allOf", "minimum", "format", "uniqueItems"}) {
        const std::string schema = std::string("{\"type\": \"string\", \"") + keyword + "\": \"x\"}";
        check(schema_error(schema).find(keyword) != std::string::npos,
              std::string("the refusal names ") + keyword);
    }
    check(schema_error(R"({"type": "object", "additionalProperties": {"type": "string"}})")
                  .find("additionalProperties") != std::string::npos,
          "additionalProperties as a schema is refused by name");
    check(schema_error(R"({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["b"]})")
                  .find("required") != std::string::npos,
          "required naming an undeclared property is refused");
    check(!schema_error(R"({"type": "weird"})").empty(), "an unknown type is refused");
    check(!schema_error(R"({"type": "string", "enum": []})").empty(), "an empty enum is refused");
    check(!schema_error(R"({"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 2})")
               .empty(),
          "maxItems below minItems is refused");
    check(!schema_error("[]").empty(), "a schema that is neither an object nor a boolean is refused");

    // The OpenAI envelope, and the conflicts that used to be impossible because the type was refused.
    auto starts_with = [](const std::string& s, const std::string& prefix) { return s.rfind(prefix, 0) == 0; };
    check(starts_with(error_of(nlohmann::json::parse(R"({"response_format": {"type": "json_schema"}})")),
                      "response_format:"),
          "json_schema without the json_schema object is refused");
    check(error_of(nlohmann::json::parse(
                       R"({"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}})"))
                  .find("schema") != std::string::npos,
          "json_schema without a schema is refused");
    const auto schema_body = options_from_body(nlohmann::json::parse(
        R"({"response_format": {"type": "json_schema", "json_schema": {"name": "p", "strict": true,
             "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}}})"));
    check(!schema_body.grammar.empty() && !schema_body.json_object, "a json_schema request becomes a grammar");
    check(error_of(nlohmann::json::parse(
                      R"({"response_format": {"type": "json_schema", "json_schema": {"schema": {}}}})"))
               .empty(),
          "the smallest valid envelope is accepted");
    check(error_of(nlohmann::json::parse(
                       R"({"grammar": "root ::= \"a\"", "response_format": {"type": "json_schema",
                           "json_schema": {"schema": {"type": "object"}}}})"))
                  .find("conflicts") != std::string::npos,
          "grammar and json_schema together are refused");
}


// Gemma 4's own markup (engine#1779, F10 fault 4), taken from the pack's tokenizer_config.json: the call body is
// `call:NAME{key:value,...}` with string values wrapped in <|"|>, which is not JSON. Asserted on the normalised
// JSON rather than on "a call was found", because the normalisation is the part that can be silently wrong.
void test_gemma4_answer_parser() {
    const auto G = AnswerFamily::gemma4;
    {
        const std::string reply =
            "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3,metric:true}<tool_call|>";
        const auto a = parse_answer(reply, G, true);
        check(a.calls.size() == 1, "gemma4: a <|tool_call> body is a call when tools were sent");
        check(a.calls[0].name == "get_weather", "gemma4: the name is what follows call:");
        check(a.calls[0].arguments == R"({"city":"Paris","days":3,"metric":true})",
              "gemma4: arguments are rewritten to JSON (escaped strings, numbers, booleans)");
        check(a.content.empty() && a.content_null, "gemma4: a call-only reply has content: null");
    }
    {
        // A comma inside an escaped string must not split the argument list -- the case a naive splitter gets wrong.
        const std::string reply =
            "<|tool_call>call:say{text:<|\"|>hello, world<|\"|>,n:2}<tool_call|>";
        const auto a = parse_answer(reply, G, true);
        check(a.calls.size() == 1 && a.calls[0].arguments == R"({"text":"hello, world","n":2})",
              "gemma4: a comma inside <|\"|>...<|\"|> stays inside the string");
    }
    {
        const std::string reply = "<|tool_call>call:ping{}<tool_call|>";
        const auto a = parse_answer(reply, G, true);
        check(a.calls.size() == 1 && a.calls[0].arguments == "{}", "gemma4: a call with no arguments is {}");
        const auto without = parse_answer(reply, G, false);
        check(without.calls.empty(), "gemma4: the same markup is not a call when no tools were sent");
        check(without.content.find("<|tool_call>") != std::string::npos,
              "gemma4: and it stays in content, marker included");
    }
    {
        const auto a = parse_answer("<|channel>thought\nlet me think<channel|>Done.", G, false);
        check(a.reasoning == "let me think" && a.content == "Done.",
              "gemma4: the thought channel is reasoning, without the newline the template puts after the tag");
        check(a.content == "Done.", "gemma4: and the reply after it is content");
    }
    {
        // A string whose closing <|"|> never arrives is malformed: no call, and the markup is back in content.
        const std::string reply = "<|tool_call>call:say{text:<|\"|>hello}<tool_call|>";
        const auto a = parse_answer(reply, G, true);
        check(a.calls.empty(), "gemma4: an unterminated <|\"|> string is not a call");
        check(a.content == reply, "gemma4: and what the model wrote is back in content, whole");
        const auto b = parse_answer("<|tool_call>call:f{a:1}tail}<tool_call|>", G, true);
        check(b.calls.empty() && !b.content.empty(), "gemma4: text after the arguments object is not a call");
        const auto c = parse_answer("<|tool_call>call:f{a:[1,2}<tool_call|>", G, true);
        check(c.calls.empty() && !c.content.empty(), "gemma4: an array that never closes is not a call");
    }
    {
        // Nested object and array, the same bare-key form one level down.
        const auto a = parse_answer(
            "<|tool_call>call:f{o:{k:<|\"|>v<|\"|>},l:[1,<|\"|>a,b<|\"|>],z:null}<tool_call|>", G, true);
        check(a.calls.size() == 1 && a.calls[0].arguments == R"({"o":{"k":"v"},"l":[1,"a,b"],"z":null})",
              "gemma4: nested objects and arrays are rewritten too");
    }
}

// The family comes from the container's chat template. The Gemma 4 lines are the two literals the pack's own
// chat_template carries (amd/gemma-4-e4b-it-mobius-int4_rai_1.9.0_npu: `<|tool_call>` x1, `<|channel>thought`
// x1, `<tool_call>` x0, `<think>` x0), not a guess at them.
void test_answer_family_of_template() {
    using onebit::npu::answer_family_of_template;
    check(answer_family_of_template("") == AnswerFamily::qwen3, "no template is qwen3, as before");
    check(answer_family_of_template("{%- if tools %}<tool_call>\n{\"name\": ...}</tool_call>{% endif %}<think>") ==
              AnswerFamily::qwen3,
          "a template with <tool_call> and <think> is qwen3");
    check(answer_family_of_template("{{- '<|tool_call>call:' + function['name'] + '{' -}}") == AnswerFamily::gemma4,
          "a template that writes <|tool_call>call: is gemma4");
    check(answer_family_of_template("{{- '<|channel>thought\\n<channel|>' -}}") == AnswerFamily::gemma4,
          "a template with the thought channel and no tools is gemma4 too");
    check(answer_family_of_template("<|im_start|>user\n{{ content }}<|im_end|>") == AnswerFamily::qwen3,
          "a template with neither marker is qwen3");
}

// Every tag every family declares must be recognised, whatever its length, whole and split across two chunks at
// every byte. The scanner once read tags through a 16-byte window, so a 17-byte tag read as text (engine#1779);
// this arm fails for any family whose tags the scanner cannot match, including one added later.
void test_every_family_tag_is_matched() {
    for (const AnswerFamily f : {AnswerFamily::qwen3, AnswerFamily::gemma4}) {
        const auto t = onebit::npu::answer_detail::tags_of(f);
        const std::string name = onebit::npu::answer_detail::family_name(f);
        const std::string think = std::string(t.think_open) + "R" + t.think_close + "C";
        const std::string call = std::string(t.call_open) + "not a call" + t.call_close;
        for (size_t cut = 0; cut <= think.size(); cut++) {
            AnswerStream s(f, false);
            std::vector<std::pair<bool, std::string>> out;
            s.feed(think.substr(0, cut), out);
            s.feed(think.substr(cut), out);
            s.finish(out);
            check(s.reasoning() == "R" && s.content() == "C",
                  (name + ": think tags are matched, split at byte " + std::to_string(cut)).c_str());
        }
        for (size_t cut = 0; cut <= call.size(); cut++) {
            AnswerStream s(f, true);
            std::vector<std::pair<bool, std::string>> out;
            s.feed(call.substr(0, cut), out);
            s.feed(call.substr(cut), out);
            s.finish(out);
            // Not a valid body, so it comes back as content: open tag, body and close tag, which proves both
            // call tags were seen (an unseen close tag would leave the body unterminated and drop the tag).
            check(s.calls().empty() && s.content() == call && s.reasoning().empty(),
                  (name + ": call tags are matched, split at byte " + std::to_string(cut)).c_str());
        }
    }
}

void test_answer_parser() {
    const auto F = AnswerFamily::qwen3;
    {   // reasoning, then content: the Qwen3 shape this route sees on every answer
        const auto a = parse_answer("<think>\nOkay, the user asked.", F, false);
        check(a.reasoning == "Okay, the user asked.",
              "reasoning_content is the inside of <think>, without the newline the model puts after the tag");
        check(a.content.empty() && a.calls.empty(), "a reasoning-only answer has no content and no calls");
        check(!a.content_null, "content is empty string, not null, when there are no calls");

        const auto b = parse_answer("<think>hmm</think>The answer is 42.", F, false);
        check(b.reasoning == "hmm" && b.content == "The answer is 42.", "reasoning and content are split apart");
    }
    {   // an unterminated <think> (a truncated reply) keeps the rest as reasoning
        const auto a = parse_answer("<think>still thinking", F, false);
        check(a.reasoning == "still thinking" && a.content.empty(), "an unclosed <think> owns the rest");
    }
    {   // markup is a call only when the request carried tools
        const std::string call = R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>)";
        const auto with = parse_answer(call, F, true);
        check(with.calls.size() == 1, "a <tool_call> is a call when tools were sent");
        check(with.calls[0].name == "get_weather", "the call carries the function name");
        check(with.calls[0].arguments == R"({"city":"Paris"})", "arguments are the JSON object as a string");
        check(with.calls[0].id == "call_0", "calls are numbered deterministically");
        check(with.content.empty() && with.content_null, "a call-only reply has content: null");

        const auto without = parse_answer(call, F, false);
        check(without.calls.empty(), "the same markup is not a call when no tools were sent");
        check(without.content.find("<tool_call>") != std::string::npos, "and it stays in content, marker included");
        check(!without.content_null, "so content is not null");
    }
    {   // text plus a call: content is what the model said, the call is separate
        const auto a = parse_answer(
            "Let me check.<tool_call>{\"name\": \"get_time\", \"arguments\": {\"zone\": \"Rome\"}}</tool_call>",
            F, true);
        check(a.content == "Let me check." && a.calls.size() == 1 && !a.content_null,
              "prose before a call stays content and the reply is not content: null");
    }
    {   // several calls, in order
        const auto a = parse_answer(
            R"(<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>)",
            F, true);
        check(a.calls.size() == 2 && a.calls[0].name == "a" && a.calls[1].name == "b", "calls keep their order");
        check(a.calls[1].id == "call_1", "ids are sequential");
    }
    {   // a malformed call block is not dropped
        const auto a = parse_answer("<tool_call>not json</tool_call>", F, true);
        check(a.calls.empty(), "an unparseable block is not a call");
        check(a.content.find("not json") != std::string::npos, "and its text survives in content");
    }
    {   // an unterminated call block is put back, not swallowed
        const auto a = parse_answer("<tool_call>{\"name\": \"x\"", F, true);
        check(a.calls.empty(), "an unterminated block makes no call");
        check(a.content.find("\"name\"") != std::string::npos, "its text survives in content");
    }
    {   // text that only looks like markup
        const auto a = parse_answer("2 < 3 and 4 < 5", F, false);
        check(a.content == "2 < 3 and 4 < 5", "a stray '<' is just text");
        const auto b = parse_answer("use <thinking> not <think>", F, false);
        check(b.content.find("<thinking>") != std::string::npos, "a longer tag is not the family's tag");
    }
}

void test_answer_stream() {
    // The invariant that matters: feeding the same text in chunks must produce exactly the pieces
    // the whole-text parse produces -- that is what makes streamed deltas add up to the reply.
    const std::string text =
        "<think>\nLet me think about this.</think>Sure.<tool_call>{\"name\": \"f\", \"arguments\": {\"a\": 1}}</tool_call>";
    const auto whole = parse_answer(text, AnswerFamily::qwen3, true);
    for (size_t step : {1u, 2u, 3u, 5u, 7u, 13u}) {
        AnswerStream s(AnswerFamily::qwen3, true);
        std::vector<std::pair<bool, std::string>> pieces;
        for (size_t i = 0; i < text.size(); i += step) s.feed(text.substr(i, step), pieces);
        s.finish(pieces);
        std::string reasoning, content;
        for (const auto& [is_reasoning, t] : pieces) (is_reasoning ? reasoning : content) += t;
        const std::string label = " at chunk size " + std::to_string(step);
        check(reasoning == whole.reasoning, "streamed reasoning equals the whole-text parse" + label);
        check(content == whole.content, "streamed content equals the whole-text parse" + label);
        check(s.calls().size() == whole.calls.size() && s.calls()[0].name == "f",
              "the call is found across chunk boundaries" + label);
    }
    {   // a tag split across the boundary is never emitted as text
        AnswerStream s(AnswerFamily::qwen3, false);
        std::vector<std::pair<bool, std::string>> pieces;
        s.feed("hello <thi", pieces);
        s.feed("nk>world</think>", pieces);
        s.finish(pieces);
        std::string all;
        for (const auto& [r, t] : pieces) all += t;
        check(all == "hello world", "a tag split across two chunks is not emitted");
        check(s.reasoning() == "world", "and the split lands in reasoning");
    }
}

// Two tools: one with a required string field, one that declares no parameters at all.
const char* kTools = R"([{"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"], "additionalProperties": false}}},
    {"type": "function", "function": {"name": "ping"}}])";

void test_tool_grammar() {
    const auto tools = nlohmann::json::parse(kTools);
    // tool_choice "required", one call at a time: names and argument types are unforgeable.
    const Grammar g(tool_call_gbnf(tools, nlohmann::json("required"), false));
    check(run(g, R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>)").can_stop,
          "a well-formed call is accepted");
    check(run(g, R"(<tool_call>{"name": "ping", "arguments": {}}</tool_call>)").can_stop,
          "a tool that declares no parameters still takes an arguments object");
    check(!run(g, R"(<tool_call>{"name": "get_weathr", "arguments": {"city": "Paris"}}</tool_call>)").alive,
          "a misspelt tool name cannot be generated");
    check(!run(g, R"(<tool_call>{"name": "get_weather", "arguments": {"city": 7}}</tool_call>)").alive,
          "an argument of the wrong type cannot be generated");
    check(!run(g, R"(<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>)").alive,
          "a missing required argument cannot be generated");
    check(!run(g, R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Paris", "x": 1}}</tool_call>)").alive,
          "an argument the schema does not declare cannot be generated");
    check(!run(g, R"(```json
{"name": "get_weather", "arguments": {"city": "Paris"}}
```)").alive,
          "a Markdown fence cannot wrap the call");
    check(!run(g, R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>
<tool_call>{"name": "ping", "arguments": {}}</tool_call>)").alive,
          "parallel_tool_calls false refuses a second call");

    // parallel_tool_calls true: one or more calls.
    const Grammar par(tool_call_gbnf(tools, nlohmann::json("required"), true));
    check(run(par, R"(<tool_call>{"name": "ping", "arguments": {}}</tool_call>
<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>)").can_stop,
          "parallel_tool_calls true accepts several calls");

    // The object form names one function: the other one is not generable.
    const auto only = nlohmann::json::parse(R"({"type": "function", "function": {"name": "ping"}})");
    const Grammar one(tool_call_gbnf(tools, only, false));
    check(run(one, R"(<tool_call>{"name": "ping", "arguments": {}}</tool_call>)").can_stop,
          "tool_choice function accepts that function");
    check(!run(one, R"(<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>)").alive,
          "tool_choice function refuses the other tools");

    // "auto" and "none" install no grammar, and say so by returning "".
    check(tool_call_gbnf(tools, nlohmann::json("auto"), true).empty(),
          "tool_choice auto installs no grammar (a call-or-text alternation is not a CFG)");
    check(tool_call_gbnf(tools, nlohmann::json("none"), true).empty(), "tool_choice none installs no grammar");
    check(tool_call_gbnf(nlohmann::json::array(), nlohmann::json("required"), true).empty(),
          "no tools means no tool grammar");
}

void test_tool_request_parsing() {
    auto starts_with = [](const std::string& s, const std::string& prefix) { return s.rfind(prefix, 0) == 0; };
    const std::string tools = std::string(R"({"tools": )") + kTools + "}";
    // required -> a grammar is installed
    auto req = options_from_body(nlohmann::json::parse(
        std::string(R"({"tools": )") + kTools + R"(, "tool_choice": "required"})"));
    check(!req.grammar.empty() && req.tool_grammar, "tool_choice required installs the call grammar");
    // auto (the default) -> no grammar, and no error either
    auto autod = options_from_body(nlohmann::json::parse(tools));
    check(autod.grammar.empty() && !autod.tool_grammar, "tools with no tool_choice leave the answer free");
    // parallel_tool_calls must be a bool, and needs tools
    check(starts_with(error_of(nlohmann::json::parse(
                          std::string(R"({"tools": )") + kTools + R"(, "tool_choice": "required", "parallel_tool_calls": 1})")),
                      "parallel_tool_calls:"),
          "parallel_tool_calls must be true or false");
    check(starts_with(error_of(nlohmann::json::parse(R"({"tool_choice": "required"})")), "tool_choice:"),
          "tool_choice without tools is refused");
    check(!error_of(nlohmann::json::parse(
                        std::string(R"({"tools": )") + kTools + R"(, "tool_choice": {"type": "function", "function": {"name": "nope"}}})"))
               .empty(),
          "tool_choice naming a tool that was not declared is refused");
    check(!error_of(nlohmann::json::parse(R"({"tools": [{"function": {"name": "a"}}, {"function": {"name": "a"}}]})"))
               .empty(),
          "a duplicate function name is refused");
    check(!error_of(nlohmann::json::parse(R"({"tools": [{"type": "other", "function": {"name": "a"}}]})")).empty(),
          "a tool whose type is not function is refused");
    check(!error_of(nlohmann::json::parse(
                        R"({"tools": [{"function": {"name": "a", "parameters": {"type": "string", "pattern": "^x$"}}}],
                            "tool_choice": "required"})"))
               .empty(),
          "under tool_choice required, an argument schema this converter cannot honour is refused");
    check(error_of(nlohmann::json::parse(
                      R"({"tools": [{"function": {"name": "a", "parameters": {"type": "string", "pattern": "^x$"}}}],
                          "tool_choice": "auto"})"))
              .empty(),
          "under tool_choice auto no grammar is built, so no schema is rejected");
    // one grammar at a time
    check(starts_with(error_of(nlohmann::json::parse(
                          std::string(R"({"grammar": "root ::= \"a\"", "tools": )") + kTools + "}")),
                      "tools:"),
          "grammar and tools together are refused");
    check(starts_with(error_of(nlohmann::json::parse(
                          std::string(R"({"response_format": {"type": "json_object"}, "tools": )") + kTools + "}")),
                      "tools:"),
          "response_format and tools together are refused");
}

void test_content_mask() {
    // One entry per generated token, in order; the mask says which of them are `content`, which is
    // what issue #651 item 3 reports logprobs for.
    const std::vector<std::string> pieces = {"<think>", "\n", "Okay", "</think>", "The", " answer", " is", " 42."};
    const auto mask = content_mask(pieces, AnswerFamily::qwen3, false);
    check(mask.size() == pieces.size(), "the mask has one bit per token");
    check(mask[0] == 0 && mask[1] == 0 && mask[2] == 0, "reasoning tokens get no entry");
    check(mask[3] == 0, "the </think> token is markup, not content");
    check(mask[4] && mask[5] && mask[6] && mask[7], "the answer's tokens do");
    {
        // A token that straddles the boundary counts as content: its text did reach the caller.
        const std::vector<std::string> straddle = {"<think>", "hmm</think>The", " answer"};
        const auto m = content_mask(straddle, AnswerFamily::qwen3, false);
        check(m[0] == 0 && m[1] == 1 && m[2] == 1, "a straddling token is content");
    }
    {
        // Call markup gets no entry either, and the same text with no tools does (it is content then).
        const std::vector<std::string> call = {"<tool_call>", "{\"name\": \"f\",", " \"arguments\": {}}", "</tool_call>"};
        const auto with = content_mask(call, AnswerFamily::qwen3, true);
        bool any = false;
        for (char c : with) any = any || c;
        check(!any, "call markup gets no logprob entry when tools were sent");
        const auto without = content_mask(call, AnswerFamily::qwen3, false);
        bool all = true;
        for (char c : without) all = all && c;
        check(all, "the same tokens are content when no tools were sent");
    }
    {
        // Text held back to the end is still content.
        const std::vector<std::string> tail = {"hello", " wor"};
        check(content_mask(tail, AnswerFamily::qwen3, false)[1] == 1, "held-back text is content");
    }
}

void test_typed_fields() {
    using nlohmann::json;
    // Absent and null are the default; a number is taken; a fractional cap is truncated (body.value did that).
    check(int_field(json::parse(R"({"a": 8})"), "a", 256) == 8, "an integer field is read");
    check(int_field(json::parse(R"({})"), "a", 256) == 256, "an absent field is the default");
    check(int_field(json::parse(R"({"a": null})"), "a", 256) == 256, "a null field is the default");
    check(int_field(json::parse(R"({"a": 8.5})"), "a", 256) == 8, "a fractional cap is truncated, not refused");
    check(throws_of([] { int_field(json::parse(R"({"max_tokens": "8"})"), "max_tokens", 256); }) ==
              "max_tokens: must be an integer",
          "a string max_tokens is a 400 naming the field (engine#1106: it used to be a 500 with no body)");
    check(throws_of([] { int_field(json::parse(R"({"max_tokens": true})"), "max_tokens", 256); }) ==
              "max_tokens: must be an integer",
          "a boolean max_tokens is refused too");

    check(bool_field(json::parse(R"({"stream": true})"), "stream"), "a boolean field is read");
    check(!bool_field(json::parse(R"({"stream": false})"), "stream"), "false is false");
    check(!bool_field(json::parse(R"({})"), "stream"), "an absent flag is false by default");
    check(throws_of([] { bool_field(json::parse(R"({"stream": "yes"})"), "stream"); }) == "stream: must be true or false",
          "a string stream is a 400 naming the field (engine#1106: it used to be a 500 with no body)");
}

void test_tools_type() {
    using nlohmann::json;
    // engine#1106: a wrong-typed tools used to short-circuit as if no tools were sent, in the default mode.
    check(throws_of([] { options_from_body(json::parse(R"({"tools": "a string"})")); }) == "tools: must be an array",
          "a non-array tools is refused even with the default tool_choice");
    check(throws_of([] { options_from_body(json::parse(R"({"tools": 7})")); }) == "tools: must be an array",
          "a numeric tools is refused too");
    const auto none = options_from_body(json::parse(R"({"tools": []})"));
    check(none.tools_sent && none.grammar.empty(), "an EMPTY tools array still means no tools, and still counts as sent");
    const auto explicit_auto = options_from_body(json::parse(R"({"tools": [], "tool_choice": "auto"})"));
    check(explicit_auto.grammar.empty(), "an empty tools array with an explicit auto installs no grammar");
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
    check(error_of(nlohmann::json::parse(R"({"response_format": {"type": "json_schema"}})")).find("json_schema") !=
              std::string::npos,
          "json_schema without its envelope is refused by name");
    check(error_of(nlohmann::json::parse(R"({"response_format": {"type": "nope"}})")).find("nope") != std::string::npos,
          "an unknown response_format type is refused");
    check(!error_of(nlohmann::json::parse(
                        R"({"grammar": "root ::= \"a\"", "response_format": {"type": "json_object"}})"))
               .empty(),
          "grammar and json_object together are refused");
    // Nothing else in the body is inspected: the parser stays backward compatible.
    check(error_of(nlohmann::json::parse(R"({"prompt": "hi", "n": 3})")).empty(),
          "fields this route does not implement are still ignored");
}

}  // namespace

// engine#2017: the logits row is `vocab_size` wide, which may exceed the tokenizer (Qwen3-0.6B: 151,936 for 151,669
// ids; the shape below is the issue's, 32,032 for 32,016). Ids past the tokenizer are padding positions: every real
// logit negative and the padding at exactly 0.0 is the row where an unbounded argmax picks one (engine#1182).
void test_vocab_bound() {
    const int V = 32016, W = 32032;
    std::vector<float> row(size_t(W), 0.0f);
    for (int v = 0; v < V; v++) row[size_t(v)] = -1.0f - float(v % 97) * 0.01f;
    row[123] = -0.5f;  // the real best
    check(argmax_token(row.data(), size_t(W)) >= V, "control: an unbounded argmax lands on a padding id");
    check(argmax_token(row.data(), size_t(W), V) == 123, "the bounded argmax picks the best real id");
    check(sample(row, SampleParams(), 1) >= V, "control: unbounded greedy sample lands on a padding id");
    std::mt19937_64 rng(1);
    check(sample_token(row, SampleParams(), rng, nullptr, V) == 123, "bounded greedy sample picks the best real id");
    SampleParams hot;
    hot.temperature = 1.0f;
    bool all_real = true;
    for (uint64_t s = 0; s < 200; s++) {
        std::mt19937_64 r(s);
        if (sample_token(row, hot, r, nullptr, V) >= V) all_real = false;
    }
    check(all_real, "a bounded draw at temperature 1 never draws a padding id");
    std::vector<char> mask(size_t(W), 1);
    std::mt19937_64 r2(2);
    check(sample_token(row, SampleParams(), r2, &mask, V) == 123, "the bound holds with a mask that allows padding");
    check(token_bound(size_t(W), 0) == size_t(W), "no tokenizer keeps the whole row");
    check(token_bound(size_t(V), W) == size_t(V), "a tokenizer larger than the row does not widen it");
}

int main() {
    test_vocab_bound();
    test_sampler();
    test_sampler_distribution();
    test_sample_probs();
    test_stop();
    test_json_grammar();
    test_gbnf();
    test_grammar_end_tokens();
    test_logprobs();
    test_logprob_options();
    test_prefix_cache();
    test_prefix_pin();
    test_json_schema();
    test_json_schema_refusals();
    test_answer_parser();
    test_answer_stream();
    test_tool_grammar();
    test_gemma4_answer_parser();
    test_every_family_tag_is_matched();
    test_answer_family_of_template();
    test_tool_request_parsing();
    test_content_mask();
    test_request_parsing();
    test_typed_fields();
    test_tools_type();
    std::printf("%s\n", failures ? "FAILED" : "npu_sampling: ok");
    return failures ? 1 : 0;
}
