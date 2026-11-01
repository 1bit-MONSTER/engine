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
using onebit::npu::Grammar;
using onebit::npu::AnswerFamily;
using onebit::npu::AnswerStream;
using onebit::npu::json_grammar;
using onebit::npu::parse_answer;
using onebit::npu::tool_call_gbnf;
using onebit::npu::kRandomSeed;
using onebit::npu::PrefixOptions;
using onebit::npu::PrefixCache;
using onebit::npu::json_schema_to_gbnf;
using onebit::npu::logprobs_of;
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

void test_answer_parser() {
    const auto F = AnswerFamily::qwen3;
    {   // reasoning, then content: the Qwen3 shape this route sees on every answer
        const auto a = parse_answer("<think>\nOkay, the user asked.", F, false);
        check(a.reasoning == "\nOkay, the user asked.", "reasoning_content is the inside of <think>");
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

int main() {
    test_sampler();
    test_sampler_distribution();
    test_stop();
    test_json_grammar();
    test_gbnf();
    test_logprobs();
    test_logprob_options();
    test_prefix_cache();
    test_json_schema();
    test_json_schema_refusals();
    test_answer_parser();
    test_answer_stream();
    test_tool_grammar();
    test_tool_request_parsing();
    test_request_parsing();
    std::printf("%s\n", failures ? "FAILED" : "npu_sampling: ok");
    return failures ? 1 : 0;
}
