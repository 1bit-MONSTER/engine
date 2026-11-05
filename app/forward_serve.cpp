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

// `1bit forward-serve -m <model.q4nx> --kernels <elf dir> -p <port>`: one
// repacked Q4NX model on the model-generic NPU forward, behind OpenAI
// endpoints. It is the serve route for `1bit serve -m <model.gguf> --device
// npu` after the GGUF has been repacked (serve.cpp).
//
//   GET  /health, /v1/health    200 once the model is on the device
//   GET  /v1/models             the one model
//   POST /v1/chat/completions   ChatML prompt, sampling controls, optional SSE stream (400 when the
//                               container's tokenizer cannot represent ChatML, app/npu_chatml.h)
//   POST /v1/completions        raw prompt
//   POST /v1/systemone          d3 decision questions (alias /v1/decisions), when the model
//   GET  /v1/systemone/models   directory carries a decision head -- see docs/npu.md "Decisions"
//
// Decoding is greedy unless the request asks for more: temperature, top_k, top_p and seed
// sample, stop cuts the answer, and grammar (GBNF) or response_format {"type":
// "json_object"} constrain it.  See app/npu_sampling.h, app/npu_grammar.h and
// docs/npu.md "Sampling and structured output".
//
// Requests run one at a time: the forward holds one KV cache, and each request
// resets it from position 0.
#include "forward_serve.h"

#include "../npu/forward/include/q4nx_forward.h"
#include "npu_device_lock.h"
#include "npu_answer_parser.h"
#include "npu_chatml.h"
#include "npu_grammar.h"
#include "npu_prefix_cache.h"
#include "npu_prompt_ids.h"
#include "npu_sampling.h"
#include "npu_stop_tokens.h"
#include "npu_systemone.h"
#include "tokenizer.h"
#include "../npu/decision.h"  // the d3 readout head (npu/decision.h, issue #653)
#include "rt.h"  // the own NPU runtime (ONEBIT_NPU_RT_DIR)

#include <httplib.h>
#include "http_guard.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace onebit {

namespace {

using json = nlohmann::json;

struct ForwardEngine {
    std::string id;
    int lock_fd = -1;  // app/npu_device_lock.h: the device lock, held for the process's life
    std::unique_ptr<npu::Tokenizer> tok;
    int  bos_id  = -1;     // config.json bos_token_id (the chat template may open with it)
    bool add_bos = false;  // config.json add_bos_token: the rendered chat prompt needs BOS
    std::unique_ptr<Q4nxNpuForward> fwd;
    std::vector<int> eos;   // config.json eos_token_id: one id or a list (app/npu_stop_tokens.h)
    int endoftext = -1;
    int max_context = 0;   // config.json max_position_embeddings: prompt plus answer
    // The chat route renders ChatML for every container (#832), so whether this container's
    // tokenizer can represent that prompt is a property of the container: decided once at load
    // (#963).  A refusal, not a load failure -- the id route stays usable on such a container.
    bool chatml_ok = true;
    std::string chatml_refusal;  // empty unless !chatml_ok; the 400's message
    // decode() of every vocabulary id, built on the first constrained request: the GBNF
    // mask tests each token's text against the grammar state, per token.
    std::vector<std::string> tok_text;
    // Saved prompts (issue #650): the slot table decides what to keep, prefix_state holds each
    // slot's attention KV and GDN state.  Only ever touched under mu.
    std::unique_ptr<npu::PrefixCache> prefix;
    std::vector<Q4nxNpuForward::StateSnapshot> prefix_state;
    // The d3 decision head (#653), loaded when decision_config.json and decision.q4nx sit next to the
    // model: it is what turns this server into a decision endpoint as well as a chat one.
    std::unique_ptr<npu::DecisionHead> head;
    std::string head_error;  // why there is no head, for the 503 the decision route answers
    // --dump-logits <dir>: write each generated position's raw logits (float32, vocabulary-wide)
    // as step_<n>.bin.  The OpenAI API cannot express this -- it caps top_logprobs at 100 -- so a
    // KL-divergence comparison against another stack (engine#671 gate b) needs the distribution
    // itself.  Empty means off, which is the default: nothing is written unless asked.
    std::string dump_logits;
    std::mutex mu;
};

// What one request did, for the `usage` object and the prefix log line (issue #650).
struct CallStats {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    // AMD's model_benchmark protocol reports two stages per request -- prompt processing (to first
    // token) and token generation -- and onnx/bench_npu_serve.py reads them from the reply's
    // `timings` (app/unified.cpp has always emitted them; this route did not, so no NPU row from
    // this route could be expressed in AMD's units at all).
    double prompt_ms = 0;
    double predicted_ms = 0;
    npu::PrefixDecision prefix{-1, 0, "off"};  // "off" when --prefix-slots 0
};

// One generated token's log-probability entry, with the text that token produced.  The two travel
// together so a streamed chunk can carry exactly the entries whose bytes it carries.
struct Emitted {
    std::string text;
    npu::TokenLogprob lp;
};

size_t complete_utf8(const std::string& s) {
    size_t lead = s.size();
    while (lead > 0 && s.size() - lead < 4 && (static_cast<unsigned char>(s[lead - 1]) & 0xC0) == 0x80) --lead;
    if (lead == 0) return s.size();
    const unsigned char c = static_cast<unsigned char>(s[lead - 1]);
    const size_t need = (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    return s.size() - (lead - 1) >= need ? s.size() : lead - 1;
}

std::string now_id(const char* prefix) {
    static std::atomic<unsigned> n{0};
    return std::string(prefix) + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" +
           std::to_string(n++);
}

int json_int(const std::string& path, const char* key, int def) {
    std::ifstream f(path);
    if (!f) return def;
    json j;
    f >> j;
    if (!j.contains(key)) return def;
    // A boolean is an int here: every HF checkpoint writes add_bos_token/add_eos_token as a
    // JSON boolean, and the alternative -- asking the packers to write 1/0 instead -- would make
    // the container's config.json disagree with the checkpoint it came from.  Reading only
    // is_number_integer() made add_bos_token a dead key: measured on Zyphra/ZR1-1.5B, forward-serve
    // reported the same 27 prompt tokens with "add_bos_token": true and with false, i.e. the chat
    // prompt never got its BOS.
    if (j[key].is_boolean()) return j[key].get<bool>() ? 1 : 0;
    if (j[key].is_number_integer()) return j[key].get<int>();
    return def;
}

void handle_generate(ForwardEngine& e, const httplib::Request& req, httplib::Response& res, bool chat) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"request body is not JSON"}})", "application/json");
        return;
    }
    std::string prompt;  // the chat template, or a string prompt; documents say nothing for an ids prompt
    // Set when /v1/completions was given the prompt as token ids (llama.cpp's shape): the ids are then
    // the prompt and nothing is tokenized or prepended -- that is what makes a two-engine parity run
    // compare the forward instead of the tokenizer (app/npu_prompt_ids.h).  `prompt` stays empty on that
    // path and nothing reads it: the token count, the context guard and the prefix-cache key all come
    // from `ids`.  Do not log `prompt` as the request's text without handling this case first.
    std::vector<int> prompt_ids;
    bool prompt_is_ids = false;
    try {
        if (chat) {
            if (!e.chatml_ok) throw std::runtime_error(e.chatml_refusal);
            prompt = npu::chatml(body.at("messages"), npu::thinking_from_body(body));
        } else {
            prompt_is_ids = npu::prompt_ids_from_body(body, e.tok->size(), prompt_ids);
            if (!prompt_is_ids) prompt = body.at("prompt").get<std::string>();
        }
    } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", std::string("bad request: ") + ex.what()}}}}.dump(), "application/json");
        return;
    }
    // Decode-time controls.  Every field is optional and absent means the greedy decode
    // this route has always done, so existing callers see no change; a field the route
    // cannot honour is a 400 rather than a silently ignored key.
    npu::Options opts;
    std::unique_ptr<npu::Grammar> grammar;
    int max_tokens = 256;
    try {
        // Both read here, not before: a wrong type must be the 400 below, not a json::type_error that escapes
        // the guard and answers 500 with an empty body (engine#1106).
        max_tokens = npu::int_field(body, "max_completion_tokens", npu::int_field(body, "max_tokens", 256));
        opts = npu::options_from_body(body);
        // logprobs is served in the chat shape; the legacy /v1/completions shape is a different
        // object this route does not implement, and a silently dropped field is what it must not do.
        if (!chat && opts.sample.logprobs)
            throw std::runtime_error("logprobs: served on /v1/chat/completions, not on /v1/completions");
        if (opts.json_object || !opts.grammar.empty())
            grammar = std::make_unique<npu::Grammar>(opts.json_object ? npu::json_grammar() : opts.grammar);
    } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
        return;
    }
    const npu::Grammar* gram = grammar.get();
    // A prompt this long cannot fit the context whatever it tokenizes to; refuse it before
    // spending time on it.  An id prompt states its own length, so the check runs on ids below.
    if (!prompt_is_ids && prompt.size() > size_t(e.max_context) * 16) {
        res.status = 400;
        res.set_content(R"({"error":{"message":"prompt longer than the context"}})", "application/json");
        return;
    }
    bool stream = false;
    try {
        stream = npu::bool_field(body, "stream");  // engine#1106: a wrong type is a 400, not a 500 with no body
    } catch (const std::exception& ex) {
        res.status = 400;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
        return;
    }
    const std::string id = now_id(chat ? "chatcmpl-" : "cmpl-");
    const long created = long(std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string object = chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion";

    CallStats stats;  // filled by run(): token counts and the prefix decision
    // Returns the finish_reason the request earned: "stop" for an end-of-sequence token,
    // a stop sequence or a grammar that allows nothing more, "length" when the token
    // budget ran out.
    auto run = [&e, &req, prompt, prompt_ids, prompt_is_ids, max_tokens, chat, &opts, gram, &stats](
                   const std::function<bool(const std::string&, const std::vector<Emitted>&)>& on_text) {
        std::string finish = "stop";
        // The tokenizer is const and keeps no per-call state, so it runs before the lock.  A caller
        // that sent ids already stated the positions: they are used exactly, with no BOS inserted
        // on top (inserting one there would re-introduce the one-token prompt difference an id
        // prompt exists to remove).
        std::vector<int> ids = prompt_is_ids ? prompt_ids : e.tok->encode(prompt);
        // A chat template that opens with {{- bos_token }} (MiniCPM5 and other
        // llama-class models) needs the BOS id: the ChatML prompt built above never
        // carries it, and those models derail into their special tokens without it.
        if (!prompt_is_ids && chat && e.add_bos && e.bos_id >= 0) ids.insert(ids.begin(), e.bos_id);
        if (ids.empty()) throw std::runtime_error("empty prompt");
        stats.prompt_tokens = int(ids.size());
        // The KV cache grows with every position: the prompt and the answer together stay
        // inside the model's context.
        const int room = e.max_context - int(ids.size());
        if (room < 1) throw std::runtime_error("prompt longer than the context");
        const int limit = std::clamp(max_tokens, 0, room);
        std::lock_guard<std::mutex> lock(e.mu);

        const auto t_request = std::chrono::steady_clock::now();
        std::vector<float> logits;
        // Prefill in batched (M>1) passes of at most the kernel's row width: each
        // projection's weight stream is then read once per chunk instead of once
        // per token, which is where prefill time goes (the per-op kernels are
        // M=128 wide, so a per-token pass leaves ~99% of the array idle).
        // step_batch falls back to per-token step() for the architectures it does
        // not model, so this is behavior-preserving.
        // (issue #650) Resume from the longest saved prompt that is a prefix of this one, then
        // prefill only the remainder.  Reuse stops one token short of the whole prompt even on a
        // full-length hit: the answer is drawn from the last position's logits and a snapshot holds
        // state, not logits, so that token is run again.
        int resume = 0;
        if (e.prefix && e.prefix->enabled()) {
            const npu::PrefixDecision d = e.prefix->lookup(ids);
            stats.prefix = d;
            // One token short of both the slot and the prompt: the snapshot stops one token short (see the save
            // below), and the last token is run again for the logits the answer is drawn from.
            const int want = npu::resume_point(d, ids.size());
            // The forward clears itself only when a batch starts at 0 (step_batch), and a resumed
            // prefill starts above 0, so the previous request's state has to go first.
            if (d.slot >= 0 && want > 0) {
                e.fwd->reset();
                if (e.fwd->restore_state(e.prefix_state[size_t(d.slot)], want)) {
                    resume = want;
                    stats.prefix.cached_tokens = resume;
                } else {
                    e.fwd->reset();
                    stats.prefix = npu::PrefixDecision{-1, 0, "slot has no usable state"};
                }
            }
            // One line per decision, as FLM logs it, so the behaviour is checkable without a debugger.
            std::fprintf(stderr, "[prefix] slot=%d cached=%d of %zu reason=%s\n", stats.prefix.slot,
                         stats.prefix.cached_tokens, ids.size(), stats.prefix.reason.c_str());
        }
        {
            const int chunk_max = std::max(1, e.fwd->max_batch_rows());
            const int n = int(ids.size());
            for (int off = resume; off < n;) {
                const int take = std::min(chunk_max, n - off);
                // The state the prefix cache keeps is taken one token short of the prompt, because the answer is
                // drawn from the last position's logits and a snapshot holds state, not logits -- so the snapshot
                // is saved just before the last token and that token is run again on a resume.  On a GDN model
                // this is also the only exact point: the conv window and the recurrent state are updated in
                // place, so a snapshot taken *after* the last token cannot be restored to just before it.
                // Splitting the final chunk costs one launch.
                if (off + take == n && take > 1) {
                    const std::vector<int> head(ids.begin() + off, ids.begin() + off + take - 1);
                    if (!e.fwd->step_batch(head, off, logits))
                        throw std::runtime_error(std::string("forward step failed: ") + e.fwd->last_error());
                    off += take - 1;
                    continue;
                }
                if (off + take == n && e.prefix && e.prefix->enabled()) {
                    Q4nxNpuForward::StateSnapshot snap;
                    if (e.fwd->snapshot_state(off, snap)) {
                        std::string why;
                        const int slot = e.prefix->insert(ids, snap.bytes(), stats.prefix.slot, &why);
                        if (slot >= 0 && size_t(slot) < e.prefix_state.size()) {
                            e.prefix_state[size_t(slot)] = std::move(snap);
                            std::fprintf(stderr, "[prefix] saved slot=%d tokens=%d bytes=%zu %s\n", slot, off,
                                         e.prefix_state[size_t(slot)].bytes(), why.c_str());
                        }
                    }
                }
                const std::vector<int> chunk(ids.begin() + off, ids.begin() + off + take);
                if (!e.fwd->step_batch(chunk, off, logits))
                    throw std::runtime_error(std::string("forward step failed: ") + e.fwd->last_error());
                off += take;
            }
        }

        // Prompt processing ends here: every prompt token has been through the model and the first
        // token's distribution is in `logits`.
        const auto t_first = std::chrono::steady_clock::now();
        stats.prompt_ms =
            std::chrono::duration<double, std::milli>(t_first - t_request).count();

        // Sampler state: one generator per request, seeded so a run can be reproduced.
        std::mt19937_64 rng(npu::resolve_seed(opts.sample.seed));
        npu::Grammar::State grammar_state;
        std::vector<char> allowed;
        if (gram) {
            grammar_state = gram->start();
            // One decoded piece per vocabulary id: the mask below tests those against the
            // grammar state.  A megabyte of small strings, built once per process.
            if (e.tok_text.size() < logits.size()) {
                e.tok_text.clear();
                e.tok_text.reserve(logits.size());
                for (size_t i = 0; i < logits.size(); i++) e.tok_text.push_back(e.tok->decode(int(i)));
            }
        }
        // allowed[i] says token i can be appended without leaving the grammar.  The first
        // byte test rejects almost every token cheaply (a JSON object at the start accepts
        // a handful of first bytes out of 256), so only the survivors pay a full feed().
        auto refresh_allowed = [&]() {
            allowed.assign(logits.size(), 0);
            const npu::ByteMask first = gram->next_bytes(grammar_state);
            for (size_t i = 0; i < logits.size(); i++) {
                const std::string& t = e.tok_text[i];
                if (t.empty()) {
                    allowed[i] = 1;  // an empty piece cannot leave the grammar
                } else if (first.has(static_cast<unsigned char>(t[0]))) {
                    npu::Grammar::State probe = grammar_state;
                    allowed[i] = gram->feed(probe, t) ? 1 : 0;
                }
            }
        };

        std::string pending;  // decoded, not yet sent: a stop sequence may still match in it
        // Entries whose bytes are still held in `pending`, each with the bytes of its piece not yet
        // flushed: an entry is released by the flush that emits its last byte, so a piece the
        // stop-holdback splits across two chunks still reports exactly once.
        std::vector<std::pair<Emitted, size_t>> held;
        const bool want_lp = opts.sample.logprobs;
        auto take_held = [&held](size_t bytes) {
            std::vector<Emitted> out;
            size_t left = bytes;
            while (!held.empty() && left > 0) {
                if (held.front().second <= left) {
                    left -= held.front().second;
                    out.push_back(held.front().first);
                    held.erase(held.begin());
                } else {
                    held.front().second -= left;
                    left = 0;
                }
            }
            return out;
        };
        bool stopped_at_eos = false, stopped_at_stop = false;
        int produced = 0;
        for (int n = 0; n < limit; n++) {
            if (req.is_connection_closed()) break;  // the client is gone: free the NPU
            if (gram) refresh_allowed();
            // Raw distribution for a reference comparison: sample_token() takes logits by const
            // reference and the copy below happens before it runs, so what lands on disk is what
            // the model produced, not what the sampler chose.
            if (!e.dump_logits.empty()) {
                const std::string path = e.dump_logits + "/step_" + std::to_string(n) + ".bin";
                if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                    std::fwrite(logits.data(), sizeof(float), logits.size(), f);
                    std::fclose(f);
                }
            }
            const int next = npu::sample_token(logits, opts.sample, rng, gram ? &allowed : nullptr);
            // -1: the grammar allows no token here, so the answer ends where it is.  EOS is
            // not masked by the grammar on purpose: a constrained answer is still allowed to
            // stop early (the client validates it), which beats a runaway inside the mask.
            if (next < 0) { stopped_at_eos = true; break; }
            if (std::find(e.eos.begin(), e.eos.end(), next) != e.eos.end() ||
                (e.endoftext >= 0 && next == e.endoftext)) { stopped_at_eos = true; break; }
            if (gram) gram->feed(grammar_state, e.tok_text[next]);
            const std::string piece = e.tok->decode(next);
            pending += piece;
            // Raw distribution: sample_token() takes logits by const reference, so this is
            // pre-top_k/top_p/temperature and the grammar mask cannot narrow it.
            if (want_lp)
                held.emplace_back(Emitted{piece, npu::logprobs_of(logits, next, opts.sample.top_logprobs)}, piece.size());
            ++produced;
            const size_t stop_at = npu::find_stop(pending, opts.stop);
            if (stop_at != std::string::npos) {
                if (stop_at > 0) on_text(pending.substr(0, stop_at), take_held(stop_at));
                stopped_at_stop = true;
                break;
            }
            // Send what no stop sequence can still match, and keep the tail: the next
            // token may complete a match that starts in it.
            const size_t cut = complete_utf8(pending);
            const size_t hold = npu::stop_holdback(opts.stop);
            size_t emit = cut > hold ? cut - hold : 0;
            while (emit > 0 && (static_cast<unsigned char>(pending[emit]) & 0xC0) == 0x80) --emit;  // keep whole characters
            if (emit > 0) {
                const std::string out = pending.substr(0, emit);
                pending.erase(0, emit);
                if (!on_text(out, take_held(emit))) break;
            }
            // No further token will be sampled after this one (limit reached), so the
            // decode step that would only produce the NEXT position's logits is a whole
            // weight stream -- 8.7 s on the 27B -- spent on a result nobody reads.  At
            // max_tokens=1 that is half the request's latency.
            if (n + 1 >= limit) break;
            if (!e.fwd->step(next, (int)ids.size() + n, logits))
                throw std::runtime_error(std::string("forward step failed: ") + e.fwd->last_error());
        }
        if (!stopped_at_stop) {
            const size_t cut = complete_utf8(pending);
            if (cut > 0) on_text(pending.substr(0, cut), take_held(cut));
        }
        stats.completion_tokens = produced;
        stats.predicted_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_first).count();
        if (produced >= limit && !stopped_at_eos && !stopped_at_stop) finish = "length";
        return finish;
    };

    // AMD's two stages, as milliseconds, and nothing derived from them: the protocol's consumers
    // (onnx/bench_npu_serve.py) compute their own rates, and a rate of ours would be a second
    // definition of the same stage -- off by one token, because the first token belongs to TTFT in
    // that protocol.  Two rates for one stage is exactly how two rows end up disagreeing later.
    auto timings_json = [](const CallStats& st) {
        return json{{"prompt_ms", st.prompt_ms}, {"predicted_ms", st.predicted_ms}};
    };

    // OpenAI's usage object plus the two fields issue #650 reports: how many prompt tokens came
    // from a saved prefix, and what the cache decided (which slot, and why).
    auto usage_json = [](const CallStats& st) {
        return json{{"prompt_tokens", st.prompt_tokens},
                    {"completion_tokens", st.completion_tokens},
                    {"total_tokens", st.prompt_tokens + st.completion_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", st.prefix.cached_tokens}}},
                    {"flm_prefix_cache", {{"slot", st.prefix.slot},
                                          {"cached_tokens", st.prefix.cached_tokens},
                                          {"reason", st.prefix.reason}}}};
    };

    // OpenAI's logprobs.content entry: the token's text, its log-probability, and the most likely
    // alternatives with theirs.  Only the chat shape is served.
    auto logprob_json = [&e](const Emitted& x) {
        json j{{"token", x.text}, {"logprob", x.lp.logprob}, {"top_logprobs", json::array()}};
        for (const auto& t : x.lp.top)
            j["top_logprobs"].push_back({{"token", e.tok->decode(t.id)}, {"logprob", t.logprob}});
        return j;
    };
    auto logprobs_json = [&logprob_json](const std::vector<Emitted>& entries) {
        json content = json::array();
        for (const auto& x : entries) content.push_back(logprob_json(x));
        return json{{"content", content}};
    };

    auto wrap = [&](const std::string& content, const std::string& finish, const std::vector<Emitted>& lps) {
        json out;
        out["id"] = id;
        out["object"] = object;
        out["created"] = created;
        out["model"] = e.id;
        if (chat) {
            // (issue #652 item 3) one shared parser splits what the model wrote into OpenAI's three
            // parts; a reply that is only tool calls says content: null.
            const npu::ParsedAnswer ans =
                npu::parse_answer(content, npu::AnswerFamily::qwen3, opts.tools_sent);
            json message{{"role", "assistant"}};
            message["content"] = ans.content_null ? json(nullptr) : json(ans.content);
            if (!ans.reasoning.empty()) message["reasoning_content"] = ans.reasoning;
            if (!ans.calls.empty()) {
                json calls = json::array();
                for (const auto& c : ans.calls)
                    calls.push_back({{"id", c.id},
                                     {"type", "function"},
                                     {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
                message["tool_calls"] = calls;
            }
            std::string why = finish;
            if (!ans.calls.empty() && why == "stop") why = "tool_calls";
            json choice{{"index", 0}, {"message", message}, {"finish_reason", why}};
            if (opts.sample.logprobs) {
                // (issue #651 item 3) entries exist for content tokens only: reasoning and call
                // markup get none, which is what the parser's own boundary says.
                std::vector<std::string> texts;
                texts.reserve(lps.size());
                for (const auto& x : lps) texts.push_back(x.text);
                const std::vector<char> keep = npu::content_mask(texts, npu::AnswerFamily::qwen3, opts.tools_sent);
                std::vector<Emitted> content_lps;
                for (size_t i = 0; i < lps.size(); i++)
                    if (i < keep.size() && keep[i]) content_lps.push_back(lps[i]);
                choice["logprobs"] = logprobs_json(content_lps);
            }
            out["choices"] = json::array({choice});
        } else {
            out["choices"] = json::array({{{"index", 0}, {"text", content}, {"finish_reason", finish}}});
        }
        out["usage"] = usage_json(stats);
        out["timings"] = timings_json(stats);
        // A piece can be half a character and dump() throws on invalid UTF-8: replace rather than
        // turn a reported logprob into a 500.
        return out.dump(-1, ' ', false, json::error_handler_t::replace);
    };

    try {
        if (!stream) {
            std::string content;
            std::vector<Emitted> lps;
            const std::string finish = run([&](const std::string& t, const std::vector<Emitted>& lp) {
                content += t;
                lps.insert(lps.end(), lp.begin(), lp.end());
                return true;
            });
            res.set_content(wrap(content, finish, lps), "application/json");
        } else {
            // Build the SSE frames while the handler still owns its locals.  The
            // content provider below runs AFTER this handler returns, so it must
            // not capture anything by reference (id/created/object die here) --
            // that was a dangling-reference crash (`std::system_error`).  Run the
            // forward here too, so it is not invoked from httplib's write path.
            json chunk_tpl;
            chunk_tpl["id"] = id;
            chunk_tpl["object"] = "chat.completion.chunk";
            chunk_tpl["created"] = created;
            chunk_tpl["model"] = e.id;
            auto chunks = std::make_shared<std::vector<std::string>>();
            // The same splitter as the non-streamed path, fed piece by piece, so a streamed reply
            // and a non-streamed one cannot disagree about what is reasoning and what is content.
            npu::AnswerStream stream(npu::AnswerFamily::qwen3, opts.tools_sent);
            auto emit_pieces = [&](const std::string& t, const std::vector<Emitted>& lp) {
                std::vector<std::pair<bool, std::string>> pieces;
                stream.feed(t, pieces);
                for (const auto& [is_reasoning, text] : pieces) {
                    json chunk = chunk_tpl;
                    json delta = is_reasoning ? json{{"reasoning_content", text}} : json{{"content", text}};
                    if (chat)
                        chunk["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
                    else
                        chunk["choices"] = json::array({{{"index", 0}, {"text", text}, {"finish_reason", nullptr}}});
                    // logprobs describe content tokens only (issue #651 item 3): a reasoning delta
                    // carries no entries.
                    if (!is_reasoning && opts.sample.logprobs && !lp.empty())
                        chunk["choices"][0]["logprobs"] = logprobs_json(lp);
                    chunks->push_back("data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");
                }
                return true;
            };
            run(emit_pieces);
            {
                std::vector<std::pair<bool, std::string>> tail;
                stream.finish(tail);
                for (const auto& [is_reasoning, text] : tail) {
                    json chunk = chunk_tpl;
                    json delta = is_reasoning ? json{{"reasoning_content", text}} : json{{"content", text}};
                    chunk["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
                    chunks->push_back("data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");
                }
                if (!stream.calls().empty()) {
                    // The calls arrive whole, in one chunk: OpenAI streams argument fragments, and a
                    // client that appends per index handles either; sending them once is what makes
                    // the streamed sum equal the non-streamed reply.
                    json calls = json::array();
                    for (const auto& c : stream.calls())
                        calls.push_back({{"index", int(calls.size())},
                                         {"id", c.id},
                                         {"type", "function"},
                                         {"function", {{"name", c.name}, {"arguments", c.arguments}}}});
                    json chunk = chunk_tpl;
                    json delta{{"tool_calls", calls}};
                    if (!chunk_tpl.contains("object") || chunk_tpl["object"].is_null()) chunk["object"] = "chat.completion.chunk";
                    chunk["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", "tool_calls"}}});
                    chunks->push_back("data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");
                }
            }
            {
                json tail = chunk_tpl;
                tail["choices"] = json::array();
                tail["usage"] = usage_json(stats);
            tail["timings"] = timings_json(stats);
                chunks->push_back("data: " + tail.dump() + "\n\n");
            }
            auto cursor = std::make_shared<size_t>(0);
            res.set_chunked_content_provider("text/event-stream",
                [chunks, cursor](size_t, httplib::DataSink& sink) {
                    while (*cursor < chunks->size()) {
                        const std::string& d = (*chunks)[*cursor];
                        if (!sink.write(d.data(), d.size())) return false;
                        ++*cursor;
                    }
                    const std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    return true;
                });
        }
    } catch (const std::exception& ex) {
        res.status = 500;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
    }
}

// The per-question input limit of the decision route: the text path's own 32,768 tokens, or the
// backbone's context when that is smaller. Either overflowing is the same failure to the model, so
// the smaller one is the limit the request is checked against.
size_t systemone_max_input(const ForwardEngine& e) {
    const size_t ctx = e.max_context > 0 ? size_t(e.max_context) : npu::kSystemOneMaxInputTokens;
    return std::min(ctx, npu::kSystemOneMaxInputTokens);
}

// DECISION_API_KEY, when set, gates the decision routes -- upstream d3_server.py's own access control,
// so a deployment keeps the property it had when it pointed vLLM Semantic Router at that server.
// /health stays open for probes and reports that authentication is on.
bool decision_authorized(const httplib::Request& req) {
    const char* key = std::getenv("DECISION_API_KEY");
    if (!key || !*key) return true;
    const std::string want = std::string("Bearer ") + key;
    const std::string got = req.get_header_value("Authorization");
    // Constant-time compare: a key checked byte by byte with an early exit leaks its prefix.
    unsigned diff = unsigned(got.size() ^ want.size());
    for (size_t i = 0; i < got.size() && i < want.size(); i++) diff |= unsigned(got[i] ^ want[i]);
    return diff == 0;
}

// POST /v1/systemone (and its /v1/decisions alias): one backbone prefill per question, then the
// pack's readout over the pooled last hidden state. No text is generated, so there is no sampling,
// no grammar and no stream -- the answer is a probability for every option of the question.
void handle_systemone(ForwardEngine& e, const httplib::Request& req, httplib::Response& res) {
    auto refuse = [&res](int status, const json& detail) {
        res.status = status;
        res.set_content(json{{"detail", detail}}.dump(-1, ' ', false, json::error_handler_t::replace),
                        "application/json");
    };
    if (!e.head) {
        refuse(503, "no decision head loaded: " + e.head_error);
        return;
    }
    // ordered_json: the request's key order is part of the prompt (npu/systemone.h).
    nlohmann::ordered_json body;
    try {
        body = nlohmann::ordered_json::parse(req.body);
    } catch (const std::exception& ex) {
        refuse(422, std::string("could not read the body as JSON (") + ex.what() + ")");
        return;
    }
    npu::SystemOneParsed parsed;
    try {
        parsed = npu::systemone_request_of(body);
    } catch (const npu::SystemOneRefusal& ex) {
        refuse(422, ex.what());
        return;
    }
    // A malformed question makes the whole request a refusal, and nothing runs: a partial answer
    // would leave the caller unable to tell an unanswered question from an unasked one.
    if (!parsed.invalid.empty()) {
        nlohmann::ordered_json bad = nlohmann::ordered_json::object();
        for (const auto& [id, why] : parsed.invalid) bad[id] = why;
        refuse(422, nlohmann::ordered_json{{"invalid_questions", bad}});
        return;
    }
    const size_t limit = systemone_max_input(e);
    const std::vector<std::string>& codes = e.head->config().codes;
    nlohmann::ordered_json answers = nlohmann::ordered_json::object();
    size_t input_tokens = 0;
    try {
        std::lock_guard<std::mutex> lock(e.mu);
        std::vector<float> logits;
        for (const auto& q : parsed.questions) {
            const std::string prompt = npu::systemone_prompt(parsed.state, q.question, codes);
            const std::vector<int> ids = e.tok->encode(prompt);
            // Nothing is truncated: an over-long question is an error entry and the next one still runs.
            if (ids.size() > limit) {
                answers[q.id] = npu::systemone_question_error(q.question.kind, "max_length_exceeded",
                                                              npu::systemone_length_message(ids.size(), limit));
                continue;
            }
            e.fwd->reset();
            if (!e.fwd->step_batch(ids, 0, logits))
                throw std::runtime_error(std::string("forward step failed: ") + e.fwd->last_error());
            std::vector<int> rows(q.question.keys.size());  // option i is readout row i (pack code order)
            for (size_t i = 0; i < rows.size(); i++) rows[i] = int(i);
            try {
                const std::vector<float> probs =
                    e.head->probabilities_over(e.head->logits(e.fwd->last_hidden()), rows);
                answers[q.id] = npu::d3::answer_of(q.question, probs);
            } catch (const std::exception& ex) {
                // The model ran but its output cannot be read as an answer: that is this question's
                // problem, not the request's (upstream's invalid_model_output).
                answers[q.id] = npu::systemone_question_error(q.question.kind, "invalid_model_output", ex.what());
                continue;
            }
            input_tokens += ids.size();
        }
    } catch (const std::exception& ex) {
        res.status = 500;
        res.set_content(json{{"detail", ex.what()}}.dump(-1, ' ', false, json::error_handler_t::replace),
                        "application/json");
        return;
    }
    res.set_content(npu::systemone_reply(e.id, answers, input_tokens).dump(-1, ' ', false,
                                                                          json::error_handler_t::replace),
                    "application/json");
}

}  // namespace

int run_forward_serve(int argc, char** argv) {
    std::string model, kernels, host = "127.0.0.1", alias, dump_logits;
    int port = 8000, streams = 1;
    npu::PrefixOptions prefix_opts;  // --prefix-slots / --prefix-cache-mb / --prefix-min-gain (#650)
    for (int i = 0; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model = next();
        else if (a == "--kernels") kernels = next();
        else if (a == "-p" || a == "--port") port = std::stoi(next());
        else if (a == "--host") host = next();
        else if (a == "--alias") alias = next();
        else if (a == "--prefix-slots") prefix_opts.slots = std::stoi(next());
        else if (a == "--prefix-cache-mb") prefix_opts.cache_mb = std::stoul(next());
        else if (a == "--prefix-min-gain") prefix_opts.min_gain = std::stoi(next());
        else if (a == "--npu-streams") streams = std::stoi(next());
        else if (a == "--dump-logits") dump_logits = next();
        else throw std::runtime_error("unknown option " + a);
    }
    if (model.empty())
        throw std::runtime_error("usage: 1bit forward-serve -m <model.q4nx> --kernels <dir> [-p PORT] [--npu-streams N] [--dump-logits <dir>]");
    if (streams < 1 || streams > kNpuMaxStreams)
        throw std::runtime_error("--npu-streams takes 1.." + std::to_string(kNpuMaxStreams) +
                                 " (the device's hwctx limit), not " + std::to_string(streams));
    if (!std::filesystem::exists(model)) throw std::runtime_error(model + " does not exist");

    const std::string dir = std::filesystem::path(model).parent_path().string();
    const std::string cfg = dir + "/config.json";
    const std::string tok_json = dir + "/tokenizer.json";

    ForwardEngine e;
    e.id = alias.empty() ? std::filesystem::path(dir).filename().string() : alias;
    if (!dump_logits.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(dump_logits, ec);
        if (ec || !std::filesystem::is_directory(dump_logits))
            throw std::runtime_error("--dump-logits: cannot use " + dump_logits + " as a directory");
        e.dump_logits = dump_logits;
        std::fprintf(stderr, "1bit forward-serve: dumping raw logits to %s\n", dump_logits.c_str());
    }
    e.tok = std::make_unique<npu::Tokenizer>(tok_json);
    e.bos_id  = json_int(cfg, "bos_token_id", -1);
    e.add_bos = json_int(cfg, "add_bos_token", 0) != 0;
    e.eos = npu::eos_token_ids_from_config(cfg);
    e.endoftext = e.tok->token_id("<|endoftext|>");
    // q4nx_config_int finds the key at any depth (a text_config's too).
    e.max_context = q4nx_config_int(cfg.c_str(), "max_position_embeddings", 0);
    if (e.max_context <= 0) e.max_context = 32768;

    // #963: the chat route's prompt is ChatML for every container, and a tokenizer with no
    // <|im_start|>/<|im_end|> added token cannot represent it -- the marker becomes ordinary pieces and
    // the model reads text its own template never produces (zyphra/ZR1-1.5B: 33 tokens rendered
    // against 15 under the model's template).  Refused per request rather than refused at load,
    // because /v1/completions with token ids (#938) is the way to serve such a container correctly.
    if (!npu::chatml_supported(*e.tok)) {
        e.chatml_ok = false;
        std::string specials;
        std::vector<std::pair<const char*, int>> named = {{"bos", e.bos_id}};
        for (int id : e.eos) named.emplace_back("eos", id);
        for (const auto& [label, id] : named) {
            const std::string text = id >= 0 ? e.tok->decode(id) : std::string();
            if (text.empty()) continue;
            specials += std::string(specials.empty() ? "" : ", ") + label + " " + text + " (" +
                        std::to_string(id) + ")";
        }
        e.chatml_refusal =
            "this container's tokenizer has no <|im_start|> / <|im_end|>, so the ChatML prompt the chat route "
            "renders would be tokenized into pieces the model never sees (#963); send /v1/completions with "
            "prompt as token ids (#938) and render the model's own template" +
            (specials.empty() ? std::string() : " (model specials: " + specials + ")");
        std::fprintf(stderr, "1bit forward-serve: chat route disabled: %s\n", e.chatml_refusal.c_str());
    }

    // The prompt cache (#650): --prefix-slots 0 turns it off, and an invalid value is refused here
    // rather than clamped, like every other option on this command.
    if (prefix_opts.slots > 0) {
        e.prefix = std::make_unique<npu::PrefixCache>(prefix_opts);
        e.prefix_state.resize(size_t(prefix_opts.slots));
        std::fprintf(stderr,
                     "1bit forward-serve: prefix cache on: %d slots, %zu MB, min gain %d tokens\n",
                     prefix_opts.slots, prefix_opts.cache_mb, prefix_opts.min_gain);
    }

    e.fwd = std::make_unique<Q4nxNpuForward>();

    // Who may have the NPU (app/npu_device_lock.h): one process by default, N when the
    // operator asks with --npu-streams. hwctx_limit = 16 is not what bounds a second
    // client -- the array partition is: 8 columns, and this build's kernel set asks for
    // all 8, so a second client is refused (-ENODEV / code 102) once 0003 is in the
    // driver, and shared the tiles and wedged the array before it (engine#1084). The
    // default stays exclusive because a kernel set that does leave columns free still
    // shares one DMA engine and copies the weights per stream: two engines on two models
    // measured 34.1 % / 70.4 % of solo, 0.41x aggregate (docs/npu.md "Two streams on one
    // NPU").
    // Held for the life of the process (fd not closed), exactly like the per-process
    // ELF runner (npu_engine_universal), which takes the same exclusive lock.
    if (!std::getenv("NPU_NO_DEVICE_LOCK"))
        e.lock_fd = take_npu_device_lock(streams);

    // Offline mode (npu/forward/include/q4nx_forward.h, init_host): with
    // NPU_INFER_HOST_GEMM=1 the forward runs the model with host GEMMs and never opens
    // the device, which is how the sampling and grammar path is exercised without an
    // NPU.  Open the device only when the forward is going to use it.
    const bool host_gemm = std::getenv("NPU_INFER_HOST_GEMM") != nullptr;
    std::unique_ptr<xrt::device> dev;
    if (!host_gemm) dev = std::make_unique<xrt::device>(0);
    const bool ok = dev ? e.fwd->init(*dev, model.c_str(), kernels) : e.fwd->init_host(model.c_str(), kernels);
    if (!ok) throw std::runtime_error(std::string("forward init failed: ") + e.fwd->last_error());

    // A d3 pack keeps its decision head next to model.q4nx (npu/decision.h). Both files present means
    // the decision routes come up beside the chat ones; a pack without them is a chat model, which is
    // the normal case, so this is not an error.
    const std::string decision_cfg = dir + "/decision_config.json";
    const std::string decision_readout = dir + "/decision.q4nx";
    if (std::filesystem::exists(decision_cfg) && std::filesystem::exists(decision_readout)) {
        auto head = std::make_unique<npu::DecisionHead>();
        if (head->load(dir)) {
            std::fprintf(stderr, "1bit forward-serve: d3 decision head: %zu options, hidden %zu, %s, pooling %s\n",
                         head->options(), head->hidden(), head->config().format_id.c_str(),
                         head->config().pooling.c_str());
            e.head = std::move(head);
        } else {
            // Present but unusable is a 503 with the reason, not a silent fall back to chat only.
            e.head_error = head->error();
            std::fprintf(stderr, "1bit forward-serve: d3 decision head unusable: %s\n", e.head_error.c_str());
        }
    } else {
        e.head_error = "no decision_config.json + decision.q4nx next to " + model;
    }

    std::atomic<bool> ready{false};
    httplib::Server srv;
    install_request_guard(srv, host);  // Host / Origin (app/http_guard.h)
    // /health is what a probe reads: the d3 fields appear only when a decision head is loaded, so a
    // chat-only deployment keeps the body it had before this route existed.
    auto health = [&](httplib::Response& r) {
        r.status = ready ? 200 : 503;
        if (!ready) {
            r.set_content(R"({"status":"loading"})", "application/json");
            return;
        }
        json h{{"status", "ok"}};
        if (e.head) {
            h["model"] = e.id;
            h["max_input_tokens"] = systemone_max_input(e);
            h["modalities"] = json::array({"text"});
            const char* key = std::getenv("DECISION_API_KEY");
            h["authentication"] = key && *key;
        }
        r.set_content(h.dump(), "application/json");
    };
    srv.Get("/health", [&](const httplib::Request&, httplib::Response& r) { health(r); });
    srv.Get("/v1/health", [&](const httplib::Request&, httplib::Response& r) { health(r); });
    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& r) {
        r.set_content(json{{"object", "list"},
                           {"data", json::array({{{"id", e.id}, {"object", "model"}, {"owned_by", "1bit"}}})}}.dump(),
                      "application/json");
    });
    auto post = [&](const httplib::Request& q, httplib::Response& r, bool chat) {
        if (!ready) {
            r.status = 503;
            r.set_content(R"({"error":{"message":"model is loading"}})", "application/json");
            return;
        }
        handle_generate(e, q, r, chat);
    };
    srv.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& r) { post(q, r, true); });
    srv.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& r) { post(q, r, false); });
    // The d3 decision routes (#653), with upstream's alias and its own 422/401 refusals: this is the
    // interop contract with vLLM Semantic Router, so it follows d3_server.py rather than the chat
    // route's 400 habit (docs/npu.md, "Decisions").
    auto decide = [&](const httplib::Request& q, httplib::Response& r) {
        if (!ready) {
            r.status = 503;
            r.set_content(R"({"detail":"model is loading"})", "application/json");
            return;
        }
        if (!decision_authorized(q)) {
            r.status = 401;
            r.set_header("WWW-Authenticate", "Bearer");
            r.set_content(R"({"detail":"a valid bearer key is required"})", "application/json");
            return;
        }
        handle_systemone(e, q, r);
    };
    srv.Post("/v1/systemone", decide);
    srv.Post("/v1/decisions", decide);
    srv.Get("/v1/systemone/models", [&](const httplib::Request& q, httplib::Response& r) {
        if (!decision_authorized(q)) {
            r.status = 401;
            r.set_header("WWW-Authenticate", "Bearer");
            r.set_content(R"({"detail":"a valid bearer key is required"})", "application/json");
            return;
        }
        if (!e.head) {
            r.status = 503;
            r.set_content(json{{"detail", "no decision head loaded: " + e.head_error}}.dump(), "application/json");
            return;
        }
        r.set_content(npu::systemone_models(e.id, systemone_max_input(e)).dump(), "application/json");
    });

    if (!srv.bind_to_port(host, port)) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
    std::thread listener([&] { srv.listen_after_bind(); });
    struct Stop {
        httplib::Server& s;
        std::thread& t;
        ~Stop() { s.stop(); if (t.joinable()) t.join(); }
    } stop{srv, listener};

    ready = true;
    std::fprintf(stderr, "1bit forward-serve: %s ready on http://%s:%d (%s)", e.id.c_str(), host.c_str(), port,
                 e.fwd->backend().c_str());
    if (streams > 1) std::fprintf(stderr, " [npu-streams=%d]", streams);
    std::fprintf(stderr, "\n");

    while (true) std::this_thread::sleep_for(std::chrono::seconds(3600));
    return 0;
}

}  // namespace onebit
