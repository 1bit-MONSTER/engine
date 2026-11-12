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
#include "q4nx_cache.h"   // engine#2263: the sweep after a forward load

#include "../npu/forward/include/file_create.h"   // dumps are created 0600, never through a symlink
#include "npu_decode_note.h"

#include "../npu/forward/include/q4nx_forward.h"
#include "../npu/target/xdna/xdna_target.h"  // XdnaTarget::kName for --npu-target
#include "npu_device_lock.h"
#include "npu_answer_parser.h"
#include "npu_chatml.h"
#include "npu_template.h"  // the container's own chat template (#963 option 2)
#include "npu_grammar.h"
#include "npu_load_retry.h"  // engine#1241: a refused NPU context names another client, not a dead device
#include "npu_prefix_cache.h"
#include "npu_prompt_ids.h"
#include "npu_sampling.h"
#include "npu_spec_sampler.h"
#include "npu_stop_tokens.h"
#include "stream_bridge.h"
#include "npu_systemone.h"
#include "tokenizer.h"
#include "../npu/decision.h"  // the d3 readout head (npu/decision.h, issue #653)
#include "../npu/mtp/prompt_lookup.h"  // --spec-lookup: the speculative loop with a prompt-lookup drafter (#1781)
#include "../npu/dflash_head.h"        // --spec-draft: the same loop with a DFlash draft package (#1781)
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

// A prompt that leaves no room for an answer inside the context (the model's, or --ctx-size).
struct ContextOverflow : std::runtime_error {
    size_t prompt_tokens;
    int context;
    ContextOverflow(size_t n, int ctx)
        : std::runtime_error("request (" + std::to_string(n) + " tokens) exceeds the available context size (" +
                             std::to_string(ctx) + " tokens)"),
          prompt_tokens(n), context(ctx) {}
};

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
    // Since #963 option 2 the route renders the container's own template when it declares one, so
    // the flag is "this container can be served on the chat route at all", not "its template is
    // ChatML": a container with a template needs only that the build carries a renderer.
    bool chat_ok = true;
    std::string chat_refusal;  // empty unless !chat_ok; the 400's message
    // The template the container declares, its sha256 and where it came from: rendered per request
    // when present (app/npu_template.h), else ChatML when the tokenizer has the anchors.
    npu::ChatTemplate tmpl;
    // The markup this container's replies are split by (app/npu_answer_parser.h): the template's own when
    // the template is what gets rendered, else qwen3, which is what the ChatML fallback prompts for.
    npu::AnswerFamily answer_family = npu::AnswerFamily::qwen3;
    std::string bos_text;  // the bos/eos *strings* the template's {{ bos_token }} renders
    std::string eos_text;
    // decode() of every vocabulary id, built on the first constrained request: the GBNF
    // mask tests each token's text against the grammar state, per token.
    std::vector<std::string> tok_text;
    // Saved prompts (issue #650): the slot table decides what to keep, prefix_state holds each
    // slot's attention KV and GDN state.  Only ever touched under mu.
    std::unique_ptr<npu::PrefixCache> prefix;
    std::vector<Q4nxNpuForward::StateSnapshot> prefix_state;
    // --prefix-pin (engine#1785): keep the state at the point where a request stops agreeing with the
    // previous one, so single-turn requests under one system prompt resume from it (PrefixCache::pin_point).
    bool prefix_pin = false;
    std::vector<int> prev_ids;  // the previous request's prompt, what pin_point() compares against
    // The d3 decision head (#653), loaded when decision_config.json and decision.q4nx sit next to the
    // model: it is what turns this server into a decision endpoint as well as a chat one.
    std::unique_ptr<npu::DecisionHead> head;
    std::string head_error;  // why there is no head, for the 503 the decision route answers
    // --dump-logits <dir>: write each generated position's raw logits (float32, vocabulary-wide)
    // as step_<response>_<n>.bin (`<response>` is 1-based per serve, so two prompts in one run cannot overwrite each
    // other's steps -- engine#803; a single-prompt run is step_1_*).  The OpenAI API cannot express this -- it caps
    // top_logprobs at 100 -- so a
    // KL-divergence comparison against another stack (engine#671 gate b) needs the distribution
    // itself.  Empty means off, which is the default: nothing is written unless asked.
    std::string dump_logits;
    // --spec-lookup K (engine#1781): a greedy request without grammar decodes through mtp::run_from() with
    // the prompt-lookup drafter, up to K drafts verified per batch; 0 = off (the default). --spec-fallback X: stop
    // drafting for the rest of a reply once its last 3 rounds averaged fewer than X tokens (0 = never).
    int spec_lookup = 0;
    double spec_fallback = 0.0;
    // --spec-sample: a sampled request (temperature > 0) takes the speculative path too, its drafts kept or refused by
    // the rule that keeps the target's distribution (npu/mtp/spec_accept.h through mtp::RowSampler). Off by default: the
    // reply then follows the same distribution but not the plain loop's random stream, so a seed gives a different
    // reply than it does without the flag.
    bool spec_sample = false;
    bool host_gemm = false;   // NPU_INFER_HOST_GEMM: the forward runs on the host, with no launch width of its own
    // --spec-draft <dir> (engine#1781): a DFlash draft package as the drafter instead of the prompt lookup, up to
    // spec_draft_k drafts a round. The forward writes the draft's tap layers into draft_taps on every pass; the sink
    // is emptied at the start of each request, so a request that resumed from a saved prefix has no rows for the
    // positions it did not run, and its head refuses to draft (it decodes with plain steps, the same text).
    std::unique_ptr<npu::dflash::DraftModel> draft;
    npu::dflash::DraftWeights draft_w;
    std::vector<float> draft_hproj;   // generation 2: the selector's hidden projection, decoded once
    std::vector<float> draft_taps;
    int spec_draft_k = 0;
    // The dump's response index. `step_<n>.bin` used the decode index alone, so a second request overwrote the first
    // request's steps: a multi-prompt comparison (the KLD row, engine#803) then silently compared only the LAST prompt's
    // dumps and looked complete. Numbering the response too makes the set of files say how many responses it holds.
    std::atomic<int> dump_seq{0};
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
    // engine#2111: each generated token's text, in order, and how many of them were reasoning. -1 = not counted (a
    // /v1/completions reply is raw text with no split), so the usage object then carries no details at all.
    std::vector<std::string> pieces;
    int reasoning_tokens = -1;
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

// `bridge` is where a streamed reply's frames go as they are produced (handle_generate below runs this on its own
// thread for a streaming request); null means the caller wants the whole response in `res`.
void handle_generate_impl(ForwardEngine& e, const httplib::Request& req, httplib::Response& res, bool chat,
                          npu::StreamBridge* bridge) {
    // One id per response, taken before any step: the dumps of two prompts in one serve must not share names.
    const int dump_req = e.dump_logits.empty() ? 0 : ++e.dump_seq;
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
    // --prefix-pin: the leading system messages and a probe user message (npu::pin_probe_messages), rendered.  Whatever
    // of that the full prompt starts with is the part every question under this system prompt shares, so it can be pinned on
    // the FIRST request (PrefixCache::pin_at) -- comparing with the previous request only finds it on the second,
    // and can run a few tokens into the question when two questions open with the same words.  Empty = the
    // request has no system message, or the route is not pinning.
    std::string system_prompt;
    try {
        if (chat) {
            if (!e.chat_ok) throw std::runtime_error(e.chat_refusal);
            if (e.tmpl.present() && npu::renderer_built()) {
                // The request's own knobs reach the template as variables, the way llama-server hands
                // `chat_template_kwargs` over: FastFlowLM's packs branch on `user_system_prompt`, Qwen3's
                // on `enable_thinking`.  Ordered re-read of the same bytes: `body` is a std::map, and a
                // template that serializes its input (a tools block through `tojson`) renders the
                // request's key order into the prompt, so it has to be the caller's order, not ours.
                const nlohmann::ordered_json ordered = nlohmann::ordered_json::parse(req.body);
                // The functions the request offers go the same way (npu::request_template_vars): the
                // template is where a model learns what it may call.
                const nlohmann::ordered_json extra = npu::request_template_vars(ordered, npu::thinking_from_body(body));
                // An earlier call's arguments arrive as a JSON string; the template is written against the
                // object (npu::messages_for_template).
                prompt = npu::render_chat_template(e.tmpl, npu::messages_for_template(ordered.at("messages")),
                                                   /*add_generation_prompt=*/true, extra, e.bos_text, e.eos_text);
                if (e.prefix_pin) {
                    // A template that cannot render the probe is no reason to refuse the request: it only means
                    // there is no early pin, and the log says so.
                    try {
                        const nlohmann::ordered_json probe = npu::pin_probe_messages(ordered.at("messages"));
                        if (!probe.empty())
                            system_prompt = npu::render_chat_template(e.tmpl, probe, /*add_generation_prompt=*/false, extra,
                                                                      e.bos_text, e.eos_text);
                    } catch (const std::exception& ex) {
                        system_prompt.clear();
                        std::fprintf(stderr, "[prefix] no early pin: the template did not render the system turn (%s)\n", ex.what());
                    }
                }
            } else {
                prompt = npu::chatml(body.at("messages"), npu::thinking_from_body(body));
                if (e.prefix_pin) {
                    const json probe = npu::pin_probe_messages(body.at("messages"));
                    if (!probe.empty()) system_prompt = npu::chatml(probe, npu::thinking_from_body(body));
                }
            }
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
    auto run = [&e, &req, prompt, system_prompt, prompt_ids, prompt_is_ids, max_tokens, chat, &opts, gram, &stats, dump_req, bridge](
                   const std::function<bool(const std::string&, const std::vector<Emitted>&)>& on_text) {
        std::string finish = "stop";
        // The tokenizer is const and keeps no per-call state, so it runs before the lock.  A caller
        // that sent ids already stated the positions: they are used exactly, with no BOS inserted
        // on top (inserting one there would re-introduce the one-token prompt difference an id
        // prompt exists to remove).
        std::vector<int> ids = prompt_is_ids ? prompt_ids : e.tok->encode(prompt);
        // A chat template that opens with {{- bos_token }} (MiniCPM5 and other
        // llama-class models) needs the BOS id: the ChatML prompt built above never
        // carries it, and those models derail into their special tokens without it.  A rendered
        // template may already have emitted it, and a second BOS is a prompt no HF tokenizer
        // produces.
        if (!prompt_is_ids && chat && e.add_bos && e.bos_id >= 0 && (ids.empty() || ids.front() != e.bos_id))
            ids.insert(ids.begin(), e.bos_id);
        if (ids.empty()) throw std::runtime_error("empty prompt");
        stats.prompt_tokens = int(ids.size());
        // The KV cache grows with every position: the prompt and the answer together stay
        // inside the model's context.
        const int room = e.max_context - int(ids.size());
        if (room < 1) throw ContextOverflow(ids.size(), e.max_context);
        const int limit = std::clamp(max_tokens, 0, room);
        std::lock_guard<std::mutex> lock(e.mu);
        e.draft_taps.clear();   // --spec-draft: this request's tap rows start at its first position run

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
        // (engine#1785) Where this request should leave a pinned snapshot, as a position: the state is saved
        // one token short of the shared part for the same reason the whole-prompt snapshot below is saved one
        // token short of the prompt. -1 = no pin for this request. pin_point() is 0 when a slot (the pin
        // itself, on every request after the one that made it) already covers the shared part, and it is
        // always less than the prompt's length, so this cut is never the last-token cut below.
        int pin_cut = -1;
        size_t pin_tokens = 0;
        if (e.prefix_pin && e.prefix && e.prefix->enabled()) {
            // The end of the system turn first: it is known on the first request and it is the right place.  Only
            // a request with no system message falls back to what it shares with the previous request.
            if (!system_prompt.empty()) {
                std::vector<int> lead = e.tok->encode(system_prompt);
                if (chat && e.add_bos && e.bos_id >= 0 && (lead.empty() || lead.front() != e.bos_id))
                    lead.insert(lead.begin(), e.bos_id);
                pin_tokens = e.prefix->pin_at(ids, npu::shared_prefix_len(lead, ids));
            } else {
                pin_tokens = e.prefix->pin_point(e.prev_ids, ids);
            }
            if (pin_tokens > 0 && int(pin_tokens) - 1 >= resume) pin_cut = int(pin_tokens) - 1;
        }
        {
            const int chunk_max = std::max(1, e.fwd->max_batch_rows());
            const int n = int(ids.size());
            for (int off = resume; off < n;) {
                int take = std::min(chunk_max, n - off);
                if (pin_cut >= 0 && off == pin_cut) {
                    Q4nxNpuForward::StateSnapshot snap;
                    if (e.fwd->snapshot_state(off, snap)) {
                        std::string why;
                        const std::vector<int> shared(ids.begin(), ids.begin() + long(pin_tokens));
                        const int slot = e.prefix->pin(shared, snap.bytes(), &why);
                        if (slot >= 0 && size_t(slot) < e.prefix_state.size()) e.prefix_state[size_t(slot)] = std::move(snap);
                        std::fprintf(stderr, "[prefix] pin slot=%d tokens=%zu state_at=%d %s\n", slot, pin_tokens, off,
                                     why.c_str());
                    }
                    pin_cut = -1;
                } else if (pin_cut > off && off + take > pin_cut) {
                    // Stop this chunk at the pin position; the next pass saves the state there. One more launch.
                    take = pin_cut - off;
                    const std::vector<int> head(ids.begin() + off, ids.begin() + off + take);
                    if (!e.fwd->step_batch(head, off, logits))
                        throw std::runtime_error(std::string("forward step failed: ") + e.fwd->last_error());
                    off += take;
                    continue;
                }
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
        if (e.prefix_pin) e.prev_ids = ids;

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
        // The ids a constrained answer may end on: the container's own eos ids plus <|endoftext|>.
        std::vector<int> end_ids = e.eos;
        if (e.endoftext >= 0) end_ids.push_back(e.endoftext);
        // Ids at or beyond the tokenizer's vocabulary are padding positions, not tokens: decode()
        // returns "" for them, which used to make them indistinguishable from a byte fragment and so
        // unconditionally allowed (engine#1182: the model drifts into them once the grammar leaves it
        // nothing to say, and they emit no text and never end the reply).
        const size_t vocab = size_t(e.tok->size());
        auto refresh_allowed = [&]() {
            allowed.assign(logits.size(), 0);
            const npu::ByteMask first = gram->next_bytes(grammar_state);
            for (size_t i = 0; i < logits.size() && i < vocab; i++) {
                const std::string& t = e.tok_text[i];
                if (t.empty()) {
                    allowed[i] = 1;  // a half-character byte fragment cannot leave the grammar
                } else if (first.has(static_cast<unsigned char>(t[0]))) {
                    npu::Grammar::State probe = grammar_state;
                    allowed[i] = gram->feed(probe, t) ? 1 : 0;
                }
            }
            npu::allow_end_tokens(allowed, grammar_state, end_ids);
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
        // One chosen token through the reply: the end tokens, the grammar, the stop sequences and the hold-back that
        // keeps a partial stop match unsent. false = the reply ends here (an end token, a stop sequence, or a client
        // that stopped reading). The plain loop and the speculative one hand every token to this, in order.
        // `row` is the logits row `next` was chosen from (the head's full width); null means `logits`.
        auto deliver = [&](int next, const float* row = nullptr) -> bool {
            if (std::find(e.eos.begin(), e.eos.end(), next) != e.eos.end() ||
                (e.endoftext >= 0 && next == e.endoftext)) { stopped_at_eos = true; return false; }
            if (gram) gram->feed(grammar_state, e.tok_text[next]);
            const std::string piece = e.tok->decode(next);
            pending += piece;
            // Raw distribution: sample_token() takes logits by const reference, so this is
            // pre-top_k/top_p/temperature and the grammar mask cannot narrow it.  Over the tokenizer's ids only, as
            // sample_token() draws: a row is as wide as the head, which can be wider (Qwen3 151,936 against 151,669,
            // engine#2272), and the padding would join the normaliser and could be named in top_logprobs.
            if (want_lp)
                held.emplace_back(Emitted{piece, npu::logprobs_of(row ? row : logits.data(),
                                                                  npu::token_bound(logits.size(), e.tok->size()), next,
                                                                  opts.sample.top_logprobs)},
                                  piece.size());
            stats.pieces.push_back(piece);   // engine#2111: reasoning_tokens is counted over these
            ++produced;
            const size_t stop_at = npu::find_stop(pending, opts.stop);
            if (stop_at != std::string::npos) {
                if (stop_at > 0) on_text(pending.substr(0, stop_at), take_held(stop_at));
                stopped_at_stop = true;
                return false;
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
                if (!on_text(out, take_held(emit))) return false;
            }
            return true;
        };
        // engine#1781: the speculative path, only where it cannot change a token. Greedy, so each token is the bounded
        // argmax the verify rows check drafts against; no grammar (its mask is per token and per state); no logit dump
        // (it writes one file per decode step). The text is then the plain loop's token for token -- what
        // tests/serve_spec_lookup_test.py holds it to. Logprobs are reported from the row each token came from
        // (Options::on_token_row), the base's own logits at that position, as the plain loop reports them.
        // --spec-sample widens it to sampled requests, whose drafts are then judged by the distribution-keeping rule
        // instead (npu::SpecSampler): the reply follows the same distribution as the plain loop's, not its stream.
        const bool spec = (e.spec_lookup > 0 || e.draft) && (opts.sample.greedy() || e.spec_sample) && !gram &&
                          e.dump_logits.empty();
        if (spec) {
            struct ServeBase : npu::mtp::BaseModel {
                Q4nxNpuForward& f;
                int nv, width, rows;
                ServeBase(Q4nxNpuForward& fw, int v, int w, int r) : f(fw), nv(v), width(w), rows(r) {}
                int vocab() const override { return nv; }
                // The forward's rows are the head's width (M x NV), wider than vocab() (the tokenizer's bound)
                // whenever the head carries padding ids (Qwen3: 151,936 against 151,669); the loop strides by this.
                int row_stride() const override { return width; }
                int hidden_size() const override { return int(f.last_hidden().size()); }
                int max_batch_rows() const override { return rows; }
                bool step(int t, int p, std::vector<float>& l) override { return f.step(t, p, l); }
                bool step_batch_all(const std::vector<int>& t, int s0, std::vector<float>& l) override {
                    f.begin_verify(int(t.size()), s0);
                    return f.step_batch_all(t, s0, l) && l.size() == t.size() * size_t(width);
                }
                bool commit_verify(int a) override { return f.commit_verify_rows(a); }
                const std::vector<float>& last_hidden() const override { return f.last_hidden(); }
            };
            // The bound sample_token() applies: ids past the tokenizer are padding and never chosen (engine#2017).
            // The device verifies as many rows as its kernels are wide; host GEMMs have no width, and 1 (what the
            // forward reports there) would leave no row for a draft, so host mode verifies the device's 16.
            ServeBase base(*e.fwd, int(npu::token_bound(logits.size(), e.tok->size())), int(logits.size()),
                           e.host_gemm ? 16 : e.fwd->max_batch_rows());
            npu::mtp::PromptLookupHead lookup(ids);
            std::unique_ptr<npu::dflash::DFlashHead> dflash;
            if (e.draft) {
                const int bound = int(npu::token_bound(logits.size(), e.tok->size()));
                npu::dflash::HeadHooks hooks;
                hooks.embed = [&e](int t, std::vector<float>& row) {
                    e.fwd->embed_row(t, row);
                    return !row.empty();
                };
                // The target's own head on the drafted row, then the package's rule: DFlash 2's selector when it has
                // one, else the head's choice, bounded like sample_token().
                hooks.pick = [&e, bound](const float* h, int prev) {
                    std::vector<float> hv(h, h + e.draft_w.hidden), lg;
                    if (!e.fwd->lm_head_logits(hv, lg) || lg.empty()) return -1;
                    if (e.draft->cfg.generation >= 2) return npu::dflash::dflash2_select(*e.draft, e.draft_hproj, h, prev, lg, bound);
                    return npu::argmax_token(lg.data(), lg.size(), bound);
                };
                dflash = std::make_unique<npu::dflash::DFlashHead>(e.draft_w, e.draft_taps, hooks,
                                                                   int(e.draft->cfg.mask_token_id),
                                                                   e.draft->cfg.block_size, e.draft->cfg.sliding_window);
            }
            npu::mtp::DraftHead& head = dflash ? static_cast<npu::mtp::DraftHead&>(*dflash) : lookup;
            npu::SpecSampler sampler(opts.sample, rng);   // used for a sampled request (npu_spec_sampler.h)
            npu::mtp::Options o;
            o.k = e.draft ? e.spec_draft_k : e.spec_lookup;
            o.max_tokens = limit;
            o.tau = 0.0f;
            o.fallback_below = e.spec_fallback;
            o.consume_last = false;   // the plain loop does not run the reply's last token either
            o.sampler = opts.sample.greedy() ? nullptr : &sampler;
            // Each token with the row it came from, so a request with logprobs reports them from the same row the
            // plain loop would have: the base's own logits at that position, at the head's full width.
            o.on_token_row = [&](int t, const float* row) {
                if (req.is_connection_closed() || (bridge && bridge->cancelled())) return false;
                return deliver(t, row);
            };
            const npu::mtp::Result r = npu::mtp::run_from(base, &head, int(ids.size()), logits, o);
            if (!r.ok) throw std::runtime_error("speculative decode: " + r.err);
            std::fprintf(stderr, "[spec] %s%s %d tokens: %d rounds, %d of %d drafts accepted, %d plain steps%s\n",
                         e.draft ? "dflash" : "lookup", o.sampler ? " sampled" : "", int(r.tokens.size()), r.stats.batches, r.stats.accepted, r.stats.drafted, r.stats.steps,
                         r.stats.fell_back_at >= 0 ? (", fell back after round " + std::to_string(r.stats.fell_back_at)).c_str()
                                                   : "");
        }
        for (int n = 0; !spec && n < limit; n++) {
            // the client is gone: free the NPU (the bridge knows first when a write to it failed)
            if (req.is_connection_closed() || (bridge && bridge->cancelled())) break;
            if (gram) refresh_allowed();
            // Raw distribution for a reference comparison: sample_token() takes logits by const
            // reference and the copy below happens before it runs, so what lands on disk is what
            // the model produced, not what the sampler chose.
            if (!e.dump_logits.empty()) {
                const std::string path = e.dump_logits + "/step_" + std::to_string(dump_req) + "_" +
                                          std::to_string(n) + ".bin";
                if (std::FILE* f = onebit_fopen_private_wb(path.c_str())) {   // 0600, no symlink (file_create.h)
                    std::fwrite(logits.data(), sizeof(float), logits.size(), f);
                    std::fclose(f);
                }
            }
            const int next = npu::sample_token(logits, opts.sample, rng, gram ? &allowed : nullptr, e.tok->size());
            // -1: the grammar allows no token here, so the answer ends where it is.  An end token is
            // in the mask only where the grammar can stop (refresh_allowed), so a constrained reply
            // that completes reports "stop" instead of padding to the budget (engine#1182).
            if (next < 0) { stopped_at_eos = true; break; }
            if (!deliver(next)) break;
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
    // engine#2111: the usage object is `onebit::usage_json_text()` now (app/forward_serve.h) - a pure function, so
    // the rule "reasoning_tokens appears iff the route could count it" has known answers without an NPU
    // (tests/serve_usage_test.cpp). The JSON stays here, where the rest of the reply is built.
    auto usage_json = [](const CallStats& st) {
        UsageCounters u;
        u.prompt_tokens = st.prompt_tokens;
        u.completion_tokens = st.completion_tokens;
        u.cached_tokens = st.prefix.cached_tokens;
        u.prefix_slot = st.prefix.slot;
        u.prefix_reason = st.prefix.reason;
        u.prefix_pinned = st.prefix.pinned;
        u.reasoning_tokens = st.reasoning_tokens;
        return json::parse(usage_json_text(u));
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

    // A tool grammar (tool_choice "required" or a named function) constrains the reply to the qwen3 call
    // markup whatever the container is (app/npu_tool_grammar.h), so that reply is split as qwen3.  Every
    // other reply is in the container's own markup.
    const npu::AnswerFamily family = opts.tool_grammar ? npu::AnswerFamily::qwen3 : e.answer_family;
    // A template whose generation prompt already opens the reasoning block leaves the answer inside it
    // (npu::prompt_opens_reasoning).  Only a rendered chat prompt can: an ids prompt and /v1/completions are raw.
    const bool starts_in_reasoning = chat && !prompt_is_ids && npu::prompt_opens_reasoning(prompt, family);
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
                npu::parse_answer(content, family, opts.tools_sent, starts_in_reasoning);
            stats.reasoning_tokens = npu::reasoning_token_count(stats.pieces, family, opts.tools_sent, starts_in_reasoning);
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
                const std::vector<char> keep = npu::content_mask(texts, family, opts.tools_sent, starts_in_reasoning);
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
            // With a bridge, each frame leaves as soon as it exists (the connection thread is already writing them);
            // without one they are collected and sent when the handler returns, as this route always did.
            auto chunks = std::make_shared<std::vector<std::string>>();
            auto send = [&](std::string frame) {
                if (bridge) bridge->push(std::move(frame));
                else chunks->push_back(std::move(frame));
            };
            // The same splitter as the non-streamed path, fed piece by piece, so a streamed reply
            // and a non-streamed one cannot disagree about what is reasoning and what is content.
            npu::AnswerStream stream(family, opts.tools_sent, starts_in_reasoning);
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
                    send("data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");
                }
                return true;
            };
            const std::string finish = run(emit_pieces);
            {
                std::vector<std::pair<bool, std::string>> tail;
                stream.finish(tail);
                for (const auto& [is_reasoning, text] : tail) {
                    json chunk = chunk_tpl;
                    json delta = is_reasoning ? json{{"reasoning_content", text}} : json{{"content", text}};
                    chunk["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
                    send("data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");
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
                    send("data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");
                }
                // OpenAI's stream ends with one choice that carries an empty delta and the reason the
                // generation ended; without it a streaming client cannot tell a `length` cut from a
                // natural `stop` (the per-delta chunks all say null, and the usage tail has no
                // choices at all). The tool-call chunk above already reports `tool_calls`, so this is
                // sent exactly once, and only for the text path.
                if (stream.calls().empty()) {
                    json chunk = chunk_tpl;
                    chunk["choices"] = json::array({{{"index", 0},
                                                     {"delta", json::object()},
                                                     {"finish_reason", finish}}});
                    send("data: " + chunk.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n");
                }
            }
            {
                json tail = chunk_tpl;
                tail["choices"] = json::array();
                // engine#2111: the same count as the non-streamed reply's, over the same pieces.
                if (chat)
                    stats.reasoning_tokens =
                        npu::reasoning_token_count(stats.pieces, family, opts.tools_sent, starts_in_reasoning);
                tail["usage"] = usage_json(stats);
            tail["timings"] = timings_json(stats);
                send("data: " + tail.dump() + "\n\n");
            }
            if (bridge) {
                bridge->push("data: [DONE]\n\n");
                return;
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
    } catch (const ContextOverflow& ex) {
        // The client's request, not a server fault: 400, with the type and the wording the GGUF route's
        // backend uses for the same refusal, so one client-side check covers both routes.
        res.status = 400;
        res.set_content(json{{"error", {{"message", ex.what()}, {"type", "exceed_context_size_error"},
                                        {"n_prompt_tokens", ex.prompt_tokens}, {"n_ctx", ex.context}}}}.dump(),
                        "application/json");
    } catch (const std::exception& ex) {
        res.status = 500;
        res.set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
    }
}

// The route's entry.  A request that does not ask for a stream is answered in place.  One that does is generated on
// its own thread, so its frames can be written while later tokens are still being produced (app/stream_bridge.h):
// this thread waits for the first frame or the end, and only then decides between an ordinary response -- the 400s
// and 500s a request can earn before it has produced anything keep their status codes -- and a stream.
void handle_generate(ForwardEngine& e, const httplib::Request& req, httplib::Response& res, bool chat) {
    // A non-causal decision pack has no decode to run (npu::decision_only_refusal): say so before any work.
    if (e.fwd && e.fwd->noncausal()) {
        res.status = 400;
        res.set_content(npu::decision_only_refusal(e.id, e.head != nullptr).dump(), "application/json");
        return;
    }
    bool wants_stream = false;
    try {
        const json body = json::parse(req.body);
        wants_stream = body.is_object() && body.contains("stream") && body["stream"].is_boolean() && body["stream"].get<bool>();
    } catch (const std::exception&) {
        // not JSON: the implementation answers that itself, without a stream
    }
    if (!wants_stream) {
        handle_generate_impl(e, req, res, chat, nullptr);
        return;
    }
    auto bridge = std::make_shared<npu::StreamBridge>();
    auto early = std::make_shared<httplib::Response>();   // what the generation answers when it never starts a stream
    // `req` outlives this handler: httplib keeps it until the response, content provider included, has been written,
    // and the releaser below joins the thread before that.
    auto worker = std::make_shared<std::thread>([&e, &req, chat, bridge, early]() {
        try {
            handle_generate_impl(e, req, *early, chat, bridge.get());
        } catch (const std::exception& ex) {
            early->status = 500;
            early->set_content(json{{"error", {{"message", ex.what()}}}}.dump(), "application/json");
        } catch (...) {
            early->status = 500;
            early->set_content(R"({"error":{"message":"generation failed"}})", "application/json");
        }
        // A failure after the stream began cannot change the status any more: say so in the stream and end it.
        // One that came before any frame pushes nothing, so the handler below still answers it as an ordinary error.
        if (early->status >= 400 && bridge->started()) bridge->push("data: " + early->body + "\n\ndata: [DONE]\n\n");
        bridge->finish();
    });
    if (!bridge->wait_first()) {
        worker->join();
        res.status = early->status > 0 ? early->status : 500;
        res.set_content(early->body, early->has_header("Content-Type") ? early->get_header_value("Content-Type")
                                                                        : std::string("application/json"));
        return;
    }
    res.set_chunked_content_provider(
        "text/event-stream",
        [bridge](size_t, httplib::DataSink& sink) {
            std::string frame;
            while (bridge->next(frame)) {
                if (!sink.write(frame.data(), frame.size())) {
                    bridge->cancel();   // the client went away: the generation stops at its next token
                    return false;
                }
            }
            sink.done();
            return true;
        },
        [bridge, worker](bool) {
            bridge->cancel();
            if (worker->joinable()) worker->join();
        });
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

// engine#1862: the sealed-route one-shot id run. `npu-run` was written for the per-role lane kernel set
// (layer_ctx{1,2,17}.elf + lmhead.elf), which no pack on disk ships any more: every pack since the sealed route carries
// SEALED-*.elf / full_i8_* driven by this forward. So an id run on a current pack goes through the SAME forward, and
// the pass it makes is the one the k-row identity check needs -- one launch, M rows, per-row logits.
int run_forward_ids(const std::string& model_dir, const std::string& kernels, const std::vector<int>& ids,
                    int max_tokens, float penalty, bool host_gemm) {
    if (ids.empty()) throw std::runtime_error("--ids: no token ids");
    const std::string model = model_dir + "/model.q4nx";
    if (!std::filesystem::exists(model)) throw std::runtime_error(model + " does not exist");
    // The tokenizer's size bounds the printed ids (engine#2017); a container without tokenizer.json keeps the whole row.
    int tok_vocab = 0;
    if (const std::string tj = model_dir + "/tokenizer.json"; std::filesystem::exists(tj))
        tok_vocab = npu::Tokenizer(tj).size();
    // Same device discipline as forward-serve: host-GEMM runs take no lock (engine#1353 -- a host-GEMM run that held it
    // for 4 h 52 m stopped two lanes' device work), a run that opens the device takes the exclusive one.
    int lock_fd = -1;
    if (npu_lock_taken(!host_gemm, std::getenv("NPU_NO_DEVICE_LOCK") != nullptr)) lock_fd = take_npu_device_lock(1);
    std::unique_ptr<xrt::device> dev;
    if (!host_gemm) dev = std::make_unique<xrt::device>(0);
    Q4nxNpuForward fwd;
    const bool ok = dev ? fwd.init(*dev, model.c_str(), kernels) : fwd.init_host(model.c_str(), kernels);
    if (!ok) throw std::runtime_error("forward init failed: " + std::string(fwd.last_error()));
    q4nx_cache_sweep_after_load();   // engine#2263: the load wrote the device form and head images into the cache
    const int width = fwd.max_batch_rows();
    if (int(ids.size()) > width)
        std::fprintf(stderr, "1bit npu-run: %zu ids, so this runs in passes of %d rows (the kernel's width)\n", ids.size(),
                     width);
    int done = 0;
    while (done < int(ids.size())) {
        const int m = std::min(width, int(ids.size()) - done);
        std::vector<int> chunk(ids.begin() + done, ids.begin() + done + m);
        std::vector<float> all, one;
        // begin_verify() is what makes the next pass norm EVERY row and write M x NV logits (the speculative verifier's
        // call); plain step_batch() returns only the last row, which cannot answer a per-row question.
        fwd.begin_verify(m, done);
        if (!fwd.step_batch_all(chunk, done, all))
            throw std::runtime_error("forward step failed: " + std::string(fwd.last_error()));
        // NV from the block itself (the vocab width is whatever the head returns) rather than from a config accessor:
        // the size has to divide by the row count either way, and a block that does not is the thing to report.
        if (all.empty() || all.size() % size_t(m) != 0)
            throw std::runtime_error("forward returned a logits block of the wrong size (" + std::to_string(all.size()) +
                                     " floats for " + std::to_string(m) + " rows)");
        const int nv = int(all.size() / size_t(m));
        for (int r = 0; r < m; ++r) {
            // Bounded by the tokenizer like the served route (engine#2017): an id past it is a padding position.
            const int best = npu::argmax_token(all.data() + size_t(r) * size_t(nv), size_t(nv), tok_vocab);
            std::printf("%d\n", best);   // token ids out, in row order: the id the model chooses at that row
        }
        done += m;
    }
    // -n only matters for a continuation, which this route does not do: say so rather than ignore the option.
    if (max_tokens > 0)
        std::fprintf(stderr,
                     "1bit npu-run: sealed route returns the model's choice at EVERY row of one pass; -n %d is not "
                     "used (a continuation is `1bit forward-serve`)\n",
                     max_tokens);
    (void)penalty;
    if (lock_fd >= 0) ::close(lock_fd);
    return 0;
}

int run_forward_serve(int argc, char** argv) {
    std::string model, kernels, host = "127.0.0.1", alias, dump_logits;
    int port = 8000, streams = 1;
    // Which NPU target serves this route (engine#799 P4). A1 compiles one, xdna; the point of the option
    // is that a name this build does not have is refused BY NAME below, before the device is opened,
    // instead of being ignored and served by whatever target happened to be linked.
    std::string npu_target = ::npu::target::xdna::XdnaTarget::kName;
    npu::PrefixOptions prefix_opts;  // --prefix-slots / --prefix-cache-mb / --prefix-min-gain (#650)
    bool prefix_pin = false;         // --prefix-pin (engine#1785)
    int ctx_size = 0;                // --ctx-size N: prompt plus answer, at most the model's own (0 = the model's)
    int spec_lookup = 0;             // --spec-lookup K (engine#1781): prompt-lookup drafts, K per verify batch
    double spec_fallback = 0.0;      // --spec-fallback X: stop drafting below X tokens per round
    bool spec_sample = false;        // --spec-sample: sampled requests speculate too (distribution-keeping rule)
    std::string spec_draft;          // --spec-draft <dir>: a DFlash draft package as the drafter
    int spec_draft_k = 0;            // --spec-draft-k K: drafts per round (default: the package's block - 1)
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
        else if (a == "--prefix-pin") prefix_pin = true;
        else if (a == "-c" || a == "--ctx-size") ctx_size = std::stoi(next());
        else if (a == "--npu-streams") streams = std::stoi(next());
        else if (a == "--npu-target") npu_target = next();
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--spec-lookup") spec_lookup = std::stoi(next());
        else if (a == "--spec-fallback") spec_fallback = std::stod(next());
        else if (a == "--spec-sample") spec_sample = true;
        else if (a == "--spec-draft") spec_draft = next();
        else if (a == "--spec-draft-k") spec_draft_k = std::stoi(next());
        else throw std::runtime_error("unknown option " + a);
    }
    if (model.empty())
        throw std::runtime_error("usage: 1bit forward-serve -m <model.q4nx> --kernels <dir> [-p PORT] [--npu-streams N] [--npu-target xdna] [--dump-logits <dir>] [--spec-lookup K | --spec-draft <dir> [--spec-draft-k K]] [--spec-fallback X] [--spec-sample]");
    if (!spec_draft.empty() && spec_lookup > 0)
        throw std::runtime_error("--spec-draft and --spec-lookup are two drafters; give one");
    if (spec_draft_k != 0 && spec_draft.empty()) throw std::runtime_error("--spec-draft-k needs --spec-draft");
    if (spec_lookup < 0 || spec_lookup > 15)
        throw std::runtime_error("--spec-lookup takes 0..15 drafts per round (a verify batch holds at most 16 rows), not " +
                                 std::to_string(spec_lookup));
    if (!(spec_fallback >= 0.0)) throw std::runtime_error("--spec-fallback takes a number of tokens >= 0");
    if (spec_sample && spec_lookup == 0 && spec_draft.empty())
        throw std::runtime_error("--spec-sample needs a drafter (--spec-lookup K or --spec-draft <dir>)");
    if (streams < 1 || streams > kNpuMaxStreams)
        throw std::runtime_error("--npu-streams takes 1.." + std::to_string(kNpuMaxStreams) +
                                 " (the device's hwctx limit), not " + std::to_string(streams));
    // The compiled target list, checked here rather than at open() so a typo cannot reach the device. A1 links
    // exactly one target; a second (P5's hexagon) adds its name to this list and to npu/target/.
    {
        const std::vector<std::string> targets = {::npu::target::xdna::XdnaTarget::kName};
        if (std::find(targets.begin(), targets.end(), npu_target) == targets.end())
            throw std::runtime_error("--npu-target " + npu_target +
                                     ": this build has no such target (compiled: " + targets.front() + ")");
    }
    if (!std::filesystem::exists(model)) throw std::runtime_error(model + " does not exist");

    const std::string dir = std::filesystem::path(model).parent_path().string();
    const std::string cfg = dir + "/config.json";
    const std::string tok_json = dir + "/tokenizer.json";

    ForwardEngine e;
    e.id = alias.empty() ? std::filesystem::path(dir).filename().string() : alias;
    e.spec_lookup = spec_lookup;
    e.spec_fallback = spec_fallback;
    e.spec_sample = spec_sample;
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
    // config.json is the first source; what it does not state, tokenizer_config.json does (the BOS text, looked up
    // in the tokenizer, and add_bos_token). Without this a Gemma3 pack was served with no BOS at all.
    if (e.bos_id < 0 || json_int(cfg, "add_bos_token", -1) < 0) {
        const nlohmann::json tc = npu::read_json_or_null(dir + "/tokenizer_config.json");
        if (e.bos_id < 0) {
            const std::string b = npu::special_token_text(tc, "bos_token");
            if (!b.empty()) e.bos_id = e.tok->token_id(b);
        }
        if (json_int(cfg, "add_bos_token", -1) < 0) {
            const int a = npu::add_bos_token_flag(tc);
            if (a >= 0) e.add_bos = a != 0;
        }
        if (e.bos_id >= 0 && e.add_bos)
            std::fprintf(stderr, "1bit forward-serve: BOS %d from tokenizer_config.json (config.json does not state it)\n",
                         e.bos_id);
    }
    e.eos = npu::eos_token_ids_from_config(cfg);
    const size_t eos_from_config = e.eos.size();
    e.endoftext = e.tok->token_id("<|endoftext|>");
    // q4nx_config_int finds the key at any depth (a text_config's too).
    e.max_context = q4nx_config_int(cfg.c_str(), "max_position_embeddings", 0);
    if (e.max_context <= 0) e.max_context = 32768;
    // --ctx-size lowers the limit and never raises it: the model's positions are what they are. The same
    // guard that refuses a prompt longer than the model's context then refuses one longer than this.
    if (ctx_size < 0) throw std::runtime_error("--ctx-size: must be zero or more");
    if (ctx_size > 0) {
        if (ctx_size > e.max_context)
            std::fprintf(stderr, "1bit forward-serve: --ctx-size %d is above the model's context %d; using %d\n", ctx_size,
                         e.max_context, e.max_context);
        e.max_context = std::min(e.max_context, ctx_size);
    }
    std::fprintf(stderr, "1bit forward-serve: context %d tokens (prompt plus answer)\n", e.max_context);

    // #963: the chat route's prompt was ChatML for every container, and a tokenizer with no
    // <|im_start|>/<|im_end|> added token cannot represent it -- the marker becomes ordinary pieces and
    // the model reads text its own template never produces (zyphra/ZR1-1.5B: 33 tokens rendered
    // against 15 under the model's template).  Since option 2 the container's own template is rendered
    // when it declares one, so the anchors are only needed by a container that declares none; a
    // container with neither is still refused per request rather than failed at load, because
    // /v1/completions with token ids (#938) serves it correctly.
    e.tmpl = npu::load_chat_template(dir);
    // The load names the template's identity, the same number a pin or a 400 quotes (#1089, #1095):
    // one line per load, and "none" is a fact worth printing rather than an absence.
    std::fprintf(stderr, "1bit forward-serve: template sha256=%s (%s)%s\n",
                 e.tmpl.sha256.empty() ? "none" : e.tmpl.sha256.c_str(),
                 e.tmpl.present() ? e.tmpl.source.c_str() : "no template",
                 e.tmpl.present() ? "" : " -- the chat route uses ChatML with anchors, else refuses");
    const bool can_render = e.tmpl.present() && npu::renderer_built();
    if (can_render) e.answer_family = npu::answer_family_of_template(e.tmpl.text);
    std::fprintf(stderr, "1bit forward-serve: answer markup=%s\n",
                 e.answer_family == npu::AnswerFamily::gemma4 ? "gemma4" : "qwen3");
    if (e.tmpl.present() && !npu::renderer_built())
        std::fprintf(stderr,
                     "1bit forward-serve: this build has no chat-template renderer (ONEBIT_JINJA=OFF), so the "
                     "container's template is NOT used and the chat route falls back to ChatML\n");
    if (!can_render && !npu::chatml_supported(*e.tok)) {
        e.chat_ok = false;
        std::string specials;
        std::vector<std::pair<const char*, int>> named = {{"bos", e.bos_id}};
        for (int id : e.eos) named.emplace_back("eos", id);
        for (const auto& [label, id] : named) {
            const std::string text = id >= 0 ? e.tok->decode(id) : std::string();
            if (text.empty()) continue;
            specials += std::string(specials.empty() ? "" : ", ") + label + " " + text + " (" +
                        std::to_string(id) + ")";
        }
        e.chat_refusal =
            "this container's tokenizer has no <|im_start|> / <|im_end|>, so the ChatML prompt the chat route "
            "renders would be tokenized into pieces the model never sees (#963); send /v1/completions with "
            "prompt as token ids (#938) and render the model's own template" +
            (specials.empty() ? std::string() : " (model specials: " + specials + ")");
        std::fprintf(stderr, "1bit forward-serve: chat route disabled: %s\n", e.chat_refusal.c_str());
    }
    // What ends a generation is more than config.json's eos_token_id (npu::stop_token_ids): the tokenizer's own
    // eos_token, and, when this route closes its prompt turns with ChatML's marker, that marker.  Decided here,
    // after the template, because whether the route renders ChatML is known only now.
    {
        const bool chatml_route = !can_render && e.chat_ok;
        // The marker this route's prompts close an assistant turn with.  On the ChatML fallback it is <|im_end|>.
        // With the container's own template it is read off the template: render a two-message exchange and take
        // what follows the assistant's text (npu::turn_end_marker).  A template that cannot render that exchange
        // contributes nothing, and the load line below shows what the serve stops on either way.
        std::string turn_end = chatml_route ? std::string("<|im_end|>") : std::string();
        if (can_render) {
            static const char* const kProbe = "\xC2\xA4turn-end-probe\xC2\xA4";
            try {
                nlohmann::ordered_json exchange = nlohmann::ordered_json::array();
                exchange.push_back({{"role", "user"}, {"content", "u"}});
                exchange.push_back({{"role", "assistant"}, {"content", kProbe}});
                const std::string rendered = npu::render_chat_template(e.tmpl, exchange, /*add_generation_prompt=*/false,
                                                                       nlohmann::ordered_json::object(), e.bos_text, e.eos_text);
                const size_t at = rendered.find(kProbe);
                if (at != std::string::npos) turn_end = npu::turn_end_marker(rendered.substr(at + std::strlen(kProbe)));
            } catch (const std::exception& ex) {
                std::fprintf(stderr, "1bit forward-serve: the template did not render a two-message exchange (%s); no turn-end marker from it\n", ex.what());
            }
        }
        e.eos = npu::stop_token_ids(npu::read_json_or_null(cfg), npu::read_json_or_null(dir + "/tokenizer_config.json"),
                                    turn_end, [&](const std::string& text) { return e.tok->token_id(text); });
        std::string named;
        for (size_t i = 0; i < e.eos.size(); i++)
            named += std::string(i ? ", " : "") + std::to_string(e.eos[i]) + " " + e.tok->decode(e.eos[i]) +
                     (i < eos_from_config ? "" : " (tokenizer / chat turn end)");
        std::fprintf(stderr, "1bit forward-serve: stops on %s\n", named.empty() ? "<|endoftext|> only" : named.c_str());
    }
    // The template variables `{{ bos_token }}`/`{{ eos_token }}` render: the strings the tokenizer
    // actually has (a null bos_token in tokenizer_config.json is "", and the engine's -1 is the same
    // fact), not a second read of the JSON that could disagree with the tokenizer.
    if (e.bos_id >= 0) e.bos_text = e.tok->decode(e.bos_id);
    if (!e.eos.empty() && e.eos.front() >= 0) e.eos_text = e.tok->decode(e.eos.front());

    // The prompt cache (#650): --prefix-slots 0 turns it off, and an invalid value is refused here
    // rather than clamped, like every other option on this command.
    if (prefix_opts.slots > 0) {
        e.prefix = std::make_unique<npu::PrefixCache>(prefix_opts);
        e.prefix_state.resize(size_t(prefix_opts.slots));
        std::fprintf(stderr,
                     "1bit forward-serve: prefix cache on: %d slots, %zu MB, min gain %d tokens\n",
                     prefix_opts.slots, prefix_opts.cache_mb, prefix_opts.min_gain);
        e.prefix_pin = prefix_pin;
        if (prefix_pin) std::fprintf(stderr, "1bit forward-serve: prefix pin on: the part a request shares with the previous one keeps a slot\n");
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
    //
    // Offline mode (npu/forward/include/q4nx_forward.h, init_host): with
    // NPU_INFER_HOST_GEMM=1 the forward runs the model with host GEMMs and never opens
    // the device, which is how the sampling and grammar path is exercised without an
    // NPU.  Such a run takes no lock (engine#1353): the lock is what protects the array,
    // and a run that holds it without ever opening the device blocks every lane that
    // does -- measured, one orphaned host-GEMM serve held it 4 h 52 m and stopped two
    // other lanes' device work.  Open the device, and the lock, only when the forward is
    // going to use it.
    const bool host_gemm = std::getenv("NPU_INFER_HOST_GEMM") != nullptr;
    e.host_gemm = host_gemm;
    if (npu_lock_taken(!host_gemm, std::getenv("NPU_NO_DEVICE_LOCK") != nullptr))
        e.lock_fd = take_npu_device_lock(streams);
    std::unique_ptr<xrt::device> dev;
    if (!host_gemm) dev = std::make_unique<xrt::device>(0);
    const bool ok = dev ? e.fwd->init(*dev, model.c_str(), kernels) : e.fwd->init_host(model.c_str(), kernels);
    if (!ok) {
        const std::string why = std::string("forward init failed: ") + e.fwd->last_error();
        // engine#1241 cure 2: the driver refuses a second client with code 102 / ENODEV, the same message a dead array
        // answers, so "No such device" reads as a broken device or ELF. If another pid holds the device lock, say so
        // now: the array is merely owned, and the fix is to wait for that process, not to rebuild anything.
        if (npu_contexts_refused(why)) {
            const std::string held = npu_lock_refusal_note();
            if (!held.empty()) std::fprintf(stderr, "%s\n", held.c_str());
        }
        throw std::runtime_error(why);
    }
    // engine#2263: loading a pack writes into the Q4NX cache (the device-form copy, the head images) and, until this, only a
    // GGUF repack swept it -- so a box that serves packs, or runs reviews through npu-run, grew it without bound.
    q4nx_cache_sweep_after_load();

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

    // Warm the weight cache before saying "ready". A first request has to touch every (layer, role) weight once, and on
    // the non-folded int8 route that means materialising the weight matrix in float32 and packing it -- ~112
    // projections, which on a 27B pack is tens of seconds. Left in the request, that looks like a hang: a 8-token
    // request returning nothing in 150 s (engine #1605), and a "ready" line that is a lie about the next minute.
    // Paid here it is startup time, which is what startup time is for. NPU_SERVE_NO_WARMUP=1 keeps the old behaviour.
    if (!getenv("NPU_SERVE_NO_WARMUP")) {
        // Announce the warm-up BEFORE it runs, so a caller polling for "ready" can tell "warming, give it a minute" from
        // "hung". Measured cost is 44.5 s on a folded 27B pack, and `tests/serve_port.sh`'s helper defaults to a 30 s
        // budget -- a caller that only knows about `ready` would time out and report "serve never reported a port",
        // which is the misleading shape #1531/#1545 spent effort removing.
        if (!spec_draft.empty()) {
            // The draft is loaded once and checked against this target before the server says ready: a package
            // for another model would otherwise fail only at the first greedy request's first draft.
            auto dm = std::make_unique<npu::dflash::DraftModel>(npu::dflash::load_draft(spec_draft));
            std::vector<float> row;
            e.fwd->embed_row(0, row);
            if (int(row.size()) != dm->cfg.hidden)
                throw std::runtime_error("--spec-draft " + spec_draft + ": the draft is " + std::to_string(dm->cfg.hidden) +
                                         " wide, this model " + std::to_string(row.size()));
            if (!e.fwd->set_layer_taps(dm->cfg.target_layer_ids, &e.draft_taps))
                throw std::runtime_error("--spec-draft " + spec_draft + ": its target layers do not fit this model");
            const int most = dm->cfg.block_size - 1;
            if (most < 1) throw std::runtime_error("--spec-draft " + spec_draft + ": block size " + std::to_string(most + 1));
            if (spec_draft_k < 0 || spec_draft_k > most)
                throw std::runtime_error("--spec-draft-k takes 1.." + std::to_string(most) + " for this package");
            e.spec_draft_k = spec_draft_k > 0 ? spec_draft_k : most;
            e.draft_w = npu::dflash::decode_draft(*dm);
            if (dm->cfg.generation >= 2) e.draft_hproj = npu::dflash::decode_plain(dm->selector_hproj);
            std::fprintf(stderr, "1bit forward-serve: speculative drafts from %s (DFlash %d, %d layers, taps",
                         spec_draft.c_str(), dm->cfg.generation, dm->cfg.layers);
            for (int l : dm->cfg.target_layer_ids) std::fprintf(stderr, " %d", l);
            std::fprintf(stderr, "), up to %d per round\n", e.spec_draft_k);
            e.draft = std::move(dm);
        }
        std::fprintf(stderr, "1bit forward-serve: warming (weight cache; first request would otherwise pay it)\n");
        std::fflush(stderr);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<float> wlogits;
        std::vector<int> wids = {131, 131, 131, 131};   // four tokens of one id: the cheapest pass that touches every weight
        e.fwd->reset();
        const bool ok = e.fwd->step_batch(wids, 0, wlogits);
        e.fwd->reset();
        const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (!ok) {
            std::fprintf(stderr, "1bit forward-serve: warm-up pass failed (%s); serving anyway -- the first request "
                                 "will pay it\n", e.fwd->last_error());
        } else {
            // The caveat is in the log line, not only the PR: the warm-up covers the TOKEN-INVARIANT weights (wkey >=
            // 0, "packed once and reused" -- xdna_target.cpp). Routed-expert weights are wkey < 0 and are deliberately
            // never cached ("whose weights change per token"), so on a MoE pack the warm-up is NOT the whole first-touch
            // cost and a reader must not expect the same 44.5 s to disappear.
            std::fprintf(stderr, "1bit forward-serve: warm-up pass %.1f ms (token-invariant weights only -- routed MoE "
                                 "experts are never cached, so a MoE pack still pays per-expert packing; "
                                 "NPU_SERVE_NO_WARMUP=1 to skip)\n", wms);
        }
    }
    ready = true;
    std::fprintf(stderr, "1bit forward-serve: %s ready on http://%s:%d (%s)", e.id.c_str(), host.c_str(), port,
                 e.fwd->backend().c_str());
    if (streams > 1) std::fprintf(stderr, " [npu-streams=%d]", streams);
    // The target that is actually serving, so a log says which one answered rather than leaving it to the build.
    std::fprintf(stderr, " [npu-target=%s]", npu_target.c_str());
    std::fprintf(stderr, "\n");
    // engine#2102: the machine line `[forward] decode family: …` stays byte-for-byte (gates grep it). This is the sentence
    // an operator needs at the ready line's level when it says 0 of N -- the route that answers correctly and 50x slower.
    {
        const std::string note = onebit::npu::decode_family_note(e.fwd->decode_roles(), e.fwd->design_roles(), kernels,
                                                                 e.fwd->decode_disabled(), e.fwd->decode_image());
        if (!note.empty()) std::fprintf(stderr, "%s\n", note.c_str());
    }

    while (true) std::this_thread::sleep_for(std::chrono::seconds(3600));
    return 0;
}

}  // namespace onebit
