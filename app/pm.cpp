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

// `1bit serve --pm`: the Project Manager tool loop (app/pm.h, docs/pm.md).
#include "pm.h"

#include "pm_experts_default.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace onebit::pm {

namespace {

using json = nlohmann::json;

const char* kBuiltinPrompt =
    "You are the Project Manager of the 1bit engine. You run on the user's own machine and lead a team of "
    "expert models. Decide for each request whether to answer it yourself or to delegate it.\n"
    "- Answer yourself: conversation, short factual questions, opinions, summaries of what is in the chat, "
    "anything an expert would not do better.\n"
    "- Delegate with delegate(expert, task, context) when the request fits an expert's domain below. "
    "Write the task as a complete, self-contained brief: the expert sees only your task and context, "
    "never the chat. Put the user's exact requirements, code and constraints in it.\n"
    "- Call experts() if you need the list again.\n"
    "- Only one expert runs at a time, and loading one takes time, so delegate once with a complete brief "
    "rather than several times.\n"
    "- When the tool result arrives, relay the expert's answer to the user in full (code included), "
    "checked and with your own short framing. If the result says the delegation failed, say so in one "
    "sentence and answer as well as you can yourself.\n"
    "Never pretend an expert answered when it did not.";

std::string json_string(const json& v) {
    if (v.is_string()) return v.get<std::string>();
    if (v.is_null()) return "";
    return v.dump();
}

// The text parts of an OpenAI message content (a string, or an array of parts).
std::string content_text(const json& c) {
    if (c.is_string()) return c.get<std::string>();
    std::string out;
    if (c.is_array())
        for (const auto& part : c)
            if (part.is_object() && part.value("type", "") == "text") out += part.value("text", "");
    return out;
}

// Cuts `s` to at most `n` bytes without splitting a UTF-8 sequence.
std::string cut_utf8(std::string s, size_t n, bool* cut) {
    *cut = false;
    if (s.size() <= n) return s;
    size_t end = n;
    while (end > 0 && (static_cast<unsigned char>(s[end]) & 0xC0) == 0x80) --end;
    s.resize(end);
    *cut = true;
    return s;
}

std::string now_id(const char* prefix) {
    static std::atomic<unsigned> n{0};
    return std::string(prefix) + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + "-" +
           std::to_string(n++);
}

long now_s() {
    return long(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

// JSON object starting at `pos` (an opening brace), by brace matching outside strings.
bool object_at(const std::string& s, size_t pos, json& out, size_t* end) {
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = pos; i < s.size(); i++) {
        const char c = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{') depth++;
        else if (c == '}') {
            if (--depth == 0) {
                try {
                    out = json::parse(s.substr(pos, i + 1 - pos));
                } catch (const std::exception&) {
                    return false;
                }
                *end = i + 1;
                return true;
            }
        }
    }
    return false;
}

json arguments_object(const json& a) {
    if (a.is_object()) return a;
    if (a.is_string()) {
        const std::string s = a.get<std::string>();
        try {
            json j = json::parse(s);
            if (j.is_object()) return j;
        } catch (const std::exception&) {
        }
        if (!s.empty()) return json{{"task", s}};
    }
    return json::object();
}

const Expert& expert_or(const Config& cfg, const std::string& key, const Expert*& found) {
    static const Expert none{};
    found = cfg.find(key);
    return found ? *found : none;
}

std::string experts_text(const Config& cfg) {
    std::string out;
    for (const auto& e : cfg.experts)
        out += "- " + e.key + " (" + e.id + "): " + e.domain + ". Use when: " + e.use_when + "\n";
    if (out.empty()) out = "(no experts configured: answer everything yourself)\n";
    return out;
}

json tools_schema(const Config& cfg) {
    std::string keys;
    for (const auto& e : cfg.experts) keys += (keys.empty() ? "" : ", ") + e.key;
    return json::array({
        {{"type", "function"},
         {"function", {{"name", "experts"},
                       {"description", "List the experts you can delegate to, with when to use each."},
                       {"parameters", {{"type", "object"}, {"properties", json::object()}}}}}},
        {{"type", "function"},
         {"function", {{"name", "delegate"},
                       {"description", "Hand a task to one expert model and get its full answer back as the tool result. "
                                       "Experts: " + keys + "."},
                       {"parameters", {{"type", "object"},
                                       {"properties", {{"expert", {{"type", "string"}, {"description", "one of: " + keys}}},
                                                       {"task", {{"type", "string"}, {"description", "a complete, self-contained brief of what to do"}}},
                                                       {"context", {{"type", "string"}, {"description", "code, data or constraints the expert needs (optional)"}}}}},
                                       {"required", json::array({"expert", "task"})}}}}}}});
}

struct Delegation {
    std::string expert, id;
    long ms = 0;
    bool ok = false;
};

// Runs delegate(): one OpenAI chat request to Lemonade for the expert's catalog id. Returns the
// tool result text; a failure is described in it (the PM must say so and answer alone).
std::string run_delegate(const Config& cfg, const json& args, std::vector<Delegation>& log) {
    const std::string key = json_string(args.value("expert", json()));
    const std::string task = json_string(args.value("task", json()));
    const std::string context = json_string(args.value("context", json()));
    const Expert* e = nullptr;
    expert_or(cfg, key, e);
    Delegation d;
    d.expert = key;
    if (!e) {
        std::string keys;
        for (const auto& x : cfg.experts) keys += (keys.empty() ? "" : ", ") + x.key;
        log.push_back(d);
        std::fprintf(stderr, "1bit serve: pm: delegate %s: unknown expert (task %zu chars)\n", key.c_str(), task.size());
        return "delegation failed: there is no expert '" + key + "'. The experts are: " + (keys.empty() ? "(none)" : keys) +
               ". Answer the user yourself.";
    }
    d.id = e->id;
    if (task.empty()) {
        log.push_back(d);
        return "delegation failed: delegate() needs a task. Call it again with the full brief in `task`.";
    }
    json messages = json::array();
    messages.push_back({{"role", "system"},
                        {"content", "You are the " + e->domain + " expert of the 1bit engine's team. The Project Manager "
                                    "sends you one task; answer it completely and directly, code included. Your answer is relayed "
                                    "to the user as it is."}});
    messages.push_back({{"role", "user"}, {"content", context.empty() ? task : task + "\n\nContext:\n" + context}});
    json req = {{"model", e->id}, {"messages", messages}, {"stream", false}};
    if (e->max_tokens > 0) req["max_tokens"] = e->max_tokens;

    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return long(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count()); };
    auto fail = [&](const std::string& why) {
        d.ms = elapsed();
        log.push_back(d);
        std::fprintf(stderr, "1bit serve: pm: delegate %s -> %s (task %zu chars) %ld ms failed: %s\n", key.c_str(),
                     e->id.c_str(), task.size(), d.ms, why.c_str());
        return "delegation to " + e->id + " failed: " + why + ". Tell the user the expert was not available and answer yourself.";
    };
    httplib::Client cli(cfg.lemonade_url);
    if (!cli.is_valid()) return fail("--lemonade-url " + cfg.lemonade_url + " is not a usable http:// URL");
    cli.set_connection_timeout(5);
    cli.set_read_timeout(cfg.delegate_timeout_s);
    cli.set_write_timeout(60);
    auto r = cli.Post(cfg.chat_path, req.dump(), "application/json");
    if (!r) return fail("Lemonade at " + cfg.lemonade_url + " did not answer (" + httplib::to_string(r.error()) + ")");
    json out;
    try {
        out = json::parse(r->body);
    } catch (const std::exception&) {
        return fail("Lemonade answered HTTP " + std::to_string(r->status) + " with a body that is not JSON");
    }
    if (r->status != 200) {
        std::string msg = out.is_object() && out.contains("error") ? json_string(out["error"].is_object() ? out["error"].value("message", json()) : out["error"]) : "";
        return fail("Lemonade answered HTTP " + std::to_string(r->status) + (msg.empty() ? "" : " (" + msg + ")"));
    }
    std::string answer;
    try {
        const json& m = out.at("choices").at(0).at("message");
        answer = content_text(m.value("content", json()));
        if (answer.empty() && m.contains("reasoning_content")) answer = json_string(m["reasoning_content"]);
    } catch (const std::exception&) {
        return fail("Lemonade's reply had no choices[0].message");
    }
    if (answer.empty()) return fail("the expert returned an empty answer");
    d.ms = elapsed();
    d.ok = true;
    log.push_back(d);
    bool cut = false;
    answer = cut_utf8(answer, cfg.max_result_bytes, &cut);
    std::fprintf(stderr, "1bit serve: pm: delegate %s -> %s (task %zu chars) %ld ms ok, %zu bytes%s\n", key.c_str(),
                 e->id.c_str(), task.size(), d.ms, answer.size(), cut ? " (cut)" : "");
    if (cut) answer += "\n[the expert's answer was cut at " + std::to_string(cfg.max_result_bytes) + " bytes]";
    return answer;
}

// The final answer as SSE, the chunks a plain forward would relay: a role chunk, the reasoning
// (if any) and the content in pieces, a finish chunk, then [DONE]. Built here, while the handler
// owns its locals; the provider runs after the handler returns.
void stream_final(httplib::Response& res, const std::string& id, const std::string& model, const json& message,
                  const std::string& finish) {
    json tpl = {{"id", id}, {"object", "chat.completion.chunk"}, {"created", now_s()}, {"model", model}};
    auto chunks = std::make_shared<std::vector<std::string>>();
    auto push = [&](json delta, json fin) {
        json c = tpl;
        c["choices"] = json::array({{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", std::move(fin)}}});
        chunks->push_back("data: " + c.dump() + "\n\n");
    };
    push({{"role", "assistant"}, {"content", ""}}, nullptr);
    auto pieces = [&](const char* field, const std::string& text) {
        // pieces of about 48 bytes, split at ASCII spaces so a UTF-8 sequence is never cut
        size_t i = 0;
        while (i < text.size()) {
            size_t j = std::min(text.size(), i + 48);
            if (j < text.size()) {
                const size_t sp = text.rfind(' ', j);
                if (sp != std::string::npos && sp > i) j = sp + 1;
                else
                    while (j < text.size() && (static_cast<unsigned char>(text[j]) & 0xC0) == 0x80) ++j;
            }
            push({{field, text.substr(i, j - i)}}, nullptr);
            i = j;
        }
    };
    if (message.contains("reasoning_content") && message["reasoning_content"].is_string())
        pieces("reasoning_content", message["reasoning_content"].get<std::string>());
    pieces("content", content_text(message.value("content", json())));
    push(json::object(), finish);
    chunks->push_back("data: [DONE]\n\n");
    auto cursor = std::make_shared<size_t>(0);
    res.set_chunked_content_provider("text/event-stream", [chunks, cursor](size_t, httplib::DataSink& sink) {
        while (*cursor < chunks->size()) {
            const std::string& d = (*chunks)[*cursor];
            if (!sink.write(d.data(), d.size())) return false;
            ++*cursor;
        }
        sink.done();
        return true;
    });
}

void error(httplib::Response& res, int status, const std::string& msg) {
    res.status = status;
    res.set_content(json{{"error", {{"message", msg}}}}.dump(), "application/json");
}

}  // namespace

const Expert* Config::find(const std::string& key) const {
    for (const auto& e : experts)
        if (e.key == key) return &e;
    return nullptr;
}

const char* default_experts_json() { return kDefaultPmExpertsJson; }

Config load_config(const std::string& file, const std::string& lemonade_url) {
    std::string text;
    if (file.empty()) {
        text = default_experts_json();
    } else {
        std::ifstream f(file);
        if (!f) throw std::runtime_error("--pm-experts " + file + ": cannot read");
        std::stringstream ss;
        ss << f.rdbuf();
        text = ss.str();
    }
    json j;
    try {
        j = json::parse(text);
    } catch (const std::exception& e) {
        throw std::runtime_error("--pm-experts " + (file.empty() ? std::string("(built in)") : file) + ": not JSON: " + e.what());
    }
    Config c;
    c.lemonade_url = lemonade_url;
    if (!j.is_object() || !j.contains("experts") || !j["experts"].is_array())
        throw std::runtime_error("--pm-experts: expected an object with an \"experts\" array");
    if (j.contains("pm") && j["pm"].is_object()) {
        const json& p = j["pm"];
        c.max_rounds = p.value("max_rounds", c.max_rounds);
        c.max_result_bytes = p.value("max_result_bytes", c.max_result_bytes);
        c.delegate_timeout_s = p.value("delegate_timeout_s", c.delegate_timeout_s);
        c.thinking = p.value("thinking", c.thinking);
        c.system_prompt = p.value("system_prompt", std::string());
        c.chat_path = p.value("chat_path", c.chat_path);
    }
    if (c.max_rounds < 1) throw std::runtime_error("--pm-experts: pm.max_rounds must be at least 1");
    for (const auto& x : j["experts"]) {
        Expert e;
        e.key = x.value("expert", std::string());
        e.id = x.value("id", std::string());
        e.domain = x.value("domain", std::string());
        e.use_when = x.value("use_when", std::string());
        e.max_tokens = x.value("max_tokens", 0);
        if (e.key.empty() || e.id.empty()) throw std::runtime_error("--pm-experts: every expert needs \"expert\" and \"id\"");
        if (c.find(e.key)) throw std::runtime_error("--pm-experts: expert \"" + e.key + "\" is listed twice");
        c.experts.push_back(std::move(e));
    }
    return c;
}

std::vector<ToolCall> parse_tool_calls(const json& message, const std::vector<std::string>& names) {
    std::vector<ToolCall> out;
    auto known = [&](const std::string& n) {
        for (const auto& k : names)
            if (k == n) return true;
        return false;
    };
    if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
        for (const auto& tc : message["tool_calls"]) {
            if (!tc.is_object() || !tc.contains("function")) continue;
            ToolCall c;
            c.id = tc.value("id", std::string());
            c.name = tc["function"].value("name", std::string());
            c.arguments = arguments_object(tc["function"].value("arguments", json()));
            if (known(c.name)) out.push_back(std::move(c));
        }
        if (!out.empty()) return out;
    }
    const std::string text = content_text(message.value("content", json()));
    if (text.empty()) return out;
    // ZAYA: <function=NAME><parameter=K>V</parameter>...</function>  (the closing tags may be missing:
    // the stop sequence ends the turn at </zyphra_tool_call>)
    static const std::regex fn_re(R"(<function=([\w.\-]+)>([\s\S]*?)(?:</function>|$))");
    static const std::regex param_re(R"(<parameter=([\w.\-]+)>\s*([\s\S]*?)\s*(?:</parameter>|(?=<parameter=)|$))");
    for (std::sregex_iterator it(text.begin(), text.end(), fn_re), end; it != end; ++it) {
        ToolCall c;
        c.name = (*it)[1];
        c.arguments = json::object();
        const std::string body = (*it)[2];
        for (std::sregex_iterator p(body.begin(), body.end(), param_re), pend; p != pend; ++p) c.arguments[(*p)[1]] = std::string((*p)[2]);
        if (known(c.name)) out.push_back(std::move(c));
    }
    if (!out.empty()) return out;
    // Qwen / JSON: {"name": "...", "arguments": {...}} anywhere in the text
    size_t pos = 0;
    while ((pos = text.find("{\"name\"", pos)) != std::string::npos) {
        json obj;
        size_t end = pos;
        if (object_at(text, pos, obj, &end)) {
            ToolCall c;
            c.name = obj.value("name", std::string());
            c.arguments = arguments_object(obj.value("arguments", obj.value("parameters", json::object())));
            if (known(c.name)) out.push_back(std::move(c));
            pos = end;
        } else {
            pos += 7;
        }
    }
    return out;
}

void handle_chat(const Config& cfg, int backend_port, const std::string& our_id, bool drop_model,
                 const std::string& set_model, const httplib::Request& req, httplib::Response& res) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        return error(res, 400, "request body is not JSON");
    }
    if (!body.is_object() || !body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
        return error(res, 400, "bad request: messages must be a non-empty array");
    const bool stream = body.value("stream", false);
    const std::string id = now_id("chatcmpl-");

    // The PM prompt goes first; a system message of the client's follows it.
    json messages = body["messages"];
    const std::string prompt = (cfg.system_prompt.empty() ? std::string(kBuiltinPrompt) : cfg.system_prompt) +
                               "\n\nYour experts:\n" + experts_text(cfg);
    if (messages[0].is_object() && messages[0].value("role", "") == "system")
        messages[0]["content"] = prompt + "\n\n" + content_text(messages[0].value("content", json()));
    else
        messages.insert(messages.begin(), json{{"role", "system"}, {"content", prompt}});

    const json tools = tools_schema(cfg);
    std::vector<std::string> names = {"experts", "delegate"};
    std::vector<Delegation> log;
    httplib::Client backend("127.0.0.1", backend_port);
    backend.set_read_timeout(3600);
    json final_message, final_reply;
    std::string finish = "stop";
    int rounds = 0;
    for (int round = 0; round < cfg.max_rounds; round++) {
        rounds = round + 1;
        const bool last = round + 1 == cfg.max_rounds;  // the cap: no tools, the PM must answer
        json rq = body;
        rq["messages"] = messages;
        rq["stream"] = false;
        if (drop_model) rq.erase("model");
        if (!set_model.empty()) rq["model"] = set_model;
        if (!rq.contains("temperature")) rq["temperature"] = 0.0;  // routing is a decision: greedy
        if (!rq.contains("chat_template_kwargs")) rq["chat_template_kwargs"] = {{"enable_thinking", cfg.thinking}};
        if (!last) {
            rq["tools"] = tools;
            // a tool call ends the turn (ZAYA otherwise writes on past its call)
            rq["stop"] = json::array({"</zyphra_tool_call>", "</tool_call>"});
        } else {
            rq.erase("tools");
            rq.erase("tool_choice");
        }
        auto r = backend.Post("/v1/chat/completions", rq.dump(), "application/json");
        if (!r) return error(res, 502, "backend did not answer");
        json out;
        try {
            out = json::parse(r->body);
        } catch (const std::exception&) {
            res.status = r->status;
            res.set_content(r->body, r->get_header_value("Content-Type"));
            return;
        }
        if (r->status != 200) {
            res.status = r->status;
            res.set_content(out.dump(), "application/json");
            return;
        }
        json message;
        try {
            message = out.at("choices").at(0).at("message");
            finish = json_string(out["choices"][0].value("finish_reason", json("stop")));
        } catch (const std::exception&) {
            return error(res, 502, "backend reply had no choices[0].message");
        }
        const std::vector<ToolCall> calls = last ? std::vector<ToolCall>{} : parse_tool_calls(message, names);
        if (calls.empty()) {
            final_message = message;
            final_reply = out;
            break;
        }
        // The PM's turn, in OpenAI form so the chat template renders the calls, then one tool
        // message per call.
        json assistant = {{"role", "assistant"}, {"content", ""}};
        json tcs = json::array();
        for (size_t i = 0; i < calls.size(); i++) {
            const std::string cid = calls[i].id.empty() ? "call_" + std::to_string(round) + "_" + std::to_string(i) : calls[i].id;
            tcs.push_back({{"id", cid}, {"type", "function"}, {"function", {{"name", calls[i].name}, {"arguments", calls[i].arguments.dump()}}}});
        }
        assistant["tool_calls"] = tcs;
        messages.push_back(assistant);
        for (size_t i = 0; i < calls.size(); i++) {
            std::string result;
            if (calls[i].name == "experts") result = experts_text(cfg);
            else result = run_delegate(cfg, calls[i].arguments, log);
            messages.push_back({{"role", "tool"}, {"tool_call_id", tcs[i]["id"]}, {"name", calls[i].name}, {"content", result}});
        }
        if (finish == "tool_calls") finish = "stop";
    }
    if (final_message.is_null()) return error(res, 502, "the project manager gave no answer");  // not reachable: the cap round has no tools
    // A PM that stopped after the tool result with nothing to say: the expert's answer is still
    // the answer; relay the last tool result rather than an empty message.
    if (content_text(final_message.value("content", json())).empty() && !messages.empty() &&
        messages.back().value("role", "") == "tool")
        final_message["content"] = messages.back()["content"];
    final_message.erase("tool_calls");

    json pm = {{"rounds", rounds}, {"delegations", json::array()}};
    for (const auto& d : log) pm["delegations"].push_back({{"expert", d.expert}, {"id", d.id}, {"ms", d.ms}, {"ok", d.ok}});
    res.set_header("X-1bit-PM", std::to_string(rounds) + " rounds, " + std::to_string(log.size()) + " delegations");
    if (stream) return stream_final(res, id, our_id, final_message, finish);
    json reply = final_reply;
    reply["id"] = id;
    reply["model"] = our_id;
    reply["choices"] = json::array({{{"index", 0}, {"message", final_message}, {"finish_reason", finish}}});
    reply["pm"] = pm;
    res.set_content(reply.dump(), "application/json");
}

}  // namespace onebit::pm
