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

// The Loom GPU route's pure parts (app/loom_route.h, docs/loom-route.md): the route table and its match, what
// loom-serve's /v1/status means for readiness, and the request/reply normalisation between a llama-server
// client and loom-serve. Header only: no server, no model, no device. tests/loom_route.sh runs `1bit serve`.
#include "loom_route.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

namespace {
int failures = 0;
void check(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) failures++;
}
using nlohmann::json;
namespace loom = onebit::loom;
namespace fs = std::filesystem;
}  // namespace

int main() {
    // sha256 over a stream: the FIPS 180-2 vectors, and one fed in odd-sized pieces across block boundaries
    {
        loom::Sha256 a;
        check("sha256 of \"\"", a.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        loom::Sha256 b;
        const std::string abc = "abc";
        b.update(reinterpret_cast<const uint8_t*>(abc.data()), abc.size());
        check("sha256 of \"abc\"", b.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        const std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        loom::Sha256 c;
        for (size_t i = 0; i < m.size(); i += 7)
            c.update(reinterpret_cast<const uint8_t*>(m.data() + i), std::min<size_t>(7, m.size() - i));
        check("sha256 of the 448-bit vector, fed 7 bytes at a time",
              c.hex() == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    }

    // the shipped table is empty and parses; malformed entries are refused with the reason
    {
        loom::LoomRoutes t;
        std::string err;
        check("an empty table parses", t.parse(R"({"about":"x","routes":[]})", err) && t.routes.empty());
        check("a table without routes is refused", !t.parse(R"({"about":"x"})", err) && err.find("routes") != std::string::npos);
        check("an entry without gate is refused",
              !t.parse(R"({"routes":[{"id":"a","card":"c","match":{"model":["m"]}}]})", err) && err.find("gate") != std::string::npos);
        check("an entry without preset is refused (a card with several presets will not start without one)",
              !t.parse(R"({"routes":[{"id":"a","card":"c","gate":"g","match":{"model":["m"]}}]})", err) && err.find("preset") != std::string::npos);
        check("an entry without match is refused",
              !t.parse(R"({"routes":[{"id":"a","card":"c","gate":"g","preset":"chat"}]})", err) && err.find("match") != std::string::npos);
        check("an unknown key is refused (a typo would silently match everything)",
              !t.parse(R"({"routes":[{"id":"a","card":"c","gate":"g","preset":"chat","match":{"modle":["m"]}}]})", err) &&
                  err.find("modle") != std::string::npos);
        check("a sha256 without its size is refused",
              !t.parse(R"({"routes":[{"id":"a","card":"c","gate":"g","preset":"chat","match":{"gguf_sha256":")" + std::string(64, 'a') +
                       R"("}}]})", err) && err.find("gguf_size") != std::string::npos);
        check("a bad sha256 is refused",
              !t.parse(R"({"routes":[{"id":"a","card":"c","gate":"g","preset":"chat","match":{"gguf_sha256":"xyz","gguf_size":3}}]})", err));
        check("a full entry parses",
              t.parse(R"({"routes":[{"id":"a","card":"c","gate":"g","preset":"chat","device":"amdgpu://",
                          "args":["--memory_bytes=1"],"env":{"K":"V"},
                          "match":{"model":["m"],"gguf_sha256":")" + std::string(64, 'A') + R"(","gguf_size":3}}]})", err) &&
                  t.routes.size() == 1 && t.routes[0].sha256 == std::string(64, 'a') && t.routes[0].env[0] == "K=V" &&
                  t.routes[0].args[0] == "--memory_bytes=1");
    }

    // the match: names, then size, then sha256; the hash is computed only for a file whose size matches
    {
        const fs::path dir = fs::temp_directory_path() / ("loom_route_test_" + std::to_string(::getpid()));
        fs::create_directories(dir);
        const std::string file = (dir / "tiny.gguf").string();
        std::ofstream(file, std::ios::binary) << "abc";
        const std::string abc_sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        int hashed = 0;
        auto sha = [&](const std::string& p) { ++hashed; return loom::sha256_of_file(p); };
        loom::LoomRoutes t;
        std::string err;
        t.parse(R"({"routes":[
            {"id":"other-size","card":"c","gate":"g","preset":"chat","match":{"gguf_sha256":")" + abc_sha + R"(","gguf_size":4}},
            {"id":"by-name","card":"c","gate":"g","preset":"chat","match":{"model":["named"]}},
            {"id":"by-sha","card":"c","gate":"g","preset":"chat","match":{"gguf_sha256":")" + abc_sha + R"(","gguf_size":3}}]})", err);
        const loom::LoomRoute* r = loom::match_loom_route(t, file, {"", "tiny", "tiny.gguf"}, sha);
        check("the checkpoint's sha256 + size selects its entry", r && r->id == "by-sha");
        check("and the file was hashed once", hashed == 1);
        hashed = 0;
        r = loom::match_loom_route(t, file, {"named", "tiny", "tiny.gguf"}, sha);
        check("a model name selects its entry (first match wins)", r && r->id == "by-name" && hashed == 0);
        loom::LoomRoutes none;
        none.parse(R"({"routes":[{"id":"x","card":"c","gate":"g","preset":"chat","match":{"gguf_sha256":")" + std::string(64, 'b') +
                       R"(","gguf_size":3}}]})", err);
        hashed = 0;
        check("another checkpoint of the same size is no match", !loom::match_loom_route(none, file, {"tiny"}, sha) && hashed == 1);
        loom::LoomRoutes empty;
        empty.parse(R"({"routes":[]})", err);
        hashed = 0;
        check("the empty table matches nothing and reads nothing", !loom::match_loom_route(empty, file, {"tiny"}, sha) && hashed == 0);
        check("sha256_of_file streams the file", loom::sha256_of_file(file) == abc_sha);

        // the card: a directory means its model.json; model.id is the registered name
        fs::create_directories(dir / "card");
        std::ofstream(dir / "card" / "model.json") << R"({"model":{"id":"Org/Model-1"}})";
        loom::LoomRoute cr;
        cr.card = "card";
        const std::string card = loom::loom_card_path(cr, dir.string());
        std::string why;
        check("a relative card directory resolves to its model.json under the cards dir",
              card == (dir / "card" / "model.json").string());
        check("the card's model.id is read", loom::loom_card_model_id(card, why) == "Org/Model-1");
        loom::LoomRoute up;
        up.card = "../card";
        check("a relative card that climbs out of the cards dir is refused", loom::loom_card_path(up, (dir / "sub").string()).empty());
        std::ofstream(dir / "card" / "other.json") << "{}";
        loom::LoomRoute wrong;
        wrong.card = "card/other.json";
        check("a card file that is not model.json is refused", loom::loom_card_path(wrong, dir.string()).empty());
        loom::LoomRoute gone;
        gone.card = "nope";
        check("a card that does not exist is refused", loom::loom_card_path(gone, dir.string()).empty());
        check("a missing card says so", loom::loom_card_model_id((dir / "none.json").string(), why).empty() &&
                                            why.find("cannot read") != std::string::npos);
        fs::remove_all(dir);
    }

    // readiness from GET /v1/status: preparing waits, ready is ready, failed carries loom-serve's message
    {
        std::string why;
        auto status = [](const std::string& state) {
            return json{{"object", "list"},
                        {"data", json::array({{{"id", "Org/M"},
                                               {"loom", {{"registration", state}, {"error", {{"message", "PARSE/003: bad"}}}}}}})}}
                .dump();
        };
        check("an empty status waits", loom::loom_ready(R"({"object":"list","data":[]})", "Org/M", why) == loom::LoomReady::kWaiting);
        check("preparing waits", loom::loom_ready(status("preparing"), "Org/M", why) == loom::LoomReady::kWaiting);
        check("ready is ready", loom::loom_ready(status("ready"), "Org/M", why) == loom::LoomReady::kReady);
        check("another model's ready is not ours", loom::loom_ready(status("ready"), "Org/Other", why) == loom::LoomReady::kWaiting);
        why.clear();
        check("failed is failed, with the reason",
              loom::loom_ready(status("failed"), "Org/M", why) == loom::LoomReady::kFailed && why.find("PARSE/003") != std::string::npos);
        check("not JSON waits", loom::loom_ready("<html>", "Org/M", why) == loom::LoomReady::kWaiting);
    }

    // requests: llama-server's hints dropped, n_predict renamed, neutral defaults dropped, real unsupported fields refused
    {
        json b = {{"model", "served-id"}, {"messages", json::array()}, {"cache_prompt", true}, {"id_slot", 0},
                  {"speculative.n_max", 3}, {"n_predict", 32}, {"presence_penalty", 0}, {"n", 1}, {"stop", json::array()},
                  {"top_k", 1}, {"temperature", 0}};
        const loom::LoomRequest r = loom::loom_request(b, "Org/M");
        check("a llama-server request is accepted", r.ok);
        check("its serving hints are dropped", !b.contains("cache_prompt") && !b.contains("id_slot") && !b.contains("speculative.n_max"));
        check("n_predict becomes max_tokens", !b.contains("n_predict") && b["max_tokens"] == 32);
        check("neutral defaults are dropped", !b.contains("presence_penalty") && !b.contains("n") && !b.contains("stop"));
        check("loom-serve's own fields stay", b["top_k"] == 1 && b["temperature"] == 0);
        check("the model is the card's", b["model"] == "Org/M");
        check("not streamed", !r.stream && !b.contains("stream_options"));

        json m = {{"messages", json::array()}, {"max_tokens", 8}, {"n_predict", 99}};
        loom::loom_request(m, "Org/M");
        check("an explicit max_tokens wins over n_predict", m["max_tokens"] == 8);
        json neg = {{"messages", json::array()}, {"max_tokens", -1}};
        loom::loom_request(neg, "Org/M");
        check("a negative max_tokens (llama-server: until the context fills) leaves the deployment default", !neg.contains("max_tokens"));

        json s = {{"messages", json::array()}, {"stop", json::array({"\n"})}};
        const loom::LoomRequest rs = loom::loom_request(s, "Org/M");
        check("a real stop list is refused by name, not ignored", !rs.ok && rs.error.find("\"stop\"") != std::string::npos);
        json p = {{"messages", json::array()}, {"frequency_penalty", 0.5}};
        check("a real penalty is refused", !loom::loom_request(p, "Org/M").ok);

        json st = {{"messages", json::array()}, {"stream", true}};
        const loom::LoomRequest rst = loom::loom_request(st, "Org/M");
        check("a streamed request asks loom-serve for usage", rst.stream && !rst.client_usage &&
                                                               st["stream_options"]["include_usage"] == true);
        json stu = {{"messages", json::array()}, {"stream", true}, {"stream_options", {{"include_usage", true}}}};
        check("a client's own include_usage is remembered", loom::loom_request(stu, "Org/M").client_usage);
    }

    // a non-streamed reply: served id, timings from usage (counts only: no first-token time)
    {
        json out = {{"model", "Org/M"}, {"choices", json::array()},
                    {"usage", {{"prompt_tokens", 20}, {"completion_tokens", 5}, {"prompt_tokens_details", {{"cached_tokens", 4}}}}}};
        loom::loom_reply(out, "served-id");
        check("the reply names the served model", out["model"] == "served-id");
        check("timings carry the counts Lemonade reads",
              out["timings"]["prompt_n"] == 16 && out["timings"]["cache_n"] == 4 && out["timings"]["predicted_n"] == 5);
        check("and no invented times", !out["timings"].contains("prompt_ms") && out["timings"].contains("measured_by"));
        json err = {{"error", {{"message", "unsupported chat field 'x'"}, {"type", "loom_request_error"}}}};
        loom::loom_reply(err, "served-id");
        check("an error reply is left as loom-serve wrote it", !err.contains("timings") && !err.contains("model"));
    }

    // a streamed reply: model renamed, finish chunk held and merged with the usage serve asked for, timings added
    {
        double clock = 0;
        auto now = [&] { return clock; };
        loom::LoomSse sse("served-id", false, now);
        std::string out;
        auto chunk = [](const json& j) { return "data: " + j.dump() + "\n\n"; };
        clock = 5;
        const std::string c0 = chunk({{"model", "Org/M"}, {"choices", json::array({{{"index", 0}, {"delta", {{"role", "assistant"}}}}})}});
        out += sse.feed(c0.data(), c0.size());
        const std::string c1 = chunk({{"model", "Org/M"}, {"choices", json::array({{{"index", 0}, {"delta", {{"content", "Par"}}}}})}});
        clock = 100;   // first text at 100 ms
        // split across two reads, mid-event
        out += sse.feed(c1.data(), 10);
        out += sse.feed(c1.data() + 10, c1.size() - 10);
        clock = 300;
        const std::string fin = chunk({{"model", "Org/M"}, {"choices", json::array({{{"index", 0}, {"delta", json::object()},
                                                                                      {"finish_reason", "stop"}}})}});
        out += sse.feed(fin.data(), fin.size());
        check("the finish chunk is held until usage arrives", out.find("finish_reason") == std::string::npos);
        clock = 500;
        const std::string use = chunk({{"model", "Org/M"}, {"choices", json::array()},
                                       {"usage", {{"prompt_tokens", 10}, {"completion_tokens", 5}}}});
        out += sse.feed(use.data(), use.size());
        const std::string fin_done = "data: [DONE]\n\n";
        out += sse.feed(fin_done.data(), fin_done.size());
        out += sse.finish();
        check("every chunk names the served model", out.find("Org/M") == std::string::npos && out.find("served-id") != std::string::npos);
        // the events, parsed back
        std::vector<json> evs;
        bool done = false;
        for (size_t pos = 0; (pos = out.find("data: ", pos)) != std::string::npos;) {
            const size_t end = out.find("\n\n", pos);
            const std::string d = out.substr(pos + 6, end - pos - 6);
            if (d == "[DONE]") done = true;
            else evs.push_back(json::parse(d));
            pos = end;
        }
        check("[DONE] ends the stream", done && sse.saw_done());
        check("three chunks: role, text, final (the usage-only chunk folded in)", evs.size() == 3);
        const json& last = evs.back();
        check("the final chunk carries finish_reason, usage and timings",
              last["choices"][0]["finish_reason"] == "stop" && last["usage"]["completion_tokens"] == 5 && last.contains("timings"));
        check("prompt_ms = first text at serve's clock", last["timings"]["prompt_ms"] == 100.0);
        check("predicted_ms = first text to usage", last["timings"]["predicted_ms"] == 400.0);
        check("predicted_per_second counts the decode steps after the first token",
              last["timings"]["predicted_per_second"] == 4 * 1000.0 / 400.0);
    }
    {
        // the client asked for include_usage: OpenAI's separate usage chunk, timings added to it
        loom::LoomSse sse("served-id", true, [] { return 1.0; });
        const std::string in = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                               "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}\n\n"
                               "data: [DONE]\n\n";
        std::string out = sse.feed(in.data(), in.size());
        out += sse.finish();
        check("include_usage: the finish chunk is passed on at once", out.find("finish_reason") < out.find("\"usage\""));
        check("and the usage chunk keeps its own place, with timings",
              out.find("\"choices\":[]") != std::string::npos && out.find("timings") != std::string::npos);
    }
    {
        // loom-serve answered a streamed request with a plain JSON error (a rejected request): one SSE error + [DONE]
        loom::LoomSse sse("served-id", false, [] { return 0.0; });
        const std::string body = R"({"error":{"message":"unsupported chat field 'x'","type":"loom_request_error"}})";
        std::string out = sse.feed(body.data(), body.size());
        out += sse.finish();
        check("a plain JSON error becomes an SSE error event and [DONE]",
              out.rfind("data: {\"error\"", 0) == 0 && out.find("unsupported chat field") != std::string::npos &&
                  out.find("data: [DONE]") != std::string::npos);
    }

    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
