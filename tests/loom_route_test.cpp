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
#include "loom_routes_default.h"  // generated from config/loom-routes.json (the table serve ships)

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
    using onebit::loom::StreamShape;
    using onebit::loom::loom_stream_shape;
    {   // engine#3254 review: the shape decides whether the client gets a stream or a status, so a partial line must
        // not be guessed. `"da"` is a prefix of `data:` and used to be classified as "not a stream" -- which handed the
        // client an SSE body under `application/json`, the exact symptom the status pass exists to remove.
        check("a SSE data line is a stream", loom_stream_shape("data: {\"x\":1}\n\n") == StreamShape::Sse);
        check("a half-read marker asks for more bytes", loom_stream_shape("da") == StreamShape::NeedMore);
        check("...and a complete marker with no payload is still a stream",
              loom_stream_shape("data:") == StreamShape::Sse);
        check("event:/id:/comment lines before data: are a stream",
              loom_stream_shape("event: m\nid: 7\n: keepalive\ndata: {}\n\n") == StreamShape::Sse);
        check("a partial event line asks for more bytes", loom_stream_shape("eve") == StreamShape::NeedMore);
        check("a JSON refusal is not a stream", loom_stream_shape("{\"error\":{\"message\":\"no\"}}") == StreamShape::NotSse);
        check("...even when it is split after one brace", loom_stream_shape("{") == StreamShape::NotSse);
        check("an HTML error page is not a stream", loom_stream_shape("<html><body>502</body></html>") == StreamShape::NotSse);
        check("whitespace only says nothing yet", loom_stream_shape("\r\n") == StreamShape::NeedMore);
        check("an empty body says nothing yet", loom_stream_shape("") == StreamShape::NeedMore);
    }

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

    // a route's default reasoning effort: a preset name or a number in [0, 1); set only on a request that chose none
    {
        loom::LoomRoutes t;
        std::string err;
        auto route = [](const std::string& eff) {
            return R"({"routes":[{"id":"a","card":"c","gate":"g","preset":"chat","match":{"model":["m"]},"reasoning_effort":)" +
                   eff + "}]}";
        };
        check("no reasoning_effort: the card's own default",
              t.parse(R"({"routes":[{"id":"a","card":"c","gate":"g","preset":"chat","match":{"model":["m"]}}]})", err) &&
                  t.routes[0].reasoning_effort.is_null());
        check("a preset name parses", t.parse(route(R"("medium")"), err) && t.routes[0].reasoning_effort == "medium");
        check("a number parses", t.parse(route("0.7"), err) && t.routes[0].reasoning_effort == 0.7);
        check("an unknown name is refused", !t.parse(route(R"("hihg")"), err) && err.find("reasoning_effort") != std::string::npos);
        check("1.0 is refused (the template takes [0, 1))", !t.parse(route("1.0"), err));
        check("a negative number is refused", !t.parse(route("-0.1"), err));
        check("a boolean is refused", !t.parse(route("true"), err));

        const json eff = "medium";
        json b = {{"messages", json::array()}};
        check("a request with no effort gets the route's", loom::loom_default_effort(b, eff) && b["reasoning_effort"] == "medium");
        json n = {{"messages", json::array()}};
        check("no route default: the request is left alone", !loom::loom_default_effort(n, json()) && !n.contains("reasoning_effort"));
        for (const json& c : {json{{"reasoning_effort", "high"}}, json{{"enable_thinking", false}},
                              json{{"reasoning", {{"effort", "low"}}}},
                              json{{"chat_template_kwargs", {{"reasoning_effort", 0.2}}}},
                              json{{"chat_template_kwargs", {{"enable_thinking", true}}}}}) {
            json q = c;
            q["messages"] = json::array();
            const json before = q;
            check("a client's own choice wins: " + c.dump(), !loom::loom_default_effort(q, eff) && q == before);
        }
        json k = {{"messages", json::array()}, {"chat_template_kwargs", {{"other", 1}}}};
        check("other template kwargs do not count as a choice", loom::loom_default_effort(k, eff) && k["reasoning_effort"] == "medium");
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

        // engine#3350: a card's parameter/ links follow the model serve was given (whatever snapshot the pull made)
        fs::create_directories(dir / "lc" / "parameter");
        fs::create_directories(dir / "snap");
        std::ofstream(dir / "lc" / "model.json") << R"({"model":{"id":"M"},"variants":{"gguf":{"services":{"chat":{"parameters":{)"
            R"("model":{"files":[{"path":"parameter/m.gguf","origin":"@checkpoint/m.gguf"}]},)"
            R"("draft":{"files":[{"path":"parameter/d.gguf","origin":"@checkpoint/d.gguf"}]},)"
            R"("tok":{"files":[{"path":"tokenizer.json","origin":"tokenizer.json"}]},)"
            R"("bad":{"files":[{"path":"../escape.gguf","origin":"@checkpoint/x.gguf"}]},)"
            R"("abs":{"files":[{"path":")" + (dir / "victim-link").string() + R"(","origin":"@checkpoint/x.gguf"}]},)"
            R"("via":{"files":[{"path":"outdir/v.gguf","origin":"@checkpoint/x.gguf"}]}}}}}}})";
        // a symlink the machine owns, outside the card: an absolute card path must not delete and re-point it
        fs::create_directories(dir / "elsewhere");
        fs::create_symlink(dir / "elsewhere" / "real-target", dir / "victim-link");
        // a symlinked directory inside the card that leads out of it
        fs::create_directory_symlink(dir / "elsewhere", dir / "lc" / "outdir");
        std::ofstream(dir / "snap" / "m.gguf") << "m";
        std::ofstream(dir / "snap" / "d.gguf") << "d";
        std::ofstream(dir / "snap" / "x.gguf") << "x";
        fs::create_symlink(dir / "old-rev" / "m.gguf", dir / "lc" / "parameter" / "m.gguf");   // the lock's revision: dangling
        std::string lwhy;
        const int made = loom::loom_link_card_parameters((dir / "lc" / "model.json").string(), (dir / "snap" / "m.gguf").string(), lwhy);
        check("a dangling weight link and a missing one are pointed beside the served model", made == 2 && lwhy.empty() &&
              fs::read_symlink(dir / "lc" / "parameter" / "m.gguf") == dir / "snap" / "m.gguf" &&
              fs::read_symlink(dir / "lc" / "parameter" / "d.gguf") == dir / "snap" / "d.gguf");
        check("a path that climbs out of the card is not linked", !fs::exists(fs::symlink_status(dir / "escape.gguf")));
        check("an ABSOLUTE card path does not re-point a symlink outside the card (serve runs as root)",
              fs::read_symlink(dir / "victim-link") == dir / "elsewhere" / "real-target");
        check("a symlinked directory in the card does not lead a link out of it",
              !fs::exists(fs::symlink_status(dir / "elsewhere" / "v.gguf")));
        check("a second call changes nothing",
              loom::loom_link_card_parameters((dir / "lc" / "model.json").string(), (dir / "snap" / "m.gguf").string(), lwhy) == 0);
        fs::remove(dir / "lc" / "parameter" / "d.gguf");
        std::ofstream(dir / "lc" / "parameter" / "d.gguf") << "a real copy";                      // --materialize=copy
        loom::loom_link_card_parameters((dir / "lc" / "model.json").string(), (dir / "snap" / "m.gguf").string(), lwhy);
        check("a real file in parameter/ is never replaced", !fs::is_symlink(fs::symlink_status(dir / "lc" / "parameter" / "d.gguf")));

        // engine#3528: a RELATIVE -m (as the payload README's own `./1bit serve -m model.gguf`) is linked by its absolute
        // path - the link target used to be written relative to the cards directory, where it dangles
        fs::create_directories(dir / "lc_rel" / "parameter");
        std::ofstream(dir / "lc_rel" / "model.json") << R"({"model":{"id":"M2"},"variants":{"gguf":{"services":{"chat":{"parameters":{)"
            R"("model":{"files":[{"path":"parameter/m.gguf","origin":"@checkpoint/m.gguf"}]}}}}}}})";
        {
            const fs::path was = fs::current_path();
            fs::current_path(dir);
            std::string rwhy;
            const int rmade = loom::loom_link_card_parameters((dir / "lc_rel" / "model.json").string(), "snap/m.gguf", rwhy);
            const fs::path tgt = fs::read_symlink(dir / "lc_rel" / "parameter" / "m.gguf");
            fs::current_path(was);
            check("a relative model path is linked by its ABSOLUTE path (engine#3528)",
                  rmade == 1 && tgt.is_absolute() && tgt == dir / "snap" / "m.gguf" && fs::exists(dir / "lc_rel" / "parameter" / "m.gguf"));
        }
        // engine#3528: the defaults when no flag and no environment says (a shell on the appliance has neither)
        check("cards: --loom-cards wins over everything", loom::default_cards_dir("/x", "/env", true, "/xdg", "/home/u") == "/x");
        check("cards: ONEBIT_LOOM_CARDS wins over the appliance dir", loom::default_cards_dir("", "/env", true, "/xdg", "/home/u") == "/env");
        check("cards: the appliance's /data/loom-cards when it exists", loom::default_cards_dir("", nullptr, true, "/xdg", "/home/u") == "/data/loom-cards");
        check("cards: an empty ONEBIT_LOOM_CARDS is ignored", loom::default_cards_dir("", "", true, nullptr, "/home/u") == "/data/loom-cards");
        check("cards: elsewhere, XDG then HOME as before",
              loom::default_cards_dir("", nullptr, false, "/xdg", "/home/u") == "/xdg/1bit/loom-cards" &&
              loom::default_cards_dir("", nullptr, false, nullptr, "/home/u") == "/home/u/.cache/1bit/loom-cards" &&
              loom::default_cards_dir("", nullptr, false, nullptr, nullptr).empty());
        check("loom-serve: flag > ONEBIT_LOOM_SERVE > the payload's beside the binary > PATH",
              loom::default_loom_serve_path("/f", "/e", "/b") == "/f" && loom::default_loom_serve_path("", "/e", "/b") == "/e" &&
              loom::default_loom_serve_path("", nullptr, "/opt/1bit-package/loom/loom-serve") == "/opt/1bit-package/loom/loom-serve" &&
              loom::default_loom_serve_path("", "", "") == "loom-serve");
        // the card's lock names a size per weight file: a sibling of another size (the repo replaced the file under the
        // card's name, as unsloth/Inkling-2-Mini-GGUF is about to do with its DSpark drafter) is refused, not linked
        fs::remove(dir / "lc" / "parameter" / "d.gguf");
        fs::remove(dir / "lc" / "parameter" / "m.gguf");
        std::ofstream(dir / "lc" / "model.json.lock") << R"({"files":[{"origin":"@checkpoint/m.gguf","size":1},)"
            R"({"origin":"@checkpoint/d.gguf","size":99}]})";
        bool refused = false;
        std::string swhy;
        const int sm = loom::loom_link_card_parameters((dir / "lc" / "model.json").string(), (dir / "snap" / "m.gguf").string(),
                                                       swhy, &refused);
        // a link the card needs but cannot write (a read-only parameter/ dir: a non-root shell vs the appliance's root-owned
        // /data/loom-cards) refuses the card, so serve takes the llama.cpp route instead of starting Loom on a missing link
        {
            fs::create_directories(dir / "lc5" / "parameter");
            std::ofstream(dir / "lc5" / "model.json") << R"({"model":{"id":"M5"},"variants":{"gguf":{"services":{"chat":{"parameters":{)"
                R"("model":{"files":[{"path":"parameter/m.gguf","origin":"@checkpoint/m.gguf"}]}}}}}}})";
            fs::permissions(dir / "lc5" / "parameter", fs::perms::owner_read | fs::perms::owner_exec);
            bool r5 = false;
            std::string w5;
            const int n5 = loom::loom_link_card_parameters((dir / "lc5" / "model.json").string(), (dir / "snap" / "m.gguf").string(), w5, &r5);
            if (geteuid() == 0) {
                std::printf("skip (running as root, a read-only dir does not stop root): a link that cannot be written refuses the card\n");
            } else {
                check("a link that cannot be written refuses the card (and says why)",
                      n5 == 0 && r5 && w5.find("cannot link") != std::string::npos);
            }
            fs::permissions(dir / "lc5" / "parameter", fs::perms::owner_all);
            // review (cl-flm): a STALE link that cannot be replaced (another model's file, read-only dir) also refuses
            std::error_code rm_ec;   // as root the first arm did link m.gguf: start the stale case from a known link
            fs::remove(dir / "lc5" / "parameter" / "m.gguf", rm_ec);
            fs::create_symlink(dir / "snap" / "d.gguf", dir / "lc5" / "parameter" / "m.gguf");
            fs::permissions(dir / "lc5" / "parameter", fs::perms::owner_read | fs::perms::owner_exec);
            bool r6 = false;
            std::string w6;
            loom::loom_link_card_parameters((dir / "lc5" / "model.json").string(), (dir / "snap" / "m.gguf").string(), w6, &r6);
            if (geteuid() == 0) {
                std::printf("skip (running as root): a stale link that cannot be replaced refuses the card\n");
            } else {
                check("a stale link that cannot be replaced refuses the card (not served on the stale target)",
                      r6 && w6.find("cannot link") != std::string::npos &&
                      fs::read_symlink(dir / "lc5" / "parameter" / "m.gguf") == dir / "snap" / "d.gguf");
            }
            fs::permissions(dir / "lc5" / "parameter", fs::perms::owner_all);
        }
        // review (cl-hf): the same refusal, ROOT-PROOF (CI's build container runs as root): `parameter` is a regular file,
        // so no link can be made under it whoever runs the test
        {
            fs::create_directories(dir / "lc6");
            std::ofstream(dir / "lc6" / "model.json") << R"({"model":{"id":"M6"},"variants":{"gguf":{"services":{"chat":{"parameters":{)"
                R"("model":{"files":[{"path":"parameter/m.gguf","origin":"@checkpoint/m.gguf"}]}}}}}}})";
            std::ofstream(dir / "lc6" / "parameter") << "not a directory";
            bool r7 = false;
            std::string w7;
            const int n7 = loom::loom_link_card_parameters((dir / "lc6" / "model.json").string(), (dir / "snap" / "m.gguf").string(), w7, &r7);
            check("a link whose directory cannot exist refuses the card, also as root (and says why)",
                  n7 == 0 && r7 && w7.find("cannot link") != std::string::npos);
        }
        // engine#3553 review: another named origin's weight (the native Inkling card's `@mxfp8/` experts, pulled from their
        // own repo) must be in place in the variant that serves this model, or the card is refused naming the repo to pull
        {
            fs::create_directories(dir / "lc7" / "parameter");
            std::ofstream(dir / "lc7" / "model.json") << R"({"model":{"id":"M7"},"origins":{"checkpoint":{"kind":"hf","repository":"u/ck"},)"
                R"("mxfp8":{"kind":"hf","repository":"org/mxfp8-experts"}},"variants":{)"
                R"("gguf":{"services":{"chat":{"parameters":{"model":{"files":[{"path":"parameter/m.gguf","origin":"@checkpoint/m.gguf"},)"
                R"({"path":"parameter/x.safetensors","origin":"@mxfp8/x.safetensors"}]}}}}},)"
                R"("other":{"services":{"chat":{"parameters":{"model":{"files":[{"path":"parameter/o.gguf","origin":"@checkpoint/o.gguf"},)"
                R"({"path":"parameter/y.safetensors","origin":"@mxfp8/y.safetensors"}]}}}}}}})";
            std::ofstream(dir / "lc7" / "model.json.lock") << R"({"files":[{"origin":"@checkpoint/m.gguf","size":1},)"
                R"({"origin":"@mxfp8/x.safetensors","size":3},{"origin":"@mxfp8/y.safetensors","size":3}]})";
            bool r8 = false;
            std::string w8;
            loom::loom_link_card_parameters((dir / "lc7" / "model.json").string(), (dir / "snap" / "m.gguf").string(), w8, &r8);
            check("a missing @mxfp8/ weight in the served variant refuses the card and names the origin, size and repo",
                  r8 && w8.find("@mxfp8/x.safetensors (3 B) is not linked") != std::string::npos &&
                  w8.find("pull org/mxfp8-experts") != std::string::npos && w8.find("y.safetensors") == std::string::npos);
            std::ofstream(dir / "snap" / "x.safetensors") << "xyz";
            fs::create_symlink(dir / "snap" / "x.safetensors", dir / "lc7" / "parameter" / "x.safetensors");
            bool r9 = false;
            std::string w9;
            loom::loom_link_card_parameters((dir / "lc7" / "model.json").string(), (dir / "snap" / "m.gguf").string(), w9, &r9);
            check("...linked at the lock's size it is served (and the unused variant's absent @mxfp8/ file does not refuse)",
                  !r9 && w9.empty());
            fs::remove(dir / "lc7" / "parameter" / "x.safetensors");
            std::ofstream(dir / "snap" / "x2.safetensors") << "another size";
            fs::create_symlink(dir / "snap" / "x2.safetensors", dir / "lc7" / "parameter" / "x.safetensors");
            bool r10 = false;
            std::string w10;
            loom::loom_link_card_parameters((dir / "lc7" / "model.json").string(), (dir / "snap" / "m.gguf").string(), w10, &r10);
            check("...and one of another size than the lock refuses the card",
                  r10 && w10.find("@mxfp8/x.safetensors (3 B) has another size") != std::string::npos);
        }
        // engine#3553 review (cl-hf): the named origin's file is in the HF cache, but in the snapshot of the repo's head (a
        // pull by `main`), not the lock's revision boot linked: serve finds it there by the lock's size, sha256 breaking a tie
        {
            const fs::path hub = dir / "hub";
            const fs::path ckdir = hub / "models--u--ck" / "snapshots" / "r1";
            fs::create_directories(ckdir);
            std::ofstream(ckdir / "m.gguf") << "m";
            const fs::path mx = hub / "models--org--mxfp8-experts" / "snapshots";
            fs::create_directories(mx / "head1");
            fs::create_directories(mx / "head2");
            std::ofstream(mx / "head1" / "x.safetensors") << "abc";   // same size, other bytes: must lose on sha
            std::ofstream(mx / "head2" / "x.safetensors") << "xyz";
            fs::create_directories(dir / "lc8" / "parameter");
            std::ofstream(dir / "lc8" / "model.json") << R"({"model":{"id":"M8"},"origins":{"checkpoint":{"kind":"hf","repository":"u/ck"},)"
                R"("mxfp8":{"kind":"hf","repository":"org/mxfp8-experts"}},"variants":{)"
                R"("gguf":{"services":{"chat":{"parameters":{"model":{"files":[{"path":"parameter/m.gguf","origin":"@checkpoint/m.gguf"},)"
                R"({"path":"parameter/x.safetensors","origin":"@mxfp8/x.safetensors"}]}}}}}}})";
            std::ofstream(dir / "lc8" / "model.json.lock") << R"({"files":[{"origin":"@checkpoint/m.gguf","size":1},)"
                R"({"origin":"@mxfp8/x.safetensors","size":3,"sha256":")" + loom::sha256_of_file((mx / "head2" / "x.safetensors").string()) + R"("}]})";
            fs::create_symlink(mx / "lockrev" / "x.safetensors", dir / "lc8" / "parameter" / "x.safetensors");   // boot's link: dangling
            bool r11 = false;
            std::string w11;
            loom::loom_link_card_parameters((dir / "lc8" / "model.json").string(), (ckdir / "m.gguf").string(), w11, &r11);
            check("a named origin's file in another snapshot of its repo is found by the lock's size + sha256 and linked",
                  !r11 && w11.empty() && fs::canonical(dir / "lc8" / "parameter" / "x.safetensors") == fs::canonical(mx / "head2" / "x.safetensors"));
            fs::remove_all(mx);
            fs::remove(dir / "lc8" / "parameter" / "x.safetensors");
            bool r12 = false;
            std::string w12;
            loom::loom_link_card_parameters((dir / "lc8" / "model.json").string(), (ckdir / "m.gguf").string(), w12, &r12);
            check("...and with no snapshot of that repo the card is refused, naming the repo to pull",
                  r12 && w12.find("pull org/mxfp8-experts") != std::string::npos);
        }
        // a file the card lists but nothing beside the model provides (an unused variant) does not refuse the card
        {
            fs::create_directories(dir / "lc2" / "parameter");
            std::ofstream(dir / "lc2" / "model.json") << R"({"model":{"id":"M2"},"variants":{"q4":{"services":{"chat":{"parameters":{)"
                R"("model":{"files":[{"path":"parameter/m.gguf","origin":"@checkpoint/m.gguf"}]}}}}},)"
                R"("q8":{"services":{"chat":{"parameters":{"model":{"files":[{"path":"parameter/q8.gguf","origin":"@checkpoint/q8.gguf"}]}}}}}}})";
            std::ofstream(dir / "lc2" / "model.json.lock") << R"({"files":[{"origin":"@checkpoint/m.gguf","size":1},)"
                R"({"origin":"@checkpoint/q8.gguf","size":7,"sha256":"0000000000000000000000000000000000000000000000000000000000000000"}]})";
            bool r4 = false;
            std::string w4;
            const int n4 = loom::loom_link_card_parameters((dir / "lc2" / "model.json").string(), (dir / "snap" / "m.gguf").string(), w4, &r4);
            check("an unused variant's absent file does not refuse the card (the used file is linked)",
                  n4 == 1 && !r4 && fs::read_symlink(dir / "lc2" / "parameter" / "m.gguf") == dir / "snap" / "m.gguf");
        }
        check("a weight whose size differs from the card's lock is not linked and the card is refused",
              sm == 1 && refused && fs::is_symlink(fs::symlink_status(dir / "lc" / "parameter" / "m.gguf")) &&
              !fs::exists(fs::symlink_status(dir / "lc" / "parameter" / "d.gguf")) &&
              swhy.find("size (99) and sha256 the card's lock gives for @checkpoint/d.gguf") != std::string::npos);
        // option (b): the lock's sha256 finds the right bytes under ANOTHER name (unsloth's dspark-..._Old.gguf), and
        // the file under the card's name (replaced, another size) is not used
        {
            std::ofstream(dir / "snap" / "d.gguf") << "a new layout";
            std::ofstream(dir / "snap" / "d_Old.gguf") << "dd";
            std::ofstream(dir / "snap" / "z_same_size.gguf") << "zz";   // same size, other bytes: must lose on sha
            std::ofstream(dir / "lc" / "model.json.lock") << R"({"files":[{"origin":"@checkpoint/m.gguf","size":1},)"
                R"({"origin":"@checkpoint/d.gguf","size":2,"sha256":")" + loom::sha256_of_file((dir / "snap" / "d_Old.gguf").string()) +
                R"("}]})";
            bool r2 = false;
            std::string w2;
            const int n2 = loom::loom_link_card_parameters((dir / "lc" / "model.json").string(),
                                                           (dir / "snap" / "m.gguf").string(), w2, &r2);
            check("a drafter replaced under the card's name is found by the lock's sha under its other name",
                  n2 == 1 && !r2 && w2.empty() &&
                  fs::read_symlink(dir / "lc" / "parameter" / "d.gguf") == dir / "snap" / "d_Old.gguf");
            std::string w3;
            check("a link that already resolves to a file of the lock's size is kept (no re-hash, no change)",
                  loom::loom_link_card_parameters((dir / "lc" / "model.json").string(), (dir / "snap" / "m.gguf").string(), w3) == 0 &&
                  fs::read_symlink(dir / "lc" / "parameter" / "d.gguf") == dir / "snap" / "d_Old.gguf");
        }
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

    // the SHIPPED table (config/loom-routes.json, compiled into serve): it parses, and every entry names the exact
    // checkpoint the gate measured (sha256 + size, so a different upload never takes the route) and its gate record
    {
        loom::LoomRoutes shipped;
        std::string err;
        const bool ok = shipped.parse(kDefaultLoomRoutesJson, err);
        check("the shipped table parses" + (ok ? std::string() : ": " + err), ok);
        bool pinned = true, gated = true;
        for (const loom::LoomRoute& r : shipped.routes) {
            pinned = pinned && r.sha256.size() == 64 && r.size > 0 && !r.card.empty() && !r.preset.empty();
            gated = gated && r.gate.find("issues/3158") != std::string::npos;
        }
        check("every shipped entry pins a sha256 + size, a card and a preset", pinned);
        check("every shipped entry links its #3158 gate record", gated);
    }

    std::printf("%s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
