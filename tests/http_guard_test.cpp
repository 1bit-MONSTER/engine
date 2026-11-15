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

// http_guard_test: which Host and Origin values app/http_guard.h lets through.
#include "http_guard.h"
#include "httplib.h"

#include <cstdio>
#include <string>
#include <thread>

static int failures = 0;

static void expect(bool allowed, const std::string& bound, const std::string& method, const std::string& host,
                   const std::string& origin) {
    const std::string why = onebit::refuse_request(bound, method, host, origin);
    if (why.empty() != allowed) {
        ++failures;
        std::printf("FAIL bound=%s %s Host=%s Origin=%s -> %s\n", bound.c_str(), method.c_str(), host.c_str(),
                    origin.c_str(), why.empty() ? "allowed" : why.c_str());
    }
}

int main() {
    // bound to loopback: loopback Host values only
    expect(true, "127.0.0.1", "GET", "127.0.0.1:8000", "");
    expect(true, "127.0.0.1", "GET", "localhost:8000", "");
    expect(true, "127.0.0.1", "GET", "LOCALHOST", "");
    expect(true, "127.0.0.1", "GET", "[::1]:8000", "");
    expect(true, "127.0.0.1", "GET", "127.1.2.3:9", "");
    expect(true, "127.0.0.1", "GET", "app.localhost:8000", "");
    expect(true, "127.0.0.1", "GET", "", "");  // HTTP/1.0, no Host
    expect(true, "localhost", "GET", "127.0.0.1:8000", "");
    expect(false, "127.0.0.1", "GET", "rebind.example:8000", "");
    expect(false, "127.0.0.1", "GET", "127.0.0.1.example:8000", "");
    expect(false, "127.0.0.1", "POST", "evil.localhost.example", "");
    expect(false, "::1", "GET", "rebind.example", "");
    // bound to every interface: any Host
    expect(true, "0.0.0.0", "GET", "192.168.1.20:8000", "");
    expect(true, "0.0.0.0", "POST", "box.lan:8000", "");
    // Origin on POST: loopback or the server's own origin only
    expect(true, "127.0.0.1", "POST", "127.0.0.1:8000", "http://localhost:3000");
    expect(true, "127.0.0.1", "POST", "127.0.0.1:8000", "http://127.0.0.1:8000");
    expect(false, "127.0.0.1", "POST", "127.0.0.1:8000", "https://evil.example");
    expect(false, "127.0.0.1", "POST", "127.0.0.1:8000", "null");
    expect(true, "127.0.0.1", "GET", "127.0.0.1:8000", "https://evil.example");  // reads stay same-origin
    expect(true, "0.0.0.0", "POST", "box.lan:8000", "http://box.lan:8000");
    expect(false, "0.0.0.0", "POST", "box.lan:8000", "http://other.lan:8000");
    expect(false, "0.0.0.0", "DELETE", "box.lan:8000", "https://evil.example");

    // engine#2482: what the HTTP layer refuses, not only what the guard refuses. cpp-httplib returns 413
    // with an EMPTY body when a request labelled `application/x-www-form-urlencoded` (which is what
    // `curl -d` sends) carries more than CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH -- 8 KiB by
    // default. The engine logged nothing, so a legitimate request looked like a deliberate refusal.
    // Same subject as the arms above: a request the server should let through.
    {
        httplib::Server srv;
        bool answered = false;
        srv.Post("/v1/completions", [&](const httplib::Request &, httplib::Response & res) {
            answered = true;
            res.set_content(R"({"ok":true})", "application/json");
        });
        const int port = srv.bind_to_any_port("127.0.0.1");
        if (port <= 0) { ++failures; std::printf("FAIL could not bind a loopback port\n"); }
        else {
            std::thread th([&] { srv.listen_after_bind(); });
            srv.wait_until_ready();   // void in v0.47.0; the POST is the real readiness check
            httplib::Client cli("127.0.0.1", port);
            cli.set_read_timeout(10, 0);
            cli.set_write_timeout(10, 0);
            auto short_r = cli.Post("/v1/completions", httplib::Headers{}, std::string(771, 'x'),
                                    "application/x-www-form-urlencoded");
            if (!(short_r && short_r->status == 200)) { ++failures; std::printf("FAIL 771-byte form body did not reach the handler\n"); }
            answered = false;
            auto long_r = cli.Post("/v1/completions", httplib::Headers{}, std::string(94 * 1024, 'x'),
                                   "application/x-www-form-urlencoded");
            if (!(long_r && long_r->status == 200)) { ++failures; std::printf("FAIL 94 KiB form body -> %d (engine#2482: the 8 KiB form limit)\n", long_r ? long_r->status : -1); }
            if (!answered) { ++failures; std::printf("FAIL the 94 KiB request never reached the handler\n"); }
            srv.stop();
            th.join();
        }
    }

    std::printf("%s\n", failures ? "FAILED" : "http_guard: ok");
    return failures ? 1 : 0;
}
