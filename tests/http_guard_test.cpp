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
#include <httplib.h>

#include "http_guard.h"

#include <thread>

#include <cstdio>
#include <string>

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

    // engine#2482: the server path itself. A JSON body over 8 KB sent the way `curl -d` sends it (labelled
    // application/x-www-form-urlencoded) must reach the route; an error httplib raises before routing must come back
    // with a JSON body, and its log line must name the cause and the request.
    {
        httplib::Server srv;
        onebit::install_request_guard(srv, "127.0.0.1");
        srv.Post("/v1/completions", [](const httplib::Request& q, httplib::Response& r) {
            r.set_content(std::to_string(q.body.size()), "text/plain");
        });
        const int port = srv.bind_to_any_port("127.0.0.1");
        std::thread t([&] { srv.listen_after_bind(); });
        srv.wait_until_ready();
        httplib::Client cli("127.0.0.1", port);
        const std::string big = "{\"prompt\":\"" + std::string(20000, 'a') + "\"}";
        auto res = cli.Post("/v1/completions", big, "application/x-www-form-urlencoded");
        if (!res || res->status != 200 || res->body != std::to_string(big.size())) {
            std::printf("FAIL: a %zu B JSON body sent as form-urlencoded: status %d, body '%s'\n", big.size(),
                        res ? res->status : -1, res ? res->body.c_str() : "");
            ++failures;
        }
        srv.set_payload_max_length(4096);
        res = cli.Post("/v1/completions", std::string(5000, 'b'), "application/json");
        if (!res || res->status != 413 || res->body.find("\"code\":413") == std::string::npos ||
            res->get_header_value("Content-Type") != "application/json") {
            std::printf("FAIL: an over-limit body: status %d, body '%s'\n", res ? res->status : -1,
                        res ? res->body.c_str() : "");
            ++failures;
        }
        srv.stop();
        t.join();
    }
    {
        httplib::Request q;
        q.method = "POST";
        q.path = "/v1/completions";
        q.headers.emplace("Content-Type", "application/x-www-form-urlencoded");
        q.body = std::string(9000, 'c');
        const std::string line = onebit::request_error_line("ExceedMaxPayloadSize", &q);
        if (line.find("ExceedMaxPayloadSize") == std::string::npos || line.find("/v1/completions") == std::string::npos ||
            line.find("application/x-www-form-urlencoded") == std::string::npos || line.find("9000 B") == std::string::npos) {
            std::printf("FAIL: the error log line: %s\n", line.c_str());
            ++failures;
        }
        httplib::Response r;
        r.status = 404;
        r.set_content("{\"error\":\"own\"}", "application/json");
        onebit::fill_empty_error_body(r);
        if (r.body != "{\"error\":\"own\"}") {
            std::printf("FAIL: a route's own error body was replaced: %s\n", r.body.c_str());
            ++failures;
        }
    }
    std::printf("%s\n", failures ? "FAILED" : "http_guard: ok");
    return failures ? 1 : 0;
}
