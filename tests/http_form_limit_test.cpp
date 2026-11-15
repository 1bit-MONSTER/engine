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
// http_form_limit_test: a form-labelled body larger than 8 KiB must reach the handler, not 413.
//
// engine#2482: `curl -d` sends `Content-Type: application/x-www-form-urlencoded` for a JSON body, so
// cpp-httplib parsed it on the form path and refused anything over CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH
// (8 KiB by default) with 413 and an empty reply -- and the engine logged nothing, so the client saw a
// rejection with no reason. This asserts the compile-time limit, which is the thing that changed: it does
// NOT need a model, a device or `1bit serve`.
//
// The arm is deliberately "the handler ran", not "not 413": a 413 is one way to fail and a dropped
// connection is another, and asserting the positive is the only version that cannot pass on a crash.
#include "httplib.h"

#include <cstdio>
#include <string>
#include <thread>

static int failures = 0;
static void chk(const char * what, bool ok) {
    if (!ok) { std::printf("FAIL %s\n", what); failures++; }
    else     { std::printf("ok   %s\n", what); }
}

int main() {
    httplib::Server srv;
    bool reached = false;
    srv.Post("/v1/completions", [&](const httplib::Request &, httplib::Response & res) {
        reached = true;
        res.set_content(R"({"ok":true})", "application/json");
    });

    const int port = srv.bind_to_any_port("127.0.0.1");
    chk("bound a port", port > 0);
    if (port <= 0) { std::printf("%d failure(s)\n", failures); return 1; }

    std::thread th([&] { srv.listen_after_bind(); });
    srv.wait_until_ready();   // void in v0.47.0; the POST below is the real readiness check
    chk("server reports running", srv.is_running());

    {
        httplib::Client cli("127.0.0.1", port);
        cli.set_read_timeout(10, 0);
        cli.set_write_timeout(10, 0);

        // 771 B: the shape the issue measured as working on the real server.
        const std::string short_body(771, 'x');
        auto r1 = cli.Post("/v1/completions", httplib::Headers{}, short_body,
                           "application/x-www-form-urlencoded");
        chk("a short form-labelled body reaches the handler", r1 && r1->status == 200);

        // ~94 KiB: the size the issue measured as a 413 with a zero-byte reply.
        reached = false;
        const std::string long_body(94 * 1024, 'x');
        auto r2 = cli.Post("/v1/completions", httplib::Headers{}, long_body,
                           "application/x-www-form-urlencoded");
        chk("a 94 KiB form-labelled body reaches the handler", r2 && r2->status == 200);
        chk("... and it is the handler that answered", reached);
    }

    srv.stop();
    th.join();
    std::printf("%s: %d failure(s)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
