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
    std::printf("%s\n", failures ? "FAILED" : "http_guard: ok");
    return failures ? 1 : 0;
}
