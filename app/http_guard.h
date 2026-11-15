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

// Which requests the engine's HTTP servers (serve, forward-serve, the NPU fast
// lane) answer. None of them authenticates, so a web page in the user's browser
// must not be able to drive them:
//
//  - bound to loopback, the Host header must name a loopback host (localhost,
//    127.x.x.x, ::1). A page whose own hostname has been rebound to 127.0.0.1
//    sends its own name, so it cannot read the server as same-origin.
//  - a request that is not GET/HEAD/OPTIONS and carries an Origin header must
//    come from a loopback origin or from the server's own origin (Origin equal
//    to Host). Browsers send Origin on cross-site POSTs, including the "simple"
//    text/plain ones that need no preflight; API clients send none.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
// install_request_guard() is defined when httplib.h was included first.

namespace onebit {

inline std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

// The host part of a Host header or an Origin's authority: "[::1]:8000" -> "::1",
// "LocalHost:80" -> "localhost".
inline std::string host_part(const std::string& authority) {
    std::string a = lower(authority);
    if (!a.empty() && a.front() == '[') {
        const size_t close = a.find(']');
        return close == std::string::npos ? a : a.substr(1, close - 1);
    }
    if (std::count(a.begin(), a.end(), ':') == 1) a = a.substr(0, a.find(':'));  // drop the port
    return a;
}

inline bool loopback_host(const std::string& host) {
    if (host == "localhost" || host == "::1" || host == "0:0:0:0:0:0:0:1") return true;
    if (host.size() > 10 && host.compare(host.size() - 10, 10, ".localhost") == 0) return true;  // RFC 6761
    // 127.0.0.0/8, dotted quad only
    if (host.rfind("127.", 0) != 0) return false;
    int dots = 0;
    for (char c : host) {
        if (c == '.') ++dots;
        else if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return dots == 3;
}

// Why the request is refused, or "" to serve it. `bound` is the address the server listens on.
inline std::string refuse_request(const std::string& bound, const std::string& method, const std::string& host_header,
                                  const std::string& origin) {
    if (loopback_host(host_part(bound)) && !host_header.empty() && !loopback_host(host_part(host_header)))
        return "the Host header does not name this server";
    if (origin.empty() || method == "GET" || method == "HEAD" || method == "OPTIONS") return "";
    const size_t scheme = origin.find("://");
    if (scheme == std::string::npos) return "cross-site requests are not allowed";  // "null", opaque origins
    const std::string authority = origin.substr(scheme + 3);
    if (loopback_host(host_part(authority))) return "";
    if (!host_header.empty() && lower(authority) == lower(host_header)) return "";  // the server's own origin
    return "cross-site requests are not allowed";
}

#ifdef CPPHTTPLIB_HTTPLIB_H
// engine#2482: an error httplib raises before a route runs (a 413 on the body, a bad request line) gave the client an
// empty body and wrote nothing to the log. Every such error now leaves one stderr line, and an error response that has
// no body gets a JSON one, so the client and the log name the same cause.
inline std::string request_error_line(const std::string& error, const httplib::Request* q) {
    std::string line = "1bit serve: request refused before routing: " + error;
    if (q) {
        line += " (" + q->method + " " + q->path + ", Content-Type '" + q->get_header_value("Content-Type") +
                "', body " + std::to_string(q->body.size()) + " B)";
    }
    return line;
}

inline void fill_empty_error_body(httplib::Response& r) {
    if (r.status < 400 || !r.body.empty()) return;
    std::string message = "HTTP " + std::to_string(r.status);
    if (r.status == 413) message += ": the request body is larger than this server accepts";
    r.set_content("{\"error\":{\"message\":\"" + message + "\",\"code\":" + std::to_string(r.status) + "}}",
                  "application/json");
}

// Answers a refused request with 403 before it reaches a route.
inline void install_request_guard(httplib::Server& srv, const std::string& bound) {
    srv.set_error_logger([](const httplib::Error& e, const httplib::Request* q) {
        std::fprintf(stderr, "%s\n", request_error_line(httplib::to_string(e), q).c_str());
    });
    srv.set_error_handler([](const httplib::Request&, httplib::Response& r) {
        if (!r.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
        fill_empty_error_body(r);
        return httplib::Server::HandlerResponse::Handled;
    });
    srv.set_pre_routing_handler([bound](const httplib::Request& q, httplib::Response& r) {
        const std::string why = refuse_request(bound, q.method, q.get_header_value("Host"), q.get_header_value("Origin"));
        if (why.empty()) return httplib::Server::HandlerResponse::Unhandled;
        r.status = 403;
        r.set_content("{\"error\":{\"message\":\"" + why + "\"}}", "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
}
#endif

}  // namespace onebit
