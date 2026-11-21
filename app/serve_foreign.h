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
//
// Who a `--port 0` backend's readiness lines belong to (engine#3130 part 2, engine#3171).
//
// serve launches a child with `--port 0` and reads the port that child reports. Two kinds of line arrive on one pipe:
//   * ours -- `1bit serve: … ready on http://…`, which parse_backend_ready_port() reads;
//   * someone else's -- llama.cpp's `llama_server: listening on http://…`, which is what a
//     `--llama-server llama-server` mistake looks like and must be refused at once rather than waited out.
// A nested `1bit serve` child tears the distinction: it tees ITS child's lines into this pipe unchanged, so a
// grandchild's llama.cpp line arrives here even though the child itself is ours. A child that has printed an
// `1bit ` line is therefore never judged by a later foreign one -- that is the whole rule, and it is a pure function so
// the case can be pinned without a process (tests/serve_foreign_ready_test.cpp).
#pragma once

#include <string>

namespace onebit::serve {

// The two shapes, from one line of the child's stderr.
inline bool is_own_line(const std::string& line) { return line.compare(0, 5, "1bit ") == 0; }

inline bool looks_like_foreign_ready(const std::string& line) {
    if (is_own_line(line)) return false;  // our own line; parse_backend_ready_port handles it
    return line.find("listening on") != std::string::npos && line.find("http://") != std::string::npos;
}

// Whether a foreign line seen on this child's pipe refuses the launch: only while the child has never spoken our format.
// A genuine `--llama-server llama-server` child never does, so the mistake is still caught; a nested `1bit serve` does,
// a moment after its grandchild's line.
inline bool foreign_refuses(bool foreign_line_seen, bool own_line_seen) {
    return foreign_line_seen && !own_line_seen;
}

}  // namespace onebit::serve
