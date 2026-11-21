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
// app/serve_backend_ready.h -- what one line of a backend's stderr means for `wait_ready` (engine#3171).
//
// The decision is a pure function of the line and of whether this child has already spoken in `1bit`'s own format, so
// it lives here rather than inside `Child`: the two `serve.cpp` regressions this week were both in this decision, and
// neither had a row that could fail (the refusal path only runs for a `--port 0` child, i.e. `--laya-models`, whose
// e2e rows skip in CI without a checkpoint).
//
// A backend `1bit serve` starts can speak two ways:
//   * our own readiness line, `1bit serve: ready on http://HOST:PORT` (also `1bit forward-serve:`), which is the ONLY
//     line that carries the port a `--port 0` launch asked the child to report; and
//   * llama.cpp's, `<something>: listening on http://HOST:PORT`, when the child is llama-server rather than our own
//     binary -- `--llama-server llama-server` is the mistake `ONEBIT_BACKEND_PORT_GRACE_S` exists for.
//
// The subtlety, and the reason `own_seen` is an argument: a `--laya-models` backend IS a `1bit serve`, and every
// `Child` passes its child's stderr through unchanged, so the outer serve sees its GRANDCHILD's llama.cpp line before
// the nested child prints its own. Refusing on the foreign line alone therefore refuses a working launch (engine#3171,
// measured: the grandchild's `listening on http://…` arrives in the outer `tail` before `1bit serve: ready on …`).
// A genuine `--llama-server llama-server` child never prints a `1bit ` line, so the distinction holds.
#pragma once

#include <cstdlib>
#include <string>

namespace onebit::serve::backend_ready {

// What a backend line means to `wait_ready`.
enum class Verdict {
    Ignored,       // nothing to do with readiness: model log chatter, metrics, a placement report
    OurPort,       // our own readiness line: parse the port out of it
    ForeignReady,  // llama.cpp announcing itself: this child is not `1bit serve`, which a `--port 0` launch cannot use
};

// True when the line is a readiness announcement in a server's own words, i.e. carries a listening address. Matched on
// the two substrings llama.cpp has printed for years rather than on a prefix, because the prefix is the part that
// differs between its builds and platforms.
inline bool carries_listening_address(const std::string& line) {
    return line.find("listening on") != std::string::npos && line.find("http://") != std::string::npos;
}

// `1bit`'s own line: `1bit serve: ready on http://…` / `1bit forward-serve: MODEL ready on http://…`.
inline bool is_own_ready_line(const std::string& line) {
    return line.compare(0, 5, "1bit ") == 0 && line.find(" ready on http://") != std::string::npos;
}

// The decision. `own_seen` says whether this child has already printed a line starting with `1bit ` -- once it has, the
// child is our own binary and a later llama.cpp-format line can only be a grandchild's (engine#3171).
inline Verdict verdict(const std::string& line, bool own_seen) {
    if (is_own_ready_line(line)) return Verdict::OurPort;
    if (line.compare(0, 5, "1bit ") == 0) return Verdict::Ignored;  // our own non-readiness line
    if (!carries_listening_address(line)) return Verdict::Ignored;
    return own_seen ? Verdict::Ignored : Verdict::ForeignReady;
}

}  // namespace onebit::serve::backend_ready
