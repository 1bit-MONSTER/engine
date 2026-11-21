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
// tests/serve_backend_ready_test.cpp -- the four cases of app/serve_backend_ready.h (engine#3171).
//
// Why this row exists at all: the decision it tests only runs for a backend launched with `--port 0`, and the one rail
// that does that (`--laya-models`) has no checkpoint in CI, so no e2e row can reach it. Both of this week's serve.cpp
// regressions were in this decision and neither had a row that could fail. A table test costs nothing and covers the
// nesting that broke.
#include "serve_backend_ready.h"

#include <cstdio>
#include <string>

namespace br = onebit::serve::backend_ready;

static int failed = 0;

static void expect(const char* what, br::Verdict got, br::Verdict want) {
    if (got == want) {
        std::printf("ok   %s\n", what);
        return;
    }
    std::printf("FAIL %s: got %d want %d\n", what, int(got), int(want));
    failed++;
}

int main() {
    // 1. our own readiness line: the port comes from here, and it is not a foreign announcement.
    expect("our own readiness line is the port",
           br::verdict("1bit serve: ready on http://127.0.0.1:8080", false), br::Verdict::OurPort);
    expect("forward-serve's line is the port too",
           br::verdict("1bit forward-serve: MODEL ready on http://127.0.0.1:8080 (backend)", false),
           br::Verdict::OurPort);

    // 2. a stranger's readiness line, before this child has spoken our format: the --llama-server llama-server mistake.
    expect("llama.cpp's line is foreign when we have heard nothing of our own",
           br::verdict("llama_server: listening on http://127.0.0.1:8080", false), br::Verdict::ForeignReady);
    expect("any prefix, same rule",
           br::verdict("srv  update_slots: all slots are idle; listening on http://0.0.0.0:8080", false),
           br::Verdict::ForeignReady);

    // 3. engine#3171: the same line AFTER the child has spoken our format is a GRANDCHILD's, not this child's role.
    expect("a grandchild's llama.cpp line is ignored once the child speaks our format",
           br::verdict("llama_server: listening on http://127.0.0.1:8080", true), br::Verdict::Ignored);

    // 4. everything else is chatter, including our own non-readiness lines (which must also not be mistaken for foreign).
    expect("our memory line is not a readiness line",
           br::verdict("1bit serve: memory: need 9.9 GiB (weights 8.0 + KV/state 1.9)", false), br::Verdict::Ignored);
    expect("a llama.cpp log line without a listening address is chatter",
           br::verdict("main: model loaded", false), br::Verdict::Ignored);
    expect("listening without http:// is not a readiness address",
           br::verdict("something: listening on unix:/tmp/sock", false), br::Verdict::Ignored);
    expect("http:// without a listening phrase is chatter",
           br::verdict("curl: http://example.invalid failed", false), br::Verdict::Ignored);

    if (failed != 0) {
        std::printf("serve_backend_ready_test: %d failure(s)\n", failed);
        return 1;
    }
    std::printf("serve_backend_ready_test: all arms pass\n");
    return 0;
}
