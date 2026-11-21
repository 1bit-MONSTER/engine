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
// engine#3171: a nested `1bit serve` child must not be refused for its grandchild's llama.cpp line. The rule is pure, so
// it is pinned here rather than by launching processes: a `/nonexistent` binary and a real model would be needed to reach
// the `--laya-models` rail where the refusal actually fired.
#include "serve_foreign.h"

#include <cstdio>

using onebit::serve::foreign_refuses;
using onebit::serve::is_own_line;
using onebit::serve::looks_like_foreign_ready;

static int failures = 0;

static void check(const char* what, bool got, bool want) {
    if (got == want) {
        std::printf("ok   %s\n", what);
        return;
    }
    std::printf("FAIL %s (got %d, want %d)\n", what, int(got), int(want));
    ++failures;
}

int main() {
    const std::string ours   = "1bit serve: Qwen3-0.6B on cpu (/path/inner.sh)";
    const std::string ready  = "1bit serve: ready on http://127.0.0.1:48387";
    const std::string foreign = "llama_server: listening on http://127.0.0.1:60679";
    const std::string noise  = "[INFO] warming up";

    check("our memory line is ours", is_own_line(ours), true);
    check("our ready line is ours", is_own_line(ready), true);
    check("llama.cpp's line is not ours", is_own_line(foreign), false);
    check("llama.cpp's line looks foreign", looks_like_foreign_ready(foreign), true);
    check("our ready line is never foreign", looks_like_foreign_ready(ready), false);
    check("an unrelated line is not foreign", looks_like_foreign_ready(noise), false);

    // the bug: the grandchild's line alone refused a nested child ...
    check("a foreign line with no own line refuses the launch", foreign_refuses(true, false), true);
    // ... and the fix: a child that has spoken our format is not judged by a later foreign line
    check("a foreign line after an own line does NOT refuse (nested 1bit serve)", foreign_refuses(true, true), false);
    check("nothing seen yet never refuses", foreign_refuses(false, false), false);

    if (failures == 0) std::printf("serve_foreign_ready: ok\n");
    return failures == 0 ? 0 : 1;
}
