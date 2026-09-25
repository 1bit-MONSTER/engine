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
// ThinkSplit (app/think_split.h): the NPU route's reasoning_content split.
#include "think_split.h"

#include <cstdio>
#include <string>
#include <vector>

struct R { std::string c, r; };
R run(bool opened, std::vector<std::string> parts) {
    onebit::ThinkSplit s(opened); R out;
    auto emit=[&](bool reasoning, const std::string& t){ (reasoning?out.r:out.c)+=t; };
    for (auto& p: parts) s.feed(p, emit);
    s.flush(emit); return out;
}
int fails=0;
void check(const char* name, R got, const char* c, const char* r){
    bool ok = got.c==c && got.r==r; fails += !ok;
    std::printf("%s %s: content=[%s] reasoning=[%s]\n", ok?"PASS":"FAIL", name, got.c.c_str(), got.r.c_str());
}
int main(){
    check("tags whole", run(false,{"<think>\nplan</think>\n\nParis"}), "Paris", "plan");
    check("tags split", run(false,{"<th","ink>pl","an</th","ink>","\n\nPar","is"}), "Paris", "plan");
    check("opened by template", run(true,{"plan it\n","</think>","\n\nParis"}), "Paris", "plan it\n");
    check("no think", run(false,{"Par","is"}), "Paris", "");
    check("literal tag after content", run(false,{"use <think> tags"}), "use <think> tags", "");
    check("unfinished think", run(false,{"<think>still going"}), "", "still going");
    return fails;
}


