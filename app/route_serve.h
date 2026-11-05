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

// `1bit route --serve`: the resident form of `1bit route` (docs/laya-first.md §8 step 1). The
// one-shot command pays the scorer load on every request, which is why the fork cannot ask it per
// request on a cold Lemonade; this holds the scorer and the route policy in memory and answers one
// decision per stdin line.
#pragma once

#include <string>
#include <vector>

namespace onebit {

// Reads newline-delimited JSON requests from stdin and writes one decision per line to stdout.
// `devices` is the candidate list, in the caller's order (see laya/route.h).
int run_route_serve(const std::string& laya_model, const std::vector<std::string>& devices,
                    const std::string& policy_path);

// One decision as the JSON line shared by `1bit route --json` and `--serve`, so the two commands can
// never disagree about field names: {"class","confidence","device","decision_ms"}.
std::string route_json_line(const std::string& cls, float confidence, const std::string& device,
                            double decision_ms);

}  // namespace onebit
