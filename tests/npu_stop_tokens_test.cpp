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

// npu_stop_tokens_test: the ids that end a generation are a LIST (app/npu_stop_tokens.h, engine#1044).
// No model, no device, no network.
//
// The case that fails on the old code is the first one and it is not hypothetical: the Llama-3.2-1B OGA
// pack declares eos_token_id [128001, 128008, 128009], forward-serve read that key with json_int() (an
// integer or the default), so its stop set was {-1, <|endoftext|>} and 128009 -- the <|eot_id|> the model
// actually emits after an answer -- was served as text with finish_reason "length".
#include "npu_stop_tokens.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using nlohmann::json;
using onebit::npu::eos_token_ids;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

static void check_ids(const std::vector<int>& got, const std::vector<int>& want, const std::string& what) {
    bool ok = got == want;
    std::string g, w;
    for (int v : got) g += std::to_string(v) + " ";
    for (int v : want) w += std::to_string(v) + " ";
    check(ok, what + " (got [" + g + "], want [" + w + "])");
}

static std::string write_cfg(const char* name, const std::string& body) {
    const std::string path = std::string("/tmp/npu_stop_tokens_") + name + ".json";
    std::ofstream f(path);
    f << body;
    return path;
}

int main() {
    // 1. the pack this was found on: three stop ids, and the one the model emits first is in there
    {
        const json j = json::parse(R"({"eos_token_id": [128001, 128008, 128009]})");
        check_ids(eos_token_ids(j), {128001, 128008, 128009}, "Llama-3.2 list is honoured whole");
        const std::vector<int> ids = eos_token_ids(j);
        check(std::find(ids.begin(), ids.end(), 128009) != ids.end(),
              "<|eot_id|> 128009 is a stop id (the token the old code served as text)");
    }
    // 2. the single-integer form every Qwen container writes must keep working unchanged
    check_ids(eos_token_ids(json::parse(R"({"eos_token_id": 151645})")), {151645}, "Qwen single id");
    // 3. no key at all: nobody stops but the "<|endoftext|>" lookup, which is what -1 used to mean
    check_ids(eos_token_ids(json::parse(R"({"hidden_size": 2048})")), {}, "missing key");
    // 4. a malformed entry is dropped, not fatal: the container's own config must not stop a serve loading
    check_ids(eos_token_ids(json::parse(R"({"eos_token_id": [128009, "128001", 2.5]})")), {128009},
              "non-integer entries are ignored");
    // 5. the file path form, including a file that is not there
    const std::string cfg = write_cfg("llama", R"({"eos_token_id": [128001, 128009]})");
    check_ids(onebit::npu::eos_token_ids_from_config(cfg), {128001, 128009}, "config file form");
    check_ids(onebit::npu::eos_token_ids_from_config("/tmp/npu_stop_tokens_does_not_exist.json"), {},
              "unreadable config");
    std::remove(cfg.c_str());

    if (failures == 0) std::printf("all stop-token checks passed\n");
    return failures == 0 ? 0 : 1;
}
