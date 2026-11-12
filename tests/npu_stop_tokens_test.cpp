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

    // 6. the tokenizer's eos_token and the chat route's turn end (engine#1785, FastFlowLM's Qwen3.5-2B container):
    //    config.json names <|endoftext|>, tokenizer_config.json names <|im_end|>, and the model ends its turn with
    //    the second, which was served as text.
    {
        using onebit::npu::eos_token_text;
        using onebit::npu::stop_token_ids;
        const auto id_of = [](const std::string& t) {
            if (t == "<|endoftext|>") return 248044;
            if (t == "<|im_end|>") return 248046;
            if (t == "<end_of_turn>") return 106;
            return -1;
        };
        const json cfg = json::parse(R"({"eos_token_id": 248044})");
        const json tok = json::parse(R"({"eos_token": "<|im_end|>"})");
        check(eos_token_text(tok) == "<|im_end|>", "eos_token as a plain string");
        check(eos_token_text(json::parse(R"({"eos_token": {"content": "<|im_end|>", "lstrip": false}})")) == "<|im_end|>",
              "eos_token as an AddedToken object");
        check(eos_token_text(json::parse(R"({"eos_token": null})")).empty() && eos_token_text(json::parse("{}")).empty() &&
                  eos_token_text(json::parse(R"({"eos_token": 7})")).empty() && eos_token_text(json(nullptr)).empty(),
              "a missing, null or non-text eos_token is no token");
        check_ids(stop_token_ids(cfg, tok, "", id_of), {248044, 248046},
                  "the tokenizer's eos_token is a stop id beside config.json's (the token that was served as text)");
        check_ids(stop_token_ids(cfg, json(nullptr), "", id_of), {248044}, "with no tokenizer_config the list is config.json's, unchanged");
        check_ids(stop_token_ids(cfg, json(nullptr), "<|im_end|>", id_of), {248044, 248046},
                  "a route that closes its turns with <|im_end|> stops on it even when neither file names it");
        check_ids(stop_token_ids(cfg, tok, "<|im_end|>", id_of), {248044, 248046}, "named twice, listed once");
        check_ids(stop_token_ids(json::parse(R"({"eos_token_id": [248046, 248044]})"), tok, "<|im_end|>", id_of), {248046, 248044},
                  "config.json's order is kept and nothing is repeated");
        check_ids(stop_token_ids(cfg, json::parse(R"({"eos_token": "<|eot_id|>"})"), "<|nope|>", id_of), {248044},
                  "a name the tokenizer does not have adds nothing");
        check_ids(stop_token_ids(json::parse("{}"), json::parse(R"({"eos_token": "<end_of_turn>"})"), "", id_of), {106},
                  "with no eos_token_id at all, the tokenizer's eos_token is the list");
        check_ids(stop_token_ids(json(nullptr), json(nullptr), "", id_of), {}, "nothing readable, nothing added");
        const std::string f = write_cfg("tokcfg", R"({"eos_token": "<|im_end|>"})");
        check(eos_token_text(onebit::npu::read_json_or_null(f)) == "<|im_end|>", "read from a file");
        check(onebit::npu::read_json_or_null("/tmp/npu_stop_tokens_does_not_exist.json").is_null(), "a missing file is null");
        const std::string bad = write_cfg("badjson", "{ not json");
        check(onebit::npu::read_json_or_null(bad).is_null(), "a malformed file is null");
        std::remove(f.c_str());
        std::remove(bad.c_str());
    }

    // 7. the turn end read off the template (engine#1786, FastFlowLM's Phi4-mini-Instruct-NPU2): neither file names
    //    <|end|>, the template closes every turn with it, and the answer ran on past it.
    {
        using onebit::npu::turn_end_marker;
        check(turn_end_marker("<|end|>") == "<|end|>", "Phi: the marker right after the assistant's text");
        check(turn_end_marker("<|im_end|>\n") == "<|im_end|>", "ChatML: the marker, without the newline after it");
        check(turn_end_marker("<|eot_id|>") == "<|eot_id|>" && turn_end_marker("<end_of_turn>\n") == "<end_of_turn>",
              "Llama 3 and Gemma");
        check(turn_end_marker("\n<|im_end|>\n<|im_start|>user") == "<|im_end|>", "whitespace before it is skipped, what follows is ignored");
        check(turn_end_marker("").empty() && turn_end_marker("\n").empty() && turn_end_marker("  \n ").empty(),
              "nothing, or only whitespace, after the turn: no marker");
        check(turn_end_marker("Human: next").empty() && turn_end_marker("x<|end|>").empty(),
              "plain text after the turn is not a stop, even with a marker later in it");
        check(turn_end_marker("<").empty() && turn_end_marker("<>").empty() && turn_end_marker("<|end").empty() &&
                  turn_end_marker("< not a token >").empty(),
              "a lone bracket, an empty pair, an unclosed one and one with spaces in it are not markers");
        // into the list: Phi's two files name only <|endoftext|>
        const auto id_of = [](const std::string& t) {
            if (t == "<|endoftext|>") return 199999;
            if (t == "<|end|>") return 200020;
            return -1;
        };
        const json cfg = json::parse(R"({"eos_token_id": 199999})");
        const json tok = json::parse(R"({"eos_token": "<|endoftext|>"})");
        check_ids(onebit::npu::stop_token_ids(cfg, tok, "", id_of), {199999}, "from the two files alone the serve stops on <|endoftext|> only (the fault)");
        check_ids(onebit::npu::stop_token_ids(cfg, tok, turn_end_marker("<|end|>"), id_of), {199999, 200020},
                  "with the template's turn end it also stops on <|end|>");
        check_ids(onebit::npu::stop_token_ids(cfg, tok, turn_end_marker("<|eot|>"), id_of), {199999},
                  "a marker the tokenizer does not have as one token adds nothing");
    }

    // engine#1786: the BOS a container's tokenizer_config.json states, for a config.json that states none
    // (FastFlowLM's Medgemma-4B-NPU2: no bos_token_id, no add_bos_token in config.json).
    {
        using onebit::npu::add_bos_token_flag;
        using onebit::npu::special_token_text;
        const json med = json::parse(R"({"bos_token": "<bos>", "add_bos_token": true, "eos_token": "<eos>"})");
        check(special_token_text(med, "bos_token") == "<bos>" && add_bos_token_flag(med) == 1,
              "Medgemma's tokenizer_config.json: BOS <bos>, added");
        check(special_token_text(json::parse(R"({"bos_token": {"content": "<s>", "lstrip": false}})"), "bos_token") == "<s>",
              "bos_token in the object spelling");
        check(add_bos_token_flag(json::parse(R"({"add_bos_token": false})")) == 0 &&
                  add_bos_token_flag(json::parse(R"({"add_bos_token": 1})")) == 1,
              "add_bos_token false and 1");
        check(add_bos_token_flag(json::parse("{}")) == -1 && add_bos_token_flag(json::parse(R"({"add_bos_token": null})")) == -1 &&
                  add_bos_token_flag(json(nullptr)) == -1 && special_token_text(json::parse(R"({"bos_token": null})"), "bos_token").empty(),
              "an absent or null key says nothing");
        check(onebit::npu::eos_token_text(med) == "<eos>", "eos_token_text is unchanged by the shared reader");
    }

    if (failures == 0) std::printf("all stop-token checks passed\n");
    return failures == 0 ? 0 : 1;
}
