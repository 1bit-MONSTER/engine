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
// A GGUF's `tokenizer.chat_template` is a Jinja2 program that runs on every request, so it can
// rewrite the system context with no weight change at all (Pillar Security's template-backdoor
// work). serve therefore names the template's sha256 once per load, and can compare it with the
// one pinned for that model (config/template-pins.json, docs/serve.md "Chat template").
#pragma once

#include "gguf_meta.h"
#include "hrx_runtime.h"  // sha256_hex
#include "template_pins_default.h"  // config/template-pins.json, compiled in by CMakeLists.txt

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

namespace onebit {

// sha256 of the file's tokenizer.chat_template, "" when it carries none (also "" for a file that is
// not a GGUF, a non-GGUF model directory, or a 1BP package). The template is read from the metadata
// header only; the tensors are never touched.
inline std::string gguf_chat_template_sha(const std::string& path) {
    const std::string tpl = gguf_string_meta(path, "tokenizer.chat_template");
    if (tpl.empty()) return "";
    return sha256_hex(reinterpret_cast<const uint8_t*>(tpl.data()), tpl.size());
}

// A model directory (a Q4NX / NPU container) has no GGUF at all: its template is the "chat_template"
// string in the container's tokenizer_config.json, or chat_template.jinja beside it.
inline std::string container_chat_template_sha(const std::string& dir) {
    auto read = [](const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        return in ? std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()) : std::string();
    };
    std::string tpl;
    const std::string cfg = read(dir + "/tokenizer_config.json");
    if (!cfg.empty()) {
        try {
            const nlohmann::json j = nlohmann::json::parse(cfg);
            if (j.contains("chat_template") && j["chat_template"].is_string())
                tpl = j["chat_template"].get<std::string>();
        } catch (const std::exception&) {
            // a container whose tokenizer_config.json does not parse has no readable template
        }
    }
    if (tpl.empty()) tpl = read(dir + "/chat_template.jinja");
    if (tpl.empty()) return "";
    return sha256_hex(reinterpret_cast<const uint8_t*>(tpl.data()), tpl.size());
}

// sha256 of the template a model would run: a GGUF's metadata key, or a model directory's
// tokenizer_config.json / chat_template.jinja. "" when the model carries none (or is neither, e.g. a
// 1BP package).
inline std::string template_sha_of(const std::string& model) {
    std::error_code ec;
    if (std::filesystem::is_directory(model, ec)) return container_chat_template_sha(model);
    if (std::filesystem::path(model).extension() == ".gguf") return gguf_chat_template_sha(model);
    return "";
}

// The template pins: model key -> expected template sha256, or "none" for a model that must carry no
// template. Keys are lower-cased and matched against the serve's model id (--alias, i.e. Lemonade's
// catalog entry) and then the file's (or directory's) own name, so one row can serve either spelling.
struct TemplatePins {
    std::map<std::string, std::string> by_key;

    bool parse(const std::string& text, std::string& err) {
        by_key.clear();
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(text);
        } catch (const std::exception& e) {
            err = std::string("not JSON: ") + e.what();
            return false;
        }
        if (!j.is_object() || !j.contains("pins") || !j["pins"].is_object()) {
            err = "expected an object with a \"pins\" object";
            return false;
        }
        for (const auto& [key, v] : j["pins"].items()) {
            std::string sha, source;
            if (v.is_string()) {
                sha = v.get<std::string>();
            } else if (v.is_object() && v.contains("sha256") && v["sha256"].is_string()) {
                sha = v["sha256"].get<std::string>();
            } else {
                err = "pins." + key + ": expected a sha256 string or an object with \"sha256\"";
                return false;
            }
            std::transform(sha.begin(), sha.end(), sha.begin(), [](unsigned char c) { return std::tolower(c); });
            if (sha != "none" && (sha.size() != 64 || sha.find_first_not_of("0123456789abcdef") != std::string::npos)) {
                err = "pins." + key + ": \"" + sha + "\" is not a sha256 (64 hex), and not \"none\"";
                return false;
            }
            std::string k = key;
            std::transform(k.begin(), k.end(), k.begin(), [](unsigned char c) { return std::tolower(c); });
            if (k.empty()) {
                err = "pins: an empty model key";
                return false;
            }
            by_key[k] = sha;
        }
        return true;
    }

    // The first of `keys` (tried in order, already lower-cased) that carries a pin, or nullptr.
    const std::string* find(const std::vector<std::string>& keys) const {
        for (const auto& k : keys) {
            auto it = by_key.find(k);
            if (it != by_key.end()) return &it->second;
        }
        return nullptr;
    }
};

inline std::string lowercased(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
    return out;
}

struct TemplateVerdict {
    std::string sha256;  // "" when the file carries no tokenizer.chat_template
    std::string key;     // the pin key that matched, "" when the model has no pin
    std::string pinned;  // what that pin expects, "none" for "no template"
    bool mismatch() const { return !key.empty() && pinned != (sha256.empty() ? "none" : sha256); }
};

// `id_keys` are the names the model may be pinned under, most specific first (serve passes the model
// id from --alias, then the file's name).
inline TemplateVerdict template_verdict(const std::string& model, const std::vector<std::string>& id_keys,
                                        const TemplatePins& pins) {
    TemplateVerdict v;
    v.sha256 = template_sha_of(model);
    std::vector<std::string> keys;
    for (const auto& k : id_keys) keys.push_back(lowercased(k));
    if (const std::string* p = pins.find(keys)) v.pinned = *p;
    // which key matched, for the log line
    for (const auto& k : keys) {
        auto it = pins.by_key.find(k);
        if (it != pins.by_key.end()) {
            v.key = k;
            break;
        }
    }
    return v;
}

}  // namespace onebit
