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

// gguf_architecture and gguf_string_meta (app/gguf_meta.h) on GGUF headers written here: the key found
// after scalars and arrays, a string value longer than one read (the template case), and "" for files
// that are not GGUFs, carry no such key, or store something else under it.
#include "gguf_meta.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Gguf {
    std::string b;
    uint64_t n_kv = 0;
    template <class T> void raw(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void str(const std::string& s) { raw<uint64_t>(s.size()); b += s; }
    void kv_u32(const std::string& k, uint32_t v) { str(k); raw<uint32_t>(4); raw(v); ++n_kv; }
    void kv_str(const std::string& k, const std::string& v) { str(k); raw<uint32_t>(8); str(v); ++n_kv; }
    void kv_strs(const std::string& k, const std::vector<std::string>& v) {
        str(k); raw<uint32_t>(9); raw<uint32_t>(8); raw<uint64_t>(v.size());
        for (const auto& s : v) str(s);
        ++n_kv;
    }
    void kv_f32s(const std::string& k, size_t n) {
        str(k); raw<uint32_t>(9); raw<uint32_t>(6); raw<uint64_t>(n);
        for (size_t i = 0; i < n; ++i) raw<float>(0.5f * i);
        ++n_kv;
    }
    std::string write(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        const uint32_t magic = 0x46554747u, version = 3;
        const uint64_t n_tensors = 0;
        f.write(reinterpret_cast<const char*>(&magic), 4);
        f.write(reinterpret_cast<const char*>(&version), 4);
        f.write(reinterpret_cast<const char*>(&n_tensors), 8);
        f.write(reinterpret_cast<const char*>(&n_kv), 8);
        f << b;
        return path;
    }
};

int failures = 0;
void expect(const std::string& what, const std::string& got, const std::string& want) {
    const bool ok = got == want;
    std::printf("%s %s: \"%s\"\n", ok ? "ok  " : "FAIL", what.c_str(), got.c_str());
    failures += !ok;
}

}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "onebit_gguf_meta_test";
    std::filesystem::create_directories(dir);

    Gguf first;
    first.kv_str("general.architecture", "qwen3");
    expect("architecture first", onebit::gguf_architecture(first.write(dir / "first.gguf")), "qwen3");

    Gguf later;
    later.kv_u32("general.quantization_version", 2);
    later.kv_strs("tokenizer.ggml.tokens", {"<bos>", "\n", "hello"});
    later.kv_f32s("tokenizer.ggml.scores", 3);
    later.kv_str("general.name", "ZAYA1 8B");
    later.kv_str("general.architecture", "zaya");
    expect("architecture after arrays", onebit::gguf_architecture(later.write(dir / "later.gguf")), "zaya");

    Gguf none;
    none.kv_u32("general.quantization_version", 2);
    expect("no architecture", onebit::gguf_architecture(none.write(dir / "none.gguf")), "");

    // the template: a long string after arrays and scalars, which is what a real GGUF looks like
    Gguf tpl;
    tpl.kv_u32("general.quantization_version", 2);
    tpl.kv_strs("tokenizer.ggml.tokens", {"<bos>", "\n", "hello"});
    tpl.kv_str("general.architecture", "qwen3");
    tpl.kv_str("tokenizer.chat_template", "{% for m in messages %}{{ m['role'] }}\n{% endfor %}");
    tpl.kv_str("tokenizer.chat_template.tool_use", "{% if tools %}{{ tools }}{% endif %}");
    tpl.kv_u32("qwen3.block_count", 28);
    const std::string tpl_path = tpl.write(dir / "tpl.gguf");
    expect("string value after arrays", onebit::gguf_string_meta(tpl_path, "tokenizer.chat_template"),
           "{% for m in messages %}{{ m['role'] }}\n{% endfor %}");
    expect("first match wins", onebit::gguf_string_meta(tpl_path, "general.architecture"), "qwen3");
    expect("a named template keeps its own key",
           onebit::gguf_string_meta(tpl_path, "tokenizer.chat_template.tool_use"),
           "{% if tools %}{{ tools }}{% endif %}");
    expect("key after the template", onebit::gguf_string_meta(tpl_path, "qwen3.block_count"), "");
    expect("missing key", onebit::gguf_string_meta(tpl_path, "tokenizer.chat_template.nope"), "");
    expect("an empty string value", onebit::gguf_string_meta((dir / "empty.gguf").string(), "x"), "");
    Gguf empty_val;
    empty_val.kv_str("tokenizer.chat_template", "");
    expect("empty value reads as none", onebit::gguf_string_meta(empty_val.write(dir / "emptyval.gguf"),
                                                                "tokenizer.chat_template"), "");
    Gguf wrong_type;
    wrong_type.kv_u32("tokenizer.chat_template", 7);
    expect("an integer under the key", onebit::gguf_string_meta(wrong_type.write(dir / "int.gguf"),
                                                               "tokenizer.chat_template"), "");

    { std::ofstream(dir / "text.gguf") << "not a gguf file at all"; }
    expect("not a GGUF", onebit::gguf_architecture((dir / "text.gguf").string()), "");
    expect("not a GGUF, string key", onebit::gguf_string_meta((dir / "text.gguf").string(), "general.architecture"), "");
    expect("missing file", onebit::gguf_architecture((dir / "missing.gguf").string()), "");

    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
