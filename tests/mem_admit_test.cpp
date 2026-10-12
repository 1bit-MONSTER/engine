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

// serve's memory admission (app/mem_admit.h) and the box readings it uses (app/sysinfo.h), on
// GGUF headers written here (sparse files, so a "100 GiB" model costs nothing) and a fake /proc
// and /sys. The KV and state sizes are the ones HRX0 server logs printed for the same shapes
// (tools/mem-budget's README, 2026-10-12): Qwen3-0.6B 448 MiB at 4096, GLM-4.7-Flash (MLA)
// 105.75 MiB at 2048, ZAYA1-8B 80 MiB at 2048, Qwen3.6-35B-A3B 40 MiB KV + 62.8 MiB state at 2048.
#include "mem_admit.h"
#include "sysinfo.h"

#include <cmath>
#include <unistd.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace onebit;

namespace {

struct Gguf {
    std::string b;
    uint64_t n_kv = 0;
    template <class T> void raw(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void str(const std::string& s) { raw<uint64_t>(s.size()); b += s; }
    Gguf& u32(const std::string& k, uint32_t v) { str(k); raw<uint32_t>(4); raw(v); ++n_kv; return *this; }
    Gguf& s(const std::string& k, const std::string& v) { str(k); raw<uint32_t>(8); str(v); ++n_kv; return *this; }
    Gguf& i32s(const std::string& k, const std::vector<int32_t>& v) {
        str(k); raw<uint32_t>(9); raw<uint32_t>(5); raw<uint64_t>(v.size());
        for (int32_t x : v) raw(x);
        ++n_kv;
        return *this;
    }
    Gguf& vocab(size_t n) {  // a tokenizer-sized string array the reader must get through
        str("tokenizer.ggml.tokens"); raw<uint32_t>(9); raw<uint32_t>(8); raw<uint64_t>(n);
        for (size_t i = 0; i < n; ++i) str("tok" + std::to_string(i));
        ++n_kv;
        return *this;
    }
    std::string write(const std::string& path, uint64_t size = 0) const {
        {
            std::ofstream f(path, std::ios::binary);
            const uint32_t magic = 0x46554747u, version = 3;
            const uint64_t n_tensors = 0;
            f.write(reinterpret_cast<const char*>(&magic), 4);
            f.write(reinterpret_cast<const char*>(&version), 4);
            f.write(reinterpret_cast<const char*>(&n_tensors), 8);
            f.write(reinterpret_cast<const char*>(&n_kv), 8);
            f << b;
        }
        if (size) fs::resize_file(path, size);   // sparse: the "weights"
        return path;
    }
};

int failures = 0;
void expect(const std::string& what, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
void expect_mib(const std::string& what, uint64_t got, double want_mib) {
    const double mib = double(got) / double(admit::kMiB);
    const bool ok = std::fabs(mib - want_mib) < 0.01;
    std::printf("%s %s: %.4f MiB (want %.4f)\n", ok ? "ok  " : "FAIL", what.c_str(), mib, want_mib);
    if (!ok) ++failures;
}
void put(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("mem_admit_test_" + std::to_string(::getpid()));
    fs::create_directories(dir);
    const std::string d = dir.string();

    // Qwen3-0.6B: 28 layers, 8 KV heads, 128 + 128
    const auto qwen3 = Gguf().s("general.architecture", "qwen3").u32("qwen3.block_count", 28)
        .u32("qwen3.context_length", 40960).u32("qwen3.embedding_length", 1024).u32("qwen3.attention.head_count", 16)
        .u32("qwen3.attention.head_count_kv", 8).u32("qwen3.attention.key_length", 128).u32("qwen3.attention.value_length", 128)
        .vocab(151936).write(d + "/qwen3.gguf", 484ull << 20);
    auto m = admit::model_need(qwen3, 4096, 1);
    expect("qwen3 architecture read past a 151936-token vocabulary", m.arch == "qwen3");
    expect_mib("Qwen3-0.6B KV at 4096", m.kv, 448);
    expect("weights = file size", m.weights == (484ull << 20));
    expect_mib("Qwen3-0.6B KV, 4 slots sharing 40960", admit::model_need(qwen3, 40960, 4).kv, 4480);
    expect_mib("q8_0 KV is 34/64 of f16", admit::model_need(qwen3, 4096, 1, 34.0 / 32, 34.0 / 32).kv, 448 * 34.0 / 64);
    m = admit::model_need(qwen3, 0, 1);
    expect("no context given: the model's own", m.n_ctx == 40960 && m.ctx_from_model);

    // GLM-4.7-Flash (deepseek2, MLA): 47 layers, kv_lora_rank 512 + 64 rope dims, no V of its own
    const auto glm = Gguf().s("general.architecture", "deepseek2").u32("deepseek2.block_count", 47)
        .u32("deepseek2.context_length", 202752).u32("deepseek2.embedding_length", 2048).u32("deepseek2.attention.head_count", 20)
        .u32("deepseek2.attention.head_count_kv", 1).u32("deepseek2.attention.key_length", 576)
        .u32("deepseek2.attention.value_length", 512).u32("deepseek2.attention.kv_lora_rank", 512)
        .u32("deepseek2.rope.dimension_count", 64).write(d + "/glm.gguf");
    expect_mib("GLM-4.7-Flash (MLA) KV at 2048", admit::model_need(glm, 2048, 1).kv, 105.75);

    // ZAYA1-8B: 40 attention layers, 2 KV heads, 128 + 128
    const auto zaya = Gguf().s("general.architecture", "zaya").u32("zaya.block_count", 40).u32("zaya.embedding_length", 2048)
        .u32("zaya.attention.head_count", 8).u32("zaya.attention.head_count_kv", 2).u32("zaya.attention.key_length", 128)
        .u32("zaya.attention.value_length", 128).write(d + "/zaya.gguf");
    expect_mib("ZAYA1-8B KV at 2048", admit::model_need(zaya, 2048, 1).kv, 80);

    // Qwen3.6-35B-A3B (qwen35moe): one layer in 4 attends; 30 gated delta-net layers keep state
    const auto q35 = Gguf().s("general.architecture", "qwen35moe").u32("qwen35moe.block_count", 40)
        .u32("qwen35moe.embedding_length", 2048).u32("qwen35moe.attention.head_count", 16)
        .u32("qwen35moe.attention.head_count_kv", 2).u32("qwen35moe.attention.key_length", 256)
        .u32("qwen35moe.attention.value_length", 256).u32("qwen35moe.full_attention_interval", 4)
        .u32("qwen35moe.ssm.conv_kernel", 4).u32("qwen35moe.ssm.state_size", 128).u32("qwen35moe.ssm.group_count", 16)
        .u32("qwen35moe.ssm.inner_size", 4096).write(d + "/q35.gguf");
    m = admit::model_need(q35, 2048, 1);
    expect_mib("Qwen3.6-35B-A3B KV at 2048", m.kv, 40);
    expect_mib("Qwen3.6-35B-A3B recurrent state, one sequence", m.state, 62.8125);

    // per-layer KV heads (a hybrid that marks its recurrent layers with 0)
    const auto mixed = Gguf().s("general.architecture", "lfm2").u32("lfm2.block_count", 4).u32("lfm2.embedding_length", 1024)
        .u32("lfm2.attention.head_count", 8).i32s("lfm2.attention.head_count_kv", {0, 4, 0, 4}).write(d + "/mixed.gguf");
    expect_mib("per-layer head_count_kv: only the layers with heads", admit::model_need(mixed, 1024, 1).kv,
               8.0 * 256 * 2 * 1024 / (1 << 20));  // 8 heads, 128 + 128, f16

    // gpt-oss: sliding window 128 on every other layer (llama.cpp's pattern for the architecture)
    const auto oss = Gguf().s("general.architecture", "gpt-oss").u32("gpt-oss.block_count", 24)
        .u32("gpt-oss.embedding_length", 2880).u32("gpt-oss.attention.head_count", 64).u32("gpt-oss.attention.head_count_kv", 8)
        .u32("gpt-oss.attention.key_length", 64).u32("gpt-oss.attention.value_length", 64)
        .u32("gpt-oss.attention.sliding_window", 128).write(d + "/oss.gguf");
    expect_mib("gpt-oss: 12 full layers at 131072 + 12 windowed at 128 + 512", admit::model_need(oss, 131072, 1).kv,
               (12.0 * 131072 + 12.0 * 640) * 8 * 128 * 2 / (1 << 20));

    // a split model counts every shard
    Gguf().s("general.architecture", "qwen3").write(d + "/big-00001-of-00002.gguf", 3ull << 30);
    Gguf().write(d + "/big-00002-of-00002.gguf", 2ull << 30);
    expect("shards add up", admit::file_bytes(d + "/big-00001-of-00002.gguf") == (5ull << 30));

    // a llama-server command line: the device decides what an unset context means
    admit::Need hrx, cpu, drafted;
    admit::add_llama_server(hrx, {"llama-server", "-m", qwen3, "--device", "HRX0", "-ngl", "99"});
    admit::add_llama_server(cpu, {"llama-server", "-m", qwen3, "-ngl", "0"});
    expect("HRX0, no -c: the model's 40960", hrx.gpu && hrx.n_ctx == 40960 && hrx.ctx_from_model);
    expect("CPU, no -c: --fit's 4096 floor", !cpu.gpu && cpu.n_ctx == 4096);
    admit::add_llama_server(drafted, {"llama-server", "-m", qwen3, "-c", "4096", "-np", "1", "-md", zaya, "--mmproj", qwen3});
    expect("-md and --mmproj count", drafted.parts.size() == 3 &&
                                     drafted.total == (484ull << 20) + 448 * admit::kMiB + 512 * admit::kMiB +
                                                      admit::file_bytes(zaya) + 160 * admit::kMiB + (484ull << 20));

    // the box: a fake /proc and /sys
    const std::string root = d + "/root";
    put(root + "/proc/meminfo", "MemTotal:       131072000 kB\nMemFree:  1000 kB\nMemAvailable:   52428800 kB\n");
    put(root + "/sys/class/drm/card0/device/mem_info_gtt_total", std::to_string(120ull << 30) + "\n");
    put(root + "/sys/class/drm/card0/device/mem_info_gtt_used", std::to_string(110ull << 30) + "\n");
    put(root + "/sys/class/drm/card0-DP-1/status", "connected\n");
    put(root + "/sys/class/hwmon/hwmon3/name", "k10temp\n");
    put(root + "/sys/class/hwmon/hwmon3/temp1_label", "Tctl\n");
    put(root + "/sys/class/hwmon/hwmon3/temp1_input", "71500\n");
    const sysinfo::Memory mem = sysinfo::memory(root);
    expect("meminfo read", mem.has_meminfo && mem.mem_available == 52428800ull * 1024);
    expect("GTT read from card0", mem.has_gtt && mem.card == "card0" && mem.gtt_total - mem.gtt_used == (10ull << 30));
    const auto t = sysinfo::tctl_celsius(root);
    expect("Tctl read", t && std::fabs(*t - 71.5) < 1e-9);
    expect("no box, no reading", !sysinfo::memory(d + "/nowhere").has_meminfo && !sysinfo::tctl_celsius(d + "/nowhere"));

    admit::Need need;
    need.gpu = true;
    need.total = 12ull << 30;
    auto v = admit::judge(need, mem);
    expect("12 GiB on HRX with 10 GiB of GTT free: refused", v.checked && !v.fits && v.free == (10ull << 30));
    need.total = 8ull << 30;
    v = admit::judge(need, mem);
    expect("8 GiB: fits, with a program-cache warning", v.fits && !v.warnings.empty());
    need.gpu = false;
    need.total = 12ull << 30;
    v = admit::judge(need, mem);
    expect("12 GiB on the CPU: GTT does not count, MemAvailable 50 GiB does", v.fits && v.free == mem.mem_available);
    need.total = 60ull << 30;
    expect("60 GiB on the CPU: refused", !admit::judge(need, mem).fits);
    expect("a box with no /proc/meminfo: not checked", !admit::judge(need, sysinfo::memory(d + "/nowhere")).checked);

    fs::remove_all(dir);
    if (failures) std::printf("%d FAILED\n", failures);
    else std::printf("PASS\n");
    return failures ? 1 : 0;
}
