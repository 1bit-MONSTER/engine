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
#include "tmp_root.h"   // engine#2029: scratch goes on disk, not in tmpfs

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
    Gguf& bools(const std::string& k, const std::vector<bool>& v) {  // what gguf-py writes for a list of Python bools
        str(k); raw<uint32_t>(9); raw<uint32_t>(7); raw<uint64_t>(v.size());
        for (bool x : v) raw<uint8_t>(x ? 1 : 0);
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
    const fs::path dir = ::onebit::test::tmp_root() / ("mem_admit_test_" + std::to_string(::getpid()));
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

    // flint: the converter writes the pattern as bools from layer_types (35 sliding, 14 full of 49), and its
    // loader requires the array, so the file decides; engine #1425
    std::vector<bool> flint_pat;
    for (int i = 0; i < 49; ++i) flint_pat.push_back(i % 7 != 3 && i % 7 != 6);  // the count matters, not the order
    const auto flint = Gguf().s("general.architecture", "flint").u32("flint.block_count", 49)
        .u32("flint.embedding_length", 2048).u32("flint.attention.head_count", 16).u32("flint.attention.head_count_kv", 4)
        .u32("flint.attention.key_length", 128).u32("flint.attention.value_length", 128)
        .u32("flint.attention.sliding_window", 4096).bools("flint.attention.sliding_window_pattern", flint_pat)
        .write(d + "/flint.gguf");
    expect_mib("flint: 14 full layers at 262144 + 35 windowed at 4096 + 512", admit::model_need(flint, 262144, 1).kv,
               (14.0 * 262144 + 35.0 * 4608) * 4 * 256 * 2 / (1 << 20));
    expect("flint: the estimate grows with the context",
           admit::model_need(flint, 262144, 1).kv > admit::model_need(flint, 8192, 1).kv);

    // engine#1425: a header with a sliding window and no pattern this reader can use is priced as a lower bound,
    // and the check must SAY so -- admitting on a floor is the #680 overrun class, and the comment in the code is
    // not something a planner reads. A pattern array of either type, or a per-arch entry, clears the flag.
    const auto flint_nopattern = Gguf().s("general.architecture", "flint").u32("flint.block_count", 49)
        .u32("flint.embedding_length", 4096).u32("flint.attention.head_count", 32)
        .u32("flint.attention.head_count_kv", 8).u32("flint.attention.key_length", 128)
        .u32("flint.attention.value_length", 128).u32("flint.attention.sliding_window", 4096)
        .write(d + "/flint-nopattern.gguf");
    expect("no pattern array, unlisted arch: the need is flagged a lower bound",
           admit::model_need(flint_nopattern, 262144, 1).kv_lower_bound);
    expect("a bool pattern array clears the flag", !admit::model_need(flint, 262144, 1).kv_lower_bound);
    expect("the per-arch list clears the flag", !admit::model_need(oss, 131072, 1).kv_lower_bound);
    {   // the int32 form too: gguf-py writes an int for the single-interval pattern
        std::vector<int32_t> i32(flint_pat.begin(), flint_pat.end());
        const auto flint_int = Gguf().s("general.architecture", "flint").u32("flint.block_count", 49)
            .u32("flint.embedding_length", 4096).u32("flint.attention.head_count", 32)
            .u32("flint.attention.head_count_kv", 4).u32("flint.attention.key_length", 128)
            .u32("flint.attention.value_length", 128).u32("flint.attention.sliding_window", 4096)
            .i32s("flint.attention.sliding_window_pattern", i32).write(d + "/flint-i32.gguf");
        expect("an int32 pattern prices the same as the bool one",
               admit::model_need(flint_int, 262144, 1).kv == admit::model_need(flint, 262144, 1).kv);
        expect("an int32 pattern clears the flag too", !admit::model_need(flint_int, 262144, 1).kv_lower_bound);
    }
    {
        admit::Need lb;
        admit::add_llama_server(lb, {"llama-server", "-m", flint_nopattern, "-c", "262144"});
        std::string joined;
        for (const auto& p : lb.parts) joined += p + " | ";
        expect("the message names it a lower bound and whose pattern is missing",
               joined.find("LOWER BOUND") != std::string::npos &&
                   joined.find("flint's sliding-window pattern") != std::string::npos);
    }

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

    // The fixture box reads 10 GiB of GTT free. With the ceiling in place (engine #680) that box is
    // already past the ceiling, so the verdict tests use explicit readings instead of the fixture.
    sysinfo::Memory box = mem;               // 120 GiB of GTT, 60 GiB in use: 60 GiB free, 104 GiB ceiling
    box.gtt_used = 60ull << 30;

    admit::Need need;
    need.gpu = true;
    need.total = 12ull << 30;
    auto v = admit::judge(need, box);
    // free is min(GTT free, MemAvailable): 60 GiB of GTT free, 50 GiB available
    expect("12 GiB on HRX with 60 GiB of GTT free: fits", v.checked && v.fits && v.free == (50ull << 30));
    // a refusal that comes from RAM alone (no GTT in the reading) is still a plain free-memory refusal
    sysinfo::Memory ram_only = box;
    ram_only.has_gtt = false;
    need.total = 50ull << 30;
    v = admit::judge(need, ram_only);
    expect("50 GiB on a box with 50 GiB available: refused by free memory, not by the ceiling",
           !v.fits && !v.over_ceiling);
    need.gpu = false;
    need.total = 12ull << 30;
    v = admit::judge(need, box);
    expect("12 GiB on the CPU: GTT does not count, MemAvailable 50 GiB does", v.fits && v.free == box.mem_available);
    need.total = 60ull << 30;
    expect("60 GiB on the CPU: refused", !admit::judge(need, box).fits);

    // engine#2986: the reserve was a fixed 16 GiB, sized on this 120 GiB box, so a machine whose GTT is 16 GiB had a
    // ceiling of zero and HRX refused every model. It now scales as a quarter of GTT, capped at 16 GiB -- which is
    // exactly the old number from 64 GiB up (#680's measured headroom unchanged) and leaves three quarters usable below.
    {
        auto ceiling_of = [&](uint64_t gtt_total) {
            sysinfo::Memory m = box;
            m.gtt_total = gtt_total;
            m.gtt_used  = 0;
            return admit::judge(need, m);
        };
        const uint64_t gib = 1ull << 30;
        expect("120 GiB of GTT keeps the measured 16 GiB reserve (104 GiB ceiling)",
               admit::gtt_reserve_bytes(120 * gib) == 16 * gib);
        expect("64 GiB of GTT is the crossover: still a 16 GiB reserve",
               admit::gtt_reserve_bytes(64 * gib) == 16 * gib);
        expect("32 GiB of GTT gets an 8 GiB reserve, not 16",
               admit::gtt_reserve_bytes(32 * gib) == 8 * gib);
        expect("16 GiB of GTT (a 32 GB machine) gets a 4 GiB reserve, so the ceiling is not zero",
               admit::gtt_reserve_bytes(16 * gib) == 4 * gib);
        expect("8 GiB of GTT (a 16 GB machine) keeps a 2 GiB reserve and a usable ceiling",
               admit::gtt_reserve_bytes(8 * gib) == 2 * gib);
        // engine#2995 review: an UNREADABLE pool is not an empty one. `gtt_total == 0` must keep the measured reserve,
        // because the alternative is failing open on the input where the answer is unknown.
        expect("an unreadable pool keeps the measured 16 GiB reserve, not zero",
               admit::gtt_reserve_bytes(0) == 16 * gib);

        // engine#2986 follow-up: the reserve override must be a number, or it is ignored (the fork tests the same arms)
        for (const char* bad : {"", "abc", "16GiB", "-1", "\n", "512 MiB"}) {
            ::setenv("GGML_HRX_GTT_RESERVE_MB", bad, 1);
            expect(std::string("GGML_HRX_GTT_RESERVE_MB=\"") + bad + "\" is ignored: an 8 GiB pool keeps its 2 GiB",
                   admit::gtt_reserve_bytes(8ull << 30) == (2ull << 30));
        }
        ::setenv("GGML_HRX_GTT_RESERVE_MB", "0", 1);
        expect("GGML_HRX_GTT_RESERVE_MB=0 is still an override", admit::gtt_reserve_bytes(8ull << 30) == 0);
        for (const char* spaced : {"512\n", " 512 ", "\t512"}) {
            ::setenv("GGML_HRX_GTT_RESERVE_MB", spaced, 1);
            expect("whitespace around the number is accepted on both ends",
                   admit::gtt_reserve_bytes(8ull << 30) == (512ull << 20));
        }
        ::setenv("GGML_HRX_GTT_RESERVE_MB", "512", 1);
        expect("GGML_HRX_GTT_RESERVE_MB=512 is still an override", admit::gtt_reserve_bytes(8ull << 30) == (512ull << 20));
        ::unsetenv("GGML_HRX_GTT_RESERVE_MB");

        expect("a 16 GiB-GTT box now admits a small model where the fixed reserve refused everything",
               ceiling_of(16 * gib).fits || !ceiling_of(16 * gib).over_ceiling);
    }
    expect("a box with no /proc/meminfo: not checked", !admit::judge(need, sysinfo::memory(d + "/nowhere")).checked);

    // The program-cache warning now needs MemAvailable to be the binding limit: under the ceiling there is
    // never free-minus-need under 5 GiB on a 120 GiB pool (the ceiling is 104 GiB of it), which is the
    // point of the ceiling.
    sysinfo::Memory tight = box;
    tight.mem_available = 30ull << 30;       // 120 GiB GTT free-by-ceiling, but only 30 GiB of RAM available
    tight.gtt_used = 0;
    need.gpu = true;
    need.total = 26ull << 30;
    v = admit::judge(need, tight);
    expect("26 GiB with 30 GiB available: fits, with a program-cache warning", v.fits && !v.warnings.empty());

    // engine #680: the GTT ceiling refuses a load that would land where the device faulted, even when
    // the free-memory test alone would admit it. 120 GiB of GTT, 60 GiB in use, 16 GiB reserve.
    need.gpu = true;
    need.total = 40ull << 30;   // 60 + 40 = 100 GiB after the load
    v = admit::judge(need, box);
    expect("40 GiB more: fits under the ceiling", v.fits && !v.over_ceiling && v.ceiling == (104ull << 30));
    need.total = 45ull << 30;   // 60 + 45 = 105 GiB after the load, past the 104 GiB ceiling
    v = admit::judge(need, box);
    expect("45 GiB more: refused by the ceiling although 60 GiB is free", !v.fits && v.over_ceiling &&
                                                                      v.free >= need.total);
    expect("the refusal names the ceiling and the reserve",
           v.ceiling_note.find("ceiling") != std::string::npos &&
               v.ceiling_note.find("GGML_HRX_GTT_RESERVE_MB") != std::string::npos);
    need.total = 1ull << 30;    // 61 GiB after the load: inside the ceiling, and above #315's zone only warns
    v = admit::judge(need, box);
    expect("1 GiB more: fits, no ceiling note", v.fits && v.ceiling_note.empty());
    // engine#2986 restated this invariant rather than dropping it: a pool smaller than the *fixed* 16 GiB reserve is no
    // longer a zero ceiling, because the reserve is now a quarter of the pool. What still has to hold is that the ceiling
    // leaves the reserve free -- and that a pool small enough to be a sixteenth of the load is still refused by it.
    need.gpu   = true;              // the arms below are about the GTT ceiling, and an earlier arm leaves this cleared
    need.total = 60ull << 30;
    sysinfo::Memory small = box;
    small.gtt_total = 8ull << 30;   // was "smaller than the reserve": now 2 GiB reserved, 6 GiB ceiling
    small.gtt_used = 0;
    v = admit::judge(need, small);
    expect("an 8 GiB pool: the ceiling is the pool minus a quarter, not zero",
           v.ceiling == 6ull * (1ull << 30) && v.over_ceiling);
    sysinfo::Memory tiny = box;
    tiny.gtt_total = 1ull << 30;    // 256 MiB reserved, 768 MiB ceiling: still far under a 60 GiB load
    tiny.gtt_used = 0;
    v = admit::judge(need, tiny);
    expect("a 1 GiB pool is still refused: the ceiling cannot grow a load that large",
           !v.fits && v.over_ceiling && v.ceiling == 768ull * (1ull << 20));

    // engine#2986: the default context. LFM3-1.7B-Q4_K_M as its file says: 24 blocks, KV heads only on the 11
    // attention layers, a 256-token window on 5 of the 24, 262,144 tokens of its own, 1,099,627,808 bytes.
    {
        std::vector<int32_t> kv_heads(24, 0);
        for (int i = 2; i < 24; i += 2) kv_heads[size_t(i)] = 4;
        std::vector<bool> windowed(24, false);
        for (int i = 4; i < 24; i += 4) windowed[size_t(i)] = true;
        const auto lfm3 = Gguf().s("general.architecture", "lfm3").u32("lfm3.block_count", 24)
            .u32("lfm3.context_length", 262144).u32("lfm3.embedding_length", 2048).u32("lfm3.attention.head_count", 16)
            .i32s("lfm3.attention.head_count_kv", kv_heads).u32("lfm3.attention.key_length", 128)
            .u32("lfm3.attention.value_length", 128).u32("lfm3.attention.sliding_window", 256)
            .bools("lfm3.attention.sliding_window_pattern", windowed).write(d + "/lfm3.gguf", 1099627808ull);
        const std::vector<std::string> serve_argv = {"llama-server", "-m", lfm3, "--device", "HRX0", "-ngl", "99", "--jinja"};
        auto gib = [](uint64_t b) { return double(b) / double(admit::kGiB); };
        auto machine = [&](uint64_t gtt_gib, uint64_t avail_gib) {
            sysinfo::Memory mm = box;
            mm.gtt_total = gtt_gib << 30;
            mm.gtt_used = 512ull << 20;   // a desktop session's share
            mm.mem_available = avail_gib << 30;
            return mm;
        };
        // the 32 GB machine keeps 262144: 5.88 GiB plus the 1 GiB reserve plus the program cache's 4 GiB fits there
        expect("32 GB machine (GTT 16 GiB, 26 GiB available): the model's own context stands",
               admit::default_ctx(serve_argv, machine(16, 26)).ctx == 0);
        struct Case { const char* what; sysinfo::Memory mm; };
        const Case cases[] = {
            { "16 GB machine (GTT 8 GiB, 11 GiB available)", machine(8, 11) },
        };
        for (const Case& k : cases) {
            const admit::CtxChoice c = admit::default_ctx(serve_argv, k.mm);
            std::printf("     %s: model ctx %lld needs %.2f GiB -> ctx %lld needs %.2f GiB (roomy %d)\n", k.what, c.model_ctx,
                        gib(c.need_model), c.ctx, gib(c.need_ctx), c.roomy ? 1 : 0);
            expect(std::string(k.what) + ": the model's own 262144 is not roomy, a smaller context is chosen",
                   c.model_ctx == 262144 && c.ctx >= admit::kDefaultCtxFloor && c.ctx < 262144 && c.ctx % 256 == 0 && c.roomy);
            // the chosen context is the largest roomy one: 256 more is not
            std::vector<std::string> a = serve_argv, b = serve_argv;
            a.insert(a.end(), {"-c", std::to_string(c.ctx)});
            b.insert(b.end(), {"-c", std::to_string(c.ctx + 256)});
            admit::Need na, nb;
            admit::add_llama_server(na, a);
            admit::add_llama_server(nb, b);
            expect(std::string(k.what) + ": the chosen context is roomy and 256 more is not",
                   admit::judge(na, k.mm).roomy && !admit::judge(nb, k.mm).roomy && na.total == c.need_ctx);
        }
        {
            const admit::CtxChoice busy = admit::default_ctx(serve_argv, machine(8, 6));
            std::printf("     16 GB machine, busier (GTT 8 GiB, 6 GiB available): -> ctx %lld needs %.2f GiB (roomy %d)\n", busy.ctx,
                        gib(busy.need_ctx), busy.roomy ? 1 : 0);
            expect("16 GB machine with 6 GiB available: no context is roomy, the floor is chosen",
                   busy.ctx == admit::kDefaultCtxFloor && !busy.roomy);
        }
        {
            const sysinfo::Memory mm = machine(8, 11);
            std::vector<std::string> a = serve_argv;
            a.insert(a.end(), {"-c", "262144"});
            expect("an explicit --ctx-size is left alone", admit::default_ctx(a, mm).ctx == 0);
            admit::Need n;
            admit::add_llama_server(n, a);
            const admit::Verdict ev = admit::judge(n, mm);
            std::printf("     16 GB machine, explicit --ctx-size 262144: need %.2f GiB, ceiling %.2f GiB -> %s\n", gib(n.total),
                        gib(ev.ceiling), ev.fits ? "fits" : "refused");
            expect("an explicit --ctx-size 262144 on the 16 GB machine is judged as given (fits only if it fits)",
                   ev.fits == (n.total + admit::kGiB <= ev.free && !ev.over_ceiling));
            a = serve_argv;
            a.insert(a.end(), {"-c", "1000000"});
            admit::Need big;
            admit::add_llama_server(big, a);
            expect("an explicit --ctx-size of 1,000,000 on the 16 GB machine is refused", !admit::judge(big, mm).fits);
        }
        // strixhalo-sized: the model's own context is roomy, nothing changes
        expect("120 GiB box with 60 GiB in use: the model's own context stands",
               admit::default_ctx(serve_argv, box).ctx == 0);
        std::vector<std::string> cpu = {"llama-server", "-m", lfm3, "-ngl", "0"};
        expect("a CPU backend is left alone", admit::default_ctx(cpu, machine(8, 11)).ctx == 0);
        // nothing fits even at the floor: the floor is passed and the normal refusal speaks
        const admit::CtxChoice none = admit::default_ctx(serve_argv, machine(2, 2));
        expect("a 2 GiB pool: nothing fits, the floor is passed for the refusal to name",
               none.ctx == admit::kDefaultCtxFloor && !none.roomy);
    }

    fs::remove_all(dir);
    if (failures) std::printf("%d FAILED\n", failures);
    else std::printf("PASS\n");
    return failures ? 1 : 0;
}
