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
// whisper_npu_device.h -- open the NPU for whisper_transcribe's encoder (engine#1783, the encoder route).
//
// Compiled only into whisper_transcribe_npu (ONEBIT_WHISPER_NPU). It reads the design list from the ENCODER GRAPH, not from
// a hand-written plan: every MatMul whose B is a rank-2 weight, and every Conv as its im2col GEMM (K = cin*k, N = cout), staged on
// the engine's grid (audio_kernel_roles.h). For each distinct staged (K, N) it finds `full_i8_<ROLE>_K<K>_N<N>_b8.elf` in the
// kernel directory, whatever the role prefix: cl-npurt's conv stems are D-named (`D K256 N1024`, `D K2304 N1024`), and the name
// is only how the generator files them. It then adopts them through open_prefill() in that order. The order is the role
// index NpuAudioGemm routes by.
//
// A staged shape with no ELF is left out and printed. Its products are declined by NpuAudioGemm and run on the host,
// counted, so a missing kernel shows in the split instead of as a crash or a silent zero.
//
// engine#1783, sealed set: when the directory holds `sealed_i8_b8.elf` and its list `sealed_i8_b8.shapes` (one
// "K N instance" line per runtime sequence in it), every design opens that ONE ELF, each on its own instance, so all of
// them share one hw context. Per-role ELFs each claim the whole array, and a role change between launches is then a
// context switch: measured on the whisper-small encoder at about 2.4 ms per launch (an O-shaped launch waited 0.38 ms
// when every launch stayed on its context and 2.9 ms in the per-role set). A staged shape the list does not hold
// exactly takes the same-K instance with the smallest wider N (NpuAudioGemm::role_for). ONEBIT_WHISPER_NPU_SEALED=0
// ignores the sealed files.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "audio_exec.h"
#include "audio_kernel_roles.h"
#include "audio_npu_gemm.h"
#include "onnx_graph.h"
#include "q4nx_pack.h"
#include "target/xdna/xdna_target.h"

namespace whisper_npu {

namespace x = ::npu::target::xdna;

struct Device {
    std::unique_ptr<xrt::device> dev;
    std::unique_ptr<x::XdnaTarget> target;
    std::unique_ptr<onebit::audio::NpuAudioGemm> gemm;
    onebit::npu::pack_import::AudioGemm hook;
    std::vector<std::string> missing;   // staged shapes with no ELF, "K<k> N<n>"
};

// The staged (K, N) of every product the executor will offer, in graph order, without repeats.
inline std::vector<std::pair<int, int>> offered_shapes(const onebit::npu::pack_import::OnnxGraphInfo& g) {
    std::vector<std::pair<int, int>> out;
    auto add = [&](int64_t k, int64_t n) {
        const auto s = onebit::audio::staged(onebit::audio::ProjectionShape{int(k), int(n)});
        for (const auto& p : out)
            if (p.first == s.k && p.second == s.n) return;
        out.emplace_back(s.k, s.n);
    };
    auto init_dims = [&](const std::string& name) -> const std::vector<int64_t>* {
        for (const auto& t : g.init_tensors)
            if (t.name == name) return &t.dims;
        return nullptr;
    };
    std::map<std::string, std::pair<int64_t, int64_t>> by_a;   // activation -> (K, summed N), the executor's fused offer
    std::map<std::string, int> members;
    std::map<std::string, bool> same_k;
    for (const auto& n : g.nodes) {
        if (n.ins.size() < 2) continue;
        const std::vector<int64_t>* d = init_dims(n.ins[1]);
        if (d == nullptr) continue;
        if (n.op == "MatMul" && d->size() == 2) {
            add((*d)[0], (*d)[1]);
            auto& e = by_a[n.ins[0]];
            if (members[n.ins[0]]++ == 0) { e.first = (*d)[0]; same_k[n.ins[0]] = true; }
            else if (e.first != (*d)[0]) same_k[n.ins[0]] = false;
            e.second += (*d)[1];
        }
        if (n.op == "Conv" && d->size() == 3) add((*d)[1] * (*d)[2], (*d)[0]);
    }
    // engine#1783: the executor offers projections that share an activation as one product (audio_exec.cpp, the fused
    // q|k|v), so its staged shape is one the device should open too. ONEBIT_AUDIO_FUSE_QKV=0 means it is never offered.
    const char* fq = std::getenv("ONEBIT_AUDIO_FUSE_QKV");
    if (fq == nullptr || fq[0] != '0')
        for (const auto& kv : by_a)
            if (members[kv.first] >= 2 && same_k[kv.first]) add(kv.second.first, kv.second.second);
    return out;
}

// `full_i8_<ROLE>_K<k>_N<n>_b8.elf` in `dir`, any role prefix; empty when none. Two matches is a refusal: which one the
// encoder ran would depend on directory order.
inline std::string find_elf(const std::string& dir, int k, int n, std::string* role, std::string* why) {
    const std::string tail = "_K" + std::to_string(k) + "_N" + std::to_string(n) + "_b8.elf";
    std::string hit;
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) {
        *why = "cannot open " + dir;
        return {};
    }
    while (dirent* e = readdir(d)) {
        const std::string f = e->d_name;
        if (f.rfind("full_i8_", 0) != 0 || f.size() <= tail.size() || f.compare(f.size() - tail.size(), tail.size(), tail) != 0)
            continue;
        if (!hit.empty()) {
            *why = "two ELFs for K" + std::to_string(k) + " N" + std::to_string(n) + ": " + hit + " and " + f;
            closedir(d);
            return {};
        }
        hit = f;
    }
    closedir(d);
    if (!hit.empty()) *role = hit.substr(8, hit.size() - tail.size() - 8);
    return hit;
}

// The sealed set's instance list (see the header): "K N instance" per line, '#' comments. Empty when the directory has
// no sealed set, when ONEBIT_WHISPER_NPU_SEALED=0, or (with *why set) when the list is malformed.
struct SealedShape {
    int k = 0, n = 0;
    std::string instance;
};
inline constexpr const char* kSealedElf = "sealed_i8_b8.elf";
inline std::vector<SealedShape> sealed_shapes(const std::string& dir, std::string* why) {
    std::vector<SealedShape> out;
    if (const char* e = std::getenv("ONEBIT_WHISPER_NPU_SEALED"); e != nullptr && e[0] == '0') return out;
    std::ifstream elf(dir + "/" + kSealedElf);
    std::ifstream lst(dir + "/sealed_i8_b8.shapes");
    if (!elf || !lst) return out;
    std::string line;
    while (std::getline(lst, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        SealedShape s;
        if (!(is >> s.k >> s.n >> s.instance) || s.k <= 0 || s.n <= 0) {
            *why = dir + "/sealed_i8_b8.shapes: \"" + line + "\" is not \"K N instance\"";
            return {};
        }
        out.push_back(s);
    }
    return out;
}

inline Q4nxDesign design_enum(const std::string& role) {
    if (role == "QKV") return Q4NX_DESIGN_QKV;
    if (role == "O") return Q4NX_DESIGN_O;
    if (role == "GU") return Q4NX_DESIGN_GU;
    if (role == "G") return Q4NX_DESIGN_G;
    if (role == "U") return Q4NX_DESIGN_U;
    return Q4NX_DESIGN_D;   // the conv stems are filed as D; the enum only names the log line
}

// Opens device 0, adopts one design per offered shape that has an ELF (M = `rows`, the ELFs' generated M), and wires the
// executor hook. False + why when the device or an ELF cannot be opened.
inline bool open(const onebit::npu::pack_import::OnnxGraphInfo& enc, const std::string& kdir, int rows, Device* out,
                 std::string* why) {
    std::vector<Q4nxDesignGeom> geoms;
    std::vector<onebit::audio::NpuAudioDesign> designs;
    std::vector<std::string> kernels;   // per design: its instance in the sealed ELF ("" = main:seq)
    std::string serr;
    const std::vector<SealedShape> sealed = sealed_shapes(kdir, &serr);
    if (!serr.empty()) {
        *why = serr;
        return false;
    }
    for (const auto& s : offered_shapes(enc)) {
        if (!sealed.empty()) {
            // exact, else the same-K instance with the smallest wider N; each instance is adopted once
            const SealedShape* hit = nullptr;
            for (const auto& c : sealed)
                if (c.k == s.first && c.n >= s.second && (hit == nullptr || c.n < hit->n)) hit = &c;
            if (hit == nullptr) {
                out->missing.push_back("K" + std::to_string(s.first) + " N" + std::to_string(s.second));
                continue;
            }
            bool have = false;
            for (const auto& d : designs) have = have || (d.K == hit->k && d.N == hit->n);
            if (have) continue;
            Q4nxDesignGeom g{};
            g.M = rows;
            g.K = hit->k;
            g.N = hit->n;
            g.role = Q4NX_DESIGN_D;   // the enum only names the log line
            std::snprintf(g.elf_name, sizeof g.elf_name, "%s", kSealedElf);
            geoms.push_back(g);
            designs.push_back(onebit::audio::NpuAudioDesign{rows, hit->k, hit->n});
            kernels.push_back("main:" + hit->instance);
            std::fprintf(stderr, "npu: role %zu = %s instance %s (K%d N%d, M=%d; staged K%d N%d)\n", geoms.size() - 1,
                         kSealedElf, hit->instance.c_str(), hit->k, hit->n, rows, s.first, s.second);
            continue;
        }
        std::string role, err;
        const std::string elf = find_elf(kdir, s.first, s.second, &role, &err);
        if (!err.empty()) {
            *why = err;
            return false;
        }
        if (elf.empty()) {
            out->missing.push_back("K" + std::to_string(s.first) + " N" + std::to_string(s.second));
            continue;
        }
        Q4nxDesignGeom g{};
        g.M = rows;
        g.K = s.first;
        g.N = s.second;
        g.role = design_enum(role);
        std::snprintf(g.elf_name, sizeof g.elf_name, "%s", elf.c_str());
        geoms.push_back(g);
        designs.push_back(onebit::audio::NpuAudioDesign{rows, s.first, s.second});
        kernels.emplace_back();
        std::fprintf(stderr, "npu: role %zu = %s (staged K%d N%d, M=%d)\n", geoms.size() - 1, elf.c_str(), s.first, s.second,
                     rows);
    }
    if (geoms.empty()) {
        *why = "no ELF in " + kdir + " covers any of the encoder's products";
        return false;
    }
    out->dev = std::make_unique<xrt::device>(0);
    out->target = std::make_unique<x::XdnaTarget>();
    // One layer of weight cache: NpuAudioGemm keys every weight at layer 0 with its own wkey.
    out->target->set_device(out->dev.get(), /*weight_layout_b8=*/true, /*weight_cache=*/true, /*n_layers=*/1);
    x::XdnaTarget::Setup setup;
    setup.kernels_dir = kdir;
    setup.b8 = true;
    setup.prefill_kernels = kernels;
    out->target->configure(setup);
    // NPU_PROFILE=1: the target's per-role launch accounting (pack / quant / wait / readback, fills apart), dumped by
    // whisper_transcribe after each encoder pass (engine#1783).
    out->target->set_profile(std::getenv("NPU_PROFILE") != nullptr);
    std::string err;
    if (!out->target->open_prefill(geoms.data(), int(geoms.size()), err)) {
        *why = "open_prefill: " + err;
        return false;
    }
    out->gemm = std::make_unique<onebit::audio::NpuAudioGemm>(*out->target, designs, /*target_caches=*/true);
    onebit::audio::NpuAudioGemm* gm = out->gemm.get();
    out->hook.gemm = [gm](const std::string& key, const float* A, int64_t M, int64_t K, const float* B, int64_t N,
                          std::vector<float>* C, std::string* w) { return gm->gemm(key, A, M, K, B, N, C, w); };
    out->hook.cached = [gm](const std::string& key) { return gm->cached(key); };
    out->hook.gemm_ep = [gm](const std::string& key, const float* A, int64_t M, int64_t K, const float* B, int64_t N,
                             std::vector<float>* C, const onebit::audio::Epilogue& ep, std::string* w) {
        return gm->gemm_ep(key, A, M, K, B, N, C, &ep, w);
    };
    return true;
}

}  // namespace whisper_npu
