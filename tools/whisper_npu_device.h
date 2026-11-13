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

#pragma once

#include <cstdio>
#include <dirent.h>
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
    for (const auto& n : g.nodes) {
        if (n.ins.size() < 2) continue;
        const std::vector<int64_t>* d = init_dims(n.ins[1]);
        if (d == nullptr) continue;
        if (n.op == "MatMul" && d->size() == 2) add((*d)[0], (*d)[1]);
        if (n.op == "Conv" && d->size() == 3) add((*d)[1] * (*d)[2], (*d)[0]);
    }
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
    for (const auto& s : offered_shapes(enc)) {
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
    out->target->configure(setup);
    std::string err;
    if (!out->target->open_prefill(geoms.data(), int(geoms.size()), err)) {
        *why = "open_prefill: " + err;
        return false;
    }
    out->gemm = std::make_unique<onebit::audio::NpuAudioGemm>(*out->target, designs, /*target_caches=*/true);
    onebit::audio::NpuAudioGemm* gm = out->gemm.get();
    out->hook.gemm = [gm](const std::string& key, const float* A, int64_t M, int64_t K, const float* B, int64_t N,
                          std::vector<float>* C, std::string* w) { return gm->gemm(key, A, M, K, B, N, C, w); };
    return true;
}

}  // namespace whisper_npu
