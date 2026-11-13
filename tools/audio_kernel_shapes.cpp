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
// audio_kernel_shapes: which GEMM shapes an audio pack's encoder needs, and whether a kernel directory covers them (engine#1783).
//
//   audio_kernel_shapes <pack dir>                     # the shapes the encoder's projections ask for
//   audio_kernel_shapes <pack dir> --write-plan <f>    # ... as a launch plan (JSON): node order, staged shapes, ELF names
//   audio_kernel_shapes <pack dir> --kernels <dir>     # ... and whether each has a full ELF in that directory
//
// Why a tool and not a test: what it reports depends on which pack and which kernel directory are given, and the useful output is a
// list a lane reads before taking a device ticket. The shape derivation is the part worth trusting -- it reads the pack's own ONNX
// graph through the engine's reader, so a pack whose projections differ (768 for small, 1280 for large-v3) reports its own numbers
// instead of a table someone maintained.
//
// The role a projection plays is decided by the tensor it multiplies, not by node order: a packed export fuses q/k/v into one
// 3N-wide matrix, and the ELF it needs is the fused one (N = 3 x hidden). A name this tool does not recognise is reported as
// `unknown`, because a shape with no role is a shape nobody has built a kernel for.

#include <algorithm>
#include "file_create.h"   // onebit_fopen_private_wb: a dump must be 0600 and must not follow a symlink
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "audio_pack.h"
#include "audio_kernel_roles.h"
#include "onnx_graph.h"

namespace fs = std::filesystem;
using onebit::npu::pack_import::OnnxGraphInfo;

namespace {

struct Need {
    std::string role;   // QKV | O | GU | D, or "unknown"
    int kindex = 0;     // the reduction (K)
    int n = 0;          // the output width (N)
    std::string weight;  // the tensor it multiplies, for the message
};

std::string role_of_unused(const std::string& weight) {
    const auto has = [&](const char* s) { return weight.find(s) != std::string::npos; };
    if (has("qkv") || (has("q_proj") && has("k_proj"))) return "QKV";
    if (has("gate_up") || (has("gate_proj") && has("up_proj"))) return "GU";
    if (has("q_proj") || has("query")) return "QKV";
    if (has("k_proj") || has("key") || has("v_proj") || has("value")) return "QKV";
    if (has("out_proj") || has("o_proj") || has("dense")) return "O";
    if (has("down_proj")) return "D";
    if (has("fc1")) return "GU";
    if (has("fc2")) return "D";
    return "unknown";
}

// A packed export names its weights generically (`val_452`) and reaches them through a Transpose; the name that says what the
// projection IS (`blocks.0.attn.query`) sits on the Transpose's input. So the role comes from following the producer chain back to
// a named tensor -- without this every shape reports `unknown`, which is what the first run of this tool did.
const std::string* source_name(const std::map<std::string, std::string>& producer, std::string name) {
    for (int hop = 0; hop < 4; ++hop) {
        const auto it = producer.find(name);
        if (it == producer.end()) return nullptr;
        name = it->second;
        // a name that looks like a parameter path rather than a val_N intermediate is the one to report
        if (name.find("val_") == std::string::npos && name.find("weight") != std::string::npos) {
            static thread_local std::string keep;
            keep = name;
            return &keep;
        }
    }
    static thread_local std::string keep;
    keep = name;
    return &keep;
}

std::vector<Need> shapes_of(const OnnxGraphInfo& g, int hidden) {
    std::map<std::string, Need> by_weight;   // one entry per weight, so a shared weight is not counted twice
    std::map<std::string, std::vector<int64_t>> dims;
    for (const auto& t : g.init_tensors) dims[t.name] = t.dims;
    std::map<std::string, std::string> producer;   // output -> its single non-initializer input, for the role lookup
    for (const auto& node : g.nodes) {
        if (node.outs.empty() || node.ins.empty()) continue;
        if (node.op != "Transpose" && node.op != "Reshape" && node.op != "Cast" && node.op != "Identity") continue;
        producer[node.outs[0]] = node.ins[0];
    }
    for (const auto& node : g.nodes) {
        if (node.op != "MatMul" || node.ins.size() < 2) continue;
        const auto it = dims.find(node.ins[1]);
        if (it == dims.end() || it->second.size() != 2) continue;
        Need need;
        const std::string* named = source_name(producer, node.ins[1]);
        need.weight = (named != nullptr && !named->empty()) ? *named : node.ins[1];
        need.kindex = int(it->second[0]);
        need.n = int(it->second[1]);
        need.role = onebit::audio::role_of_shape(onebit::audio::ProjectionShape{need.kindex, need.n}, hidden);   // after the shape is set: this read 0, 0 on the first run
        by_weight[need.weight] = need;
    }
    std::vector<Need> out;
    for (auto& kv : by_weight) out.push_back(kv.second);
    std::sort(out.begin(), out.end(), [](const Need& a, const Need& b) {
        return a.role == b.role ? a.n < b.n : a.role < b.role;
    });
    return out;
}

// The ELF names the family generator produces: full_i8_<ROLE>_K<K>_N<N>_b8[_m16].elf
std::string elf_name(const Need& n, bool m16) {
    return onebit::audio::elf_name_for(onebit::audio::ProjectionShape{n.kindex, n.n}, 0, m16) == "" ? "" :
           "full_i8_" + n.role + "_K" + std::to_string(onebit::audio::q4nx_pad_k(n.kindex)) + "_N" +
           std::to_string(onebit::audio::q4nx_pad_n(n.n)) + "_b8" +
               (m16 ? "_m16" : "") + ".elf";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <pack dir> [--kernels <dir>]\n", argv[0]);
        return 2;
    }
    onebit::audio::AudioPack pack;
    std::string why;
    if (!onebit::audio::discover_audio_pack(argv[1], &pack, &why)) {
        std::fprintf(stderr, "pack: %s\n", why.c_str());
        return 1;
    }
    OnnxGraphInfo g;
    try {
        g = onebit::npu::pack_import::parse_onnx_graph(pack.encoder);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "encoder: %s\n", e.what());
        return 1;
    }
    // The hidden width comes from the graph's own biggest projection: the widest N among the distinct shapes.
    int hidden = 0;
    {
        // The hidden width is the most common SQUARE weight width: q/k/v/o are h x h in every layer, so the mode of that width is
        // h, where a median over all weights is skewed by the embeddings and the 4h MLP.
        // ONE implementation of the rule, in the header the tests drive: the mode of the square widths, and 0 when two widths
        // tie. A second copy here is the drift this file exists to prevent -- and it had one: this loop took whichever
        // width came first out of a std::map, i.e. the SMALLEST of a tie, silently.
        std::vector<int> square;
        for (const auto& t : g.init_tensors)
            if (t.dims.size() == 2 && t.dims[0] == t.dims[1] && t.dims[0] > 256) square.push_back(int(t.dims[0]));
        hidden = onebit::audio::hidden_from_square_widths(square.data(), int(square.size()));
        if (hidden <= 0) {
            std::fprintf(stderr,
                         "audio_kernel_shapes: the hidden width is ambiguous or absent in %s (%zu square weight width(s), no "
                         "single most common one). Every role would be reported `unknown`, and a role this tool cannot name is "
                         "one whose kernel cannot be checked -- refusing rather than picking one.\n",
                         pack.encoder.c_str(), square.size());
            return 2;
        }
    }
    const std::vector<Need> needs = shapes_of(g, hidden);
    std::printf("pack %s: %zu projection shape(s) in %s (hidden %d)\n", argv[1], needs.size(), pack.encoder.c_str(), hidden);
    std::map<std::string, int> counts;
    for (const auto& n : needs) {
        ++counts[n.role + " K" + std::to_string(n.kindex) + " N" + std::to_string(n.n)];
    }
    for (const auto& kv : counts) std::printf("  %-18s x%d\n", kv.first.c_str(), kv.second);

    std::string kernels, plan_path;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--kernels") == 0) kernels = argv[i + 1];
        else if (std::strcmp(argv[i], "--write-plan") == 0) plan_path = argv[i + 1];
    }
    if (!plan_path.empty()) {
        // The launch plan a device route would read: every projection IN TOPOLOGICAL ORDER (the graph's own order, because a
        // route executes them in it), with the shape the model declares beside the shape the kernel is staged at and the ELF
        // name that staging produces. It is written here rather than derived by the route so that the one place which knows the
        // staging rule is the one place that names the kernel -- the defect this file exists to prevent (engine#1783).
        std::FILE* f = onebit_fopen_private_wb(plan_path.c_str());   // 0600, no symlink (file_create.h)
        if (f == nullptr) {
            std::fprintf(stderr, "audio_kernel_shapes: cannot write %s\n", plan_path.c_str());
            return 1;
        }
        std::fprintf(f, "{\n \"pack\": \"%s\",\n \"encoder\": \"%s\",\n \"hidden\": %d,\n \"launches\": [\n",
                     argv[1], pack.encoder.c_str(), hidden);
        int idx = 0;
        std::vector<int64_t> dims;
        std::vector<std::string> mats;
        for (const auto& t : g.init_tensors)
            if (t.dims.size() == 2 && t.dims[1] > 1) mats.push_back(t.name);
        for (const auto& node : g.nodes) {
            if (node.op != "MatMul" || node.ins.size() < 2) continue;
            const auto need = std::find_if(needs.begin(), needs.end(), [&](const Need& n) {
                return n.weight == node.ins[1] || n.weight == node.ins[0];
            });
            if (need == needs.end()) continue;
            const auto staged = onebit::audio::staged(onebit::audio::ProjectionShape{need->kindex, need->n});
            std::fprintf(f,
                         "%s    {\"order\": %d, \"op\": \"MatMul\", \"weight\": \"%s\", \"role\": \"%s\", "
                         "\"k\": %d, \"n\": %d, \"staged_k\": %d, \"staged_n\": %d, \"elf_b8\": \"%s\", "
                         "\"elf_m16\": \"%s\"}",
                         idx == 0 ? "" : ",\n", idx, need->weight.c_str(), need->role.c_str(), need->kindex, need->n,
                         staged.k, staged.n, elf_name(*need, false).c_str(), elf_name(*need, true).c_str());
            ++idx;
        }
        std::fprintf(f, "\n ],\n \"projections\": %d,\n \"note\": \"staged_* are the widths the ELF is named at "
                        "(K%%256, N%%512); the forward asks for those names, not the model's\"\n}\n", idx);
        std::fclose(f);
        std::printf("plan: %d projection(s) in topological order -> %s\n", idx, plan_path.c_str());
        if (idx == 0) {
            std::fprintf(stderr, "audio_kernel_shapes: the plan is empty -- the pack's graph has no 2-D MatMul on a "
                                 "recognised weight, so nothing would launch\n");
            return 1;
        }
    }

    if (kernels.empty()) return 0;

    // One ELF per distinct (role, K, N), in both M tiers: the family the encoder needs is the same parameterised generator's output.
    std::map<std::string, int> distinct;
    for (const auto& n : needs) distinct[n.role + ":" + std::to_string(n.kindex) + ":" + std::to_string(n.n)] = 1;
    int missing = 0;
    for (const auto& kv : distinct) {
        const std::string key = kv.first;
        const size_t a = key.find(':'), b = key.find(':', a + 1);
        Need n;
        n.role = key.substr(0, a);
        n.kindex = std::stoi(key.substr(a + 1, b - a - 1));
        n.n = std::stoi(key.substr(b + 1));
        for (bool m16 : {false, true}) {
            const std::string name = elf_name(n, m16);
            const bool present = fs::is_regular_file(fs::path(kernels) / name);
            if (!present) ++missing;
            std::printf("  %-6s %5s %s\n", present ? "have" : "MISSING", m16 ? "m16" : "b8", name.c_str());
        }
    }
    std::printf("kernels %s: %d missing of %zu x 2\n", kernels.c_str(), missing, distinct.size());
    if (missing != 0) {
        std::printf("note: a role this tool does not recognise is reported as `unknown` above, so an %s* ELF will always be MISSING\n",
                    "unknown");
    }
    return missing == 0 ? 0 : 1;
}
