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
// moe_expert_bench: times one layer's routed-expert block for one decode token on a ggml
// device, with the model's real expert weights: gate and up MUL_MAT_ID, SwiGLU, down
// MUL_MAT_ID, for top-k random distinct experts (new ones every run). The GPU side of the
// three-way split in docs/moe-streaming.md.
//
//   moe_expert_bench model.gguf layer device top_k [runs]     device: Vulkan0, HRX0, CPU
//
// Build against a libllama build's ggml: g++ -std=c++17 -O2 -I<llama.cpp>/ggml/include
//   tools/moe_expert_bench.cpp -L<build>/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,<build>/bin
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "moe_expert_bench model.gguf layer device top_k [runs]\n"); return 2; }
    const std::string path = argv[1], dev_name = argv[3];
    const int layer = atoi(argv[2]), top_k = atoi(argv[4]), runs = argc > 5 ? atoi(argv[5]) : 50;

    // the shard holding this layer's experts
    std::vector<std::string> files = { path };
    const auto pos = path.find("-00001-of-");
    if (pos != std::string::npos) {
        files.clear();
        const int n = atoi(path.c_str() + pos + 10);
        for (int i = 1; i <= n; i++) { char b[32]; snprintf(b, sizeof b, "-%05d-of-", i); files.push_back(path.substr(0, pos) + b + path.substr(pos + 10)); }
    }
    const std::string b = "blk." + std::to_string(layer) + ".";
    const char* names[3] = { "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
    ggml_tensor* meta_t[3] = {};
    std::vector<uint8_t> bytes[3];
    for (auto& fn : files) {
        ggml_context* meta = nullptr;
        gguf_context* g = gguf_init_from_file(fn.c_str(), { true, &meta });
        if (!g) continue;
        FILE* f = fopen(fn.c_str(), "rb");
        for (int i = 0; i < 3; i++) {
            const int64_t id = gguf_find_tensor(g, (b + names[i]).c_str());
            if (id < 0) continue;
            meta_t[i] = ggml_get_tensor(meta, (b + names[i]).c_str());
            bytes[i].resize(gguf_get_tensor_size(g, id));
            fseeko(f, gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id), SEEK_SET);
            if (fread(bytes[i].data(), 1, bytes[i].size(), f) != bytes[i].size()) return 1;
        }
        fclose(f);
    }
    for (int i = 0; i < 3; i++) if (!meta_t[i]) { fprintf(stderr, "no %s%s\n", b.c_str(), names[i]); return 1; }
    const int64_t n_embd = meta_t[0]->ne[0], n_ff = meta_t[0]->ne[1], n_exp = meta_t[0]->ne[2];

    ggml_backend_load_all();
    ggml_backend_t be = dev_name == "CPU" ? ggml_backend_cpu_init()
                                          : ggml_backend_dev_init(ggml_backend_dev_by_name(dev_name.c_str()), nullptr);
    if (!be) { fprintf(stderr, "no device %s\n", dev_name.c_str()); return 1; }
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* w[3];
    for (int i = 0; i < 3; i++) w[i] = ggml_new_tensor_3d(ctx, meta_t[i]->type, meta_t[i]->ne[0], meta_t[i]->ne[1], meta_t[i]->ne[2]);
    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, 1);
    ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, 1);
    ggml_tensor* gate = ggml_mul_mat_id(ctx, w[0], x, ids);
    ggml_tensor* up = ggml_mul_mat_id(ctx, w[1], x, ids);
    ggml_tensor* act = ggml_swiglu_split(ctx, gate, up);
    ggml_tensor* out = ggml_mul_mat_id(ctx, w[2], act, ids);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int i = 0; i < 3; i++) ggml_backend_tensor_set(w[i], bytes[i].data(), 0, bytes[i].size());
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> xv(n_embd);
    for (auto& v : xv) v = nd(rng);
    ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (!ggml_backend_supports_op(be, out) || !ggml_backend_supports_op(be, gate)) {
        fprintf(stderr, "%s does not support this MUL_MAT_ID\n", dev_name.c_str());
        return 1;
    }
    std::vector<int32_t> perm(n_exp);
    std::iota(perm.begin(), perm.end(), 0);
    std::vector<double> ms;
    for (int r = 0; r < runs + 5; r++) {
        std::shuffle(perm.begin(), perm.end(), rng);
        ggml_backend_tensor_set(ids, perm.data(), 0, top_k * 4);
        ggml_backend_synchronize(be);
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        if (r >= 5) ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    const double mb = double(bytes[0].size() + bytes[1].size() + bytes[2].size()) / n_exp * top_k / 1e6;
    printf("%s layer %d on %s: %s/%s/%s, %lld x %lld, %lld experts, top-%d: median %.3f ms (p10 %.3f, p90 %.3f), %.1f MB read, %.1f GB/s\n",
           path.c_str(), layer, dev_name.c_str(), ggml_type_name(meta_t[0]->type), ggml_type_name(meta_t[1]->type),
           ggml_type_name(meta_t[2]->type), (long long) n_embd, (long long) n_ff, (long long) n_exp, top_k, med,
           ms[ms.size() / 10], ms[ms.size() * 9 / 10], mb, mb / med);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
