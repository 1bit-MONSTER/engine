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
#include "vision_tower.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace onebit::npu::vision {

namespace {

// Rows of work split over a few threads. Every unit (a token, a head) is computed by exactly one thread with exactly
// the arithmetic the single-threaded loop used, so the result does not depend on the thread count: the goldens in
// tests/npu_vision_tower_test.cpp hold bit for bit at any setting. ONEBIT_VISION_THREADS sets the count (1 turns
// threading off); the default is the machine's, capped at 16. Small jobs stay on the calling thread.
std::atomic<int> g_threads{0};            // 0: not set, use the environment / the machine
std::atomic<double> g_min_work{4e6};      // multiply-adds below which a loop stays on the calling thread
int tower_threads() {
    const int forced = g_threads.load();
    if (forced > 0) return forced;
    static const int n = [] {
        int v = 0;
        if (const char* e = std::getenv("ONEBIT_VISION_THREADS")) v = std::atoi(e);
        if (v <= 0) v = int(std::thread::hardware_concurrency());
        return v < 1 ? 1 : (v > 16 ? 16 : v);
    }();
    return n;
}
template <typename Fn>
void parallel_for(int count, double work_per_item, Fn fn) {   // fn(first, last) over [0, count)
    const int nt = tower_threads();
    if (nt <= 1 || count < 2 || work_per_item * count < g_min_work.load()) {
        fn(0, count);
        return;
    }
    const int parts = count < nt ? count : nt;
    std::vector<std::thread> pool;
    pool.reserve(size_t(parts - 1));
    for (int p = 1; p < parts; ++p)
        pool.emplace_back(fn, int(int64_t(count) * p / parts), int(int64_t(count) * (p + 1) / parts));
    fn(0, int(int64_t(count) / parts));
    for (std::thread& t : pool) t.join();
}

// y[n, rows] = x[n, in] * W[rows_stored.., in_stride]^T + b, using the first `rows` rows and the first `in` columns of
// a weight stored `in_stride` wide. The padded MLPs are read through this: the stride is the tensor's, the extent the
// model's. Eight partial sums so the compiler can vectorise without being allowed to reorder a single chain.
void linear(const float* x, int n, int in, const float* w, int in_stride, const float* b, int rows, float* y) {
  parallel_for(n, double(in) * rows, [=](int t0, int t1) {
    // Tokens are taken a few at a time and every weight row is applied to the whole handful before the next row is
    // touched. A weight matrix here is tens of megabytes and a token's input a few kilobytes, so walking the matrix
    // once per token (the obvious loop) is bound by memory, not arithmetic; this walks it once per handful. Each
    // dot product is the same sequence of operations either way, so the result is the same bits.
    const int kBlock = 16;
    for (int tb = t0; tb < t1; tb += kBlock) {
      const int te = tb + kBlock < t1 ? tb + kBlock : t1;
      for (int r = 0; r < rows; ++r) {
        const float* wr = w + size_t(r) * in_stride;
        const float bias = b ? b[r] : 0.0f;
        for (int t = tb; t < te; ++t) {
            const float* xr = x + size_t(t) * in;
            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            int i = 0;
            for (; i + 8 <= in; i += 8)
                for (int k = 0; k < 8; ++k) acc[k] += xr[i + k] * wr[i + k];
            float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            for (; i < in; ++i) s += xr[i] * wr[i];
            y[size_t(t) * rows + r] = s + bias;
        }
      }
    }
  });
}

// torch.nn.LayerNorm: the population variance, eps inside the root.
void layer_norm(const float* x, int n, int width, const float* w, const float* b, float* y) {
    for (int t = 0; t < n; ++t) {
        const float* xr = x + size_t(t) * width;
        float* yr = y + size_t(t) * width;
        double mean = 0;
        for (int i = 0; i < width; ++i) mean += xr[i];
        mean /= width;
        double var = 0;
        for (int i = 0; i < width; ++i) var += (xr[i] - mean) * (xr[i] - mean);
        const float inv = float(1.0 / std::sqrt(var / width + 1e-6));
        for (int i = 0; i < width; ++i) yr[i] = float(xr[i] - mean) * inv * w[i] + b[i];
    }
}

inline float gelu_tanh(float v) {
    return 0.5f * v * (1.0f + std::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}
inline float gelu_erf(float v) { return 0.5f * v * (1.0f + std::erf(v * 0.7071067811865476f)); }

struct Attn {
    std::vector<float> qkv_w, qkv_b, proj_w, proj_b;
};
Attn load_attn(const Pack& p, const std::string& stem) {
    return {p.f32(stem + ".attn.qkv.weight"), p.f32(stem + ".attn.qkv.bias"), p.f32(stem + ".attn.proj.weight"),
            p.f32(stem + ".attn.proj.bias")};
}

// out[n, H] = proj(attention(x)), where tokens attend only inside consecutive groups of `group` tokens (group == n is
// full attention). qkv is [3H, H]: q, k, v stacked in that order, heads contiguous inside each.
void attention(const float* x, int n, int H, int heads, int head_dim, int group, const Attn& a, float* out) {
    std::vector<float> qkv(size_t(n) * 3 * H), ctx(size_t(n) * H);
    linear(x, n, H, a.qkv_w.data(), H, a.qkv_b.data(), 3 * H, qkv.data());
    const float scale = 1.0f / std::sqrt(float(head_dim));
    const int groups = n / group;
    // one unit = one head of one group; each writes its own slice of ctx
    parallel_for(groups * heads, double(group) * group * head_dim * 2, [&, scale](int u0, int u1) {
      std::vector<float> prob(static_cast<size_t>(group));
      for (int u = u0; u < u1; ++u) {
        const int g0 = (u / heads) * group, h = u % heads;
        {
            for (int i = 0; i < group; ++i) {
                const float* q = qkv.data() + size_t(g0 + i) * 3 * H + size_t(h) * head_dim;
                float mx = -INFINITY;
                for (int j = 0; j < group; ++j) {
                    const float* k = qkv.data() + size_t(g0 + j) * 3 * H + H + size_t(h) * head_dim;
                    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    int d = 0;
                    for (; d + 8 <= head_dim; d += 8)
                        for (int u = 0; u < 8; ++u) acc[u] += q[d + u] * k[d + u];
                    float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
                    for (; d < head_dim; ++d) s += q[d] * k[d];
                    prob[size_t(j)] = s * scale;
                    if (prob[size_t(j)] > mx) mx = prob[size_t(j)];
                }
                double sum = 0;
                for (int j = 0; j < group; ++j) {
                    prob[size_t(j)] = std::exp(prob[size_t(j)] - mx);
                    sum += prob[size_t(j)];
                }
                float* o = ctx.data() + size_t(g0 + i) * H + size_t(h) * head_dim;
                std::memset(o, 0, sizeof(float) * size_t(head_dim));
                for (int j = 0; j < group; ++j) {
                    const float pj = float(prob[size_t(j)] / sum);
                    const float* v = qkv.data() + size_t(g0 + j) * 3 * H + 2 * size_t(H) + size_t(h) * head_dim;
                    for (int d = 0; d < head_dim; ++d) o[d] += pj * v[d];
                }
            }
        }
      }
    });
    linear(ctx.data(), n, H, a.proj_w.data(), H, a.proj_b.data(), H, out);
}

// fc2(act(fc1(x))): `mid` rows of fc1 and `mid` columns of an fc2 stored `mid_stored` wide.
template <typename Act>
void mlp(const float* x, int n, int in, const std::vector<float>& w1, const std::vector<float>& b1, int mid,
         const std::vector<float>& w2, const std::vector<float>& b2, int mid_stored, int out_w, Act act, float* out) {
    std::vector<float> h(size_t(n) * mid);
    linear(x, n, in, w1.data(), in, b1.data(), mid, h.data());
    for (float& v : h) v = act(v);
    linear(h.data(), n, mid, w2.data(), mid_stored, b2.data(), out_w, out);
}

// The four tokens of each 2x2 window of a gh x gw row-major grid, side by side: out[(gh/2)*(gw/2), 4*H], windows
// row-major, inside a window (0,0) (0,1) (1,0) (1,1). This is both the vit_merger's window order and the merger's view.
std::vector<float> windows(const float* x, int gh, int gw, int H) {
    std::vector<float> o(size_t(gh) * gw * H);
    size_t at = 0;
    for (int wy = 0; wy < gh / 2; ++wy)
        for (int wx = 0; wx < gw / 2; ++wx)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx) {
                    std::memcpy(o.data() + at, x + (size_t(2 * wy + dy) * gw + size_t(2 * wx + dx)) * H, sizeof(float) * size_t(H));
                    at += size_t(H);
                }
    return o;
}

}  // namespace

void tower_threading(int threads, double min_work) {
    g_threads.store(threads > 16 ? 16 : (threads < 0 ? 0 : threads));
    g_min_work.store(min_work < 0 ? 0 : min_work);
}

std::vector<float> tower_forward(const Pack& p, const float* pixels, int gh, int gw, std::vector<float>* tower) {
    const Geometry& g = p.geometry();
    const std::string pre = "model.visual.";
    if (g.heads <= 0 || g.head_dim <= 0)
        throw std::runtime_error(p.path() + ": the head count is only in config.json, and none was given");
    if (g.insert_layer < 0 || g.insert_layer >= g.layers)
        throw std::runtime_error(p.path() + ": the vit_merger runs after block " + std::to_string(g.insert_layer) +
                                 ", which this tower of " + std::to_string(g.layers) + " blocks does not have");
    if (!pixels || gh <= 0 || gw <= 0 || gh % 4 != 0 || gw % 4 != 0)
        throw std::runtime_error(p.path() + ": a " + std::to_string(gh) + " x " + std::to_string(gw) +
                                 " patch grid cannot be merged twice by 2 x 2 (both extents must be multiples of 4)");
    if (gh > 4096 || gw > 4096) throw std::runtime_error(p.path() + ": the patch grid is larger than any image this tower takes");
    const int H = g.hidden, P = g.patch, C = g.channels, M = g.merge_hidden;
    int n = gh * gw;

    // 1 + 2: patches (c, dy, dx) in the weight's own order, then the nearest position row
    std::vector<float> x(size_t(n) * H);
    {
        const std::vector<float> w = p.f32(pre + "patch_embed.proj.weight"), b = p.f32(pre + "patch_embed.proj.bias");
        const std::vector<float> pos = p.f32(pre + "pos_embed.weight");
        const int pd = C * P * P, W = gw * P;
        std::vector<float> patch(static_cast<size_t>(pd));
        for (int y = 0; y < gh; ++y)
            for (int xx = 0; xx < gw; ++xx) {
                size_t at = 0;
                for (int c = 0; c < C; ++c)
                    for (int dy = 0; dy < P; ++dy)
                        for (int dx = 0; dx < P; ++dx)
                            patch[at++] = pixels[(size_t(c) * gh * P + size_t(y * P + dy)) * W + size_t(xx * P + dx)];
                float* row = x.data() + (size_t(y) * gw + xx) * H;
                linear(patch.data(), 1, pd, w.data(), pd, b.data(), H, row);
                const float* pr = pos.data() + size_t(nearest_position(y, xx, gh, gw, g.grid)) * H;
                for (int i = 0; i < H; ++i) row[i] += pr[i];
            }
    }

    std::vector<float> t(size_t(n) * H), u(size_t(n) * H);
    const auto tanh_act = [](float v) { return gelu_tanh(v); };
    for (int layer = 0; layer < g.layers; ++layer) {
        const std::string b = pre + "blocks." + std::to_string(layer);
        // 3 / 5: one block
        layer_norm(x.data(), n, H, p.f32(b + ".norm1.weight").data(), p.f32(b + ".norm1.bias").data(), t.data());
        attention(t.data(), n, H, g.heads, g.head_dim, n, load_attn(p, b), u.data());
        for (size_t i = 0; i < size_t(n) * H; ++i) x[i] += u[i];
        layer_norm(x.data(), n, H, p.f32(b + ".norm2.weight").data(), p.f32(b + ".norm2.bias").data(), t.data());
        mlp(t.data(), n, H, p.f32(b + ".mlp.linear_fc1.weight"), p.f32(b + ".mlp.linear_fc1.bias"), g.model_intermediate,
            p.f32(b + ".mlp.linear_fc2.weight"), p.f32(b + ".mlp.linear_fc2.bias"), g.intermediate, H, tanh_act, u.data());
        for (size_t i = 0; i < size_t(n) * H; ++i) x[i] += u[i];

        if (layer != g.insert_layer) continue;
        // 4: the vit_merger
        const std::string v = pre + "vit_merger";
        layer_norm(x.data(), n, H, p.f32(v + ".norm1.weight").data(), p.f32(v + ".norm1.bias").data(), t.data());
        const std::vector<float> normed = windows(t.data(), gh, gw, H);
        std::vector<float> res = windows(x.data(), gh, gw, H);        // the residual, in window order too
        attention(normed.data(), n, H, g.heads, g.head_dim, 4, load_attn(p, v), u.data());
        for (size_t i = 0; i < size_t(n) * H; ++i) res[i] += u[i];     // (n/4 windows) x (4 tokens side by side)
        const int nw = n / 4;
        std::vector<float> flat(size_t(nw) * M), y(size_t(nw) * H);
        layer_norm(res.data(), nw, M, p.f32(v + ".pre_norm.weight").data(), p.f32(v + ".pre_norm.bias").data(), flat.data());
        mlp(flat.data(), nw, M, p.f32(v + ".mlp.linear_fc1.weight"), p.f32(v + ".mlp.linear_fc1.bias"),
            g.model_vit_merger_intermediate, p.f32(v + ".mlp.linear_fc2.weight"), p.f32(v + ".mlp.linear_fc2.bias"),
            g.vit_merger_intermediate, H, tanh_act, y.data());
        for (int wdw = 0; wdw < nw; ++wdw)
            for (int i = 0; i < H; ++i) {
                const float* r = res.data() + size_t(wdw) * M;
                y[size_t(wdw) * H + i] += 0.25f * (r[i] + r[H + i] + r[2 * H + i] + r[3 * H + i]);
            }
        x.swap(y);
        gh /= 2;
        gw /= 2;
        n = nw;
    }

    // 5: the tower's last LayerNorm
    std::vector<float> last(size_t(n) * H);
    layer_norm(x.data(), n, H, p.f32(pre + "merger.norm.weight").data(), p.f32(pre + "merger.norm.bias").data(), last.data());
    if (tower) *tower = last;

    // 6: the merger
    const int no = n / 4;
    const std::vector<float> grouped = windows(last.data(), gh, gw, H);
    std::vector<float> normed(size_t(no) * M), out(size_t(no) * g.out_hidden);
    layer_norm(grouped.data(), no, M, p.f32(pre + "merger.pre_norm.weight").data(), p.f32(pre + "merger.pre_norm.bias").data(),
               normed.data());
    mlp(normed.data(), no, M, p.f32(pre + "merger.linear_fc1.weight"), p.f32(pre + "merger.linear_fc1.bias"), M,
        p.f32(pre + "merger.linear_fc2.weight"), p.f32(pre + "merger.linear_fc2.bias"), M, g.out_hidden,
        [](float v) { return gelu_erf(v); }, out.data());
    return out;
}

}  // namespace onebit::npu::vision
