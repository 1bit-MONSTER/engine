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
// The lm-head tile cache (engine#1716): load one packed head tile from a file a previous serve wrote, or pack it and
// write that file. The contract is in q4nx_head_image.h; this file is the only place that decides "hit".
//
// It is separate from the forward so it runs without a device: the forward hands it the weight BO's mapping as `codes`,
// a test hands it a malloc'd buffer, and both get the same bytes.

#include "q4nx_head_image.h"
#include "q4nx_pack.h"
#include "q4nx_weight_layout.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

inline size_t bpos(size_t i, size_t j, int KD, int ND, int blocked) {
    return blocked ? q4nx_layout::blocked_index(i, j, KD, ND) : q4nx_layout::rowmajor_index(i, j, KD, ND);
}

// The sampled columns of a tile holding `ncols` real columns: both ends and an even spread between them.
int sample_columns(int ncols, int out[Q4NX_HEAD_TILE_SAMPLE]) {
    if (ncols <= 0) return 0;
    int n = 0;
    for (int s = 0; s < Q4NX_HEAD_TILE_SAMPLE; s++) {
        const int j = Q4NX_HEAD_TILE_SAMPLE > 1
                          ? (int)(((long long)(ncols - 1) * s) / (Q4NX_HEAD_TILE_SAMPLE - 1))
                          : 0;
        if (n == 0 || out[n - 1] != j) out[n++] = j;
    }
    return n;
}

// Does the image agree with the packer on the sampled columns? Each column is packed ALONE through the production
// packer (K x 1, row-major) and compared with the image at the layout's own index, so nothing here restates the
// quantiser: if the packer changes, this changes with it.
bool sample_agrees(const Q4nxHeadImageKey& key, int n0, int ncols, int k, int kd, int nd, Q4nxHeadColFn col, void* ctx,
                   const int8_t* codes, const float* scales, Q4nxHeadTileResult* res) {
    int cols[Q4NX_HEAD_TILE_SAMPLE];
    const int ns = sample_columns(ncols, cols);
    std::vector<float> w((size_t)k);
    std::vector<int8_t> c((size_t)k);
    for (int s = 0; s < ns; s++) {
        const int j = cols[s];
        std::fill(w.begin(), w.end(), 0.0f);
        col(ctx, n0 + j, w.data());
        float sc = 0.0f;
        double ssum = 0.0;
        if (q4nx_pack_B_int8_blocked(c.data(), w.data(), k, 1, k, 1, /*blocked=*/0, &sc, &ssum) != 0) return false;
        if (std::memcmp(&sc, &scales[j], sizeof sc) != 0) {
            std::snprintf(res->why, sizeof res->why,
                          "the cached head image's column %d has scale %.9g where these weights pack to %.9g: it was "
                          "written for different head weights or by a different packer",
                          n0 + j, (double)scales[j], (double)sc);
            return false;
        }
        for (int i = 0; i < k; i++) {
            const int8_t got = codes[bpos((size_t)i, (size_t)j, kd, nd, (int)key.blocked)];
            if (got != c[(size_t)i]) {
                std::snprintf(res->why, sizeof res->why,
                              "the cached head image's column %d row %d holds code %d where these weights pack to %d: "
                              "it was written for different head weights or by a different packer",
                              n0 + j, i, (int)got, (int)c[(size_t)i]);
                return false;
            }
        }
        res->sampled++;
    }
    return true;
}

}  // namespace

extern "C" uint64_t q4nx_head_source_hash(const void* data, size_t nbytes, const uint32_t* extra, int n_extra) {
    if (!data || nbytes == 0) return 0;
    uint64_t h = 1469598103934665603ull;
    const unsigned char* b = (const unsigned char*)data;
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= (uint64_t)b[i];
        h *= 1099511628211ull;
    }
    const unsigned char* e = (const unsigned char*)extra;
    for (size_t i = 0; extra && i < (size_t)(n_extra > 0 ? n_extra : 0) * sizeof(uint32_t); ++i) {
        h ^= (uint64_t)e[i];
        h *= 1099511628211ull;
    }
    return h ? h : 1;   // 0 is "no data"
}

extern "C" int q4nx_head_tile_image(const char* dir, const Q4nxHeadImageKey* key, int tile, int n0, int ncols, int k, int t,
                                    int kd, int nd, Q4nxHeadColFn col, void* ctx, int8_t* codes, float* scales,
                                    Q4nxHeadTileResult* res) {
    Q4nxHeadTileResult local;
    if (!res) res = &local;
    std::memset(res, 0, sizeof *res);
    if (!key || !col || !codes || !scales || tile < 0 || n0 < 0 || ncols <= 0 || k <= 0 || t <= 0 || ncols > t || k > kd ||
        t > nd || (uint32_t)k != key->kpad || (uint32_t)t != key->t) {
        std::snprintf(res->why, sizeof res->why, "head tile %d: unusable geometry (k %d t %d kd %d nd %d ncols %d)", tile, k,
                      t, kd, nd, ncols);
        return 0;
    }
    if (key->blocked && !q4nx_layout::blocked_fits(kd, nd)) {
        std::snprintf(res->why, sizeof res->why, "head tile %d: %d x %d has no blocked layout", tile, kd, nd);
        return 0;
    }
    const size_t image_bytes = (size_t)kd * (size_t)nd;

    char path[4096];
    path[0] = 0;
    if (dir && *dir) {
        char name[64];
        if (q4nx_head_image_key_name(key, name, (int)sizeof name)) {
            // "headimg-<digest>.q4hi" -> "headimg-<digest>-t<tile>.q4hi": one file per tile of the same key.
            char* dot = std::strrchr(name, '.');
            if (dot) *dot = 0;
            const int n = std::snprintf(path, sizeof path, "%s/%s-t%03d.q4hi", dir, name, tile);
            if (n <= 0 || n >= (int)sizeof path) path[0] = 0;
        }
    }

    if (path[0]) {
        Q4nxHeadImageInfo info;
        char why[256];
        if (q4nx_head_image_read(path, codes, (int)image_bytes, scales, nd, &info, why, (int)sizeof why)) {
            if (sample_agrees(*key, n0, ncols, k, kd, nd, col, ctx, codes, scales, res)) {
                res->hit = 1;
                return 1;
            }
            res->refused = 1;   // `why` was set by the sample
        } else if (!std::strstr(why, "not there")) {
            res->refused = 1;
            std::snprintf(res->why, sizeof res->why, "%s", why);
        }
    }

    // MISS: exactly what I8Ctx::packB_into() does -- zero the image, gather the tile, pack it.
    std::memset(codes, 0, image_bytes);
    std::fill(scales, scales + nd, 0.0f);
    std::vector<float> bt((size_t)k * (size_t)t, 0.0f);   // padded columns must be zero
    {
        std::vector<float> w((size_t)k);
        for (int j = 0; j < ncols; j++) {
            std::fill(w.begin(), w.end(), 0.0f);
            col(ctx, n0 + j, w.data());
            for (int i = 0; i < k; i++) bt[(size_t)i * (size_t)t + (size_t)j] = w[(size_t)i];
        }
    }
    double ssum = 0.0;
    if (q4nx_pack_B_int8_blocked(codes, bt.data(), k, t, kd, nd, (int)key->blocked, scales, &ssum) != 0) {
        std::snprintf(res->why, sizeof res->why, "head tile %d: the packer refused %d x %d into %d x %d", tile, k, t, kd, nd);
        return 0;
    }
    if (path[0]) res->wrote = q4nx_head_image_write(path, codes, scales, kd, nd);
    return 1;
}
