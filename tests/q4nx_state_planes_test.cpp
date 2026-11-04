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
// q4nx_state_planes_test.cpp -- the snapshot's two kinds of plane, at the shipped Bonsai-2-27B geometry.
//
// The bug this pins down (2026-11-04): snapshot_state() cut the GDN conv window and the recurrent state with the
// per-position count `tokens * NKV * HD`, exactly as it cut the KV.  Those planes are one fixed block per layer
// that each position *updates*, so at 180 tokens the snapshot kept 23 % of the recurrent plane, and every
// restore_state() then failed its "covers the positions being restored" test: the serve logged
// `[prefix] slot=-1 cached=0 ... reason=slot has no usable state` on 5 of 5 long requests and re-prefilled the
// whole prompt every time.  The first block below is that reproducer, on the same arithmetic.
//
// The second half of the contract is *where* a snapshot may be taken: the whole planes are updated in place, so
// `resume_point()` (app/npu_prefix_cache.h) may only resume one token short of the snapshot and of the prompt --
// restoring `cached - 1` positions out of a snapshot taken at `cached` applies the last token's GDN update twice,
// which the serve measured as the same prompt answered with different text.
//
// Geometry (state inventory, TBONSAI-HYBRID.md §3): NKV=4, HD=256 -> per_pos = 1024 floats; the conv window is
// CD*(CK-1) = 30,720 floats per layer; the recurrent state is 48*128*128 = 786,432 floats per layer.
#include "npu_prefix_cache.h"
#include "q4nx_state_planes.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using q4nx_state::give_per_pos;
using q4nx_state::give_whole;
using q4nx_state::take_per_pos;
using q4nx_state::take_whole;

constexpr size_t kPerPos = 1024;    // NKV * HD
constexpr size_t kConv = 30720;     // CD * (CK - 1)
constexpr size_t kRec = 786432;     // lin_nvh * lin_khd * lin_vhd
constexpr size_t kKvLayers = 64;    // a plane per layer; empty on a layer that shares its KV
constexpr size_t kGdnLayers = 64;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        failures++;
    }
}

std::vector<std::vector<float>> plane(size_t layers, size_t n, float fill) {
    return std::vector<std::vector<float>>(layers, std::vector<float>(n, fill));
}

// The pre-fix semantics, kept here only to prove they are what the serve was doing.
void old_snapshot(const std::vector<std::vector<float>>& conv, const std::vector<std::vector<float>>& rec,
                  size_t want, std::vector<std::vector<float>>& conv_out, std::vector<std::vector<float>>& rec_out) {
    auto cut = [want](const std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst) {
        dst.assign(src.size(), {});
        for (size_t l = 0; l < src.size(); l++)
            dst[l].assign(src[l].begin(), src[l].begin() + long(std::min(want, src[l].size())));
    };
    cut(conv, conv_out);
    cut(rec, rec_out);
}

// The pre-fix restore test, verbatim: one `want` for every plane.
bool old_put(const std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst, size_t want) {
    if (dst.size() < src.size()) return false;
    for (size_t l = 0; l < src.size(); l++) {
        if (!src[l].empty() && src[l].size() < want) return false;
        if (src[l].empty()) {
            dst[l].clear();
            continue;
        }
        dst[l].assign(src[l].begin(), src[l].begin() + long(std::min(want, src[l].size())));
    }
    return true;
}

void test_old_planes_were_truncated() {
    // Two symptoms from the same line, one on each side of the plane size.
    //
    // (1) A prompt longer than 30 tokens: `want` exceeds the 30,720-float conv plane, so the restore's
    //     "covers the positions" test fails and the serve logs `slot has no usable state`.  Every prompt the
    //     cache was asked about on 2026-11-04 was 849 or 704 tokens, i.e. this side.
    // (2) A prompt shorter than 768 tokens: `want` is *below* the 786,432-float recurrent plane, so the snapshot
    //     silently keeps only `want` of it (23 % at 180 tokens) and the install would short the plane.
    for (size_t tokens : {size_t(180), size_t(848)}) {
        const size_t want = tokens * kPerPos;
        const auto conv = plane(kGdnLayers, kConv, 1.0f);
        const auto rec = plane(kGdnLayers, kRec, 2.0f);
        std::vector<std::vector<float>> conv_saved, rec_saved;
        old_snapshot(conv, rec, want, conv_saved, rec_saved);
        check(conv_saved[0].size() == std::min(want, kConv), "old: conv plane cut with the per-position count");
        check(rec_saved[0].size() == std::min(want, kRec), "old: rec plane cut with the per-position count");
        if (want < kRec) {
            check(rec_saved[0].size() < kRec, "old: recurrent plane truncated below its real size");
            check(rec_saved[0].size() == tokens * kPerPos, "old: it kept exactly `want` floats of it");
        }
        // And the restore test the serve applies: the conv plane can never cover `want` past 30 tokens.
        auto dst = plane(kGdnLayers, kConv, 0.0f);
        check(!old_put(conv_saved, dst, want), "old: restore refused the conv plane (slot has no usable state)");
    }
    // The two sides in one line each, so the numbers are in the test and not only in this comment.
    check(size_t(180) * kPerPos == 184320 && 184320 * 100 / kRec == 23, "180-token prompt kept 23 % of the plane");
    check(size_t(848) * kPerPos > kConv, "848-token prompt's `want` is past the conv plane's whole size");
}

void test_whole_planes_survive() {
    const auto conv = plane(kGdnLayers, kConv, 1.0f);
    const auto rec = plane(kGdnLayers, kRec, 2.0f);
    for (size_t tokens : {size_t(180), size_t(848), size_t(8192)}) {
        const size_t want = tokens * kPerPos;
        std::vector<std::vector<float>> conv_saved, rec_saved;
        take_whole(conv, conv_saved);
        take_whole(rec, rec_saved);
        check(conv_saved[0].size() == kConv, "conv plane carried whole");
        check(rec_saved[0].size() == kRec, "rec plane carried whole");
        // Snapshot size is now the same at every prompt length for the GDN half: that is the property that was
        // missing -- a snapshot is the KV for `tokens` positions plus two fixed planes.
        check(conv_saved.size() == kGdnLayers && rec_saved.size() == kGdnLayers, "one plane per layer");
        auto conv_dst = plane(kGdnLayers, kConv, 0.0f), rec_dst = plane(kGdnLayers, kRec, 0.0f);
        check(give_whole(conv_saved, conv_dst), "restore accepts a whole conv plane");
        check(give_whole(rec_saved, rec_dst), "restore accepts a whole rec plane");
        check(conv_dst[7] == std::vector<float>(kConv, 1.0f), "conv plane content round-trips");
        check(rec_dst[63] == std::vector<float>(kRec, 2.0f), "rec plane content round-trips");
        check(want >= kPerPos, "sanity: the prompt is at least one position");
    }
}

void test_kv_planes_still_per_position() {
    const auto kv = plane(kKvLayers, 4096 * kPerPos, 3.0f);   // a 4k-position cache
    std::vector<std::vector<float>> saved;
    take_per_pos(kv, saved, 848 * kPerPos);
    check(saved[0].size() == 848 * kPerPos, "KV snapshot keeps only the positions asked for");
    auto dst = plane(kKvLayers, 4096 * kPerPos, 0.0f);
    check(give_per_pos(saved, dst, 848 * kPerPos), "restore accepts a KV plane that covers the positions");
    check(dst[0].size() == 848 * kPerPos, "restored KV is cut to the positions being restored");
    check(!give_per_pos(saved, dst, 849 * kPerPos), "restore refuses KV that does not cover the positions");
    // A shared-KV layer carries an empty plane: the destination is left empty, not filled with another layer's.
    auto shared = saved;
    shared[3].clear();
    check(give_per_pos(shared, dst, 848 * kPerPos), "restore accepts a shared-KV layer's empty plane");
    check(dst[3].empty(), "shared-KV layer is left empty");
}

void test_truncated_whole_plane_is_refused() {
    // The safety property: a snapshot whose GDN plane is short must be refused (the caller re-prefills), never
    // installed short, which would leave the GDN step reading past the end of the plane.
    auto rec_short = plane(kGdnLayers, 848 * kPerPos, 2.0f);
    auto rec_dst = plane(kGdnLayers, kRec, 0.0f);
    check(!give_whole(rec_short, rec_dst), "a truncated rec plane is refused");
    check(rec_dst[0] == std::vector<float>(kRec, 0.0f), "a refused restore leaves the plane untouched");
    // The short side matters as much: a 180-token snapshot's plane is *smaller* than the real one, and installing
    // it short is the silent-wrongness case (the GDN step would read past the end of it).
    auto rec_short_small = plane(kGdnLayers, 180 * kPerPos, 2.0f);
    check(!give_whole(rec_short_small, rec_dst), "a below-size rec plane is refused too");
    check(rec_dst[0] == std::vector<float>(kRec, 0.0f), "a below-size plane does not resize the destination");
    // Wrong layer count is refused too.
    auto few = plane(kGdnLayers + 1, kRec, 2.0f);
    check(!give_whole(few, rec_dst), "a plane set with more layers than the forward is refused");
}

void test_resume_point_is_one_token_short() {
    using onebit::npu::PrefixDecision;
    using onebit::npu::resume_point;
    PrefixDecision hit;
    hit.slot = 0;
    hit.cached_tokens = 180;              // the slot covers 180 tokens...
    check(resume_point(hit, 180) == 179, "a full hit resumes one token short of the snapshot");
    check(resume_point(hit, 400) == 179, "an extension resumes one token short of the slot, not of the prompt");
    check(resume_point(hit, 40) == 39, "a short prompt still resumes one token short of itself");
    PrefixDecision miss;
    check(resume_point(miss, 180) == 0 && resume_point(miss, 1) == 0, "a miss resumes nothing");
    PrefixDecision one;
    one.slot = 3;
    one.cached_tokens = 1;
    check(resume_point(one, 1) == 0, "a one-token hit has nothing to restore");
}

}  // namespace

int main() {
    test_old_planes_were_truncated();
    test_whole_planes_survive();
    test_kv_planes_still_per_position();
    test_truncated_whole_plane_is_refused();
    test_resume_point_is_one_token_short();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("q4nx_state_planes: all checks passed\n");
    return 0;
}
