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
// ffn_cache_policy_test: the size rule for the dense-tail cache's default (npu/forward/include/ffn_cache_policy.h).
// The two measured models are the fixtures: both must be on, and the one engine#1785 measured is the one the
// earlier 256 MB floor left off.

#include "ffn_cache_policy.h"

#include <cstdio>
#include <string>

using onebit::npu::ffn_cache_worth_it;
using onebit::npu::ffn_tail_bytes;
using onebit::npu::kFfnCacheFloorBytes;

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("FAIL %s\n", what.c_str());
        ++failures;
    }
}

int main() {
    const double MB = 1024.0 * 1024.0;
    check(ffn_tail_bytes(6144, 2048) == 3.0 * 6144 * 2048 * 4, "the tail is three IM x H f32 matrices");
    check(ffn_tail_bytes(17408, 5120) > 1000.0 * MB && ffn_cache_worth_it(17408, 5120), "Qwen3.8-27B (1.07 GB tail): on, as before");
    check(ffn_tail_bytes(6144, 2048) == 144.0 * MB, "Qwen3.5-2B's tail is 144 MiB (151 MB)");
    check(ffn_cache_worth_it(6144, 2048), "Qwen3.5-2B: on -- measured 485-520 -> 362-367 ms per token, same text (engine#1785)");
    check(ffn_tail_bytes(6144, 2048) < 256.0 * MB, "and it is the model the earlier 256 MB floor left off");
    // Below the smallest measured tail nothing is switched on unasked.
    check(!ffn_cache_worth_it(3584, 1024), "a 0.8B-class tail (42 MiB) stays off by default: not measured");
    check(!ffn_cache_worth_it(4096, 2048), "a 96 MiB tail stays off by default: not measured");
    check(kFfnCacheFloorBytes == 128.0 * MB, "the floor is 128 MiB");
    check(!ffn_cache_worth_it(5460, 2048) && ffn_cache_worth_it(5462, 2048), "the rule turns exactly at the floor (5461.33 x 2048)");
    check(!ffn_cache_worth_it(0, 2048) && !ffn_cache_worth_it(6144, 0) && !ffn_cache_worth_it(-6144, -2048),
          "a geometry that is not one is never on");
    if (failures == 0) std::printf("ffn_cache_policy: ok\n");
    return failures == 0 ? 0 : 1;
}
