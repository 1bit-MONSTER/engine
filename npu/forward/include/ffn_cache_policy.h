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
// ffn_cache_policy.h -- when the dense-tail cache is on by default (engine#587, engine#1785).
//
// On a hybrid (GDN) model the forward used to re-dequantise each layer's dense FFN tail on every token, even after the
// packed weights were resident on the device. The dense-tail cache skips that reload. Whether it is on when
// NPU_FFN_CACHE says nothing is a size rule, and the floor of that rule is set by what has been measured, not guessed:
//
//   model                          tail per layer   decode, cache off -> on            text
//   Qwen3.8-27B-NPU2  (IM 17408, H 5120)  1.07 GB   117.4 -> 104.5 s per 8-token request   identical (~/lb/587-cache-20261030)
//   Qwen3.5-2B-NPU2   (IM 6144,  H 2048)   151 MB   485-520 -> 362-367 ms per token        identical in 12 of 12 runs
//
// The second row is engine#1785's measurement (A B B A, one ticket; the off figure is the quieter off arm, the other
// was 478-707): the reload was about a fifth of every decode token on a model the old 256 MB floor left out. The floor is therefore just under the smallest tail measured.
// Below it nothing has been measured, so nothing is switched on unasked; NPU_FFN_CACHE=1 still forces it.

#pragma once

namespace onebit::npu {

// Bytes of f32 the three dense-tail matrices of one layer occupy (gate, up, down: IM x H each).
inline double ffn_tail_bytes(int intermediate, int hidden) {
    return 3.0 * double(intermediate) * double(hidden) * 4.0;
}

// The smallest tail the cache has been measured to pay for (see the table above).
inline constexpr double kFfnCacheFloorBytes = 128.0 * 1024.0 * 1024.0;

// The size half of the default: a tail at or above the floor. The caller adds the conditions that are about the
// run, not the model (a GDN model, the packed-weight cache on, not the host-GEMM mode).
inline bool ffn_cache_worth_it(int intermediate, int hidden) {
    return intermediate > 0 && hidden > 0 && ffn_tail_bytes(intermediate, hidden) >= kFfnCacheFloorBytes;
}

}  // namespace onebit::npu
