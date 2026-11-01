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
// `1bit npu-pack flm-head`: turn FASTFLOWLM'S MTP-head container into a Q4NX sidecar directory of ours.
//
// The head is one Qwen3.5 full-attention layer plus an input fusion, shipped by FastFlowLM as a
// safetensors-shaped container whose quantised tensors are rank-3 `[nrb, ncb, 4736]` and whose dtype string
// says "I8". That 4,736-byte cell is byte-for-byte our Q4_K chunk: u8 scale[g*32+n] at 0, u8 min[g*32+n] at
// 256, nibbles at 512 (byte k*16 + n/2, even n low), bf16 S[n] at 4608, bf16 M[n] at 4672, and
// w = q4 * (scale*S) + min*M -- measured bit-for-bit against FastFlowLM's own decoder on six tiles from six
// head matrices (evidence: amd-oss/npu-re, docs/FACTS-flm-q4nx-container-and-mtp-head.md §5). So this tool
// copies every tile unchanged: no requantisation, no zero point to fold, nothing to lose.
//
// Inputs are validated against that shape law and refused by name if they do not fit; the written container is
// re-read and every tile is decoded twice (from the source bytes and from the copy) and compared, so "the bytes
// are the same" is checked rather than asserted.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace onebit::npu::flm_head {

// The head's geometry, as the container's own shapes state it (FACTS §4.2 hard-asserts these for the 27B).
struct Dims {
    int hidden = 0, intermediate = 0;
    long long matrices = 0, norms = 0, tiles = 0, tile_bytes = 0;
};

// The form the head's weights are written in.
//   Q4K    the container's own cell, copied unchanged: byte-exact, no loss at all (our Q4_K chunk)
//   Q4_1   our mode-1 chunk (5,120 B: bf16 S[256], bf16 M[256], 4096 code bytes), the form the engine's
//          q4_1 kernels stream; the only loss is rounding S*scale and M*min to bf16, measured in the report
enum class Form { Q4K, Q4_1 };

struct Report {
    std::string out_dir;
    Dims dims;
    // Q4_1 form only: the bf16 factorisation error against this container's own Q4_K decode.
    long long values = 0, exactly_equal = 0;
    double max_abs = 0, rel_l1 = 0;
    // Per-tensor facts for the manifest: name, tile grid, bytes.
    std::vector<std::string> manifest;
    // The self-check: tiles re-decoded from the source and from the written container, compared bit for bit.
    long long checked_tiles = 0, differing_tiles = 0;
    // Anything that makes the container unrepresentable. Non-empty means nothing was written.
    std::vector<std::string> refused;
};

// Reads `container` (FastFlowLM's model.mtp.q4nx), validates it, writes `out_dir`/model.q4nx + config.json +
// manifest.json. Throws only on I/O; a container it cannot represent comes back in Report::refused.
Report import_head(const std::string& container, const std::string& out_dir, Form form = Form::Q4K);

// flm-head-import <container> -o <out dir> [--form q4_k|q4_1]
int run(int argc, char** argv);

}  // namespace onebit::npu::flm_head
