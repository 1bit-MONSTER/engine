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

// The buffers the Qwen3.6-35B-A3B `lax` kernels read (docs/npu-lax.md), packed from a
// Q4NX container. Pure host code.
//
// The layout is the one the open kernels' recipe (third_party/OpenFlowLM-Next,
// open_kernels/recipes/qwen36moe.py pack_plan and recipes/pack.py, MIT) produces for this
// model with every projection at q4_1 (OPEN_KERNELS_FORCE_Q4_1, make_decode.py --requant),
// which is how the lax kernels are built. It is re-implemented here from that
// specification, not copied, and checked byte for byte against the Python packer's
// output (tests/npu_lax_test.cpp, docs/npu-lax.md "The C++ driver").
//
// Per layer the kernel takes
//   pool    512 MiB: the routed experts' up/gate stripes, their down slices, the shared
//           expert, then the attention projections (DeltaNet qkv + z gate, or q/k/v/o)
//   consts  norms, the router, the DeltaNet constants and (linear layers) ssm_out
// and the model-wide buffers are the q8 lm head pool, the final norm and the position
// table (ptab: per position [i32 valid | i32 nf | f32 cos @512 | f32 sin]).
//
// Every quantized chunk is a 32-row x 256-column tile. A q4_1 chunk (5120 B) is bf16
// d[g*32+r] @0, bf16 m[g*32+r] @512, then nibbles: nibble (r/16)*4096 + g*512 + i*16 + r%16
// holds column g*32+i of row r (even nibble in the low half of its byte). A q8 chunk
// (8704 B) is bf16 d[g*32+r] @0 and int8 codes @512 at the same index.
#pragma once

#include "model.h"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace onebit::npu::lax {

constexpr size_t kQ41 = 5120, kQ8 = 8704, kQ4K = 4736;

enum class Kind : uint8_t { Linear, Full };  // DeltaNet (lax_l) or full attention (lax_a)

// The model dimensions the kernels were built for, read from config.json and the
// container. Anything else is refused: the kernels are compiled for these.
struct Config {
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, rotary_dim = 0;
    int experts = 0, topk = 0, moe_inter = 0, shared_inter = 0;
    int lin_key_heads = 0, lin_value_heads = 0, lin_key_dim = 0, lin_value_dim = 0, conv_kernel = 0;
    int vocab = 0;  // embedding / lm head rows (248320; the tokenizer's ids stop at 248070)
    double rope_theta = 0;
    std::vector<Kind> kinds;

    static Config from_model(const Model& m, const std::string& dir);
};

// Buffer sizes of the lax program (open_kernels model/lax_decode_cfg.py).
constexpr size_t kPoolBytes = 512u << 20;
constexpr size_t kConstsBytes = 11882496;     // the buffer; a full-attention layer fills less
constexpr size_t kConstsFull = 1062912;
constexpr size_t kActBytes = 190464;
constexpr size_t kCfgBytes = 4096;
constexpr size_t kStateBytes = 2342912;       // DeltaNet recurrent + conv state
constexpr int kMaxContext = 4096;             // KV and ptab rows
constexpr size_t kKvRow = 2048;               // one position's [K | V]: 2 kv heads x 256 bf16 each
constexpr size_t kKvBytes = kKvRow * kMaxContext;
constexpr size_t kPtabRow = 1024;
constexpr size_t kPtabBytes = kPtabRow * kMaxContext;
constexpr size_t kLmPoolBytes = 542113792;
constexpr size_t kHidden = 2048;
constexpr size_t kLogitsBytes = 993280;       // f32 logits over the padded 248320 rows

// Where the MoE block lies in the pool (also what the on-device router retargets).
struct MoeLayout {
    size_t stripe = 163840;      // 128 rows of up (or gate) x 2048
    size_t up_bytes = 655360;    // one expert's up (= gate = down)
    size_t pool_down = 335544320;
    size_t share_up = 503316480, share_gate = 503971840, share_down = 504627200;
    size_t attn = 505282560;     // first byte after the shared expert
};

size_t consts_bytes(Kind k);

// One layer's pool; dst holds kPoolBytes and is fully written (zeros included).
void pack_pool(const Model& m, const Config& c, int layer, uint8_t* dst);
// One layer's consts; dst holds consts_bytes(kind) and is fully written.
void pack_consts(const Model& m, const Config& c, int layer, uint8_t* dst);
// The q8 lm head pool; dst holds kLmPoolBytes.
void pack_lmhead(const Model& m, uint8_t* dst);
// The final norm, bf16 [hidden].
void pack_norm(const Model& m, uint8_t* dst);
// rows position records of kPtabRow bytes.
void pack_ptab(const Config& c, int rows, uint8_t* dst);

// The chunk transforms, exposed for their unit tests.
// q8 -> q4_1 per 32-value block: m = min rounded down to bf16, d = (max - m) / 15 rounded
// up, nibble = trunc((v - m) / d + 0.5) clipped to 0..15, all in float32.
void requant_q8(const uint8_t* q8, uint8_t* q41);
// The signed-nibble form of a 5120-byte chunk (every min zero) as q4_1: m = -8d, nibble ^ 8.
void q4_0_to_q4_1(const uint8_t* src, uint8_t* dst);
bool is_signed_q4(const uint8_t* first_chunk);
// Pool chunk c of a std band [out, in_dim] reads source chunk std_src(c, in_dim).
size_t std_src(size_t c, size_t in_dim);
size_t down_src(size_t c);          // one expert's down slice (128 chunks)
size_t stripe_src(size_t c);        // inside one 128-row up/gate stripe (32 chunks)
size_t lmhead_src(size_t k, size_t hidden);

// SHA-256 (FIPS 180-4), for the byte-identity tables.
std::array<uint8_t, 32> sha256(std::span<const uint8_t> data);
std::string hex(std::span<const uint8_t> digest);

}  // namespace onebit::npu::lax
