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
// `1bit npu-pack`: turn a Ryzen AI OGA NPU model pack (`*_rai_*_npu_*`, the
// `model.onnx` + `model.onnx.data` + `dd_metastate_*.fconst` + `genai_config.json`
// set AMD publishes) into the engine's Q4NX model directory. The pack's weights
// and its tokenizer/config contract are the model; every line of this reader is
// written from the documented facts in
// amd-oss/npu-re: docs/FACTS-oga-pack-weight-layout.md (pack-side layout) and
// docs/npu.md here (container side). No vendor library, wheel or compiled
// program is called, linked or shipped.
//
// Two commands:
//   npu-pack info   <pack dir>                 what the pack says, as JSON
//   npu-pack import <pack dir> -o <out dir>    write model.q4nx + config.json
//
// A pack the importer cannot represent is refused by name, with the reason and
// the tensor, and nothing is written: a container that silently carries 1 % of
// a model is worse than no container. `--allow-incomplete` writes what is
// representable and records the rest in the container's manifest, so the gap is
// visible to whatever loads it next.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace onebit::npu::pack_import {

// Is [offset, offset+length) inside a mapped file of `size` bytes? Both comparisons are overflow-proof —
// each is against a value already smaller than `size` — so a crafted `model.onnx` cannot wrap them.
// (`offset + length > size` does wrap: offset = UINT64_MAX - 4, length = 8 sums to 3.)
inline bool ext_range_ok(uint64_t size, uint64_t offset, uint64_t length) {
    return offset <= size && length <= size - offset;
}

// a*b without wrapping. Every read whose length comes from the file's own metadata (a shape product, N*K/block,
// N*(K/block)*block/2) is computed through this, so a crafted model.onnx cannot wrap a length into a small
// allocation and then fill it with the declared count.
inline bool mul_ok(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    out = a * b;
    return true;
}

// The model geometry the container's config.json needs.
struct Dims {
    int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, vocab = 0;
    float rope_theta = 0;
    // config.json's rms_norm_eps, and the two bytes it is stored as in the .fconst's block headers.
    // engine#734: the layer-block locator looked for bf16 1e-6 (86 35) only, which is what the Qwen
    // packs declare; Llama-3.2-1B declares 1e-5 (28 37) and so had no anchors at all.
    float rms_norm_eps = 1e-6f;
    unsigned short eps_bits = 0x3586;
    std::string model_type;
    std::vector<int> eos;
};

// One tensor the pack describes, and where its values would come from.
struct Planned {
    std::string name;     // Q4NX tensor name (model.layers.N.<...>.weight)
    std::string source;   // "model.onnx.data", "fconst:Llm_Token_Token", "fconst:Llm_Prefill", "config.json"
    std::string dtype;    // "q4_1" tiles, or "BF16"
    std::vector<int64_t> shape;
    std::string refusal;  // empty when importable; otherwise the named reason
    // What the pack's own cache/*_meta.json declares for this tensor (attrs N/K and the const
    // tensor's byte size), when it declares it. The pack stating its own geometry is a stronger
    // check on our plan than our own arithmetic, and it is what engine#734 needs to generalise
    // the .fconst offsets beyond hidden 896.
    int64_t declared_n = 0, declared_k = 0;
    std::string declared_check;  // empty when it agrees, else what disagrees
};

struct Plan {
    std::string dir;
    std::string pack;  // *_rai_1.8.0_npu_4K, from the directory name
    Dims dims;
    std::vector<Planned> tensors;
    // The attention payload's geometry, MEASURED from the .fconst rather than assumed: how many 64-row
    // tiles follow the block's first epsilon header, and the reason when that does not match the shape
    // this decoder was established on. Zero tiles plus a note is what the projections' refusal cites.
    // (engine#734's lesson: `info` must report the measurement, not the plan.)
    int64_t attn_tiles = 0;
    std::string attn_note;
    // Where the four projections actually sit, once the section count is known (engine#979). k/q/v are
    // always in the block's first payload at K = hidden. o_proj has a payload of its own whose K is
    // q_proj's output width -- the same payload as q/k/v on the Qwen2/Llama families, a separate one
    // behind the two per-head norms on the Qwen3 families, where the pack itself declares
    // `o_proj N=2560 K=4096` (q_out, not hidden). o_off is relative to the block start.
    int64_t o_off = 0;
    int o_k = 0;
    // Per-head QK RMSNorm gains (Qwen3 and later): their offsets inside the block, 0 when the family has
    // none. Identified and verified byte-exact against the public Qwen3-1.7B/4B checkpoints (0 of 128
    // values differ on either pack, and the swap differs on 127-128 of 128).
    int64_t knorm_off = 0, qnorm_off = 0;
    int64_t mlp_off = 0;  // the MLP section starts just past the block's post-attention norm
    // The MLP section's geometry, likewise measured: gate and up share one fused 4,288-byte-subvolume
    // tensor of `mlp_gu_units` x 128 columns, then a 64-byte parameter subvolume, then down_proj in
    // 2,176-byte subvolumes. All of it has to add up to the block stride exactly, which is the check
    // that keeps this from being an assumption about one pack.
    int64_t mlp_gu_units = 0, mlp_down_groups = 0;
    std::string mlp_note;

    std::vector<std::string> refused() const;
    std::string manifest() const;  // JSON: format, layout version, source, geometry, tensors
};

// Reads the pack (model.onnx, its external data, both .fconst headers, config.json).
// Throws only when the pack is structurally unreadable.
Plan plan(const std::string& pack_dir);

struct Options {
    bool allow_incomplete = false;  // write what is representable and name the rest
};

struct Report {
    std::string out_dir;
    int tensors = 0;
    long long tiles = 0;
    std::vector<std::string> refused;

    // Self-check 1: the 4-bit decoder against the pack's own independent data. With
    // tie_word_embeddings the pack's lm_head must reproduce its fp16 embedding, so the
    // decode is checked against the embedding, and two wrong decoders are run as controls.
    // An untied pack has no such reference and the numbers are left at 0 with
    // selfcheck_reference false -- a wrong number would be worse than none.
    bool selfcheck_reference = false;
    double lmhead_corr = 0, lmhead_rel = 0, ctrl_corr_nibble = 0, ctrl_corr_nozp = 0;

    // Self-check 2: the written container, read back, against the pack's dequantised values.
    double q4nx_rel = 0, q4nx_max_abs = 0, zero_exact_pct = 0;
    long long codes_exact = 0, codes_total = 0;

    // The .fconst layer-block geometry, measured from the RMSNorm epsilon anchors rather than assumed
    // (engine#734): the constants measured on one pack are functions of hidden size and of each
    // projection's N and K, so they only work for that pack. stride/second_header are 0 when the
    // anchors could not be resolved, with fconst_note saying why.
    int64_t stride = 0, second_header = 0;
    std::string fconst_note;
    int norm_anchors_ok = 0, norm_anchors_total = 0;
    // The final norm is the last tensor the .fconst tail carries. `final_norm_crosscheck` is only
    // meaningful when both facts below hold, so they are reported separately rather than inferred from it:
    // false means "no agreeing duplicate", which also covers "no duplicate to compare" and "the norm was
    // refused", and those are different things to say about an import.
    bool final_norm_imported = false;      // model.norm.weight was written into the container
    bool prefill_tail_seen = false;        // a second copy was readable in the prefill .fconst
    bool final_norm_crosscheck = false;    // ... and it is byte-identical to the Token_Token copy

    // Self-check 3: the attention tiles' own round trip. Decoding a shipped tile and re-encoding it has to
    // give the same 34,816 bytes back, every field at once (quants, zero points, scales and the writer's
    // bias slots) -- so it fails if any index, nibble order or field offset in decode_tile() is wrong.
    long long attn_tiles_ok = 0, attn_tiles_total = 0;
    int attn_projections = 0, attn_bias_tensors = 0;
    double attn_zero_inexact_pct = 0;  // per-32-group zero points that are not bf16-exact in the container

    // Self-check 4: the same round trip over the MLP section -- every 128-column gate/up unit and every
    // 64-column down_proj group decoded and re-encoded, which has to give the shipped bytes back.
    long long mlp_groups_ok = 0, mlp_groups_total = 0;
    int mlp_projections = 0;
    double mlp_zero_inexact_pct = 0;
};

Report import_pack(const std::string& pack_dir, const std::string& out_dir, const Options&);

// `1bit npu-pack <info|import> ...`; argv[0] is the subcommand.
int run(int argc, char** argv);

// Exposed for the tests: a decoded MatMulNBits tensor as (codes, scale, zero) per
// 32-column group, ready to tile.
struct Q4 {
    int N = 0, K = 0, block = 0;
    std::vector<uint8_t> codes;  // [N, K], 0..15
    std::vector<float> scale;    // [N, K/32]
    std::vector<float> zero;     // [N, K/32]
};
Q4 decode_matmulnbits(const uint8_t* qweight, const float* scales, const uint8_t* zeros,
                      int N, int K, int block, bool swap_nibbles = false, bool drop_zero = false);

// One 64-row tile of the .fconst attention payload, as the pack stores it. Established byte-exactly
// (amd-oss/npu-re docs/FACTS-oga-pack-weight-layout.md §7.13, §7.15): the tile is 2*(K/128)+2 subvolumes
// of 2,176 B = one pair per (column half, K block of 128) plus the pair holding the writer's bias input.
// Codes and zero points are stored as int4 (two's complement); `scale` is bf16 per (row, K block).
// Both narrow fields pair with rows the same way -- byte `i` of a subvolume carries rows 2i and 2i+1
// (low nibble = even row) -- so a per-row bug in one of them cannot hide behind the other, and a
// decode->encode round trip cannot see the pairing at all (engine#849).
//
// The same subvolume also carries the MLP section at a second width: `rows` is the group's column
// count, which fixes the subvolume's own width (rows/2 = 32 columns at 2,176 B, 64 at 4,288 B).
struct TileW {
    int k = 0;                  // the projection's K
    int rows = 64;              // columns covered: 64 in the attention payload and down_proj, 128 where
                                // gate and up are fused into one 4,288-byte subvolume
    std::vector<uint8_t> code;   // [rows][k] stored nibbles
    std::vector<float> scale;    // [rows][k/128], widened from bf16 (exact, and bf16 again on the way back)
    std::vector<uint8_t> zp;     // [rows][k/128] stored zero points
    std::vector<uint16_t> bias;  // [rows] bf16
};
// Bytes in one group of that layout: 2*(k/128) + 2 subvolumes, each 64*(rows/2) + 64 + rows bytes.
// 0 when k is not a positive multiple of 128 or rows is not a positive multiple of 64, i.e. when the
// payload cannot be groups of this shape.
int64_t group_bytes(int64_t k, int rows);
// Byte count below is the 64-row tile: the attention payload's and down_proj's shape.
// The layer-block geometry of a `.fconst`, measured from its epsilon anchors (engine#734). The anchors
// are the pack's OWN bf16 rms_norm_eps, so `eps_bits` comes from its config.json via
// `bf16_bits_of()`; the default is bf16 1e-6, which is what the Qwen family declares.
struct FconstGeom {
    int64_t stride = 0;   // bytes per layer block
    int64_t h1 = 0;       // offset of the second (post-attention) header within a block
    int period = 0;       // headers per block
    int blocks = 0;       // block starts seen
    // Every in-block header offset after the block's first (so hdr[0] == h1), relative to the block
    // start. These are the block's section boundaries: 1 header on the Qwen2/Llama packs (q/k/v/o share
    // one payload) and 3 on the Qwen3 packs (q/k/v, then one per-head QK norm each, then o_proj), so
    // which layout a pack uses is measured from the pack rather than guessed from a version.
    std::vector<int64_t> hdr;
    std::string why;      // non-empty when the geometry could not be established
};
unsigned short bf16_bits_of(float v);
FconstGeom locate_fconst(const std::string& path, unsigned short eps_bits = 0x3586);

int64_t attn_tile_bytes(int64_t k);
// Where a layer block's attention payload starts: the block's 64-byte header, then its
// input_layernorm weight as bf16 [hidden]. So it is `64 + 2 x hidden`, not a constant. It reads 1,856 on
// the 0.5B pack (hidden 896) and 4,160 on Llama-3.2-1B (hidden 2,048); reading it as the constant is what
// made every pack with a different hidden look like "whole tiles plus a stray run" (engine#979).
int64_t attn_payload_off(int64_t hidden);
TileW decode_tile(const uint8_t* tile, int k, int rows = 64);
// The exact inverse. Used only as the importer's self-check: a tile that does not survive
// encode(decode(t)) == t is a tile whose fields we do not understand, and it is refused by name.
std::vector<uint8_t> encode_tile(const TileW& t);
// A projection assembled from consecutive tiles, in the form the container's writer takes.
Q4 q_from_tiles(const std::vector<TileW>& tiles);
// gate and up are ONE tensor in the .fconst, interleaved in 32-column chunks, so a 128-column group holds
// 64 gate rows and 64 up rows: this takes one of the two as a 64-row tile (rows 0..31 from the group's
// first column half, rows 32..63 from the second).
TileW gu_half(const TileW& group, bool up);

// Q4NX q4_1 tiles of a Q4 tensor: [N/32 * K/256, 5120] bytes, plus the parameters
// that were not exactly representable in bf16 (counted, not fixed up).
void tile_q4_1(const Q4& q, std::vector<uint8_t>& out, long long* zero_inexact = nullptr);

}  // namespace onebit::npu::pack_import
