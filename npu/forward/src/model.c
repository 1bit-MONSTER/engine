// Copyright (c) 2026 bong-water-water-bong
#include "model.h"
#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#ifndef MIN64
#define MIN64(a,b) ((a)<(b)?(a):(b))
#endif
#include <math.h>

// ========= BF16 conversion helpers =========
float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t float_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    // Round to nearest even for BF16
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    uint32_t truncated = (bits + rounding_bias) >> 16;
    return (uint16_t)truncated;
}

// ========= Q4NX I8 format =========
// Q4NX stores I8 weights as raw int8 values with NO embedded scale factors.
// For inference, I8 values are dequantized using per-group absmax:
//   For each group of 32 I8 values:
//     scale = max(|I8_values|) / 127
//     BF16_val = I8_val * scale
// This converts [-128, 127] I8 range to [-absmax, absmax] BF16 range.
//
// FLM's reorder_cpy rearranges I8 data into NPU's blocked format.
// Our engine: convert I8→BF16 per group, then pack into [npu_block_rows, npu_block_cols] blocks.

// ========= NPU blocked format =========
//
// NPU expects weights in blocked format:
// For weight matrix [out_features, in_features]:
//   - Column-blocks of 1024 columns
//   - Row-blocks of 256 rows
//   - Each block = [min(256,out_rem), 1024] BF16 values in row-major
//   - Padded within a 1MB BO (second half zero padding)

// ========= Q4NX format =========
// Q4NX "I8" tensors are NOT raw BF16 bytes. Each 5120-byte I8 row is ONE
// torch2aie tile of [32 BF16 rows x 256 BF16 cols]:
//   [0..511]    256 bf16 scales  (GROUP-major: scales[g*32+lr], g = col/32, lr = tile row)
//   [512..1023] 256 bf16 zero points (Qwen3 is SUBTRACTIVE: W = (q - zp) * scale)
//   [1024..5119] packed int4: lane = lr/16, byte = lane*2048 + cc*8 + (lr%16)/2
//                low nibble = even tile row, UNSIGNED (q in [0,15])
// The tensor is a tile grid: n_tile_cols = in_features/256, n_tile_rows =
// i8_rows/n_tile_cols; logical rows = n_tile_rows*32, cols = n_tile_cols*256.
// The dequant must expand these tiles; reading pairs as BF16 (the old code)
// misinterprets scale/nibble bytes as weights and is numerically invalid.
// NPU blocking: each BO holds [npu_block_rows, npu_block_cols] BF16 values = 512KB data + 512KB zeros.
// Number of column blocks = ceil(cols / 2 / block_cols).

int npu_weight_num_blocks(const TensorDesc* desc, const ModelConfig* config,
                          int in_features) {
    if (desc->ndim != 2 || in_features <= 0) return 0;
    // Each I8 row is a 32x256 tile; the I8 row count spans the whole tile
    // grid, so logical_rows = i8_rows * 8192 / in_features.
    int64_t i8_rows = desc->shape[0];
    int64_t logical_rows = i8_rows * 8192 / in_features;
    int n_rb = (int)((logical_rows + config->npu_block_rows - 1) / config->npu_block_rows);
    // Full-width blocks for wide weights (FFN down: in=3072 > block_cols):
    // the K=3072 insts expect a [256, 3072] weight in ONE BO, so n_cb=1 and
    // the block holds the whole width (issue #2006 / NPU_GEMM_FIX.md — the
    // previous [256, 1024] slice layout made the down GEMM read 2048 columns
    // past the 1 MB BO -> intermittent aie2_set_cmd_timeout).
    int n_cb = (in_features > config->npu_block_cols) ? 1
               : (int)((in_features + config->npu_block_cols - 1) / config->npu_block_cols);
    return n_rb * n_cb;
}

// Read a bf16 at byte offset off of a tile row (explicit bytes: the q4nx
// tensor can start at an odd file offset, so a (uint16_t*) cast is UB).
static inline uint16_t q4nx_bf16(const uint8_t* p) {
    return (uint16_t)(p[0]) | ((uint16_t)(p[1]) << 8);
}

static inline uint16_t f32_to_bf16(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint32_t r = ((b >> 16) & 1) + 0x7FFF;
    return (uint16_t)((b + r) >> 16);
}

int npu_dequant_block(void* out, const void* in,
                       const TensorDesc* desc, const ModelConfig* config,
                       int block_idx, int in_features) {
    const int TILE_ROWS = 32, TILE_COLS = 256, TILE_BYTES = 5120;
    const int block_rows = config->npu_block_rows;   // 256
    const int block_cols = config->npu_block_cols;   // 1024
    int64_t i8_rows = desc->shape[0];
    int n_tile_cols = in_features / TILE_COLS;
    if (n_tile_cols <= 0) return 0;
    int64_t logical_rows = i8_rows * 8192 / in_features;
    int n_row_blocks = (int)((logical_rows + block_rows - 1) / block_rows);
    int n_col_blocks = (in_features > block_cols) ? 1
                       : (int)((in_features + block_cols - 1) / block_cols);
    int block_width = (in_features > block_cols) ? in_features : block_cols;
    int rb = block_idx / n_col_blocks;      // row block
    int cb = block_idx % n_col_blocks;      // col block
    if (rb >= n_row_blocks || cb >= n_col_blocks) return 0;
    int64_t row_start = (int64_t)rb * block_rows;
    int col_start = cb * block_width;
    int num_rows = (int)MIN64(logical_rows - row_start, block_rows);
    int num_cols = (int)MIN64(in_features - col_start, block_width);
    if (num_rows <= 0 || num_cols <= 0) return 0;

    const uint8_t* data = (const uint8_t*)in;
    uint16_t* bf16_out = (uint16_t*)out;
    memset(bf16_out, 0, (size_t)num_rows * block_width * 2);

    for (int r = 0; r < num_rows; r++) {
        int64_t lr_global = row_start + r;
        int tile_row = (int)(lr_global / TILE_ROWS);
        int lr = (int)(lr_global % TILE_ROWS);
        const uint8_t* row0 = data + (size_t)(tile_row * n_tile_cols) * TILE_BYTES;
        int lane = lr / 16;
        int byte_idx = (lr % 16) / 2;
        int nib = lr % 2;
        for (int c = 0; c < num_cols; c++) {
            int cc_global = col_start + c;
            int tile_col = cc_global / TILE_COLS;
            int cc = cc_global % TILE_COLS;
            int g = cc / 32;
            const uint8_t* row = row0 + (size_t)tile_col * TILE_BYTES;
            const uint8_t* packed = row + 1024 + (size_t)lane * (TILE_COLS * 8);
            // Qwen3 (unsigned) scale layout is GROUP-major: scales[g*32+lr]
            // (the Zaya signed converter is row-major scales[lr*8+g] instead —
            // verified empirically: g*32+lr dequantizes Qwen3 weights to the
            // plausible [-0.57, 0.64] range, lr*8+g to garbage ±1e3).
            int sc_off = (g * 32 + lr) * 2;
            float scale = bf16_to_float(q4nx_bf16(row + sc_off));
            float zp    = bf16_to_float(q4nx_bf16(row + 512 + sc_off));
            if (!isfinite(scale) || fabs(scale) > 100.0f) scale = 0.0f;
            if (!isfinite(zp) || fabs(zp) > 100.0f) zp = 0.0f;
            uint8_t b = packed[cc * 8 + byte_idx];
            int q = nib == 0 ? (b & 0x0F) : ((b >> 4) & 0x0F);
            // FastFlowLM Qwen3 q4nx formula — verified BIT-EXACT (maxdiff 0.0)
            // against the runtime's own q4nx_dequantize (libq4_npu_eXpress.so):
            //   W = (q - zp) * scale
            // with scales/zero-points bf16 at the GROUP-major index g*32+lr
            // (the torch2aie/zaya convention W = q*scale + zp does NOT match
            // this file — it mis-dequantizes every element).
            float w = ((float)q - zp) * scale;
            bf16_out[r * block_width + c] = f32_to_bf16(w);
        }
    }
    return num_rows * num_cols;
}

int npu_pack_weight_bo(uint8_t* bo_buffer, const void* in,
                        const TensorDesc* desc, const ModelConfig* config,
                        int block_idx, int in_features) {
    int bo_size = config->npu_weight_bo_size;
    memset(bo_buffer, 0, bo_size);
    
    int num_written = npu_dequant_block(bo_buffer, in, desc, config, block_idx, in_features);
    if (num_written < 0) return num_written;
    
    return 0;
}


// ===========================================================================
// Runtime-layout weight packer (issues #2006/#2015) — decoded byte-exact from
// the real FastFlowLM runtime's captured weight BOs (2026-09-01):
//
// Per-layer weight BO (10 MB = 1920 x 5120-B Q4NX tiles, layers in model
// order):
//   [0, 256)   q_proj tiles      G=8    (reorder group)
//   [256, 384) k_proj tiles      G=8
//   [384, 512) v_proj tiles      G=8
//   [512, 768) o_proj tiles      G=16
//   [768,1536) up/gate ALTERNATING 64-tile chunks: up0, gate0, up1, gate1...
//               each chunk reordered with G=8
//   [1536,1920) down_proj tiles  G=24
// Tile reorder within a group G (out[o] = in[G*(o/G) + (o/2)%(G/2) + (G/2)*(o%2)]):
//   G=8:  [0,4,1,5,2,6,3,7]  (q/k/v/gate/up)
//   G=16: [0,8,1,9,...,7,15] (o_proj)
//   G=24: stride 12           (down_proj)
// The mm/layer kernels DEQUANTIZE IN-KERNEL from these raw 5120-B tiles —
// the host never dequantizes (the old npu_dequant_block path is NOT the
// runtime layout).
// ===========================================================================
#define NPU_TILE_BYTES 5120


// Tiles in a projection tensor. shape[0] is the ROW count and shape[1] is the row width IN
// BYTES (a q4nx byte extent; for bf16 tensors shape[-1] counts elements, so the two must not
// be conflated). A tile is NPU_TILE_BYTES = 5120 B. Every model until Gemma3-1B had
// shape[1] == 5120, i.e. one row == one tile, so shape[0] was also the tile count and this
// returns exactly shape[0]. Gemma3-1B has shape[1] == 1280 -- a quarter-tile row -- where
// using shape[0] overstates the count 4x and every derived offset and BO size is wrong.
// THE SAME RULE MUST BE USED EVERYWHERE a tile count is derived from a tensor: the source
// read, the destination offsets, and the BO size. Fixing one of the three and not the others
// moves the fault rather than removing it -- which is how this was found.
int npu_desc_tiles(const TensorDesc* d) {
    if (!d || d->ndim != 2) return 0;
    long long rows = (long long)d->shape[0];
    long long rb   = (long long)d->shape[1];
    if (rows <= 0) return 0;
    if (rb > 0 && rb != NPU_TILE_BYTES)
        return (int)((rows * rb + NPU_TILE_BYTES - 1) / NPU_TILE_BYTES);
    return (int)rows;
}

static void npu_reorder_tiles(uint8_t* dst, const uint8_t* src, int n_tiles, int G) {
    // S is the HALF-GROUP length. It was G/2, which is integer division and therefore breaks
    // for an ODD G: with G=9 (Gemma3-1B, H=1152 -> 1152/128 = 9) the map o -> i was not a
    // permutation -- o=8 and o=0 both landed on source tile 0 -- so one tile was written twice
    // and another never, silently corrupting the weights.
    //
    // (G+1)/2 is identical to G/2 for every EVEN G, so this is a no-op for every model that
    // worked before (all of them have an even G on every projection), and a permutation for
    // odd G as well. Verified by exhaustive permutation check for G = 8, 16, 20, 24, 54, 84
    // (unchanged) and 9 (fixed).
    //
    // SECOND FIX (2026-09-14): that permutation check was run over the IN-GROUP domain
    // o in [0, G), where the map IS a permutation for odd G. The full map runs over
    // o in [0, n_tiles), and the in-group term was computed from the RAW o rather than
    // from o % G, so it kept growing past the group instead of cycling inside it.
    // For EVEN G that is harmless, because (o/2) % S + S*(o%2) is periodic with period
    // 2S = G, so raw-o and o%G agree exactly. For ODD G, 2S = G+1 != G, so they diverge:
    // at G=9, n=27 it emitted i = 27 = n_tiles (out of range) and a duplicate.
    // Using o % G makes the in-group term periodic with period G for every G. It is
    // BIT-IDENTICAL for every even G (checked for G = 8, 16, 54 over full and ragged
    // tails), so it is a provable no-op for every model that worked before, and for G=9,
    // n=27 the map becomes a clean permutation (0 duplicates, 0 out-of-range, max 26).
    //
    // Caveat, stated because it matters: this is the minimal rule that restores the necessary
    // permutation property, NOT a derivation of the vendor's layout -- the reorder was only
    // ever verified byte-exact for G=8 and G=16 (both powers of two; see the note on
    // npu_pack_layer_bo). Odd G needs a device run behind it before it is called correct.
    const int S = (G + 1) / 2;
    for (int o = 0; o < n_tiles; o++) {
        const int og = o % G;   // in-group position; raw `o` grows past the group for odd G
        int i = G * (o / G) + (og / 2) % S + S * (og % 2);
        memcpy(dst + (size_t)o * NPU_TILE_BYTES,
               src + (size_t)i * NPU_TILE_BYTES, NPU_TILE_BYTES);
    }
}

// Pack one projection's reordered tiles into the layer BO at `tile_offset`.
static void npu_pack_proj(uint8_t* bo, const TensorDesc* desc, ModelWeights* mw,
                          int tile_offset, int G, const char* name) {
    if (desc->ndim != 2) return;
    // shape[0] is the ROW count and shape[1] is the row width IN BYTES (this is a q4nx
    // byte extent, not an element count -- for bf16 tensors shape[-1] counts elements,
    // which is why the two must not be conflated).
    //
    // A tile is NPU_TILE_BYTES = 5120 B. Every model until Gemma3-1B had shape[1] == 5120,
    // i.e. one row == one tile, and shape[0] was therefore the tile count as well. Gemma3-1B's
    // tensors have shape[1] == 1280 -- a QUARTER-tile row -- so shape[0] (576 for q_proj) is
    // FOUR TIMES the tile count (144), and npu_reorder_tiles read 4x the tensor's bytes and
    // segfaulted. Deriving the count from the byte extent is IDENTICAL whenever
    // shape[1] == 5120 (checked for Qwen3-0.6B: 256*5120/5120 == 256) and correct otherwise.
    int n_tiles = npu_desc_tiles(desc);
    if (getenv("RT_PACK_DEBUG"))
        fprintf(stderr, "  pack %-5s n_tiles=%4d G=%3d off_tile=%5d ndim=%d shape0=%d shape1=%d\n",
                name ? name : "?", n_tiles, G, tile_offset,
                desc->ndim, desc->ndim > 0 ? (int)desc->shape[0] : -1,
                desc->ndim > 1 ? (int)desc->shape[1] : -1);
    const uint8_t* data = (const uint8_t*)model_tensor_data(mw, (TensorDesc*)desc);
    // A tensor can be 2-D and still have no data (absent from the bundle, or not mapped).
    // Without this the reorder memcpy's from NULL and the process segfaults, which reads
    // as a crash in the packing path with no statement of WHICH projection is missing.
    // Returning here keeps the fault in the caller, which knows the projection name.
    if (!data) {
        fprintf(stderr, "RuntimeLayer: pack_proj: tensor has no data (n_tiles=%d G=%d offset=%d)\n",
                n_tiles, G, tile_offset);
        return;
    }
    npu_reorder_tiles(bo + (size_t)tile_offset * NPU_TILE_BYTES, data, n_tiles, G);
}

// Pack a full layer (all 7 projections) into the runtime's per-layer BO layout.
// Geometry is DERIVED from the model dims (verified byte-exact vs the runtime
// for Qwen3-0.6B AND 1.7B):
//   reorder group G = K/128  (K = contraction dim; q/k/v/up/gate=H, o=NH*HD, down=IM)
//   gate/up alternating chunk CH = H/16 (== 8 * G_gateup)
//   tile offsets = cumulative tile counts (q, k, v, o, up/gate, down)
// Returns the total number of tiles written, or 0 on error.
int npu_pack_layer_bo(uint8_t* bo_buffer, ModelWeights* mw,
                      const ModelConfig* config, int layer_idx) {
    if (!bo_buffer || !mw || !config || layer_idx < 0 || layer_idx >= config->num_layers)
        return 0;
    LayerWeights* lw = &mw->layers[layer_idx];

    // Group counts are the number of 128-wide K-groups, i.e. ceil(K/128). Integer division
    // was used here, which silently TRUNCATES when the contraction dim is not a multiple of
    // 128 -- Gemma3-1B has IM=24864 and 24864 % 128 == 32, so G_d came out 194 instead of 195
    // and the last group was dropped. For every model whose dims are already aligned this is
    // exactly the old value, so the change is a no-op for all of them.
    const int G_h = (config->hidden_size + 127) / 128;                            // q/k/v/up/gate
    const int G_o = (config->num_attention_heads * config->head_dim + 127) / 128; // o_proj
    const int G_d = (config->intermediate_size + 127) / 128;                      // down_proj
    const int CH  = config->hidden_size / 16;                                    // 8 * G_h

    const int q_t  = npu_desc_tiles(&lw->q_proj_weight);
    const int k_t  = npu_desc_tiles(&lw->k_proj_weight);
    const int v_t  = npu_desc_tiles(&lw->v_proj_weight);
    const int o_t  = npu_desc_tiles(&lw->o_proj_weight);
    const int up_t = npu_desc_tiles(&lw->up_proj_weight);
    const int gate_t = npu_desc_tiles(&lw->gate_proj_weight);
    const int d_t  = npu_desc_tiles(&lw->down_proj_weight);

    // LFM2 hybrid: a short-conv layer has no q/k/v/o at all and carries its own block.
    const int sp_t = npu_desc_tiles(&lw->shortconv_in_proj_weight);
    const int so_t = npu_desc_tiles(&lw->shortconv_out_proj_weight);
    // G is the CONTRACTION-dim group count (see the rule above: G = K/128). The short-conv
    // in_proj is H -> 3H, so K = H and G must be H/128 -- writing 3H/128 here used the OUTPUT
    // dim instead, and the byte diff against FLM's own LFM2 weight BO showed exactly that: the
    // engine's conv-layer packing matched FLM through gate/up and down_proj (6,144 of 8,192
    // tiles, in order) and diverged at the short-conv block, which is the only block this G
    // touches.
    const int G_sp = (config->hidden_size + 127) / 128;  // in_proj:  K = H   -> G = H/128
    const int G_so = (config->hidden_size + 127) / 128;  // out_proj: K = H   -> G = H/128

    // LAYOUT, read off FLM's own BO (not inferred): for a conv layer the SHORT-CONV block comes
    // FIRST, then gate/up and down. Measured placements of the engine's blocks in FLM's BO:
    //   sp (1536 tiles) -> FLM 0..1535      so (512) -> FLM 1536..2047
    //   gu (4096)       -> FLM 2048..6143   d (2048) -> FLM 6144..8191
    // Every block's INTERNAL tile order is identical (sp's first eight land at 0..7, so's at
    // 1536..1543), so this is a pure block reordering -- which is also why the earlier version,
    // which appended the short-conv after down_proj, matched FLM for exactly the 6,144 tiles of
    // the gu+d prefix and diverged for the rest.
    // Attention layers have no short-conv, so their layout is unchanged: q,k,v,o,gu,d from 0.
    const int off_sp = 0;                        // shortconv.in_proj  (conv layers only)
    const int off_so = off_sp + sp_t;            // shortconv.out_proj
    const int off_q  = off_so + so_t;
    const int off_k  = off_q + q_t;
    const int off_v  = off_k + k_t;
    const int off_o  = off_v + v_t;
    const int off_gu = off_o + o_t;
    const int off_d  = off_gu + up_t + gate_t;
    const int total  = off_d + d_t;

    memset(bo_buffer, 0, (size_t)total * NPU_TILE_BYTES);

    npu_pack_proj(bo_buffer, &lw->q_proj_weight, mw, off_q, G_h, "q");
    npu_pack_proj(bo_buffer, &lw->k_proj_weight, mw, off_k, G_h, "k");
    npu_pack_proj(bo_buffer, &lw->v_proj_weight, mw, off_v, G_h, "v");
    npu_pack_proj(bo_buffer, &lw->o_proj_weight, mw, off_o, G_o, "o");

    // Short-conv block (LFM2 conv layers only; absent elsewhere so these no-op).
    if (sp_t > 0) npu_pack_proj(bo_buffer, &lw->shortconv_in_proj_weight,  mw, off_sp, G_sp, "scin");
    if (so_t > 0) npu_pack_proj(bo_buffer, &lw->shortconv_out_proj_weight, mw, off_so, G_so, "scout");

    // gate/up: alternating CH-tile chunks (up0, gate0, up1, gate1, ...)
    const uint8_t* up = (const uint8_t*)model_tensor_data(mw, &lw->up_proj_weight);
    const uint8_t* gate = (const uint8_t*)model_tensor_data(mw, &lw->gate_proj_weight);
    int n_chunks = (up_t + CH - 1) / CH;
    for (int c = 0; c < n_chunks; c++) {
        int up_n = (up_t - c * CH > CH) ? CH : up_t - c * CH;
        int gate_n = (gate_t - c * CH > CH) ? CH : gate_t - c * CH;
        int base = off_gu + c * 2 * CH;
        if (up_n > 0 && up)
            npu_reorder_tiles(bo_buffer + (size_t)base * NPU_TILE_BYTES,
                              up + (size_t)c * CH * NPU_TILE_BYTES, up_n, G_h);
        if (gate_n > 0 && gate)
            npu_reorder_tiles(bo_buffer + (size_t)(base + CH) * NPU_TILE_BYTES,
                              gate + (size_t)c * CH * NPU_TILE_BYTES, gate_n, G_h);
    }

    npu_pack_proj(bo_buffer, &lw->down_proj_weight, mw, off_d, G_d, "down");
    return total;
}

// Per-layer tile offsets (in 5120-byte Q4NX tiles) for the dequant weight_offset.
// Mirrors the off_* computation in npu_pack_layer_bo (byte-verified layout).
void npu_layer_tile_offsets(ModelWeights* mw, int layer_idx,
                            int* off_q, int* off_k, int* off_v, int* off_o,
                            int* off_gu, int* off_d) {
    if (off_q) *off_q = 0;
    LayerWeights* lw = &mw->layers[layer_idx];
    int q_t  = npu_desc_tiles(&lw->q_proj_weight);
    int k_t  = npu_desc_tiles(&lw->k_proj_weight);
    int v_t  = npu_desc_tiles(&lw->v_proj_weight);
    int o_t  = npu_desc_tiles(&lw->o_proj_weight);
    int up_t = npu_desc_tiles(&lw->up_proj_weight);
    int gate_t = npu_desc_tiles(&lw->gate_proj_weight);
    int oq = 0, ok = q_t, ov = q_t + k_t, oo = q_t + k_t + v_t;
    int ogu = oo + o_t, od = ogu + up_t + gate_t;
    if (off_q)  *off_q  = oq;
    if (off_k)  *off_k  = ok;
    if (off_v)  *off_v  = ov;
    if (off_o)  *off_o  = oo;
    if (off_gu) *off_gu = ogu;
    if (off_d)  *off_d  = od;
}

// Short-conv tile offsets within the per-layer BO (LFM2 hybrid conv layers only).
// Both are 0 on attention layers, where the shortconv tensors are absent. Additive
// deliberately: npu_layer_tile_offsets() keeps its 6-output signature.
void npu_layer_shortconv_offsets(ModelWeights* mw, int layer_idx, int* off_sp, int* off_so) {
    if (off_sp) *off_sp = 0;
    if (off_so) *off_so = 0;
    if (!mw || layer_idx < 0 || layer_idx >= mw->config.num_layers) return;
    LayerWeights* lw = &mw->layers[layer_idx];
    int q_t  = npu_desc_tiles(&lw->q_proj_weight);
    int k_t  = npu_desc_tiles(&lw->k_proj_weight);
    int v_t  = npu_desc_tiles(&lw->v_proj_weight);
    int o_t  = npu_desc_tiles(&lw->o_proj_weight);
    int up_t = npu_desc_tiles(&lw->up_proj_weight);
    int gate_t = npu_desc_tiles(&lw->gate_proj_weight);
    int d_t  = npu_desc_tiles(&lw->down_proj_weight);
    int sp_t = npu_desc_tiles(&lw->shortconv_in_proj_weight);
    const int od = q_t + k_t + v_t + o_t + up_t + gate_t + d_t;
    if (off_sp) *off_sp = od;
    if (off_so) *off_so = od + sp_t;
}

// Total per-layer weight BO bytes (all layers share the same geometry).
int npu_layer_bo_bytes(ModelWeights* mw, const ModelConfig* config) {
    if (!mw || !config) return 0;
    // Size the BO for the LARGEST layer, not layer 0. Hybrid models mix layer types:
    // LFM2-1.2B layer 0 is a gated short-conv layer with no q/k/v/o at all, while its
    // attention layers (2,5,8,10,12,14) have four more tensors. Sizing from layer 0
    // therefore under-allocates and npu_pack_layer_bo() writes past the end -- observed
    // as a SIGSEGV in __memset_avx512_unaligned_erms <- npu_pack_layer_bo <-
    // npu_bf16_pack_layer when first running LFM2. Taking the max over all layers is
    // correct for homogeneous models too (every layer is identical there).
    int tmax = 0;
    int nl = mw->config.num_layers;   // bound by the calloc'd layers[] array
    if (nl <= 0) nl = 1;
    for (int l = 0; l < nl; l++) {
        LayerWeights* lw = &mw->layers[l];
        int t = 0;
        t += npu_desc_tiles(&lw->q_proj_weight);
        t += npu_desc_tiles(&lw->k_proj_weight);
        t += npu_desc_tiles(&lw->v_proj_weight);
        t += npu_desc_tiles(&lw->o_proj_weight);
        t += npu_desc_tiles(&lw->up_proj_weight);
        t += npu_desc_tiles(&lw->gate_proj_weight);
        t += npu_desc_tiles(&lw->down_proj_weight);
        // The short-conv block also needs room when present (LFM2 conv layers).
        t += npu_desc_tiles(&lw->shortconv_in_proj_weight);
        t += npu_desc_tiles(&lw->shortconv_out_proj_weight);
        if (t > tmax) tmax = t;
    }
    return tmax * NPU_TILE_BYTES;
}

// ===========================================================================
// 35B MoE weight-BO packing — v0.9.46 layout (byte-verified 2026-09-18).
// ---------------------------------------------------------------------------
// The v0.9.46 runtime (lib md5 39a6c36a) weight BO is:
//   region A (expert pool) = 98304 tiles x 5120 B = 0x1E000000
//   region B               = 3456 tiles x 8704 B  = 0x1cb0000, base 0x1E000000
//   total                  = 0x1fcb0000 = 533,397,504 B
// Every tile is A/B-interleaved with its pair at 512-B chunk granularity, with
// the scale (and, for Q4NX, zero) blocks emitted before the payload chunks:
//
//   Q4NX expert tile (5120 B = 512 scale + 512 zeros + 4096 int4):
//     [scale_A 512][scale_B 512][zeros_A 512][zeros_B 512]
//     [int4_A0 512][int4_B0 512] .. [int4_A7 512][int4_B7 512]
//   Q8_0 region-B tile (8704 B = 512 scale + 8192 int8):
//     [scale_A 512][scale_B 512]
//     [int8_A0 512][int8_B0 512] .. [int8_A15 512][int8_B15 512]
//
// Byte-verified this session against a freshly regenerated vendor oracle
// (/tmp/lin5dump0946): region B 30081024/30081024, expert pool
// 503316480/503316480 (0 mismatches). The arg3 norms BO
// (npu_pack_moe_linear5_bo) is DERIVED from the ELF's arg3 read pattern
// (self-consistent, covers all reads) and has NO byte-exact oracle —
// lin5_b1/lin5_b2 are separate bf16 vendor buffers, not the arg3 BO.
// ===========================================================================
#define NPU_MOE_Q4NX_TILE 5120   // Q4NX expert tile bytes
#define NPU_MOE_Q80_TILE  8704   // Q8_0 region-B tile bytes

// Emit one A/B pair of Q4NX expert tiles (2 x 5120 = 10240 B) into dst.
static void moe_emit_q4nx_pair(uint8_t* dst, const uint8_t* A, const uint8_t* B) {
    memcpy(dst, A, 512);        dst += 512;   // scale_A
    memcpy(dst, B, 512);        dst += 512;   // scale_B
    memcpy(dst, A + 512, 512);  dst += 512;   // zeros_A
    memcpy(dst, B + 512, 512);  dst += 512;   // zeros_B
    for (int k = 0; k < 8; k++) {
        memcpy(dst, A + 1024 + k * 512, 512); dst += 512;   // int4_A chunk k
        memcpy(dst, B + 1024 + k * 512, 512); dst += 512;   // int4_B chunk k
    }
}

// Emit one A/B pair of Q8_0 region-B tiles (2 x 8704 = 17408 B) into dst.
static void moe_emit_q80_pair(uint8_t* dst, const uint8_t* A, const uint8_t* B) {
    memcpy(dst, A, 512); dst += 512;   // scale_A
    memcpy(dst, B, 512); dst += 512;   // scale_B
    for (int k = 0; k < 16; k++) {
        memcpy(dst, A + 512 + k * 512, 512); dst += 512;   // int8_A chunk k
        memcpy(dst, B + 512 + k * 512, 512); dst += 512;   // int8_B chunk k
    }
}

// A/B-interleave a Q8_0 tensor of n_tiles tiles (H = shape[1] tiles per row)
// into dst. Blocks of B=2H tiles: pair (blk*B+j) with (blk*B+j+H).
static void moe_pack_q80_interleave(uint8_t* dst, const uint8_t* src,
                                    int n_tiles, int H) {
    const int B = 2 * H;
    for (int blk = 0; blk < n_tiles / B; blk++) {
        for (int j = 0; j < H; j++) {
            const uint8_t* A  = src + (size_t)(blk * B + j)     * NPU_MOE_Q80_TILE;
            const uint8_t* Bt = src + (size_t)(blk * B + j + H) * NPU_MOE_Q80_TILE;
            moe_emit_q80_pair(dst, A, Bt);
            dst += 2 * NPU_MOE_Q80_TILE;
        }
    }
}

// Pack one linear layer's expert pool (up + gate + down) in the v0.9.46
// layout. up/gate: 1024 blocks of 32 tiles with window order
// j = base + 8*(i%4) + i/4, then A/B (A=even, B=odd reordered index). down:
// 2048 blocks of 16 tiles (two 8-window groups [0,2,4,6,1,3,5,7]), A/B.
// Writes 98304 x 5120 = 503,316,480 B.
int64_t npu_pack_moe_expert_pool(uint8_t* bo, ModelWeights* mw, int layer) {
    if (!bo || !mw || layer < 0 || layer >= mw->config.num_layers) return 0;
    LayerWeights* lw = &mw->layers[layer];
    if (lw->up_exps_weight.ndim == 0 || lw->gate_exps_weight.ndim == 0 ||
        lw->down_exps_weight.ndim == 0) return 0;
    const uint8_t* up   = (const uint8_t*)model_tensor_data(mw, &lw->up_exps_weight);
    const uint8_t* gate = (const uint8_t*)model_tensor_data(mw, &lw->gate_exps_weight);
    const uint8_t* down = (const uint8_t*)model_tensor_data(mw, &lw->down_exps_weight);
    if (!up || !gate || !down) return 0;

    uint8_t* dst = bo;
    // up + gate: 1024 alternating 32-tile blocks, A/B within each.
    for (int blk = 0; blk < 1024; blk++) {
        const int base = blk * 32;
        for (int p = 0; p < 16; p++) {
            int ia = base + 8 * ((2 * p) % 4)     + (2 * p) / 4;
            int ib = base + 8 * ((2 * p + 1) % 4) + (2 * p + 1) / 4;
            moe_emit_q4nx_pair(dst, up + (size_t)ia * NPU_MOE_Q4NX_TILE,
                                    up + (size_t)ib * NPU_MOE_Q4NX_TILE);
            dst += 2 * NPU_MOE_Q4NX_TILE;
        }
        for (int p = 0; p < 16; p++) {
            int ia = base + 8 * ((2 * p) % 4)     + (2 * p) / 4;
            int ib = base + 8 * ((2 * p + 1) % 4) + (2 * p + 1) / 4;
            moe_emit_q4nx_pair(dst, gate + (size_t)ia * NPU_MOE_Q4NX_TILE,
                                    gate + (size_t)ib * NPU_MOE_Q4NX_TILE);
            dst += 2 * NPU_MOE_Q4NX_TILE;
        }
    }
    // down: 2048 blocks of 16 tiles (8-window groups [0,2,4,6,1,3,5,7]).
    static const int DORD[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    for (int blk = 0; blk < 2048; blk++) {
        const int base = blk * 16;
        for (int p = 0; p < 8; p++) {
            int ia = base + ((2 * p)     / 8) * 8 + DORD[(2 * p)     % 8];
            int ib = base + ((2 * p + 1) / 8) * 8 + DORD[(2 * p + 1) % 8];
            moe_emit_q4nx_pair(dst, down + (size_t)ia * NPU_MOE_Q4NX_TILE,
                                    down + (size_t)ib * NPU_MOE_Q4NX_TILE);
            dst += 2 * NPU_MOE_Q4NX_TILE;
        }
    }
    return (int64_t)(dst - bo);   // 503,316,480
}

// Pack one linear layer's region-B weight content (share_* + qkv + gate_proj)
// in the v0.9.46 A/B-interleaved Q8_0 layout. Returns 3456 x 8704 = 30,081,024 B.
int64_t npu_pack_moe_region_b(uint8_t* bo, ModelWeights* mw, int layer) {
    if (!bo || !mw || layer < 0 || layer >= mw->config.num_layers) return 0;
    LayerWeights* lw = &mw->layers[layer];
    TensorDesc* tens[5] = { &lw->share_up_exps_weight, &lw->share_gate_exps_weight,
                            &lw->share_down_exps_weight, &lw->qkv_proj_weight,
                            &lw->self_attn_gate_proj_weight };
    uint8_t* dst = bo;
    for (int i = 0; i < 5; i++) {
        if (tens[i]->ndim == 0) return 0;   // all five must exist (linear layer)
        const uint8_t* d = (const uint8_t*)model_tensor_data(mw, tens[i]);
        if (!d) return 0;
        // shape[0] = tile rows, shape[1] = tiles per row (the A/B H), shape[2] =
        // tile bytes. n_tiles = shape[0] * shape[1].
        int n_tiles = (int)(tens[i]->shape[0] * tens[i]->shape[1]);
        int H = (int)tens[i]->shape[1];
        moe_pack_q80_interleave(dst, d, n_tiles, H);
        dst += (size_t)n_tiles * NPU_MOE_Q80_TILE;
    }
    return (int64_t)3456 * NPU_MOE_Q80_TILE;   // 30,081,024
}

// Pack one FULL-ATTENTION layer's region-B weight content in the v0.9.46
// A/B-interleaved Q8_0 layout. The full-attn layers (model indices 3,7,...,39)
// have self_attn.q/k/v/o_proj instead of linear_attn.qkv_proj/ssm_out_proj, and
// the vendor (load_attn_weights) lays them out as:
//   share_up [16,8,8704]  H=8   128 tiles   @ tile 0
//   share_gate [16,8,8704] H=8  128 tiles   @ tile 128
//   share_down [64,2,8704] H=2  128 tiles   @ tile 256
//   q_proj rows 0..127     H=8   1024 tiles  @ tile 384   (first half)
//   k_proj [16,8,8704]    H=8   128 tiles   @ tile 1408
//   v_proj [16,8,8704]    H=8   128 tiles   @ tile 1536
//   q_proj rows 128..255   H=8   1024 tiles  @ tile 1664  (second half)
//   o_proj [64,16,8704]   H=16  1024 tiles  @ tile 2688
// Total 3712 x 8704 = 32,309,248 B. Byte-exact vs /tmp/attndump0946 oracle.
int64_t npu_pack_moe_region_b_full(uint8_t* bo, ModelWeights* mw, int layer) {
    if (!bo || !mw || layer < 0 || layer >= mw->config.num_layers) return 0;
    LayerWeights* lw = &mw->layers[layer];
    const uint8_t* q = (const uint8_t*)model_tensor_data(mw, &lw->q_proj_weight);
    if (!q || lw->q_proj_weight.ndim == 0) return 0;   // must be a full-attn layer
    const int q_tiles = (int)(lw->q_proj_weight.shape[0] * lw->q_proj_weight.shape[1]);
    if (q_tiles != 2048) return 0;
    const size_t T = NPU_MOE_Q80_TILE;
    uint8_t* dst = bo;
    const uint8_t* d;
    d = (const uint8_t*)model_tensor_data(mw, &lw->share_up_exps_weight);
    if (!d) return 0; moe_pack_q80_interleave(dst, d, 128, 8); dst += 128 * T;
    d = (const uint8_t*)model_tensor_data(mw, &lw->share_gate_exps_weight);
    if (!d) return 0; moe_pack_q80_interleave(dst, d, 128, 8); dst += 128 * T;
    d = (const uint8_t*)model_tensor_data(mw, &lw->share_down_exps_weight);
    if (!d) return 0; moe_pack_q80_interleave(dst, d, 128, 2); dst += 128 * T;
    moe_pack_q80_interleave(dst, q, 1024, 8); dst += 1024 * T;   // q rows 0..127
    d = (const uint8_t*)model_tensor_data(mw, &lw->k_proj_weight);
    if (!d) return 0; moe_pack_q80_interleave(dst, d, 128, 8); dst += 128 * T;
    d = (const uint8_t*)model_tensor_data(mw, &lw->v_proj_weight);
    if (!d) return 0; moe_pack_q80_interleave(dst, d, 128, 8); dst += 128 * T;
    moe_pack_q80_interleave(dst, q + 1024 * T, 1024, 8); dst += 1024 * T;   // q rows 128..255
    d = (const uint8_t*)model_tensor_data(mw, &lw->o_proj_weight);
    if (!d) return 0; moe_pack_q80_interleave(dst, d, 1024, 16); dst += 1024 * T;
    return (int64_t)(3712 * T);   // 32,309,248
}

// Pack one FULL-ATTENTION layer's norms BO (the v0.9.46 load_attn_weights b2
// buffer): 128 zero bytes, then q_norm [256] BF16 (512 B), then k_norm [256]
// BF16 (512 B). 1152 B total. Byte-exact vs /tmp/attndump0946/attn_b2_L3.bin.
int64_t npu_pack_moe_attn_norms_bo(uint8_t* bo, ModelWeights* mw, int layer) {
    if (!bo || !mw || layer < 0 || layer >= mw->config.num_layers) return 0;
    LayerWeights* lw = &mw->layers[layer];
    if (lw->q_norm_weight.ndim == 0 || lw->k_norm_weight.ndim == 0) return 0;
    const uint8_t* qn = (const uint8_t*)model_tensor_data(mw, &lw->q_norm_weight);
    const uint8_t* kn = (const uint8_t*)model_tensor_data(mw, &lw->k_norm_weight);
    if (!qn || !kn) return 0;
    memset(bo, 0, 1152);
    memcpy(bo + 128, qn, 512);
    memcpy(bo + 640, kn, 512);
    return 1152;
}

// Pack one layer's router BO (arg-2): input_layernorm @0, post_attention_layernorm
// @0x1000, shared_expert_gate @0x2000 (BF16), moe_router @0x3000 (BF16 [H,
// N_EXPERTS], e-major transpose). The layernorm offsets are the best-effort
// region-A placement (R37 desc: shared_gate @0x2000, moe_router @0x3000; the
// arg-2 BD decode shows a 3072-B read @0 — the layernorms).
int64_t npu_pack_moe_router_bo(uint8_t* bo, ModelWeights* mw, int layer) {
    if (!bo || !mw || layer < 0 || layer >= mw->config.num_layers) return 0;
    LayerWeights* lw = &mw->layers[layer];
    if (lw->moe_router_weight.ndim != 2) return 0;
    const uint8_t* rt = (const uint8_t*)model_tensor_data(mw, &lw->moe_router_weight);
    const uint8_t* seg = (const uint8_t*)model_tensor_data(mw, &lw->shared_expert_gate_weight);
    const uint8_t* iln = (const uint8_t*)model_tensor_data(mw, &lw->input_layernorm_weight);
    const uint8_t* paln = (const uint8_t*)model_tensor_data(mw, &lw->post_attention_layernorm_weight);
    if (!rt || !seg) return 0;
    const int64_t n_in = lw->moe_router_weight.shape[0];      // H = 2048
    const int64_t n_out = lw->moe_router_weight.shape[1];     // N_EXPERTS = 256
    const size_t seg_bytes = (size_t)lw->shared_expert_gate_weight.data_size;
    memset(bo, 0, 0x3000);
    if (iln && lw->input_layernorm_weight.ndim == 1)
        memcpy(bo + 0x0000, iln, (size_t)lw->input_layernorm_weight.data_size);
    if (paln && lw->post_attention_layernorm_weight.ndim == 1)
        memcpy(bo + 0x1000, paln, (size_t)lw->post_attention_layernorm_weight.data_size);
    if (seg_bytes) memcpy(bo + 0x2000, seg, seg_bytes);
    // e-major TRANSPOSE (addendum 155): the layer ELF reads the router at
    // @12288 as dst[e*2048 + h] = router[h][e], i.e. transposed [256 experts]
    // [2048 hidden], read [2048h x 32e] per DMA.
    const uint16_t* src = (const uint16_t*)rt;
    uint16_t* dst = (uint16_t*)(bo + 0x3000);
    // The raw moe_router.weight is plain row-major [H=2048, N=256]. The old
    // stride-8 interleave (dst[(i%8)*65536 + j*256 + i/8]) was a guess and
    // contradicts the layer ELF's e-major transpose read. Verified against the
    // reference FFN: plain transpose reproduces the reference top-8
    // [91,160,46,153,22,131,33,121] with probs [.0263,.0248,.0171,.0168,.0156,
    // .0152,.0150,.0143]; the stride-8 interleave gives [228,131,175,5,16,54,
    // 91,2] — wrong expert selection, wrong routed FFN, ~0 correlation.
    for (int64_t j = 0; j < n_out; j++)
        for (int64_t i = 0; i < n_in; i++)
            dst[j * n_in + i] = src[i * n_out + j];
    return (int64_t)(0x3000 + n_in * n_out * 2);   // 0x3000 + 1 MB
}

// Pack one linear layer's v0.9.46 norms BO (9,241,088 B): 328,192-B head
// (ssm_conv1d, ssm_norm, ssm_a, ssm_dt.bias, ssm_alpha_proj, ssm_beta_proj,
// plain concat) then ssm_out (1024 x 8704-B Q8_0 tiles, A/B H=16).
int64_t npu_pack_moe_linear5_bo(uint8_t* bo, ModelWeights* mw, int layer) {
    if (!bo || !mw || layer < 0 || layer >= mw->config.num_layers) return 0;
    LayerWeights* lw = &mw->layers[layer];
    if (lw->ssm_conv1d_weight.ndim == 0 || lw->ssm_out_proj_weight.ndim == 0) return 0;
    const size_t BO = 9241088;   // 328,192 head + 1024 x 8704 ssm_out
    memset(bo, 0, BO);
    uint8_t* dst = bo;

    TensorDesc* head[6] = { &lw->ssm_conv1d_weight, &lw->ssm_norm_weight,
                            &lw->ssm_a,             &lw->ssm_dt_bias,
                            &lw->ssm_alpha_proj_weight, &lw->ssm_beta_proj_weight };
    for (int h = 0; h < 6; h++) {
        if (head[h]->ndim == 0) return 0;   // all six must exist on a linear layer
        const uint8_t* s = (const uint8_t*)model_tensor_data(mw, head[h]);
        if (!s) return 0;
        size_t n = (size_t)head[h]->data_size;
        memcpy(dst, s, n);
        dst += n;
    }

    const uint8_t* ssm = (const uint8_t*)model_tensor_data(mw, &lw->ssm_out_proj_weight);
    if (!ssm) return 0;
    // ssm_out [64,16,8704] = 1024 tiles, A/B interleave H = 16.
    moe_pack_q80_interleave(dst, ssm, 1024, 16);
    return (int64_t)BO;
}

// Pack the lm_head weight (tied embedding) into the runtime's 98,566,144 B BO.
// The q4nx stores lm_head as [18992 tiles x 5120B] (8 vocab rows per tile);
// the runtime BO = the same tiles reordered with G=8 (npu_reorder_tiles) —
// byte-verified against the captured runtime lm_head BO (Round 36).
// The tensor's own data_offset (metadata, relative to data_base) points at
// the physical data. Returns bytes written or 0 on error.
#ifdef __cplusplus
extern "C"
#endif
int npu_pack_lmhead_bo(uint8_t* bo_buffer, ModelWeights* mw, const ModelConfig* config) {
    (void)config;
    if (!bo_buffer || !mw || mw->lm_head_weight.ndim != 2) return 0;
    const int TILE = 5120;
    // SAME RULE AS THE LAYER PACKING: shape[0] is a ROW count and shape[1] is a row width IN
    // BYTES, so a row is one tile only when shape[1] == 5120. Gemma3-1B's lm_head tensor has
    // shape[1] == 1280, which made this read 4x the tensor and fault -- the same defect as
    // npu_pack_layer_bo, in the lm_head path. The caller sizes the BO with the same helper, so
    // the two stay consistent.
    int n_tiles = npu_desc_tiles(&mw->lm_head_weight);
    if (n_tiles <= 0) return 0;
    const uint8_t* data = (const uint8_t*)model_tensor_data(mw, &mw->lm_head_weight);
    if (!data) return 0;
    npu_reorder_tiles(bo_buffer, data, n_tiles, config->hidden_size / 128);
    return n_tiles * TILE;
}

// ========= Simple JSON Parser =========

static int parse_json_metadata(const uint8_t* json_data, uint64_t json_len,
                                TensorDesc* tensors, int max_tensors);
static int find_tensor(const char* name, TensorDesc* tensors, int count);
// find a per-layer tensor by name with both layer namings
static int find_layer_tensor(const char* name_plural, const char* name_singular,
                             TensorDesc* tensors, int count, TensorDesc* out) {
    int idx = find_tensor(name_plural, tensors, count);
    if (idx < 0 && name_singular) idx = find_tensor(name_singular, tensors, count);
    if (idx >= 0) { memcpy(out, &tensors[idx], sizeof(TensorDesc)); return 1; }
    return 0;
}

// ========= Model Loader =========

ModelWeights* model_load(const char* path, ModelConfig config) {
    ModelWeights* mw = calloc(1, sizeof(ModelWeights));
    if (!mw) return NULL;
    
    memcpy(&mw->config, &config, sizeof(config));
    
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("Cannot open model file: %s", path);
        free(mw);
        return NULL;
    }
    
    struct stat st;
    fstat(fd, &st);
    mw->file_size = st.st_size;
    
    mw->file_data = mmap(NULL, mw->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mw->file_data == MAP_FAILED) {
        LOG_ERROR("mmap failed: %s", strerror(errno));
        free(mw);
        return NULL;
    }
    
    uint64_t header_size;
    memcpy(&header_size, mw->file_data, 8);
    // Tensor offsets are relative to the data section that follows the 8-byte
    // length prefix and the JSON header.
    mw->data_base = 8 + header_size;
    
    LOG_INFO("Model file: %s (%lu MB)", path, (unsigned long)(mw->file_size / 1024 / 1024));
    LOG_INFO("Header: %lu bytes JSON", (unsigned long)header_size);
    
    const char* json_start = (const char*)(mw->file_data + 8);
    size_t json_len = header_size;
    
    // Tensor index cap.  Gemma4-E2B has 709 tensors (the whole model is 4455 MB),
    // so the old 512 silently truncated the index: layers whose tensors sat past
    // entry 512 were absent, and load_layer() then failed with "dequant failed"
    // part-way through the FIRST step.  Raised, and the cap now reports itself.
    int max_tensors = 4096;
    TensorDesc* tensors = calloc(max_tensors, sizeof(TensorDesc));
    int num_tensors = parse_json_metadata((const uint8_t*)json_start, json_len,
                                           tensors, max_tensors);
    
    LOG_INFO("Found %d tensors in metadata", num_tensors);
    
    // Embed tokens
    int idx_emb = find_tensor("model.embed_tokens.weight", tensors, num_tensors);
    // LFM2 names the embedding model.token_embd.weight (everything else it names the
    // canonical way), so accept both.
    if (idx_emb < 0) idx_emb = find_tensor("model.token_embd.weight", tensors, num_tensors);
    if (idx_emb >= 0) memcpy(&mw->embed_tokens, &tensors[idx_emb], sizeof(TensorDesc));

    // Final RMSNorm (model.norm.weight), applied to the last hidden state before
    // the lm_head (host logits path needs it).
    int idx_norm = find_tensor("model.norm.weight", tensors, num_tensors);
    if (idx_norm >= 0) memcpy(&mw->norm_weight, &tensors[idx_norm], sizeof(TensorDesc));
    
    // ── Derive the actual model config from the parsed tensors, overriding
    //    the hardcoded 0.6B profile when they disagree (40-layer MoE, etc.).
    {
        int max_layer = -1;
        for (int t = 0; t < num_tensors; t++) {
            const char* n = tensors[t].name;
            if (!n || !strstr(n, ".input_layernorm.weight")) continue;
            const char* p = strstr(n, "model.layer");
            if (!p) continue;
            p += strlen("model.layer");
            if (*p == 's') p++;          // "model.layers." vs "model.layer."
            if (*p == '.') p++;
            int ln = atoi(p);
            if (ln > max_layer) max_layer = ln;
        }
        if (max_layer >= 0) {
            int derived = max_layer + 1;
            if (derived != config.num_layers) {
                LOG_INFO("Derived %d layers from metadata (was %d)", derived, config.num_layers);
                config.num_layers = derived;
                mw->config = config;
            }
        }
    }
    if (mw->embed_tokens.shape[0] > 0 &&
        ((int)mw->embed_tokens.shape[0] != config.vocab_size ||
         (int)mw->embed_tokens.shape[1] != config.hidden_size)) {
        LOG_INFO("Derived vocab=%lld hidden=%lld from embed_tokens",
                 (long long)mw->embed_tokens.shape[0], (long long)mw->embed_tokens.shape[1]);
        config.vocab_size = (int)mw->embed_tokens.shape[0];
        config.hidden_size = (int)mw->embed_tokens.shape[1];
        mw->config = config;
    }
    
    // Allocate per-layer weights
    mw->layers = calloc(config.num_layers, sizeof(LayerWeights));
    
    char name_buf[128];
    char name_buf2[128];
    for (int l = 0; l < config.num_layers; l++) {
        LayerWeights* layer = &mw->layers[l];
        
                snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.input_layernorm.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.input_layernorm.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->input_layernorm_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.post_attention_layernorm.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.post_attention_layernorm.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->post_attention_layernorm_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_norm.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.q_norm.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->q_norm_weight);

        // ---- LFM2 gated short convolution (present only on conv layers) ------
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.shortconv.in_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.shortconv.in_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors,
                          &layer->shortconv_in_proj_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.shortconv.conv.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.shortconv.conv.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors,
                          &layer->shortconv_conv_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.shortconv.out_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.shortconv.out_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors,
                          &layer->shortconv_out_proj_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_norm.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.k_norm.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->k_norm_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.q_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->q_proj_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.k_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->k_proj_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.v_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.v_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->v_proj_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.q_proj.bias", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.q_proj.bias", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->q_proj_bias);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.k_proj.bias", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.k_proj.bias", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->k_proj_bias);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.v_proj.bias", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.v_proj.bias", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->v_proj_bias);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.o_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.o_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->o_proj_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.gate_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.gate_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->gate_proj_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.up_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.up_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->up_proj_weight);
        
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.down_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.down_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->down_proj_weight);

        // ---- MoE routed + shared expert tensors ----
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.up_exps_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.up_exps_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->up_exps_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.gate_exps_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.gate_exps_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->gate_exps_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.down_exps_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.down_exps_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->down_exps_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.share_up_exps_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.share_up_exps_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->share_up_exps_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.share_gate_exps_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.share_gate_exps_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->share_gate_exps_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.mlp.share_down_exps_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.mlp.share_down_exps_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->share_down_exps_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.moe_router.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.moe_router.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->moe_router_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.shared_expert_gate.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.shared_expert_gate.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->shared_expert_gate_weight);

        // ---- linear-attn (GateDeltaNet) tensors ----
        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.self_attn.gate_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.self_attn.gate_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->self_attn_gate_proj_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.qkv_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.linear_attn.qkv_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->qkv_proj_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.ssm_out_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.linear_attn.ssm_out_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->ssm_out_proj_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.ssm_conv1d.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.linear_attn.ssm_conv1d.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->ssm_conv1d_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.ssm_norm.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.linear_attn.ssm_norm.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->ssm_norm_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.ssm_a", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.linear_attn.ssm_a", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->ssm_a);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.ssm_dt.bias", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.linear_attn.ssm_dt.bias", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->ssm_dt_bias);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.ssm_alpha_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.linear_attn.ssm_alpha_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->ssm_alpha_proj_weight);

        snprintf(name_buf, sizeof(name_buf),
                 "model.layers.%d.linear_attn.ssm_beta_proj.weight", l);
        snprintf(name_buf2, sizeof(name_buf2),
                 "model.layer.%d.linear_attn.ssm_beta_proj.weight", l);
        find_layer_tensor(name_buf, name_buf2, tensors, num_tensors, &layer->ssm_beta_proj_weight);
    }
    
    // Final norm
    int idx_fn = find_tensor("model.norm.weight", tensors, num_tensors);
    if (idx_fn >= 0) memcpy(&mw->norm_weight, &tensors[idx_fn], sizeof(TensorDesc));
    
    // LM head
    int idx_lm = find_tensor("lm_head.weight", tensors, num_tensors);
    if (idx_lm >= 0) memcpy(&mw->lm_head_weight, &tensors[idx_lm], sizeof(TensorDesc));
    
    mw->all_tensors = tensors;
    mw->all_count = num_tensors;
    
    LOG_INFO("Model loaded: %d tensors, %d layers", num_tensors, config.num_layers);
    return mw;
}

void model_free(ModelWeights* mw) {
    if (!mw) return;
    free(mw->all_tensors);
    if (mw->file_data) munmap(mw->file_data, mw->file_size);
    free(mw->layers);
    free(mw);
}

TensorDesc* model_tensor_by_name(ModelWeights* mw, const char* name) {
    if (!mw || !mw->all_tensors || !name) return NULL;
    for (int i = 0; i < mw->all_count; i++) {
        const char* n = mw->all_tensors[i].name;
        if (n[0] && strcmp(n, name) == 0) return &mw->all_tensors[i];
    }
    return NULL;
}

void* model_tensor_data(ModelWeights* mw, TensorDesc* desc) {
    if (!mw || !desc) return NULL;
    // data_offsets in the JSON metadata are relative to the tensor-data start
    // (right after the 8-byte length + JSON header). Without data_base every
    // tensor is read header_len bytes early (garbage scales/nibbles).
    return mw->file_data + mw->data_base + desc->data_offset;
}

int model_find_tensor(const char* name, ModelWeights* mw) {
    (void)name;
    (void)mw;
    return -1;
}

static int parse_json_metadata(const uint8_t* json_data, uint64_t json_len,
                                TensorDesc* tensors, int max_tensors) {
    const char* s = (const char*)json_data;
    uint64_t len = json_len;
    int count = 0;
    
    const char* p = s;
    const char* end = s + len;
    
    while (p < end && count < max_tensors) {
        while (p < end && *p != '"') p++;
        if (p >= end) break;
        
        const char* key_start = p + 1;
        const char* key_end = key_start;
        while (key_end < end && *key_end != '"') key_end++;
        if (key_end >= end) break;
        
        ptrdiff_t key_len = key_end - key_start;
        const char* key_str = key_start;
        
        /* A tensor entry is any key whose value object carries a "dtype" (or a
         * "...weight" name).  The plain decoder uses exact ".weight" keys and
         * Gemma4 adds "weight_layer{N}" / "weight.scale"; a GDN (qwen3_5) model
         * adds `ssm_a` and `ssm_dt.bias`, which have NO ".weight" at all, so a
         * name-only filter silently drops them and the forward then reports
         * "no ssm_a/ssm_dt".  The dtype check keeps every real tensor reachable
         * through model_tensor_by_name() without keying on a model name. */
        bool is_tensor = 0;
        for (const char* c = key_str; c + 7 <= key_end; c++) {
            if (memcmp(c, ".weight", 7) == 0) { is_tensor = 1; break; }
        }
        if (!is_tensor) {
            /* The value must be an object IMMEDIATELY after the key: scan for the
             * next '{' but stop at a '}' first.  Without the '}' stop, the keys
             * `dtype` / `shape` / `data_offsets` (whose values are a string / an
             * array) would look ahead into the NEXT entry's object, see its
             * "dtype", and be accepted as tensors -- which inflated Gemma4-E4B's
             * index to the 4096 cap and truncated away
             * `model.per_layer_token_embd.weight` (E2B: 601 -> 3541 entries). */
            const char* brace = key_end;
            while (brace < end && *brace != '{' && *brace != '}') brace++;
            if (brace < end && *brace == '{') {
                const char* close = brace;
                while (close < end && *close != '}') close++;
                for (const char* c = brace; c + 7 <= close; c++) {
                    if (memcmp(c, "\"dtype\"", 7) == 0) { is_tensor = 1; break; }
                }
            }
        }

        if (!is_tensor) {
            p = key_end + 1;
            continue;
        }
        
        TensorDesc* t = &tensors[count];
        memset(t, 0, sizeof(TensorDesc));
        
        int name_len = key_len < (int)sizeof(t->name) - 1 ? key_len : (int)sizeof(t->name) - 1;
        memcpy(t->name, key_str, name_len);
        t->name[name_len] = '\0';
        
        p = key_end + 1;
        while (p < end && *p != '{') p++;
        if (p >= end) break;
        
        const char* dtype_pos = strstr(p, "\"dtype\"");
        if (dtype_pos) {
            const char* val_start = strchr(dtype_pos, ':');
            if (val_start) {
                val_start++;
                while (*val_start == ' ' || *val_start == '"') val_start++;
                const char* val_end = val_start;
                while (*val_end && *val_end != '"') val_end++;
                int dt_len = val_end - val_start;
                int copy_len = dt_len < (int)sizeof(t->dtype) - 1 ? dt_len : (int)sizeof(t->dtype) - 1;
                memcpy(t->dtype, val_start, copy_len);
                t->dtype[copy_len] = '\0';
            }
        }
        
        const char* shape_pos = strstr(p, "\"shape\"");
        if (shape_pos) {
            const char* arr_start = strchr(shape_pos, '[');
            if (arr_start) {
                arr_start++;
                t->ndim = 0;
                const char* sp = arr_start;
                while (*sp != ']' && sp < end && t->ndim < 4) {
                    while (*sp == ' ' || *sp == ',') sp++;
                    if (*sp >= '0' && *sp <= '9') {
                        t->shape[t->ndim] = strtol(sp, (char**)&sp, 10);
                        t->ndim++;
                    } else break;
                }
            }
        }
        
        const char* off_pos = strstr(p, "\"data_offsets\"");
        if (off_pos) {
            const char* arr_start = strchr(off_pos, '[');
            if (arr_start) {
                uint64_t offsets[2] = {0, 0};
                int off_count = 0;
                const char* sp = arr_start + 1;
                while (*sp != ']' && sp < end && off_count < 2) {
                    while (*sp == ' ' || *sp == ',') sp++;
                    if (*sp >= '0' && *sp <= '9') {
                        offsets[off_count] = strtoull(sp, (char**)&sp, 10);
                        off_count++;
                    } else break;
                }
                if (off_count == 2) {
                    t->data_offset = offsets[0];
                    t->data_size = offsets[1] - offsets[0];
                }
            }
        }
        
        t->num_elements = 1;
        for (int d = 0; d < t->ndim; d++) {
            t->num_elements *= t->shape[d];
        }
        
        count++;
        p = key_end + 1;
    }
    
    return count;
}

static int find_tensor(const char* name, TensorDesc* tensors, int count) {
    for (int i = 0; i < count; i++) {
        size_t nlen = strlen(name);
        if (strncmp(tensors[i].name, name, nlen) == 0 &&
            strlen(tensors[i].name) == nlen) {
            return i;
        }
    }
    return -1;
}
