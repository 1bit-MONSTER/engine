// Copyright (c) 2026 bong-water-water-bong
// q4nx_dequant.h — Q4NX (FastFlowLM model.q4nx) INT4 weight dequantizer.
//
// Layout (established empirically; see npu-infer/tools/q4nx_dequant.py and
// npu-infer/tests/verify_q4nx_format.py for the verification):
//
//   Tensor = grid of 32x256 tiles, 5120 bytes each:
//     [0..511]     256 BF16 scales,      index i = g*32 + r   (GROUP-MAJOR)
//     [512..1023]  256 BF16 zero_points, same indexing
//     [1024..5119] 4096 B packed INT4 codes
//
//   The packed region is 16 regions of 256 B, region index = h*8 + g, where
//   h = row/16 (row half) and g = col/32 (column group).  For column j = col%32
//   and rr = row%16:
//       byte   = region*256 + 8*j + rr/2
//       nibble = LOW if rr is even, HIGH if rr is odd
//   and
//       value[row][col] = code * scale[g][row] + zero_point[g][row]
//
// NOTE: this is NOT the layout asserted by the older tests/verify_i4_dequant.py
// and tests/verify_q4nx_format.py (per-row 20-byte [scale][zp][16B nibbles]
// groups with signed nibbles).  That layout drives the reconstruction to NaN
// and contradicts the byte structure; the tie test vs the BF16 embed_tokens
// (corr 0.997, max |d|/step 0.554) settles it in favour of the layout here.
#ifndef NPU_INFER_Q4NX_DEQUANT_H
#define NPU_INFER_Q4NX_DEQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define Q4NX_TILE_ROWS 32
#define Q4NX_TILE_COLS 256
#define Q4NX_GROUP_SIZE 32
#define Q4NX_GROUPS_PER_ROW (Q4NX_TILE_COLS / Q4NX_GROUP_SIZE)   /* 8 */
#define Q4NX_TILE_BYTES 5120

/* ── The newer chunk formats (OFLM 1.0.x / Qwen3.5-4B) ─────────────────────
 * The unpacked Qwen3.5-4B container does NOT use the 5120-byte q4_1 tile
 * above.  Its projections are OFLM's `q4k_block_t` (Q4_K) and its embedding /
 * ssm_alpha/beta projections are Q8 chunks.  `flm_dtype_t == 8` is special-
 * cased to exactly 4736 B in the FastFlowLM runtime (libqwen3_5_omni_npu.so
 * @ 0x52ae0; the Qwen3.6 MoE lib likewise), and dtype 1 is the 8704-B Q8
 * chunk; see npu-infer/docs/qwen35-4b-weight-format.md for the derivation and
 * the cross-validation against the checkpoint's own `.bf16` twin. */
#define Q4NX_CHUNK_Q4K 4736
#define Q4NX_CHUNK_Q8  8704

/* Q4_K chunk: 32 rows x 256 K, 4736 B.
 *   scales[8][32] uint8 @[0,256)    index g*32 + r
 *   mins  [8][32] uint8 @[256,512)  index g*32 + r
 *   qs    [256][16]     @[512,4608) byte k*16 + r/2, even row = low nibble
 *   S[32] bf16 @[4608,4672)   M[32] bf16 @[4672,4736)  (M already negated)
 *   value(r,k) = S[r]*scales[g][r]*nib + M[r]*mins[g][r] */
void q4nx_dequant_tile_q4k(const uint8_t* chunk, float* out, int ld);

/* Q8 chunk: 32 rows x 256 K, 8704 B.
 *   d[256] bf16 @[0,512)   index g*32 + r
 *   q[8192] i8  @[512,8704) byte k*32 + r
 *   value(r,k) = d[g*32+r] * q[k*32+r] */
void q4nx_dequant_tile_q8(const uint8_t* chunk, float* out, int ld);

/* Dequantize a whole [rows, cols] tensor whose tiles are `chunk_bytes` each
 * (5120 = q4_1, 4736 = Q4_K, 8704 = Q8).  Picks the matching tile decoder and
 * lays the row-tile-major grid out into out[rows*cols].  Returns 0, or -1 on a
 * geometry/size mismatch. */
int q4nx_dequant_tensor_chunked(const uint8_t* data, size_t nbytes, int rows,
                                int cols, int chunk_bytes, float* out);

/* Dequantize one 5120-byte tile into out[32][ld] (row-major, strided by ld). */
void q4nx_dequant_tile(const uint8_t* tile, float* out, int ld);

/* Dequantize a whole INT4 tensor of logical shape [rows, cols] whose packed
 * bytes are a row-major grid of 32x256 tiles.  out must hold rows*cols floats
 * (row-major).  Returns 0 on success, -1 on a geometry mismatch. */
int q4nx_dequant_tensor(const uint8_t* data, size_t nbytes, int rows, int cols,
                        float* out);

/* Bytes a [rows, cols] tensor occupies in the packed grid. */
size_t q4nx_tensor_bytes(int rows, int cols);

/* PLAIN int8 with a per-32-column F32 scale (Gemma4 stores `embed_tokens` and
 * `per_layer_token_embd` this way instead of the packed tile grid):
 *
 *   out[r*cols + c] = (float)data[r*cols + c] * scale[r*(cols/32) + c/32]
 *
 * `scale` may be NULL, in which case the raw int8 codes are cast to float.
 * Returns 0 on success, -1 on a geometry mismatch. */
int q4nx_dequant_plain_i8(const uint8_t* data, const float* scale, int rows,
                          int cols, float* out);

#ifdef __cplusplus
}
#endif
#endif /* NPU_INFER_Q4NX_DEQUANT_H */
