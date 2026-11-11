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
//   A tile of that size comes in two kinds. The one above is the asymmetric kind (FastFlowLM lists such packs as
//   "Q4_1"): codes 0..15 and a zero-point that is the group's offset. The symmetric kind (listed as "Q4_0"; the
//   LFM2 packs) has the SAME byte layout with every zero-point stored as 0, a scale that carries a sign, and the
//   code as a two's-complement nibble:
//       value[row][col] = (code >= 8 ? code - 16 : code) * scale[g][row]
//   Nothing in the container names the kind (no metadata, no config key; the name is in FastFlowLM's model list,
//   which is not part of a pack), so it is read from the weights, with three rules:
//     - a tile is SYMMETRIC on its own evidence only when all 256 zero-points are zero AND some scale is below
//       zero. The sign is what an asymmetric tile with no offset does not have: a zero offset with positive scales
//       is a legitimate asymmetric tile (weights 0..15 * scale; tests/npu_moe_container_test.cpp writes one), and it
//       reads as it always did. A symmetric scale is max/-8 of its group, so it is negative whenever the group's
//       largest weight is positive: about half of them, and never none of 256 in a real tile.
//     - a TENSOR is one kind. The tensor-level readers look at all of its tiles first: any symmetric tile makes
//       the tensor symmetric, and its undecided tiles (all zero-points zero, no negative scale: padding, mostly)
//       follow it; a tensor with both a symmetric and an asymmetric tile is refused, by name, on stderr.
//       The fused int8 packer (q4nx_pack_q4_1_tiles_to_int8) takes the kind the same way, over the whole tensor.
//     - the one-row reader (q4nx_dequant_tile_row) takes it from the tiles of the row's own tile-row: it runs once
//       per token on an embedding it exists to avoid paging in, so it does not scan the table.
//     - q4nx_dequant_tile() on one tile, with no tensor around it, uses that tile's own evidence.
//   Measured 2026-11-11 on every 5120-byte tile of the fifteen packs on the box: 1,889,236 tiles in the other
//   packs, none with an all-zero zero-point block and none with a negative scale; 285,696 tiles in the two LFM2
//   packs, every one with both. Read as asymmetric, an LFM2 projection correlates -0.39 with the same tensor of
//   the upstream GGUF; read as above, 0.99.
//   The same tie test that settled the asymmetric layout settles this reading: LFM2.5-1.2B's lm_head is the
//   quantised copy of its own bf16 token_embd, and over the first 2048 rows |dequant - embedding| / |scale| has
//   mean 0.243 and maximum 1.000 (0.48 % of elements above 0.5, all of them the clip of +8 steps to +7). The
//   other two candidate readings, (code - 8) * scale and unsigned code * scale, are 8 and 7 steps off on average.
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
/* The symmetric 4-bit chunk with no zero-point block (FastFlowLM lists the pack as "Q4_0"; Hy-MT2-1.8B-NPU2):
 *   scales [256] bf16 @[0,512)     index g*32 + r (group-major, as in the 5120-byte tile)
 *   codes  [256][16]  @[512,4608)  byte c*16 + r/2, even row = low nibble (column-major, as in the Q4K chunk)
 *   value[r][c] = (code >= 8 ? code - 16 : code) * scale[g][r]
 * Settled by the tie test on that pack (2026-11-11): its lm_head is the quantised copy of its own bf16
 * embed_tokens, and over the first 1024 rows |dequant - embedding| / |scale| has mean 0.244 and maximum 1.000 with
 * this reading; the eleven other combinations of scale order, code order and code sign are 3 to 8 steps off.
 * The size names the kind (no tile of another kind is 4608 bytes), so nothing is inferred from the data. */
#define Q4NX_CHUNK_Q4S 4608
void q4nx_dequant_tile_q4s(const uint8_t* chunk, float* out, int ld);

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
/* ONE row of the same chunked (tiled) tensor, decoded without decoding the table: row `r` lives as row `r % 32` inside
 * each of `cols / 256` tiles, so this reads those tiles and nothing else. A loader that reads a row per token needs no
 * table-sized buffer, which is the difference between 0.74 GiB of packed container and its 4.74 GiB f32 expansion
 * (engine#1784). Returns 0 and writes `cols` floats, or -1 for the same conditions q4nx_dequant_tensor_chunked()
 * refuses, plus an out-of-range row. The values are the ones that function would place at that row, bit for bit. */
int q4nx_dequant_tile_row(const uint8_t* data, size_t nbytes, int rows, int cols, int chunk_bytes, int r,
                          float* out /* cols */);
/* The chunk kind q4nx_dequant_tile_row() would accept for a [rows, cols] table of `nbytes`, or 0 when it would refuse
 * every row: rows a multiple of 32, cols a multiple of 256, and nbytes exactly (rows/32)*(cols/256) tiles of one of the
 * three kinds. A loader that picks the per-row path asks this first (engine#1842): choosing it on `nbytes % tiles == 0`
 * alone let a table the reader refuses load fine and then serve every embedding read as 0.0f. */
int q4nx_tile_row_kind(size_t nbytes, int rows, int cols);

/* The (rows, cols) tile grid a container actually stores, when it is LARGER than the model's.
 *
 * data_size pins it exactly -- it is (rows_p/32)*(cols_p/256)*chunk_bytes for one of the three chunk
 * kinds -- so the grid is recoverable without knowing the kind first (engine#1097: the FastFlowLM
 * repack pads intermediate_size to its 512-column grid, IM 8960 -> 9216, so a ZR1 MLP weight holds
 * 1728 tiles where the model's dims imply 1680).
 *
 * Returns 1 and the grid when exactly one candidate fits, 0 otherwise. A container that needs no
 * padding is 0 (the caller keeps the plain decoder), and so is anything unsuitable: no chunk kind
 * dividing data_size, a grid smaller than the model's, more than Q4NX_PACKED_MAX_EXCESS extra tiles,
 * or **two different grids with the same least excess**. That last one is not a guess worth making:
 * (ntr, ntc+1) and (ntr+1, ntc) with equal excess are a transposed decode waiting to happen, so an
 * ambiguous tile count is refused rather than resolved by loop order. */
#define Q4NX_PACKED_MAX_EXCESS 64
int q4nx_packed_grid(size_t nbytes, int rows, int cols, int* cb_out, int* rp_out, int* cp_out);

/* Dequantize a [rows, cols] tensor out of a container whose tile grid is LARGER: (rows_packed,
 * cols_packed).  A converter may pad a weight to a wider grid than the model's dimensions
 * (engine#1097: the FastFlowLM repack pads intermediate_size to 512 columns, 8960 -> 9216, so the
 * weight holds 1728 tiles where the model's dims imply 1680).  The container's grid is the read
 * geometry; only the model's [rows, cols] are written, so the caller's buffers, its row stride and
 * the kernel's B layout are unchanged.  rows_packed/cols_packed must be on the 32x256 tile grid and
 * must dominate rows/cols; equal values make this exactly q4nx_dequant_tensor_chunked().
 * Returns 0, or -1 on a geometry/size mismatch. */
int q4nx_dequant_tensor_packed(const uint8_t* data, size_t nbytes, int rows, int cols,
                               int rows_packed, int cols_packed, int chunk_bytes, float* out);

/* What one 5120-byte tile says about its kind (see the layout notes above). */
#define Q4NX_TILE_ASYMMETRIC 0   /* a zero-point is present */
#define Q4NX_TILE_SYMMETRIC  1   /* every zero-point is +-0 and some scale is below zero */
#define Q4NX_TILE_UNDECIDED  2   /* every zero-point is +-0, no scale below zero: asymmetric unless its tensor says otherwise */
int q4nx_tile_kind(const uint8_t* tile);

/* The kind of `ntiles` 5120-byte tiles of ONE tensor, `stride` bytes apart: Q4NX_TILE_SYMMETRIC when any tile is
 * symmetric and none asymmetric, Q4NX_TILE_ASYMMETRIC when none is symmetric, -1 when both occur. */
int q4nx_tiles_kind(const uint8_t* data, size_t ntiles, size_t stride);

/* Dequantize one 5120-byte tile into out[32][ld] (row-major, strided by ld), on the tile's own evidence. */
void q4nx_dequant_tile(const uint8_t* tile, float* out, int ld);
/* The same with the kind given by the caller, which knows the tensor the tile belongs to (symmetric != 0). */
void q4nx_dequant_tile_as(const uint8_t* tile, float* out, int ld, int symmetric);

/* Dequantize a whole INT4 tensor of logical shape [rows, cols] whose packed
 * bytes are a row-major grid of 32x256 tiles, or a PLAIN bf16 tensor of that
 * shape.  out must hold rows*cols floats (row-major).  `nbytes` must be exactly
 * rows*cols*2 (plain bf16) or exactly the tile grid's size; anything else is a
 * format this decoder does not know and is refused.  Returns 0 on success, -1
 * otherwise. */
int q4nx_dequant_tensor(const uint8_t* data, size_t nbytes, int rows, int cols,
                        float* out);

/* Bytes a [rows, cols] tensor occupies in the packed grid. */
size_t q4nx_tensor_bytes(int rows, int cols);

/* PLAIN bf16, row-major, no tile header (`tools/safetensors_to_q4nx.py pack --dtype
 * bf16` writes these): out[r*cols + c] = bf16(data)[r*cols + c].  q4nx_dequant_tensor()
 * takes this path when the buffer is exactly rows*cols*2 bytes, so a bf16 container is
 * no longer decoded as 5120-byte q4_1 tiles (engine#875).  Returns 0, or -1 on a
 * geometry argument that cannot describe a tensor. */
int q4nx_dequant_plain_bf16(const uint8_t* data, int rows, int cols, float* out);

/* PLAIN int8 with a per-32-column F32 scale (Gemma4 stores `embed_tokens` and
 * `per_layer_token_embd` this way instead of the packed tile grid):
 *
 *   out[r*cols + c] = (float)data[r*cols + c] * scale[r*(cols/32) + c/32]
 *
 * `scale` may be NULL, in which case the raw int8 codes are cast to float.
 * Returns 0 on success, -1 on a geometry mismatch. */
int q4nx_dequant_plain_i8(const uint8_t* data, const float* scale, int rows,
                          int cols, float* out);

/* One element of the same table `q4nx_dequant_plain_i8()` decodes, without decoding the table: what a loader needs
 * to read a row per token out of a read-only mapping instead of expanding the whole table into RAM (engine#1784).
 * `row`/`col` out of range give 0.0f. `scale` may be NULL, which is the unscaled int8 table. The arithmetic is the
 * one in the loop above -- `(float)q * scale[row * groups + col / group]` -- so a row read this way is bit-identical
 * to the same element of the fully decoded table, which is what the test pins. */
float q4nx_plain_i8_at(const uint8_t* data, const float* scale, int rows, int cols, int row, int col);

#ifdef __cplusplus
}
#endif
#endif /* NPU_INFER_Q4NX_DEQUANT_H */
