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
// The packed lm-head image (engine#1716): the bytes `I8Ctx::packB_into` writes into a weight BO, on disk.
//
// The forward packs the head per tile on EVERY serve -- `memset`, `q4nx_pack_B_int8_blocked`, a BO sync -- which the issue
// measured at 10-17 s quiet and minutes contended for the h4096 shapes. Nothing device-side enters that image, so it can be
// produced on the host, written once at import and copied on later serves.
//
// What makes that safe is INTEGRITY AND GEOMETRY, not provenance (engine#1802 audit). The image is written with a format
// version, its geometry, and a hash over the payload; `read` recomputes the hash and refuses rather than returning bytes it
// cannot vouch for. So a truncated write, a corrupted byte, or a geometry change is a loud refusal and never a
// plausible-looking image -- the same rule the .fconst canonical copy follows after engine#1662/#1669.
//
// What the hash does NOT prove: that the image was built from THESE weights. An image from another pack with the same kd/nd
// hashes faithfully to its own bytes and passes every check here, so a stale image left beside a different container of the
// same head geometry would be accepted. q4nx_head_tile_image() below is the caller, and it closes that gap twice: the file name carries a digest of the head's
// own bytes (Q4nxHeadImageKey), and a hit re-packs a sample of columns from the live weights before the image is used.
//
// Absence is not an error: a container without an image simply packs as it did before.
#ifndef Q4NX_HEAD_IMAGE_H
#define Q4NX_HEAD_IMAGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bumped whenever the packer's output changes in any way. A reader that does not recognise the version refuses.
#define Q4NX_HEAD_IMAGE_VERSION 1u

typedef struct {
    uint32_t kd;      // rows of the image (K padded, as the BO holds it)
    uint32_t nd;      // columns of the image
    uint64_t hash;    // FNV-1a over the codes and then the scales, in that order
} Q4nxHeadImageInfo;

// Writes header + codes + scales. Returns 1 on success, 0 on failure (no partial file is left: the write goes to a
// temporary name beside `path` and is renamed into place only after it is fully written).
int q4nx_head_image_write(const char *path, const int8_t *codes, const float *scales, int kd, int nd);

// Reads into caller-provided buffers: `codes` must hold kd*nd int8 and `scales` nd floats (the packer's contract is one
// scale per COLUMN). Returns 1 on success, 0 on refusal with `why` set -- wrong version, wrong geometry, short payload,
// trailing bytes, or a hash that does not reproduce. Never a silent success with partial data.
int q4nx_head_image_read(const char *path, int8_t *codes, int codes_n, float *scales, int scales_n, Q4nxHeadImageInfo *info,
                         char *why, int why_n);

// The hash the writer would compute, so a caller that built an image itself can compare against a file it read.
uint64_t q4nx_head_image_hash(const int8_t *codes, const float *scales, int kd, int nd);

// The cache KEY (engine#1716, from reading run_lm_head): a head image is only valid for one combination of the kernel-set
// geometry and the container's head bytes, because the forward packs with
//
//     const int T = designs_[i_qkv].N;                 // the widest design with K == kpad_
//     const size_t tile_elems = (size_t)kpad_ * (size_t)T;
//     lctx.packB_into(bo, bt.data(), kpad_, T, ...);   // and the design's `blocked` flag changes the byte order
//
// so the SAME container packs to a DIFFERENT image under a different kernel set. A key that omitted `kpad`, `T` or
// `blocked` would serve one set's image to another -- and the aggregate correlation would still look plausible, which is
// the failure this whole contract exists to make impossible.
//
// The key is a 64-bit digest of exactly those four facts, one of which is a digest of the head bytes themselves.
typedef struct {
    uint32_t kpad;      // K padded to the kernel grid (the BO's row count)
    uint32_t t;         // the design's N: columns per tile
    uint32_t blocked;   // 1 when the design packs blocked (b8), 0 otherwise
    uint64_t head_hash; // FNV-1a over the container's lm_head.weight bytes
} Q4nxHeadImageKey;

// Returns a digest over {version, key}, so a key can name a file without embedding four numbers and a hash in a filename.
uint64_t q4nx_head_image_key_digest(const Q4nxHeadImageKey *key);

// The suggested file name for a key: "headimg-<16 hex>.q4hi". `out` is written NUL-terminated; returns 1 on success.
int q4nx_head_image_key_name(const Q4nxHeadImageKey *key, char *out, int out_n);


// ── The tile cache itself (engine#1716): what a serve calls instead of packing ────────────────────────────────────────
//
// One head tile, filled into `codes` (kd*nd bytes, the weight BO's image) and `scales` (nd floats):
//
//   * HIT   `<dir>/headimg-<key digest>-t<tile>.q4hi` reads back whole (version, geometry, payload hash) AND a bounded
//           sample of its columns equals what the packer produces for those columns from the weights the caller holds
//           NOW. Then the image is used and the tile is not gathered or packed.
//   * MISS  anything else: no file, a refused file, or a sample that disagrees. The tile is gathered through `col`,
//           packed by q4nx_pack_B_int8_blocked() -- the packer the forward has always called -- and written for the next
//           serve. A refused file is replaced, and `why` says what was wrong with it.
//
// The sample is what makes a hit safe rather than plausible. The payload hash proves the file is intact; the key proves
// it was written for this geometry and these head bytes; the sample proves that the bytes about to be copied into a
// weight BO are the bytes this build's packer writes for this model's weights, by packing those columns again. It is
// possible because the packer is column-separable: a column's scale is the max over its own K values and each code is
// rounded with that column's scale alone, so one column packs to the same bytes alone as inside its tile.
//
// `col(ctx, n, out)` writes the `k` values of vocabulary column `n` (rows past the model's width as 0). `n0` is the
// tile's first vocabulary column and `ncols` how many real columns it holds; the image's remaining columns are zero.
// `k`/`t` are the packer's logical K/N (the key's kpad and T), `kd`/`nd` the BO's padded geometry.
//
// Never fails for a cache reason: with `dir` NULL or unwritable it packs and reports wrote = 0. Returns 1 when `codes` and
// `scales` hold the tile's image, 0 only for unusable arguments or a packer refusal (nothing usable was produced).
typedef void (*Q4nxHeadColFn)(void *ctx, int n, float *out_k);

typedef struct {
    int hit;       // 1: the image came from the cache and its sample verified
    int wrote;     // 1: the image was packed and written for the next serve
    int refused;   // 1: a cached file was present and NOT used; `why` names the reason
    int sampled;   // columns re-packed to verify a hit
    char why[320];
} Q4nxHeadTileResult;

// Columns re-packed per tile to verify a cached image before it is used.
#define Q4NX_HEAD_TILE_SAMPLE 8

int q4nx_head_tile_image(const char *dir, const Q4nxHeadImageKey *key, int tile, int n0, int ncols, int k, int t, int kd,
                         int nd, Q4nxHeadColFn col, void *ctx, int8_t *codes, float *scales, Q4nxHeadTileResult *res);

// The key's `head_hash`: FNV-1a over the head's stored bytes, then over `extra` (the facts that change what those bytes
// decode to without changing them -- the caller lists them). 0 for no data.
uint64_t q4nx_head_source_hash(const void *data, size_t nbytes, const uint32_t *extra, int n_extra);

#ifdef __cplusplus
}
#endif

#endif  // Q4NX_HEAD_IMAGE_H
