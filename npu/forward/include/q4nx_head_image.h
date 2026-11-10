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
// same head geometry would be accepted. Nothing calls this API yet; whoever first does must key the path on the container's
// identity (or add a source hash beside the payload hash) before a stale file can be read as this pack's head.
//
// Absence is not an error: a container without an image simply packs as it did before.
#ifndef Q4NX_HEAD_IMAGE_H
#define Q4NX_HEAD_IMAGE_H

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


#ifdef __cplusplus
}
#endif

#endif  // Q4NX_HEAD_IMAGE_H
