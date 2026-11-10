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
// The packed lm-head image, on disk (engine#1716). See the header for why the identity matters more than the speed: a
// head image is only ever a copy of what the packer would produce for the same tile, so the file carries a version, its
// geometry and a hash, and a reader that cannot reproduce the hash refuses instead of returning bytes it cannot vouch for.

#include "q4nx_head_image.h"

#include <stdio.h>
#include <string.h>

// "Q4HI" then the version as four ASCII digits, so a hexdump of the first eight bytes says what the file is.
#define Q4NX_HEAD_IMAGE_MAGIC "Q4HI"

typedef struct {
    char magic[4];
    char version[4];
    uint32_t kd;
    uint32_t nd;
    uint64_t hash;
} HeadImageHeader;

static uint64_t fnv1a(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)b[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t q4nx_head_image_hash(const int8_t *codes, const float *scales, int kd, int nd) {
    if (kd <= 0 || nd <= 0) return 0;
    uint64_t h = 1469598103934665603ull;
    h = fnv1a(h, codes, (size_t)kd * (size_t)nd);
    h = fnv1a(h, scales, (size_t)nd * sizeof(float));
    return h;
}

static void version_digits(char out[4]) {
    const unsigned v = Q4NX_HEAD_IMAGE_VERSION;
    out[0] = (char)('0' + (v / 100) % 10);
    out[1] = (char)('0' + (v / 10) % 10);
    out[2] = (char)('0' + v % 10);
    out[3] = 0;
}

int q4nx_head_image_write(const char *path, const int8_t *codes, const float *scales, int kd, int nd) {
    if (!path || !codes || !scales || kd <= 0 || nd <= 0) return 0;
    HeadImageHeader h;
    memcpy(h.magic, Q4NX_HEAD_IMAGE_MAGIC, 4);
    version_digits(h.version);
    h.kd = (uint32_t)kd;
    h.nd = (uint32_t)nd;
    h.hash = q4nx_head_image_hash(codes, scales, kd, nd);
    // Written beside the destination and renamed into place: a truncated image must never be readable as a valid one,
    // because the reader would then copy it into a weight BO and the model would answer wrongly with no message.
    char tmp[4096];
    if (snprintf(tmp, sizeof tmp, "%s.tmp", path) <= 0) return 0;
    FILE *f = fopen(tmp, "wb");
    if (!f) return 0;
    int ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(codes, (size_t)kd * (size_t)nd, 1, f) == 1 &&
             fwrite(scales, (size_t)nd * sizeof(float), 1, f) == 1;
    ok = ok && fflush(f) == 0;
    if (fclose(f) != 0) ok = 0;
    if (!ok || rename(tmp, path) != 0) {
        remove(tmp);
        return 0;
    }
    return 1;
}

int q4nx_head_image_read(const char *path, int8_t *codes, int codes_n, float *scales, int scales_n, Q4nxHeadImageInfo *info,
                         char *why, int why_n) {
    if (why && why_n > 0) why[0] = 0;
    char dv[4];
    version_digits(dv);
    FILE *f = path ? fopen(path, "rb") : NULL;
    if (!f) {
        if (why) snprintf(why, (size_t)why_n, "the head image is not there (%s)", path ? path : "(no path)");
        return 0;
    }
    HeadImageHeader h;
    if (fread(&h, sizeof h, 1, f) != 1) {
        fclose(f);
        if (why) snprintf(why, (size_t)why_n, "the head image is shorter than its header");
        return 0;
    }
    if (memcmp(h.magic, Q4NX_HEAD_IMAGE_MAGIC, 4) != 0) {
        fclose(f);
        if (why) snprintf(why, (size_t)why_n, "the head image does not start with %s", Q4NX_HEAD_IMAGE_MAGIC);
        return 0;
    }
    if (memcmp(h.version, dv, 4) != 0) {
        fclose(f);
        if (why)
            snprintf(why, (size_t)why_n, "the head image is format version %.4s, this build writes %s", h.version, dv);
        return 0;
    }
    if (h.kd == 0 || h.nd == 0 || (int64_t)h.kd * (int64_t)h.nd != (int64_t)codes_n || (int)h.nd != scales_n) {
        fclose(f);
        if (why)
            snprintf(why, (size_t)why_n,
                     "the head image is %ux%u (so %d codes and %u scales), not the %d x %d this container needs",
                     h.kd, h.nd, (int)(h.kd * h.nd), h.nd, codes_n > 0 && scales_n > 0 ? codes_n / scales_n : 0, scales_n);
        return 0;
    }
    const size_t n_codes = (size_t)h.kd * (size_t)h.nd, n_scales = (size_t)h.nd * sizeof(float);
    int ok = fread(codes, n_codes, 1, f) == 1 && fread(scales, n_scales, 1, f) == 1;
    if (ok) {
        // anything after the payload means the file is not the one the header describes
        char extra;
        ok = fread(&extra, 1, 1, f) == 0;
    }
    fclose(f);
    if (!ok) {
        if (why) snprintf(why, (size_t)why_n, "the head image payload is short or has trailing bytes");
        return 0;
    }
    const uint64_t got = q4nx_head_image_hash(codes, scales, (int)h.kd, (int)h.nd);
    if (got != h.hash) {
        if (why)
            snprintf(why, (size_t)why_n,
                     "the head image does not match its own hash (%016llx in the file, %016llx over the bytes): it was "
                     "written by a different packer or damaged",
                     (unsigned long long)h.hash, (unsigned long long)got);
        return 0;
    }
    if (info) {
        info->kd = h.kd;
        info->nd = h.nd;
        info->hash = h.hash;
    }
    return 1;
}
