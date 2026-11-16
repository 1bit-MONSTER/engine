/* Copyright 2026 bong-water-water-bong
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// engine#2673: chatglm3's attention nodes state rotary_interleaved = 1 -- RoPE on consecutive pairs (2i, 2i+1) over the
// first rot channels -- while the forward rotated half-split pairs (i, i + rot/2). Arms: the interleaved rotation acts on
// (2i, 2i+1) with frequency i; it equals half-split under the pair permutation (so the two differ only in which channels
// form a pair); channels past rot are untouched; the control shows the two orders really disagree on one vector; and
// the config key `rope_interleaved` reaches Q4nxSemantics (default 0).
#include "q4nx_rope.h"
#include "q4nx_semantics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
static void check(int ok, const char* what) {
    if (!ok) { printf("FAIL %s\n", what); ++failures; }
}

int main(void) {
    enum { HD = 12, ROT = 8, HALF = ROT / 2 };
    float c[HALF], sn[HALF];
    for (int i = 0; i < HALF; i++) { const double a = 0.3 + 0.7 * i; c[i] = (float)cos(a); sn[i] = (float)sin(a); }
    float x[HD], inter[HD], half[HD], perm[HD];
    for (int k = 0; k < HD; k++) x[k] = (float)(k + 1) * 0.5f;

    // 1. interleaved: pair i is (2i, 2i+1), rotated by angle i
    memcpy(inter, x, sizeof x);
    q4nx_rope_rotate_pairs(inter, HALF, c, sn, 1);
    int ok = 1;
    for (int i = 0; i < HALF; i++) {
        const float a = x[2 * i], b = x[2 * i + 1];
        ok &= fabsf(inter[2 * i] - (a * c[i] - b * sn[i])) < 1e-6f && fabsf(inter[2 * i + 1] - (a * sn[i] + b * c[i])) < 1e-6f;
    }
    check(ok, "interleaved: pair i = (2i, 2i+1) rotated by frequency i");
    // 2. it is half-split under the permutation (2i, 2i+1) -> (i, i+half)
    for (int k = 0; k < HD; k++) perm[k] = x[k];
    for (int i = 0; i < HALF; i++) { perm[i] = x[2 * i]; perm[i + HALF] = x[2 * i + 1]; }
    memcpy(half, perm, sizeof perm);
    q4nx_rope_rotate_pairs(half, HALF, c, sn, 0);
    ok = 1;
    for (int i = 0; i < HALF; i++) ok &= half[i] == inter[2 * i] && half[i + HALF] == inter[2 * i + 1];
    check(ok, "interleaved == half-split of the pair-permuted vector, bit for bit");
    // 3. channels past rot pass through, in both orders
    ok = 1;
    for (int k = ROT; k < HD; k++) ok &= inter[k] == x[k];
    float h2[HD];
    memcpy(h2, x, sizeof x);
    q4nx_rope_rotate_pairs(h2, HALF, c, sn, 0);
    for (int k = ROT; k < HD; k++) ok &= h2[k] == x[k];
    check(ok, "channels past rot are untouched");
    // 4. control: on the same unpermuted vector the two orders disagree -- the bug was real, not a relabelling
    float maxd = 0;
    for (int k = 0; k < ROT; k++) { const float d = fabsf(inter[k] - h2[k]); if (d > maxd) maxd = d; }
    check(maxd > 0.1f, "control: interleaved and half-split give different q/k for the same input");

    // 5. the config key reaches the semantics (default 0)
    char dir[] = "./npu-rope-interleave-XXXXXX";
    if (!mkdtemp(dir)) { printf("FAIL mkdtemp\n"); return 1; }
    char cfg[512];
    snprintf(cfg, sizeof cfg, "%s/config.json", dir);
    Q4nxSemantics s;
    FILE* f = fopen(cfg, "w");
    fputs("{\"model_type\": \"chatglm\", \"hidden_size\": 64, \"num_hidden_layers\": 2, \"rope_interleaved\": true}", f);
    fclose(f);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.rope_interleaved == 1, "config rope_interleaved: true -> semantics rope_interleaved = 1");
    f = fopen(cfg, "w");
    fputs("{\"model_type\": \"llama\", \"hidden_size\": 64, \"num_hidden_layers\": 2}", f);
    fclose(f);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.rope_interleaved == 0, "control: no key -> rope_interleaved = 0 (half-split stays the default)");
    remove(cfg);
    rmdir(dir);

    if (failures) { printf("%d failure(s)\n", failures); return 1; }
    printf("rope interleave: 6 arms passed\n");
    return 0;
}
