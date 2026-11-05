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

/* `rope_type: "llama3"` — the RoPE frequency scaling of Llama-3.1/3.2/3.3 (engine#1048).
 *
 * A Llama-3.x config pairs a very large `rope_theta` (500000) with a scaling block:
 *
 *   "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "high_freq_factor": 4.0,
 *                    "low_freq_factor": 1.0, "original_max_position_embeddings": 8192}
 *
 * and the frequency of every dim pair is classified by its wavelength, in units of the ORIGINAL
 * context length: wavelengths shorter than `original_max/low_freq_factor` keep their frequency,
 * wavelengths longer than `original_max/high_freq_factor` are divided by `factor`, and the band in
 * between is linearly interpolated between the two (transformers' `_compute_llama3_parameters`).
 *
 * Without it, a Llama-3.x model is served with `theta^(-i/half)` and nothing else, which is what
 * this forward did: measured on Llama-3.2-1B, 17 of 32 dim pairs are scaled (14 divided by exactly
 * 32, 3 interpolated) and the resulting logit error grows with position — KL 0.0058 at ctx 104,
 * 0.0107 at 754, 0.0252 at 1497 against the same checkpoint with the scaling applied.
 *
 * The scaling is applied to the *frequency*, before it is multiplied by the position, so it costs
 * nothing per token.  `on == 0` is a no-op and is the value for every model that does not declare
 * this rope type, which is what keeps the change from touching anything else.
 */
#pragma once

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int on;                        /* config declared rope_type "llama3" with usable numbers */
    double factor;                 /* 32.0 for Llama-3.2-1B/3B, 8.0 for Llama-3.1-8B */
    double high_freq_factor;       /* 4.0 */
    double low_freq_factor;        /* 1.0 */
    double original_max_position;  /* 8192 — the band is in THIS length's units, not the served ctx */
} Q4nxRopeLlama3;

/* One dim pair's inverse frequency, after the llama3 rule.  `inv` is the plain theta^(-i/half).
 * Returns `inv` unchanged when `s` is null or off, or when the block is not usable (a factor of 0,
 * a band with high <= low), because a malformed scaling block must not silently re-frequency a model
 * in a way no reference can reproduce. */
static inline double q4nx_rope_llama3_inv_freq(double inv, const Q4nxRopeLlama3* s) {
    if (!s || !s->on) return inv;
    if (!(s->factor > 0.0) || !(s->original_max_position > 0.0)) return inv;
    if (!(s->high_freq_factor > s->low_freq_factor)) return inv;

    const double wavelen = 2.0 * 3.14159265358979323846 / inv;
    const double low_wavelen = s->original_max_position / s->low_freq_factor;
    const double high_wavelen = s->original_max_position / s->high_freq_factor;

    if (wavelen > low_wavelen) return inv / s->factor;   /* low frequency: scaled down */
    if (wavelen < high_wavelen) return inv;              /* high frequency: untouched */
    const double smooth = (s->original_max_position / wavelen - s->low_freq_factor) /
                          (s->high_freq_factor - s->low_freq_factor);
    return (1.0 - smooth) * inv / s->factor + smooth * inv;
}

/* The scaling a config.json DECLARES, and whether this forward implements it (engine#1087).
 *
 * A declared block that is not applied is worse than a failed load: the forward runs plain
 * theta^(-i/half), the reply stays fluent and is wrong past the model's original window, and no line
 * in the log says the model was served something other than it asked for.  Two rules are implemented
 * — `llama3` (above) and LongRoPE's `short_factor` (read_rope_short_factor in q4nx_forward.cpp) — so
 * every other declared type is refused at load BY NAME instead.
 *
 * q4nx_rope_declared_type() writes "" when the config declares no scaling block at all (the case for
 * every model that has no rope_scaling key), and the declared type otherwise.
 */
static inline void q4nx_rope_declared_type(const char* json, char* out, int cap) {
    if (!out || cap <= 0) return;
    out[0] = '\0';
    if (!json) return;
    const char* rs = strstr(json, "\"rope_scaling\"");
    if (!rs) return;
    {
        /* "rope_scaling": null declares nothing — and must not let the search below find some other
         * block's "type": that is the one way this reader could refuse a model that asked for nothing. */
        const char* c = strchr(rs, ':');
        if (!c) return;
        const char* v = c + 1;
        while (*v == ' ' || *v == '\t' || *v == '\n' || *v == '\r') ++v;
        if (*v != '{') return;
    }
    /* the block is a flat object in every config we read; 800 bytes is past its closing brace */
    const char* end = rs + 800;
    const char* p = strstr(rs, "\"rope_type\"");      /* current HF spelling */
    if (!p || p > end) p = strstr(rs, "\"type\"");   /* pre-4.45 spelling: {"type": "yarn"} */
    if (!p || p > end) return;
    const char* c = strchr(p, ':');
    if (!c || c > end) return;
    const char* q = strchr(c, '"');
    if (!q || q > end) return;
    const char* e = strchr(q + 1, '"');
    if (!e) return;
    int n = (int)(e - q - 1);
    if (n > cap - 1) n = cap - 1;
    memcpy(out, q + 1, (size_t)n);
    out[n] = '\0';
}

/* 1 when the declared type is one this forward applies, 0 when it must be refused.  `llama3` needs
 * usable numbers (a declared-but-unusable block is not the same thing as no block) and `longrope`
 * needs the short_factor the forward reads. */
static inline int q4nx_rope_type_supported(const char* type, const Q4nxRopeLlama3* llama3,
                                           int has_short_factor) {
    if (!type || !type[0]) return 1;                        /* no block declared: plain rope */
    if (!strcmp(type, "default") || !strcmp(type, "none")) return 1;
    if (!strcmp(type, "llama3")) return llama3 && llama3->on;
    if (!strcmp(type, "longrope")) return has_short_factor ? 1 : 0;
    return 0;                                               /* yarn, linear, dynamic, mrope, ... */
}

#ifdef __cplusplus
}
#endif
