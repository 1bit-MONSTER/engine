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

/* q4nx_gemma3_semantics_test -- what the forward reads from a gemma3 config (engine#1786).
 *
 * Medgemma-4B-NPU2's config.json states its layer mix as `sliding_window_pattern: 6` (no layer_types), its sliding
 * layers' RoPE base as `rope_local_base_freq` and its attention scale as `query_pre_attn_scalar`. Before this, none
 * of the three was read: the 29 sliding layers ran as full attention with no window, and a sliding layer of a config
 * that does list layer_types (Gemma3-270M-NPU2) was rotated with theta 0. The gemma3 containers also carry the norm's
 * `1 +` already, so the plain form is the right one for them, where gemma2 keeps (1 + w).
 *
 * The strings are cut down from the configs on this box to the fields the rules read. No model, no device. */
#include "q4nx_semantics.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

static int sem_of(const char* dir, const char* json, Q4nxSemantics* s) {
    char path[512];
    snprintf(path, sizeof path, "%s/config.json", dir);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE* f = fd >= 0 ? fdopen(fd, "w") : NULL;
    if (!f) { printf("FAIL cannot write %s\n", path); failures++; return -1; }
    fputs(json, f);
    fclose(f);
    if (q4nx_semantics_from_config(path, dir, s) != 0) { printf("FAIL cannot read %s\n", path); failures++; return -1; }
    return 0;
}

static const char* kMedgemma =
    "{\"head_dim\": 256, \"hidden_size\": 2560, \"model_type\": \"gemma3_text\", \"num_attention_heads\": 8,"
    " \"num_hidden_layers\": 34, \"num_key_value_heads\": 4, \"query_pre_attn_scalar\": 256,"
    " \"rope_local_base_freq\": 10000.0, \"rope_scaling\": {\"factor\": 8.0, \"rope_type\": \"linear\"},"
    " \"rope_theta\": 1000000.0, \"sliding_window\": 1024, \"sliding_window_pattern\": 6}";

int main(void) {
    char dir[] = "/tmp/q4nx-gemma3-sem-XXXXXX";
    if (!mkdtemp(dir)) { printf("FAIL mkdtemp\n"); return 1; }
    Q4nxSemantics s;

    if (sem_of(dir, kMedgemma, &s) == 0) {
        int full = 0, sliding = 0, pattern_ok = 1;
        for (int i = 0; i < s.n_layer_types; i++) {
            if (s.layer_types[i] == Q4NX_LAYER_FULL) full++;
            else if (s.layer_types[i] == Q4NX_LAYER_SLIDING) sliding++;
            const int want = ((i + 1) % 6) ? Q4NX_LAYER_SLIDING : Q4NX_LAYER_FULL;
            if (s.layer_types[i] != want) pattern_ok = 0;
        }
        check(s.n_layer_types == 34 && full == 5 && sliding == 29,
              "Medgemma: sliding_window_pattern 6 over 34 layers gives 5 full and 29 sliding layers");
        check(pattern_ok, "... every 6th layer (5, 11, ..., 29) is the full one, as HF's (i + 1) % 6 says");
        check(s.hybrid_layers == 29, "... and the hybrid count says 29");
        check(s.rope_theta == 1000000.0f && s.rope_theta_sliding == 10000.0f && s.rope_local_base,
              "Medgemma: full layers theta 1e6, sliding layers rope_local_base_freq 10000");
        check(s.query_pre_attn_scalar == 256.0f, "Medgemma: query_pre_attn_scalar is read");
        check(s.extra_layer_norms == 2, "gemma3: the two feedforward norms, without Gemma4's post-PLE norm and scalar");
        check(s.rmsnorm_offset == 0, "gemma3 containers carry the norm's 1: the plain form");
        check(s.sliding_window == 1024, "the window is unchanged");
    }

    /* Gemma3-270M-NPU2 lists layer_types and has no local base key: the list wins, and the sliding layers take the
     * model's theta rather than 0. */
    if (sem_of(dir, "{\"model_type\": \"gemma3_text_only\", \"num_hidden_layers\": 3, \"rope_theta\": 1000000.0,"
                    " \"sliding_window\": 512, \"sliding_window_pattern\": 6,"
                    " \"layer_types\": [\"sliding_attention\", \"full_attention\", \"sliding_attention\"]}",
               &s) == 0) {
        check(s.n_layer_types == 3 && s.layer_types[1] == Q4NX_LAYER_FULL && s.layer_types[0] == Q4NX_LAYER_SLIDING,
              "a stated layer_types list is not replaced by the pattern");
        check(s.rope_theta_sliding == 1000000.0f && !s.rope_local_base,
              "with no rope_local_base_freq a sliding layer takes the model's theta, not 0");
        check(s.extra_layer_norms == 2 && s.rmsnorm_offset == 0, "gemma3_text_only is gemma3 too");
    }

    /* Gemma4-12B-IT-NPU2 (engine#1786): a dense Gemma4. Its global layers have no v_proj, and attention_k_eq_v says
     * their values are the k projection; no per-layer embedding, so no post_layernorm to read. */
    if (sem_of(dir, "{\"model_type\": \"gemma4_unified\", \"num_hidden_layers\": 6, \"head_dim\": 256,"
                    " \"global_head_dim\": 512, \"num_global_key_value_heads\": 1, \"attention_k_eq_v\": true,"
                    " \"hidden_size_per_layer_input\": 0, \"sliding_window\": 1024}", &s) == 0)
        check(s.attention_k_eq_v == 1 && s.extra_layer_norms == 1 && s.hidden_size_per_layer_input == 0 &&
                  s.global_head_dim == 512 && s.num_global_kv_heads == 1,
              "Gemma4-12B: attention_k_eq_v read, Gemma4 norms, no PLE, global head 512 x 1 kv head");
    if (sem_of(dir, "{\"model_type\": \"gemma4_text\", \"num_hidden_layers\": 2}", &s) == 0)
        check(s.attention_k_eq_v == 0, "a Gemma4 config without the key keeps its own v_proj");

    /* The neighbours keep what they had. */
    if (sem_of(dir, "{\"model_type\": \"gemma2\", \"hidden_size\": 2304}", &s) == 0)
        check(s.rmsnorm_offset == 1 && s.extra_layer_norms == 0, "gemma2 keeps (1 + w) and is not gemma3");
    if (sem_of(dir, "{\"model_type\": \"gemma3n_text\", \"hidden_size\": 2048}", &s) == 0)
        check(s.extra_layer_norms == 0, "gemma3n is another architecture, not gemma3");
    if (sem_of(dir, "{\"model_type\": \"qwen3\", \"rope_theta\": 1000000.0, \"sliding_window\": 4096,"
                    " \"num_hidden_layers\": 4}", &s) == 0)
        check(s.n_layer_types == 0 && s.query_pre_attn_scalar == 0.0f && !s.rope_local_base,
              "a config with a window but no pattern gets no layer list and no gemma keys");

    char path[512];
    snprintf(path, sizeof path, "%s/config.json", dir);
    unlink(path);
    rmdir(dir);
    printf(failures ? "q4nx_gemma3_semantics: %d FAILED\n" : "q4nx_gemma3_semantics: ok\n", failures);
    return failures ? 1 : 0;
}
