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

/* q4nx_qk_norm_test -- which models get the per-head q/k RMSNorm (engine#1889).
 *
 * The forward applies it only when the semantics flag is set, and the flag used to come from a model_type /
 * architectures table alone (qwen3*, gemma*). Hunyuan-7B is `hunyuan_v1_dense` with `use_qk_norm: true`: the table had
 * no row for it, so its container's self_attn.{q,k}_norm.weight were loaded and never applied. The config key now
 * decides when the config states it; the table stays the answer for the families that do not state it.
 *
 * The strings below are the real keys of the configs on this box (Hunyuan-7B-Instruct's config.json, a Qwen3 and a
 * Llama container's), cut down to the fields the rule reads. No model, no device. */
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

/* <dir>/config.json holding exactly `json`, 0600 (mkdtemp gives 0700; the file's own mode is what CodeQL reads). */
static int after_rope = -1;   /* the last config's qk_norm_after_rope */

static int qk_norm_of(const char* dir, const char* json) {
    char path[512];
    snprintf(path, sizeof path, "%s/config.json", dir);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE* f = fd >= 0 ? fdopen(fd, "w") : NULL;
    if (!f) { printf("FAIL cannot write %s\n", path); failures++; return -1; }
    fputs(json, f);
    fclose(f);
    Q4nxSemantics s;
    if (q4nx_semantics_from_config(path, dir, &s) != 0) { printf("FAIL cannot read %s\n", path); failures++; return -1; }
    after_rope = s.qk_norm_after_rope;
    return s.qk_norm;
}

int main(void) {
    char dir[] = "/tmp/q4nx-qk-norm-XXXXXX";
    if (!mkdtemp(dir)) { printf("FAIL mkdtemp\n"); return 1; }

    /* The measured case: a family the name table has never heard of, declaring the norm. */
    check(qk_norm_of(dir, "{\"architectures\": [\"HunYuanDenseV1ForCausalLM\"], \"model_type\": \"hunyuan_v1_dense\","
                          " \"num_attention_heads\": 32, \"num_key_value_heads\": 8, \"head_dim\": 128,"
                          " \"use_qk_norm\": true}") == 1,
          "hunyuan_v1_dense with use_qk_norm: true normalises q and k (engine#1889)");
    /* The same family saying it has none: the key is read, not assumed from the family. */
    check(qk_norm_of(dir, "{\"architectures\": [\"HunYuanDenseV1ForCausalLM\"], \"model_type\": \"hunyuan_v1_dense\","
                          " \"use_qk_norm\": false}") == 0,
          "hunyuan_v1_dense with use_qk_norm: false does not");
    /* A family with no row and no key keeps the neutral convention. */
    check(qk_norm_of(dir, "{\"architectures\": [\"HunYuanDenseV1ForCausalLM\"], \"model_type\": \"hunyuan_v1_dense\"}") == 0,
          "an unlisted family that does not state the key keeps the plain convention");

    /* The table is unchanged for the families that never state the key. */
    check(qk_norm_of(dir, "{\"architectures\": [\"Qwen3ForCausalLM\"], \"model_type\": \"qwen3\"}") == 1,
          "qwen3 without the key still normalises (the name table)");
    check(qk_norm_of(dir, "{\"architectures\": [\"Gemma3ForCausalLM\"], \"model_type\": \"gemma3_text\"}") == 1,
          "gemma without the key still normalises (the name table)");
    check(qk_norm_of(dir, "{\"architectures\": [\"LlamaForCausalLM\"], \"model_type\": \"llama\"}") == 0,
          "llama without the key still does not");
    check(qk_norm_of(dir, "{\"architectures\": [\"Phi3ForCausalLM\"], \"model_type\": \"phi3\"}") == 0,
          "phi without the key still does not");

    /* A stated key beats the table in both directions, and a null is not a statement. */
    check(qk_norm_of(dir, "{\"model_type\": \"llama\", \"use_qk_norm\": true}") == 1,
          "a llama-named config that states use_qk_norm: true normalises");
    check(qk_norm_of(dir, "{\"model_type\": \"qwen3\", \"use_qk_norm\": false}") == 0,
          "a qwen3-named config that states use_qk_norm: false does not");
    check(qk_norm_of(dir, "{\"model_type\": \"qwen3\", \"use_qk_norm\": null}") == 1,
          "use_qk_norm: null states nothing, so the table answers");
    /* Only the top-level key counts: a nested one belongs to another component's config. */
    check(qk_norm_of(dir, "{\"model_type\": \"llama\", \"vision_config\": {\"use_qk_norm\": true}}") == 0,
          "a use_qk_norm nested under another component's config is not this model's");

    /* The ORDER (engine#1889). Hunyuan's graph is q_proj -> RotaryEmbedding -> q_norm -> attention in every layer, so
     * its norm runs after RoPE; Qwen3 and Gemma keep norm-then-RoPE. */
    (void)qk_norm_of(dir, "{\"architectures\": [\"HunYuanDenseV1ForCausalLM\"], \"model_type\": \"hunyuan_v1_dense\","
                          " \"use_qk_norm\": true}");
    check(after_rope == 1, "hunyuan_v1_dense normalises q and k AFTER RoPE, as its graph does");
    (void)qk_norm_of(dir, "{\"architectures\": [\"HunYuanDenseV1ForCausalLM\"]}");
    check(after_rope == 1, "... and the architectures entry alone is enough to say so");
    (void)qk_norm_of(dir, "{\"architectures\": [\"Qwen3ForCausalLM\"], \"model_type\": \"qwen3\"}");
    check(after_rope == 0, "qwen3 keeps norm, then RoPE");
    (void)qk_norm_of(dir, "{\"architectures\": [\"Gemma3ForCausalLM\"], \"model_type\": \"gemma3_text\"}");
    check(after_rope == 0, "gemma keeps norm, then RoPE");
    (void)qk_norm_of(dir, "{\"model_type\": \"llama\", \"use_qk_norm\": true}");
    check(after_rope == 0, "a non-Hunyuan config that states use_qk_norm gets the default order");

    check(strstr(q4nx_semantics_field(Q4NX_SEM_QK_NORM), "use_qk_norm") != NULL,
          "the startup line names use_qk_norm as the flag's source");

    /* The missing-gain refusal is about attention layers. LFM2: q/k norm on, layer 0 a short-convolution layer with no
     * attention and so no q_norm. Applied to every layer, the rule refused both LFM2 packs at load. */
    check(q4nx_qk_gain_missing(1, 1, 0) == 1, "q/k norm on, an attention layer with no q gain: refused (the engine#1889 case)");
    check(q4nx_qk_gain_missing(1, 1, 1) == 0, "q/k norm on, an attention layer with its gain: fine");
    check(q4nx_qk_gain_missing(1, 0, 0) == 0, "q/k norm on, a layer with no attention and no gain: not a refusal (LFM2's conv layers)");
    check(q4nx_qk_gain_missing(0, 1, 0) == 0, "q/k norm off: no gain is expected anywhere");

    char path[512];
    snprintf(path, sizeof path, "%s/config.json", dir);
    unlink(path);
    rmdir(dir);
    printf(failures ? "q4nx_qk_norm: %d FAILED\n" : "q4nx_qk_norm: ok\n", failures);
    return failures ? 1 : 0;
}
