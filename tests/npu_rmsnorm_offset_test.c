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

// npu_rmsnorm_offset_test -- qwen3_5 packs the two RMSNorm conventions under one model_type, so the
// forward decides from the layer-0 input_layernorm weight (engine#796). This checks that rule against
// the two real shapes on this box, with no model and no device:
//
//   * the official release's Qwen3_5RMSNorm weight is a delta trained from 0 -- mean 0.2377, min
//     -0.1143 on d3-lite layer 0 -- and must resolve to the (1 + w) form;
//   * the hybrid 27B pack's weight is a gain trained from 1 -- mean 0.9704, min 0.8711 -- and must
//     stay the plain w form.
//
// The measured consequence, from the same diagnostic: numpy rmsnorm(embed, w) reproduces the engine's
// own xn (cos 1.000000) and rmsnorm(embed, 1 + w) reproduces the reference's (cos 0.999999), so this
// one bit is the whole 0.6367-token-probability gap in engine#796's causal arm.
#include "q4nx_semantics.h"
#include "file_create.h"   /* fixture files are created owner-only and never through a symlink */

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

int main(void) {
    /* Shape 1: zero-centered (the d3-lite / Qwen3.5 release form). Values echo the real tensor's
     * statistics: a delta around 0.24 with negatives in it. */
    float zero_centered[64];
    for (int i = 0; i < 64; i++) zero_centered[i] = 0.2377f + (float)sin((double)i) * 0.15f;
    check(q4nx_rmsnorm_offset_from_norm(zero_centered, 64) == 1,
          "mean 0.24 with negatives -> (1 + w)");

    /* A delta whose negatives happen to be absent still has a mean far below 1. */
    float no_negatives[64];
    for (int i = 0; i < 64; i++) no_negatives[i] = 0.35f + 0.01f * (float)(i % 7);
    check(q4nx_rmsnorm_offset_from_norm(no_negatives, 64) == 1,
          "all-positive delta, mean 0.38 -> (1 + w)");

    /* Shape 2: plain gain (the hybrid 27B pack). Every value around 1, none negative. */
    float plain[64];
    for (int i = 0; i < 64; i++) plain[i] = 0.9704f + (float)sin((double)i) * 0.06f;
    check(q4nx_rmsnorm_offset_from_norm(plain, 64) == 0, "mean 0.97, min 0.87 -> plain w");

    /* Exactly 1 everywhere (an untrained plain norm) must stay plain. */
    float ones[8] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    check(q4nx_rmsnorm_offset_from_norm(ones, 8) == 0, "all ones -> plain w");

    /* A plain gain with a single negative gain is the ambiguous case; the rule takes the (1 + w) form
     * because a plain norm weight with a negative channel is not a trained gain. */
    float mixed[4] = {1.1f, 0.9f, -0.2f, 1.0f};
    check(q4nx_rmsnorm_offset_from_norm(mixed, 4) == 1, "a negative plain gain -> (1 + w)");

    /* Degenerate inputs must not be read as a convention. */
    check(q4nx_rmsnorm_offset_from_norm(NULL, 16) == 0, "null -> plain w");
    check(q4nx_rmsnorm_offset_from_norm(ones, 0) == 0, "empty -> plain w");

    /* And the real tensors' two statistics, as constants: if a future pack ever lands between the two
     * shapes, whichever side of 0.5 it falls on is visible here rather than in a serving log. */
    float real_d3[4] = {1.0f, 0.2949f, 0.5352f, -0.1143f};
    float real_27b[4] = {0.9704f, 0.8711f, 1.2344f, 0.9500f};
    check(q4nx_rmsnorm_offset_from_norm(real_d3, 4) == 1, "d3-lite layer-0 head -> (1 + w)");
    check(q4nx_rmsnorm_offset_from_norm(real_27b, 4) == 0, "bonsai 27B layer-0 head -> plain w");

    /* The mask rule (engine#796). A d3 pack declares `noncausal_full_attention` in its own
     * decision_config.json; that declaration has to decide when there is no plan, and a plan that
     * contradicts it has to be refused. Measured on the real pack: honouring the declaration gives
     * worst |dp| 0.0246 against the pack's d3_runtime.py, ignoring it 0.3702 with a choice flip. */
    char why[512];
    check(q4nx_attention_mode("noncausal_full_attention", 0, 0, why, sizeof why) == 1,
          "a noncausal declaration with no plan is honoured (the pack's own word)");
    check(q4nx_attention_mode("noncausal_full_attention", 1, 1, why, sizeof why) == 1,
          "a noncausal declaration and a noncausal plan agree");
    check(q4nx_attention_mode(NULL, 1, 1, why, sizeof why) == 1,
          "no declaration: the plan decides (noncausal)");
    check(q4nx_attention_mode(NULL, 0, 0, why, sizeof why) == 0,
          "no declaration and no plan: causal, every other model on the box");
    check(q4nx_attention_mode("causal", 0, 0, why, sizeof why) == 0,
          "a causal declaration is causal");
    check(q4nx_attention_mode("", 0, 0, why, sizeof why) == 0,
          "an empty declaration (a decision_config.json with no attention_mode) is causal");
    check(q4nx_attention_mode("noncausal_full_attention", 1, 0, why, sizeof why) == -1 &&
          strstr(why, "stale") && strstr(why, "tools/npu_plan.py"),
          "a plan that says causal for a noncausal pack is refused, naming itself stale and the command");
    check(q4nx_attention_mode("causal", 1, 1, why, sizeof why) == -1 && strstr(why, "stale"),
          "the disagreement is refused in both directions, not only the one this issue measured");

    /* The prefill block width (engine#1582): a note only when the batch was blocked BELOW the prefill width, and
     * it must carry the launch counts that make the cost legible (14 launches per projection for 219 rows at 16). */
    check(strstr(q4nx_block_note(219, 128, 128), "blocked") == NULL,
          "a 219-row prefill at the 128-row width says nothing (it was not blocked)");
    const char* blocked = q4nx_block_note(219, 16, 128);
    check(strstr(blocked, "M=219") && strstr(blocked, "14 launch") && strstr(blocked, "make it 2"),
          "a 219-row prefill at 16 rows/launch names the width, the launch count and the prefill alternative");
    check(strstr(q4nx_block_note(219, 1, 128), "219 launch") != NULL,
          "at one row per launch the note says 219, which is the pathological case");
    check(q4nx_block_note(0, 16, 128)[0] == '\0' && q4nx_block_note(10, 16, 128)[0] == '\0',
          "a batch inside the block, or an empty one, prints nothing");

    /* engine#1788: per-head q/k norm, decided by the container when the name does not decide it. The config is the
     * published MiniCPM-V 4.7 1B pack's: model_type minicpmv4_7 around a qwen3_5 text model whose full-attention
     * layers (3, 7, 11, ...) ship q_norm and k_norm. */
    {
        const char* path = "npu_rmsnorm_offset_test.config.json";
        const unsigned char full[8] = {0, 0, 0, 1, 0, 0, 0, 1};   /* layers 3 and 7 carry the tensor */
        const unsigned char none[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        Q4nxSemantics sem;
        FILE* f = onebit_fopen_private_wb(path);
        if (f) {
            fputs("{\"model_type\": \"minicpmv4_7\", \"architectures\": [\"MiniCPMV4_7ForConditionalGeneration\"], "
                  "\"hidden_size\": 1024, \"num_hidden_layers\": 8, \"hidden_act\": \"silu\"}", f);
            fclose(f);
        }
        check(f && q4nx_semantics_from_config(path, ".", &sem) == 0 && sem.qk_norm == 0,
              "minicpmv4_7: the name alone says no q/k norm (the gap)");
        check(q4nx_semantics_qk_norm_from_tensors(&sem, full, full, 8) == 3 && sem.qk_norm == 1,
              "... and a layer carrying q_norm and k_norm turns it on, naming layer 3");
        check(q4nx_semantics_qk_norm_from_tensors(&sem, full, full, 8) == -1 && sem.qk_norm == 1,
              "already on: nothing changes and no layer is named");
        sem.qk_norm = 0;
        check(q4nx_semantics_qk_norm_from_tensors(&sem, none, none, 8) == -1 && sem.qk_norm == 0,
              "no layer carries the tensors: it stays off (llama, phi)");
        check(q4nx_semantics_qk_norm_from_tensors(&sem, full, none, 8) == -1 && sem.qk_norm == 0,
              "q_norm without k_norm is not the pair: it stays off");
        check(q4nx_semantics_qk_norm_from_tensors(&sem, none, full, 8) == -1 && sem.qk_norm == 0,
              "k_norm without q_norm is not the pair either");
        check(q4nx_semantics_qk_norm_from_tensors(&sem, full, full, 3) == -1 && sem.qk_norm == 0,
              "only the layers that exist are read (3 layers: the pair at layer 3 is out of range)");
        check(q4nx_semantics_qk_norm_from_tensors(&sem, NULL, full, 8) == -1 && q4nx_semantics_qk_norm_from_tensors(NULL, full, full, 8) == -1 &&
                  q4nx_semantics_qk_norm_from_tensors(&sem, full, full, 0) == -1,
              "null or empty input changes nothing");
        f = onebit_fopen_private_wb(path);
        if (f) {
            fputs("{\"model_type\": \"qwen3_5\", \"hidden_size\": 1024, \"num_hidden_layers\": 8}", f);
            fclose(f);
        }
        check(f && q4nx_semantics_from_config(path, ".", &sem) == 0 && sem.qk_norm == 1 &&
                  q4nx_semantics_qk_norm_from_tensors(&sem, none, none, 8) == -1 && sem.qk_norm == 1,
              "qwen3_5: on by name, and absent tensors never turn it off");
        remove(path);
    }

    printf(failures ? "npu_rmsnorm_offset: %d FAILED\n" : "npu_rmsnorm_offset: ok\n", failures);
    return failures ? 1 : 0;
}
