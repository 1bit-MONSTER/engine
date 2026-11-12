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

/* q4nx_lfm3_semantics_test -- what the forward reads from an LFM3 config and container (engine#1786).
 *
 * FastFlowLM's LFM3-1.7B-NPU2 is LFM2's conv/attention hybrid with four more rules (LiquidAI's modeling_lfm3.py):
 * layer_types names the conv layers ("conv") beside full and sliding attention; the norms are a sandwich around
 * both blocks (use_sandwich_norm); each attention head's context is gated by sigmoid(gate_proj . x); and the full
 * layers' partial RoPE keeps the full head_dim's frequency progression with a zero tail. Its container also keeps
 * the upstream tensor names, which model_load renames to the canonical ones. Before this, the forward refused the
 * pack by name at "conv".
 *
 * The config string is cut down from the pack's config.json to the fields the rules read. No model, no device. */
#include "model.h"
#include "q4nx_semantics.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

static int sem_of(const char* dir, const char* json, Q4nxSemantics* s) {
    char path[512];
    std::snprintf(path, sizeof path, "%s/config.json", dir);
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE* f = fd >= 0 ? fdopen(fd, "w") : nullptr;
    if (!f) { std::printf("FAIL cannot write %s\n", path); failures++; return -1; }
    std::fputs(json, f);
    std::fclose(f);
    if (q4nx_semantics_from_config(path, dir, s) != 0) { std::printf("FAIL cannot read %s\n", path); failures++; return -1; }
    return 0;
}

static const char* kLfm3 =
    "{\"architectures\": [\"Lfm3ForCausalLM\"], \"model_type\": \"lfm3\", \"head_dim\": 128, \"hidden_size\": 2048,"
    " \"num_attention_heads\": 16, \"num_key_value_heads\": 4, \"num_hidden_layers\": 6, \"norm_eps\": 1e-05,"
    " \"layer_types\": [\"conv\", \"conv\", \"full_attention\", \"conv\", \"sliding_attention\", \"conv\"],"
    " \"rope_parameters\": {\"full_attention\": {\"partial_rotary_factor\": 0.5, \"rope_theta\": 1000000.0,"
    " \"rope_type\": \"default\"}, \"sliding_attention\": {\"rope_theta\": 10000, \"rope_type\": \"default\"}},"
    " \"sliding_window\": 256, \"use_sandwich_norm\": true, \"conv_L_cache\": 7}";

static int find(const std::vector<TensorDesc>& t, const char* name) {
    for (size_t i = 0; i < t.size(); i++)
        if (std::strcmp(t[i].name, name) == 0) return (int)i;
    return -1;
}

static std::vector<TensorDesc> table(const std::vector<const char*>& names) {
    std::vector<TensorDesc> t(names.size());
    for (size_t i = 0; i < names.size(); i++) {
        std::memset(&t[i], 0, sizeof(TensorDesc));
        std::snprintf(t[i].name, sizeof(t[i].name), "%s", names[i]);
        t[i].data_offset = (uint64_t)i;   // the tag that follows a tensor through a rename
    }
    return t;
}

int main() {
    char dir[] = "/tmp/q4nx-lfm3-sem-XXXXXX";
    if (!mkdtemp(dir)) { std::printf("FAIL mkdtemp\n"); return 1; }
    Q4nxSemantics s;

    if (sem_of(dir, kLfm3, &s) == 0) {
        int conv = 0, full = 0, sliding = 0;
        for (int i = 0; i < s.n_layer_types; i++) {
            conv += s.layer_types[i] == Q4NX_LAYER_CONV;
            full += s.layer_types[i] == Q4NX_LAYER_FULL;
            sliding += s.layer_types[i] == Q4NX_LAYER_SLIDING;
        }
        check(s.unknown_layer_type[0] == '\0', "LFM3: \"conv\" in layer_types is a layer kind the forward knows");
        check(s.n_layer_types == 6 && conv == 4 && full == 1 && sliding == 1 &&
                  s.layer_types[0] == Q4NX_LAYER_CONV && s.layer_types[2] == Q4NX_LAYER_FULL,
              "... 4 conv, 1 full and 1 sliding layer, in the stated order");
        check(s.extra_layer_norms == 2, "use_sandwich_norm: the two extra norms around both blocks");
        check(s.attn_head_gate == 1, "LFM3 gates each attention head");
        check(s.proportional_rope == 1 && s.rotary_dim == 64 && s.rope_theta == 1000000.0f,
              "full layers: zero-tail partial RoPE over the whole head (64 of 128 channels' frequencies), theta 1e6");
        check(s.rope_theta_sliding == 10000.0f, "sliding layers: their own theta");
        check(s.qk_norm == 1 && s.rmsnorm_offset == 0, "q/k norms on, plain norm form");
    }
    if (sem_of(dir, "{\"model_type\": \"lfm3\", \"num_hidden_layers\": 2, \"use_sandwich_norm\": false,"
                    " \"layer_types\": [\"conv\", \"full_attention\"]}", &s) == 0)
        check(s.extra_layer_norms == 0, "use_sandwich_norm false: no extra norms");
    if (sem_of(dir, "{\"model_type\": \"lfm2\", \"hidden_size\": 1024, \"full_attn_idxs\": [2]}", &s) == 0)
        check(s.attn_head_gate == 0 && s.extra_layer_norms == 0 && s.proportional_rope == 0,
              "LFM2 keeps what it had: no gate, no sandwich, no zero-tail RoPE");
    if (sem_of(dir, "{\"model_type\": \"qwen3\", \"num_hidden_layers\": 2,"
                    " \"layer_types\": [\"conv_attention\", \"full_attention\"]}", &s) == 0)
        check(std::strcmp(s.unknown_layer_type, "conv_attention") == 0,
              "an exact match: \"conv_attention\" is still an unknown kind, refused by name");

    {
        std::vector<TensorDesc> t = table({
            "model.embed_tokens.weight", "model.embedding_norm.weight",
            "model.layers.0.conv.in_proj.weight", "model.layers.0.conv.conv.weight", "model.layers.0.conv.out_proj.weight",
            "model.layers.0.operator_norm.weight", "model.layers.0.post_operator_norm.weight",
            "model.layers.0.ffn_norm.weight", "model.layers.0.post_ffn_norm.weight",
            "model.layers.0.feed_forward.w1.weight", "model.layers.0.feed_forward.w2.weight",
            "model.layers.0.feed_forward.w3.weight",
            "model.layers.1.operator_norm.weight", "model.layers.1.ffn_norm.weight",
            "model.layers.1.self_attn.gate_proj.weight"});
        const int n = model_alias_lfm_names(t.data(), (int)t.size());
        auto at = [&](const char* name) { const int i = find(t, name); return i < 0 ? -1 : (int)t[(size_t)i].data_offset; };
        check(n == 13, "13 of the 15 names are renamed (the embedding and the gate keep theirs)");
        check(at("model.norm.weight") == 1, "embedding_norm is the final norm");
        check(at("model.layers.0.shortconv.in_proj.weight") == 2 && at("model.layers.0.shortconv.conv.weight") == 3 &&
                  at("model.layers.0.shortconv.out_proj.weight") == 4,
              "conv.* -> shortconv.* (conv.conv is the taps)");
        check(at("model.layers.0.input_layernorm.weight") == 5, "operator_norm -> input_layernorm");
        check(at("model.layers.0.post_attention_layernorm.weight") == 6 &&
                  at("model.layers.0.pre_feedforward_layernorm.weight") == 7 &&
                  at("model.layers.0.post_feedforward_layernorm.weight") == 8,
              "sandwich layer: post_operator_norm is the block output's norm, ffn_norm the MLP input's");
        check(at("model.layers.0.mlp.gate_proj.weight") == 9 && at("model.layers.0.mlp.down_proj.weight") == 10 &&
                  at("model.layers.0.mlp.up_proj.weight") == 11,
              "w1 is gate, w2 is down, w3 is up (Lfm3MLP: w2(silu(w1 x) * w3 x))");
        check(at("model.layers.1.post_attention_layernorm.weight") == 13,
              "a layer without post_operator_norm (LFM2's upstream form): ffn_norm is post_attention_layernorm");
        check(at("model.layers.1.self_attn.gate_proj.weight") == 14, "the attention gate keeps its name");
    }
    {
        std::vector<TensorDesc> t = table({"model.norm.weight", "model.embedding_norm.weight",
                                           "model.layers.0.input_layernorm.weight", "model.layers.0.operator_norm.weight"});
        check(model_alias_lfm_names(t.data(), (int)t.size()) == 0 && find(t, "model.embedding_norm.weight") == 1 &&
                  find(t, "model.layers.0.operator_norm.weight") == 3,
              "a canonical name already present is never overwritten");
        std::vector<TensorDesc> q = table({"model.layers.0.shortconv.conv.weight", "model.layers.0.mlp.gate_proj.weight"});
        check(model_alias_lfm_names(q.data(), (int)q.size()) == 0, "an LFM2 pack's canonical names are left alone");
    }

    char path[512];
    std::snprintf(path, sizeof path, "%s/config.json", dir);
    unlink(path);
    rmdir(dir);
    std::printf(failures ? "q4nx_lfm3_semantics: %d FAILED\n" : "q4nx_lfm3_semantics: ok\n", failures);
    return failures ? 1 : 0;
}
