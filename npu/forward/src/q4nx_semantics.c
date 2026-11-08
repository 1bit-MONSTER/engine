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
// q4nx_semantics.c — see q4nx_semantics.h.
//
// A deliberately tiny JSON field reader: these config.json files are small flat
// maps, and the goal is only to read named scalar fields with defined defaults.
#include "q4nx_semantics.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* kFields[Q4NX_SEM_COUNT] = {
    /* RMSNORM_OFFSET  */ "model_type/architectures (gemma* uses the 1+w form)",
    /* QK_NORM         */ "model_type/architectures (q_norm/k_norm tensors)",
    /* ROPE_THETA      */ "rope_theta",
    /* ROTARY_DIM      */ "partial_rotary_factor * head_dim",
    /* MLP_ACT         */ "hidden_act (falls back to model_type)",
    /* ATTENTION_BIAS  */ "attention_bias (or presence of q/k/v bias tensors)",
    /* MLP_BIAS        */ "mlp_bias (or presence of gate/up/down bias tensors)",
    /* EMBED_SCALE     */ "model_type (gemma* uses sqrt(hidden_size))",
    /* LOGIT_SOFTCAP   */ "final_logit_softcapping",
    /* SLIDING_WINDOW  */ "sliding_window",
    /* EPS             */ "rms_norm_eps",
    /* HYBRID_LAYERS   */ "layer_types (count of entries != full_attention)",
    /* MROPE           */ "rope_parameters.mrope_interleaved (rope_type)",
    /* PER_LAYER_ROPE  */ "rope_parameters.{full,sliding}_attention sub-blocks",
    /* LAYER_TYPES     */ "layer_types",
};

const char* q4nx_semantics_field(int flag) {
    if (flag < 0 || flag >= Q4NX_SEM_COUNT) return "?";
    return kFields[flag];
}

void q4nx_semantics_default(Q4nxSemantics* s) {
    if (!s) return;
    memset(s, 0, sizeof(*s));
    s->rope_theta = 10000.0f;   /* HF default when rope_theta is absent */
    s->mlp_act = Q4NX_ACT_SILU;
    s->embed_scale = 1.0f;
    s->eps = 1e-6f;
}

// ---- minimal JSON helpers -------------------------------------------------

static char* slurp(const char* path, long* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)n, f);
    buf[rd] = 0;
    fclose(f);
    if (out_len) *out_len = (long)rd;
    return buf;
}

// Find "key" and return a pointer to the value that follows the colon.
static const char* value_for(const char* json, const char* key) {
    if (!json || !key) return NULL;
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char* p = json;
    while ((p = strstr(p, pat)) != NULL) {
        const char* q = p + strlen(pat);
        while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
        if (*q == ':') {
            q++;
            while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
            return q;
        }
        p += strlen(pat);
    }
    return NULL;
}

// Return a pointer to the '}' matching the '{' at p (string-aware), or NULL.
// A plain strchr(p, '}') is wrong here: it stops at the FIRST closing brace,
// which for rope_parameters is the end of the full_attention sub-object, leaving
// the sliding_attention block outside the extracted slice (that is why Gemma4's
// sliding theta read back as the full-attention theta).
static const char* match_brace(const char* p) {
    if (!p || *p != '{') return NULL;
    int depth = 0, in_str = 0;
    for (const char* q = p; *q; q++) {
        if (in_str) {
            if (*q == '\\') { if (q[1]) q++; continue; }
            if (*q == '"') in_str = 0;
            continue;
        }
        if (*q == '"') { in_str = 1; continue; }
        if (*q == '{') depth++;
        else if (*q == '}') { if (--depth == 0) return q; }
    }
    return NULL;
}

// Like value_for(), but only matches keys inside the OUTERMOST object (brace
// depth 1).  Without this a flat scan reads sub-tower configs: Gemma4's
// hidden_size came from audio_config (1024 instead of 1536), and its rope_theta
// from rope_parameters.full_attention.  Passing a sub-object (e.g. the
// rope_parameters text) works too, since its own outermost braces are depth 1.
static const char* value_for_top(const char* json, const char* key) {
    if (!json || !key) return NULL;
    char pat[128];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    int depth = 0, in_str = 0;
    for (const char* p = json; *p; p++) {
        if (in_str) {
            if (*p == '\\') { if (p[1]) p++; continue; }
            if (*p == '"') in_str = 0;
            continue;
        }
        if (*p == '"') {
            if (depth == 1 && strncmp(p, pat, strlen(pat)) == 0) {
                const char* q = p + strlen(pat);
                while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                if (*q == ':') {
                    q++;
                    while (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r') q++;
                    return q;
                }
            }
            in_str = 1;
            continue;
        }
        if (*p == '{' || *p == '[') depth++;
        else if (*p == '}' || *p == ']') depth--;
    }
    return NULL;
}

static int get_str(const char* json, const char* key, char* out, size_t out_sz) {
    const char* v = value_for_top(json, key);
    if (!v || *v != '"') return 0;
    v++;
    size_t i = 0;
    while (*v && *v != '"' && i + 1 < out_sz) out[i++] = *v++;
    out[i] = 0;
    return 1;
}

static int get_num(const char* json, const char* key, double* out) {
    const char* v = value_for_top(json, key);
    if (!v) return 0;
    if (strncmp(v, "null", 4) == 0) return 0;
    if (strncmp(v, "true", 4) == 0) { *out = 1; return 1; }
    if (strncmp(v, "false", 5) == 0) { *out = 0; return 1; }
    char* end = NULL;
    double d = strtod(v, &end);
    if (end == v) return 0;
    *out = d;
    return 1;
}

static int starts_with(const char* s, const char* p) {
    return s && p && strncmp(s, p, strlen(p)) == 0;
}

int q4nx_semantics_from_config(const char* json_path, const char* model_dir,
                               Q4nxSemantics* s) {
    (void)model_dir;   /* reserved: tensor-level probing lives in the caller */
    if (!json_path || !s) return -1;
    q4nx_semantics_default(s);

    long len = 0;
    char* json = slurp(json_path, &len);
    if (!json) return -1;

    char model_type[64] = {0}, arch[128] = {0}, act[64] = {0};
    get_str(json, "model_type", model_type, sizeof(model_type));
    /* architectures is an array; read its first string element */
    {
        const char* v = value_for_top(json, "architectures");
        if (v && *v == '[') {
            const char* q = strchr(v, '"');
            if (q) {
                q++;
                size_t i = 0;
                while (*q && *q != '"' && i + 1 < sizeof(arch)) arch[i++] = *q++;
                arch[i] = 0;
            }
        }
    }
    /* activation: HF uses hidden_act or hidden_activation */
    if (!get_str(json, "hidden_act", act, sizeof(act)))
        get_str(json, "hidden_activation", act, sizeof(act));

    const int is_gemma = starts_with(model_type, "gemma") || starts_with(arch, "Gemma");
    /* Gemma4 is NOT the (1+w) form: Gemma4RMSNorm does `x/rms * w` with w
     * initialised to ones, whereas gemma2/gemma3 do `x/rms * (1 + w)`. */
    const int is_gemma4 = strstr(model_type, "gemma4") != NULL ||
                          strstr(arch, "Gemma4") != NULL;
    const int is_qwen3 = starts_with(model_type, "qwen3") || starts_with(arch, "Qwen3");
    const int is_phi = starts_with(model_type, "phi") || starts_with(arch, "Phi");
    const int is_qwen35 = starts_with(model_type, "qwen3_5") || starts_with(arch, "Qwen3_5");

    /* RMSNorm convention: gemma2/gemma3 scale by (1 + w); gemma4 by w. */
    s->rmsnorm_offset = (is_gemma && !is_gemma4) ? 1 : 0;
    /* qwen3_5 ships both conventions under the same model_type, so the forward resolves it from the
     * layer-0 input_layernorm weight it loads (a config-only answer would be a guess). */
    s->rmsnorm_offset_probe = (is_qwen35 && !is_gemma) ? 1 : 0;

    /* per-head q/k RMSNorm: qwen3* and gemma* have q_norm/k_norm; llama/phi do not */
    s->qk_norm = (is_qwen3 || is_gemma) ? 1 : 0;

    /* RoPE base.  Qwen3.5 nests it under rope_parameters (scalar); Gemma4 nests
     * it PER LAYER TYPE under rope_parameters.{full,sliding}_attention with
     * different thetas and types.  The latter is a variant this forward does not
     * implement, so it is flagged via per_layer_rope/mrope rather than silently
     * picking one of the two. */
    double d = 0;
    double prf_rope = 0;   /* partial_rotary_factor found inside rope_parameters */
    /* Deepstack layers: the vision tower emits extra feature sets that the decoder adds at these layer indexes. */
    {
        const char* vc = value_for_top(json, "vision_config");
        const char* at = vc ? strstr(vc, "deepstack_visual_indexes") : NULL;
        if (at) at = strchr(at, '[');
        if (at) {
            int got = 0;
            const char* p3 = at + 1;
            while (*p3 && *p3 != ']' && got < 8) {
                if (*p3 >= '0' && *p3 <= '9') {
                    char* end = NULL;
                    s->deepstack_layers[got++] = (int)strtol(p3, &end, 10);
                    p3 = end;   /* strtol already stopped on the separator; a for-clause p3++ would skip the ']' */
                } else {
                    p3++;
                }
            }
            s->n_deepstack = got;
        }
    }
    /* The sections say WHICH rotary frequencies belong to the (t, h, w) axes; without them an interleaved-MRoPE
     * consumer has nothing but a guess, so they are read next to the flag (both spellings) and a model that
     * declares mrope without them is refused by the consumer rather than assumed. */
    {
        const char* blocks[2] = {value_for_top(json, "rope_scaling"), value_for_top(json, "rope_parameters")};
        for (int b = 0; b < 2 && s->n_mrope_section == 0; b++) {
            const char* blk = blocks[b];
            if (!blk || *blk != '{') continue;
            const char* at = strstr(blk, "mrope_section");
            if (!at) continue;
            at = strchr(at, '[');
            if (!at) continue;
            int got = 0;
            const char* p2 = at + 1;
            while (*p2 && *p2 != ']' && got < 3) {
                if (*p2 >= '0' && *p2 <= '9') {
                    char* end = NULL;
                    s->mrope_section[got++] = (int)strtol(p2, &end, 10);
                    p2 = end;   /* as above: let strtol move the cursor, not the loop */
                } else {
                    p2++;
                }
            }
            s->n_mrope_section = got;
        }
    }
    /* Interleaved MRoPE is declarable in two spellings and only one was read here.  FastFlowLM/OGA packs and
     * Qwen3.5 nest the rotary block under `rope_parameters` (the block below); a Hugging Face `qwen3_vl` config
     * (Qwen3-VL, engine #1311/TQVL) declares the same thing under `rope_scaling`.  Reading only the first meant
     * such a container loaded as mrope=0 -- plain RoPE, silently: exact for a text-only prompt (with
     * pos_t = pos_h = pos_w the two are bit-identical, measured), wrong by ~0.29 rad on the rotary block once
     * image tokens are in it.  tests/q4nx_mrope_test.c pins both spellings and the controls (a yarn/linear
     * rope_scaling block must NOT be read as mrope). */
    {
        const char* rs = value_for_top(json, "rope_scaling");
        double mr = 0;
        if (rs && *rs == '{' && get_num(rs, "mrope_interleaved", &mr) && mr != 0) s->mrope = 1;
    }
    const char* rope_blk = value_for_top(json, "rope_parameters");
    if (rope_blk && *rope_blk == '{') {
        const char* end = match_brace(rope_blk);
        size_t blen = end ? (size_t)(end - rope_blk) + 1 : strlen(rope_blk);
        char* sub = (char*)malloc(blen + 1);
        if (sub) {
            memcpy(sub, rope_blk, blen);
            sub[blen] = 0;
            const int has_scalar = get_num(sub, "rope_theta", &d);
            if (has_scalar && d > 0) s->rope_theta = (float)d;
            get_num(sub, "partial_rotary_factor", &prf_rope);
            if (get_num(sub, "mrope_interleaved", &d) && d != 0) s->mrope = 1;
            char rt[32] = {0};
            if (get_str(sub, "rope_type", rt, sizeof(rt)) && rt[0] &&
                strcmp(rt, "default") != 0)
                s->rope_type_named = 1;   /* e.g. gemma4 "proportional" */
            if (!has_scalar) {
                /* per-layer-type form: read the full_attention block as the
                 * representative theta, and flag the model as unimplemented. */
                const char* fa = value_for(sub, "full_attention");
                if (fa && *fa == '{') {
                    const char* fe = match_brace(fa);
                    size_t fl = fe ? (size_t)(fe - fa) + 1 : strlen(fa);
                    char* fsub = (char*)malloc(fl + 1);
                    if (fsub) {
                        memcpy(fsub, fa, fl);
                        fsub[fl] = 0;
                        if (get_num(fsub, "rope_theta", &d) && d > 0)
                            s->rope_theta = (float)d;
                        get_num(fsub, "partial_rotary_factor", &prf_rope);
                        char frt[32] = {0};
                        if (get_str(fsub, "rope_type", frt, sizeof(frt)) && frt[0] &&
                            strcmp(frt, "default") != 0)
                            s->rope_type_named = 1;
                        free(fsub);
                    }
                    s->per_layer_rope = 1;
                    /* the sliding_attention block, used by sliding layers */
                    s->rope_theta_sliding = s->rope_theta;
                    s->rotary_dim_sliding = s->rotary_dim;
                    const char* sa = value_for(sub, "sliding_attention");
                    if (sa && *sa == '{') {
                        const char* se = match_brace(sa);
                        size_t sl = se ? (size_t)(se - sa) + 1 : strlen(sa);
                        char* ssub = (char*)malloc(sl + 1);
                        if (ssub) {
                            memcpy(ssub, sa, sl);
                            ssub[sl] = 0;
                            if (get_num(ssub, "rope_theta", &d) && d > 0)
                                s->rope_theta_sliding = (float)d;
                            double sprf = 0, shd = 0;
                            if (get_num(ssub, "partial_rotary_factor", &sprf) && sprf > 0 &&
                                sprf < 1.0 && get_num(json, "head_dim", &shd) && shd > 0)
                                s->rotary_dim_sliding = (int)(sprf * shd + 0.5);
                            free(ssub);
                        }
                    }
                }
            }
            free(sub);
        }
    } else if (get_num(json, "rope_theta", &d) && d > 0) {
        s->rope_theta = (float)d;
    }

    /* partial rotary: top-level key, or nested in rope_parameters (scalar form),
     * or nested per layer type (Gemma4's full_attention.partial_rotary_factor). */
    double hs = 0, hd = 0, prf = 0;
    const int have_hd = get_num(json, "head_dim", &hd) && hd > 0;
    int have_prf = get_num(json, "partial_rotary_factor", &prf) && prf > 0 && prf < 1.0;
    if (!have_prf && prf_rope > 0 && prf_rope < 1.0) { prf = prf_rope; have_prf = 1; }
    /* Gemma4's full_attention layers use global_head_dim (512), not head_dim
     * (256), so the FULL rotary window must be sized from that dimension; with
     * no num_global_key_value_heads the head count is the same for both types. */
    if (get_num(json, "global_head_dim", &d) && d > 0)
        s->global_head_dim = (int)d;
    if (get_num(json, "num_global_key_value_heads", &d) && d > 0)
        s->num_global_kv_heads = (int)d;
    const double full_hd = s->global_head_dim > 0 ? (double)s->global_head_dim : hd;
    if (have_prf && (have_hd || s->global_head_dim > 0))
        s->rotary_dim = (int)(prf * full_hd + 0.5);
    else
        s->rotary_dim = 0;   /* full head_dim */

    /* Gemma4 attention semantics: scaling is 1.0 (q/k carry an RMSNorm), v is
     * normalised with a SCALE-FREE RMSNorm, and the full-attention layers use
     * "proportional" RoPE (rope_angles = partial*head_dim//2, rest identity). */
    s->attn_scaling_one = is_gemma4 ? 1 : 0;
    s->v_norm = is_gemma4 ? 1 : 0;
    s->proportional_rope = (is_gemma4 && s->per_layer_rope &&
                            strstr(json, "\"proportional\"") != NULL) ? 1 : 0;

    /* LM head: tied to the token embedding (the common case) or a SEPARATE
     * lm_head.weight (Nanbeige4.1-3B sets tie_word_embeddings = false). */
    s->tie_embeddings = 1;
    if (get_num(json, "tie_word_embeddings", &d)) s->tie_embeddings = d != 0;

    /* layer_types -> the per-layer attention kind the forward dispatches on */
    {
        const char* lt = value_for(json, "layer_types");
        if (lt && *lt == '[') {
            const char* end = strchr(lt, ']');
            const char* p = lt;
            int hybrid = 0, total = 0;
            while (p && (!end || p < end)) {
                const char* q = strchr(p, '"');
                if (!q || (end && q > end)) break;
                q++;
                const char* qe = strchr(q, '"');
                if (!qe) break;
                char nm[48];
                size_t n = (size_t)(qe - q);
                if (n >= sizeof(nm)) n = sizeof(nm) - 1;
                memcpy(nm, q, n);
                nm[n] = 0;
                int kind = Q4NX_LAYER_OTHER;
                /* EXACT match, not a substring, on the spellings that name attention math this forward has.  A
                 * substring test made the guard as wide as the substring: "chunked_linear_attention",
                 * "mamba2_linear" and "nonlinear_attention" all contain "linear" and were mapped to the GDN
                 * recurrence and served, fluently and silently, instead of refused (engine#1156).  Anything else
                 * stays OTHER, which is what the forward refuses by name below. */
                if (strcmp(nm, "full_attention") == 0) kind = Q4NX_LAYER_FULL;
                else if (strcmp(nm, "sliding_attention") == 0) kind = Q4NX_LAYER_SLIDING;
                else if (strcmp(nm, "linear_attention") == 0) kind = Q4NX_LAYER_LINEAR;
                else if (!s->unknown_layer_type[0]) {
                    /* Remember the spelling: the forward refuses this container by name rather than serving an
                     * unimplemented layer kind as full attention (engine#1087's shape). */
                    size_t k = strlen(nm);
                    if (k >= sizeof(s->unknown_layer_type)) k = sizeof(s->unknown_layer_type) - 1;
                    memcpy(s->unknown_layer_type, nm, k);
                    s->unknown_layer_type[k] = 0;
                }
                if (total < Q4NX_MAX_LAYERS) s->layer_types[total] = kind;
                total++;
                if (kind != Q4NX_LAYER_FULL) hybrid++;
                p = qe + 1;
            }
            if (total > 0) {
                s->hybrid_layers = hybrid;
                s->n_layer_types = total < Q4NX_MAX_LAYERS ? total : Q4NX_MAX_LAYERS;
            }
        }
    }

    /* MLP activation */
    if (strstr(act, "gelu")) s->mlp_act = Q4NX_ACT_GELU;
    else if (act[0]) s->mlp_act = Q4NX_ACT_SILU;
    else s->mlp_act = is_gemma ? Q4NX_ACT_GELU : Q4NX_ACT_SILU;  /* gemma omits hidden_act */

    /* biases: absent field means "no bias" for llama-family configs, but Phi
     * and Gemma configs omit it as well, so default to 0 and let phase 2's
     * tensor-level probe (bias tensors present?) override it. */
    if (get_num(json, "attention_bias", &d)) s->attention_bias = d != 0;
    if (get_num(json, "mlp_bias", &d)) s->mlp_bias = d != 0;

    /* embedding scale: gemma scales embeddings by sqrt(hidden_size) */
    if (is_gemma && get_num(json, "hidden_size", &hs) && hs > 0)
        s->embed_scale = (float)sqrt(hs);
    else
        s->embed_scale = 1.0f;

    /* final-logit soft cap */
    if (get_num(json, "final_logit_softcapping", &d) && d > 0) s->logit_softcap = (float)d;
    else if (get_num(json, "final_logit_softcap", &d) && d > 0) s->logit_softcap = (float)d;

    /* sliding window: only when the config also enables it (qwen3 sets
     * use_sliding_window=false and sliding_window=null) */
    double sw = 0, use_sw = 1;
    get_num(json, "use_sliding_window", &use_sw);
    if (get_num(json, "sliding_window", &sw) && sw > 0 && use_sw != 0)
        s->sliding_window = (int)sw;
    else
        s->sliding_window = 0;

    /* epsilon */
    if (get_num(json, "rms_norm_eps", &d) && d > 0) s->eps = (float)d;

    /* ── Gemma4-style per-layer embeddings (PLE) + KV sharing ── */
    if (get_num(json, "hidden_size_per_layer_input", &d) && d > 0)
        s->hidden_size_per_layer_input = (int)(d + 0.5);
    if (get_num(json, "vocab_size_per_layer_input", &d) && d > 0)
        s->vocab_size_per_layer_input = (int)(d + 0.5);
    if (get_num(json, "num_kv_shared_layers", &d) && d > 0)
        s->num_kv_shared_layers = (int)(d + 0.5);
    /* Gemma4 adds pre/post feedforward norms and a post-PLE norm. */
    s->extra_layer_norms = is_gemma4 ? 1 : 0;
    /* `use_double_wide_mlp` is a JSON boolean, so read it as "present in text". */
    s->double_wide_mlp = (strstr(json, "use_double_wide_mlp\":true") != NULL ||
                          strstr(json, "use_double_wide_mlp\": true") != NULL) ? 1 : 0;

    (void)is_phi;
    (void)len;
    free(json);
    return 0;
}

int q4nx_config_int(const char* json_path, const char* key, int def) {
    if (!json_path || !key) return def;
    long len = 0;
    char* json = slurp(json_path, &len);
    if (!json) return def;
    double d = 0;
    const int ok = get_num(json, key, &d);
    free(json);
    return ok ? (int)(d + (d < 0 ? -0.5 : 0.5)) : def;
}

int q4nx_semantics_uses_ple(const Q4nxSemantics* s) {
    return s && s->hidden_size_per_layer_input > 0;
}

int q4nx_rmsnorm_offset_from_norm(const float* w, int n) {
    if (!w || n <= 0) return 0;
    double sum = 0.0;
    float mn = w[0];
    for (int i = 0; i < n; i++) {
        sum += (double)w[i];
        if (w[i] < mn) mn = w[i];
    }
    /* A (1 + w) checkpoint's norm weight is a delta trained from 0, so it carries negatives and a mean
     * well under 1; a plain norm weight is a gain trained from 1. Either signal alone is enough. */
    return (mn < 0.0f || (sum / (double)n) < 0.5) ? 1 : 0;
}

int q4nx_semantics_linear_layers(const Q4nxSemantics* s) {
    if (!s) return 0;
    int n = 0;
    for (int i = 0; i < s->n_layer_types; i++)
        if (s->layer_types[i] == Q4NX_LAYER_LINEAR) n++;
    return n;
}

void q4nx_semantics_layer_mix(const Q4nxSemantics* s, char* out, size_t out_sz) {
    if (!out || out_sz == 0) return;
    int full = 0, slid = 0, lin = 0, oth = 0;
    const int n = s ? s->n_layer_types : 0;
    for (int i = 0; i < n; i++) {
        switch (s->layer_types[i]) {
            case Q4NX_LAYER_FULL:    full++; break;
            case Q4NX_LAYER_SLIDING: slid++; break;
            case Q4NX_LAYER_LINEAR:  lin++;  break;
            default:                 oth++;  break;
        }
    }
    if (n == 0) { snprintf(out, out_sz, "no layer_types (all full attention)"); return; }
    snprintf(out, out_sz, "full=%d sliding=%d linear=%d other=%d", full, slid, lin, oth);
}

void q4nx_semantics_dump(const Q4nxSemantics* s) {
    if (!s) return;
    fprintf(stderr, "  [semantics] rmsnorm_offset=%d  <- %s\n", s->rmsnorm_offset,
            q4nx_semantics_field(Q4NX_SEM_RMSNORM_OFFSET));
    fprintf(stderr, "  [semantics] qk_norm=%d  <- %s\n", s->qk_norm,
            q4nx_semantics_field(Q4NX_SEM_QK_NORM));
    fprintf(stderr, "  [semantics] rope_theta=%.6g  <- %s\n", s->rope_theta,
            q4nx_semantics_field(Q4NX_SEM_ROPE_THETA));
    fprintf(stderr, "  [semantics] rotary_dim=%d  <- %s\n", s->rotary_dim,
            q4nx_semantics_field(Q4NX_SEM_ROTARY_DIM));
    fprintf(stderr, "  [semantics] mlp_act=%s  <- %s\n",
            s->mlp_act == Q4NX_ACT_GELU ? "gelu" : "silu",
            q4nx_semantics_field(Q4NX_SEM_MLP_ACT));
    fprintf(stderr, "  [semantics] attention_bias=%d mlp_bias=%d  <- %s / %s\n",
            s->attention_bias, s->mlp_bias,
            q4nx_semantics_field(Q4NX_SEM_ATTENTION_BIAS),
            q4nx_semantics_field(Q4NX_SEM_MLP_BIAS));
    fprintf(stderr, "  [semantics] embed_scale=%.6g  <- %s\n", s->embed_scale,
            q4nx_semantics_field(Q4NX_SEM_EMBED_SCALE));
    fprintf(stderr, "  [semantics] logit_softcap=%.6g  <- %s\n", s->logit_softcap,
            q4nx_semantics_field(Q4NX_SEM_LOGIT_SOFTCAP));
    fprintf(stderr, "  [semantics] sliding_window=%d  <- %s\n", s->sliding_window,
            q4nx_semantics_field(Q4NX_SEM_SLIDING_WINDOW));
    fprintf(stderr, "  [semantics] eps=%.6g  <- %s\n", s->eps,
            q4nx_semantics_field(Q4NX_SEM_EPS));
    fprintf(stderr, "  [semantics] hybrid_layers=%d mrope=%d  <- %s / %s\n",
            s->hybrid_layers, s->mrope,
            q4nx_semantics_field(Q4NX_SEM_HYBRID_LAYERS),
            q4nx_semantics_field(Q4NX_SEM_MROPE));
    fprintf(stderr, "  [semantics] per_layer_rope=%d  <- %s\n", s->per_layer_rope,
            q4nx_semantics_field(Q4NX_SEM_PER_LAYER_ROPE));
    {
        char mix[128];
        q4nx_semantics_layer_mix(s, mix, sizeof(mix));
        fprintf(stderr, "  [semantics] layer mix: %s  <- %s\n", mix,
                q4nx_semantics_field(Q4NX_SEM_LAYER_TYPES));
        fprintf(stderr, "  [semantics] per-type rope: full(theta=%.6g rot=%d) "
                        "sliding(theta=%.6g rot=%d)  rope_type_named=%d\n",
                s->rope_theta, s->rotary_dim, s->rope_theta_sliding,
                s->rotary_dim_sliding, s->rope_type_named);
        fprintf(stderr, "  [semantics] ple=%d (dim=%d vocab=%d) kv_shared_layers=%d "
                        "extra_layer_norms=%d double_wide_mlp=%d\n",
                q4nx_semantics_uses_ple(s), s->hidden_size_per_layer_input,
                s->vocab_size_per_layer_input, s->num_kv_shared_layers,
                s->extra_layer_norms, s->double_wide_mlp);
    }
}

int q4nx_attention_mode(const char* declared, int plan_loaded, int plan_noncausal, char* why, size_t why_len) {
    if (why && why_len) why[0] = '\0';
    const int declared_noncausal = declared && strcmp(declared, "noncausal_full_attention") == 0;
    if (plan_loaded && declared && *declared && declared_noncausal != plan_noncausal) {
        if (why && why_len)
            snprintf(why, why_len,
                     "plan.json says attention.mode %s but decision_config.json declares %s: the plan is "
                     "stale, since tools/npu_plan.py writes it from that declaration. Rerun "
                     "tools/npu_plan.py <model dir> --kernels <elf dir>; for a deliberate measurement arm "
                     "change the pack's own decision_config.json, not the plan",
                     plan_noncausal ? "noncausal_full_attention" : "causal", declared);
        return -1;
    }
    if (declared_noncausal) return 1;
    if (declared && *declared) return 0;    /* the pack declares causal, and any plan agrees */
    return plan_noncausal ? 1 : 0;          /* no declaration: the plan decides */
}
