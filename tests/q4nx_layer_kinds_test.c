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

/* q4nx_layer_kinds_test: layer_types -> per-layer kind, and the name a refusal can quote.
 * No model, no device, no network: writes a config.json to a temp dir and parses it.
 *
 * Why this is a test rather than a comment: q4nx_semantics maps every layer_types entry it does not recognise to
 * Q4NX_LAYER_OTHER, and the forward's attention path treats OTHER as full attention. A container that declares a
 * layer kind nobody implements would therefore be served as a plain decoder - fluent, wrong at every position, and
 * silent. The forward now refuses such a container by name (the engine#1087 shape), and the name it quotes is what
 * these assertions check - the first unknown spelling, not the last, and not the count.
 *
 * The known kinds are checked at the same time so a future edit to the string matching (which uses strstr for
 * "sliding" and "linear", the spellings every container on this box uses) cannot pass this test by matching nothing.
 */
#include "q4nx_semantics.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

static void check(int ok, const char* what) {
    if (!ok) {
        printf("FAIL %s\n", what);
        failures++;
    } else {
        printf("ok   %s\n", what);
    }
}

/* Write <dir>/config.json holding exactly `json`; returns the path in `out`. */
static void write_config(const char* dir, const char* json, char* out, size_t out_sz) {
    snprintf(out, out_sz, "%s/config.json", dir);
    FILE* f = fopen(out, "w");
    if (!f) { printf("FAIL cannot write %s\n", out); failures++; return; }
    fputs(json, f);
    fclose(f);
}

int main(void) {
    char dir[] = "/tmp/q4nx-layer-kinds-XXXXXX";
    if (!mkdtemp(dir)) { printf("FAIL mkdtemp\n"); return 1; }
    char cfg[512];

    /* 1. The three spellings the containers on this box use, plus one that nothing implements. */
    write_config(dir,
                 "{\"num_hidden_layers\": 4, \"layer_types\": [\"full_attention\", \"linear_attention\","
                 " \"sliding_attention\", \"conv_attention\"], \"hidden_size\": 64}",
                 cfg, sizeof cfg);
    Q4nxSemantics s;
    q4nx_semantics_from_config(cfg, dir, &s);

    check(s.n_layer_types == 4, "four layer_types entries parsed");
    check(s.layer_types[0] == Q4NX_LAYER_FULL, "full_attention -> Q4NX_LAYER_FULL");
    check(s.layer_types[1] == Q4NX_LAYER_LINEAR, "linear_attention -> Q4NX_LAYER_LINEAR (the GDN path)");
    check(s.layer_types[2] == Q4NX_LAYER_SLIDING, "sliding_attention -> Q4NX_LAYER_SLIDING");
    check(s.layer_types[3] == Q4NX_LAYER_OTHER, "conv_attention -> Q4NX_LAYER_OTHER (nothing implements it)");
    check(strcmp(s.unknown_layer_type, "conv_attention") == 0,
          "the refusal can quote the unknown kind by name");
    check(s.hybrid_layers == 3, "hybrid count covers linear + sliding + other, as before");

    /* 2. Only known kinds: nothing to refuse, so the name must stay empty (this is every container on this box). */
    write_config(dir,
                 "{\"num_hidden_layers\": 2, \"layer_types\": [\"full_attention\", \"linear_attention\"]}",
                 cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.unknown_layer_type[0] == '\0', "known kinds only -> no refusal name (existing containers unaffected)");
    check(s.n_layer_types == 2, "two entries parsed in the known-only case");

    /* 3. No layer_types at all (qwen3/qwen2/qwen3moe containers): also nothing to refuse. */
    write_config(dir, "{\"num_hidden_layers\": 2}", cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.unknown_layer_type[0] == '\0', "no layer_types -> no refusal name");
    check(s.n_layer_types == 0, "no layer_types -> zero kinds (forward uses full attention)");

    /* 4. Only the FIRST unknown spelling is kept: the message must not depend on the layer order. */
    write_config(dir,
                 "{\"num_hidden_layers\": 3, \"layer_types\": [\"mamba2\", \"full_attention\", \"conv_attention\"]}",
                 cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(strcmp(s.unknown_layer_type, "mamba2") == 0, "the first unknown kind is the one quoted");

    unlink(cfg);
    rmdir(dir);
    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
