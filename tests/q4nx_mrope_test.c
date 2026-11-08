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

/* q4nx_mrope_test: is an interleaved-MRoPE declaration SEEN, whichever spelling the config uses?
 * No model, no device, no network: writes a config.json to a temp dir and parses it.
 *
 * Why this is a test rather than a comment: two spellings reach the engine. FastFlowLM/OGA packs and Qwen3.5 nest
 * the rotary block under `rope_parameters`; a Hugging Face `qwen3_vl` config (Qwen3-VL-2B-Instruct, the model engine
 * #1311's lane runs for TQVL) nests it under `rope_scaling`. The reader only looked in `rope_parameters`, so a VL
 * container loaded with `mrope=0` - plain RoPE - and nothing said so. For a text-only prompt that is exactly right
 * (with pos_t = pos_h = pos_w the two are bit-identical, measured), but for a prompt with image tokens the two
 * differ by ~0.29 rad on the rotary block, i.e. a silently wrong attention geometry. The flag is what a refusal,
 * a plan field or a route can act on, so it has to be seen.
 */
#include "q4nx_semantics.h"

#include <fcntl.h>
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

static void write_config(const char* dir, const char* json, char* out, size_t out_sz) {
    snprintf(out, out_sz, "%s/config.json", dir);
    int fd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    FILE* f = fd >= 0 ? fdopen(fd, "w") : NULL;
    if (!f) { printf("FAIL cannot write %s\n", out); failures++; return; }
    fputs(json, f);
    fclose(f);
}

int main(void) {
    char dir[] = "/tmp/q4nx-mrope-XXXXXX";
    if (!mkdtemp(dir)) { printf("FAIL mkdtemp\n"); return 1; }
    char cfg[512];
    Q4nxSemantics s;

    /* 1. The Hugging Face qwen3_vl spelling: rope_scaling, mrope_interleaved, mrope_section, rope_type default. */
    write_config(dir,
                 "{\"num_hidden_layers\": 2, \"hidden_size\": 64, \"rope_theta\": 5000000,"
                 " \"rope_scaling\": {\"mrope_interleaved\": true, \"mrope_section\": [24, 20, 20],"
                 " \"rope_type\": \"default\"}}",
                 cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.mrope == 1, "qwen3_vl rope_scaling.mrope_interleaved -> mrope=1");
    check(s.rope_theta == 5000000.0f, "rope_theta still read from the top level (5e6, not lost)");
    check(s.rope_type_named == 0, "rope_type \"default\" is not flagged as a named variant");

    /* 2. The FastFlowLM/Qwen3.5 spelling the reader already handled: rope_parameters. */
    write_config(dir,
                 "{\"num_hidden_layers\": 2, \"hidden_size\": 64,"
                 " \"rope_parameters\": {\"mrope_interleaved\": true, \"rope_theta\": 1000000}}",
                 cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.mrope == 1, "rope_parameters.mrope_interleaved -> mrope=1 (unchanged)");
    check(s.rope_theta == 1000000.0f, "rope_parameters.rope_theta still read");

    /* 3. A plain decoder must not be flagged: this is the control that keeps the new lookup from firing on
     *    every config that happens to have a rope_scaling block (yarn, linear, dynamic all do). */
    write_config(dir,
                 "{\"num_hidden_layers\": 2, \"hidden_size\": 64, \"rope_theta\": 10000,"
                 " \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 4.0}}",
                 cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.mrope == 0, "a rope_scaling block without mrope_interleaved is NOT mrope");
    check(s.rope_theta == 10000.0f, "theta untouched by the control case");

    /* 4. mrope_interleaved: false is a declaration of plain RoPE, not a missing field. */
    write_config(dir,
                 "{\"num_hidden_layers\": 2, \"hidden_size\": 64,"
                 " \"rope_scaling\": {\"mrope_interleaved\": false}}",
                 cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.mrope == 0, "mrope_interleaved:false stays plain");

    if (failures) { printf("q4nx_mrope: %d failure(s)\n", failures); return 1; }
    printf("q4nx_mrope: ok\n");
    return 0;
}
