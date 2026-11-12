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

    /* 5. The interleaved axis rule (engine #1498).  The sections are stride-3 windows, not contiguous runs: at
     *    [24, 20, 20] there must be 24 t, 20 h and 20 w frequencies out of 64, and the h/w windows must sit on
     *    i % 3 == 1 / 2 and stop at 3 * section[dim].  A contiguous split puts 40 of the 64 on the wrong axis. */
    {
        int sec[3] = {24, 20, 20};
        int count[3] = {0, 0, 0};
        for (int i = 0; i < 64; i++) count[q4nx_mrope_axis(i, 3, sec)]++;
        check(count[0] == 24 && count[1] == 20 && count[2] == 20, "interleaved [24,20,20] -> 24 t / 20 h / 20 w");
        check(q4nx_mrope_axis(1, 3, sec) == 1 && q4nx_mrope_axis(4, 3, sec) == 1, "i%3==1 is the h window");
        check(q4nx_mrope_axis(2, 3, sec) == 2 && q4nx_mrope_axis(5, 3, sec) == 2, "i%3==2 is the w window");
        check(q4nx_mrope_axis(0, 3, sec) == 0 && q4nx_mrope_axis(3, 3, sec) == 0, "i%3==0 stays t");
        check(q4nx_mrope_axis(60, 3, sec) == 0, "past 3 * section[1] the h window closes (i=60 is t)");
        check(q4nx_mrope_axis(61, 3, sec) == 0 && q4nx_mrope_axis(62, 3, sec) == 0,
              "past 3 * section[2] the w window closes too");
        /* the contiguous mapping this replaces, for the record: 43 of 64 differ (measured, and the number the
         * reviewer gave for engine #1498) */
        int diff = 0;
        for (int i = 0; i < 64; i++) {
            int old = i < 24 ? 0 : (i < 44 ? 1 : 2);
            if (old != q4nx_mrope_axis(i, 3, sec)) diff++;
        }
        check(diff == 43, "the contiguous reading differs on 43 of 64 frequencies");
        /* and with no sections nothing rotates off the t plane */
        check(q4nx_mrope_axis(1, 0, sec) == 0, "no mrope_section -> plain (t) axis");
    }

    /* 6. Qwen2-VL / Qwen2.5-VL (engine#1786): rope_scaling {"type": "mrope", "mrope_section": [16, 24, 24]}, the
     * CONTIGUOUS form. transformers' apply_multimodal_rotary_pos_emb splits cos by mrope_section * 2 and takes chunk
     * k from plane k % 3, so frequency i is t below 16, h below 40 and w below 64. FastFlowLM's
     * Qwen2.5-VL-3B-Instruct-NPU2 was refused at load ("mrope" was taken for a frequency scaling). */
    write_config(dir,
                 "{\"num_hidden_layers\": 2, \"hidden_size\": 2048, \"rope_theta\": 1000000.0,"
                 " \"rope_scaling\": {\"type\": \"mrope\", \"mrope_section\": [16, 24, 24]}}",
                 cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.mrope == 1 && s.mrope_sectioned == 1, "qwen2.5_vl rope_scaling.type mrope -> mrope=1, sectioned");
    check(s.n_mrope_section == 3 && s.mrope_section[0] == 16 && s.mrope_section[1] == 24 && s.mrope_section[2] == 24,
          "its sections are read: 16 / 24 / 24");
    {
        const int sec[3] = {16, 24, 24};
        /* the reference's own construction, over the 64 frequencies of a 128-wide head */
        int ref[64], k = 0, agree = 0;
        const int split[6] = {16, 24, 24, 16, 24, 24};
        for (int c = 0; c < 6 && k < 128; c++)
            for (int j = 0; j < split[c]; j++, k++)
                if (k < 64) ref[k] = c % 3;
        for (int i = 0; i < 64; i++) agree += q4nx_mrope_axis_sectioned(i, 3, sec) == ref[i];
        check(agree == 64, "contiguous axis = the reference's split of cos by mrope_section * 2, all 64 frequencies");
        check(q4nx_mrope_axis_sectioned(15, 3, sec) == 0 && q4nx_mrope_axis_sectioned(16, 3, sec) == 1 &&
                  q4nx_mrope_axis_sectioned(39, 3, sec) == 1 && q4nx_mrope_axis_sectioned(40, 3, sec) == 2,
              "the run boundaries: 15 t, 16 h, 39 h, 40 w");
        int differ = 0;
        for (int i = 0; i < 64; i++) differ += q4nx_mrope_axis_sectioned(i, 3, sec) != q4nx_mrope_axis(i, 3, sec);
        check(differ > 0, "and it is not the interleaved layout (the two disagree on some frequencies)");
    }
    /* the Qwen3-VL spelling is untouched by the new form */
    write_config(dir,
                 "{\"num_hidden_layers\": 2, \"rope_scaling\": {\"mrope_interleaved\": true,"
                 " \"mrope_section\": [24, 20, 20], \"rope_type\": \"default\"}}",
                 cfg, sizeof cfg);
    q4nx_semantics_from_config(cfg, dir, &s);
    check(s.mrope == 1 && s.mrope_sectioned == 0, "qwen3_vl stays interleaved (mrope_sectioned 0)");

    if (failures) { printf("q4nx_mrope: %d failure(s)\n", failures); return 1; }
    printf("q4nx_mrope: ok\n");
    return 0;
}
