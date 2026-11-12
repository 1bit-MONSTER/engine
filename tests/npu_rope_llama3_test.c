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

/* npu_rope_llama3_test: rope_type "llama3" frequencies are fixed data (npu/forward/include/q4nx_rope.h).
 * No model, no device, no network.
 *
 * The expected table below was produced by transformers' own implementation and then reproduced by an
 * independent port to max|diff| 4.2e-8 (ROPE_GAP_REPORT / engine#1048), so it is a known answer, not a
 * snapshot of what this code happens to do.  The assertion that fails on the old code is the first one:
 * on the old code every entry came back as the plain theta^(-i/half), and 17 of these 32 entries are not.
 *
 * The two configurations are the two published llama3 shapes: Llama-3.2-1B (head_dim 64, factor 32, so
 * 15 unchanged / 14 divided by exactly 32 / 3 interpolated at i=15..17) and Llama-3.1-8B (head_dim 128,
 * factor 8, so 29 / 29 / 6 at i=29..34).  A rule that only divided the low frequencies, or switched at
 * the band edge instead of interpolating, fails on the interpolated entries.
 */
#include "q4nx_rope.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char* what) {
    if (!ok) {
        printf("FAIL %s\n", what);
        ++failures;
    }
}

static double plain_inv(int i, int half, double theta) {
    return 1.0 / pow(theta, (double)i / (double)half);
}

static int near(double a, double b) {
    const double d = fabs(a - b);
    const double m = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return d <= 1e-12 * (m > 1.0 ? m : 1.0);
}

/* llama3_inv_freq(int half, theta, factor, high, low, orig_max) -> the scaled table the model must use */
static int test_table(const char* tag, int half, double theta, double factor, double high, double low,
                      double orig, const double* want, int want_n, int n_unchanged, int n_divided,
                      int first_interp, int n_interp) {
    Q4nxRopeLlama3 s;
    s.on = 1;
    s.factor = factor;
    s.high_freq_factor = high;
    s.low_freq_factor = low;
    s.original_max_position = orig;

    int unchanged = 0, divided = 0, interp = 0, differed = 0;
    for (int i = 0; i < want_n; i++) {
        const double plain = plain_inv(i, half, theta);
        const double got = q4nx_rope_llama3_inv_freq(plain, &s);
        if (!near(got, want[i])) {
            printf("FAIL %s: i=%d got %.17g want %.17g\n", tag, i, got, want[i]);
            ++failures;
        }
        if (near(plain, want[i])) ++unchanged;
        else if (near(plain / factor, want[i])) ++divided;
        else ++interp;
        if (fabs(got - plain) > 1e-12) ++differed;
    }
    char buf[192];
    snprintf(buf, sizeof buf, "%s: unchanged %d == %d", tag, unchanged, n_unchanged);
    check(unchanged == n_unchanged, buf);
    snprintf(buf, sizeof buf, "%s: divided by factor %d == %d", tag, divided, n_divided);
    check(divided == n_divided, buf);
    snprintf(buf, sizeof buf, "%s: interpolated %d == %d", tag, interp, n_interp);
    check(interp == n_interp, buf);
    snprintf(buf, sizeof buf, "%s: %d entries differ from plain (old code gave 0)", tag, differed);
    check(differed == n_divided + n_interp, buf);
    /* the band must be contiguous and start where the table says it starts */
    snprintf(buf, sizeof buf, "%s: first interpolated index %d", tag, first_interp);
    {
        int seen = -1;
        for (int i = 0; i < want_n && seen < 0; i++) {
            const double plain = plain_inv(i, half, theta);
            if (!near(plain, want[i]) && !near(plain / factor, want[i])) seen = i;
        }
        check(seen == first_interp, buf);
    }
    return 0;
}

/* The declared scaling type, and whether it is one this forward applies (engine#1087).  A type it does
 * not implement must be refused by name: served plain it stays fluent and is wrong past the original
 * window.  The reader is exercised on the config shapes that exist in the wild, including the two
 * spellings of the key and a config whose "model_type" must not be mistaken for a scaling type. */
static void test_declared_type(void) {
    struct {
        const char* json;
        const char* type;
        int llama3_on;
        int has_short;
        int supported;
    } cases[] = {
        /* no rope_scaling at all: plain rope, and "model_type" is not a scaling type */
        {"{\"model_type\": \"qwen3\", \"rope_theta\": 1000000.0}", "", 0, 0, 1},
        /* the pack-imported Llama-3.2 config, verbatim shape */
        {"{\"rope_theta\": 500000.0, \"rope_scaling\": {\"factor\": 32.0, \"high_freq_factor\": 4.0, "
         "\"low_freq_factor\": 1.0, \"original_max_position_embeddings\": 8192, \"rope_type\": \"llama3\"}}",
         "llama3", 1, 0, 1},
        /* declared llama3 whose numbers are unusable: not the same thing as no block */
        {"{\"rope_scaling\": {\"factor\": 0.0, \"high_freq_factor\": 4.0, \"low_freq_factor\": 1.0, "
         "\"original_max_position_embeddings\": 8192, \"rope_type\": \"llama3\"}}",
         "llama3", 0, 0, 0},
        /* YaRN, both spellings: refused */
        {"{\"rope_scaling\": {\"factor\": 4.0, \"original_max_position_embeddings\": 32768, "
         "\"rope_type\": \"yarn\"}}",
         "yarn", 0, 0, 0},
        {"{\"rope_scaling\": {\"type\": \"yarn\", \"factor\": 4.0}}", "yarn", 0, 0, 0},
        /* dynamic NTK and linear: refused */
        {"{\"rope_scaling\": {\"type\": \"dynamic\", \"factor\": 2.0}}", "dynamic", 0, 0, 0},
        {"{\"rope_scaling\": {\"type\": \"linear\", \"factor\": 8.0}}", "linear", 0, 0, 0},
        /* explicit no-scaling spellings: supported */
        {"{\"rope_scaling\": null}", "", 0, 0, 1},
        /* null followed by another block that does carry a "type": still nothing declared */
        {"{\"rope_scaling\": null, \"other\": {\"type\": \"yarn\"}}", "", 0, 0, 1},
        {"{\"rope_scaling\": {\"type\": \"none\"}}", "none", 0, 0, 1},
        {"{\"rope_scaling\": {\"rope_type\": \"default\"}}", "default", 0, 0, 1},
        /* LongRoPE: only with the short_factor the forward reads */
        {"{\"rope_scaling\": {\"rope_type\": \"longrope\", \"short_factor\": [1.0, 1.1]}}", "longrope", 0,
         1, 1},
        {"{\"rope_scaling\": {\"rope_type\": \"longrope\"}}", "longrope", 0, 0, 0},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        char type[64];
        q4nx_rope_declared_type(cases[i].json, type, (int)sizeof type);
        Q4nxRopeLlama3 s;
        s.on = cases[i].llama3_on;
        s.factor = 32.0;
        s.high_freq_factor = 4.0;
        s.low_freq_factor = 1.0;
        s.original_max_position = 8192.0;
        const int got = q4nx_rope_type_supported(type, &s, cases[i].has_short, 0, 0);
        char buf[192];
        snprintf(buf, sizeof buf, "declared type \"%s\" reads as \"%s\"", cases[i].type, type);
        check(!strcmp(type, cases[i].type), buf);
        snprintf(buf, sizeof buf, "declared \"%s\" supported %d == %d", cases[i].type, got,
                 cases[i].supported);
        check(got == cases[i].supported, buf);
    }
    /* The review's two edges (PR #1090): the window is the block's own closing brace, not a byte count.
     * (a) a type that sits past a fixed window must still be found — reading it as "no block" would
     *     serve the model plain, which is the failure this reader exists to stop (it fails OPEN);
     * (b) a block with no type must not take a neighbour's "type" — that refuses a model that asked
     *     for nothing. */
    {
        char big[1600];
        char pad[1000];
        memset(pad, 'x', sizeof pad - 1);
        pad[sizeof pad - 1] = '\0';
        snprintf(big, sizeof big, "{\"rope_scaling\": {\"pad\": \"%s\", \"rope_type\": \"yarn\"}}", pad);
        char type[64];
        q4nx_rope_declared_type(big, type, (int)sizeof type);
        check(!strcmp(type, "yarn"), "a type 1,000 bytes into the block is still read (fails closed)");
        check(q4nx_rope_type_supported(type, (const Q4nxRopeLlama3*)0, 0, 0, 0) == 0,
              "... and is refused");
    }
    {
        char type[64];
        q4nx_rope_declared_type("{\"rope_scaling\": {\"factor\": 8.0}, "
                                "\"quantization_config\": {\"type\": \"fp8\"}}", type, (int)sizeof type);
        check(type[0] == '\0', "a block with no type does not take a neighbour's type");
        check(q4nx_rope_type_supported(type, (const Q4nxRopeLlama3*)0, 0, 0, 0) == 1, "... and is not refused");
        /* nested arrays/objects before the type, and a '}' inside a string, both keep the block open */
        q4nx_rope_declared_type("{\"rope_scaling\": {\"mrope_section\": [1, 2], \"rope_type\": \"yarn\"}}",
                                type, (int)sizeof type);
        check(!strcmp(type, "yarn"), "a nested array before the type does not end the block");
        q4nx_rope_declared_type("{\"rope_scaling\": {\"note\": \"} not the end\", \"rope_type\": \"yarn\"}}",
                                type, (int)sizeof type);
        check(!strcmp(type, "yarn"), "a '}' inside a string does not end the block");
        q4nx_rope_declared_type("{\"rope_scaling\": {\"rope_type\": \"yarn\"", type, (int)sizeof type);
        check(type[0] == '\0', "an unterminated block declares nothing rather than guessing");
    }

    /* no config text at all: nothing declared, nothing to refuse */
    char none[8];
    q4nx_rope_declared_type((const char*)0, none, (int)sizeof none);
    check(none[0] == '\0', "null config declares no type");
    check(q4nx_rope_type_supported(none, (const Q4nxRopeLlama3*)0, 0, 0, 0) == 1, "no type is supported");
    /* engine#1740 W2: the declared-type extraction must SEE Hunyuan's block, because everything downstream (the
     * reader, the predicate, the base rule) is gated on this string. A config that declares dynamic must report it,
     * and one that declares nothing must report nothing -- the second half is what keeps every other model on plain
     * rope. */
    {
        const char* hy = "{\"rope_scaling\": {\"type\": \"dynamic\", \"factor\": 1.0}, \"max_position_embeddings\": 262144}";
        char t2[64] = {0};
        q4nx_rope_declared_type(hy, t2, (int)sizeof t2);
        check(!strcmp(t2, "dynamic"), "a config declaring rope_scaling dynamic is read as dynamic");
        const char* none = "{\"max_position_embeddings\": 32768}";
        char t3[64] = {1};
        q4nx_rope_declared_type(none, t3, (int)sizeof t3);
        check(t3[0] == 0, "and a config with no rope_scaling block reports nothing");
    }

    /* engine#1740 W2: dynamic NTK. The support predicate accepts it ONLY when the numbers were read, and the base
     * rule is the plain theta until a position passes the model's own window -- which is the property that makes
     * accepting Hunyuan's factor-1.0 block safe rather than a guess. */
    check(q4nx_rope_type_supported("dynamic", (const Q4nxRopeLlama3*)0, 0, 0, 0) == 0,
          "dynamic NTK is refused when no numbers were read (a declared block must not be served as plain rope)");
    check(q4nx_rope_type_supported("dynamic", (const Q4nxRopeLlama3*)0, 0, 1, 0) == 1,
          "... and accepted when they were");
    check(q4nx_rope_type_supported("yarn", (const Q4nxRopeLlama3*)0, 0, 1, 0) == 0,
          "... which does not open the door to the other types");
    {
        Q4nxRopeDynamic d = {1, 1.0, 262144.0};            /* Hunyuan-0.5B's own numbers */
        check(q4nx_rope_dynamic_base(1e6, &d, 0.0, 128) == 1e6, "below the window the base IS theta (position 0)");
        check(q4nx_rope_dynamic_base(1e6, &d, 262143.0, 128) == 1e6, "... and still is one position short of it");
        check(q4nx_rope_dynamic_base(1e6, &d, 262144.0, 128) >= 1e6, "... and does not fall below theta once past it");
        /* factor 1 at exactly the window is the no-op case: scaled == 1, so base == theta */
        check(q4nx_rope_dynamic_base(1e6, &d, 262144.0, 128) == 1e6, "factor 1 at the window is a no-op, bit for bit");
        Q4nxRopeDynamic f4 = {1, 4.0, 8192.0};
        const double at_2x = q4nx_rope_dynamic_base(1e4, &f4, 16384.0, 128);
        /* factor 4 at 2x the window: scaled = 4*2 - 3 = 5, base = theta * 5^(dim/(dim-2)) */
        check(fabs(at_2x - 1e4 * pow(5.0, 128.0 / 126.0)) < 1e-6, "the dynamic base follows the NTK formula exactly");
        Q4nxRopeDynamic bad = {1, 0.0, 8192.0};
        check(q4nx_rope_dynamic_base(1e4, &bad, 1e6, 128) == 1e4, "an unusable factor leaves the base alone");
        Q4nxRopeDynamic bad2 = {1, 4.0, 0.0};
        check(q4nx_rope_dynamic_base(1e4, &bad2, 1e6, 128) == 1e4, "an unusable window leaves the base alone");
        Q4nxRopeDynamic off = {0, 4.0, 8192.0};
        check(q4nx_rope_dynamic_base(1e4, &off, 1e6, 128) == 1e4, "and off is a no-op");
        /* THE EXPONENT, against a KNOWN VALUE (review of #1792). Every arm above asserts a property -- "below the window
         * the base is theta, past it not below theta" -- and every one of them is satisfied by ANY exponent, which is how
         * a wrong rotary width (the caller passing head_dim instead of the rotary width) survived a green suite. With
         * factor 1.0 the rule collapses to theta * (pos/original)^(dim/(dim-2)), so one printed constant pins it:
         *   pos 524288, original 262144 -> ratio 2; dim 64 -> 1e6 * 2^(64/62) = 2.0435e6
         * and dim 128 -> 1e6 * 2^(128/126) = 2.0219e6, i.e. the arm distinguishes the two widths the bug confused. */
        {
            Q4nxRopeDynamic k = {1, 1.0, 262144.0};
            const double want64 = 1e6 * pow(2.0, 64.0 / 62.0);
            const double want128 = 1e6 * pow(2.0, 128.0 / 126.0);
            const double got64 = q4nx_rope_dynamic_base(1e6, &k, 524288.0, 64);
            const double got128 = q4nx_rope_dynamic_base(1e6, &k, 524288.0, 128);
            check(fabs(got64 - want64) < 1e-6 * want64,
                  "the base at twice the window with dim 64 is theta * 2^(64/62), to 1e-6 relative");
            check(fabs(got128 - want128) < 1e-6 * want128,
                  "... and with dim 128 it is theta * 2^(128/62)");
            /* the pair is the point: if the exponent were ignored, or the width were not passed through, these would be
             * equal -- so this asserts the arms CAN tell the rotary width from the head dim. */
            check(fabs(got64 - got128) > 1e3,
                  "and the two widths differ by more than 1e3, so this arm cannot pass with the widths confused");
        }
        /* engine#1889: Hunyuan's block is `{"alpha": 1000.0, "factor": 1.0, "type": "dynamic"}`, and with alpha set its
         * rotary embedding uses ONE fixed base, theta * alpha^(dim/(dim-2)), at every position. Read as factor-only the
         * block was a no-op (base 10000), which rotated every position by the wrong angle. The expected numbers are the
         * pack's, not the formula's: Hunyuan-7B's model.onnx.data carries the graph's cos_cache/sin_cache (FP16
         * [32768, 64]); at position 1 the angle of pair i is its inverse frequency, and the table reads
         *   pair 1: 0.77608   pair 2: 0.60209   pair 5: 0.28159
         * (FP16, so good to about 5e-4), against 0.86596 / 0.74989 / 0.48697 for plain theta 10000. */
        {
            Q4nxRopeDynamic hy = {1, 1.0, 262144.0, 1000.0};   /* Hunyuan-7B's own numbers */
            const double base = q4nx_rope_dynamic_base(1e4, &hy, 1.0, 128);
            check(fabs(base - 11158839.9) < 1.0, "alpha 1000 at theta 10000, dim 128: the base is 11,158,840");
            check(q4nx_rope_dynamic_base(1e4, &hy, 0.0, 128) == base && q4nx_rope_dynamic_base(1e4, &hy, 300000.0, 128) == base,
                  "... at every position, inside the window and past it (the alpha form does not move with pos)");
            check(fabs(pow(base, -1.0 / 64.0) - 0.77608) < 5e-4, "pair 1's inverse frequency is the pack table's 0.77608");
            check(fabs(pow(base, -2.0 / 64.0) - 0.60209) < 5e-4, "pair 2's is the pack table's 0.60209");
            check(fabs(pow(base, -5.0 / 64.0) - 0.28159) < 5e-4, "pair 5's is the pack table's 0.28159");
            check(fabs(pow(1e4, -1.0 / 64.0) - 0.77608) > 0.05,
                  "and plain theta 10000 misses pair 1 by more than 0.05, so these arms cannot pass on the no-op");
            Q4nxRopeDynamic hy64 = {1, 1.0, 262144.0, 1000.0};
            check(fabs(q4nx_rope_dynamic_base(1e4, &hy64, 1.0, 64) - 1e4 * pow(1000.0, 64.0 / 62.0)) < 1e-3,
                  "the exponent is dim/(dim-2) over the rotary width here too (dim 64 -> 64/62)");
            Q4nxRopeDynamic no_alpha = {1, 1.0, 262144.0, 0.0};
            check(q4nx_rope_dynamic_base(1e4, &no_alpha, 5.0, 128) == 1e4,
                  "alpha 0 is the factor rule: inside the window the base is theta, as before");
            Q4nxRopeDynamic off_alpha = {0, 1.0, 262144.0, 1000.0};
            check(q4nx_rope_dynamic_base(1e4, &off_alpha, 5.0, 128) == 1e4, "and an alpha on a block that is off does nothing");
            Q4nxRopeDynamic tiny = {1, 1.0, 262144.0, 1000.0};
            check(q4nx_rope_dynamic_base(1e4, &tiny, 5.0, 2) == 1e4, "a rotary width of 2 has no exponent, so the base stays theta");
        }
    }

}

int main(void) {
    /* Llama-3.2-1B: head_dim 64, theta 500000, factor 32, high 4, low 1, original_max 8192 */
    static const double t1b[32] = {
        1, 0.66360123769608848, 0.44036660267178046, 0.29222782257301516, 0.19392274474868576,
        0.12868737343265052, 0.085397100285765609, 0.056669621445291044, 0.037606030930863933,
        0.024955408670558694, 0.016560440080994446, 0.010989528534539826, 0.0072926647372171085,
        0.0048394213457198928, 0.0032114459947525913, 0.001290547928209264, 0.00042955679655936815,
        9.7082878026276702e-05, 1.9461638184831125e-05, 1.2914767187047391e-05,
        8.5702554898814781e-06, 5.6872321504570466e-06, 3.7740542941082826e-06,
        2.5044671007024935e-06, 1.6619674677953091e-06, 1.102883668639601e-06,
        7.3187496754404192e-07, 4.8567313430101075e-07, 3.2229329303788936e-07,
        2.1387422816109151e-07, 1.4192720251899592e-07, 9.4183067254349113e-08};
    test_table("Llama-3.2-1B", 32, 500000.0, 32.0, 4.0, 1.0, 8192.0, t1b, 32, 15, 14, 15, 3);

    /* Llama-3.1-8B: head_dim 128, theta 500000, factor 8, same band */
    static const double t8b[64] = {
        1, 0.81461723385654472, 0.66360123769608848, 0.54058100463576697, 0.44036660267178046,
        0.35873022375128993, 0.29222782257301516, 0.2380538204803507, 0.19392274474868576,
        0.15797280990904317, 0.12868737343265052, 0.10483095217796996, 0.085397100285765609,
        0.069565949614160322, 0.056669621445291044, 0.04616405026546052, 0.037606030930863933,
        0.030634520893224042, 0.024955408670558694, 0.020329105980970156, 0.016560440080994446,
        0.013490419890226748, 0.010989528534539826, 0.0089522593361943993, 0.0072926647372171085,
        0.0059407303756749668, 0.0048394213457198928, 0.0039422760301166563, 0.0032114459947525913,
        0.0021665707635033591, 0.0013718935677611381, 0.0008567514129196324, 0.00052484616099295468,
        0.00031269375038406517, 0.00017850781276799638, 9.5562123539646833e-05,
        7.7846552739324498e-05, 6.3415143457776151e-05, 5.1659068748189563e-05,
        4.2082367687255257e-05, 3.4281021959525912e-05, 2.7925911282444464e-05,
        2.2748928601828186e-05, 1.8531669290821308e-05, 1.509621717643313e-05,
        1.2297638677963614e-05, 1.0017868402809974e-05, 8.1607282474359434e-06,
        6.6478698711812362e-06, 5.4154693654999229e-06, 4.4115346745584042e-06,
        3.593712173650999e-06, 2.9274998701761677e-06, 2.3847918463583037e-06,
        1.942692537204043e-06, 1.5825508208909101e-06, 1.2891731721515574e-06,
        1.0501826834601687e-06, 8.5549691264436605e-07, 6.9690252855116753e-07,
        5.6770881007598369e-07, 4.6246538050008837e-07, 3.7673226901739645e-07,
        3.0689259889145111e-07};
    test_table("Llama-3.1-8B", 64, 500000.0, 8.0, 4.0, 1.0, 8192.0, t8b, 64, 29, 29, 29, 6);

    /* off (every non-Llama-3 model): the plain table, untouched */
    {
        Q4nxRopeLlama3 s;
        s.on = 0;
        s.factor = 32.0;
        s.high_freq_factor = 4.0;
        s.low_freq_factor = 1.0;
        s.original_max_position = 8192.0;
        int bad = 0;
        for (int i = 0; i < 32; i++) {
            const double p = plain_inv(i, 32, 500000.0);
            if (q4nx_rope_llama3_inv_freq(p, &s) != p) ++bad;
        }
        check(bad == 0, "off: every frequency is the plain one");
        check(q4nx_rope_llama3_inv_freq(0.5, (const Q4nxRopeLlama3*)0) == 0.5, "null block is a no-op");
    }
    /* a malformed block is refused rather than applied: no reference could reproduce a guess */
    {
        Q4nxRopeLlama3 s;
        s.on = 1;
        s.factor = 0.0;
        s.high_freq_factor = 4.0;
        s.low_freq_factor = 1.0;
        s.original_max_position = 8192.0;
        check(q4nx_rope_llama3_inv_freq(1e-5, &s) == 1e-5, "factor 0 is refused");
        s.factor = 32.0;
        s.high_freq_factor = 1.0;
        check(q4nx_rope_llama3_inv_freq(1e-5, &s) == 1e-5, "an empty band is refused");
        s.high_freq_factor = 4.0;
        s.original_max_position = 0.0;
        check(q4nx_rope_llama3_inv_freq(1e-5, &s) == 1e-5, "original_max 0 is refused");
    }

    test_declared_type();

    /* engine#1786: yarn (GPT-OSS-20B: factor 32, original 4096, beta 32 / 1, truncate false, base 150000, dim 64).
     * Accepted only with usable numbers; the per-pair rule against transformers' _compute_yarn_parameters, computed
     * by hand for pairs 0 (pure extrapolation), 31 (pure interpolation) and one inside the ramp. */
    check(q4nx_rope_type_supported("yarn", (const Q4nxRopeLlama3*)0, 0, 0, 1) == 1, "yarn with its numbers is applied");
    check(q4nx_rope_type_supported("yarn", (const Q4nxRopeLlama3*)0, 0, 0, 0) == 0, "yarn without them is refused");
    {
        Q4nxRopeYarn y = {1, 32.0, 4096.0, 32.0, 1.0, 0.1 * log(32.0) + 1.0, 0};
        const double base = 150000.0;
        const double low = (64 * log(4096.0 / (32.0 * 2 * 3.14159265358979323846))) / (2 * log(base));
        const double high = (64 * log(4096.0 / (1.0 * 2 * 3.14159265358979323846))) / (2 * log(base));
        const double e0 = 1.0, e31 = 1.0 / pow(base, 62.0 / 64.0);
        check(fabs(q4nx_rope_yarn_inv_freq(0, 64, base, &y) - e0) < 1e-15, "yarn pair 0 (below the ramp) keeps the plain frequency");
        check(fabs(q4nx_rope_yarn_inv_freq(31, 64, base, &y) - e31 / 32.0) < 1e-18, "yarn pair 31 (past the ramp) is the plain one / factor");
        const int i = 12;
        const double ex = 1.0 / pow(base, 2.0 * i / 64.0), r = (i - low) / (high - low), ef = 1.0 - (r < 0 ? 0 : r > 1 ? 1 : r);
        check(fabs(q4nx_rope_yarn_inv_freq(i, 64, base, &y) - ((ex / 32.0) * (1 - ef) + ex * ef)) < 1e-15,
              "yarn pair 12 blends the two by the linear ramp (no floor/ceil with truncate false)");
        check(low > 0 && low < 12 && high > 12 && high < 31, "... and pair 12 is inside GPT-OSS's ramp");
    }

    if (failures == 0) printf("all llama3 and yarn rope checks passed\n");
    return failures == 0 ? 0 : 1;
}
