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

    if (failures == 0) printf("all llama3 rope checks passed\n");
    return failures == 0 ? 0 : 1;
}
