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
//
// tests/audio_kernel_roles_test.cpp -- the shape-to-role rule of tools/audio_kernel_shapes.cpp, as a pure function
// (npu/forward/include/audio_kernel_roles.h, engine#1783). Hand-computed cases; no graph, no device.
//
// The rule was wrong twice before it was right, both times in ways a compiler cannot see: a name-based classifier reported
// `unknown` for every projection (packed exports name their weights `val_N`), and a version that classified BEFORE assigning
// the shape read (0, 0, hidden) and said `unknown` for everything. These arms are the ones that would have caught both.

#include <cstdio>
#include <string>

#include "audio_kernel_roles.h"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

onebit::audio::ProjectionShape sh(int k, int n) { return onebit::audio::ProjectionShape{k, n}; }

}  // namespace

using onebit::audio::elf_name_for;
using onebit::audio::q4nx_pad_k;
using onebit::audio::q4nx_pad_n;

int main() {
    using onebit::audio::elf_name_for;
    using onebit::audio::hidden_from_square_widths;
    using onebit::audio::role_of_shape;

    // 1. whisper-small's four shapes at h = 768
    check(std::string(role_of_shape(sh(768, 768), 768)) == "O", "h x h is O (q, k, v and o share it)");
    check(std::string(role_of_shape(sh(768, 3072), 768)) == "GU", "h -> 4h is GU");
    check(std::string(role_of_shape(sh(3072, 768), 768)) == "D", "4h -> h is D");
    check(std::string(role_of_shape(sh(768, 2304), 768)) == "QKV", "h -> 3h is fused QKV");

    // 2. the ELF the family emits, both M tiers
    // The emitted name is on the engine's staging grid (768 -> 1024), not the model's own width -- the defect the synth768
    // device run exposed: the forward asked for O_K768_N1024 while this said N768 and the set "covered".
    check(elf_name_for(sh(768, 768), 768, false) == "full_i8_O_K768_N1024_b8.elf", "b8 ELF name, staged");
    check(elf_name_for(sh(768, 3072), 768, true) == "full_i8_GU_K768_N3072_b8_m16.elf", "m16 ELF name");

    // 3. a shape that is none of the four is `unknown`, never a guess -- asking for the wrong ELF wastes a ticket
    check(std::string(role_of_shape(sh(1000, 2000), 768)) == "unknown", "an unrecognised shape is unknown");
    check(std::string(role_of_shape(sh(768, 768), 0)) == "unknown", "no hidden width means no roles at all");

    // 4. the hidden width comes from the most common SQUARE width, not the median of everything: embeddings and the 4h MLP
    //    would drag a median away from h, which is the bug the first version had.
    {
        const int widths[] = {768, 768, 768, 768, 3072, 3072, 51865, 1500, 128};
        check(hidden_from_square_widths(widths, 9) == 768, "the mode of the square widths is the hidden width");
        const int one[] = {1280};
        check(hidden_from_square_widths(one, 1) == 1280, "a single width is that width");
        // A tie is refused, not broken: two widths occurring equally often (here 768 and 896, four times each) mean the pack
        // has two candidate hidden widths, and whichever one a loop happened to see first used to decide every role after it.
        {
            const int tied_widths[] = {768, 768, 768, 768, 896, 896, 896, 896};
            check(hidden_from_square_widths(tied_widths, 8) == 0, "an equal-mode tie is refused, not broken by order");
            const int clear[] = {768, 768, 768, 896, 896};
            check(hidden_from_square_widths(clear, 5) == 768, "a clear mode is still taken");
            check(hidden_from_square_widths(nullptr, 0) == 0, "no evidence is 0, not a guess");
        }
        check(hidden_from_square_widths(nullptr, 0) == 0, "no widths means no hidden width, not a crash");
    }

    // 5. large-v3's shapes at h = 1280, so the rule is not tuned to one pack
    check(std::string(role_of_shape(sh(1280, 1280), 1280)) == "O", "lv3: h x h is O");
    check(std::string(role_of_shape(sh(1280, 5120), 1280)) == "GU", "lv3: h -> 4h is GU");
    check(std::string(role_of_shape(sh(5120, 1280), 1280)) == "D", "lv3: 4h -> h is D");

    // The engine stages every container on its own grid -- K a multiple of 256, N of 512 (npu/forward/src/q4nx_pack.c) -- and
    // the forward asks for the ELF at THOSE widths. Measured on the synth768 device run (engine#1783): with the set emitted at
    // the model's own widths the forward said "missing full ELF ... QKV_K768_N2560_b8" (2304 rounded up) and then "...O_K768_N1024"
    // (768 rounded up), while this header was reporting the family covered.
    check(q4nx_pad_n(768) == 1024, "N 768 stages to 1024");
    check(q4nx_pad_n(2304) == 2560, "N 2304 (a fused QKV at hidden 768) stages to 2560");
    check(q4nx_pad_n(3072) == 3072, "N 3072 is already on the 512 grid");
    check(q4nx_pad_k(768) == 768 && q4nx_pad_k(3000) == 3072, "K pads to 256: 768 in place, 3000 -> 3072");
    check(q4nx_pad_n(0) == 0 && q4nx_pad_k(-5) == -5, "a non-positive width is left alone, not rounded into a shape");
    check(elf_name_for({768, 2304}, 768, false) == "full_i8_QKV_K768_N2560_b8.elf",
          "the QKV name the forward asks for at hidden 768 (2304 staged to 2560)");
    check(elf_name_for({768, 768}, 768, true) == "full_i8_O_K768_N1024_b8_m16.elf",
          "the O name, m16 tier: the hidden width stages too (768 to 1024)");
    check(elf_name_for({3072, 768}, 768, false) == "full_i8_D_K3072_N1024_b8.elf",
          "the D name: K already on its grid, N staged");
    check(elf_name_for({768, 3072}, 768, false) == "full_i8_GU_K768_N3072_b8.elf",
          "the MLP name: 4h is on the 512 grid, so whisper-small's own width is the name");

    if (failures != 0) {
        std::printf("audio kernel roles: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("audio kernel roles: all cases ok\n");
    return 0;
}
