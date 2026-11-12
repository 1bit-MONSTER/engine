// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
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
    check(elf_name_for(sh(768, 768), 768, false) == "full_i8_O_K768_N768_b8.elf", "b8 ELF name");
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
        check(hidden_from_square_widths(nullptr, 0) == 0, "no widths means no hidden width, not a crash");
    }

    // 5. large-v3's shapes at h = 1280, so the rule is not tuned to one pack
    check(std::string(role_of_shape(sh(1280, 1280), 1280)) == "O", "lv3: h x h is O");
    check(std::string(role_of_shape(sh(1280, 5120), 1280)) == "GU", "lv3: h -> 4h is GU");
    check(std::string(role_of_shape(sh(5120, 1280), 1280)) == "D", "lv3: 4h -> h is D");

    if (failures != 0) {
        std::printf("audio kernel roles: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("audio kernel roles: all cases ok\n");
    return 0;
}
