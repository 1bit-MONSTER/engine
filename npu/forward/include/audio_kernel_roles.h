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
// audio_kernel_roles.h -- which GEMM shape an encoder's projection is, decided by SHAPE against the hidden width (engine#1783).
//
// Why by shape and not by name: a packed export reaches its weights through generically named initializers (`val_452`) and the
// human name sits on a tensor the MatMul never touches, so a name-based classifier reports every projection as unknown. Shape is
// also what a kernel actually depends on -- one ELF serves every role with the same (K, N), because q, k, v and o are all h x h.
//
// The four roles are the ones the family generator emits (see tools/audio_kernel_shapes.cpp and the private family's
// `full_i8_<ROLE>_K<K>_N<N>_b8[_m16].elf` naming):
//
//   fused QKV   K = h,  N = 3h        attention's q|k|v packed into one matrix
//   O           K = h,  N = h         q, k, v and o all land here -- one ELF serves all four
//   GU          K = h,  N = 4h        the MLP's fused gate|up (Whisper's 4x expansion)
//   D           K = 4h, N = h         the MLP's down projection
//
// A shape that is none of these is `unknown`, and that is deliberate: an unrecognised shape is a projection nobody has built a
// kernel for, and saying so is more useful than guessing a role and asking for the wrong ELF.

#pragma once

#include <string>

namespace onebit::audio {

struct ProjectionShape {
    int k = 0;   // reduction (K)
    int n = 0;   // output width (N)
};

// The hidden width, from the model's own square projections: q/k/v/o are h x h in every layer, so the most common square width
// IS h. A median over all weights would be skewed by the embedding and the 4h MLP, which is what the first version did.
// The hidden width is the MODE of the square projection widths (h x h is q/k/v/o, and there are four of them per layer).
// A tie is refused rather than broken by whichever width came first: two widths occurring equally often means the pack has
// two candidate hidden widths, and picking one silently would decide every role after it -- the same "confident guess in the
// middle of its own evidence" that a 0.5 threshold on a correlation has. Returns 0 for "no evidence" and for "ambiguous".
inline int hidden_from_square_widths(const int* widths, int count) {
    int best_count = 0, hidden = 0, tied = 0;
    for (int i = 0; i < count; ++i) {
        int occurrences = 0;
        for (int j = 0; j < count; ++j)
            if (widths[j] == widths[i]) ++occurrences;
        if (occurrences > best_count) {
            best_count = occurrences;
            hidden = widths[i];
            tied = 0;
        } else if (occurrences == best_count && widths[i] != hidden) {
            ++tied;
        }
    }
    return tied == 0 ? hidden : 0;
}

inline const char* role_of_shape(const ProjectionShape& s, int hidden) {
    if (hidden <= 0) return "unknown";
    if (s.k == hidden && s.n == 3 * hidden) return "QKV";
    if (s.k == hidden && s.n == hidden) return "O";       // q, k, v and o share this shape and this ELF
    if (s.k == hidden && s.n == 4 * hidden) return "GU";
    if (s.k == 4 * hidden && s.n == hidden) return "D";
    return "unknown";
}

// The ELF the family generator emits for this shape, whether or not that role is real: the caller decides what to do with a
// name for an `unknown` shape (the tool reports it, a gate would refuse it).
// The engine's own staging grid, from npu/forward/src/q4nx_pack.c (`q4nx_pad_k` / `q4nx_pad_n`): a container's K is a
// multiple of 256 and its N a multiple of 512, and the ELF the forward asks for is named at THOSE widths, not at the model's.
// Measured on the synth768 run (engine#1783): QKV 2304 -> the forward asks for N2560, O 768 -> N1024, and a set emitted at the
// unpadded widths is "missing full ELF" while this header said the family was covered.
constexpr int kQ4nxKGrid = 256;
constexpr int kQ4nxNGrid = 512;

inline int q4nx_pad_k(int k) { return k <= 0 ? k : ((k + kQ4nxKGrid - 1) / kQ4nxKGrid) * kQ4nxKGrid; }
inline int q4nx_pad_n(int n) { return n <= 0 ? n : ((n + kQ4nxNGrid - 1) / kQ4nxNGrid) * kQ4nxNGrid; }

// The shape as the forward names it: padded on the engine's grid.
inline ProjectionShape staged(const ProjectionShape& s) { return ProjectionShape{q4nx_pad_k(s.k), q4nx_pad_n(s.n)}; }

inline std::string elf_name_for(const ProjectionShape& s, int hidden, bool m16) {
    const ProjectionShape p = staged(s);
    return std::string("full_i8_") + role_of_shape(s, hidden) + "_K" + std::to_string(p.k) + "_N" + std::to_string(p.n) +
           "_b8" + (m16 ? "_m16" : "") + ".elf";
}

}  // namespace onebit::audio
