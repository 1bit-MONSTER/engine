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
// engine#1783: the audio executor ships as two compiled flavours and this file picks one, once, at load.
//
// Why a whole-TU split and not a per-function clone: the flags' effect is spread across every loop in
// audio_exec.cpp (elementwise, softmax rows, transpose gathers, the four-row GEMM). Measured on this box, a
// TU-wide -mavx2 build is 1.252x on the whisper-small encoder while a clone of just the GEMM inner loop was
// 1.010x -- the width helps all of them, and only the TU split captures all of it.
//
// Why runtime and not build-time: the compiler is chosen per repo, the ISA is per box, and the 1bit OS
// appliance must run on whatever it is installed on (engine#2403). The baseline flavour is always linked; the
// AVX2 one is only entered when the CPU reports both features. No `fma` in either flavour: a fused multiply-add
// rounds once where the ordered sum rounds twice, and audio_parallel asserts the ordered sum bit for bit.
#include "audio_exec.h"

#include <cstdio>
#include <cstdlib>

namespace onebit::npu::pack_import {

namespace tu_base {
bool run_audio_graph(const OnnxGraphInfo&, const float*, int64_t, int64_t, const std::string&, AudioRunResult*,
                     std::string*);
bool run_audio_graph(const OnnxGraphInfo&, const float*, int64_t, int64_t, const std::string&, AudioRunResult*,
                     std::string*, AudioGemm*);
bool run_decoder_graph(const OnnxGraphInfo&, const int64_t*, int64_t, const float*, const std::vector<int64_t>&,
                       const std::string&, AudioRunResult*, std::string*);
bool run_decoder_graph(const OnnxGraphInfo&, const int64_t*, int64_t, const float*, const std::vector<int64_t>&,
                       const std::string&, AudioRunResult*, std::string*, DecoderStepCache*);
int prefill_audio_device(const OnnxGraphInfo&, int64_t, AudioGemm*, std::string*);
}  // namespace tu_base

#if defined(__x86_64__) && defined(__GNUC__)
namespace tu_avx2 {
bool run_audio_graph(const OnnxGraphInfo&, const float*, int64_t, int64_t, const std::string&, AudioRunResult*,
                     std::string*);
bool run_audio_graph(const OnnxGraphInfo&, const float*, int64_t, int64_t, const std::string&, AudioRunResult*,
                     std::string*, AudioGemm*);
bool run_decoder_graph(const OnnxGraphInfo&, const int64_t*, int64_t, const float*, const std::vector<int64_t>&,
                       const std::string&, AudioRunResult*, std::string*);
bool run_decoder_graph(const OnnxGraphInfo&, const int64_t*, int64_t, const float*, const std::vector<int64_t>&,
                       const std::string&, AudioRunResult*, std::string*, DecoderStepCache*);
int prefill_audio_device(const OnnxGraphInfo&, int64_t, AudioGemm*, std::string*);
}  // namespace tu_avx2
#endif

// Read once: the answer cannot change inside a process. ONEBIT_AUDIO_AVX2=0 forces the baseline; =1 requires
// the fast flavour and says so when the CPU cannot run it, rather than executing it anyway.
static bool use_avx2_tu() {
#if defined(__x86_64__) && defined(__GNUC__)
    static const bool pick = [] {
        const bool fast = __builtin_cpu_supports("avx2");
        const char* e = std::getenv("ONEBIT_AUDIO_AVX2");
        if (e != nullptr && e[0] == '0' && e[1] == '\0') return false;
        if (e != nullptr && e[0] == '1' && e[1] == '\0' && !fast) {
            std::fprintf(stderr, "[audio] ONEBIT_AUDIO_AVX2=1 but this CPU has no AVX2: keeping the baseline\n");
            return false;
        }
        return fast;
    }();
    return pick;
#else
    return false;
#endif
}

bool run_audio_graph(const OnnxGraphInfo& g, const float* mel, int64_t mel_bins, int64_t frames,
                     const std::string& out_name, AudioRunResult* out, std::string* why) {
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2_tu()) return tu_avx2::run_audio_graph(g, mel, mel_bins, frames, out_name, out, why);
#endif
    return tu_base::run_audio_graph(g, mel, mel_bins, frames, out_name, out, why);
}

bool run_audio_graph(const OnnxGraphInfo& g, const float* mel, int64_t mel_bins, int64_t frames,
                     const std::string& out_name, AudioRunResult* out, std::string* why, AudioGemm* dev) {
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2_tu()) return tu_avx2::run_audio_graph(g, mel, mel_bins, frames, out_name, out, why, dev);
#endif
    return tu_base::run_audio_graph(g, mel, mel_bins, frames, out_name, out, why, dev);
}

bool run_decoder_graph(const OnnxGraphInfo& g, const int64_t* tokens, int64_t n_tokens, const float* xa,
                       const std::vector<int64_t>& xa_dims, const std::string& out_name, AudioRunResult* out,
                       std::string* why) {
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2_tu()) return tu_avx2::run_decoder_graph(g, tokens, n_tokens, xa, xa_dims, out_name, out, why);
#endif
    return tu_base::run_decoder_graph(g, tokens, n_tokens, xa, xa_dims, out_name, out, why);
}

bool run_decoder_graph(const OnnxGraphInfo& g, const int64_t* tokens, int64_t n_tokens, const float* xa,
                       const std::vector<int64_t>& xa_dims, const std::string& out_name, AudioRunResult* out,
                       std::string* why, DecoderStepCache* cache) {
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2_tu()) return tu_avx2::run_decoder_graph(g, tokens, n_tokens, xa, xa_dims, out_name, out, why, cache);
#endif
    return tu_base::run_decoder_graph(g, tokens, n_tokens, xa, xa_dims, out_name, out, why, cache);
}

// engine#1783: either copy builds the same keys and the same weights; the pick follows the executor's, for one rule.
int prefill_audio_device(const OnnxGraphInfo& g, int64_t rows, AudioGemm* dev, std::string* why) {
#if defined(__x86_64__) && defined(__GNUC__)
    if (use_avx2_tu()) return tu_avx2::prefill_audio_device(g, rows, dev, why);
#endif
    return tu_base::prefill_audio_device(g, rows, dev, why);
}

}  // namespace onebit::npu::pack_import
