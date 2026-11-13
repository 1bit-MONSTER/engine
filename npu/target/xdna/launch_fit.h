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
// npu/target/xdna/launch_fit.h -- does a launch fit the context it was routed to? Header-only so a test with no device
// arms it (xdna_target.cpp includes the private runtime header, so no CI job compiles it).
//
// engine#1783: a standalone caller passed role 0 for every shape (O K768 N1024, GU K768 N6144, D K3072 N1024), so
// GU and D ran on role 0's context. The int8 batched path packs K x N weights into a BO sized KD x ND and reads M x K
// activations into one sized MD x KD; nothing compared the two, so the process died with SIGSEGV after both ELFs had
// initialised and nothing said which launch was wrong. Only the pb16i branch checked M. This is the refusal for all of
// them, with the numbers in the message.
//
// A geometry field of 0 means the context's shape is not known yet, and is not checked: contexts are value-initialised
// and only init_elf() assigns MD (npu_engine_i8ctx_inc.h).
#ifndef ONEBIT_NPU_TARGET_XDNA_LAUNCH_FIT_H
#define ONEBIT_NPU_TARGET_XDNA_LAUNCH_FIT_H

#include <string>

// Empty when an M x K by K x N launch fits a context built for MD x KD by KD x ND; otherwise why it does not.
inline std::string xdna_launch_misfit(int M, int K, int N, int MD, int KD, int ND) {
    if (M < 1 || K < 1 || N < 1)
        return "launch shape M=" + std::to_string(M) + " K=" + std::to_string(K) + " N=" + std::to_string(N) +
               " has an empty dimension";
    std::string why;
    auto over = [&](const char* name, int got, int cap) {
        if (cap > 0 && got > cap) {
            if (!why.empty()) why += ", ";
            why += std::string(name) + "=" + std::to_string(got) + " > " + std::to_string(cap);
        }
    };
    over("M", M, MD);
    over("K", K, KD);
    over("N", N, ND);
    if (why.empty()) return {};
    return "launch does not fit the role's context (" + why + "; context is M=" + std::to_string(MD) +
           " K=" + std::to_string(KD) + " N=" + std::to_string(ND) + "): wrong role index for this shape?";
}

#endif
