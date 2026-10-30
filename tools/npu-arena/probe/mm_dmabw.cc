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
// mm_dmabw.cc -- the arena's DMA-ceiling probe microkernel.
//
// It exports the SAME two symbols as mm_kernel_reference.cc (matmul_i8_i32,
// zero_i32) with the same signature, and does no arithmetic: the design's
// objectfifos, taps, tile geometry and column-group loop are byte-identical to
// the phase's real kernel, so what a probe launch measures is the data path only
// (weight stream + activation + result) with the matrix unit idle.
//
// The two `volatile` touches exist so neither the compiler nor the linker can
// drop the buffers: without them an empty kernel makes the design look faster
// than the DMA it is supposed to measure.
//
//   bytes/second from this kernel = the attainable ceiling for that phase's
//   tap/tile geometry on this box.

#include <stdint.h>

#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_K
#define DIM_K 64
#endif
#ifndef DIM_N
#define DIM_N 128
#endif

extern "C" void matmul_i8_i32(int8_t* a_in, int8_t* b_in, int32_t* c_out) {
    volatile int8_t sink = b_in[0];
    sink = (int8_t)(sink + a_in[0]);
    c_out[0] = (int32_t)sink;          // one store per tile: keeps c_out live
}

extern "C" void zero_i32(int32_t* c_out) {
    for (int i = 0; i < DIM_M * DIM_N; i += DIM_N) c_out[i] = 0;
}
