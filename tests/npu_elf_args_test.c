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

// npu_elf_args_test -- q4nx_elf_arg_sizes() reads the argument widths a full ELF declares.
//
// The widths are the launcher's contract: the kernel's shim DMA touches exactly the bytes the ELF's
// .dynsym declares for each argument, so a narrower buffer makes the DMA walk past the mapping.
// That is not hypothetical: on 2026-11-02 an OGA pack's ELF family -- (weights, activations,
// output) = (655,360, 2,048, 4,096) B -- was driven by the int8 launcher, which passes
// (MD*KD, KD*ND, MD*ND*4) = (16,384, 1,048,576, 65,536) B, and the array logged IO_PAGE_FAULTs
// while returning garbage instead of failing.  The gate in I8Ctx::init_elf compares these two sets
// and refuses; this test pins the reader it depends on.
//
// Host-only: the fixtures are hand-built ELF32 images, so no aiecc build and no device are needed.
#include "q4nx_pack.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(int cond, const char* what) {
    printf("%s %s\n", cond ? "ok  " : "FAIL", what);
    if (!cond) failures++;
}

static void put32(unsigned char* p, uint32_t v) { memcpy(p, &v, 4); }
static void put16(unsigned char* p, uint16_t v) { memcpy(p, &v, 2); }

// A minimal ELF32 (LSB) with [ehdr][.dynsym][.strtab][.shstrtab][shtab] and one STT_OBJECT symbol
// per argument, named "0".."2" -- the shape aiecc --get-full-elf writes for a full ELF.
static unsigned char* build_elf(const uint32_t sizes[3], uint32_t* out_len) {
    enum { NSYM = 4, NDYN = NSYM * 16, NSTR = 8, NSHSTR = 24, NSH = 5 };
    const uint32_t off_dyn = 52, off_str = off_dyn + NDYN, off_shstr = off_str + NSTR;
    const uint32_t off_sht = off_shstr + NSHSTR, total = off_sht + NSH * 40;
    unsigned char* elf = calloc(1, total);
    if (!elf) return NULL;
    memcpy(elf, "\x7f" "ELF", 4);
    elf[4] = 1; elf[5] = 1; elf[6] = 1;                      // ELF32, little-endian, version 1
    put32(elf + 0x20, off_sht); put16(elf + 0x2E, 40); put16(elf + 0x30, NSH); put16(elf + 0x32, 4);
    memcpy(elf + off_str, "\0" "0" "\0" "1" "\0" "2" "\0", NSTR);       // names at 1, 3, 5
    memcpy(elf + off_shstr, "\0.dynsym\0.strtab\0.shstrtab\0", NSHSTR);
    for (int i = 1; i < NSYM; i++) {
        unsigned char* sym = elf + off_dyn + i * 16;
        put32(sym, (uint32_t)(2 * i - 1));                    // 1, 3, 5
        put32(sym + 8, sizes[i - 1]);
        sym[12] = 0x11;                                       // STB_GLOBAL | STT_OBJECT
        put16(sym + 14, 5);
    }
    struct { const char* name; uint32_t type, off, size, link, entsize; } sh[NSH] = {
        {"", 0, 0, 0, 0, 0},
        {".dynsym", 11, off_dyn, NDYN, 2, 16},
        {".strtab", 3, off_str, NSTR, 0, 0},
        {".shstrtab", 3, off_shstr, NSHSTR, 0, 0},
        {"", 0, 0, 0, 0, 0}};
    const uint32_t name_off[NSH] = {0, 1, 9, 17, 0};
    for (int i = 0; i < NSH; i++) {
        unsigned char* p = elf + off_sht + i * 40;
        const uint32_t f[10] = {name_off[i], sh[i].type, 0, 0, sh[i].off, sh[i].size, sh[i].link, 0, 4,
                                sh[i].entsize};
        memcpy(p, f, sizeof f);
    }
    *out_len = total;
    return elf;
}

int main(void) {
    // the OGA pack's attention ELF: weights, activations, output
    const uint32_t sizes[3] = {163840, 2048, 1024};
    uint32_t len = 0;
    unsigned char* elf = build_elf(sizes, &len);
    if (!elf) { printf("FAIL out of memory\n"); return 1; }

    uint64_t got[3] = {0, 0, 0};
    check(q4nx_elf_arg_sizes(elf, len, got) == 3, "three argument widths are found");
    check(got[0] == 163840 && got[1] == 2048 && got[2] == 1024, "widths match the declared symbols");

    // the tile family declares much narrower arguments; the reader must report them as they are
    const uint32_t tile[3] = {512, 1024, 4096};
    unsigned char* e2 = build_elf(tile, &len);
    uint64_t got2[3] = {0, 0, 0};
    check(e2 && q4nx_elf_arg_sizes(e2, len, got2) == 3 && got2[0] == 512 && got2[2] == 4096,
          "a narrow tile-family ELF reads back as declared");

    // refusals: not an ELF, no .dynsym, a truncated header -- and out[] is zeroed first either way
    uint32_t len0 = 0;
    unsigned char* bad = build_elf(sizes, &len0);
    bad[0] = 'X';
    uint64_t z[3] = {7, 7, 7};
    check(q4nx_elf_arg_sizes(bad, len0, z) == -1, "a non-ELF is refused");
    check(z[0] == 0 && z[1] == 0 && z[2] == 0, "out[] is zeroed before the refusal");
    bad[0] = 0x7f;
    const uint32_t zero_type = 0;
    const uint32_t off_dyn = 52, off_sht = off_dyn + 4 * 16 + 8 + 24;
    memcpy(bad + off_sht + 40 + 4, &zero_type, 4);            // .dynsym -> SHT_NULL
    check(q4nx_elf_arg_sizes(bad, len0, z) == -1, "an ELF with no .dynsym is refused");
    check(q4nx_elf_arg_sizes(elf, 16, z) == -1, "a truncated header is refused");
    check(q4nx_elf_arg_sizes(NULL, len, z) == -1, "a null image is refused");

    // ── the gemv family's rule, which decides how a launch is shaped (engine#855) ──
    // Declarations measured on the OGA 0.5B pack's five ELFs: (stream/4, 2K, N) at K=1024/4864.
    // The per-core split is real, so a rule that compared arg0 or arg2 against the FULL width would
    // mis-classify every one of them -- which is exactly what happened before this rule existed.
    {
        struct { int K, N; uint64_t decl[3]; int want; const char* what; } cases[] = {
            {1024, 1536,   {245760, 2048, 1536},    1, "QKV K=1024 N=1536 (per-core stream, whole row)"},
            {1024, 1024,   {163840, 2048, 1024},    1, "O K=1024 N=1024"},
            {1024, 9728,   {1556480, 2048, 9728},   1, "GU K=1024 N=9728"},
            {4864, 1024,   {778240, 9728, 1024},    1, "D K=4864 N=1024 (arg1 is 2*K)"},
            {1024, 152064, {24330240, 2048, 152064}, 1, "head K=1024 N=152064"},
            // the int8 per-tile family: (tile, K int8, N int32) -- must NOT be taken for gemv
            {6144, 5120,   {512, 1024, 4096},       0, "an int8 per-tile ELF"},
            // near misses: full-width stream, wrong output width, wrong K grid, arg1 not 2K
            {1024, 1536,   {983040, 2048, 1536},    0, "stream given whole instead of per core"},
            {1024, 1536,   {245760, 2048, 1536 * 4}, 0, "output given in bytes instead of elements"},
            {1024, 1536,   {245760, 1024, 1536},    0, "int8-width activations"},
            {1000, 1536,   {245760, 2000, 1536},    0, "K not on the 256 grid"},
            {1024, 1500,   {0, 2048, 1536},         0, "zero stream width"},
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
            const int got = q4nx_gemv_family(cases[i].decl, cases[i].K, cases[i].N);
            if (got != cases[i].want) {
                printf("FAIL gemv family: %s\n", cases[i].what);
                failures++;
            }
        }
        printf("gemv family: %zu declarations classified\n", sizeof cases / sizeof cases[0]);
    }

    free(elf); free(e2); free(bad);
    printf("%s\n", failures ? "FAILED" : "all cases ok");
    return failures ? 1 : 0;
}
