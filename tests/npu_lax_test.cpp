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

// npu_lax_test                                    (CI: pure host, no model)
// npu_lax_test --model <dir> --sha256 <tsv> [--ref <dir>] [--only a,b,...] [--out <dir>] [--threads N]
// npu_lax_test --insts <lax_a insts.bin> --patches <tsv>
// npu_lax_test --elf <kernel dir> --sites <tsv> [--golden <dir>]
//
// The lax decode's host side (npu/lax_pack.h, npu/lax_stream.h) against the reference
// packer, the open kernels' Python (third_party/OpenFlowLM-Next open_kernels/recipes/
// pack.py at the pin; tests/golden/npu_lax/ holds what it produced):
//
// CI: SHA-256 test vectors; the q8 -> q4_1 re-quantization and the signed-nibble
// transcode on synthetic chunks from a 64-bit LCG (x = x * 6364136223846793005 +
// 1442695040888963407, byte = x >> 56), hashed against the reference's output for the
// same bytes; every chunk permutation's index array (int64) hashed against the
// reference's; the 4096-row position table; the position patches on a synthetic stream;
// the cfg words.
//
// --model: packs every buffer of the model (ptab, normw, lmpool, pool_<L>, consts_<L>)
// and compares its SHA-256 with the reference's table; with --ref, the files there
// (pool_0.bin, ...) are also compared byte for byte, naming the first difference; --out
// writes ours there.
//
// --insts: the position patches found in the lax_a stream equal the tsv (word, kind,
// flags) the reference harness's table produced for the same build.
//
// CI also runs the full-ELF position sites (npu/lax_stream.h, elf_position_sites) on a
// synthetic instruction ELF: which words and addends move, their values at a position,
// the flag bit kept until assemble_full_elf clears it, and the malformed inputs refused.
//
// --elf: the position sites of the kernel directory's lax_a/insts.elf equal the tsv (the
// investigation's attn_sites for the same build); with --golden, every full ELF
// (npu/lax_elf.h) is byte-identical to the one the reference harness assembled and ran
// (harness/full_elf.py lax: lax_init, lxf, axf_p<N>, ln, lm).
#include "full_elf.h"
#include "lax_elf.h"
#include "lax_pack.h"
#include "lax_stream.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace onebit::npu;
using namespace onebit::npu::lax;

namespace {

int failures = 0;
std::mutex mu;
void check(bool ok, const std::string& what) {
    std::lock_guard lk(mu);
    if (!ok) ++failures;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

std::string sha(const std::vector<uint8_t>& v) { return hex(sha256(v)); }
std::string sha_str(const std::string& s) { return hex(sha256({reinterpret_cast<const uint8_t*>(s.data()), s.size()})); }

struct Lcg {
    uint64_t x;
    uint8_t byte() {
        x = x * 6364136223846793005ull + 1442695040888963407ull;
        return uint8_t(x >> 56);
    }
};

std::vector<uint8_t> q8_chunks(int n) {
    Lcg g{1};
    std::vector<uint8_t> out(size_t(n) * kQ8, 0);
    for (int c = 0; c < n; ++c) {
        uint8_t* p = out.data() + size_t(c) * kQ8;
        for (int k = 0; k < 256; ++k) {
            uint16_t u = uint16_t((c % 4 != 3 ? 0x3A00 : 0xBA00) + g.byte());
            if (c == 4 && k % 3 == 0) u = 0;  // zero-scale blocks: +0 and -0 values, the min's sign
            p[2 * k] = uint8_t(u), p[2 * k + 1] = uint8_t(u >> 8);
        }
        for (int i = 0; i < 8192; ++i) {
            uint8_t b = g.byte();
            if (c == 1) b = 0;          // constant blocks: d = 0
            if (c == 2) b &= 0x7F;      // non-negative codes
            p[512 + i] = b;
        }
    }
    return out;
}

std::vector<uint8_t> q4_chunks(int n) {
    Lcg g{2};
    std::vector<uint8_t> out(size_t(n) * kQ41, 0);
    for (int c = 0; c < n; ++c) {
        uint8_t* p = out.data() + size_t(c) * kQ41;
        for (int k = 0; k < 256; ++k) {
            const uint16_t u = uint16_t(0x3A00 + g.byte());
            p[2 * k] = uint8_t(u), p[2 * k + 1] = uint8_t(u >> 8);
        }
        for (size_t i = 1024; i < kQ41; ++i) p[i] = g.byte();
    }
    return out;
}

template <class F> std::string perm_sha(size_t n, F f) {
    std::vector<uint8_t> b(n * 8);
    for (size_t i = 0; i < n; ++i) {
        const int64_t v = int64_t(f(i));
        std::memcpy(b.data() + 8 * i, &v, 8);
    }
    return sha(b);
}

// ---- a synthetic instruction ELF (the layout aiebu-asm writes) ----------------------------

struct Rel {
    uint32_t offset, sym;
    uint32_t addend;
};

// ELF32 LE with .shstrtab, .ctrltext, .dynstr, .dynsym (symbol 0 null, then one per
// argument name) and .rela.dyn: what elf_position_sites and assemble_full_elf read.
Bytes make_insts_elf(const std::vector<uint32_t>& ctrl, const std::vector<std::string>& args, const std::vector<Rel>& rels) {
    auto put32 = [](Bytes& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i))); };
    auto put16 = [](Bytes& b, uint16_t v) { b.push_back(uint8_t(v)); b.push_back(uint8_t(v >> 8)); };
    Bytes shstr = {0}, ct, dynstr = {0}, dynsym(16, 0), rela;
    const std::vector<std::string> names = {"", ".shstrtab", ".ctrltext", ".dynstr", ".dynsym", ".rela.dyn"};
    std::vector<uint32_t> name_off;
    for (const auto& n : names) {
        name_off.push_back(uint32_t(shstr.size()));
        shstr.insert(shstr.end(), n.begin(), n.end());
        shstr.push_back(0);
    }
    for (uint32_t v : ctrl) put32(ct, v);
    for (const auto& a : args) {
        put32(dynsym, uint32_t(dynstr.size()));
        put32(dynsym, 0), put32(dynsym, 0);
        dynsym.push_back(0x11), dynsym.push_back(0), put16(dynsym, 0);
        dynstr.insert(dynstr.end(), a.begin(), a.end());
        dynstr.push_back(0);
    }
    for (const auto& r : rels) put32(rela, r.offset), put32(rela, (r.sym << 8) | 5), put32(rela, r.addend);
    Bytes e(52, 0);
    const std::vector<const Bytes*> blobs = {&shstr, &ct, &dynstr, &dynsym, &rela};
    std::vector<uint32_t> off;
    for (const Bytes* b : blobs) {
        while (e.size() % 4) e.push_back(0);
        off.push_back(uint32_t(e.size()));
        e.insert(e.end(), b->begin(), b->end());
    }
    while (e.size() % 4) e.push_back(0);
    const uint32_t shoff = uint32_t(e.size());
    e.resize(e.size() + 40 * names.size(), 0);
    const uint8_t ident[] = {0x7f, 'E', 'L', 'F', 1, 1, 1};
    std::memcpy(e.data(), ident, sizeof ident);
    auto w32 = [&](size_t o, uint32_t v) { std::memcpy(e.data() + o, &v, 4); };
    auto w16 = [&](size_t o, uint16_t v) { std::memcpy(e.data() + o, &v, 2); };
    w32(0x20, shoff);
    w16(0x2E, 40), w16(0x30, uint16_t(names.size())), w16(0x32, 1);
    const uint32_t types[] = {0, 3, 1, 3, 11, 4};
    for (size_t i = 1; i < names.size(); ++i) {
        const size_t h = shoff + 40 * i;
        w32(h, name_off[i]), w32(h + 4, types[i]), w32(h + 16, off[i - 1]), w32(h + 20, uint32_t(blobs[i - 1]->size()));
    }
    return e;
}

uint32_t rd(const Bytes& b, size_t o) {
    uint32_t v;
    std::memcpy(&v, b.data() + o, 4);
    return v;
}

void elf_sites_unit() {
    // TXN header, then per transfer the BD blockwrite (register at +2, BD words from +4:
    // length, address = 0) and the DDR_PATCH on the BD's address register (arg at +8,
    // offset at +10). Transfers: the KV window fill (arg 3, offset 0), a weight fill (arg
    // 0), the row drain (arg 3, one row), the record fill (arg 5, folded: bit 31 set).
    std::vector<uint32_t> w = {0x06040100, 0, 0, 0};
    std::vector<Rel> rels;
    // symbols: 1 -> "3" (arg 0), 2 -> "6" (arg 3), 3 -> "8" (arg 5)
    auto xfer = [&](uint32_t reg, uint32_t sym, uint32_t arg, uint32_t len, uint32_t off) {
        const size_t bd = w.size();
        w.insert(w.end(), {0x01, 0, reg, 32, len, 0, 0, 0, 0, 0, 0, 0});
        w.insert(w.end(), {0x81, 0, 0, 0, 0, 0, reg + 4, 0, arg, 0, off, 0});
        rels.push_back({uint32_t(4 * (bd + 4)), sym, off});
    };
    xfer(0x1D000, 2, 3, 512, 0);
    w.insert(w.end(), {0x00, 0, 0, 0, 0, 0});
    xfer(0x1D020, 1, 0, 99, 0x1E1E0000);
    xfer(0x1D040, 2, 3, 64, uint32_t(kKvRow));
    w.insert(w.end(), {0x80, 0, 0, 0});
    xfer(0x1D060, 3, 5, 64, uint32_t(kPtabRow) | 0x80000000u);
    w[2] = 10, w[3] = uint32_t(4 * w.size());
    const std::vector<std::string> args = {"3", "6", "8"};
    Bytes e = make_insts_elf(w, args, rels);

    const auto sites = elf_position_sites(e, kKvRow);
    check(sites.size() == 7, "ELF: 7 position sites (window length, 3 x (DDR_PATCH word, addend))");
    auto find = [&](PosPatch::Kind k, ElfPosSite::Where where) {
        for (const auto& x : sites)
            if (x.kind == k && x.where == where) return x;
        throw std::runtime_error("site missing");
    };
    check(find(PosPatch::WindowLength, ElfPosSite::CtrlWord).index == 8, "ELF: window length = the fill BD's first word");
    check(find(PosPatch::WindowOffset, ElfPosSite::CtrlWord).index == 26 &&
              find(PosPatch::WindowOffset, ElfPosSite::Addend).index == 0,
          "ELF: window offset = DDR_PATCH word 26 + relocation 0");
    check(find(PosPatch::RowDrain, ElfPosSite::CtrlWord).index == 80 &&
              find(PosPatch::RowDrain, ElfPosSite::Addend).index == 2,
          "ELF: row drain = its DDR_PATCH word + relocation 2 (the weight fill's is skipped)");
    const auto rec = find(PosPatch::Record, ElfPosSite::Addend);
    check(rec.index == 3 && rec.flags == 0x80000000u, "ELF: record = relocation 3, folded bit kept");

    apply_elf_position(e, sites, 7, kKvRow, kPtabRow);
    const ElfSection ct = elf_section(e, ".ctrltext"), rl = elf_section(e, ".rela.dyn");
    auto word = [&](size_t i) { return rd(e, ct.offset + 4 * i); };
    auto addend = [&](size_t k) { return rd(e, rl.offset + 12 * k + 8); };
    check(word(8) == 7 * kKvRow / 4, "ELF position 7: window length 7 rows in words");
    check(word(26) == 0 && addend(0) == 0, "ELF position 7: window from row 0 (word and addend)");
    check(word(80) == 7 * kKvRow && addend(2) == 7 * kKvRow, "ELF position 7: row drain at row 7 (word and addend)");
    check(word(108) == (7 * kPtabRow | 0x80000000u) && addend(3) == (7 * kPtabRow | 0x80000000u),
          "ELF position 7: record at ptab row 7, folded bit kept (word and addend)");
    check(addend(1) == 0x1E1E0000 && word(56) == 0x1E1E0000, "ELF position 7: the weight fill untouched");
    apply_elf_position(e, sites, 0, kKvRow, kPtabRow);
    check(word(8) == kKvRow / 4 && addend(2) == 0 && addend(3) == 0x80000000u, "ELF position 0: one masked row, row 0");

    // What XRT gets: the assembled full ELF has the fold cleared in the addend and the word.
    apply_elf_position(e, sites, 5, kKvRow, kPtabRow);
    const Bytes full = assemble_full_elf(e, Bytes(64, 0xAB), "axf5", PdiMode::kNone);
    const ElfSection fct = elf_section(full, ".ctrltext.0"), frl = elf_section(full, ".rela.dyn");
    check(rd(full, frl.offset + 12 * 3 + 8) == 5 * kPtabRow && rd(full, fct.offset + 4 * 80) == 5 * kKvRow && rd(full, fct.offset + 4 * 108) == 5 * kPtabRow,
          "ELF assembled at position 5: record addend and word without bit 31, drain word 5 rows");

    // Refused: a relocation naming another argument, and two relocations on one BD.
    auto throws = [](auto f) { try { f(); } catch (const std::exception&) { return true; } return false; };
    auto bad = rels;
    bad[2].sym = 1;
    check(throws([&] { elf_position_sites(make_insts_elf(w, args, bad), kKvRow); }), "ELF: a drain relocation on arg 0 is refused");
    bad = rels;
    bad.push_back(rels[2]);
    check(throws([&] { elf_position_sites(make_insts_elf(w, args, bad), kKvRow); }), "ELF: two relocations on one BD are refused");
    bad = rels;
    bad[3].addend = 0x80000800;
    check(throws([&] { elf_position_sites(make_insts_elf(w, args, bad), kKvRow); }),
          "ELF: an addend that disagrees with its DDR_PATCH word is refused");
}

void unit() {
    check(sha_str("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 of \"\"");
    check(sha_str("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 of \"abc\"");
    check(sha_str("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "sha256 of the two-block vector");

    const auto q8 = q8_chunks(8);
    check(sha(q8) == "8647fb97cb1e8d15fbbfeb9e180f948100c7907244fd8d40b6d2d207f447825c", "q8 test chunks (LCG)");
    std::vector<uint8_t> q41(8 * kQ41);
    for (int c = 0; c < 8; ++c) requant_q8(q8.data() + c * kQ8, q41.data() + c * kQ41);
    check(sha(q41) == "a854c6d6d5e6b455e5b8f3398c257b9b283c1b5ee6919ec11d785729dc40ecb7", "q8 -> q4_1 = reference");

    const auto q4 = q4_chunks(2);
    check(sha(q4) == "b70797fdc38cded3eb1220a5d99f5eef76cfb3a90777c0e495c7afdaad2c9c0f", "q4 test chunks (LCG)");
    std::vector<uint8_t> q4c(2 * kQ41);
    for (int c = 0; c < 2; ++c) q4_0_to_q4_1(q4.data() + c * kQ41, q4c.data() + c * kQ41);
    check(sha(q4c) == "46a78bf613286056665d8699b4fd6e4ec637c062bb0b95935562fbcfa1b1b302", "signed q4 -> q4_1 = reference");
    check(is_signed_q4(q4.data()) && !is_signed_q4(q4c.data()), "signed-nibble detection (every min zero)");

    check(perm_sha(2048, [](size_t c) { return std_src(c, 2048); }) ==
              "b9125fb145c43a8f1fce01a94677f04d8c529e93aceb61d812c88611a1eda2cd",
          "std band, 2048 chunks x 2048 wide");
    check(perm_sha(128, [](size_t c) { return std_src(c, 512); }) ==
              "6f1d98e68120d1228af04ce8065bd7d3448e025a993cf52b9802a8d940b068ca",
          "std band, 128 chunks x 512 wide");
    check(perm_sha(1024, [](size_t c) { return std_src(c, 4096); }) ==
              "9510e09e730a6d3b904cb53ddfc1680ea268d97aef846eb1b428a64a84e1c355",
          "std band, 1024 chunks x 4096 wide");
    check(perm_sha(128, down_src) == "8a4458d562f3f2881bc485dcf5a26d0b6e688869ff197bcffdee7ec71b5dbcb4",
          "expert down slice");
    check(perm_sha(32, stripe_src) == "23f038693437780b8fbac60e5bf7e22c328ba308129ea0d43ceda6c22fdab70d",
          "expert up/gate stripe");
    check(perm_sha(62080, [](size_t k) { return lmhead_src(k, 2048); }) ==
              "03075dcdb60acada1095d81c64493328285f9104fed5222475af20c1ff298290",
          "q8 lm head supertiles");

    Config c;
    c.rotary_dim = 64;
    c.rope_theta = 1e7;
    std::vector<uint8_t> pt(kPtabBytes);
    pack_ptab(c, kMaxContext, pt.data());
    check(sha(pt) == "2a11c3c044d7cd4cd3a07abb7c2b5e40e6fe11b2793c891b44ccf5e28d2ad8f7", "position table, 4096 rows");

    // A synthetic stream: header, a BD write (register 0x1D000, length word), the KV window
    // fill on arg 3 at offset 0 for that register + 4, the row drain at one row with the
    // translation bit, the record fill on arg 5, and filler ops of every length.
    std::vector<uint32_t> w = {0x06030100, 0, 0, 0};
    auto op = [&](std::vector<uint32_t> o) { w.insert(w.end(), o.begin(), o.end()); };
    op({0x80, 0, 0, 0});
    op({0x01, 0, 0x1D000, 0, 512, 0, 0, 0, 0, 0, 0, 0});
    op({0x81, 0, 0, 0, 0, 0, 0x1D004, 0, 3, 0, 0, 0});
    op({0x00, 0, 0, 0, 0, 0});
    op({0x81, 0, 0, 0, 0, 0, 0x1D024, 0, 3, 0, uint32_t(kKvRow) | 0x80000000u, 0});
    op({0x03, 0, 0, 0, 0, 0, 0});
    op({0x81, 0, 0, 0, 0, 0, 0x1D044, 0, 5, 0, uint32_t(kPtabRow) | 0x80000000u, 0});
    op({0x81, 0, 0, 0, 0, 0, 0x1D064, 0, 0, 0, 12345, 0});  // a weight fill: untouched
    const auto patches = position_patches(w, kKvRow);
    check(patches.size() == 4, "four position patches in the synthetic stream");
    apply_position(w, patches, 7, kKvRow, kPtabRow);
    check(w[4 + 4 + 4] == 7 * kKvRow / 4, "window length: 7 rows in words");
    check(w[4 + 4 + 12 + 10] == 0, "window offset: row 0");
    check(w[4 + 4 + 12 + 12 + 6 + 10] == (7 * kKvRow | 0x80000000u), "row drain: row 7, translation bit kept");
    check(w[4 + 4 + 12 + 12 + 6 + 12 + 7 + 10] == (7 * kPtabRow | 0x80000000u), "record: ptab row 7");
    check(w[4 + 4 + 12 + 12 + 6 + 12 + 7 + 12 + 10] == 12345, "weight fill untouched");
    apply_position(w, patches, 0, kKvRow, kPtabRow);
    check(w[4 + 4 + 4] == kKvRow / 4, "position 0 streams one (masked) row");

    elf_sites_unit();

    const auto cw = cfg_words(0x1'2345'6780'0000ull);
    check(cw[0] == 0x67800000u && cw[1] == 0x12345u && cw[2] == 0x1D21C && cw[3] == 0x1D214 && cw[9] == 0x1D214,
          "cfg words: pool address lo/hi, then the eight column queues");
}

std::map<std::string, std::pair<size_t, std::string>> read_table(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::map<std::string, std::pair<size_t, std::string>> t;
    for (std::string line; std::getline(f, line);) {
        std::istringstream in(line);
        std::string name, digest;
        size_t bytes = 0;
        if (in >> name >> bytes >> digest) t[name] = {bytes, digest};
    }
    return t;
}

void compare_file(const std::string& path, const std::vector<uint8_t>& ours, const std::string& name) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return;
    std::vector<uint8_t> ref(size_t(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(ref.data()), std::streamsize(ref.size()));
    size_t first = 0;
    const size_t n = std::min(ref.size(), ours.size());
    while (first < n && ref[first] == ours[first]) ++first;
    check(ref.size() == ours.size() && first == n,
          name + " byte for byte vs " + path + (first < n ? " (first difference at byte " + std::to_string(first) + ")" : ""));
}

int model_mode(const std::string& dir, const std::string& table_path, const std::string& ref, const std::string& only,
               const std::string& out, int threads) {
    const Model m(dir);
    const Config c = Config::from_model(m, dir);
    const auto table = read_table(table_path);
    std::vector<std::string> names = {"ptab", "normw", "lmpool"};
    for (int L = 0; L < c.layers; ++L) {
        names.push_back("pool_" + std::to_string(L));
        names.push_back("consts_" + std::to_string(L));
    }
    if (!only.empty()) {
        std::vector<std::string> keep;
        std::stringstream ss(only);
        for (std::string s; std::getline(ss, s, ',');) keep.push_back(s);
        names = keep;
    }
    std::atomic<size_t> next{0};
    auto work = [&] {
        for (size_t i; (i = next++) < names.size();) {
            const std::string& n = names[i];
            std::vector<uint8_t> b;
            try {
                if (n == "ptab") {
                    b.resize(kPtabBytes);
                    pack_ptab(c, kMaxContext, b.data());
                } else if (n == "normw") {
                    b.resize(kHidden * 2);
                    pack_norm(m, b.data());
                } else if (n == "lmpool") {
                    b.resize(kLmPoolBytes);
                    pack_lmhead(m, b.data());
                } else if (n.rfind("pool_", 0) == 0) {
                    b.resize(kPoolBytes);
                    pack_pool(m, c, std::stoi(n.substr(5)), b.data());
                } else if (n.rfind("consts_", 0) == 0) {
                    const int L = std::stoi(n.substr(7));
                    b.resize(consts_bytes(c.kinds[size_t(L)]));
                    pack_consts(m, c, L, b.data());
                } else throw std::runtime_error("unknown buffer " + n);
            } catch (const std::exception& e) {
                check(false, n + ": " + e.what());
                continue;
            }
            const auto it = table.find(n);
            const std::string d = sha(b);
            if (it == table.end()) check(false, n + ": not in the reference table");
            else
                check(it->second.first == b.size() && it->second.second == d,
                      n + " sha256 " + d + (it->second.second == d ? "" : " != reference " + it->second.second));
            if (!ref.empty()) compare_file(ref + "/" + n + ".bin", b, n);
            if (!out.empty()) std::ofstream(out + "/" + n + ".bin", std::ios::binary).write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
        }
    };
    if (threads <= 0) threads = 4;
    std::vector<std::thread> ts;
    for (int t = 0; t < threads; ++t) ts.emplace_back(work);
    for (auto& t : ts) t.join();
    return 0;
}

int insts_mode(const std::string& insts, const std::string& expect) {
    std::ifstream f(insts, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + insts);
    std::vector<uint32_t> w(size_t(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(w.data()), std::streamsize(w.size() * 4));
    const auto p = position_patches(w, kKvRow);
    std::ifstream e(expect);
    if (!e) throw std::runtime_error("cannot read " + expect);
    std::vector<PosPatch> want;
    for (std::string line; std::getline(e, line);) {
        std::istringstream in(line);
        size_t word;
        int kind;
        uint32_t flags;
        if (in >> word >> kind >> std::hex >> flags) want.push_back({word, PosPatch::Kind(kind), flags});
    }
    bool same = p.size() == want.size();
    for (size_t i = 0; same && i < p.size(); ++i)
        same = p[i].word == want[i].word && p[i].kind == want[i].kind && p[i].flags == want[i].flags;
    for (const auto& x : p) std::printf("  word %zu kind %d flags %08x\n", x.word, int(x.kind), x.flags);
    check(same, "lax_a position patches = the reference harness's table");
    return 0;
}

void pdi_less_positions(const ElfInputs& in);

// tsv lines: where (word | reloc), index, kind, flags (hex)
int elf_mode(const std::string& dir, const std::string& expect, const std::string& golden) {
    const auto in = ElfInputs::read(dir);
    std::ifstream f(expect);
    if (!f) throw std::runtime_error("cannot read " + expect);
    std::vector<ElfPosSite> want;
    for (std::string line; std::getline(f, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream l(line);
        std::string where;
        size_t index;
        int kind;
        uint32_t flags;
        if (l >> where >> index >> kind >> std::hex >> flags)
            want.push_back({0, PosPatch::Kind(kind), flags, where == "reloc" ? ElfPosSite::Addend : ElfPosSite::CtrlWord, index});
    }
    bool same = in.sites.size() == want.size();
    for (size_t i = 0; same && i < want.size(); ++i)
        same = in.sites[i].where == want[i].where && in.sites[i].index == want[i].index && in.sites[i].kind == want[i].kind &&
               in.sites[i].flags == want[i].flags;
    for (const auto& x : in.sites)
        std::printf("  %s %zu kind %d flags %08x\n", x.where == ElfPosSite::Addend ? "reloc" : "word ", x.index, int(x.kind),
                    x.flags);
    check(same, "lax_a insts.elf position sites = the investigation's attn_sites");
    pdi_less_positions(in);
    if (golden.empty()) return 0;
    auto cmp = [&](const Bytes& ours, const std::string& file) {
        const std::string p = golden + "/" + file;
        if (!std::filesystem::exists(p)) {
            check(false, file + " missing in " + golden);
            return;
        }
        const Bytes ref = read_file(p);
        size_t first = 0;
        while (first < std::min(ours.size(), ref.size()) && ours[first] == ref[first]) ++first;
        check(ours == ref, "full ELF " + file + " = the harness's" +
                               (ours == ref ? "" : " (sizes " + std::to_string(ours.size()) + " / " + std::to_string(ref.size()) +
                                                       ", first difference at byte " + std::to_string(first) + ")"));
    };
    cmp(init_elf(in), "lax_init.elf");
    cmp(linear_elf(in), "lxf.elf");
    for (size_t p = 0; std::filesystem::exists(golden + "/axf_p" + std::to_string(p) + ".elf"); ++p)
        cmp(full_elf(in, p, true), "axf_p" + std::to_string(p) + ".elf");
    cmp(norm_elf(in), "ln.elf");
    cmp(head_elf(in), "lm.elf");
    return 0;
}

// What the decode adds per position is the same ELF without the PDI (npu/lax_elf.h).
void pdi_less_positions(const ElfInputs& in) {
    bool same = true;
    for (size_t p : {size_t(0), size_t(1), size_t(2), size_t(4095)}) {
        const Bytes a = full_elf(in, p), b = full_elf(in, p, true);
        same = same && elf_section(a, ".pdi.1").size == 0;
        for (const char* sec : {".ctrltext.0", ".rela.dyn", ".dynsym", ".dynstr", ".symtab", ".strtab", ".note.xrt.UID"}) {
            const ElfSection x = elf_section(a, sec), y = elf_section(b, sec);
            same = same && x.size == y.size && std::equal(a.begin() + x.offset, a.begin() + x.offset + x.size, b.begin() + y.offset);
        }
    }
    check(same, "position configs: no PDI, control code, relocations and symbols as with it (positions 0, 1, 2, 4095)");
}

}  // namespace

int main(int argc, char** argv) {
    std::string model, table, ref, only, out, insts, patches, elf, sites, golden;
    int threads = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : throw std::runtime_error(a + " needs a value"); };
        if (a == "--model") model = next();
        else if (a == "--sha256") table = next();
        else if (a == "--ref") ref = next();
        else if (a == "--only") only = next();
        else if (a == "--out") out = next();
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--insts") insts = next();
        else if (a == "--patches") patches = next();
        else if (a == "--elf") elf = next();
        else if (a == "--sites") sites = next();
        else if (a == "--golden") golden = next();
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    try {
        if (!model.empty()) model_mode(model, table, ref, only, out, threads);
        else if (!insts.empty()) insts_mode(insts, patches);
        else if (!elf.empty()) elf_mode(elf, sites, golden);
        else unit();
    } catch (const std::exception& e) {
        std::printf("FAIL %s\n", e.what());
        return 1;
    }
    std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
