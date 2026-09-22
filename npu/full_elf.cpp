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

#include "full_elf.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>

namespace onebit::npu {
namespace {

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error("full ELF: " + what); }

uint32_t rd32(const Bytes& b, size_t o) {
    if (o + 4 > b.size()) fail("read past the end");
    uint32_t v;
    std::memcpy(&v, b.data() + o, 4);
    return v;
}
uint16_t rd16(const Bytes& b, size_t o) {
    if (o + 2 > b.size()) fail("read past the end");
    uint16_t v;
    std::memcpy(&v, b.data() + o, 2);
    return v;
}
void wr32(Bytes& b, size_t o, uint32_t v) { std::memcpy(b.data() + o, &v, 4); }
void put32(Bytes& b, uint32_t v) { b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 4); }
void put16(Bytes& b, uint16_t v) { b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 2); }
void put_str(Bytes& b, const std::string& s) { b.insert(b.end(), s.begin(), s.end()); b.push_back(0); }

// ---- ELF32 little-endian reading ----

struct Section {
    uint32_t offset, size;
};

std::map<std::string, Section> sections(const Bytes& e) {
    if (e.size() < 52 || std::memcmp(e.data(), "\x7f" "ELF\x01\x01", 6) != 0) fail("not an ELF32 LE file");
    const uint32_t shoff = rd32(e, 0x20);
    const uint16_t n = rd16(e, 0x30), si = rd16(e, 0x32);
    auto hdr = [&](int i, int field) { return rd32(e, shoff + 40u * i + 4u * field); };
    const uint32_t strtab = hdr(si, 4);
    std::map<std::string, Section> out;
    for (int i = 0; i < n; ++i) {
        const size_t name = strtab + hdr(i, 0);
        const auto end = std::find(e.begin() + long(name), e.end(), 0);
        out[std::string(e.begin() + long(name), end)] = {hdr(i, 4), hdr(i, 5)};
    }
    return out;
}

Section section(const std::map<std::string, Section>& s, const char* name) {
    auto it = s.find(name);
    if (it == s.end()) fail(std::string("no ") + name + " section");
    return it->second;
}

// The 16-byte descriptor of the XRT UID note: namesz 4 ("XRT\0"), descsz 16.
size_t uid_offset(const Bytes& e) {
    const Section n = section(sections(e), ".note.xrt.UID");
    if (n.size != 32 || rd32(e, n.offset) != 4 || rd32(e, n.offset + 4) != 16) fail("unexpected UID note layout");
    return n.offset + 16;
}

}  // namespace

Bytes read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    return {std::istreambuf_iterator<char>(f), {}};
}

// ---- Per-context control code ----

ContextMap ContextMap::derive(const Bytes& ctx1, const Bytes& ctx2, const Bytes& ctx17) {
    if (ctx1.size() != ctx2.size() || ctx1.size() != ctx17.size()) fail("context ELFs differ in size");
    const size_t uid = uid_offset(ctx1);
    ContextMap map;
    for (size_t o = 0; o + 4 <= ctx1.size(); o += 4) {
        if (o >= uid && o < uid + 16) continue;
        const uint32_t v1 = rd32(ctx1, o), v2 = rd32(ctx2, o), v17 = rd32(ctx17, o);
        if (v1 == v2 && v1 == v17) continue;
        if (v2 != v1) {
            const uint32_t step = v2 - v1;
            if (v17 != v1 + 16u * step) fail("word at " + std::to_string(o) + " is not linear in the context length");
            map.words.push_back({uint32_t(o), v1, step});
        } else {
            if (v17 != 2u * v1) fail("word at " + std::to_string(o) + " is not a 16-row block count");
            map.words.push_back({uint32_t(o), v1, 0});
        }
    }
    return map;
}

Bytes derive_context(const Bytes& ctx1, const ContextMap& map, int n) {
    if (n < 1) fail("context length must be at least 1");
    Bytes e = ctx1;
    const uint32_t blocks = uint32_t(n + ContextMap::kBlock - 1) / ContextMap::kBlock;
    for (const auto& w : map.words)
        wr32(e, w.offset, w.step ? w.base + uint32_t(n - 1) * w.step : w.base * blocks);
    const Section ct = section(sections(e), ".ctrltext");
    const auto digest = md5({e.data() + ct.offset, ct.size});
    std::memcpy(e.data() + uid_offset(e), digest.data(), 16);
    return e;
}

// ---- Full-ELF assembly ----

Bytes assemble_full_elf(const Bytes& inst, const Bytes& pdi, const std::string& name, PdiMode mode,
                        uint32_t config) {
    const auto S = sections(inst);
    const Section ct = section(S, ".ctrltext"), dsym = section(S, ".dynsym"), dstr = section(S, ".dynstr"),
                  rel = section(S, ".rela.dyn");

    struct Sym {
        uint32_t name, value, size;
        uint8_t info, other;
        uint16_t shndx;
    };
    struct Rela {
        uint32_t offset, info;
        int32_t addend;
    };
    std::vector<Sym> syms;
    for (uint32_t i = 0; i < dsym.size / 16; ++i) {
        const size_t o = dsym.offset + 16u * i;
        syms.push_back({rd32(inst, o), rd32(inst, o + 4), rd32(inst, o + 8), inst[o + 12], inst[o + 13], rd16(inst, o + 14)});
    }
    std::vector<Rela> relas;
    for (uint32_t i = 0; i < rel.size / 12; ++i) {
        const size_t o = rel.offset + 12u * i;
        relas.push_back({rd32(inst, o), rd32(inst, o + 4), int32_t(rd32(inst, o + 8))});
    }

    // 1. Control code: keep exactly the first transaction (the bytes its header
    // declares). Captured layer streams carry a second, identical, never-relocated
    // copy (the double-buffer slot); the xclbin path only ever executes the first.
    Bytes ctrl(inst.begin() + ct.offset, inst.begin() + ct.offset + ct.size);
    if (rd32(ctrl, 0) != 0x06040100) fail("control code does not start with a TXN header");
    uint32_t ops = rd32(ctrl, 8), size = rd32(ctrl, 12);
    if (size > ctrl.size()) fail("TXN header claims more bytes than the section holds");
    for (const auto& r : relas)
        if (r.offset >= size) fail("a relocation points past the first transaction");
    ctrl.resize(size);
    if (mode == PdiMode::kInitOnly) {
        ops = 0;
        size = 16;
        ctrl.resize(16);
        relas.clear();
        syms.resize(2);  // keep one buffer argument, referenced by nothing
    }
    // load_pdi after the header: op count +1, byte size +16, relocations shift.
    if (mode != PdiMode::kNone) {
        wr32(ctrl, 8, ops + 1);
        wr32(ctrl, 12, size + 16);
        static const uint32_t kLoadPdi[4] = {0x00010008, 0, 0, 0};
        ctrl.insert(ctrl.begin() + 16, reinterpret_cast<const uint8_t*>(kLoadPdi),
                    reinterpret_cast<const uint8_t*>(kLoadPdi) + 16);
        for (auto& r : relas)
            if (r.offset >= 16) r.offset += 16;
    }
    // DDR_PATCH ops carry a flag in bit 31 of their address word; the full-ELF
    // loader does not want it. Walk the ops (lengths in words) to find them.
    static const std::map<uint32_t, uint32_t> kOpWords = {{0x00, 6}, {0x01, 12}, {0x03, 7}, {0x80, 4}, {0x81, 12}};
    for (size_t i = mode == PdiMode::kNone ? 4 : 8; i < ctrl.size() / 4;) {
        const uint32_t w = rd32(ctrl, 4 * i);
        if (w == 0x81) wr32(ctrl, 4 * (i + 10), rd32(ctrl, 4 * (i + 10)) & 0x7FFFFFFF);
        auto it = kOpWords.find(w);
        if (it == kOpWords.end()) break;  // unknown op: stop rather than mis-step
        i += it->second;
    }

    // 2. Argument symbols: xclbin convention (3 = first buffer) -> full-ELF (0).
    auto symname = [&](const Sym& s) {
        const size_t o = dstr.offset + s.name;
        return std::string(inst.begin() + long(o), std::find(inst.begin() + long(o), inst.end(), 0));
    };
    Bytes new_dynstr = {0};
    put_str(new_dynstr, ".pdi.1");
    std::vector<Sym> new_syms = {syms[0], {1, 0, 0, 0x11, 0, 5}};  // .pdi.1: GLOBAL OBJECT
    int nargs = 0;
    for (size_t i = 1; i < syms.size(); ++i) {
        const int idx = std::stoi(symname(syms[i])) - 3;
        if (idx < 0) fail("argument symbol below 3");
        nargs = std::max(nargs, idx + 1);
        const uint32_t off = uint32_t(new_dynstr.size());
        put_str(new_dynstr, std::to_string(idx));
        new_syms.push_back({off, syms[i].value, syms[i].size, syms[i].info, syms[i].other, 5});
    }
    for (auto& r : relas) {
        r.info = (((r.info >> 8) + 1) << 8) | (r.info & 0xFF);  // symbols shifted by one
        if (r.addend < 0) r.addend &= 0x7FFFFFFF;                // no DDR_PATCH flag bit
    }
    if (mode != PdiMode::kNone) relas.insert(relas.begin(), {0x18, (1u << 8) | 8u, 0});  // load_pdi -> .pdi.1

    // 3. Static symbols: kernel FUNC and the COMDAT signature "sequence".
    std::string mangled = "_Z" + std::to_string(name.size()) + name;
    for (int i = 0; i < nargs; ++i) mangled += "Pc";
    Bytes strtab = {0};
    put_str(strtab, mangled);
    put_str(strtab, "sequence");
    Bytes symtab(16, 0);
    auto put_sym = [](Bytes& b, const Sym& s) {
        put32(b, s.name); put32(b, s.value); put32(b, s.size);
        b.push_back(s.info); b.push_back(s.other); put16(b, s.shndx);
    };
    put_sym(symtab, {1, 0, 0, 0x22, 0, 0});                              // WEAK FUNC UND
    put_sym(symtab, {uint32_t(1 + mangled.size() + 1), 0, 0, 0x21, 0, 1});  // WEAK OBJECT in .pdi.1

    Bytes note_cfg;
    put32(note_cfg, 4); put32(note_cfg, 4); put32(note_cfg, 6);
    put_str(note_cfg, "XRT");
    put32(note_cfg, config);
    Bytes note_uid;
    if (auto it = S.find(".note.xrt.UID"); it != S.end())
        note_uid.assign(inst.begin() + it->second.offset, inst.begin() + it->second.offset + it->second.size);
    else {
        put32(note_uid, 4); put32(note_uid, 16); put32(note_uid, 3);
        put_str(note_uid, "XRT");
        note_uid.resize(note_uid.size() + 16, 0);
    }

    // 4. Layout: ehdr, phdrs, .pdi.1, .ctrltext.0, then the tables.
    static const char* kNames[] = {"", ".shstrtab", ".strtab", ".symtab", ".pdi.1", ".ctrltext.0", ".dynstr",
                                   ".dynsym", ".rela.dyn", ".dynamic", ".group.0", ".note.xrt.configuration",
                                   ".note.xrt.UID"};
    constexpr int kSections = 13;
    Bytes shstrtab;
    uint32_t name_off[kSections];
    for (int i = 0; i < kSections; ++i) {
        name_off[i] = uint32_t(shstrtab.size());
        put_str(shstrtab, kNames[i]);
    }
    Bytes dynsym, rela;
    for (const auto& s : new_syms) put_sym(dynsym, s);
    for (const auto& r : relas) { put32(rela, r.offset); put32(rela, r.info); put32(rela, uint32_t(r.addend)); }
    Bytes dynamic;
    put32(dynamic, 7); put32(dynamic, 8); put32(dynamic, 8); put32(dynamic, uint32_t(rela.size()));
    Bytes group;
    put32(group, 1); put32(group, 5);  // GRP_COMDAT, member .ctrltext.0

    auto align = [](uint32_t x, uint32_t a) { return (x + a - 1) / a * a; };
    constexpr uint32_t kEhdr = 52, kPhnum = 3, kPhent = 32;
    uint32_t off = kEhdr + kPhnum * kPhent;
    const uint32_t pdi_off = align(off, 16);
    off = pdi_off + uint32_t(pdi.size());
    const uint32_t ct_off = align(off, 16);
    off = ct_off + uint32_t(ctrl.size());
    struct Blob { uint32_t offset; const Bytes* data; };
    auto place = [&](const Bytes& d, uint32_t a) { const uint32_t o = align(off, a); off = o + uint32_t(d.size()); return Blob{o, &d}; };
    const Blob b_shstr = place(shstrtab, 1), b_str = place(strtab, 1), b_sym = place(symtab, 4),
               b_dynstr = place(new_dynstr, 1), b_dynsym = place(dynsym, 8), b_rela = place(rela, 8),
               b_dyn = place(dynamic, 8), b_group = place(group, 4), b_cfg = place(note_cfg, 1),
               b_uid = place(note_uid, 1);
    const uint32_t shoff = align(off, 4);

    Bytes buf(shoff + 40u * kSections, 0);
    const uint8_t ident[16] = {0x7f, 'E', 'L', 'F', 1, 1, 1, 69, 16};
    std::memcpy(buf.data(), ident, 16);
    auto w16 = [&](size_t o, uint16_t v) { std::memcpy(buf.data() + o, &v, 2); };
    w16(16, 2); w16(18, 1);  // ET_EXEC, machine 1
    wr32(buf, 20, 1); wr32(buf, 24, 0); wr32(buf, 28, kEhdr); wr32(buf, 32, shoff); wr32(buf, 36, 0);
    w16(40, kEhdr); w16(42, kPhent); w16(44, kPhnum); w16(46, 40); w16(48, kSections); w16(50, 1);
    auto phdr = [&](int i, std::array<uint32_t, 8> f) { for (int k = 0; k < 8; ++k) wr32(buf, kEhdr + kPhent * i + 4 * k, f[k]); };
    phdr(0, {6, kEhdr, 0, 0, kPhnum * kPhent, kPhnum * kPhent, 4, 0});                                     // PHDR
    phdr(1, {1, pdi_off, 0, 0, uint32_t(pdi.size()), uint32_t(pdi.size()), 5, 16});                        // LOAD pdi
    phdr(2, {1, ct_off, uint32_t(pdi.size()), 0, uint32_t(ctrl.size()), uint32_t(ctrl.size()), 5, 16});    // LOAD ctrl
    std::copy(pdi.begin(), pdi.end(), buf.begin() + pdi_off);
    std::copy(ctrl.begin(), ctrl.end(), buf.begin() + ct_off);
    for (const Blob& b : {b_shstr, b_str, b_sym, b_dynstr, b_dynsym, b_rela, b_dyn, b_group, b_cfg, b_uid})
        std::copy(b.data->begin(), b.data->end(), buf.begin() + b.offset);
    auto sh = [&](int i, uint32_t type, uint32_t flags, uint32_t addr, uint32_t o, uint32_t sz, uint32_t link = 0,
                  uint32_t info = 0, uint32_t al = 1, uint32_t ent = 0) {
        const uint32_t f[10] = {name_off[i], type, flags, addr, o, sz, link, info, al, ent};
        for (int k = 0; k < 10; ++k) wr32(buf, shoff + 40u * i + 4 * k, f[k]);
    };
    auto sz = [](const Blob& b) { return uint32_t(b.data->size()); };
    sh(1, 3, 0, 0, b_shstr.offset, sz(b_shstr));
    sh(2, 3, 0, 0, b_str.offset, sz(b_str));
    sh(3, 2, 0, 0, b_sym.offset, sz(b_sym), 2, 1, 4, 16);
    sh(4, 1, 6, 0, pdi_off, uint32_t(pdi.size()), 0, 0, 16);
    sh(5, 1, 6 | 0x200, uint32_t(pdi.size()), ct_off, uint32_t(ctrl.size()), 0, 0, 16);  // SHF_GROUP
    sh(6, 3, 0, 0, b_dynstr.offset, sz(b_dynstr));
    sh(7, 11, 0, 0, b_dynsym.offset, sz(b_dynsym), 6, 1, 8, 16);
    sh(8, 4, 0, 0, b_rela.offset, sz(b_rela), 7, 0, 8, 12);
    sh(9, 6, 0, 0, b_dyn.offset, 16, 6, 0, 8, 8);
    sh(10, 17, 0, 0, b_group.offset, 8, 3, 2, 4, 4);
    sh(11, 7, 0, 0, b_cfg.offset, sz(b_cfg));
    sh(12, 7, 0, 0, b_uid.offset, sz(b_uid));
    return buf;
}

// ---- MD5 (RFC 1321) ----

std::array<uint8_t, 16> md5(std::span<const uint8_t> data) {
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    static const int R[16] = {7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21};
    uint32_t h[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
    Bytes m(data.begin(), data.end());
    const uint64_t bits = uint64_t(data.size()) * 8;
    m.push_back(0x80);
    while (m.size() % 64 != 56) m.push_back(0);
    for (int i = 0; i < 8; ++i) m.push_back(uint8_t(bits >> (8 * i)));
    for (size_t c = 0; c < m.size(); c += 64) {
        uint32_t w[16];
        std::memcpy(w, m.data() + c, 64);
        uint32_t a = h[0], b = h[1], cc = h[2], d = h[3];
        for (int i = 0; i < 64; ++i) {
            uint32_t f;
            int g;
            if (i < 16) { f = (b & cc) | (~b & d); g = i; }
            else if (i < 32) { f = (d & b) | (~d & cc); g = (5 * i + 1) % 16; }
            else if (i < 48) { f = b ^ cc ^ d; g = (3 * i + 5) % 16; }
            else { f = cc ^ (b | ~d); g = (7 * i) % 16; }
            const uint32_t t = d;
            d = cc;
            cc = b;
            b = b + std::rotl(a + f + K[i] + w[g], R[(i / 16) * 4 + i % 4]);
            a = t;
        }
        h[0] += a; h[1] += b; h[2] += cc; h[3] += d;
    }
    std::array<uint8_t, 16> out;
    std::memcpy(out.data(), h, 16);
    return out;
}

}  // namespace onebit::npu
