#include "formats.h"

#include <cstring>
#include <stdexcept>

namespace onebit::v2 {

namespace {

struct Entry {
    uint32_t id;
    FmtInfo info;
};

constexpr Entry kFormats[] = {
    {0, {"F32", 1, 4, true}},       {1, {"F16", 1, 2, true}},       {2, {"Q4_0", 32, 18, true}},
    {3, {"Q4_1", 32, 20, true}},    {6, {"Q5_0", 32, 22, true}},    {7, {"Q5_1", 32, 24, true}},
    {8, {"Q8_0", 32, 34, true}},    {9, {"Q8_1", 32, 36, false}},   {10, {"Q2_K", 256, 84, false}},
    {11, {"Q3_K", 256, 110, false}}, {12, {"Q4_K", 256, 144, true}}, {13, {"Q5_K", 256, 176, true}},
    {14, {"Q6_K", 256, 210, true}}, {15, {"Q8_K", 256, 292, false}}, {16, {"IQ2_XXS", 256, 66, false}},
    {17, {"IQ2_XS", 256, 74, false}}, {18, {"IQ3_XXS", 256, 98, false}}, {19, {"IQ1_S", 256, 50, false}},
    {20, {"IQ4_NL", 32, 18, false}}, {21, {"IQ3_S", 256, 110, false}}, {22, {"IQ2_S", 256, 82, false}},
    {23, {"IQ4_XS", 256, 136, false}}, {24, {"I8", 1, 1, false}},     {25, {"I16", 1, 2, false}},
    {26, {"I32", 1, 4, false}},     {27, {"I64", 1, 8, false}},     {28, {"F64", 1, 8, false}},
    {29, {"IQ1_M", 256, 56, false}}, {30, {"BF16", 1, 2, true}},     {34, {"TQ1_0", 256, 54, false}},
    {35, {"TQ2_0", 256, 66, false}}, {39, {"MXFP4", 32, 17, false}},
};

inline uint16_t rd16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}

// Q4_K / Q5_K: the 6-bit scale and min of sub-block j from the 12 packed bytes.
inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

void deq_q4_0(const uint8_t* b, float* y) {
    const float d = f16_to_f32(rd16(b));
    const uint8_t* qs = b + 2;
    for (int j = 0; j < 16; ++j) {
        y[j] = float(int(qs[j] & 0xF) - 8) * d;
        y[j + 16] = float(int(qs[j] >> 4) - 8) * d;
    }
}

void deq_q4_1(const uint8_t* b, float* y) {
    const float d = f16_to_f32(rd16(b)), m = f16_to_f32(rd16(b + 2));
    const uint8_t* qs = b + 4;
    for (int j = 0; j < 16; ++j) {
        y[j] = float(qs[j] & 0xF) * d + m;
        y[j + 16] = float(qs[j] >> 4) * d + m;
    }
}

void deq_q5_0(const uint8_t* b, float* y) {
    const float d = f16_to_f32(rd16(b));
    uint32_t qh;
    std::memcpy(&qh, b + 2, 4);
    const uint8_t* qs = b + 6;
    for (int j = 0; j < 16; ++j) {
        const uint8_t h0 = uint8_t(((qh >> j) << 4) & 0x10);
        const uint8_t h1 = uint8_t((qh >> (j + 12)) & 0x10);
        y[j] = float(int((qs[j] & 0xF) | h0) - 16) * d;
        y[j + 16] = float(int((qs[j] >> 4) | h1) - 16) * d;
    }
}

void deq_q5_1(const uint8_t* b, float* y) {
    const float d = f16_to_f32(rd16(b)), m = f16_to_f32(rd16(b + 2));
    uint32_t qh;
    std::memcpy(&qh, b + 4, 4);
    const uint8_t* qs = b + 8;
    for (int j = 0; j < 16; ++j) {
        const uint8_t h0 = uint8_t(((qh >> j) << 4) & 0x10);
        const uint8_t h1 = uint8_t((qh >> (j + 12)) & 0x10);
        y[j] = float((qs[j] & 0xF) | h0) * d + m;
        y[j + 16] = float((qs[j] >> 4) | h1) * d + m;
    }
}

void deq_q8_0(const uint8_t* b, float* y) {
    const float d = f16_to_f32(rd16(b));
    const auto* qs = reinterpret_cast<const int8_t*>(b + 2);
    for (int j = 0; j < 32; ++j) y[j] = float(qs[j]) * d;
}

// block_q4_K: f16 d, f16 dmin, u8 scales[12], u8 qs[128]
void deq_q4_k(const uint8_t* b, float* y) {
    const float d = f16_to_f32(rd16(b)), dmin = f16_to_f32(rd16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* q = b + 16;
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is, sc, s, m);
        const float d1 = d * s, m1 = dmin * m;
        scale_min_k4(is + 1, sc, s, m);
        const float d2 = d * s, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * float(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * float(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

// block_q5_K: f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]
void deq_q5_k(const uint8_t* b, float* y) {
    const float d = f16_to_f32(rd16(b)), dmin = f16_to_f32(rd16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s, m;
        scale_min_k4(is, sc, s, m);
        const float d1 = d * s, m1 = dmin * m;
        scale_min_k4(is + 1, sc, s, m);
        const float d2 = d * s, m2 = dmin * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * float((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * float((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        is += 2;
        u1 = uint8_t(u1 << 2);
        u2 = uint8_t(u2 << 2);
    }
}

// block_q6_K: u8 ql[128], u8 qh[64], i8 scales[16], f16 d
void deq_q6_k(const uint8_t* b, float* y) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const auto* sc = reinterpret_cast<const int8_t*>(b + 192);
    const float d = f16_to_f32(rd16(b + 208));
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * float(sc[is + 0]) * float(q1);
            y[l + 32] = d * float(sc[is + 2]) * float(q2);
            y[l + 64] = d * float(sc[is + 4]) * float(q3);
            y[l + 96] = d * float(sc[is + 6]) * float(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

}  // namespace

float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                --exp;
            }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float bf16_to_f32(uint16_t h) {
    const uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

const FmtInfo* fmt_info(uint32_t id) {
    for (const auto& e : kFormats)
        if (e.id == id) return &e.info;
    return nullptr;
}

std::string fmt_name(uint32_t id) {
    const FmtInfo* f = fmt_info(id);
    return f ? f->name : "type" + std::to_string(id);
}

uint64_t fmt_bytes(uint32_t id, uint64_t n) {
    const FmtInfo* f = fmt_info(id);
    if (!f || n % f->block) return 0;
    return n / f->block * f->bytes;
}

void dequantize(uint32_t id, const void* src, float* dst, size_t n) {
    const FmtInfo* f = fmt_info(id);
    if (!f || !f->cpu_dequant) throw std::runtime_error("no CPU dequantizer for " + fmt_name(id));
    if (n % f->block) throw std::runtime_error("dequantize: " + std::to_string(n) + " elements is not whole " + f->name + " blocks");
    const auto* b = static_cast<const uint8_t*>(src);
    const size_t nb = n / f->block;
    switch (Fmt(id)) {
        case Fmt::F32: std::memcpy(dst, src, n * 4); return;
        case Fmt::F16: for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(rd16(b + 2 * i)); return;
        case Fmt::BF16: for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(rd16(b + 2 * i)); return;
        case Fmt::Q4_0: for (size_t i = 0; i < nb; ++i) deq_q4_0(b + i * 18, dst + i * 32); return;
        case Fmt::Q4_1: for (size_t i = 0; i < nb; ++i) deq_q4_1(b + i * 20, dst + i * 32); return;
        case Fmt::Q5_0: for (size_t i = 0; i < nb; ++i) deq_q5_0(b + i * 22, dst + i * 32); return;
        case Fmt::Q5_1: for (size_t i = 0; i < nb; ++i) deq_q5_1(b + i * 24, dst + i * 32); return;
        case Fmt::Q8_0: for (size_t i = 0; i < nb; ++i) deq_q8_0(b + i * 34, dst + i * 32); return;
        case Fmt::Q4_K: for (size_t i = 0; i < nb; ++i) deq_q4_k(b + i * 144, dst + i * 256); return;
        case Fmt::Q5_K: for (size_t i = 0; i < nb; ++i) deq_q5_k(b + i * 176, dst + i * 256); return;
        case Fmt::Q6_K: for (size_t i = 0; i < nb; ++i) deq_q6_k(b + i * 210, dst + i * 256); return;
        default: break;
    }
    throw std::runtime_error("no CPU dequantizer for " + fmt_name(id));
}

}  // namespace onebit::v2
