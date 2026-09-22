#include "unicode.h"

#include <algorithm>
#include <array>
#include <span>

namespace onebit::unicode {

namespace {

#include "unicode_tables.inc"

bool in_ranges(std::span<const uint32_t[2]> table, char32_t cp) {
    auto it = std::upper_bound(table.begin(), table.end(), uint32_t(cp),
                               [](uint32_t v, const uint32_t(&r)[2]) { return v < r[0]; });
    if (it == table.begin()) return false;
    --it;
    return cp <= (*it)[1];
}

uint8_t ccc(char32_t cp) {
    if (cp < 0x300) return 0;
    std::span<const uint32_t[3]> table(kCccRanges);
    auto it = std::upper_bound(table.begin(), table.end(), uint32_t(cp),
                               [](uint32_t v, const uint32_t(&r)[3]) { return v < r[0]; });
    if (it == table.begin()) return 0;
    --it;
    return cp <= (*it)[1] ? uint8_t((*it)[2]) : 0;
}

// Hangul syllable algorithm constants (Unicode ch. 3.12).
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

void decompose(char32_t cp, std::vector<char32_t>& out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
        const uint32_t s = cp - kSBase;
        out.push_back(kLBase + s / kNCount);
        out.push_back(kVBase + (s % kNCount) / kTCount);
        if (s % kTCount) out.push_back(kTBase + s % kTCount);
        return;
    }
    auto it = std::lower_bound(std::begin(kDecomp), std::end(kDecomp), uint32_t(cp),
                               [](const DecompEntry& e, uint32_t v) { return e.cp < v; });
    if (it != std::end(kDecomp) && it->cp == cp) {
        out.insert(out.end(), kDecompData + it->offset, kDecompData + it->offset + it->len);
    } else {
        out.push_back(cp);
    }
}

std::optional<char32_t> compose_pair(char32_t a, char32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount)
        return a + (b - kTBase);
    const uint64_t key = uint64_t(a) << 21 | b;
    auto it = std::lower_bound(std::begin(kCompose), std::end(kCompose), key,
                               [](const ComposeEntry& e, uint64_t v) { return e.key < v; });
    if (it != std::end(kCompose) && it->key == key) return it->composite;
    return std::nullopt;
}

}  // namespace

std::optional<std::vector<char32_t>> decode_utf8(std::string_view s) {
    std::vector<char32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        size_t n;
        char32_t cp, min;
        if (b0 < 0x80) { n = 1; cp = b0; min = 0; }
        else if ((b0 & 0xE0) == 0xC0) { n = 2; cp = b0 & 0x1F; min = 0x80; }
        else if ((b0 & 0xF0) == 0xE0) { n = 3; cp = b0 & 0x0F; min = 0x800; }
        else if ((b0 & 0xF8) == 0xF0) { n = 4; cp = b0 & 0x07; min = 0x10000; }
        else return std::nullopt;
        if (i + n > s.size()) return std::nullopt;
        for (size_t k = 1; k < n; ++k) {
            const auto b = static_cast<unsigned char>(s[i + k]);
            if ((b & 0xC0) != 0x80) return std::nullopt;
            cp = cp << 6 | (b & 0x3F);
        }
        if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return std::nullopt;
        out.push_back(cp);
        i += n;
    }
    return out;
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += char(cp);
    } else if (cp < 0x800) {
        out += char(0xC0 | cp >> 6);
        out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += char(0xE0 | cp >> 12);
        out += char(0x80 | (cp >> 6 & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    } else {
        out += char(0xF0 | cp >> 18);
        out += char(0x80 | (cp >> 12 & 0x3F));
        out += char(0x80 | (cp >> 6 & 0x3F));
        out += char(0x80 | (cp & 0x3F));
    }
}

std::string encode_utf8(const std::vector<char32_t>& cps) {
    std::string out;
    out.reserve(cps.size());
    for (char32_t cp : cps) append_utf8(out, cp);
    return out;
}

bool is_letter(char32_t cp) {
    if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
    return in_ranges(kLetterRanges, cp);
}

bool is_number(char32_t cp) {
    if (cp < 0x80) return cp >= '0' && cp <= '9';
    return in_ranges(kNumberRanges, cp);
}

bool is_whitespace(char32_t cp) {
    switch (cp) {
        case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20: case 0x85: case 0xA0:
        case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000:
            return true;
        default:
            return cp >= 0x2000 && cp <= 0x200A;
    }
}

std::vector<char32_t> nfc(const std::vector<char32_t>& in) {
    // Fast path: nothing below U+0300 decomposes or combines with a predecessor.
    if (std::all_of(in.begin(), in.end(), [](char32_t c) { return c < 0x300; })) return in;

    std::vector<char32_t> d;
    d.reserve(in.size() + 8);
    for (char32_t cp : in) decompose(cp, d);

    // Canonical ordering: stable sort each run of non-starters by combining class.
    for (size_t i = 0; i < d.size();) {
        if (ccc(d[i]) == 0) { ++i; continue; }
        size_t j = i;
        while (j < d.size() && ccc(d[j]) != 0) ++j;
        std::stable_sort(d.begin() + i, d.begin() + j, [](char32_t a, char32_t b) { return ccc(a) < ccc(b); });
        i = j;
    }

    // Canonical composition.
    std::vector<char32_t> out;
    out.reserve(d.size());
    size_t starter = SIZE_MAX;  // index in out of the last starter
    int last_ccc = -1;          // ccc of the last character appended after the starter
    for (char32_t cp : d) {
        const int c = ccc(cp);
        if (starter != SIZE_MAX && (last_ccc < c || (last_ccc == 0 && out.size() == starter + 1))) {
            if (auto comp = compose_pair(out[starter], cp)) {
                out[starter] = *comp;
                continue;
            }
        }
        if (c == 0) {
            starter = out.size();
            last_ccc = 0;
        } else {
            last_ccc = c;
        }
        out.push_back(cp);
    }
    return out;
}

}  // namespace onebit::unicode
