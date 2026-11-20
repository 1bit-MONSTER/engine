#include "buckets.h"

#include <algorithm>
#include <stdexcept>

namespace onebit::v2 {

const char* phase_name(Phase p) {
    switch (p) {
        case Phase::Decode: return "decode";
        case Phase::Prefill: return "prefill";
        case Phase::Any: return "any";
    }
    return "?";
}

int BucketSet::bucket_for(Phase p, int n) const {
    if (n <= 0) return -1;
    for (int b : of(p))
        if (b >= n) return b;
    return -1;
}

BucketSet BucketSet::clamped(int n_ctx) const {
    BucketSet out = *this;
    out.prefill.clear();
    for (int b : prefill) {
        out.prefill.push_back(b);
        if (b >= n_ctx) break;
    }
    return out;
}

void BucketSet::validate() const {
    for (const auto* v : {&decode, &prefill}) {
        if (v->empty()) throw std::runtime_error("a bucket list is empty");
        for (size_t i = 0; i < v->size(); ++i) {
            if ((*v)[i] <= 0) throw std::runtime_error("bucket sizes must be positive");
            if (i && (*v)[i] <= (*v)[i - 1]) throw std::runtime_error("bucket sizes must be strictly increasing");
        }
    }
}

std::string BucketSet::str() const {
    std::string s = "decode {";
    for (size_t i = 0; i < decode.size(); ++i) s += (i ? "," : "") + std::to_string(decode[i]);
    s += "} prefill {";
    for (size_t i = 0; i < prefill.size(); ++i) s += (i ? "," : "") + std::to_string(prefill[i]);
    return s + "}";
}

std::vector<int> prefill_chunks(int n, const BucketSet& b) {
    std::vector<int> out;
    const int big = std::max(1, b.max_prefill());
    while (n > 0) {
        const int c = std::min(n, big);
        out.push_back(c);
        n -= c;
    }
    return out;
}

}  // namespace onebit::v2
