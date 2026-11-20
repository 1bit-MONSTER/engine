#pragma once

// [IR] Static shape buckets (design D3): every step runs a graph instantiated at load for a fixed
// token count, so the number of programs is bounded and nothing is compiled while serving. A step
// of n tokens is padded up to the smallest bucket that holds it. Attention's KV length is not part
// of the key: kernels read the paged KV and mask by length.

#include <string>
#include <vector>

namespace onebit::v2 {

enum class Phase : unsigned char { Decode, Prefill, Any };
const char* phase_name(Phase p);

struct BucketSet {
    std::vector<int> decode{1, 2, 4, 8, 16, 32};             // decode / MTP verify / --parallel tokens per step
    std::vector<int> prefill{128, 512, 2048, 4096, 8192};    // prefill chunk sizes (per model, tunable)

    const std::vector<int>& of(Phase p) const { return p == Phase::Decode ? decode : prefill; }
    // The smallest bucket of the phase holding n tokens, or -1 when n is larger than the largest.
    int bucket_for(Phase p, int n) const;
    int max_prefill() const { return prefill.empty() ? 0 : prefill.back(); }
    // Drops prefill buckets larger than the context (keeps the smallest one that covers it).
    BucketSet clamped(int n_ctx) const;
    // Sorted, unique, positive; throws std::runtime_error otherwise.
    void validate() const;
    std::string str() const;
};

// Splits a prompt of n tokens into prefill chunks no larger than the largest bucket
// (Sarathi-style chunked prefill, D6).
std::vector<int> prefill_chunks(int n, const BucketSet& b);

}  // namespace onebit::v2
