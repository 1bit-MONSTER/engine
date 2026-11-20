#pragma once

// [IR] The kernel table (design 2.2 [IR], D1, D11): (op, weight formats, phase, shape bucket,
// target) -> kernel id. The planner picks exactly one entry per IR node; an unmatched node is a
// load-time error, never a silent fallback, and there is no matcher priority race: the most
// specific key wins (exact formats beat "*", an exact bucket beats 0, an exact phase beats Any),
// and two entries with the same key are refused at registration.
//
// The table format is public; the tuned entries for GPU targets (the private HIP kernels and the
// tuning DB) register through the add-on hook later. This slice registers the CPU reference
// kernels (exec/cpu).

#include "../ir/ir.h"
#include "buckets.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace onebit::v2 {

enum class Target : unsigned char { CpuRef, Gfx1151, Gfx1150, NpuAie2p };
const char* target_name(Target t);
bool parse_target(const std::string& s, Target& out);

struct KernelKey {
    Op op = Op::EmbedNorm;
    std::string formats = "*";   // Node::formats() ("Q4_K+Q6_K"), or "*" for any
    Phase phase = Phase::Any;
    int bucket = 0;              // 0: any bucket
    Target target = Target::CpuRef;
    std::string str() const;
};

struct KernelDesc {
    std::string id;
    // "" when the kernel can run this node; otherwise why not (checked explicitly by the planner).
    std::function<std::string(const Node&)> precondition;
    // Scratch the kernel needs beyond the graph's activations for one step of `tokens` tokens.
    std::function<uint64_t(const Node&, const HParams&, int tokens)> scratch_bytes;
};

class KernelTable {
public:
    // Throws std::runtime_error when the key is already taken.
    void add(const KernelKey& key, KernelDesc desc);
    // The most specific entry whose key matches and whose precondition holds; nullptr with *why set
    // otherwise.
    const KernelDesc* lookup(const Node& n, Phase phase, int bucket, Target target, std::string* why = nullptr) const;
    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        KernelKey key;
        KernelDesc desc;
    };
    std::vector<Entry> entries_;
};

}  // namespace onebit::v2
