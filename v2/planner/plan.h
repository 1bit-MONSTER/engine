#pragma once

// [IR] The graph plan, built once at load (design D3): one GraphPlan per (phase, bucket), each a
// total map from IR nodes to kernels. On the GPU each GraphPlan becomes one instantiated HRX graph
// (next slice); on the CPU reference executor it is the kernel list a step runs. The plan's
// scratch figure feeds the memory budget (D5: everything allocated at load).

#include "../ir/ir.h"
#include "buckets.h"
#include "kernel_table.h"

#include <cstdint>
#include <string>
#include <vector>

namespace onebit::v2 {

struct PlanStep {
    int node = -1;
    std::string kernel;
};

struct GraphPlan {
    Phase phase = Phase::Decode;
    int bucket = 0;
    std::vector<PlanStep> steps;
    uint64_t scratch_bytes = 0;   // activations (peak live x bucket) + kernel scratch
};

struct Plan {
    Target target = Target::CpuRef;
    BucketSet buckets;
    std::vector<GraphPlan> graphs;
    const GraphPlan* find(Phase p, int bucket) const;
    uint64_t max_scratch() const;
    size_t program_count() const { return graphs.size(); }   // bounded by construction
    std::string report(const Graph& g) const;
};

// Throws std::runtime_error listing every (node, phase, bucket) without a kernel.
Plan build_plan(const Graph& g, const KernelTable& table, Target target, const BucketSet& buckets);

}  // namespace onebit::v2
