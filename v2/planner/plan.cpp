#include "plan.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>

namespace onebit::v2 {

const GraphPlan* Plan::find(Phase p, int bucket) const {
    for (const auto& g : graphs)
        if (g.phase == p && g.bucket == bucket) return &g;
    return nullptr;
}

uint64_t Plan::max_scratch() const {
    uint64_t m = 0;
    for (const auto& g : graphs) m = std::max(m, g.scratch_bytes);
    return m;
}

std::string Plan::report(const Graph& g) const {
    std::string s = "plan for " + std::string(target_name(target)) + ": " + std::to_string(graphs.size()) + " graphs (" +
                    buckets.str() + "), " + std::to_string(g.nodes.size()) + " nodes each\n";
    char b[160];
    for (const auto& gp : graphs) {
        std::snprintf(b, sizeof b, "  %-7s bucket %5d: scratch %.1f MiB\n", phase_name(gp.phase), gp.bucket,
                      double(gp.scratch_bytes) / double(1 << 20));
        s += b;
    }
    // kernel use, from the largest decode graph
    if (!graphs.empty()) {
        std::map<std::string, int> uses;
        for (const auto& st : graphs.front().steps) uses[st.kernel]++;
        s += "  kernels (decode bucket " + std::to_string(graphs.front().bucket) + "):";
        for (const auto& [k, n] : uses) s += " " + k + " x" + std::to_string(n);
        s += "\n";
    }
    return s;
}

Plan build_plan(const Graph& g, const KernelTable& table, Target target, const BucketSet& buckets) {
    buckets.validate();
    Plan plan;
    plan.target = target;
    plan.buckets = buckets;
    const uint64_t act_per_token = g.peak_live_bytes_per_token();
    std::vector<std::string> missing;
    for (Phase phase : {Phase::Decode, Phase::Prefill}) {
        for (int bucket : buckets.of(phase)) {
            GraphPlan gp;
            gp.phase = phase;
            gp.bucket = bucket;
            uint64_t kernel_scratch = 0;
            for (size_t i = 0; i < g.nodes.size(); ++i) {
                std::string why;
                const KernelDesc* k = table.lookup(g.nodes[i], phase, bucket, target, &why);
                if (!k) {
                    missing.push_back(why);
                    continue;
                }
                gp.steps.push_back({int(i), k->id});
                if (k->scratch_bytes) kernel_scratch = std::max(kernel_scratch, k->scratch_bytes(g.nodes[i], g.hp, bucket));
            }
            gp.scratch_bytes = act_per_token * uint64_t(bucket) + kernel_scratch;
            plan.graphs.push_back(std::move(gp));
        }
    }
    if (!missing.empty()) {
        std::string msg = "plan: " + std::to_string(missing.size()) + " node/bucket pairs have no kernel on " + target_name(target) + ":";
        for (size_t i = 0; i < missing.size() && i < 8; ++i) msg += "\n  " + missing[i];
        if (missing.size() > 8) msg += "\n  ...";
        throw std::runtime_error(msg);
    }
    return plan;
}

}  // namespace onebit::v2
