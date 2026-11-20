#include "kernel_table.h"

#include <stdexcept>

namespace onebit::v2 {

const char* target_name(Target t) {
    switch (t) {
        case Target::CpuRef: return "cpu-ref";
        case Target::Gfx1151: return "gfx1151";
        case Target::Gfx1150: return "gfx1150";
        case Target::NpuAie2p: return "npu-aie2p";
    }
    return "?";
}

bool parse_target(const std::string& s, Target& out) {
    for (Target t : {Target::CpuRef, Target::Gfx1151, Target::Gfx1150, Target::NpuAie2p})
        if (s == target_name(t)) {
            out = t;
            return true;
        }
    return false;
}

std::string KernelKey::str() const {
    return std::string(op_name(op)) + "/" + formats + "/" + phase_name(phase) + "/" + (bucket ? std::to_string(bucket) : "*") + "/" +
           target_name(target);
}

void KernelTable::add(const KernelKey& key, KernelDesc desc) {
    if (desc.id.empty()) throw std::runtime_error("kernel table: an entry needs an id");
    for (const auto& e : entries_)
        if (e.key.op == key.op && e.key.formats == key.formats && e.key.phase == key.phase && e.key.bucket == key.bucket &&
            e.key.target == key.target)
            throw std::runtime_error("kernel table: key " + key.str() + " is taken by " + e.desc.id + "; refusing " + desc.id);
    entries_.push_back({key, std::move(desc)});
}

const KernelDesc* KernelTable::lookup(const Node& n, Phase phase, int bucket, Target target, std::string* why) const {
    const std::string fmts = n.formats();
    const Entry* best = nullptr;
    int best_score = -1;
    std::string reasons;
    for (const auto& e : entries_) {
        const KernelKey& k = e.key;
        if (k.op != n.op || k.target != target) continue;
        if (k.formats != "*" && k.formats != fmts) continue;
        if (k.phase != Phase::Any && k.phase != phase) continue;
        if (k.bucket != 0 && k.bucket != bucket) continue;
        if (e.desc.precondition) {
            const std::string r = e.desc.precondition(n);
            if (!r.empty()) {
                reasons += (reasons.empty() ? "" : "; ") + e.desc.id + ": " + r;
                continue;
            }
        }
        const int score = (k.formats != "*" ? 4 : 0) + (k.bucket != 0 ? 2 : 0) + (k.phase != Phase::Any ? 1 : 0);
        if (score > best_score) {
            best = &e;
            best_score = score;
        }
    }
    if (!best && why)
        *why = "no kernel for " + n.label() + " [" + fmts + "] " + phase_name(phase) + " bucket " + std::to_string(bucket) + " on " +
               target_name(target) + (reasons.empty() ? "" : " (" + reasons + ")");
    return best ? &best->desc : nullptr;
}

}  // namespace onebit::v2
