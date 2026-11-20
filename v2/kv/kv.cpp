#include "kv.h"

#include <algorithm>
#include <stdexcept>

namespace onebit::v2 {

namespace {
inline uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
}  // namespace

uint64_t KvManager::chain_key(uint64_t parent, const int* tokens, int n) {
    uint64_t h = mix(parent ^ 0x6B76u);
    for (int i = 0; i < n; ++i) h = mix(h ^ uint64_t(uint32_t(tokens[i])));
    return h;
}

KvManager::KvManager(KvConfig cfg, ZeroFn zero) : cfg_(cfg), zero_(std::move(zero)) {
    if (cfg_.block_tokens <= 0 || cfg_.n_blocks <= 0) throw std::runtime_error("KV: block_tokens and n_blocks must be positive");
    blocks_.resize(size_t(cfg_.n_blocks));
    for (int b = cfg_.n_blocks - 1; b >= 0; --b) free_.push_back(b);
}

KvManager::Seq& KvManager::seq(int id) {
    if (id < 0 || size_t(id) >= seqs_.size() || !seqs_[size_t(id)].open) throw std::runtime_error("KV: no open sequence " + std::to_string(id));
    return seqs_[size_t(id)];
}

const KvManager::Seq& KvManager::seq(int id) const {
    if (id < 0 || size_t(id) >= seqs_.size() || !seqs_[size_t(id)].open) throw std::runtime_error("KV: no open sequence " + std::to_string(id));
    return seqs_[size_t(id)];
}

int KvManager::open() {
    for (size_t i = 0; i < seqs_.size(); ++i)
        if (!seqs_[i].open) {
            seqs_[i] = Seq{};
            seqs_[i].open = true;
            return int(i);
        }
    seqs_.push_back(Seq{});
    seqs_.back().open = true;
    return int(seqs_.size() - 1);
}

int KvManager::take_block() {
    int b = -1;
    if (!free_.empty()) {
        b = free_.back();
        free_.pop_back();
    } else if (!lru_.empty()) {
        b = lru_.front();
        lru_.pop_front();
        by_key_.erase(blocks_[size_t(b)].key);
        blocks_[size_t(b)].published = false;
        blocks_[size_t(b)].tokens.clear();
    } else {
        return -1;
    }
    blocks_[size_t(b)].ref = 1;
    // zero on reuse: no reader may see another request's rows
    if (zero_) zero_(b);
    ++zero_calls_;
    return b;
}

void KvManager::unref(int b) {
    Block& k = blocks_[size_t(b)];
    if (k.ref <= 0) throw std::runtime_error("KV: block released twice");
    if (--k.ref > 0) return;
    if (k.published) lru_.push_back(b);
    else free_.push_back(b);
}

int KvManager::match_prefix(int id, const std::vector<int>& tokens) {
    Seq& s = seq(id);
    if (s.length != 0) throw std::runtime_error("KV: match_prefix on a non-empty sequence");
    const int bt = cfg_.block_tokens;
    const int full = tokens.empty() ? 0 : int((tokens.size() - 1) / size_t(bt));
    for (int i = 0; i < full; ++i) {
        const int* t = tokens.data() + size_t(i) * size_t(bt);
        const uint64_t key = chain_key(s.chain, t, bt);
        auto it = by_key_.find(key);
        if (it == by_key_.end()) break;
        Block& b = blocks_[size_t(it->second)];
        if (!std::equal(b.tokens.begin(), b.tokens.end(), t)) break;   // a hash collision: not ours
        if (b.ref == 0) lru_.remove(it->second);
        ++b.ref;
        s.blocks.push_back(it->second);
        s.tokens.insert(s.tokens.end(), t, t + bt);
        s.length += bt;
        s.chain = key;
        ++prefix_hits_;
    }
    return s.length;
}

bool KvManager::reserve(int id, int n) {
    Seq& s = seq(id);
    if (n < 0) throw std::runtime_error("KV: negative reserve");
    const int bt = cfg_.block_tokens;
    const int need = (s.length + n + bt - 1) / bt - int(s.blocks.size());
    if (need <= 0) return true;
    std::vector<int> got;
    for (int i = 0; i < need; ++i) {
        const int b = take_block();
        if (b < 0) {
            for (int g : got) {
                blocks_[size_t(g)].ref = 0;
                free_.push_back(g);
            }
            return false;
        }
        got.push_back(b);
    }
    s.blocks.insert(s.blocks.end(), got.begin(), got.end());
    return true;
}

void KvManager::commit(int id, const std::vector<int>& tokens) {
    Seq& s = seq(id);
    const int bt = cfg_.block_tokens;
    if (size_t(s.length) + tokens.size() > s.blocks.size() * size_t(bt)) throw std::runtime_error("KV: commit past the reserved blocks");
    for (int t : tokens) {
        s.tokens.push_back(t);
        ++s.length;
        if (s.length % bt) continue;
        const int bi = s.length / bt - 1;
        const uint64_t key = chain_key(s.chain, s.tokens.data() + size_t(bi) * size_t(bt), bt);
        s.chain = key;
        Block& b = blocks_[size_t(s.blocks[size_t(bi)])];
        if (b.published || by_key_.count(key)) continue;   // already cached (ours or an identical one)
        b.published = true;
        b.key = key;
        b.tokens.assign(s.tokens.end() - bt, s.tokens.end());
        by_key_[key] = s.blocks[size_t(bi)];
    }
}

void KvManager::truncate(int id, int new_length) {
    Seq& s = seq(id);
    if (new_length < 0 || new_length > s.length) throw std::runtime_error("KV: bad truncate length");
    const int bt = cfg_.block_tokens;
    const size_t keep = size_t((new_length + bt - 1) / bt);
    // the block that becomes partial will be written again: it must not stay a shared cache entry
    if (new_length % bt && keep > 0) {
        Block& b = blocks_[size_t(s.blocks[keep - 1])];
        if (b.published) {
            if (b.ref > 1) throw std::runtime_error("KV: truncate into a block shared with another sequence (copy-on-write is a later slice)");
            by_key_.erase(b.key);
            b.published = false;
            b.tokens.clear();
        }
    }
    while (s.blocks.size() > keep) {
        unref(s.blocks.back());
        s.blocks.pop_back();
    }
    s.tokens.resize(size_t(new_length));
    s.length = new_length;
    s.chain = 0;
    for (int i = 0; i < new_length / bt; ++i) s.chain = chain_key(s.chain, s.tokens.data() + size_t(i) * size_t(bt), bt);
}

void KvManager::close(int id) {
    Seq& s = seq(id);
    for (int b : s.blocks) unref(b);
    s = Seq{};
}

int KvManager::length(int id) const { return seq(id).length; }
const std::vector<int>& KvManager::blocks(int id) const { return seq(id).blocks; }
int KvManager::free_blocks() const { return int(free_.size() + lru_.size()); }
int KvManager::cached_blocks() const { return int(lru_.size()); }

// ── StateSlots ───────────────────────────────────────────────────────────────

StateSlots::StateSlots(int n_slots, int n_snapshots, uint64_t slot_bytes, ZeroFn zero, CopyFn copy)
    : n_slots_(n_slots), n_snaps_(n_snapshots), slot_bytes_(slot_bytes), zero_(std::move(zero)), copy_(std::move(copy)),
      used_(size_t(std::max(0, n_slots)), false), snap_used_(size_t(std::max(0, n_snapshots)), false) {
    if (n_slots <= 0 || n_snapshots < 0) throw std::runtime_error("state slots: bad counts");
}

int StateSlots::acquire() {
    for (int i = 0; i < n_slots_; ++i)
        if (!used_[size_t(i)]) {
            used_[size_t(i)] = true;
            if (zero_) zero_(i);
            return i;
        }
    return -1;
}

void StateSlots::release(int slot) {
    if (slot < 0 || slot >= n_slots_ || !used_[size_t(slot)]) throw std::runtime_error("state slots: release of a free slot");
    used_[size_t(slot)] = false;
}

int StateSlots::snapshot(int slot) {
    if (slot < 0 || slot >= n_slots_ || !used_[size_t(slot)]) throw std::runtime_error("state slots: snapshot of a free slot");
    for (int i = 0; i < n_snaps_; ++i)
        if (!snap_used_[size_t(i)]) {
            snap_used_[size_t(i)] = true;
            if (copy_) copy_(slot, n_slots_ + i);
            return n_slots_ + i;
        }
    return -1;
}

void StateSlots::rollback(int slot, int snap) {
    const int s = snap - n_slots_;
    if (slot < 0 || slot >= n_slots_ || !used_[size_t(slot)] || s < 0 || s >= n_snaps_ || !snap_used_[size_t(s)])
        throw std::runtime_error("state slots: bad rollback");
    if (copy_) copy_(snap, slot);
}

void StateSlots::drop_snapshot(int snap) {
    const int s = snap - n_slots_;
    if (s < 0 || s >= n_snaps_ || !snap_used_[size_t(s)]) throw std::runtime_error("state slots: bad snapshot");
    snap_used_[size_t(s)] = false;
}

int StateSlots::in_use() const { return int(std::count(used_.begin(), used_.end(), true)); }

}  // namespace onebit::v2
