#pragma once

// [KV] Paged KV block manager (design 2.2 [KV], D7).
//
//   - One pool of fixed-size blocks (block_tokens tokens of K and V for every layer); a sequence
//     owns a block table.
//   - Prefix cache: hash-chained full blocks (vLLM APC style): block i's key is
//     hash(key of block i-1, the block's tokens), and the tokens are stored with the block, so a hash
//     collision cannot share a wrong block. A closed sequence's full blocks stay cached (ref 0) and
//     are evicted least-recently-used when the pool runs out.
//   - Zero/mask on reuse (the engine #400 lesson: stale rows past the position): a block handed to
//     a new owner without a prefix hit goes through the zero hook before any read, and every reader
//     is given the sequence length so it masks rows past it. Tests run a different request first.
//
// The manager is device-independent bookkeeping; the executor owns the storage and supplies the
// zero hook (memset on the CPU, a fill kernel on the GPU).

#include <cstdint>
#include <functional>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

namespace onebit::v2 {

struct KvConfig {
    int block_tokens = 64;   // D7: 64, tunable 16-128
    int n_blocks = 0;
};

class KvManager {
public:
    using ZeroFn = std::function<void(int block)>;
    KvManager(KvConfig cfg, ZeroFn zero);

    int open();                 // a new, empty sequence; returns its id
    // Reuses cached full blocks matching the start of `tokens` (only whole blocks, and never the
    // whole prompt: the last token must still be computed for its logits). Call on an empty
    // sequence. Returns the number of tokens reused.
    int match_prefix(int seq, const std::vector<int>& tokens);
    // Makes room for n more tokens. False (and nothing changed) when the pool cannot supply the
    // blocks even after evicting cached free blocks.
    bool reserve(int seq, int n);
    // Records n tokens as written at the end of the sequence (after the executor wrote their K/V).
    // Blocks that become full are published to the prefix cache.
    void commit(int seq, const std::vector<int>& tokens);
    // Drops the last n tokens (speculation rollback); blocks emptied are released.
    void truncate(int seq, int new_length);
    void close(int seq);

    int length(int seq) const;
    const std::vector<int>& blocks(int seq) const;
    int block_tokens() const { return cfg_.block_tokens; }
    int n_blocks() const { return cfg_.n_blocks; }
    int free_blocks() const;      // never used or evictable
    int cached_blocks() const;    // published, ref 0
    uint64_t zero_calls() const { return zero_calls_; }
    uint64_t prefix_hits() const { return prefix_hits_; }

private:
    struct Block {
        int ref = 0;
        bool published = false;
        uint64_t key = 0;
        std::vector<int> tokens;   // the block's tokens once published
    };
    struct Seq {
        bool open = false;
        int length = 0;
        std::vector<int> blocks;
        std::vector<int> tokens;   // all tokens committed (for hashing)
        uint64_t chain = 0;        // key of the last full block
    };
    KvConfig cfg_;
    ZeroFn zero_;
    std::vector<Block> blocks_;
    std::vector<int> free_;                     // never published, ref 0
    std::list<int> lru_;                        // published, ref 0: front = oldest
    std::unordered_map<uint64_t, int> by_key_;  // published blocks
    std::vector<Seq> seqs_;
    uint64_t zero_calls_ = 0, prefix_hits_ = 0;

    Seq& seq(int id);
    const Seq& seq(int id) const;
    int take_block();   // -1 when none
    void unref(int b);
    static uint64_t chain_key(uint64_t parent, const int* tokens, int n);
};

// [KV] Fixed-size recurrent-state slots (GDN / SSM / conv state) with snapshots for speculation
// rollback. A slot is zeroed when acquired (never handed out with another request's state).
class StateSlots {
public:
    using ZeroFn = std::function<void(int slot)>;
    using CopyFn = std::function<void(int from_slot, int to_slot)>;
    // n_slots live slots plus n_snapshots snapshot slots, all of slot_bytes each.
    StateSlots(int n_slots, int n_snapshots, uint64_t slot_bytes, ZeroFn zero, CopyFn copy);
    int acquire();                 // -1 when all are taken; zeroes the slot
    void release(int slot);
    int snapshot(int slot);        // copies the slot into a free snapshot; -1 when none free
    void rollback(int slot, int snap);   // copies the snapshot back
    void drop_snapshot(int snap);
    int in_use() const;
    uint64_t bytes_total() const { return uint64_t(n_slots_ + n_snaps_) * slot_bytes_; }

private:
    int n_slots_, n_snaps_;
    uint64_t slot_bytes_;
    ZeroFn zero_;
    CopyFn copy_;
    std::vector<bool> used_, snap_used_;
};

}  // namespace onebit::v2
