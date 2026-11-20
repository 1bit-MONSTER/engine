#pragma once

// [IR] V2's model graph at fused-layer granularity (design 2.2 [IR], D1, D4). A node is one fused
// layer op with explicit weight formats, not ggml's elementwise graph. The node set is the fusion
// pattern V1's private kernels already encode:
//
//   EmbedNorm        token rows -> residual x, and h = rmsnorm(x) with layer 0's attn_norm
//   QkvRopeKvWrite   q = rope(qnorm(h Wq)), k = rope(knorm(h Wk)), v = h Wv; K/V written to the
//                    paged KV cache at the tokens' positions; outputs q
//   Attention        causal GQA over the sequence's KV blocks (length-masked), outputs per head
//   OProjAddNorm     x += attn Wo; h = rmsnorm(x) with ffn_norm
//   GateUpSwiGLU     act = silu(h Wgate) * (h Wup)
//   DownAddNorm      x += act Wdown; h = rmsnorm(x) with the next layer's attn_norm, or
//                    output_norm after the last layer
//   LmHead           logits = h Wout (Wout = token_embd when the embeddings are tied)
//   Sample           logits -> next token (the sampler; on the device in later slices)
//
// Every activation is per token: a Value is [T x cols] for the T tokens of a step, so one graph
// serves every shape bucket. KV lives in the KV manager, not in a Value.

#include "hparams.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace onebit::v2 {

class GgufFile;

enum class Op : uint8_t { EmbedNorm, QkvRopeKvWrite, Attention, OProjAddNorm, GateUpSwiGLU, DownAddNorm, LmHead, Sample };
const char* op_name(Op op);
constexpr int kNumOps = 8;

enum class DType : uint8_t { F32, I32 };

struct Value {
    int id = -1;
    std::string name;
    DType dtype = DType::F32;
    int64_t cols = 0;     // elements per token
    uint64_t bytes_per_token() const { return uint64_t(cols) * 4; }
};

struct WeightRef {
    std::string role;     // "wq", "attn_norm", ...
    std::string tensor;   // GGUF tensor name
    uint32_t fmt = 0;
    std::vector<uint64_t> ne;
};

struct Node {
    Op op = Op::EmbedNorm;
    int layer = -1;                   // -1 outside the layer stack
    std::vector<int> in, out;         // Value ids; for in-place residual updates the residual is
                                      // both read (in) and a new SSA value written (out)
    std::vector<WeightRef> w;
    std::map<std::string, double> attr;
    const WeightRef* weight(const std::string& role) const;
    std::string formats() const;      // sorted unique weight formats, "Q4_K+Q6_K"; "-" if none
    std::string label() const;        // "L3.QkvRopeKvWrite"
};

struct Graph {
    std::string arch;
    HParams hp;
    std::vector<Value> values;
    std::vector<Node> nodes;
    int tokens_in = -1;     // I32 token ids
    int logits_out = -1;
    int token_out = -1;

    int add_value(std::string name, DType dt, int64_t cols);
    // Checks SSA form (each value written once, read after it is written), node arity, and that
    // every weight's shape matches the hyperparameters. Throws std::runtime_error.
    void validate() const;
    uint64_t weight_bytes() const;
    // Peak bytes of live activations per token over a topological walk (values die after
    // their last reader), the per-token part of a plan's scratch.
    uint64_t peak_live_bytes_per_token() const;
    std::string dump() const;
};

// The architecture library: the IR for a GGUF's general.architecture. Throws for an architecture
// with no V2 description yet.
Graph build_graph(const GgufFile& g);
// Architectures build_graph knows.
std::vector<std::string> known_architectures();

// One architecture's builder (ir/arch_*.cpp).
Graph build_qwen3(const GgufFile& g, const HParams& hp);

}  // namespace onebit::v2
