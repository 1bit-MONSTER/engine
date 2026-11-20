// Qwen3 dense (general.architecture "qwen3") -> the V2 IR: pre-norm GQA attention with per-head
// RMS norm on q and k before NeoX RoPE, SwiGLU FFN, RMS norm, optionally tied embeddings.

#include "../loader/formats.h"
#include "../loader/gguf.h"
#include "ir.h"

#include <stdexcept>

namespace onebit::v2 {

namespace {

WeightRef weight(const GgufFile& g, const std::string& role, const std::string& name, std::vector<uint64_t> want) {
    const TensorInfo* t = g.tensor(name);
    if (!t) throw std::runtime_error("qwen3: missing tensor " + name);
    // GGUF may drop trailing 1-dims; compare the significant ones
    std::vector<uint64_t> ne = t->ne;
    while (ne.size() > want.size() && ne.back() == 1) ne.pop_back();
    if (ne != want) {
        std::string w;
        for (size_t i = 0; i < want.size(); ++i) w += (i ? "x" : "") + std::to_string(want[i]);
        throw std::runtime_error("qwen3: " + name + " is " + t->shape_str() + ", expected " + w);
    }
    if (!fmt_info(t->fmt)) throw std::runtime_error("qwen3: " + name + " has unknown format " + std::to_string(t->fmt));
    return WeightRef{role, name, t->fmt, t->ne};
}

}  // namespace

Graph build_qwen3(const GgufFile& g, const HParams& hp) {
    Graph G;
    G.arch = hp.arch;
    G.hp = hp;
    const uint64_t E = uint64_t(hp.n_embd), F = uint64_t(hp.n_ff), V = uint64_t(hp.n_vocab);
    const uint64_t Q = uint64_t(hp.n_head) * uint64_t(hp.head_dim_k);
    const uint64_t KV = uint64_t(hp.n_head_kv) * uint64_t(hp.head_dim_k);
    const uint64_t VV = uint64_t(hp.n_head_kv) * uint64_t(hp.head_dim_v);
    const uint64_t O = uint64_t(hp.n_head) * uint64_t(hp.head_dim_v);
    const double eps = hp.rms_eps;

    G.tokens_in = G.add_value("tokens", DType::I32, 1);
    int x = G.add_value("x.emb", DType::F32, int64_t(E));
    int h = G.add_value("h.L0.attn_in", DType::F32, int64_t(E));
    {
        Node n;
        n.op = Op::EmbedNorm;
        n.in = {G.tokens_in};
        n.out = {x, h};
        n.w = {weight(g, "token_embd", "token_embd.weight", {E, V}), weight(g, "norm", "blk.0.attn_norm.weight", {E})};
        n.attr = {{"eps", eps}};
        G.nodes.push_back(std::move(n));
    }
    for (int l = 0; l < hp.n_layer; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const std::string L = ".L" + std::to_string(l);
        const int q = G.add_value("q" + L, DType::F32, int64_t(Q));
        Node qkv;
        qkv.op = Op::QkvRopeKvWrite;
        qkv.layer = l;
        qkv.in = {h};
        qkv.out = {q};
        qkv.w = {weight(g, "wq", b + "attn_q.weight", {E, Q}), weight(g, "wk", b + "attn_k.weight", {E, KV}),
                 weight(g, "wv", b + "attn_v.weight", {E, VV}),
                 weight(g, "q_norm", b + "attn_q_norm.weight", {uint64_t(hp.head_dim_k)}),
                 weight(g, "k_norm", b + "attn_k_norm.weight", {uint64_t(hp.head_dim_k)})};
        qkv.attr = {{"eps", eps}, {"n_head", hp.n_head}, {"n_head_kv", hp.n_head_kv}, {"head_dim_k", hp.head_dim_k},
                    {"head_dim_v", hp.head_dim_v}, {"rope_base", hp.rope_base}, {"rope_neox", 1}};
        G.nodes.push_back(std::move(qkv));

        const int att = G.add_value("attn" + L, DType::F32, int64_t(O));
        Node at;
        at.op = Op::Attention;
        at.layer = l;
        at.in = {q};
        at.out = {att};
        at.attr = {{"n_head", hp.n_head}, {"n_head_kv", hp.n_head_kv}, {"head_dim_k", hp.head_dim_k},
                   {"head_dim_v", hp.head_dim_v}, {"causal", 1}};
        G.nodes.push_back(std::move(at));

        const int x1 = G.add_value("x.attn" + L, DType::F32, int64_t(E));
        const int h1 = G.add_value("h.ffn_in" + L, DType::F32, int64_t(E));
        Node op;
        op.op = Op::OProjAddNorm;
        op.layer = l;
        op.in = {att, x};
        op.out = {x1, h1};
        op.w = {weight(g, "wo", b + "attn_output.weight", {O, E}), weight(g, "norm", b + "ffn_norm.weight", {E})};
        op.attr = {{"eps", eps}};
        G.nodes.push_back(std::move(op));

        const int act = G.add_value("act" + L, DType::F32, int64_t(F));
        Node gu;
        gu.op = Op::GateUpSwiGLU;
        gu.layer = l;
        gu.in = {h1};
        gu.out = {act};
        gu.w = {weight(g, "wgate", b + "ffn_gate.weight", {E, F}), weight(g, "wup", b + "ffn_up.weight", {E, F})};
        G.nodes.push_back(std::move(gu));

        const bool last = l + 1 == hp.n_layer;
        const int x2 = G.add_value("x.ffn" + L, DType::F32, int64_t(E));
        const int h2 = G.add_value(last ? "h.final" : "h.L" + std::to_string(l + 1) + ".attn_in", DType::F32, int64_t(E));
        Node dn;
        dn.op = Op::DownAddNorm;
        dn.layer = l;
        dn.in = {act, x1};
        dn.out = {x2, h2};
        dn.w = {weight(g, "wdown", b + "ffn_down.weight", {F, E}),
                weight(g, "norm", last ? "output_norm.weight" : "blk." + std::to_string(l + 1) + ".attn_norm.weight", {E})};
        dn.attr = {{"eps", eps}};
        G.nodes.push_back(std::move(dn));
        x = x2;
        h = h2;
    }
    G.logits_out = G.add_value("logits", DType::F32, int64_t(V));
    {
        Node n;
        n.op = Op::LmHead;
        n.in = {h};
        n.out = {G.logits_out};
        n.w = {hp.tied_embeddings ? weight(g, "wout", "token_embd.weight", {E, V}) : weight(g, "wout", "output.weight", {E, V})};
        n.attr = {{"tied", hp.tied_embeddings ? 1 : 0}};
        G.nodes.push_back(std::move(n));
    }
    G.token_out = G.add_value("token", DType::I32, 1);
    {
        Node n;
        n.op = Op::Sample;
        n.in = {G.logits_out};
        n.out = {G.token_out};
        G.nodes.push_back(std::move(n));
    }
    return G;
}

}  // namespace onebit::v2
