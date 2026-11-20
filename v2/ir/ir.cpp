#include "ir.h"

#include "../loader/formats.h"
#include "../loader/gguf.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

namespace onebit::v2 {

const char* op_name(Op op) {
    switch (op) {
        case Op::EmbedNorm: return "EmbedNorm";
        case Op::QkvRopeKvWrite: return "QkvRopeKvWrite";
        case Op::Attention: return "Attention";
        case Op::OProjAddNorm: return "OProjAddNorm";
        case Op::GateUpSwiGLU: return "GateUpSwiGLU";
        case Op::DownAddNorm: return "DownAddNorm";
        case Op::LmHead: return "LmHead";
        case Op::Sample: return "Sample";
    }
    return "?";
}

const WeightRef* Node::weight(const std::string& role) const {
    for (const auto& x : w)
        if (x.role == role) return &x;
    return nullptr;
}

std::string Node::formats() const {
    std::set<std::string> s;
    for (const auto& x : w) s.insert(fmt_name(x.fmt));
    if (s.empty()) return "-";
    std::string out;
    for (const auto& f : s) out += (out.empty() ? "" : "+") + f;
    return out;
}

std::string Node::label() const {
    return (layer >= 0 ? "L" + std::to_string(layer) + "." : std::string()) + op_name(op);
}

int Graph::add_value(std::string name, DType dt, int64_t cols) {
    Value v;
    v.id = int(values.size());
    v.name = std::move(name);
    v.dtype = dt;
    v.cols = cols;
    values.push_back(std::move(v));
    return values.back().id;
}

void Graph::validate() const {
    std::vector<int> written(values.size(), -1);
    if (tokens_in < 0 || size_t(tokens_in) >= values.size()) throw std::runtime_error("IR: no token input");
    written[size_t(tokens_in)] = -2;  // graph input
    static const int arity_in[kNumOps] = {1, 1, 1, 2, 1, 2, 1, 1};
    static const int arity_out[kNumOps] = {2, 1, 1, 2, 1, 2, 1, 1};
    for (size_t i = 0; i < nodes.size(); ++i) {
        const Node& n = nodes[i];
        const int o = int(n.op);
        if (int(n.in.size()) != arity_in[o] || int(n.out.size()) != arity_out[o])
            throw std::runtime_error("IR: " + n.label() + " has the wrong number of inputs or outputs");
        for (int v : n.in) {
            if (v < 0 || size_t(v) >= values.size()) throw std::runtime_error("IR: " + n.label() + " reads an unknown value");
            if (written[size_t(v)] == -1) throw std::runtime_error("IR: " + n.label() + " reads " + values[size_t(v)].name + " before it is written");
        }
        for (int v : n.out) {
            if (v < 0 || size_t(v) >= values.size()) throw std::runtime_error("IR: " + n.label() + " writes an unknown value");
            if (written[size_t(v)] != -1) throw std::runtime_error("IR: " + values[size_t(v)].name + " is written twice");
            written[size_t(v)] = int(i);
        }
        for (const auto& w : n.w)
            if (!fmt_info(w.fmt)) throw std::runtime_error("IR: " + n.label() + " weight " + w.tensor + " has unknown format " + std::to_string(w.fmt));
    }
    if (logits_out < 0 || written[size_t(logits_out)] < 0) throw std::runtime_error("IR: logits are never written");
    if (token_out < 0 || written[size_t(token_out)] < 0) throw std::runtime_error("IR: the sampled token is never written");
}

uint64_t Graph::weight_bytes() const {
    std::set<std::string> seen;   // a tied embedding is read by two nodes, stored once
    uint64_t n = 0;
    for (const auto& node : nodes)
        for (const auto& w : node.w) {
            if (!seen.insert(w.tensor).second) continue;
            uint64_t e = 1;
            for (uint64_t d : w.ne) e *= d;
            n += fmt_bytes(w.fmt, e);
        }
    return n;
}

uint64_t Graph::peak_live_bytes_per_token() const {
    std::vector<int> last_read(values.size(), -1);
    for (size_t i = 0; i < nodes.size(); ++i)
        for (int v : nodes[i].in) last_read[size_t(v)] = int(i);
    uint64_t live = values[size_t(tokens_in)].bytes_per_token(), peak = live;
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (int v : nodes[i].out) live += values[size_t(v)].bytes_per_token();
        peak = std::max(peak, live);
        for (int v : nodes[i].in)
            if (last_read[size_t(v)] == int(i)) live -= values[size_t(v)].bytes_per_token();
        for (int v : nodes[i].out)   // written and never read (the last residual)
            if (last_read[size_t(v)] < 0 && v != logits_out && v != token_out) live -= values[size_t(v)].bytes_per_token();
    }
    return peak;
}

std::string Graph::dump() const {
    std::ostringstream o;
    o << "graph " << arch << ": " << nodes.size() << " nodes, " << values.size() << " values, weights "
      << weight_bytes() << " bytes\n";
    for (const auto& n : nodes) {
        o << "  " << n.label() << " (";
        for (size_t i = 0; i < n.in.size(); ++i) o << (i ? ", " : "") << values[size_t(n.in[i])].name;
        o << ") -> (";
        for (size_t i = 0; i < n.out.size(); ++i) o << (i ? ", " : "") << values[size_t(n.out[i])].name;
        o << ")";
        for (const auto& w : n.w) o << " " << w.role << "=" << fmt_name(w.fmt);
        o << "\n";
    }
    return o.str();
}

std::vector<std::string> known_architectures() { return {"qwen3"}; }

Graph build_graph(const GgufFile& g) {
    const HParams hp = read_hparams(g);
    Graph graph;
    if (hp.arch == "qwen3") graph = build_qwen3(g, hp);
    else throw std::runtime_error("architecture '" + hp.arch + "' has no V2 description yet (V2 runs: qwen3; more in phase P3)");
    graph.validate();
    return graph;
}

}  // namespace onebit::v2
