// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The program this drives is the one open_kernels' model/lax_decode_cfg.py (setup_packed,
// position) writes for its XRT harness (third_party/OpenFlowLM-Next, MIT): the same
// buffers, argument order and per-token steps, re-implemented against XRT directly.
#include "lax.h"

#include "lax_kernels.h"
#include "lax_stream.h"

#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace onebit::npu::lax {

namespace {

using clk = std::chrono::steady_clock;
double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

constexpr size_t kBoAlign = 1u << 20;  // XDNA buffers in whole MiB, as the harness sizes them
size_t padup(size_t n) { return (n + kBoAlign - 1) / kBoAlign * kBoAlign; }
constexpr auto kTimeout = std::chrono::milliseconds(60000);

std::vector<uint8_t> slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<uint8_t> v(size_t(f.tellg()));
    f.seekg(0);
    if (!v.empty() && !f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size())))
        throw std::runtime_error("short read " + path);
    return v;
}

// ---- the classic kernel set --------------------------------------------------------------

class Classic final : public KernelSet {
    struct Stream {
        xrt::kernel kernel;
        xrt::bo instr;
        size_t nwords = 0;
        uint32_t* words() { return instr.map<uint32_t*>(); }
    };

public:
    Classic(xrt::device& dev, const std::string& dir) : dev_(dev), dir_(dir) {
        // Both control texts come from one design build, so both run on one context and one
        // runlist: lax_l's xclbin serves lax_a's stream too, as the reference driver does
        // (lax_a's own xclbin differs only in its UUID and metadata).
        layer_ctx_ = context(dir + "/lax_l/final.xclbin");
        ln_ctx_ = context(dir + "/ln/final.xclbin");
        lm_ctx_ = context(dir + "/lm_head_q8/final.xclbin");
        lin_ = stream(layer_ctx_, dir + "/lax_l/insts.bin");
        full_ = stream(layer_ctx_, dir + "/lax_a/insts.bin");
        ln_ = stream(ln_ctx_, dir + "/ln/insts.bin");
        lm_ = stream(lm_ctx_, dir + "/lm_head_q8/insts.bin");
        patches_ = position_patches({full_.words(), full_.nwords}, kKvRow);
    }

    xrt::hw_context& layer_context() override { return layer_ctx_; }

    xrt::run make_run(Stage s) override {
        Stream& st = s == Stage::LayerLinear ? lin_ : s == Stage::LayerFull ? full_ : s == Stage::Norm ? ln_ : lm_;
        xrt::run r(st.kernel);
        r.set_arg(0, 3);  // opcode: run the instruction stream
        r.set_arg(1, st.instr);
        r.set_arg(2, int(st.nwords));
        return r;
    }

    int first_buffer_arg() const override { return 3; }

    void set_position(size_t pos) override {
        apply_position({full_.words(), full_.nwords}, patches_, pos, kKvRow, kPtabRow);
        full_.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    bool position_in_run() const override { return false; }

    std::optional<xrt::run> head_run() override { return std::nullopt; }

    std::string describe() const override { return "classic xclbin + insts.bin from " + dir_; }

private:
    xrt::hw_context context(const std::string& path) {
        const xrt::xclbin x(path);
        return xrt::hw_context(dev_, dev_.register_xclbin(x));
    }

    Stream stream(xrt::hw_context& ctx, const std::string& path) {
        const auto bytes = slurp(path);
        if (bytes.empty() || bytes.size() % 4) throw std::runtime_error(path + ": not whole instruction words");
        Stream s{xrt::kernel(ctx, "MLIR_AIE"), {}, bytes.size() / 4};
        s.instr = xrt::bo(dev_, bytes.size(), xrt::bo::flags::cacheable, s.kernel.group_id(1));
        std::memcpy(s.instr.map<void*>(), bytes.data(), bytes.size());
        s.instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return s;
    }

    xrt::device& dev_;
    std::string dir_;
    xrt::hw_context layer_ctx_, ln_ctx_, lm_ctx_;
    Stream lin_, full_, ln_, lm_;
    std::vector<PosPatch> patches_;
};

}  // namespace

std::unique_ptr<KernelSet> classic_kernels(xrt::device& dev, const std::string& dir) {
    return std::make_unique<Classic>(dev, dir);
}

// ---- the decoder -----------------------------------------------------------------------

struct Decoder::Impl {
    using Bo = std::unique_ptr<xrt::ext::bo>;
    const Model& model;
    const Config cfg;
    xrt::device dev{0u};
    std::unique_ptr<KernelSet> ks;
    Bo xres, zero, normw, xresf, hn, logits, lmpool, ptab, dkv, dstate;
    std::vector<Bo> pool, consts, act, lcfg, cache;  // cache: DeltaNet state or KV, per layer kind
    // A token's runlist, for one position. With full ELFs, the next position's slot is
    // prepared while the device runs the current one (the kernel set's position_in_run);
    // classic uses slot 0 only and patches its stream in place.
    struct Slot {
        std::vector<xrt::run> runs;  // one per layer, in the runlist's order
        std::unique_ptr<xrt::runlist> rl;
        size_t pos = SIZE_MAX;
    };
    std::vector<xrt::run> linear_runs;  // position-free, shared by both slots (empty for full layers)
    Slot slots[2];
    std::optional<xrt::run> ln, lm;
    const uint8_t* embed = nullptr;
    LoadStats stats;
    RunTimes times;

    Bo bo(size_t bytes, bool zeroed = true) {
        auto b = std::make_unique<xrt::ext::bo>(dev, padup(bytes));
        if (zeroed) {
            std::memset(b->map(), 0, padup(bytes));
            b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        return b;
    }

    Impl(const Model& m, const Config& c, const std::string& dir, Transport t, int threads) : model(m), cfg(c) {
        const auto t0 = clk::now();
        ks = make_kernels(dev, dir, t);
        const Tensor& et = m.tensor("model.embed_tokens.weight");
        if (et.dtype != "BF16" || et.bytes != size_t(c.vocab) * kHidden * 2) throw std::runtime_error("unexpected embedding table");
        embed = et.data;

        // Buffers. The packed ones are written whole by their packer (no zero pass).
        const auto t1 = clk::now();
        xres = bo(kHidden * 4);
        zero = bo(kHidden * 4);
        normw = bo(kHidden * 2);
        xresf = bo(kHidden * 4);
        hn = bo(kHidden * 2);
        logits = bo(kLogitsBytes);
        dkv = bo(kKvBytes);
        dstate = bo(kStateBytes);
        lmpool = bo(kLmPoolBytes, false);
        ptab = bo(kPtabBytes, false);
        for (int L = 0; L < c.layers; ++L) {
            const Kind k = c.kinds[size_t(L)];
            pool.push_back(bo(kPoolBytes, false));
            consts.push_back(bo(kConstsBytes, false));
            act.push_back(bo(kActBytes));
            lcfg.push_back(bo(kCfgBytes));
            cache.push_back(bo(k == Kind::Linear ? kStateBytes : kKvBytes));
        }
        stats.buffers_ms = ms_since(t1);

        // Pack in parallel, each job straight into its buffer's host mapping.
        const auto t2 = clk::now();
        std::vector<std::function<void()>> jobs;
        auto sync = [](xrt::ext::bo& b) { b.sync(XCL_BO_SYNC_BO_TO_DEVICE); };
        jobs.emplace_back([&] {
            pack_lmhead(model, static_cast<uint8_t*>(lmpool->map()));
            std::memset(static_cast<uint8_t*>(lmpool->map()) + kLmPoolBytes, 0, padup(kLmPoolBytes) - kLmPoolBytes);
            sync(*lmpool);
        });
        jobs.emplace_back([&] {
            pack_ptab(cfg, kMaxContext, static_cast<uint8_t*>(ptab->map()));
            sync(*ptab);
            pack_norm(model, static_cast<uint8_t*>(normw->map()));
            sync(*normw);
        });
        for (int L = 0; L < c.layers; ++L) {
            jobs.emplace_back([&, L] {
                pack_pool(model, cfg, L, static_cast<uint8_t*>(pool[size_t(L)]->map()));
                sync(*pool[size_t(L)]);
            });
            jobs.emplace_back([&, L] {
                auto* d = static_cast<uint8_t*>(consts[size_t(L)]->map());
                const size_t n = consts_bytes(cfg.kinds[size_t(L)]);
                pack_consts(model, cfg, L, d);
                std::memset(d + n, 0, padup(kConstsBytes) - n);
                sync(*consts[size_t(L)]);
            });
        }
        if (threads <= 0) threads = int(std::clamp(std::thread::hardware_concurrency(), 1u, 16u));
        std::atomic<size_t> next{0};
        std::exception_ptr err;
        std::mutex err_mu;
        std::vector<std::thread> pool_threads;
        for (int t = 0; t < threads; ++t)
            pool_threads.emplace_back([&] {
                for (size_t i; (i = next++) < jobs.size();) {
                    try {
                        jobs[i]();
                    } catch (...) {
                        std::lock_guard lk(err_mu);
                        if (!err) err = std::current_exception();
                        next = jobs.size();
                    }
                }
            });
        for (auto& t : pool_threads) t.join();
        if (err) std::rethrow_exception(err);
        stats.pack_ms = ms_since(t2);
        stats.packed_bytes = kLmPoolBytes + kPtabBytes + size_t(c.layers) * kPoolBytes;
        for (int L = 0; L < c.layers; ++L) stats.packed_bytes += consts_bytes(c.kinds[size_t(L)]);

        // Layer configs: the pool's device address and the weight-stream queues.
        for (int L = 0; L < c.layers; ++L) {
            const auto w = cfg_words(pool[size_t(L)]->address() + 0x80000000ull);
            std::memcpy(lcfg[size_t(L)]->map(), w.data(), sizeof w);
            lcfg[size_t(L)]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        // The token's runlist: 40 runs on one context, submitted once per token.
        for (int L = 0; L < c.layers; ++L)
            linear_runs.push_back(c.kinds[size_t(L)] == Kind::Linear ? layer_run(size_t(L)) : xrt::run());
        prepare(slots[0], 0);
        const int a = ks->first_buffer_arg();
        ln = ks->make_run(Stage::Norm);
        for (int i = 0; auto* b : {&xres, &zero, &normw, &xresf, &hn}) ln->set_arg(a + i++, **b);
        lm = ks->make_run(Stage::Head);
        for (int i = 0; auto* b : {&lmpool, &hn, &logits}) lm->set_arg(a + i++, **b);
        stats.total_ms = ms_since(t0);
    }

    // A new sequence: the DeltaNet state and the layers' scratch back to zero, as at load.
    // The KV cache needs nothing: a position reads only the rows written before it.
    void reset() {
        for (int L = 0; L < cfg.layers; ++L) {
            for (auto* b : {&act[size_t(L)], cfg.kinds[size_t(L)] == Kind::Linear ? &cache[size_t(L)] : nullptr}) {
                if (!b) continue;
                std::memset((*b)->map(), 0, (*b)->size());
                (*b)->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            }
        }
    }

    // The linear layers' state buffers, in layer order: what save_state/load_state copy.
    size_t state_bytes() const {
        return size_t(std::ranges::count(cfg.kinds, Kind::Linear)) * kStateBytes;
    }

    void save_state(uint8_t* out) {
        for (int L = 0; L < cfg.layers; ++L) {
            if (cfg.kinds[size_t(L)] != Kind::Linear) continue;
            auto& b = *cache[size_t(L)];
            b.sync(XCL_BO_SYNC_BO_FROM_DEVICE, kStateBytes, 0);
            std::memcpy(out, b.map(), kStateBytes);
            out += kStateBytes;
        }
    }

    void load_state(const uint8_t* in) {
        for (int L = 0; L < cfg.layers; ++L) {
            if (cfg.kinds[size_t(L)] != Kind::Linear) continue;
            auto& b = *cache[size_t(L)];
            std::memcpy(b.map(), in, kStateBytes);
            b.sync(XCL_BO_SYNC_BO_TO_DEVICE, kStateBytes, 0);
            in += kStateBytes;
        }
    }

    void write_xres() { xres->sync(XCL_BO_SYNC_BO_TO_DEVICE, kHidden * 4, 0); }

    void wait(xrt::run& r, const char* what) {
        r.start();
        if (r.wait(kTimeout) != ERT_CMD_STATE_COMPLETED) throw std::runtime_error(std::string(what) + " run did not complete");
    }

    // Layer L's run: pool xres consts kv|dkv act ptab state|dstate cfg (a linear layer gets
    // the dummy kv, a full one the dummy state).
    xrt::run layer_run(size_t l) {
        const bool lin = cfg.kinds[l] == Kind::Linear;
        xrt::run r = ks->make_run(lin ? Stage::LayerLinear : Stage::LayerFull);
        const int a = ks->first_buffer_arg();
        r.set_arg(a + 0, *pool[l]);
        r.set_arg(a + 1, *xres);
        r.set_arg(a + 2, *consts[l]);
        r.set_arg(a + 3, lin ? *dkv : *cache[l]);
        r.set_arg(a + 4, *act[l]);
        r.set_arg(a + 5, *ptab);
        r.set_arg(a + 6, lin ? *cache[l] : *dstate);
        r.set_arg(a + 7, *lcfg[l]);
        return r;
    }

    // Slot s runs position pos: the full-attention runs made for it (full ELFs), or the
    // shared stream patched (classic); the head run first when the kernel set has one.
    void prepare(Slot& s, size_t pos) {
        ks->set_position(pos);
        if (s.rl && !ks->position_in_run()) {
            s.pos = pos;
            return;
        }
        s.pos = SIZE_MAX;
        s.runs.clear();
        for (size_t l = 0; l < linear_runs.size(); ++l)
            s.runs.push_back(cfg.kinds[l] == Kind::Linear ? linear_runs[l] : layer_run(l));
        if (!s.rl) s.rl = std::make_unique<xrt::runlist>(ks->layer_context());
        s.rl->reset();
        if (auto h = ks->head_run()) s.rl->add(*h);
        for (auto& r : s.runs) s.rl->add(r);
        s.pos = pos;
    }

    void run(size_t pos, bool head) {
        if (pos >= size_t(kMaxContext)) throw std::runtime_error("position " + std::to_string(pos) + " past the context");
        const auto t0 = clk::now();
        times = {};
        const bool ahead = ks->position_in_run();
        Slot* s = &slots[0];
        if (ahead && slots[1].pos == pos) s = &slots[1];
        else if (!ahead || slots[0].pos != pos) prepare(*s, pos);
        times.prep_ms = ms_since(t0);
        const auto t1 = clk::now();
        s->rl->execute();
        if (ahead && pos + 1 < size_t(kMaxContext)) {
            // The next position, while the device runs this one; a failure here only means
            // the next run() prepares it itself.
            Slot& o = s == &slots[0] ? slots[1] : slots[0];
            const auto t = clk::now();
            try {
                prepare(o, pos + 1);
            } catch (const std::exception&) {
                o.pos = SIZE_MAX;
            }
            times.ahead_ms = ms_since(t);
        }
        if (s->rl->wait(kTimeout) == std::cv_status::timeout) throw std::runtime_error("layer runlist timed out");
        times.layers_ms = ms_since(t1);
        if (head) {
            const auto t2 = clk::now();
            wait(*ln, "norm");
            wait(*lm, "lm head");
            times.head_ms = ms_since(t2);
        }
    }

    const float* read_logits() {
        logits->sync(XCL_BO_SYNC_BO_FROM_DEVICE, kLogitsBytes, 0);
        return static_cast<const float*>(logits->map());
    }
};

Decoder::Decoder(const Model& model, const Config& config, const std::string& kernel_dir, Transport t, int threads)
    : p_(std::make_unique<Impl>(model, config, kernel_dir, t, threads)) {}
Decoder::~Decoder() = default;

const LoadStats& Decoder::load_stats() const { return p_->stats; }
const RunTimes& Decoder::last_run() const { return p_->times; }
std::string Decoder::kernels() const { return p_->ks->describe(); }

void Decoder::feed(int token) {
    if (token < 0 || token >= p_->cfg.vocab) throw std::runtime_error("token id out of range");
    const uint8_t* row = p_->embed + size_t(token) * kHidden * 2;
    auto* x = static_cast<float*>(p_->xres->map());
    for (size_t i = 0; i < kHidden; ++i) {
        const uint32_t u = uint32_t(row[2 * i] | (row[2 * i + 1] << 8)) << 16;
        std::memcpy(x + i, &u, 4);
    }
    p_->write_xres();
}

void Decoder::set_residual(const float* v) {
    std::memcpy(p_->xres->map(), v, kHidden * 4);
    p_->write_xres();
}

void Decoder::run(size_t pos, bool head) { p_->run(pos, head); }
void Decoder::reset() { p_->reset(); }
size_t Decoder::state_bytes() const { return p_->state_bytes(); }
void Decoder::save_state(uint8_t* out) { p_->save_state(out); }
void Decoder::load_state(const uint8_t* in) { p_->load_state(in); }

int Decoder::argmax(int n) {
    const float* v = p_->read_logits();
    n = std::clamp(n, 1, int(kLogitsBytes / 4));
    int best = 0;
    for (int i = 1; i < n; ++i)
        if (v[i] > v[best]) best = i;
    return best;
}

void Decoder::logits(std::vector<float>& out) {
    const float* v = p_->read_logits();
    out.assign(v, v + kLogitsBytes / 4);
}

// ---- generation ---------------------------------------------------------------------------

int Session::pick(const GenerateOptions& opt, const std::vector<int>& history) {
    if (!(opt.repetition_penalty > 1.0f)) return dec_.argmax(vocab_);
    dec_.logits(buf_);
    buf_.resize(size_t(vocab_));
    const int n = int(history.size()), from = std::max(0, n - opt.penalty_window);
    for (int i = from; i < n; ++i) {
        const int t = history[size_t(i)];
        if (t < 0 || t >= vocab_) continue;
        if (std::find(history.begin() + from, history.begin() + i, t) != history.begin() + i) continue;
        float& l = buf_[size_t(t)];
        l = l > 0.0f ? l / opt.repetition_penalty : l * opt.repetition_penalty;
    }
    return int(std::max_element(buf_.begin(), buf_.end()) - buf_.begin());  // first maximum
}

Session::Session(Decoder& dec, int vocab, SessionOptions o)
    : dec_(dec), vocab_(vocab), turn_tokens_(std::move(o.turn_tokens)), snaps_(o.snapshots) {}

GenerateResult Session::generate(const std::vector<int>& prompt, const GenerateOptions& opt) {
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if (prompt.size() + size_t(std::max(opt.max_tokens, 0)) > size_t(kMaxContext))
        throw std::runtime_error("prompt plus max_tokens exceeds the context (" + std::to_string(kMaxContext) + ")");
    GenerateResult res;
    auto t0 = clk::now();
    // The live cache, a snapshot or nothing: whichever leaves the fewest tokens to feed
    // (npu/lax_turns.h). The last prompt token runs anyway, for its logits.
    // After a failed request the device may be past cache_: then only a snapshot or a reset.
    const Plan plan = lax::plan(dirty_ ? std::span<const int>{} : std::span<const int>(cache_), snaps_, prompt);
    res.source = plan.source;
    if (plan.source == Plan::Snapshot) {
        const auto t = clk::now();
        const auto& e = snaps_.at(size_t(plan.snapshot));
        dec_.load_state(e.state.data());
        cache_ = e.tokens;
        snaps_.touch(size_t(plan.snapshot));
        res.restore_ms = ms_since(t);
    } else if (plan.source == Plan::Scratch) {
        if (dirty_ || !cache_.empty()) dec_.reset();
        cache_.clear();
    }
    res.reused = cache_.size();
    dirty_ = true;  // until this request completes: a throw may leave a token half run

    const auto points = snapshot_points(prompt, cache_.size(), turn_tokens_);
    auto snap = points.begin();
    for (size_t i = cache_.size(); i < prompt.size(); ++i) {
        if (snap != points.end() && *snap == i) {
            ++snap;
            if (!snaps_.contains(cache_)) {
                const auto t = clk::now();
                if (auto* e = snaps_.put(cache_)) {
                    e->state.resize(dec_.state_bytes());
                    dec_.save_state(e->state.data());
                    ++res.snapshots_taken;
                }
                res.snapshot_ms += ms_since(t);
            }
        }
        dec_.feed(prompt[i]);
        dec_.run(i, i + 1 == prompt.size());  // the head only where a next token is wanted
        cache_.push_back(prompt[i]);
    }
    res.prefill_ms = ms_since(t0);

    t0 = clk::now();
    std::vector<int> history = prompt;
    for (int i = 0; i < opt.max_tokens; ++i) {
        const int t = pick(opt, history);
        res.tokens.push_back(t);
        history.push_back(t);
        if (std::find(opt.stop.begin(), opt.stop.end(), t) != opt.stop.end()) {
            res.stopped_at_eos = true;
            break;
        }
        if ((opt.on_token && !opt.on_token(t)) || i + 1 == opt.max_tokens) break;
        dec_.feed(t);
        dec_.run(cache_.size(), true);
        cache_.push_back(t);
    }
    dirty_ = false;
    res.decode_ms = ms_since(t0);
    return res;
}

}  // namespace onebit::npu::lax
