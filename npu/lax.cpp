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
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

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
    std::deque<xrt::run> runs;  // the runlist refers to these; a deque keeps them in place
    std::unique_ptr<xrt::runlist> rl;
    std::optional<xrt::run> ln, lm;
    const uint8_t* embed = nullptr;
    LoadStats stats;

    Bo bo(size_t bytes, bool zeroed = true) {
        auto b = std::make_unique<xrt::ext::bo>(dev, padup(bytes));
        if (zeroed) {
            std::memset(b->map(), 0, padup(bytes));
            b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        return b;
    }

    Impl(const Model& m, const Config& c, const std::string& dir, int threads) : model(m), cfg(c) {
        const auto t0 = clk::now();
        ks = classic_kernels(dev, dir);
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
        rl = std::make_unique<xrt::runlist>(ks->layer_context());
        const int a = ks->first_buffer_arg();
        for (int L = 0; L < c.layers; ++L) {
            const bool lin = c.kinds[size_t(L)] == Kind::Linear;
            xrt::run& r = runs.emplace_back(ks->make_run(lin ? Stage::LayerLinear : Stage::LayerFull));
            const size_t l = size_t(L);
            r.set_arg(a + 0, *pool[l]);
            r.set_arg(a + 1, *xres);
            r.set_arg(a + 2, *consts[l]);
            r.set_arg(a + 3, lin ? *dkv : *cache[l]);
            r.set_arg(a + 4, *act[l]);
            r.set_arg(a + 5, *ptab);
            r.set_arg(a + 6, lin ? *cache[l] : *dstate);
            r.set_arg(a + 7, *lcfg[l]);
            rl->add(r);
        }
        ln = ks->make_run(Stage::Norm);
        for (int i = 0; auto* b : {&xres, &zero, &normw, &xresf, &hn}) ln->set_arg(a + i++, **b);
        lm = ks->make_run(Stage::Head);
        for (int i = 0; auto* b : {&lmpool, &hn, &logits}) lm->set_arg(a + i++, **b);
        stats.total_ms = ms_since(t0);
    }

    void write_xres() { xres->sync(XCL_BO_SYNC_BO_TO_DEVICE, kHidden * 4, 0); }

    void wait(xrt::run& r, const char* what) {
        r.start();
        if (r.wait(kTimeout) != ERT_CMD_STATE_COMPLETED) throw std::runtime_error(std::string(what) + " run did not complete");
    }

    void run(size_t pos, bool head) {
        if (pos >= size_t(kMaxContext)) throw std::runtime_error("position " + std::to_string(pos) + " past the context");
        ks->set_position(pos);
        rl->execute();
        if (rl->wait(kTimeout) == std::cv_status::timeout) throw std::runtime_error("layer runlist timed out");
        if (head) {
            wait(*ln, "norm");
            wait(*lm, "lm head");
        }
    }

    const float* read_logits() {
        logits->sync(XCL_BO_SYNC_BO_FROM_DEVICE, kLogitsBytes, 0);
        return static_cast<const float*>(logits->map());
    }
};

Decoder::Decoder(const Model& model, const Config& config, const std::string& kernel_dir, int threads)
    : p_(std::make_unique<Impl>(model, config, kernel_dir, threads)) {}
Decoder::~Decoder() = default;

const LoadStats& Decoder::load_stats() const { return p_->stats; }
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

}  // namespace onebit::npu::lax
