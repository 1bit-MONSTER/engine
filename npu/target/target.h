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
// npu/target/target.h — the NPU target interface (engine#799, phase P4 / #803).
//
// One model description, many NPUs.  The model (npu/model/) owns the maths and the state; a target owns
// the runtime, the kernels and the weight residency.  A target is usable the moment it can run `gemv`;
// every other op is optional and the model falls back to its own host implementation when
// `supports(op)` is false.
//
// This header is deliberately free of any XDNA, XRT or Hexagon detail: a target's own headers
// (npu/target/xdna/xdna_target.h) hold those.  The plan it reads is npu/plan/ (phase P1); until that
// lands, `open()` is handed the same shape information the forward derives today and the plan struct is
// only forward-declared here.
#ifndef ONEBIT_NPU_TARGET_TARGET_H
#define ONEBIT_NPU_TARGET_TARGET_H

#include <cstdint>   // uint8_t: TileFn's tile buffer.  Listed rather than relied on from an includer.
#include <string>
#include <vector>

namespace onebit::npu::plan {
struct Plan;   // npu/plan/plan.h (engine #799 P1): the description a target reads and refuses by name
}  // namespace onebit::npu::plan

namespace npu::target {

// The ops a target may implement.  `gemv` is mandatory; the rest are optional and fall back to the host.
enum class Op {
    gemv = 0,
    gdn_step,     // the GDN delta-rule step (per layer)
    attn_step,    // the attention step (per layer)
    lm_head,      // the final projection
};

// One projection GEMM: y = x . B, where B is either supplied as f32 host weights (the container's
// dequantised form, `Bmat`) or already resident in the target under `wkey`.
//
// `x` is float today because that is the byte-identical path the XDNA launch layer was moved out of
// (the amax scan, the Hadamard fold and the int32 result's dequantisation all live with the launch).
// The plan's int8-activation contract (x + xscale) is a separate, byte-exact-gated step — see
// docs/npu-p4-target-split.md §5.
struct GemvArgs {
    int         role = 0;      // index into the target's role/design table
    int         layer = -1;    // layer in flight, for weight residency and profiling
    int         M = 1;         // rows: 1..max_rows for decode, prefill_rows+ for a prefill chunk
    int         K = 0;         // contracted dim
    int         N = 0;         // output width
    const float* x = nullptr;  // activations, host memory, M x K row-major
    // `a_k` is the activation's REAL width per row (the design's real_k, i.e. the model's hidden or
    // intermediate size).  It is K whenever the model is already on the container's 256-column K grid,
    // and smaller when it is not (Qwen2.5-0.5B: 896 of 1024), in which case the activation is
    // zero-padded to K before the kernel sees it -- see q4nx_pad_a_f32() for why that is exact.
    int         a_k = 0;
    const float* Bmat = nullptr;   // f32 weights [K][N]; nullptr when the weight is resident
    int         wkey = -1;     // residency key for those weights (-1 = not cacheable, re-pack per call)
    const char* tag = nullptr; // the projection's container tag (log/debug identity)
    // The caller asked for the batched (M>1) launch body.  M==1 with rows=true still runs that body:
    // the two bodies differ in how they take per-row activation scales, so which one runs is the
    // caller's choice and not a function of M.
    bool        rows = false;
    // Prefer this role's narrow (decode-twin) kernel even for a wider launch.  The pb16i prefill path
    // uses this when a projection has no wide kernel at all.
    bool        decode_twin = false;
    // The gemv family (the OGA pack's q4_1 GEMV ELFs) never reads f32 weights: it dequantises from the
    // container's own 5,120-byte q4_1 tiles, so its weight image is container state and the model builds
    // it.  The target asks for it on a residency miss only; a null `tiles` means the model has no tile
    // stream, which is a refusal by name and not a silent repack of Bmat.
    using TileFn = bool (*)(void* self, int layer, int role, std::vector<uint8_t>& out, std::string& err);
    TileFn      tiles = nullptr;
    void*       tiles_self = nullptr;
};

struct Caps {
    int  max_rows = 16;        // rows per decode launch
    int  prefill_rows = 128;   // rows per prefill launch
    bool t2_expansion = false; // 2-bit ternary codes expanded on the core
    bool gdn = false;
    bool attn = false;
    bool lm_head = false;
    bool int8_kv = false;
};

class NpuTarget {
public:
    virtual ~NpuTarget() = default;

    // Stable name for logs and for `--npu-target <name>`.
    virtual const char* name() const = 0;

    // Reads the plan, loads the kernel manifest and the signed-ELF set.  A target MUST refuse anything
    // it cannot run, with a named reason in `err`, rather than silently running a different geometry.
    // "Disagree with the plan" is a refusal, not a fallback: the plan is what the model directory says
    // the kernels are, and a target that ignores it is the filename-probe trap with extra steps.
    // `family` is the artifact-name suffix of the container's weight family ("", "_b8", "_pb16i",
    // "_t2s", "_t2s2"), which is the forward's to derive while the container parsing lives there (A1).
    // It is an argument rather than something the target recomputes, because a second copy of that rule
    // is how a plan and the images it names drift apart silently.
    virtual bool open(const onebit::npu::plan::Plan& plan, const std::string& kernels_dir,
                      const char* family, std::string& err) = 0;

    // Make the weights of [first_layer, first_layer + n_layers) resident (mapped or uploaded).  A target
    // that has no block residency returns true and does nothing; `gemv` then packs per call.
    virtual bool load_block(int first_layer, int n_layers, std::string& err) = 0;

    // The one mandatory op.  Returns false with `err` set; the caller then uses its host fallback.
    virtual bool gemv(const GemvArgs& a, std::vector<float>& out, std::string& err) = 0;

    // Optional ops.  A target that leaves one alone returns false and `supports()` stays false for it.
    //
    // CONTRACT, and why it is written out (engine#799 rule (b)): an op that a plan may name must be able to
    // ship a host reference, and a reference can only exist if the signature carries its operands.  These
    // three spell out what each one is, so the reference has something to be checked against:
    //
    //   gdn_step(layer, M, x, y)   the GDN delta-rule step for ONE layer over M rows.  `x` is [M][H] float
    //                              row-major input to the layer; `y` is [M][H] the layer's output.  The
    //                              layer's parameters (conv kernel, wa/wb, ssm_a, dt bias, norm) and its
    //                              recurrence state come from the layer that `load_block()` made resident,
    //                              NOT from the arguments: a target that holds them on the device runs the
    //                              step there, and a host reference for it must take the same parameters
    //                              explicitly.  Not idempotent -- the step advances the layer's state.
    //   attn_step(layer, M, q, o)  the attention step for ONE layer over M rows: `q` is [M][NH*HD] float
    //                              row-major, `o` is [M][NH*HD].  Keys/values come from the layer's own
    //                              cache state (built by the forward, resident per load_block()), never from
    //                              the arguments -- which is the same reason a reference needs them passed.
    //   lm_head(h, logits)         one hidden row [H] to the vocabulary vector [V]; the weights are the
    //                              model's, resident, and this one IS idempotent (no state).
    //
    // The first two are the reason rule (b) is two slices: a reference cannot be written against an
    // argument list that omits the state and parameters the op consumes.  Specifying them here is the
    // first step; `npu/plan/op_registry.cpp` registers them once their references and known answers exist.
    virtual bool gdn_step(int /*layer*/, int /*M*/, const float* /*x*/, float* /*y*/) { return false; }
    virtual bool attn_step(int /*layer*/, int /*M*/, const float* /*q*/, float* /*o*/) { return false; }
    virtual bool lm_head(const float* /*h*/, float* /*logits*/) { return false; }

    virtual bool supports(Op op) const { return op == Op::gemv; }
    virtual Caps caps() const = 0;
};

// The plan describes what the model directory asks for; a target's caps describe what it can actually run.
// A plan that asks for something the caps do not cover is a refusal BY NAME, never a silent downgrade --
// for the KV cache the downgrade would be a model that answers the same text out of a different contract,
// which is the failure this whole layer exists to remove.
//
// One function rather than a check inside each target: the xdna target is not the last one (P5 adds
// Hexagon), and a rule that has to be re-remembered per target is a rule that will be missing from one.
// Takes the plan's two dtype strings rather than the Plan so this header stays free of nlohmann/json.
inline bool check_kv_caps(const std::string& key, const std::string& value, const Caps& caps, std::string& err) {
    if (key != "int8" && value != "int8") return true;
    if (caps.int8_kv) return true;
    err = "plan asks for an int8 KV cache (kv.key=\"" + key + "\", kv.value=\"" + value +
          "\") but this target has no int8 KV path (caps().int8_kv is false): refusing rather than "
          "storing f16 in its place";
    return false;
}

}  // namespace npu::target

#endif  // ONEBIT_NPU_TARGET_TARGET_H
