<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# NPU plan layer, P4: extracting `NpuTarget` + `NpuModel` (engine#803)

Part of engine#799. This is the design for P4: split `npu/forward/src/q4nx_forward.cpp` along the line
between *the model's math* (portable) and *the NPU's runtime + kernels* (per target), so a second target
(Hexagon/P5) can be written against an interface instead of a fork of a 4.5 k-line file.

Nothing here changes numerics. The gate is unchanged outputs and an unchanged profile table.

## 1. What the line is, today

`Q4nxNpuForward` is one class holding four unrelated things:

| # | Cluster | Today | Wants to be |
|---|---|---|---|
| 1 | container + config + semantics | `q4nx.h` reads, `init_impl()` parsing, `Q4nxSemantics` | `NpuModel` setup (shared) |
| 2 | **design derivation** from filenames | `q4nx_derive_designs` + `file_exists()` probes in `init_impl()` | `npu/plan/` (P1) — a target must refuse, not probe |
| 3 | **device + launch** | `xrt::device`, `I8Ctx ctx_[]/ctx16_[]/ctx_head_`, `ctx_for()`, `design_index()`, `npu_gemm()`, `npu_gemm_rows()`, weight BO cache `wcache_`, the M=16 bit-equality gate, `NPU_PROFILE` accumulators, pb16i sidecar loads, Hadamard rotation of the activation | `NpuTarget` (`npu/target/xdna/` for XDNA) |
| 4 | **model math** | embedding, norms, rope, attention, GDN recurrence, MLA/MoE, FFN, lm head, snapshot/restore, sampling-facing logits | `NpuModel` (`npu/model/`) |

Cluster 3 is the one that makes a second NPU expensive, and it is also the one where a mistake costs
correctness silently, so it moves first.

## 2. The interface (`npu/target/target.h`)

```cpp
namespace npu::plan { struct Plan; }          // P1 owns it; a target only reads it

enum class Op { gemv, gdn_step, attn_step, lm_head };

struct GemvArgs {          // one projection GEMM, the only op every target must have
    int      role;         // index into the design/plan table (today: Q4nxDesign role index)
    int      layer;        // layer in flight, for the weight residency key
    int      M, K, N;      // rows, K, N
    const float*   x;      // activations, host, M x K row-major
    const float*   Bmat;   // f32 weights [K][N]; nullptr when the weight is already resident
    int      wkey;         // residency key for those weights (-1 = not cacheable)
    const char* tag;       // pb16i sidecar tag (nullptr = the target has no image for this op)
};

struct Caps {
    int  max_rows      = 16;    // rows per decode launch
    int  prefill_rows  = 128;   // rows per prefill launch
    bool t2_expansion  = false;
    bool gdn           = false;
    bool attn          = false;
    bool lm_head       = false;
    bool int8_kv       = false;
};

class NpuTarget {
public:
    virtual ~NpuTarget() = default;
    virtual const char* name() const = 0;
    // Reads the plan, loads the kernel manifest, refuses (named reason in err) anything it cannot run.
    virtual bool open(const npu::plan::Plan& plan, const std::string& kernels_dir, std::string& err) = 0;
    // Weights resident / mapped for [first_layer, first_layer+n_layers).
    virtual bool load_block(int first_layer, int n_layers, std::string& err) = 0;
    // The one mandatory op. Returns false with err set; the caller (NpuModel) then uses its host fallback.
    virtual bool gemv(const GemvArgs& a, std::vector<float>& out, std::string& err) = 0;
    // Optional ops: a target that leaves these alone is still usable, NpuModel falls back to the host.
    virtual bool gdn_step(int layer, int M, const float* x, float* y) { return false; }
    virtual bool attn_step(int layer, int M, const float* q, float* o) { return false; }
    virtual bool lm_head(const float* h, float* logits) { return false; }
    virtual bool supports(Op op) const { return op == Op::gemv; }
    virtual Caps caps() const = 0;
};
```

`gemv` takes float activations today because that is the byte-identical path being moved: the activation
quantisation, the amax scale, the Hadamard fold and the int32 result's dequantisation all live inside the
current `npu_gemm()`. Moving the quantiser to the model side (so the target sees `int8_t* + scale`, as the
plan describes) is a **second, separate step**, because it changes where the amax scan happens and must be
proved by the byte-exact gate rather than assumed. It is listed in §5.

## 3. What moves, and in what order

**A1 (this branch, compiles, no behaviour change)** — the launch cluster:

- `npu/target/target.h`: the interface above.
- `npu/target/xdna/xdna_target.{h,cpp}`: `NpuTargetXdna` owns `ctx_[]`, `ctx16_[]`, `ctx_head_`,
  `ctx_for()`, `design_index()`, `wcache_` (packed weight BOs), the M=16 bit-equality gate
  (`m16_check_gemm*`), the `NPU_PROFILE` accumulators, and the bodies of `npu_gemm()` /
  `npu_gemm_rows()` moved verbatim, including the pb16i path and the Hadamard rotation.
- `Q4nxNpuForward` keeps its public API and its math; `npu_gemm()`/`npu_gemm_rows()` become thin
  delegations that (a) keep the host-GEMM fallback for `init_host()`, and (b) hand the target an `int`
  role instead of an `I8Ctx&`. The ~50 call sites change from `npu_gemm(*ctx_[i_x], …)` to
  `npu_gemm(i_x, …)` — a textual change, so the numerics cannot move with it.
- The **forge q4_1 gemv family** (#885, landed after this branch started) is ported the same way, with one
  boundary that had to be chosen rather than slid: its weight image is built from the *container's* chunks,
  which is model state, while the launch is the target's. So `GemvArgs` carries a `tiles` callback
  (`bool (*)(void*, int layer, int role, std::vector<uint8_t>&, std::string&)`) that the target calls **only
  on a weight-residency miss**, and the model builds the tile stream with the layer it was given instead of
  reading its own `cur_layer_`. `role_from_tag()` is deleted rather than moved: `GemvArgs::role` already is
  the design index that mapping produced. The head's tile stream is container state too, so
  `run_lm_head_gemv()` stays in the model and asks the target for its context (`head_ctx()`), the same
  accessor the int8 head already used.
- Context creation stays in `init_impl()` for now, but goes through the target:
  `t_->adopt_ctx(role, std::move(ctx))`, `adopt_ctx16()`, `adopt_head()`. That 1:1 seam is deliberate: it
  isolates the *declaration* of the interface (this step) from the *derivation* rewrite (A2/P1), which is
  the part with the filename-probe trap in it.

**A2 (P1-dependent)** — the setup side:

- `engine-src`/`init_impl()`'s design derivation (`q4nx_derive_designs` + `file_exists()` probes) is
  replaced by `plan.json`; the target's `open(plan, kernels_dir, err)` does the ELF/manifest loading and
  the refusals, and `adopt_ctx*` disappears.
- `load_block(first_layer, n_layers, err)` takes over the weight residency role of `load_layer()`'s
  packing half (`wcache_` fill + pb16i image load), so a target that maps weights zero-copy can do it.

**A3** — `NpuModel`:

- The math + its state (`c_`, `sem_`, `embed_*`, `in_norm_`…, KV, GDN state, `mla_*`, `moe_*`) becomes
  `npu/model/npu_model.{h,cpp}` with `NpuTarget&` injected; `Q4nxNpuForward` becomes a thin adapter that
  owns a model + a target (the serve path keeps calling the same methods, so `app/forward_serve.cpp` and
  `npu/mtp/draft.h` do not change).
- Optional ops get their host fallbacks here: `gdn_step`, `attn_step`, `lm_head` return false from a target
  that does not implement them, and the model runs its own implementation (today's code, unchanged).

A3 is the step with the large diff and no numerical content; it goes after the gate proves A1+A2.

## 4. Gate (unchanged from #803)

1. `window11`: 3 greedy prompts, text identical to the current binary.
2. Logits KLD **0** on 8 prompts (the same prompts the hw-gate uses), same model dir and kernel set.
3. `NPU_PROFILE=1` profile table: every row within 2 % of the pre-split binary, same row count.
4. `ctest` green; `--npu-target xdna` is the default and an unknown target name is refused by name.
   The flag is `forward-serve`'s (the per-op route's own command) and `serve` carries it through to that child,
   so the list of compiled targets and the refusal live in one place. The refusal is checked **before the
   device is opened**, which is what makes the row measurable without a ticket: `--npu-target hexagon` prints
   `--npu-target hexagon: this build has no such target (compiled: xdna)` and exits, and the same message comes
   out of `serve --device npu --npu-target hexagon` because it reaches the forward unchanged.

Runs go through `npu-hold` tickets; a new ELF is not involved (no kernel changes), so tickets are the normal
kind. The reference binary is the `c20`-era build with the same model dir, so a difference can only be the
split.

## 5. Deliberately not in A1

- The int8-activation contract for `gemv` (see §2). Separate step, byte-exact gate.
- `plan.json` consumption, and therefore the `open()` refusals. A1's `open()` exists but derives nothing;
  it returns true when the model dir has the shapes A1 was given, and A2 replaces that with the plan.
- Any Hexagon code. P5 is written against this header; if the header is wrong, P5 says so first.
- Any kernel, weight layout or numerics change. A1 moves code; it does not read it.
