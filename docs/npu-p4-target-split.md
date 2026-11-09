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

**A2 (P1-dependent)** — the setup side, and its state in the tree:

- the design set is **named by the plan** (`designs[]`, `Plan::designs_agree()`): when the section is
  present the engine refuses a plan that disagrees with what the container derives, naming the role and
  both sides. `tools/npu_plan.py` writes the section for the families it can state completely and writes
  **nothing** for the ones it cannot (MLA, GDN mixes, Gemma4's second attention type / PLE / double-wide
  MLP), because an incomplete section would be refused at load. **Done** (#1330, #1333).
- **the M=128 ELF set is the target's** (`XdnaTarget::open_prefill()`): it adopts each geometry, probes
  the artifact, refuses by name, and owns `init_elf`; the forward no longer creates an `I8Ctx` for the
  prefill set. The device is passed at construction (`XdnaTarget(xrt::device*)`), which keeps
  `NpuTarget::open(plan, kernels_dir, family, err)` exactly as documented here. **Done** (#1380).
- **still in the forward, and therefore still owed**: the M=16 decode family and the lm-head context.
  Moving them needs only the setup state that `configure(Setup)` already carries (dirs, family, plan,
  layout flags), which is why that struct exists.
- `load_block(first_layer, n_layers, err)` — **not started, and its premise is now measured, which changes
  what it is for.** It was described here as taking over "`load_layer()`'s packing half (`wcache_` fill +
  pb16i image load)"; at this tree that half is **already the target's**: `wcache_` lives in
  `npu/target/xdna/xdna_target.h` and fills **lazily inside `gemv`**, keyed on the weight's identity, and
  `load_layer()` is the dequantisation path (host float vectors), not the packing one. So `load_block()`
  would not move a cost, it would move *when* a cost is paid — and the measurement says that cost is already
  amortised. `NPU_PROFILE=1` separates it (`LH_PACK_UPLOAD` = "weight BO create + pack + upload (`wcache_`
  miss only)"); on the 27B t2s2 container, one ticket, two runs:

  | window | `pack` per role (ms) | share of the window |
  |---|---|---|
  | first touch (fills the cache) | 86.7 / 149.4 / 170.4 / 209.2 / 117.4 / 82.4 / 102.4 / 60.1 (O/G/U/D/QKV/Z/QKVFAQ/QKVFAKV) | **≈978 ms total, 99 % of each role's first window** |
  | next window, same process (≥15 launches each) | 0.11 / 0.15 / 0.12 / 0.29 / 0.10 / 0.09 | **0.86 ms total** |

  So the residue is **one ~1 s residency per process**, and a pass after the first pays none of it. What
  `load_block()` would buy is **time to first token** (pay it up front, per block, instead of interleaving it
  with the first pass's launches) — *not* throughput, which is what "residency" reads like. That reprices the
  step: worth doing when someone wants first-token latency on a resident model, needing a quiet-box
  measurement to state the gain, and **not required** for this page's acceptance. The numbers above were taken
  at load 41.6 → 23.1 on a shared box, so the *share* is the result and the absolute ms are not.
- the filename probes are gone for everything the plan names, and **`adopt_geometry`/`make_ctx` are still
  on the target's surface** because the decode family and the lm head use them; they become private with
  the move above.

**A3** — `NpuModel`:

- The math + its state (`c_`, `sem_`, `embed_*`, `in_norm_`…, KV, GDN state, `mla_*`, `moe_*`) becomes
  `npu/model/npu_model.{h,cpp}` with `NpuTarget&` injected; `Q4nxNpuForward` becomes a thin adapter that
  owns a model + a target (the serve path keeps calling the same methods, so `app/forward_serve.cpp` and
  `npu/mtp/draft.h` do not change).
- Optional ops get their host fallbacks here: `gdn_step`, `attn_step`, `lm_head` return false from a target
  that does not implement them, and the model runs its own implementation (today's code, unchanged).

### A3a — the fallback seam, and what it can honestly wire today

The interface's promise ("the model falls back to its own host implementation when `supports(op)` is false") had
**no caller anywhere in the tree**: `git grep 'supports(Op::'` found nothing in `npu/` or `app/`, so the contract was
declared and unimplemented. `npu/model/op_dispatch.h` is that missing side, and it is the part of A3 with logic in it
— the rest of A3 is a move.

One function per optional op, each returning "the target did it, do not run yours" or "it did not, run yours". The
answer is `supports(op) && <op>(...)`, never `supports(op)` on its own: a target that **claims** an op and then fails
it (its state is not resident, say) must not swallow the op, or the layer silently produces whatever the
half-executed op left behind. `tests/npu_op_dispatch_test.cpp` (ctest `npu_op_dispatch`, stub targets, no device, no
runtime library) pins both failure modes plus the claiming-and-running case, and runs in CI by hooking onto
`npu_sampling_test`'s target — `ci.yml` builds a named list, so an unhooked test target reports "Not Run" (#635).

**Wired today: `lm_head` only.** Its contract is complete without residency — one hidden row `[H]` to `[V]`, the
weights are the model's, and it is idempotent — so `Q4nxNpuForward::run_lm_head` asks the target first, behind a
`supports()` check so that no per-token `[V]` buffer is allocated while no target claims it. No target on this box
claims it (XdnaTarget's `supports()` is `gemv`-only), so the numerics are unchanged by construction.

**Not wired, and why:** `gdn_step` and `attn_step` require the layer's parameters and state to be **resident**
(`load_block()`), and `XdnaTarget::load_block()` is still the documented no-op ("weights still fill lazily per
projection"). Wiring the dispatch before that residency exists would hand a target an op it cannot honour and hide
the reason inside a fallback that always fires. The helpers exist and are tested; they are the seam those two ops
land on when A2c/`load_block()` arrives.

What remains of A3 proper: the state move (`c_`, `sem_`, `embed_*`, KV, GDN state, `mla_*`, `moe_*`) into
`npu/model/npu_model.{h,cpp}`, and its gate — window11 text identity, logits KLD 0 on 8 prompts, a profile table
within 2 % — which needs device windows and a quiet box.

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

### The one place the move is not a pure copy, and why it is not a dropped safety

`main`'s `npu_gemm_rows` checks `kctx.canary_short()` unconditionally after a launch, and again through the
engine#666 retry. In the target both checks read `!kctx.gemv && kctx.canary_short()`: `I8Ctx::readback()`
runs `verify_canary()` only for a non-gemv context (`npu_engine_i8ctx_inc.h`), so `short_row_` is never set on
a gemv context and `main`'s unconditional check was dead there. The guard is identical on the contexts that
can set it; the conditional just says so. Everything else in `npu_gemm`/`npu_gemm_rows` is the same code with
the mechanical renames (`cur_layer_` → `ar.layer`, `err_` → `err`, `LhScope` → `q4nx_lh::`,
`hadamard_rotate` → `q4nx_hadamard::rotate`).

The host fallback is **not** part of the move, and that is where review found a real defect: the engine#845
activation pad (`q4nx_pad_a_f32`) lives in the target for the device leg, so the host branch had to keep its own
pad — and it did not, reading `A[real_k .. K)` past the caller's buffer and, in the rows form, at stride `K`
from a buffer laid out at stride `real_k`. Both host branches now call `q4nx_host_gemm_f32_padded()`, which owns
the pad-and-read pair so a caller cannot separate them, and `tests/host_gemm_pad_test.c` pins it with a NaN guard
tail after the activation buffer.

## 4b. The two plan-layer rules the owner added after this split (engine#799)

Both are in the tree and both are about *not* discovering a problem at a launch:

- **every op a plan may name ships a host reference with known answers** — `npu/plan/op_registry.{h,cpp}`,
  `ctest: npu_plan_op_reference`. `gemv` and `lm_head` are registered with hand-computed cases (including a
  zero row, negative operands and `m=2`, so the reference is checked on a path the easy case misses); the
  test walks the whole `OpId` vocabulary, so an op added with no entry fails it rather than passing silently.
  `gdn_step` and `attn_step` are the remaining gap, and the contract above them now says why: their
  signatures carry none of the layer's parameters or state, which a reference has to take explicitly.
- **completeness at load, never a silent host fallback** — `Plan::check_artifacts()` refuses a plan whose
  `designs[]` names an artifact no kernel directory has (present and non-empty, first hit wins), naming the
  design, the role and the artifact, **before any context is opened**; `tools/npu_plan.py --check` asks the
  same question with the same rule at import time. What the plan layer deliberately does not check is a
  tensor's quantisation — that is the container's (`npu/q4nx.h`), and a second opinion is how two answers
  start disagreeing.

## 5. Deliberately not in A1

- The int8-activation contract for `gemv` (see §2). Separate step, byte-exact gate.
- `plan.json` consumption, and therefore the `open()` refusals. A1's `open()` exists but derives nothing;
  it returns true when the model dir has the shapes A1 was given, and A2 replaces that with the plan.
- Any Hexagon code. P5 is written against this header; if the header is wrong, P5 says so first.
- Any kernel, weight layout or numerics change. A1 moves code; it does not read it.
