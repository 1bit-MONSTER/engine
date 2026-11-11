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
- **the lm-head context and the M=16 decode family are the target's** too: `XdnaTarget::open_head()`
  (#1458, #1478) and `XdnaTarget::open_decode_family()` (#2166, A2b). The forward still names the decode
  designs (`open_decode_family_local()`, from the plan's `decode.dirs`) and the target opens and configures
  the M=16 contexts. What remains of context setup in the forward is the optional wide and narrow sealed
  families (`load_family` in the same function), which it still configures field by field.
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

### A3 as landed — five slices, each with a device arm

`npu/model/npu_model.h` owns what the line above calls "the maths + its state", and `Q4nxNpuForward` is the adapter this
page describes: it keeps every member name as a **reference** into `NpuModel`, so no expression in the class changes and
each slice is reviewable as the ownership change it is.

| slice | what moved | expressions unchanged |
|---|---|---|
| target | `std::unique_ptr<XdnaTarget>` | `xd_->…` |
| semantics | `Q4nxSemantics` | `sem_.…` |
| geometry | `Q4nxForwardConfig` (the type moved with it) | `c_.…` |
| norm scratch | `in_norm_`, `post_norm_`, `q_norm_`, `k_norm_` | those four names |
| KV / MLA / recurrent | `kcache_`, `vcache_`, `mla_*`, `conv_state_`, `rec_state_`, `kv_owner_` | those members |

The column carried counts (94 / 82 / 112 / 16 / 54) until a reviewer tried to reproduce them. They were true on each slice's own base and
**not** from the merged tree, where the file has grown: `grep -o` over `npu/forward/src/q4nx_forward.cpp` +
`npu/forward/include/q4nx_forward.h` gives **104** for `xd_->`, **112** for `sem_.` and **169** for `c_.` at the merge. So the integers are
gone rather than wrong — a table of five exact numbers nothing checks is the shape this repository has been bitten by, and the claim that
matters ("every expression unchanged, because the member name is still a reference") does not need them.

Why the last one is safe to move and the aliasing question that had to be answered first: nothing outside the model
aliases those buffers — the model's KV sharing is *between its own layers* (`kv_owner_`), and the dma-buf shared region
lives in the fork's llama.cpp side, a different process. The hazard a reference cannot survive is a vector *re-seated*
(moved out and replaced) rather than grown in place; every mutation here is `insert`/`reserve`/`assign`/`resize` on the
same object.

**The gate as it is actually evaluable, and why "within 2 %" died.** This box moves a row by **up to 7.7 %** when the
*same binary* is run twice (engine#834), and in the 2026-11-09 arms the same-binary control pair's worst row moved
**22 %**. A constant 2 % assertion was therefore never evaluable here. What the arms did instead, via
`scripts/npu-window11.sh`: run the pre-move binary twice (**control pair**) for that window's noise floor, then pre-move
vs post-move, scored on the paired statistic with the tolerance taken from the control. Every slice came back with
**3/3 byte-identical transcripts**, **13/13 identical rows** (same phase/role/M set and launch counts) and no row
outside the control's own spread; the KV/recurrent arm's ratios were **−1.3 % … +2.3 %**. The same five slices against
the shipped **c25 reference** binary also passed (`window11: PASS`, 3/3 texts, 13/13 rows).

What is still not wired, and stated as such: `gdn_step`/`attn_step` dispatch needs `load_block()` residency, and
`XdnaTarget::load_block()` remains the documented no-op; `lm_head` is wired behind `supports()` in `run_lm_head`.

## 4. Gate (unchanged from #803)

1. `window11`: 3 greedy prompts, text identical to the current binary.
2. Logits KLD **0** on 8 prompts, same model dir and kernel set in both arms. (This line used to ask for the hw-gate's prompts; the row below was run on eight fixed completions and says so.)

   **Measured (2026-11-11, on the NPU, under `npu-hold` tickets).** Baseline: the binary at `540e1fd`
   (before the decode-family move), run twice as the control. Candidate: `main` at `d0fa51c` (with #2166).
   Both built against the same runtime. Model `ZR1-1.5B-npu-q4km-v2` with its 8 kernels (4 roles at M=16),
   8 fixed prompts through `/v1/completions` (so both arms tokenise the same text), 8 greedy tokens each,
   `--dump-logits` on, compared with `tools/logits_kld.py`:

   | comparison | positions | mean KLD | max KLD | dumps byte-identical | texts identical |
   |---|---|---|---|---|---|
   | baseline run 1 vs run 2 (control) | 64 | 0.000000 | 0.000000 | 64 of 64 | 8 of 8 |
   | baseline vs `main` | 64 | 0.000000 | 0.000000 | 64 of 64 | 8 of 8 |

   The dumps are the same bytes, so the KLD is 0 exactly, not to a tolerance. Two limits: the prompts are
   eight fixed completions, not the hw-gate's list, and the model is the 1.5B one, because the 27B container
   did not fit above the box's memory floor that day; the 27B row of this gate is item 1's (window11, three
   texts identical across the same two binaries, on #803).
3. `NPU_PROFILE=1` profile table: every row within 2 % of the pre-split binary, same row count.
4. `ctest` green; `--npu-target xdna` is the default and an unknown target name is refused by name.
   The flag is `forward-serve`'s (the per-op route's own command) and `serve` carries it through to that child,
   so the list of compiled targets and the refusal live in one place. The refusal is checked **before the
   device is opened**, which is what makes the row measurable without a ticket: `--npu-target hexagon` prints
   `--npu-target hexagon: this build has no such target (compiled: xdna)` and exits, and the same message comes
   out of `serve --device npu --npu-target hexagon` because it reaches the forward unchanged.

   **Measured (2026-11-09, `ONEBIT_NPU=ON` against the shipping runtime, no ticket and no device — the refusal is
   before both):**

   | form | what it prints |
   |---|---|
   | `forward-serve … --npu-target hexagon` | `--npu-target hexagon: this build has no such target (compiled: xdna)` |
   | `forward-serve …` (default) | proceeds to its own checks — `xdna` is the default |
   | `serve --device npu --npu-target hexagon` | the same by-name message, carried through the child |

   **One caveat the `serve` row earned:** it only reaches the child after `serve`'s own kernel resolution succeeds.
   With a pack whose full ELFs resolve nothing, `serve` refuses first with *its* message (`… no NPU kernel (full ELF)
   directory resolves for it: set ONEBIT_NPU_KERNELS …`) and the target refusal is never printed — measured on
   `Qwen3-0.6B` and on `bonsai2-27b-t2s2-serve`, whose `kernels-m16/` does not satisfy that check. That is not a
   defect to fix in code: the route cannot run without kernels either, and duplicating the compiled-target list into
   `serve` to reorder the two refusals would break the rule that keeps the list in one place. It is a reason to set
   the kernels up before testing this row, which is what the table's third line did.

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
