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
# The NPU plan layer (plan.json)

`plan.json` is the model's **stream plan**: the decisions the NPU forward used to make by looking at
which files happened to exist in `--kernels`. It sits beside `model.q4nx` in the model directory,
alongside `config.json` and `tokenizer.json` (see [npu.md](npu.md), "plan.json: the stream plan").

The file is **generated, not hand-written**: `tools/npu_plan.py` writes it from the container's own
tensor table and `config.json`, and `--check` re-generates and diffs against the file on disk. A model
directory **without** a `plan.json` keeps the old filename probes, which is what makes the plan additive.
The loader is `npu/plan/plan.h`; the forward reads the plan at init (`Q4nxNpuForward::init_impl`, the
`plan.json` block in `npu/forward/src/q4nx_forward.cpp`) and derives its designs from it instead of probing the
kernel directory.

Four generated plans are checked in next to the code in `configs/plans/` (one per model directory on the
build box); `tools/npu_plan.py <dir> --check` proves a directory's copy still matches.

## Who writes it, and how

```
tools/npu_plan.py <model dir> [--kernels DIR] [--out PATH] [--check] [--print]
```

- Sources of authority, in order: the container's own tensor table (`model.q4nx`) for layers, roles and
  projection shapes; `config.json` for layout and language-model dimensions; the kernel directory only for
  "which of these artifacts exists".
- `--out` changes where the plan is written (default `<model dir>/plan.json`); `--print` shows it without
  writing; `--check` re-generates and compares (exit 1 on any difference, so a stale plan is a gate failure).
- The generator writes schema version `1bit-npu-plan/1`; `generated_by` is informational.

## Schema (schema version 1)

`format` must be exactly `1bit-npu-plan/1`; any other string is refused by name. The remaining keys:

| key | meaning |
|---|---|
| `model.hidden`, `model.layers` | positive integers; `model.arch` is informational |
| `container.layout` | `rm`, `b8`, `pb16i` or `t2s` -- the weight layout the container declares |
| `container.chunk_kind` | informational (`mode0`/`mode1`/`mode2`/`pb16i`); the chunk bytes are described in `docs/npu.md` |
| `container.pb16i` | true when the container is the integer-exact ternary family |
| `container.tile_order` | q4_1 tile order; only `engine-1` is implemented. Absent means `engine-1`, so containers written before the field keep loading; an unknown id is refused. The loader is its only reader in this tree |
| `fusion.gate_up` | `fused` or `split` (replaces the `full_i8_GU_*` filename probe) |
| `prefill.M` | prefill activation rows; absent/0 means "not stated". Recorded and validated, but the current forward still takes its prefill rows from `Q4NX_DESIGN_M_PREFILL` (128) |
| `decode.M` | decode activation rows; absent/0 means not stated. Recorded for P4’s `NpuModel`; the current forward selects the M=16 family through `decode.dirs`, not through this value |
| `decode.enabled` | false disables the decode twin family |
| `decode.dirs[]` | where the decode family is looked for, in order: `root` is `model` (the `-m` directory) or `kernels` (`--kernels`), `path` is relative under that root, `.` is the root itself. Empty, absolute or `..` paths are refused |
| `decode.sealed_elf` | pb16i only: the sealed ELF the decode roles run |
| `kernels.tag` | artifact-name suffix: `""`, `_b8`, `_pb16i`, `_t2s` or `_t2s2`; must agree with the container layout and `container.pb16i` |
| `attention.mode` | `causal` (default) or `noncausal_full_attention`. The latter is a prompt-encoding mode: every full-attention layer sees the whole prompt, and the forward refuses to generate with it |
| `kv.key`, `kv.value` | what the KV cache is stored as, per side: `f16` (the meaning of an absent `kv` block, and of every plan written before P7) or `int8`. `int8` is int8 with **one fp32 scale per (head, token)**, taken *after* a per-head Hadamard rotation — the rotation is what makes one scale affordable, because it collapses the per-channel magnitude spread the scale would otherwise have to cover. `key` is the side that costs no un-rotation (`(QH)(KH)ᵀ = QKᵀ`), so storing K rotated and rotating Q needs no inverse transform; `value` is a separate choice because an int8 V needs one on the output, which is why K-only is the cheap variant to prove the path with. This is a description until a target opens it: a target without the capability refuses `int8` **by name**, never serves f16 in its place. That check is `check_kv_caps` in `npu/target/target.h`, called from every target's `open()`, and it is gated device-free (`ctest -R npu_target_caps`) |
| `kv.hadamard.block`, `kv.hadamard.signs` | the rotation itself: `block` is `0` (no rotation) or a power of two in `[2, 65536]` — the head width the rotation runs over — and `signs` is an optional **bare file name** (no directory part, under the model dir) for per-channel signs. A `signs` file with `block: 0` is refused rather than ignored |
| `moe.batched_expert_gu` | `{enabled, K, N}`; when enabled, K and N must be named |
| `layers[]` | one entry per layer, in order: `index`, `role` (`attn`/`gdn`/`moe`/`mla`), and projections `qkv {K, N, fused}`, `attn_gate {K, N}`, `gate_up {K, N}`, `down {K, N}`. The count must equal `model.layers`. The projection shapes are recorded for P4’s `NpuModel`; the current forward reads only the count (for its [plan] log line) and derives designs from the container |

Unknown layer keys are ignored by the loader (the current generator also writes `ssm_out`; the forward does
not read it from the plan).

## Refusal rules

The loader never substitutes a default for something it cannot honour; every refusal names the field:

- `format` other than `1bit-npu-plan/1`;
- missing/non-positive `model.hidden` or `model.layers`;
- `container.layout` outside the four values; `container.tile_order` other than `engine-1`;
- `fusion.gate_up` outside `fused`/`split`;
- `attention.mode` missing, or not `causal`/`noncausal_full_attention`;
- `kernels.tag` outside the five values, or disagreeing with `container.layout` / `container.pb16i`
  (the ternary tags `_pb16i`/`_t2s`/`_t2s2` require `container.pb16i: true`; `_b8` requires `b8`;
  `_t2s`/`_t2s2` require `t2s`);
- `decode.dirs` with an empty array, a `root` other than `model`/`kernels`, or a path that is empty,
  absolute, contains `..` or a backslash;
- `kv.key`/`kv.value` outside `f16`/`int8`; a `kv.hadamard.block` that is not `0` or a power of two in
  `[2, 65536]`; a `kv.hadamard.signs` that is a path rather than a bare name, or that names a file while
  `kv.hadamard.block` is `0`;
- `kv.key`/`kv.value` asking for `int8` on a target whose caps say `int8_kv` is false — this one is the
  **target's** refusal, not the loader's (the loader accepts the plan; the target refuses to run it), and it
  names both the plan fields and the missing capability. Today every target reports `int8_kv = false`, so
  every `int8` plan is refused: the cache writer does not exist yet (see P7 below), and the refusal is what
  keeps a request for it from being answered out of an f16 cache;
- `moe.batched_expert_gu.enabled` without positive K and N;
- `layers` missing/out of order, a role outside the four, a projection naming only one of K and N, or a
  layer count that disagrees with `model.layers`;
- `container.pb16i` with an empty `decode.sealed_elf` when `kernels.tag` is `_pb16i` and decode is enabled;
  or a non-empty `sealed_elf` on a non-pb16i container.

The forward adds one more check on load: `kernels.tag` must equal the tag the container itself implies
(same block, `Q4nxNpuForward::init_impl` in `npu/forward/src/q4nx_forward.cpp`), so a plan and a container cannot
disagree about the artifact family. `attention.mode` is the one plan field the container can also state on its own:
`decision_config.json`'s `attention_mode` decides when there is no plan, and a plan that disagrees with it is refused
as stale (see [npu.md](npu.md), "the pack's own declaration").

## What the int8 KV contract is worth, measured

The two claims `kv` rests on are arithmetic, not device behaviour, so they are gated in CI with no model and
no device (`ctest -R npu_kv_int8`, `tests/npu_kv_int8_test.cpp`). Attention-weight **KLD against the f16
cache**, head width 256, one scale per (head, token), 512 keys, 8 draws per cell:

| KV vectors | f16 + rotation | int8 K, unrotated | int8 K + rotation | rotation's worth |
|---|---:|---:|---:|---:|
| uniform (control) | 3.9e-15 | 2.5e-05 | 2.4e-05 | **1.0x** — nothing to concentrate, nothing to claim |
| 8x outlier channels | 8.6e-15 | 4.5e-04 | 4.7e-05 | **9.6x** |
| 30x outlier channels | 4.5e-14 | **8.2e-03** | **1.5e-04** | **54.5x** |

Two readings matter. The rotation is **free**: rotating both sides and taking the dot product preserves the
score to 1.6e-09 relative, which is *smaller* than one f32 dot product's own rounding error (0.01x of it), so
a rotated f16 cache is indistinguishable from today's. And the rotation is what **keeps int8 inside the
plan's gate**: at 30x outliers an unrotated int8 K cache lands at 8.2e-03, an order of magnitude outside
1e-03, while the rotated one stays at 1.5e-04. At 8x outliers the unrotated cache is already inside the gate
— so on a model whose channels are mild, the rotation buys accuracy this gate does not ask for, and on one
whose channels are not, it is the difference between passing and failing.

What this does **not** establish is the gate itself: the vectors are synthetic and generated to the outlier
structure this page describes, and #806's KLD is taken on real KV from a real model on the device. The test
bounds the mechanism; the model run decides.

## How the forward consumes it

At init the forward looks for `<model dir>/plan.json`. If present it loads it and uses the plan for the
three probe decisions (fused gate/up, artifact family, decode dirs), the sealed-ELF choice, and the
attention mode and the optional batched-expert geometry. The activation M values and per-layer projection shapes are validated but not yet read (the forward keeps its design defaults until P4’s `NpuModel` lands). A broken plan is a
named init failure ("plan <path>: ..."), never a silent fallback. A missing plan leaves the existing probe
path untouched.

`scripts/npu-window11.sh` is the gate the plan change was validated with: three fixed greedy prompts plus
the prof table, compared between a run with the plan and one without. `--compare` refuses to run without
`--tol PCT` or `--control C1.log C2.log`; `--self-test` checks that policy.

Tests: `tests/npu_plan_test.cpp` (ctest `npu_plan`) pins the loader and the refusals;
`tests/npu_plan_generate_test.py` (ctest `npu_plan_generate`) runs the generator and `--check` on a
synthetic model directory.

## Relation to the t2 and t2s chunk kinds

The plan states **which family of artifacts** a container needs (`container.layout`, `container.pb16i`,
`kernels.tag`); it does not describe chunk bytes. The chunk formats are documented in `docs/npu.md`:
"Q4NX chunks" (and "The t2 chunk kind") for the 2-bit ternary chunk, and "The t2s stream" for the 2-bit
stream the XDNA decode kernel reads, including the `_t2s`/`_t2s2` family selection the plan's `tags`
mirror.

## Worked example

`~/models/npu/bonsai2-27b-pb16i/plan.json` (built 2026-11-05), key fields only:

```
format        "1bit-npu-plan/1"
generated_by  "tools/npu_plan.py 1.1"
model         {"hidden": 5120, "layers": 64, "arch": "qwen3_5"}
container     {"layout": "pb16i", "chunk_kind": "pb16i", "pb16i": true}   # no tile_order -> engine-1
fusion        {"gate_up": "split"}
prefill       {"M": 128}
decode        {"M": 16, "enabled": true,
               "dirs": [{"root": "model", "path": "kernels-m16"}, {"root": "kernels", "path": "."}],
               "sealed_elf": "pb16i_sealed.elf"}
kernels       {"tag": "_pb16i"}
attention     {"mode": "causal"}
layers[0]     {"index": 0, "role": "gdn", "qkv": {"K": 5120, "N": 10240, "fused": true},
               "attn_gate": {"K": 5120, "N": 6144}, "gate_up": {"K": 5120, "N": 17408},
               "down": {"K": 17408, "N": 5120}}
```

## Not yet

- **P4** (#803): **A1 landed** in #808 (`0a2cbb9`) — the `NpuTarget` interface (`npu/target/target.h`), the XDNA
  target behind it, and `open()` consuming the plan instead of duplicating the derivation. **A2 is in progress and is
  tracked on its own page of record**, [npu-p4-target-split.md](npu-p4-target-split.md) §3 (steps A1/A2/A3), so this
  list does not try to snapshot it — as of #1380 (`A2b, part 1`) the M=128 ELF set is adopted in
  `XdnaTarget::open_prefill()` instead of in the forward. What is still true from *this* page: `XdnaTarget::load_block`
  remains a named no-op that says why, so weights still fill lazily per projection.
- **P5** (#804): Hexagon target (t2 gemv on HVX/HMX, QEMU v73 oracle) has no engine rows yet.
- **P6** (#805): the shared drafter protocol is on main; tau=0.6 and Hexagon acceptance rows are still missing.
- **P7** (#806): the `kv` contract above is in the plan and its arithmetic is gated in CI (`ctest -R npu_kv_int8`,
  which needs no device: see below), and the **target-side refusal** is enforced as well (`npu_target_caps`, so an
  `int8` plan cannot be run as f16 by any target, including P5's Hexagon). Still open, in the order it can be taken:
  the cache **writer/reader** in the
  forward (an int8 K store plus its scale sidecar, the prefix-cache snapshot's byte accounting with it), the **KLD
  gate on real KV from a model on the device** — the CI test bounds the mechanism on synthetic vectors, not the
  model — and **part B**, prefill in 1024/2048-token chunks with block-fp expansion, which waits on P3's on-core
  expansion, P4's `NpuModel`, and the `M>16` decision (prefill *is* M≫16; nothing in P7b asks for M>16 in the
  decode path).
