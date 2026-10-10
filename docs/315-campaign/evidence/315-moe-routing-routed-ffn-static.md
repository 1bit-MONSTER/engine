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
<!-- Copyright 2026 bong-water-water-bong
     SPDX-License-Identifier: Apache-2.0

     engine#315 evidence: the routed-FFN path, read statically, now that section 47 has narrowed the fault
     to one token's hidden row entering at the MoE aggregation. Tree read: ~/wt/llama-315copy
     (pin e44c9d01a4d5 + the diagnostic patches) on strixhalo. Read-only; no GPU, no arm. -->

# The routed-FFN path on HRX: what builds the routing tables, and the sites that can lose one token's row

Section 47 moved the fault onto the MoE aggregation, so the static leg is re-pointed from the decode-split
flash-attention kernel to the routed FFN. What follows is a map with symbols and line numbers, one hard
negative, and the candidate sites ranked by whether they can produce a *single* token's worth of damage.

## 1. What GLM actually dispatches on this path

`evidence/315-glm-dispatch-inventory.md` (same pin, `GGML_HRX_LOG_DISPATCH=1`) gives the kernel set. The
routed-FFN half is:

```text
common.argsort_rows_f32          the top-k over the router logits
common.get_rows.f32              common.get_rows_small_f32      common.set_rows
common.mul_mat_id.f32_f32_wmma              the prefill expert matmul (T > 1)
common.mul_mat_id_swiglu.f32_f32_wmma       gate+up fused with SwiGLU
common.mul_mat_id.decode_f32_wave64         the T=1 GEMV variant
common.gather_add_f32   common.sum_rows_f32 common.concat_dim0_f32
```

The prefill graph is where the NaN is first seen (T of 87-342 in section 47's table), so the relevant
consumers are `mul_mat_id.f32_f32_wmma` and the fused `mul_mat_id_swiglu.f32_f32_wmma`.

## 2. `route_ids` is a strided view of the argsort, and the stride is a contract

`dispatch_registration/common/dispatch-mul-mat-id-common.h:229-238`:

```c
// llama.cpp's route ids are a [n_expert_used, n_tokens] view of the [n_expert, n_tokens] argsort,
// so rows are n_expert apart; the routing tables must read them at that stride
const int64_t route_stride = static_cast<int64_t>(route_ids->nb[1] / sizeof(int32_t));
if (token_count > 1 && (route_stride < route_count || route_stride > expert_count)) { ... return {}; }
```

`route_stride` is derived from the tensor's *view stride* (`nb[1]`), not from any shape, and it is passed to
every kernel that touches routing: the two table builders (as `integer_parameters.emplace("route_stride", ...)`,
`:346`) and the matmul itself. A view whose `nb[1]` disagrees with the layout `argsort_rows_f32` writes would
make every reader disagree about where a token's route list starts - but the guard above refuses the dispatch
unless the stride is inside `[route_count, expert_count]`, so a *gross* mismatch is already excluded. A
subtle one (correct stride, wrong row base for some token) is not.

## 3. The routing tables are built by two dispatches inside the same plan - the stale-table hypothesis dies here

`common_mul_mat_id_ensure_moe_routing_bundle` (`dispatch-mul-mat-id-common.h:277`) allocates the expert table
and the partition table as **transient** values, records them in plan metadata
(`append_generated_resource`, roles `MoeExpertTable` / `MoePartitionTable`, `:319-334`) and then appends a
producer dispatch for each (`:342-368`):

```c
Dispatch expert_table_dispatch;
expert_table_dispatch.kernel = make_kernel_specialization(kMoeBuildExpertTableKernel);   // ggml_moe_build_expert_table
... .integer_parameters.emplace("route_stride", route_stride); ...
expert_table_dispatch.bindings.push_back({ match.route_ids->id, 0, route_ids_length });
expert_table_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
dispatch_match.dispatches.push_back(std::move(expert_table_dispatch));

Dispatch partition_table_dispatch;                                    // ggml_moe_build_expert_partition_table
partition_table_dispatch.bindings.push_back({ expert_table_value, 0, expert_table_bytes });
partition_table_dispatch.bindings.push_back({ partition_table_value, 0, partition_table_bytes });
dispatch_match.dispatches.push_back(std::move(partition_table_dispatch));
```

Hard negative, and a useful one: the tables are **rebuilt as part of the plan they are consumed by**, bound to
the live `route_ids` value, so "a table left over from an earlier graph feeds this graph's routing" is not
available as a mechanism. `has_routing_bundle` reuse (`:280-283`) only short-circuits the *second* consumer of
the same `route_ids` within one plan.

What that leaves is ordering *inside* the program: two producers, then the consumer, in one dispatch list whose
order is the only thing tying them together. That is the same class as the AQL split lead (mechanism doc
section 32) and it is where a pressure-sensitive window would live.

## 4. The consumer's bindings, and the fused path's real hazard

`dispatch_registration/common/dispatch-gated-mul-mat-id.cpp:185-190` binds, in order: input, expert table,
partition table, gate weight, up weight, output. The fused kernel
(`kernel-corpus/kernels/loom-libs/ops/mul_mat_id_swiglu_f32_f32_wmma.loom`) takes the *descriptor* out of the
partition table and unpacks it (`:32-53`):

```text
%expert            = index.assume %expert0            [range(%expert0, 0, 511)]
%route_tile_base   = index.assume %route_tile_base0   [range(%route_tile_base0, 0, 2016)]
%partition_row_count = index.assume %partition_row_count0 [range(%partition_row_count0, 1, 32)]
```

Three things are worth recording about this:

* these are `index.assume` with ranges, i.e. **assertions used by the optimizer, not checks**: a descriptor
  outside those ranges is not clamped and not reported, it is undefined behaviour by construction. A stale or
  half-published descriptor is therefore free to become an out-of-range tile base rather than a visible error;
* `route_tile_base` is the *token* coordinate of the tile, so a wrong value is precisely a *per-token* error -
  the shape section 47 measured;
* the fused SiLU uses fast-math flags on the division
  (`%sigmoid = vector.divf<reassoc|nnan|ninf|nsz|arcp> %c1_f32x4, %denominator`, `:62`). With `nnan`/`ninf` the
  compiler is entitled to assume its inputs are finite, so an infinite or NaN *input* to this kernel is not
  required to produce an infinite or NaN output - it may produce anything. That matters for how a Face-B
  (finite, wrong) output is read: it does not have to mean the corruption was finite going in.

The prefill aggregate kernel carries a publication protocol of its own
(`mul_mat_id_f32_f32_wmma.loom:8-10`, `:45-48`, `:49`): `partition_row_count`,
`maximum_partition_count`, `expert_table`, `completion_counters`, and a `publish_vector4(publish_word,
completion_counters ...)` template - i.e. partition-ordered multi-workgroup output with an atomic completion
counter. An early or unordered completion publication is another way for exactly one tile's worth of rows to be
published before they are written.

## 5. Candidate sites, ranked, each with the thing that would falsify it

| # | site | why it can lose one token's row | falsified by |
|---|---|---|---|
| 1 | intra-plan ordering between the two table builders (`dispatch-mul-mat-id-common.h:342-368`) and the consumer binding them (`dispatch-gated-mul-mat-id.cpp:185`) | a descriptor consumed before its producer's writes land gives that tile a wrong `route_tile_base`/`partition_row_count` | block-size dose-response (`HRX_AQL_BLOCK_SIZE` 262144 vs 32768) moving the fault, or a trace showing the producer and consumer in the same block with a barrier |
| 2 | `completion_counters` publication in `mul_mat_id_f32_f32_wmma.loom:8-10,45-48` | a row published before it is written is a row of previous contents - NaN if that is what the buffer held | reading the counter's expected value against the published rows in a faulting arm |
| 3 | transient allocation of the two tables (`:311-317`, `expert_table_bytes` / `partition_table_bytes`, alignment 256) aliasing with another live transient | one clobbered descriptor is one bad tile | the packer's own log for the aliasing decision in a faulting arm (`GGML_HRX_LOG_TRANSIENT_REUSE=1`) |
| 4 | `route_stride` contract, section 2 above | a stride/row-base disagreement shifts a token's route list | the argsort output's actual layout vs `nb[1]` in a faulting arm |

## 6. Hard negative recorded, so it is not re-walked

The engine's own MoE expert *streaming* cache (`moe/expert_cache.cpp`, `moe/expert_cache.h`, reached by
`serve --moe-slots` / `--moe-subst`) is a Vulkan/streaming feature: experts live in pinned host RAM slots and
`acquire()` blocks until an expert is resident. Every #315 arm runs
`llama-server -m GLM-4.7-Flash-Q4_K_M.gguf -dev HRX0 -ngl 99 -fa on`, which loads the model into GTT through
the HRX backend and never enters that cache. It is off this path, and its "blocks until resident" pattern is
therefore not a candidate for the fault.
