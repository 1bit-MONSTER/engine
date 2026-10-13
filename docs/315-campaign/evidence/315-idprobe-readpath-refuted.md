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
<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The read path is REFUTED, and the i32 argsort tensor genuinely holds floats

## The pre-committed readout, fixed before the run

The id probe's first result (`315-idprobe-result.md`) left one unexplained observation: `ffn_moe_argsort-N` is
an **i32** tensor that by construction must hold ranks in `[0, ne0)` — `ggml.c:5347` allocates it as
`ggml_new_tensor(ctx, GGML_TYPE_I32, ...)` with `op = GGML_OP_ARGSORT`, and `ggml_argsort_top_k` merely
`view_4d`s it — yet the instrument read **float-shaped values** there. Two explanations, separable:

- **(a)** the read path reported memory that is not the tensor's contents — `buffer_get` does a plain `memcpy`
  and **returns** for a host-visible buffer (`context->base != GGML_HRX_FAKE_PTR_BASE`), never reaching
  `hrx_synchronous_d2h`;
- **(b)** the argsort op genuinely did not put ranks in that buffer.

So the instrument was patched (`tools/patch-id-instrument-sync.py`) to read **each tensor twice in the same
line**: once the ordinary way, and once through `hrx_synchronous_d2h`. Readout, fixed in advance:

- memcpy floats **and** sync ranks → (a): the read path was the defect and the family becomes testable;
- **both float-shaped → (b)**;
- both ranks → the earlier float reading did not reproduce.

## The result: (b)

Arm `p1l` of the sync run (`315idsy`), `ffn_moe_argsort`, per tag:

| read | lines | plausible ranks (`0 < max < ne0`) | top `max` values |
|---|---|---|---|
| `hrx-ids-fp` (plain memcpy) | 60,678 | **0** | 16,616 × `0`; 3,036 × `1072430089`; 2,921 × `1069385080` |
| `hrx-ids-sync` (synchronising d2h) | 60,822 | **0** | 16,614 × `0`; 3,036 × `1072430089`; 2,921 × `1069385080` |

And pairwise over every tensor and call in the arm:

    fp lines: 138,278     sync lines: 138,330     identical: 138,256

The 22 fp-only and 74 sync-only residuals are **extraction artifacts**, not disagreements: they are
`bytes=0 count=0` lines and one line clipped mid-number (`call=10354` against `call=103549`). So the two reads
agree on essentially every observation.

**The plain memcpy and the synchronising device→host read return the same values.** The instrument therefore
has no bias from the host-visible shortcut, and *the float contents are really in the tensor*.

**Branch (b) fires: the argsort output buffer does not hold ranks — not as a read artifact, but as the
buffer's actual contents, confirmed by two independent read paths.**

## What this means, and what it does not

It **does not** mean the model routes on garbage: this arm produced correct output, as clean arms always do.
The consistent explanation is that **the argsort output tensor is vestigial in this graph** — the fused HRX MoE
path performs its routing internally and never consumes `selected_experts`'s buffer contents, so no kernel
writes ranks there and the buffer keeps a previous occupant (float probabilities) or zeros.

Two consequences, and they are the point of recording this:

1. **The route-id question cannot be answered by instrumenting ggml tensors in this graph.** The ids that
   matter live inside whatever op actually performs the routing, not in the named tensor. This closes the
   measurement avenue by demonstration rather than leaving it merely untried.
2. The read-path hypothesis — which would have been a large claim, implicating every host-side tensor read in
   this campaign — is **refuted for this arena** by its own pre-registered test. It is worth stating plainly,
   because it was a plausible and attractive explanation and it is now dead on evidence.

**The id-producing / argsort family is therefore not "untested for lack of effort" but unmeasurable with this
instrument, for a now-named reason.** It is still neither excluded nor implicated. No criterion changes and no
arm from this probe is eligible.

## Scope note

This settles the mechanism for the *MoE-path* tensors. It does not by itself clear the same read path for
other buffers in other arenas; it says these two reads agree *here*. The campaign's head-side NaN readings went
through `tail:N` position selection on other tensors and are unaffected by this result.

## Corrections carried in this file

The comparison above was first attempted with a mis-indexed `awk` field test (`$1` against `$2` on a line whose
first field is `[hrx-ids-X]` and second is `call=…`), which reported a meaningless "133,418 differing pairs".
That number is withdrawn; the pairwise comparison above replaces it. This is the **fourth** parse slip of the
campaign in the same family as the first three (a truncated sorted head, a line counter on newline-free output,
a sum over streaming responses): a comparison whose fields did not mean what the expression assumed.

---

# Correction: the floats are a REUSED ARENA SLOT, not a vestigial tensor

The conclusion above — *"the argsort output tensor is vestigial in this graph"* — was an inference from the
float contents, made before checking how those tensors are allocated. **Checking the allocator changes it.**

## The arena

`ggml/src/ggml-hrx/backend-context.h:63` holds a single `ggml::hrx::TransientArena transient_arena;`, and
`runtime/transient-arena.cpp` shows it is **one buffer**, resized on demand:

    Status TransientArena::ensure_capacity_locked(device, stream, required_size) {
        const bool diagnostic_fresh = std::getenv("GGML_HRX_DIAGNOSTIC_FRESH_TRANSIENT_ARENA") != nullptr;

`buffer_` is a single `hrx_buffer_t`, released only in `clear()`; `acquire_allocation_lease()` hands out a
lease (a held `unique_lock`) over that same memory. The decisive detail is the **diagnostic flag whose whole
purpose is to allocate a fresh arena** — `GGML_HRX_DIAGNOSTIC_FRESH_TRANSIENT_ARENA`. An option exists to
escape reuse, therefore **reuse is the default**.

Within a graph the standard ggml allocator places tensors at offsets in this buffer, and it **reuses the slot
of a tensor whose lifetime has ended**. The fingerprint reads *after the whole graph has executed*
(`ggml_backend_synchronize(backend)` then a loop over `graph->nodes`). By then, an intermediate tensor's slot
may already have been reassigned to a later node — here, a later f32 tensor, which is exactly why an i32 rank
tensor reads back as **probabilities**.

## What this explains, and what it does not need

It accounts for every observation without needing the tensor to be vestigial:

| observation | explanation |
|---|---|
| floats in an i32 tensor | a later float tensor now occupies the slot |
| zeros | slots the allocator never wrote |
| identical under memcpy **and** synchronising d2h | both reads return the same real memory — the read path was never the issue |
| same in faulting and clean arms | slot reuse is normal allocation behaviour |
| the model still routes correctly | the ids were correct **when consumed** |

**So the earlier "vestigial tensor" reading is withdrawn.** The ids are very likely computed correctly and
consumed correctly; the instrument simply looks at the memory too late.

## The consequence that matters for this campaign

**Post-execution reads of graph-allocated tensors are not tensor-faithful in general.** This is not a defect of
one tensor: it affects any intermediate tensor whose slot is reused before the read. That has a benign
corollary for the campaign's existing evidence — the NaN localisations used **`tail:N`**, the *last* N nodes,
whose slots are the least likely to have been overwritten, which is consistent with those readings being
informative while name-based intermediate reads are not. It is not a clean bill of health either: within a
`tail:16` window, earlier nodes can still be overwritten by later ones, so the per-node NaN *attribution* is
weaker than the NaN *totals*.

To read an intermediate tensor faithfully, one of:

1. **read at production** — hook the node's own execution and read immediately after that node runs, rather than
   after the graph; this is the real fix;
2. `GGML_HRX_DIAGNOSTIC_FRESH_TRANSIENT_ARENA`, which removes reuse but also changes the allocation behaviour
   under study, so it is not a neutral instrument;
3. read only tensors whose lifetime reaches the end of the graph.

Until one of those is used, **the route-id question stays unanswerable and the family stays neither excluded nor
implicated** — now for a reason that is understood, demonstrated, and general, rather than a missing run.
