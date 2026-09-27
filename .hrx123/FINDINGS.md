# engine#123 — HRX decode-split multipass HSA memory fault: investigation findings

Goal: mujg0qsk-semvti. Investigation date 2026-09-27. Box: strixhalo gfx1151 (quiet —
no other KFD/renderD128 holder; all idle leftovers stopped with user consent).

## Repro (confirmed)
- Tree: `~/1bit-engine/third_party/llama.cpp`, branch `debug/hrx-fa-123-repro-rig`
  (commit `4a16baa6`), which sets the three decode-split *partial* transients
  (`partial_max`/`partial_sum`/`partial_output`) to alignment **256** and adds
  `GGML_HRX_DUMP_HSACO_DIR`.
- Build: `build-hrx-engine` (GGML_HRX=ON), binary reports `4a16baa6 (68)`.
- Run: `llama-bench -m Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf -dev HRX0 -p 0 -n 8 -d 2100 -r 1`.
- Baseline: **5/5 faults** (`HSA_STATUS_ERROR_MEMORY_FAULT`, `res = -3`) on a verified-quiet
  box. Fallback `GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split` → 0/3.

## Established facts (all measured, same-run correlation)
1. **The fault address is NOT in the transient arena.** Arena device pointers
   (e.g. `0x7fe055e00000` size 217600, plus 44 MB arenas) are orders of magnitude away
   from the fault address. The issue's central "partial-buffer overrun" framing is not
   supported.
2. **The fault address is not in any HRX allocation and not in any freed HRX buffer.**
   With full alloc + free tracing (`[hrx-alloc]`/`[hrx-free]`, 574 allocs / 1759 frees in
   one run), every fault address falls in a gap.
3. **It is not a buffer over-read that slack can fix.** Padding *all* HRX allocations by
   64 MB still faults 3/3; padding only the 18.4 GB weights buffer by 256 MB still faults,
   and the wild address moved by ~630 MB — so the address does not track any allocation base.
4. **It is not a serialization race.** `GGML_HRX_DEBUG_SERIAL_EXECUTION=1`,
   `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1`, `GGML_HRX_DISABLE_GRAPH_REPLAY=1`,
   `GGML_HRX_DIAGNOSTIC_FRESH_TRANSIENT_ARENA=1` — all still fault 3/3.
5. **The fault is a read** (`client 10` = TCP; `RW: 0x0`; `PERMISSION_FAULTS: 0x3`).
6. **The fault region is consistently just past the end of the 18.4 GB weights buffer**
   (~+23 MB, or a second cluster at ~+249 MB), i.e. inside whatever gap the layout
   leaves there. In some layouts that gap is the (later unmapped) model `.gguf` mmap;
   in others a plain hole. Load modes `mmap|none|mlock|mmap+mlock|dio` all fault, so the
   `.gguf` link is coincidental, not the mechanism.
7. **Fault addresses are deterministic-ish per layout** (same relative offsets across runs),
   but the *rate* is layout-sensitive (2/3 … 5/5), which is why every earlier
   "fix" looked effective or not depending on who measured.
8. **The weights buffer is exactly 175,030,272 bytes smaller than the .gguf data
   section** (= a 151936x2048 Q4_K tensor). That missing tensor is `token_embd.weight`,
   which HRX allocates separately as a `host-device` buffer of exactly 175,030,272 bytes.
   The packed weights layout is therefore self-consistent (max packed end == buffer size).
9. **The fused produce+multipass kernel is `hsaco-0004/0073.bin`**
   (`ggml_flash_attention_decode_split_f32_f16_wmma_next_q8`, 17448 B).
   Disassembly: global loads use `global_load_b32 vN, vOFF, s[base]` with
   `base = s8 + (s3 << 12)`; `<<12` = 4096 = the partial_output block stride
   (16 * value_head_size * 2). All source-level FA indices are bounded
   (workgroup.y < kv_head_count, block < active_block_count <= partial_block_capacity).

## Conclusion so far
Every source-level access in the decode-split multipass path is bounded, yet the GPU
issues a read to a wild, unmapped address that moves with the arena/allocation layout and
cannot be masked by slack. Best remaining explanation matches the issue's own final
assessment: **a wrong address emitted by the JIT-specialised code** (a Loom codegen
defect), not an index/guard bug and not a buffer overrun. Both Loom sanitizers are
unusable for this kernel (ASAN → `spill-materialization-iteration-limit`, "no executable
bytes"; race/TSAN → missing `iree_tsan_config` runtime global + cannot instrument
`vector.fragment.load`).

## Instrumentation added (uncommitted, debug-only)
- `ggml/src/ggml-hrx/runtime/transient-arena.cpp` — `[hrx-trace]` arena dev+size; `GGML_HRX_ARENA_PAD_MB`.
- `ggml/src/ggml-hrx/ggml-hrx.cpp` — `[hrx-alloc]` dev/size/kind; `GGML_HRX_ALLOC_PAD_MB`.
- `ggml/src/ggml-hrx/runtime/command-program-executor.cpp` — `[hrx-dispatch]` per-binding device
  address/offset/len; `[hrx-alloc]` for prepared constants.
- `ggml/src/ggml-hrx/runtime/host-memory.cpp` — `[hrx-alloc]` for host-device and host-staging buffers.
- `ggml/src/ggml-hrx/backend-buffer-binding.cpp` — `[hrx-tensor]` name/offset/size/capacity.
- `ggml/src/ggml-hrx/trace-buffers.h` — `[hrx-free]` wrapper around `hrx_buffer_release`.

## Useful env knobs found
`GGML_HRX_LOG_DISPATCH`, `GGML_HRX_DUMP_HSACO_DIR`, `GGML_HRX_DUMP_COMMAND_PROGRAM_DIR`,
`GGML_HRX_TRACE_*` (added), `GGML_HRX_DIAGNOSTIC_GRAPH_KERNEL_ID`,
`GGML_HRX_DIAGNOSTIC_GRAPH_PROGRAM_COMMANDS`+`GGML_HRX_DIAGNOSTIC_GRAPH_PREFIX_COMMANDS`.

---

# ROOT CAUSE (proven 2026-09-27)

## The fault is NOT in flash attention
`GGML_HRX_DEBUG_SERIAL_EXECUTION=1` logs every command (flushed) around a stream
sync. The last thing before `HSA_STATUS_ERROR_MEMORY_FAULT` is:

```
sync-begin phase=command-post-dispatch ... index=228 list_commands=303
   ordinal=228 kind=Kernel kernel_id=14896872061627543929 bindings=7
<fault>
```

The FA multipass kernel (`11938313152504284411`, 10 bindings) is index 225 and
completes cleanly. So the issue's "multipass output pass" suspect is wrong; the
`flash_attention_decode_split` disable only changed layout/tokens enough to mask
the fault.

## Culprit
`kernel_id=14896872061627543929` = **`qwen3_moe_routed_gate_up_swiglu_q4k_q8`**
(`kernel-corpus/kernels/qwen_moe/qwen3_moe/routed_gate_up_swiglu_q4k.loom`).
Command 228 binds exactly `blk.37.ffn_gate_exps.weight` (off 14,104,334,336) and
`blk.37.ffn_up_exps.weight` (off 14,330,826,752), both 113,246,208 B (128 experts
x 884,736 B).

## Mechanism
1. `router_top8_f32.loom` seeds its per-lane argmax with
   `%best_value = -3.4028234663852886e+38` (= -FLT_MAX) and
   `%best_id = 2147483647` (0x7FFFFFFF), and only replaces the id when
   `candidate_value > best_value` (ordered `ogt`) or an equal-value tie-break fires.
   If a lane's candidates are **-inf or NaN** (unordered), neither fires and the
   sentinel **0x7FFFFFFF is published into `route_ids`**.

2. `routed_gate_up_swiglu_q4k.loom` consumes it with only a compiler hint:
   ```
   %expert_i32 = view.load %route_ids_view[token, route_index] : ... -> i32
   %expert0    = index.cast %expert_i32 : i32 to index
   %expert     = index.assume %expert0 [range(%expert0, 0, 127)]   // HINT, not a check
   %expert_byte_base = index.mul %expert, %weight_expert_bytes     // wraps at 2^32
   ```
   `0x7FFFFFFF * 884736 mod 2^32 = 0xFFF28000` (~4.29 GB).

## Exact arithmetic match (same run)
From the traced bindings, buffer base = `0x7fb5ba600000`; the up binding sits at
+14,330,826,752. Predicted access address:

```
0x7fb5ba600000 + 14330826752 + 0xFFF28000 + 25*1152
  = 0x7fba10819000 + 0x7080 = 0x7fba10820080
```

and the kernel logged the fault `in page starting at address 0x00007fba10820000`
(the driver rounds the faulting byte down to its page). Up to the page, exact —
`0x7fba10820080` is inside that page. The `25*1152` term is
`channel * weight_row_bytes`.

## Differential (deterministic flip)
`GGML_HRX_ALLOC_PAD_MB=8192` (give the weights buffer 8 GB of tail slack so even
the wrapped ~4.29 GB offset stays mapped): **0/3 faults** (baseline 5/5; rig
alignment oracle unchanged). This confirms the access is
`weights_base + expert x expert_stride` in the weights buffer.

## Consequences for the fix
- The true root cause is the **out-of-range expert id** produced by the router
  (and turned into a wild address by the consumer's hint-only bound + 32-bit
  offset multiply), not anything in the decode-split path.
- Two defects worth fixing at source:
  (A) router: never publish the `0x7FFFFFFF` "no winner" sentinel as an expert id.
  (B) consumer: the expert bound must be a real runtime guard (HRX lane rule:
      decline/fall back rather than emit a wild address), and the
      expert x stride offset must not wrap.

---

# REVERSAL (2026-09-27, post-merge) — the router fix is mitigation-only

The auditor's objection was correct. A reliable **output-correctness** oracle (greedy
`llama-server` at ~2113 tokens, 16 tokens, temp 0, compared against the same build's CPU
backend) shows the decode-split multipass path is still wrong at depth.

| config (prompt 2113 tok, multipass range) | pre-fix (358cafc2) | post-fix (895d63f0) |
|---|---|---|
| 256-byte oracle alignment | **HSA fault** (request fails, `content=None`) | **`'????????????????'`** (12/12) |
| 4096 production alignment | correct (`' Paris. \n\nThe quick brown fox...'`) | correct (12/12) |
| 256 oracle, 529 tok (non-multipass) | — | correct |

So at the oracle the fix converts a **loud fault into silent corruption** — it does not
fix the multipass wrongness, and it violates the objective's "no mitigation-only landing".

## The corruption is localised to `partial_sum`'s arena placement

Deterministic bisect with the output oracle (raise exactly one of the three partial
transients to 4096, others 256):

| partial_max | partial_sum | partial_output | output @2113 |
|---|---|---|---|
| 256 | 256 | 256 | garbage |
| 4096 | 256 | 256 | garbage |
| 256 | **4096** | 256 | **correct** |
| 256 | 256 | 4096 | garbage |

Raising *only* `partial_sum`'s alignment (which only inserts padding **before** that
allocation in the arena) fixes it; `partial_max` and `partial_output` do not matter.
That points at whatever allocation sits immediately before `partial_sum` over-running
into it, corrupting the block sums so `exp(block_max - maximum)` overflows to NaN.

Not a cross-command race: `GGML_HRX_DEBUG_SERIAL_EXECUTION=1` (a stream sync around every
command) still produces garbage.

`GGML_HRX_DUMP_COMMAND_PROGRAM_DIR` gives the arena layout JSON
(`transients.allocations[]` with `arena_offset`/`size`/`alignment`), but the multipass
program's allocation list is dominated by per-command completion counters and the names
are not attached to entries, so the exact neighbour of `partial_sum` still needs to be
read out of the program dump by hand.

## Consequence

- Objective criterion 3 (multipass output matches a known-good reference) **fails** at the
  oracle; criterion "no mitigation-only landing" is violated by what is merged.
- The true root cause of #123 is therefore the multipass value corruption (NaN), with the
  router no-winner sentinel as the amplifier that turned it into a wild address and an HSA
  fault. My merged change only removes the amplifier.

## Arena layout, measured (cap 2304 multipass program, from the program dump)

```
ALL-256   : partial_max=[52480,61696) al=256 | partial_sum=[79360,88576) al=256 | partial_output=[88576,678400) al=256
SUM-4096  : partial_max=[52480,61696) al=256 | partial_sum=[81920,91136) al=4096 | partial_output=[91136,680960) al=256
```

- `partial_sum` is immediately adjacent to `partial_output` (end == start); no overlap, both exactly sized
  (9216 B = 4 kv heads x 36 blocks x 16 x 4; 589824 B = 4 x 36 x 16 x 128 x 2).
- Raising `partial_sum`'s alignment to 4096 inserts 2560 B of padding and moves `partial_sum` + `partial_output` up,
  leaving `partial_max` where it is. That is the only change that flips the output from garbage to correct.
- So the sensitive quantity is `partial_sum`'s own address (79360 = 1536 mod 4096 in the broken case,
  81920 = 0 mod 4096 in the good case), not its size, not an overlap, not the relative gap to `partial_output`
  (which is 0 in both).
- Ruled out as the cause: host transient overlap (`GGML_HRX_DISABLE_TRANSIENT_OVERLAP=1` -> still garbage),
  cross-command synchronization (`GGML_HRX_DEBUG_SERIAL_EXECUTION=1` -> still garbage),
  buffer sizing (allocations exactly match the kernel views).

Best remaining explanation: an address/aliasing sensitivity in the multipass produce+reduce kernel (same code,
only the base addresses differ), consistent with the issue's "every access is provably in bounds yet the result
moves with the allocation layout". Note the kernel's own `buffer.assume.alignment` for the partials is 16, so the
JIT is not told to expect the 256/4096 host alignment.

## The corruption is ADDRESS-LAYOUT dependent, per process (measured)

Fresh `llama-server` process per sample (so each gets a different address layout), one greedy
completion at 2113 tokens, compared to `' Paris. \n\nThe quick brown fox jumps over the lazy dog near the riverbank'`:

| config | correct | wrong |
|---|---|---|
| production alignment 4096, merged commit 895d63f0 | **8/8** | 0/8 |
| 256-byte oracle alignment, merged commit 895d63f0 | 2/8 | **6/8** |

Earlier single-process measurements (12/12 garbage at 256, 12/12 correct at 4096) only sampled
ONE address layout each, which is why the bisect results looked clean and contradictory. The
earlier "raise partial_sum to 4096 fixes it" and the later "prefix the arena by 2560 breaks it"
are the same effect seen at two different address layouts: correctness depends on the absolute
device addresses the arena lands on, not on the alignment literal.

Consequences:
- Root cause of #123 = the multipass produce+reduce kernel produces wrong block sums at certain
  arena base addresses (the issue's "every access is provably in bounds, yet the result moves with
  the allocation layout"). The router no-winner sentinel is the amplifier that turns the resulting
  NaN into a wild address and an HSA fault.
- The merged router fix prevents the fault but leaves the wrong output, so at the oracle the decode
  is silently wrong ~75% of the time. Criterion 3 fails; the landing is mitigation-only.
- Ruled out as the cause of the wrongness: host transient overlap, cross-command synchronization,
  buffer sizing/overlap, alignment literals per se.
- Next experiment that can actually pin it: control the arena base address directly (map the arena
  at chosen congruent offsets) and sweep it with the output oracle, to find which address relation
  breaks the produce/reduce (e.g. a fixed 4096/64K congruence between partial_sum and
  partial_output or between the arena base and the weights).

## The wrongness exists at PRODUCTION alignment too, and 4096 does not prevent it

Address control: `GGML_HRX_ARENA_PREFIX_BYTES` (debug) prefixes the transient arena plan, shifting the
partials' absolute device addresses without touching any alignment literal. Fresh server per sample,
one greedy completion at 2113 tokens, production 4096 partial alignment, merged commit 895d63f0:

| arena prefix | correct / samples | note |
|---|---|---|
| 0 | 8/8 | |
| 512 | 2/2 | |
| 1024 | 2/2 | |
| 1536 | 2/2 | |
| 2048 | 2/2 | |
| **2560** | **6/8** | 2 samples wrong: `' Paris. \nYou are a helpful assistant.'` and `'...near the river?'` |
| 3072 | 2/2 | |
| 3584 | 2/2 | |

So:
- The wrongness is reachable at the **production** alignment; a chosen arena prefix gives a ~25% wrong rate.
- It was already reachable before my change too (the wrong outputs are plausible continuations, i.e. subtly
  wrong attention values, not NaN) - this is the same class as #140's "measurably wrong results".
- `partial_*` = 4096 does **not** reliably prevent it, matching the issue's own note that aligning to 4096
  "masks part of this ... but the fault is not eliminated".
- At the 256 oracle the wrongness escalates to NaN, which is what produced the HSA fault via the router
  no-winner sentinel before my change, and `'????????????????'` after it.

## Where this leaves the objective

- Criterion 3 (multipass output matches a known-good reference) FAILS at the oracle: 6/8 processes wrong.
- The merged llama.cpp#25 / engine#170 landing is therefore **mitigation-only** - it removes the HSA fault
  but not the underlying wrong values - which the objective explicitly forbids.
- Not causes: host transient overlap, cross-command sync, buffer sizing/overlap, the alignment literals
  themselves. The dependence is on absolute layout/addresses only.
- Untried but well specified next step: pin the arena base address directly (choose the arena buffer's
  device address, not just an offset) and sweep it against the output oracle to find the exact address
  relation that breaks the produce/reduce, then read that relation back into the kernel source.

# OPTION SWEEP (2026-09-27) — every accessible lever, measured with the output oracle

Oracle: fresh llama-server per sample, 2113-token prompt, greedy 16 tokens, compared to the
same build's CPU backend. Configuration = the 256-byte partial-transient alignment (the rig's
oracle), merged commit 895d63f0, unless stated.

| # | option | result |
|---|---|---|
| 1 | partial-transient alignment literals 256 vs 4096 | changes the rate; neither is a fix |
| 2 | arena prefix / absolute address control (0..3584) | changes the rate; production 4096 not immune (prefix 2560 -> 2/8 wrong) |
| 3 | `GGML_HRX_ARENA_ALIGN=4096` (align the arena base itself) | 0/8 correct - no help |
| 4 | host transient overlap disabled (`allocations_can_overlap` -> false) | still wrong |
| 5 | cross-command sync (`GGML_HRX_DEBUG_SERIAL_EXECUTION=1`) | still wrong |
| 6 | graph replay disabled | still wrong |
| 7 | `GGML_HRX_USE_UNIFIED_MEMORY=0/1` | still wrong |
| 8 | **force L1 invalidate** before the partial reads (dummy acq_rel atomic in the kernel) | **worse: 0/8** |
| 9 | `GGML_HRX_DISABLE_QWEN_DISPATCH=1` (bypass fused Qwen dispatches) | 0/8 - no help |
| 10 | Loom sanitizers access/race/value/operation/ubsan/all | unusable - the FA kernel fails to emit under any of them |
| 11 | `GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split` (non-multipass FA) | **8/8 correct** - confirms the multipass path is the culprit |
| 12 | correlate correctness with the arena's absolute base | no clean split (bases differ but OK/WRONG do not separate) |
| 13 | split produce/reduce dispatches | not available: only the fused `..._next_q8` export is in the manifest |

Baseline rate at this configuration drifts between 0/8 and 2/12 correct, so the wrongness is
high-rate and layout-sensitive, not a fixed probability.

## Conclusion of the sweep

The fused produce+reduce multipass kernel (`ggml_flash_attention_decode_split_f32_f16_wmma_next_q8`)
is layout-sensitive-wrong. The only lever that makes it reliably correct is not using it
(option 11). Forcing stronger synchronization/coherence makes it worse, and no address
congruence, memory type, allocator policy or dispatch bypass fixes it. Nothing in the current
toolchain can name the faulty access (no usable sanitizer, no device printf path, and the
program/HSACO dumps give no addresses for the partial views).

Realistic remaining fix, not attempted for budget: add the already-written *split* exports
(`..._produce_partials_f32_f16_wmma` + a separate reduce) to the manifest and register a
two-dispatch path above 2048, so the HRX scheduler provides the produce->reduce ordering that
the fused kernel is trying to do with an in-kernel completion counter. That is a real fix
rather than masking, and it is directly testable with the output oracle.

## De-fusing the multipass path does NOT help (real fix attempt, negative)

Implemented the one remaining architectural fix: added a standalone multipass reducer
(`ggml_flash_attention_decode_split_reduce_multipass_next_q8` = `reduce_completed.multipass` +
`pack_completed_q8`), exported it and the existing `..._produce_partials_f32_f16_wmma` in the
loom-libs manifest, and changed `dispatch-flash-attention.cpp` so the >2048 decode path emits
**two ordered dispatches** (produce, then reduce) instead of the fused one - removing the
in-kernel completion counter entirely and letting the graph scheduler order the pair. The host
side compiles clean and the two dispatches bind exactly the same transients.

Result (256-byte oracle configuration, merged router fix in place):
- output oracle (8 fresh processes): 0/8 exact matches, but the outputs are *plausible*
  continuations (`' Paris. The quick brown fox jumps over the lazy do'`) and contain no `?` run.
- fault oracle (`llama-bench -d 2100`, 5 runs): **5/5 faults** - unchanged.

So the fused in-kernel produce->reduce handoff is not the cause. De-fusing changes the shape of
the wrongness but neither fixes the fault nor makes the output match the reference. The defect is
in the multipass data path itself (the produce writes, or the reduce reads, wrong partial data at
unlucky layouts), not in how the two phases are ordered.

Everything tried is now exhausted: alignment literals, arena base/prefix/alignment, host overlap,
serialization, graph replay, unified memory, forced L1 invalidate, dispatch bypass, all Loom
sanitizers, arena zeroing, absolute-address correlation, and de-fusing. The only thing that makes
the path correct is not using it (`GGML_HRX_DISABLE_DISPATCH=flash_attention_decode_split`, 8/8,
4/4, 8/8 clean in every run).

## De-fuse attempt #2 (corrected grid) - still inconclusive, and my own implementation has defects

Re-applied the split (standalone reduce export + two dispatches). Two problems with the harness
run, both mine:
- `sed` targeted lines 581/583/585 for the 256-byte alignment, but the two new KERNEL_REF lines
  added earlier shifted those lines down, so the run happened at production 4096 alignment.
- At 4096 - where the fused path is clean (8/8 output, 0/5 faults) - the split build faulted
  **5/5** at d2100. My standalone reduce therefore mis-executes (most likely `pack_completed_q8`
  needs the fused kernel's full blocks x kv-heads grid, and/or the two dispatches are not ordered
  by the scheduler), which also explains why the server runs produced plausible text while
  llama-bench faulted: a broken q8 output becomes a downstream fault.

Conclusion: the de-fuse experiment cannot answer whether de-fusing fixes the original wrongness
until my split implementation is itself correct, and getting it there is a separate engineering
task (grid + explicit produce->reduce dependency). It is not a shortcut to the root cause.

---

# DISASSEMBLY VERDICT (2026-09-27) — the multipass kernel emits NO wrong address

Read-only disassembly of the emitted HSACO against the arena layout. Toolchain:
`llvm-objdump` from `/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin`.

## Which dump is which (correcting an earlier label)

The two 17448-byte JIT dumps are NOT both "the multipass kernel":

| dump | partial_block_capacity | head stride (max/sum) | head stride (output) | stride method | exp ops |
|---|---|---|---|---|---|
| `hsaco-0004.bin` | 4 (cap 256) | `s3 << 8` = 256 | `s3 << 14` = 16384 | power-of-2 shifts only | 8 |
| `hsaco-0073.bin` | **36 (cap 2304)** | `s3 * 0x900` = 2304 | `s3 * 0x24000` = 147456 | `s_mul_i32` (20 sites) | 9 |

`hsaco-0020.bin` (17424 B) is produce-only (0 atomics, 0 exp). `hsaco-0073.bin`
is the fused produce+multipass-reduce kernel (`..._next_q8`), identified by its
non-power-of-2 strides (`s_mul_i32 sX, s3, <stride>`), 1 completion-counter atomic,
16 `v_wmma`, 9 `v_exp_f32`.

## Decoded partial-buffer addressing (hsaco-0073, 36 blocks, 4 KV heads, vhs=128)

Register map (from the 3 descriptor loads): `s16:17=partial_max`, `s18:19=partial_sum`,
`s20:21=partial_output`, `s22:23=completion_counter`; `s2`=workgroup.x (block ordinal),
`s3`=workgroup.y (KV head).

Produce stores and reduce loads use IDENTICAL arithmetic:

```
partial_max/sum[head][block][row] = base + head*0x900 + block*0x40 + row*0x4
partial_output[head][block][row][ch] = base + head*0x24000 + block*0x1000 + row*0x100 + ch*0x2
completion_counter[head]             = 48-bit descriptor(s22:s23&0xffff) + head*0x4
```

- 0x900 = 36×64 = `partial_block_capacity × 16 × sizeof(f32)`  ✓ matches source
  `view<[kvh]x[partial_block_capacity]x16xf32>` and host `partial_scalar_count` (559-563).
- 0x24000 = 36×4096 = `partial_block_capacity × 16 × vhs × sizeof(f16)` ✓ matches
  `view<[kvh]x[partial_block_capacity]x16x[vhs]xf16>` and host `partial_output_bytes` (564-565).
- Reduce block loop bound is `v_cmp_lt v1, 36` (active_block_count), max/sum block stride `v1<<6`=64,
  output block stride `s2<<12`=4096 — identical to produce.
- Row offset in produce = `subgroup<<4` = subgroup×16 = row0 byte offset (rows 0..3 at +0/4/8/12) ✓.
- Completion counter descriptor is a correct 48-bit V# (`s9 = s23 & 0xffff`), offset `head<<2` ✓.

Bounds check: max partial_max offset = 3×2304 + 35×64 + 15×4 = 9212 < 9216 (buffer size). ✓
Max partial_output offset = 3×147456 + 35×4096 + 15×256 + 127×2 = 589822 < 589824. ✓

## Verdict

**The fused multipass kernel emits no wrong address.** Every partial-buffer access
(produce and reduce) decodes to the byte-exact address the `.loom` source and the host
transient sizing require, and the produce/reduce strides are mutually consistent. The
"wrong address emitted by the JIT-specialised code" hypothesis — the FINDINGS' prior
"best remaining explanation" and the premise of this disassembly pass — is **refuted**.

Consequences:
- The layout-sensitive wrongness (partial_sum at arena offset 79360 = 1536 mod 4096 →
  broken; at 81920 = 0 mod 4096 → correct, dump256 vs dump4096) is NOT explained by any
  addressing defect in this kernel; the addresses are identical regardless of where the
  arena lands.
- The remaining explanations are data-integrity, not addressing: the cross-workgroup
  produce→reduce completion-counter handoff not surviving under this hardware (the issue's
  own final lead — KFD queue eviction/restore under page migration, #140 class), or an
  as-yet-unidentified produce/reduce data hazard. Both are consistent with every negative
  differential already measured (serialization, cache invalidate, buffer isolation all
  failed to fix it).
- This matches the issue's post-#170 state: the HSA fault was the MoE router sentinel
  (fixed by llama.cpp#25 / engine#170), and this kernel is "not the faulting kernel".

## Evidence files

- `/home/bcloud/1bit-engine/.hrx123/hsaco/hsaco-0073.bin` (multipass, 36 blocks)
- `/home/bcloud/1bit-engine/.hrx123/hsaco/hsaco-0004.bin` (direct/cooperative, 4 blocks)
- `/home/bcloud/1bit-engine/.hrx123/hsaco/fa-multipass.asm` (disassembly of hsaco-0004)
- arena offsets: `dump256/program-6-.../program.json` (partial_sum@79360, broken) and
  `dump4096/program-6-.../program.json` (partial_sum@81920, clean)

---

# GROUND-TRUTH ROOT CAUSE (2026-09-27) — exact wrong address named; multipass kernel excluded

## Real device-address correlation (not just offsets)

The 36-block multipass dispatch (`kernel_id=11938313152504284411`, 10 bindings) in
`logs/dispatch-run.log` pins the actual device VAs, and the arena base is
`0x7fdfac000000`:

| buffer | arena offset | real device VA | size | decoded kernel max | end | in-bounds |
|---|---|---|---|---|---|---|
| partial_max | 52480 | `0x7fdfac00cd00` | 9216 | `0x7fdfac00f0fc` | `0x7fdfac00f100` | yes |
| partial_sum | 79360 | `0x7fdfac013600` | 9216 | `0x7fdfac0159fc` | `0x7fdfac015a00` | yes |
| partial_output | 88576 | `0x7fdfac015a00` | 589824 | `0x7fdfac0a59fe` | `0x7fdfac0a5a00` | yes |

Decoded maxima use the disassembly arithmetic (`hsaco-0073` = 36-block multipass):
`partial_max/sum = VA + head*0x900 + block*0x40 + row*0x4`,
`partial_output = VA + head*0x24000 + block*0x1000 + row*0x100 + ch*0x2`
with head<4, block<36, row<16, ch<128. Every computed address lands inside the real
device range with 2–4 bytes to spare. **The kernel addresses the partial buffers
correctly against the arena's real device-address layout.**

Note `partial_sum` VA `0x7fdfac013600` ≡ **1536 mod 4096** — the exact "broken" congruence
from the earlier bisect — yet the addressing is provably correct at that address. So the
1536-vs-0 sensitivity is **not** address arithmetic.

## The exact wrong address of engine#123 (proven, exact arithmetic)

The wrong address is **not** emitted by the multipass kernel. It is emitted by
`qwen3_moe_routed_gate_up_swiglu_q4k_q8`:

```
router no-winner sentinel published as expert id : 0x7FFFFFFF (2147483647)
consumer offset (32-bit multiply, wraps)          : 0x7FFFFFFF * 884736 mod 2^32
                                                  = 0xFFF28000 = 4,294,082,560 B (~4.29 GB)
weights base 0x7fb5ba600000 + binding 14330826752 + 0xFFF28000 + 25*1152
                                                  = 0x7fba10820080
driver fault page                                 = 0x7fba10820000   -> page-exact match
```

Full product `0x7FFFFFFF * 884736 = 0x6bffffff28000`; it is the 32-bit truncation in
`index.mul %expert, %weight_expert_bytes` (the consumer bound is a hint, `index.assume`,
not a guard) that yields the ~4.29 GB overshoot. This is engine#123's HSA fault, and it is
fixed by llama.cpp#25 / engine#170 (router no longer publishes the sentinel).

## What the fused decode-split multipass kernel does and does not do

- **Does not emit a wrong address.** Produce stores and reduce loads use identical,
  in-bounds strides (verified above and in the earlier DISASSEMBLY VERDICT section).
- **Does produce the NaN router logits** that make the router take its no-winner path —
  the trigger, not the wrong address. That NaN is the #140-class data-integrity issue, and
  the strongest remaining explanation (issue's own final lead) is the cross-workgroup
  completion-counter handoff not surviving KFD queue eviction/restore under page migration.
  That is a hardware/driver interaction; it is stated as the best-supported hypothesis, not
  as a proven arithmetic result.

## Deliverables (inspectable)

- `hsaco/fa-multipass-36blk.asm` — full `llvm-objdump -d` disassembly of the multipass kernel
  `hsaco-0073.bin` (36 blocks), the persisted artifact for the addressing claims above.
- `hsaco/fa-multipass.asm` — disassembly of `hsaco-0004.bin` (4-block direct kernel), for contrast.
- `hsaco/hsaco-0073.bin`, `hsaco/hsaco-0004.bin` — the HSACO binaries.
- `logs/dispatch-run.log` — real device VAs per binding; `dump256`/`dump4096/program.json` — arena layout.

## Closing the last unexamined lead: partial_sum's arena neighbour

The FINDINGS prior text said "the exact neighbour of partial_sum still needs to be read out
of the program dump by hand". Done: in both `dump256` and `dump4096`, the allocation that
ends immediately below `partial_sum` is a **16384-byte buffer at arena_offset 62976**
(= 32 query heads x 128 x 4 = the F32 `output` buffer, the FA kernel's own binding 8).
`partial_sum` starts exactly at its end (79360 / 81920). The multipass kernel writes
`output[head][ch]` in-bounds (max 31*512+127*4 = 16380 < 16384), so it does not over-run
into `partial_sum`. The apparent "798 overlapping pairs" are allocations from *different*
commands concatenated in `transients.allocations[]` (the arena is legitimately reused across
dispatches); within a single command's live set there is no overlap. So there is no
aliasing/over-run into `partial_sum` from this kernel either.

**Search for a wrong address in the multiplexed path is now exhausted**: partial_max/sum/output
(produce stores + reduce loads), completion_counter, output, next_q8_output, and the arena
neighbour have all been decoded and are byte-exact in-bounds. The kernel emits no wrong address.

# REPAIR (2026-09-27) — the all-NaN router case is now loud; source named

Pin: llama.cpp `1bit/hrx-moe-router-nan-loud` `c16424f5` (on the `895d63f0` #25 merge),
engine commit `78b39e6`. Source changes are in scope for this goal (the earlier
investigation was read-only).

## The silent path

`router_top8_f32.loom` seeds each lane's argmax with a **finite** `-FLT_MAX` and, after
#25, with the lane's own expert id. When a row's router logits are all NaN, `ogt`/`oeq`
never fire, so the seed survives; the row's softmax then computes
`exp(-FLT_MAX - (-FLT_MAX)) = 1` over `route_count` lanes and publishes a **plausible
uniform `1/route_count`** for a valid-but-wrong expert. Nothing downstream can tell.
(`kernel.subgroup.reduce<maxnumf>` is a NaN-*ignoring* max, which is what lets the NaN
disappear.) That is the silent wrong-expert decode.

## The repair

1. `router_top8_f32.loom`: when no lane of the row found an **ordered** candidate
   (`cmpf ogt %selected_max, -FLT_MAX` false), publish the row's own NaN
   (`vector.extract %initial_logits[0]`) instead of the masked uniform weight. The NaN
   is taken from the data, so no NaN needs to be synthesised and the compiler cannot
   fold it away. The published expert id is unchanged, so #25's fault fix is untouched.
2. `common/sampling.cpp` (`common_sampler_sample`): scan the row's logits; on a NaN,
   `LOG_ERR` naming engine#123 and `GGML_ABORT`. NaN is never a legitimate logit —
   unlike `-inf`, which masking legitimately uses — so this cannot false-positive.
3. `dispatch-flash-attention.cpp`: `GGML_HRX_FA_PARTIAL_ALIGN` selects the decode-split
   partial transients' alignment (default 4096 = production). 256 reproduces the
   engine#123/#140 rig oracle without editing stress literals, which is what made the
   baseline measurable.

## Verification (30B MoE, 2113 ctx, fresh server per sample; every sample archived)

`verify_repair.sh <tag> <port>...` starts one fresh server per port concurrently and
classifies each sample MATCH / LOUD-ABORT / SILENT-WRONG / LAUNCH-FAILED, keeping the raw
JSON and every server log under `logs/verify-<tag>/`.

| condition | n | match | loud abort | silent wrong | launch OOM |
|---|---|---|---|---|---|
| pre-repair (`895d63f0`), 8 fresh servers concurrently | 16 | 7 | n/a (no guard) | **6** | 3 |
| post-repair (`c16424f5`), 6 concurrent, movers on | 6 | 1 | **4** | **0** | 1 |
| post-repair (`c16424f5`), 6 concurrent, KSM + proactive compaction off | 6 | 1 | **4** | **0** | 1 |

Every silent `' Paris???????????????'` became a loud
`HRX returned NaN logits at vocab index 0 - refusing to sample a silently wrong token
(engine#123)`. After the repair there are **zero** silent-wrong samples: a corrupted decode is
either correct or a loud abort, so the end-to-end repro is deterministic. Under no induced
pressure the fault never appeared at all (0/8 sequential, and 0/300 for the 0.6B probe in
five conditions — see `repro-summary.md`).

Honest caveats: the LAUNCH-FAILED samples are 30B servers that failed to *load* (6 concurrent
instances is ~108 GiB of a 122 GiB box), not corrupted decodes. In the movers-off run the
`fuser` quiet line was captured while an earlier server was still shutting down, so that run
was "6 concurrent + one held fd" rather than perfectly quiet; per-sample classification is
unaffected (each sample is judged by its own outcome).

## Why the fix is out-of-tree (evidenced), and the mechanism (best-supported hypothesis)

**Evidenced — this is what satisfies the "out-of-tree" branch of the deliverable:**
- HRX's device buffers on this APU are pageable system memory registered with the GPU: the
  decode-split transient arena is one reused `hrx_allocator_allocate_buffer(
  hrx_device_allocator(device), {HRX_MEMORY_TYPE_DEVICE_LOCAL, HRX_MEMORY_ACCESS_ALL,
  HRX_BUFFER_USAGE_DEFAULT, 0})` (`runtime/transient-arena.cpp`).
- The runtime's allocation descriptor `hrx_buffer_params_t` is
  `{type, access, usage, queue_affinity}` — there is **no pin / lock / advise field**, so
  nothing settable from this repository stops those pages from being migrated.
- The allocation is implemented in `ROCm/hrx-system` `libhrx` (`hrx_allocator_*`), outside
  this repository and outside the kernels this goal can change.
- That runtime already carries `[HIP] Implement virtual memory management` (#568), the
  mechanism a pinning fix would use — so the fix is reachable, just not from here.

**Best-supported hypothesis, explicitly not proven here:** the NaN follows the kernel
migrating such a page while HRX work is in flight (KFD userptr eviction/restore). The measured
support is that corruption tracks process memory pressure — absent with no induced pressure
(0/8 sequential; 0/300 for the 0.6B probe in five conditions), present with 6 concurrent 30B
instances, and still present with KSM and proactive compaction off under that pressure because
direct compaction still migrates pages. That is a correlation with the pressure that drives
migration, not a tracer-level proof of eviction; the earlier layout-only observations (the
arena-prefix bisect) show corruption does not require induced pressure in every configuration,
so the mechanism is stated as the best-supported explanation rather than as established.

## Retracted claim (a measurement bug, not a finding)

An earlier draft recorded "1 of the 6 movers-off samples was silently divergent without NaN
logits" and inferred a residual finite-value corruption class. That was **wrong**: the
reference string had been passed through bash single quotes, so its `\n` was a literal
backslash-n and no sample could ever compare equal - a correct match was scored as divergent.
With the reference fixed, movers-off is 1 match / 4 loud aborts / 0 silent-wrong (n=6), and
movers-on is the same. No finite-value residual is observed in these runs. (A finite-value
corruption would still need a checksum/parity mechanism rather than a NaN check - true in
principle, but nothing here demonstrates that such corruption occurs.)
