<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The runtime does no cache management at all — so correctness is provided by the platform, and that is why pressure matters

This is the structural fact that ties together the two strongest observations in the whole campaign: the
**trigger's rate contrast** (forcing eviction takes the fault rate from ~5 % to ~95–100 %) and the corruption's
**256-byte granularity**. It closes no candidate by itself, but it makes the surviving ones much more specific.

## The fact

The HRX runtime performs **no CPU-cache management and exposes no coherence control**:

| searched for | found |
|---|---|
| `invalidate` in `binding/common/memory.c` | **nothing** |
| `invalidate` anywhere in libhrx | only **stream-**capture invalidation (`graph.c:2326,2358,2375,2384-2397`) — a graph-capture concept, not a cache one |
| `clflush`, `__builtin_amdgcn`, `dcache`, cache-control | **nothing** |
| a coherence API in `hrx_runtime.h` | none — the only "flush" is `hrx_stream_flush` (line 582), which the header itself describes as *"Use `hrx_stream_synchronize()` to flush and then wait"*, i.e. a **command-stream** flush |

And every host-backed buffer is declared **non-coherent**: `HOST_COHERENT = 0x04` is not consulted on the pool
path, and the register path **hardcodes** `HOST_LOCAL | DEVICE_VISIBLE = 0x52`
(`315-coherence-closed-from-source-both-paths.md`).

So the runtime declares the memory non-coherent **and** never invalidates anything. Those two together would
mean every host read of device-written data is wrong — and they are overwhelmingly right.

## What that forces

**Correctness is being provided by the platform, not by the runtime.** On this part — Strix Halo, an APU where
CPU and GPU share physical DRAM — host and device sit behind the same fabric, so the hardware supplies
visibility the software never asks for. That is why the reads work without a single invalidate in the code.

Two consequences follow, and the first is the important one:

1. **The fault must be whatever makes that implicit, platform-provided visibility *usually* hold and
   *sometimes* not.** It cannot be "the memory is non-coherent" (a constant — it would break every read, not
   one arm in twenty), and it cannot be "a missing invalidate" (there is no invalidate anywhere, so its
   absence is not variable). It has to be a mechanism that is exercised by *conditions*, which is exactly what
   the trigger is.
2. **That is precisely the trigger's signature.** Forced eviction moves the rate from ~5 % to ~95–100 %
   (`docs/315-mechanism.md`; 28/29 eviction arms vs 4/57 controls, Fisher p ≈ 1.3e-17), and the quiet cell
   occasionally faults too. `mprotect` churn drives **page reclaim and cache/TLB pressure**, so the trigger is
   a *pressure* generator — and a mechanism whose correctness rests on the platform's caches is exactly the
   kind of mechanism that fails under pressure and recovers when it eases.

## What the 256-byte granularity looks like under this reading

256 bytes is **four 64-byte cache lines**. The measured granularity is 64 *elements* of f32 = 256 B, which is
**not** a single cache line (16 elements) but is an exact multiple of it:

| reading | unit | is 256 B natural? |
|---|---|---|
| **platform cache/coherency granule** | 64 B line; 256 B = 4 lines, i.e. a sector or a prefetch/attribution granule | **yes** — and it is the only reading that needs no wave or table concept |
| one wave64 | 64 lanes × 4 B | yes |
| the record's MoE-table block | 256 B per the record | yes |

So this **adds a third candidate** rather than choosing between them — but it is the only one that is
*implied* by the structure just established (no runtime cache management + hardware-provided visibility +
pressure-sensitive failure). The previously-favoured wave64 reading is not supported by anything measured;
it was favoured because a wave is 256 bytes, which is a coincidence of arithmetic shared by all three.

## The experiment that would separate them

The three readings make different predictions about **where** the 256-byte boundary falls relative to the
buffer, and one of them makes a prediction the others do not:

- *cache/coherency granule*: the corrupted region should align to the **physical** granule — i.e. its start
  should be a multiple of 256 B **from the buffer's base, as allocated**, and should move if the base moves.
  The allocation base is recoverable (`hrx_buffer_get_device_ptr`, `hrx_buffer_map`).
- *wave64*: the region aligns to the **tensor's** row/wave index — it should track `ne0` and the row, and be
  invariant to the base address.
- *MoE-table block*: it should appear only in tensors that index those tables, which the family split already
  contradicts (`l_out` and `node_2948` align just as tightly as `ffn_moe_down`).

**So the discriminating measurement is the buffer's base address against the corruption's start**, and the
existing `[hrx-buf]`/`build` traces plus `hrx_buffer_get_device_ptr` can supply it without a new instrument.

**Nothing here names the mechanism.** It establishes where correctness comes from, why the trigger is a
pressure generator rather than a data path, and which single measurement separates the remaining three
readings — and it retires "missing invalidate" as a *variable*, since there is no invalidate to be missing.
