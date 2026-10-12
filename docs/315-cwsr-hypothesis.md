# engine#315: a different approach — the CWSR / VGPR-region hypothesis

**Written** 2026-10-12, after an external-research pass. Supersedes the in-repo frame ("the fault
is *computed*, and the ordering/visibility super-family is closed"), not by contradicting the
measurements but by explaining them from outside.

## 0. Why this is a different approach

Every lever the issue falsified was **GPU-side and ggml-side**: coherence, consumer ordering,
AQL barriers, `-fa off`, transient reuse, PM4, the MoE routing tables. All were toggled *inside
one process on one kernel*. Nothing tested the **KFD/amdgpu/HSA boundary** — the kernel↔userspace
contract that governs how compute waves are saved and restored when a queue is preempted.

That is where the external record points, and it is a boundary this issue never varied.

## 1. The mechanism

### 1.1 CWSR is the preemption save/restore path

Strix Halo (gfx1151) supports **Compute Wave Save/Restore (CWSR)**: on queue preemption the
hardware/firmware spills a wave's registers to a per-queue **context save/restore area** and
restores them later. The area is sized by a **userspace-declared** value,
`ctx_save_restore_area_size`, which must be ≥ what the kernel/topology requires
(`topo_dev->node_props.cwsr_size` + debug memory, × `NUM_XCC`).

### 1.2 There was a validation bug, and it is a *size* mismatch

`torvalds/linux d15deafab5d7` ("drm/amdkfd: fix CWSR size check and use userspace size") changes:

```diff
- if (properties->ctx_save_restore_area_size != topo_dev->node_props.cwsr_size) {
-   pr_debug("queue cwsr size 0x%x not equal to node cwsr size 0x%x\n", ...);
+ if (properties->ctx_save_restore_area_size < topo_dev->node_props.cwsr_size) {
+   pr_debug("queue cwsr size 0x%x not sufficient for node cwsr size 0x%x\n", ...);
     err = -EINVAL;

- total_cwsr_size = (topo_dev->node_props.cwsr_size + debug_memory_size) * NUM_XCC(...);
+ total_cwsr_size = (properties->ctx_save_restore_area_size + debug_memory_size) * NUM_XCC(...);
```

So the kernel used to (a) demand **exact** equality and (b) size the buffers from the
**topology** value, ignoring what userspace actually allocated.

### 1.3 The companion userspace change is unmerged, and that is the trap

`hammmmy`, on the freedesktop tracker and again on ROCm#5590:

> "These two patches fix a mismatch between userspace and kernel space on the size of the
> **VGPR region**."
> — `d15deafab5d7` (kernel, **in OEM kernel 6.18-rc6**) and `ROCm/rocm-systems#1807` (**"not merged
> yet"**)

**A half-applied fix is worse than none.** Our box has:

| side | state on strixhalo | evidence |
|---|---|---|
| kernel check | **FIXED** — `queue cwsr size 0x%x **not sufficient for** node cwsr size` is in the built module | `strings` on the decompressed `amdgpu.ko` |
| kernel control-stack check | **UNFIXED** — `queue ctl stack size 0x%x **not equal to** node ctl stack size` | same |
| userspace (`libhsa-runtime64.so.1`) | **unknown / likely pre-#1807** — has `reserved_vgpr_count`, `granulated_workitem_vgpr_count`, no ctx-save-area symbol | `strings` |
| `cwsr_enable` | **1 — CWSR is ON** | `/sys/module/amdgpu/parameters/cwsr_enable` |
| kernel | `7.3.0-rc4-1bit` | `uname -r` |

The kernel now **accepts** a userspace size it previously **rejected**, and then sizes the
save/restore buffers from that userspace value. If userspace still computes the region the old
way (per-context VGPR counts that do not match what the kernel will spill), the area is
**undersized for the wave state it must hold**.

### 1.4 How an undersized save area produces exactly our two faces

That is the part that fits the measurements better than anything else in the record:

- a wave is preempted (eviction) → its registers are spilled into the area;
- restore reads them back;
- if the area is short, the tail of the saved state is **truncated or overwritten**:
  - the truncated part holding **finite** data → wrong arithmetic → **Face B** (finite but wrong
    output, `HTTP 500` on unparseable text);
  - the truncated part holding **exponent/NaN-bearing** data → **Face A** (`count=154880 of
    154880` NaN, whole LM head).

This is the **same two-observables-one-event shape** as the #123/#140 record ("the f32 output was
right and `next_q8` was wrong, and NaN when the stale bytes decoded as NaN") — but the defect is
in the *restore buffer*, not in a kernel barrier. It also explains, for the first time:

- **the GTT/pressure dependence**: eviction and preemption only happen under memory pressure —
  "the window opens when a dispatch is preempted, and any process's KFD queue eviction does
  that, which is why the rate followed those settings" (`docs/hrx.md`);
- **why `-fa off`, barriers, coherence and ordering all did nothing**: none of them changes
  whether a wave is preempted or how much state is spilled;
- **why the corruption is already whole-batch at layer 0**: a truncated restore corrupts the
  wave, not the memory of one tensor, so it cascades immediately;
- **why our arms fault at 97-98 GiB and not below the gate** (run 7: `gtt_start=97G` ×3 faults,
  and 8 clean arms at the same 97-98 GiB — the *occurrence* of preemption is the intermittent
  variable, not the pressure itself).

## 2. The discriminator — one variable, cheap, and pre-registered

`amdgpu.cwsr_enable` is a **kernel cmdline parameter**: `1` (on, current) vs `0` (off). The
community workaround is precisely `amdgpu.cwsr_enable=0`, reported by multiple independent
Strix Halo users to stop this hang class (ROCm#5590, closed 2025-11-16 on exactly that basis).

**Prediction, stated before the run:** setting `amdgpu.cwsr_enable=0` and re-running the *same*
pre-registered campaign will produce **0 faults across ≥60 in-regime arms**, while the same
campaign at `cwsr_enable=1` faults at the observed ~25-30 %.

If instead the fault rate is unchanged at `cwsr_enable=0`, **this hypothesis is dead** and the
fault is not in the wave save/restore path at all — which would be a strong, informative
negative, because CWSR is the only preemption-state mechanism the hardware has.

## 3. Why this is falsifiable where the old frame stalled

The old frame concluded "the fix stays unwritten on purpose … there is nothing left that a
minimal patch could be aimed at." That was correct *within* the frame (all GPU-side levers
tested). This hypothesis names:

- a **specific kernel/userspace contract** (the CWSR area size),
- a **specific patch pair** (`d15deaf` + `rocm-systems#1807`),
- a **specific one-variable toggle** (`cwsr_enable`),
- a **pre-committed prediction** (0/≥60 at `cwsr_enable=0`),
- and a **kill condition** (unchanged rate ⇒ hypothesis dead).

## 4. Verification path (what to actually run)

1. **Record the current state** — `cwsr_enable`, kernel, MES firmware, HSA version, and the
   `ctx_save_restore_area_size` our runtime requests vs `cwsr_size` the topology reports.
   *This is the single most important missing measurement*: if the two already disagree, the
   mechanism is confirmed as a size mismatch before any campaign.
2. **Baseline campaign** at `cwsr_enable=1` (the measured configuration): ≥20 in-regime arms,
   expect ~25-30 % faults — this run already exists as run 7 (11 arms, 3 faults) and can be
   extended.
3. **Treatment campaign** at `cwsr_enable=0`: same harness, same composition, ≥60 in-regime arms,
   pre-registered, stopping rule unchanged.
4. **Report** both with CI and the power statement (60 clean ⇒ ~95 % the true rate is <5 %).
5. **If clean:** the fix is a **forward-only** change — either the pin moves to a kernel carrying
   both patches, or the userspace side takes `rocm-systems#1807`. Confirm
   `git merge-base --is-ancestor <prev> <new>`.
6. **If not clean:** record the negative and the fault is not CWSR; the next discriminant is the
   VGPR-size mismatch measured directly (step 1).

## 5. What this does NOT claim

- It does **not** contradict any existing measurement. Coherence, ordering, barriers and `-fa off`
  remain falsified; the corruption is still "computed" in the sense that the device writes the
  bad bytes — but the *reason* the bytes are bad would be a truncated restore, not the kernels.
- It is **not** yet a mechanism. It becomes one when step 1 measures a size disagreement, or when
  step 3 gives 0/≥60 where step 2 gave ~27 %.
- It does **not** excuse the earlier null: run 7 is still the baseline, and it faulted.

## 6. Evidence / sources

| claim | source |
|---|---|
| CWSR disable stops this hang class on Strix Halo | `ROCm/legacy-rocm-build#5590` (closed 2025-11-16 on `amdgpu.cwsr_enable=0`) |
| VGPR-region mismatch between userspace and kernel; two patches, one unmerged | `hammmmy`, freedesktop `drm/amd#4632` and `#5590` comments |
| the kernel patch and its exact diff | `torvalds/linux d15deafab5d722afb9e2f83c5edcdef9d9d98bd1` |
| userspace companion, unmerged | `ROCm/rocm-systems#1807` |
| MES-firmware hang family on the same SoC | `ROCm/legacy-rocm-build#6165`, freedesktop `drm/amd#5171`, `#5289` |
| our box's CWSR state and module strings | measured on strixhalo 2026-10-12, §1.3 |
