# After the size explanation: three surviving explanations, and their discriminating tests

**Date** 2026-10-12. Supersedes the causal story in `315-cwsr-hypothesis.md`, which is
falsified (`315-cwsr-sizes-excluded.md`).

## Where we are

- **Signal:** `cwsr_enable=0` → 0 faults in 11 in-regime arms, every arm at `gtt_start=97G`,
  against a 26% multi-session baseline (34 faulting / 131 in-regime arms) at
  `cwsr_enable=1`. Binomial p = 0.030; Fisher p = 0.107; rule-of-three bound 27.3% does not
  exclude 26%. **Suggestive, not significant.**
- **Falsified:** the area is *undersized*. Both sizes our runtime declares for gfx1151 are
  correct — VGPR 0x60000 (384 KB, the 1.5× value from `#1807`, present) and SGPR 0x4000
  (16 KB, standard). The kernel's `cwsr_size` check is also the fixed `<` form, so a
  mismatch can no longer silently under-allocate. **Size is excluded.**

That leaves exactly three explanations, each with a cheap discriminating test. They are
listed here *before* the 60-arm result so the next step is not chosen after the fact.

## H1 — fresh-boot effect (the confound)

The reboot changed more than the one cmdline token: run 7 ran on a long-uptime box with
fragmented device memory, the `cwsr=0` run on a fresh boot.

**Falsified by:** a **fresh boot with `cwsr_enable=1`**, running the identical campaign
(same PAIRS, same hour of day, same composition). If that also gives ~0 faults, CWSR is not
the cause and the 11-arm signal was a boot effect. If it gives ~26%, H1 is dead.
**Cost:** one reboot + ~50 min.

## H2 — CWSR save/restore is not bit-exact under preemption on gfx1151

Not a size bug: a hardware/firmware characteristic of the spill itself. A preempted wave
resumes with subtly wrong state, and what the damaged state holds decides the observable —
finite → Face B, exponent/NaN → Face A.

This is the explanation that fits **everything** measured, including what nothing else did:

- the GTT dependence: preemption happens under memory pressure;
- why coherence, ordering, AQL barriers, `-fa off`, transient reuse, PM4 and the MoE routing
  levers all did nothing — none of them changes whether a wave is saved and restored;
- why the corruption is already whole-batch at layer 0 — a damaged wave corrupts its whole
  tile, not one tensor's memory;
- why run 7 faulted 3× at `gtt_start=97G` while 8 clean arms sat at the same 98 GiB — the
  *occurrence* of preemption is the intermittent variable, not the pressure level.

**Discriminating test (cheap, no campaign):** count CWSR preemptions per arm and correlate
with faults. If faulting arms have preemptions and clean arms do not, that is direct
evidence. Candidate observables: the `hqds`/`mqds` debugfs nodes, the CWSR control-register
writes visible in a ctrlcode-style trace, or an eBPF probe on the MES/queue path.
**Cost:** instrumentation, no reboot.

## H3 — the partial kernel-side fix matters

Our module carries **one fixed and one unfixed** size check:

```
"queue cwsr size ... not sufficient for"   FIXED   (d15deaf)
"queue ctl stack size ... not equal to"    UNFIXED
```

Both sizes our runtime declares happen to be correct, so the `!=` check passes today — but
it is a latent constraint: any future change that makes userspace declare a *larger* control
stack would be rejected outright (EINVAL, loud) rather than accepted. That is a robustness
gap, not (on current values) a defect.

**Falsified by:** anything above. It is listed for completeness and because it is the only
actual code asymmetry found in the kernel path.

## What is NOT a candidate

- arena knobs, transient reuse — falsified earlier in the issue's record;
- MoE routing / partition tables — rebuilt in-plan, falsified;
- the routing-tensor and by-name locator routes — withdrawn (reused arena slots);
- **area size on either the VGPR or SGPR half — excluded today;**
- stream ordering / visibility / barriers — closed by run *and* at source.

## The order of the next steps

1. finish the 60-arm `cwsr=0` campaign (running) — establishes the rate, not the cause;
2. **H1 control**: fresh boot, `cwsr_enable=1`, identical campaign. Without this, step 1 is
   not attributable to CWSR at all;
3. only if H1 is survived, **H2 instrumentation** to catch the preemption directly.

Doing step 3 before step 2 would risk instrumenting a boot effect.

## Control sizing (so the H1 control is not accidentally over-sized)

The 60-arm bar in the objective applies to the **fix** campaign, not to the control. The
control only has to distinguish "the 26% baseline rate persists on a fresh boot" from "the
fresh boot suppressed it".

At the baseline's 26%, with `n` arms the expected fault count and P(0 faults) are:

| n | expected faults | P(0 faults \| 26%) |
|---|---|---|
| 11 (what we have at cwsr=0) | 2.9 | 0.030 |
| **20** | **5.2** | **0.002** |
| 30 | 7.8 | 0.0001 |

So **20 arms is sufficient for the control**: it either shows ~5 faults (baseline persists,
H1 dead, the cwsr=0 signal is real) or 0 faults (fresh boot suppressed it, H1 confirmed).
Running the control to 60 arms would cost ~3 extra hours for no additional discriminating
power.

Recorded so the control is sized to its question rather than copied from the fix campaign's
bar.

## H4 (new): the firmware save/restore list sizes the context for standard GFX11, not gfx1151's 1.5x VGPR layout

Found 2026-10-12 while waiting, and it is the first hypothesis that names a *specific
firmware artifact* rather than a userspace value.

The driver reads a **save/restore list** out of firmware and logs its geometry:

```
save_restore_list_srm_ucode_ver: %u
save_restore_list_srm_feature_ver: %u
save_restore_list_cntl_size_bytes %u          <- the CONTROL stack size
save_restore_list_cntl_offset_bytes: %u
save_restore_list_gpm_ucode_ver: %u
save_restore_list_gpm_feature_ver: %u
save_restore_list_gpm_offset_bytes: %u
RLC_SRM_DATA_RAM / RLC_SRM_ADDR_RAM           <- RLC holds the SRM blob in its own RAM
```

Reading the pieces together:

- `node_props.ctl_stack_size` — the value our kernel's **UNFIXED `!=` check** compares against
  — is exactly what `save_restore_list_cntl_size_bytes` supplies. So the control-stack size is
  **firmware-declared**, not computed by the kernel.
- The patch pair everyone cites (`d15deaf` kernel + `#1807` userspace) aligns the **VGPR /
  cwsr** area only. Nothing in that pair touches the **control stack**, whose only check in
  our module is still the exact-equality form.
- Spill/restore is executed by the **RLC firmware** using that list.

**The hypothesis:** the gfx1151 firmware's save/restore list is built for standard GFX11
context geometry, not for gfx1151's **1.5× VGPR** memory. The kernel and userspace then agree
on a control-stack size that *passes* the `!=` check (so nothing is rejected loudly) while the
RLC firmware's actual spill needs more room than the declared list provides. A wave preempted
under pressure overruns the declared region and restores truncated state.

That would make the defect **firmware-side**, which explains why:

- every userspace and kernel-side lever returned null — none of them can change what the RLC
  spills;
- the size checks all pass — the declared values are mutually consistent, just not
  sufficient for the hardware;
- `cwsr_enable=0` helps — it bypasses the RLC SRM path entirely.

**How to falsify it (bounded RE, no GPU needed):**

1. extract `save_restore_list_cntl_size_bytes`, `..._offset_bytes` and the SRM ucode version
   the driver logs at init for gfx1151 (raise the log level, or read them out of the RLC SRM
   RAM region);
2. compare against another GFX11 part's list (e.g. gfx1100) where the fault is not reported —
   if gfx1151's control-stack size is the same as a 1.0× VGPR part's, the list is wrong for
   the 1.5× layout;
3. inspect `gc_11_0_0_rlc.bin` / `gc_11_0_0_mes*.bin` (present at `/lib/firmware/amdgpu/`) for
   the SRM blob to see whether the geometry is encoded there.

Step 2 alone is decisive and cheap: **same declared control-stack size on a 1.5× VGPR part as
on a 1.0× part would confirm the list is not gfx1151-aware.**

If it instead shows a proportionally larger value, H4 is dead and the firmware is aware of the
layout.

Recorded as a hypothesis with a bounded test, not as a finding.

### H4 first measurement (kernel-side values, previously unknown)

Read from `/sys/class/kfd/kfd/topology/nodes/1/properties`:

```
cwsr_size       19185664   (18.3 MiB)
ctl_stack_size     16384   (16 KiB)
```

Three consequences, and they cut against H4:

1. **`ctl_stack_size = 16384` exactly equals libhsakmt's SGPR return (`0x4000` = 16384).**
   So the kernel's `!=` check compares 16384 against 16384 and passes. That is consistent
   with the arms running at all, and it means the control stack is **not** mis-sized today —
   kernel and userspace agree.
2. Therefore the control-stack half is *coherent*: whatever the firmware's save/restore list
   supplied, userspace declares a matching value. **H4's control-stack variant is weakened** —
   a mismatch would have failed loudly with EINVAL, and it does not.
3. `cwsr_size = 19185664` is **not a clean multiple** of the per-CU VGPR value:
   `19185664 / 393216 = 48.79…`. So the node's cwsr_size is not simply
   `vgpr_per_cu × n_cu`, and the arithmetic behind it is not reconstructible from these two
   numbers alone. That is a gap in my understanding, not evidence of a bug.

What H4 would still need to survive: a **comparison part**. If a standard-GFX11 GPU (1.0×
VGPR) reports the same `ctl_stack_size` *and* a `cwsr_size` whose ratio to its VGPR size
matches gfx1151's, the firmware is layout-aware and H4 is dead. Without a second part on this
box, H4 stays untested — and it is now the *weakest* of the four hypotheses, because the one
number it predicts should be wrong (the control-stack size) is coherent.

**Net effect on the hypothesis set:** H4 down-weighted. H1 (fresh-boot confound) remains the
gate that must be passed before any of H2/H4 can be assessed, and it is still unrun because it
requires a reboot.

## H2 refined: it is *mid-wave preemption* that matters, not the spill correctness

The kernel's own parameter documentation settles a point I had been treating loosely:

> `cwsr_enable` *(int)* CWSR (compute wave store and resume) **allows the GPU to preempt shader
> execution in the middle of a compute wave**. Default is 1 to enable this feature. Setting 0
> disables it. — `docs.kernel.org/gpu/amdgpu/module-parameters.html`

So `amdgpu.cwsr_enable=0` does **not** merely skip saving and restoring a wave. It removes the
hardware's ability to preempt a wave mid-execution at all. A dispatch either runs to completion
or waits.

**That reframes the mechanism, and it fits better than "the spill is wrong":**

> Under memory pressure a wave is preempted mid-execution; the buffers that wave is actively
> reading are then evicted and/or migrated by TTM/amdgpu. When the wave resumes — or as it
> continues — it reads memory that has moved or been recycled, and the damaged values surface
> as Face A or Face B depending on what those bytes now hold.

This is the same *window* the engine's own #123/#140 record describes — *"the window opens when
a dispatch is preempted, and any process's KFD queue eviction (page compaction, KSM, memory
pressure) does that, which is why the rate followed those memory-pressure settings"* — generalised
from a barrier bug to preemption itself.

Why it explains everything that was measured:

| observation | explanation |
|---|---|
| GTT/pressure dependence | preemption-plus-eviction needs pressure to occur at all |
| coherence, ordering, AQL barriers, `-fa off`, transient reuse, PM4, MoE routing all returned null | none of them changes **whether a wave is preempted mid-execution**; they all operate *within* execution |
| corruption already whole-batch at layer 0, cascading by layer 1 | a preempted-and-resumed wave corrupts its own tile, which cascades immediately |
| run 7 faulted 3× at `gtt_start=97G` while 8 clean arms sat at 98 GiB | the **occurrence** of a mid-wave preemption is the intermittent variable, not the pressure level |
| both faces, never in the same arm | what the moved bytes now hold decides finite-but-wrong (B) or NaN-bearing (A) |

**And it no longer depends on the size defect** — which is important, because that defect is
proven fixed here (`315-cwsr-size-excluded-PROOF.md`). A preemption-window mechanism is
compatible with a fully patched CWSR area: the area can be perfectly sized and the *memory the
wave reads* can still be moved underneath it.

**How this differs from H1, and why H1 still gates it.** H1 says the signal is a fresh-boot
effect. H2 says it is preemption. They predict the *same* result for the running campaign
(0 faults at `cwsr=0`), so the campaign cannot separate them. **Only the fresh-boot `cwsr=1`
control separates them**: if a fresh boot still faults ~26% with CWSR on, H1 is dead and
preemption is the difference. If it does not, H1 is confirmed.

**A cheap secondary discriminator worth noting** (no reboot, but needs the box free): count
**mid-wave preemptions** per arm — via the MES/queue trace, or by comparing per-arm eviction
counts between faulting and clean arms (the `server.pid` capture in `315-pid-capture.md`
exists for exactly this and is still unapplied). If faulting arms show preemptions and clean
arms at the same GTT do not, H2 gains direct support.

## H1 quantified: the two compared runs differ by ~32 hours of uptime

Measured 2026-10-12:

| | uptime when the arms ran | faults |
|---|---|---|
| run 7 (`cwsr_enable=1`) | **~32.5 h** — box booted 2026-10-11 05:33 | 3 / 11 |
| campaign `315cwsr60` (`cwsr_enable=0`) | started **692 s (~11.5 min)** after the 15:15:40 boot | 0 / 2 so far |

So the two arms of the comparison are not just "CWSR on vs off" — they are also "32-hour-old
box vs 11-minute-old box". H1 is not hypothetical; it is quantified, and the gap is large.

**This gives the running campaign a testable prediction.** It runs ~6 h, so it ends at roughly
**7 h post-boot**:

- if the fault rate rises with uptime, the campaign should show faults **late**, not uniformly
  — and that would support H1 over H2;
- if it stays at 0 across all ~66 arms spanning 11 min → 7 h of uptime, that argues the rate
  does not depend on uptime over that window, which **weakens H1** and leaves H2 standing.

Either outcome is informative, and neither was designed in — I am recording the prediction now,
before the result, so it cannot be read post hoc.

**Note the control's design is already right for this:** the H1 control is a *fresh* boot with
`cwsr_enable=1`, i.e. ~1 h uptime, which matches the campaign's condition rather than run 7's.
That is the correct comparison; run 7 alone is not.

## Side quests while the campaign ran (2026-10-12)

### The H1 control's guard is verified

Dry-ran `315-cwsr-control.sh` with only the destructive step neutered (the final
`systemctl reboot` replaced by an echo). It correctly refused:

```
=== H1 control: fresh boot with cwsr_enable=1 ===
  current cwsr_enable : 0
  drop-in present     : yes
  campaign to mirror  : 315-evict-trace.sh 48
ABORT: a 315 campaign is still running. Wait for it - this would kill its evidence.
```

So it cannot accidentally destroy a running campaign's evidence — the property that matters most
for a script that ends in a reboot.

### H2 has NO documented upstream backing (attempted to find one, and failed)

Three targeted searches for a gfx1151 *mid-wave preemption* corruption beyond the VGPR-size
defect returned **nothing relevant** — only the generic kernel-changelog noise. Specifically
searched for:

- gfx1151 mid-wave preemption compute corruption beyond the VGPR size fix;
- gfx1151 CWSR preemption data corruption 2026 kernel fix missing;
- amdkfd preemption wave-state corruption gfx11.

The gfx1151 CWSR record that *does* exist is the size defect — which this session proved is
**fully fixed on our box** (`315-cwsr-size-excluded-PROOF.md`). Nothing documents a second,
execution-level defect.

**That absence is evidence, and it points the same way as everything else in this session:**

- the one documented gfx1151 CWSR corruption is fixed here;
- no second one is documented anywhere;
- every prior lever returned null;
- the only new fact is that `cwsr_enable=0` coincided with a **fresh boot**.

So H2 now rests on a mechanism with no upstream support and no direct measurement, while H1
(fresh-boot effect) explains the signal with a single, already-quantified confound (~32.5 h
uptime vs 692 s). The honest ranking after this pass: **H1 is the leading explanation, not H2.**

I am not deleting H2 — it is the only candidate that explains the GTT dependence — but it should
not be presented as the likely mechanism on current evidence, and the H1 control is the only
thing that can settle it.

### Can H1 be tested retroactively? Attempted — no, and here is why that matters

I tried to settle H1 from existing data instead of a fresh-boot control: each KFD SMI stream
records `boot_ns` (CLOCK_BOOTTIME at the event), so in principle a session's boot wall-clock is
recoverable as `stream_start_wall − first_boot_ns`, and every arm's uptime follows from its
directory mtime.

**The method works only when the stream's START wall time is known.** For a completed stream its
file mtime is the *last* write, not the first, so `mtime − first_boot_ns` is wrong by the whole
stream duration. Using the campaign's own start timestamp instead recovers the true boot
(the `cwsr60` stream's 52.5 min offset against a 16:08:06 campaign start gives 15:15:40, which is
exactly the boot I measured) — but that requires a per-run start timestamp.

Streams available and what they cover:

| stream | session | usable |
|---|---|---|
| `315evict` | 10-11, boot ~05:33 | yes, but it is run 1 (the truncated 2-arm run) |
| `315evict2` | 10-11 05:44 | yes — run 1's continuation |
| `315evict6` | 10-12 14:04 | only 2 arms (self-deadlocked) |
| `315evict7` | 10-12, boot 10-11 05:33 (~32.5 h uptime) | yes — the faulting baseline |
| `315cwsr0`, `315cwsr60` | 10-12, boot 15:15:40 | yes — the treatment |

**The decisive problem: the historical faulting arms predate the tracer.** The 34 in-regime
faulting arms of 131 accumulated across 10-09…10-11 — the multi-session baseline H1 would need to
be tested against — were run before any SMI stream existed, and no boot times survive for those
sessions (no `wtmp`/`last reboot` on this box). So there is no retroactive H1 test.

**Consequence, stated plainly:** H1 can only be settled by a *new* fresh-boot run at
`cwsr_enable=1`, on the same harness, on the same day — i.e. exactly the control that is scripted
and waiting. This is now not merely "the honest thing to do" but the *only* remaining route,
because the alternative data does not exist.

### Control check: does `cwsr_enable=0` reduce the pressure itself? No.

If disabling CWSR merely made the box quieter (fewer evictions), then the fault difference would
be about pressure, not about preemption. It does not:

| run | config | Q_EVICT | stream span | rate |
|---|---|---|---|---|
| `315evict7` | `cwsr_enable=1` | 8,419 | 2,011 s | **4.19 /s** |
| `315cwsr60` | `cwsr_enable=0` | 9,997 | 2,634 s | **3.80 /s** |
| | | | | (not-rescheduled: 8,417 vs 9,997 — both ~100%) |

The eviction load is comparable — slightly *lower* per second in the treatment, but the same order,
and every restore is still not-rescheduled. So the memory pressure and the queue-eviction activity
are present in both arms of the comparison.

**Why that matters:** it isolates the *variable* to what CWSR actually controls. `Q_EVICT` is a
queue-level event; `cwsr_enable` governs whether a **wave** can be preempted mid-execution. With
pressure held roughly constant across the two runs, a difference in fault rate is attributable to
the preemption capability rather than to how hard the box was being squeezed.

That is the distinction H2 rests on: evictions happen either way; what CWSR-off removes is the
hardware's ability to preempt a wave *mid-execution* while its buffers are being moved.

Incidental QA: run 7 had a server living **1,975 s** (the p3l fault arm, which outlived its arm and
which I killed by hand); the campaign's longest-lived server is **267 s**, i.e. its arm servers are
being cleaned up properly. That is the harness discipline the PID-capture work was part of, and it
is why run 7's per-arm eviction counts were unusable (see `315-pid-capture.md`).

## Criterion 2 revisited: H2 suggests an ENGINE-side remedy after all

Earlier I recorded that criterion 2 (a forward-only fix in `1bit-MONSTER/engine`) had "nothing to
repair", because the documented CWSR *size* defect is proved fixed here and H2's cause would live
in the driver. That was too quick. If H2 is the mechanism, it points at an engine-side change that
is a **repair, not a guard**:

> **Do not let the GPU compute on buffers that the driver is free to evict.**

H2's premise is that a wave preempted mid-execution finds its buffers evicted or migrated
underneath it. That can only happen to memory the driver *may* move — i.e. the evictable classes:
GTT/TTM allocations and, above all, **userptr/SVM** buffers, which the #315 investigation already
showed are evicted in the 95–110 GiB band (`315-TOOLS.md`: ~18 queue evictions in 20 s at
`GTT >= 95 GiB`).

So the engine-side remedy is a **buffer-residency change**, not a configuration guard:

- stop placing the buffers the GPU reads during compute (weights, KV, activations) in
  evictable/userptr memory;
- move them to a class the driver will not reclaim under pressure (a VRAM carve-out or a
  driver-owned allocation, rather than GTT-backed userptr);
- optionally, keep a residency ledger and refuse to start work whose working set cannot be made
  resident.

**This is an engine change, forward-only, and it is the same conclusion Engine V2's D5 already
reached independently** — "weights/KV in DRIVER-OWNED memory (no userptr/SVM/SVA) to kill the #315
eviction class by construction", with the P1 soak as its gate. Two separate lines of reasoning
arriving at the same fix is a good sign for both.

What it still needs before it can be landed as *this* issue's fix:

1. the mechanism confirmed by n (≥60 clean arms at `cwsr=0`) — H2 is currently the only surviving
   candidate but is not yet measured directly;
2. a demonstration that buffer residency, not merely CWSR-off, removes the fault — i.e. the
   `Q_EVICT` events for the engine's own buffers go to zero;
3. the byte-identical-output and no-regression gates the objective already requires.

**So criterion 2 is not hopeless after all** — I was wrong to close it earlier. The correct
statement is that the fix is now *identifiable* (residency) but not yet *earned* (it needs the
mechanism confirmed and the residency change shown to remove the evictions of the engine's own
buffers).
