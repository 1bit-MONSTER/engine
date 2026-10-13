<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Out-of-scope proposal: KFD eviction/restore must preserve or invalidate in-flight consumer state

The objective's boundaries say driver/runtime changes are out of scope but should be "record[ed] as a concrete
proposal with evidence, as the earlier runtime proposal did". No such proposal existed on this record, so this
is it, written from the evidence the #315 work has accumulated.

## What the evidence establishes

**The trigger.** `~/wt/evictonly.hip` performs `mprotect` churn over a `hipHostRegister`'d range; this makes
the kernel **evict and restore KFD queues**. Measured per-arm, each arm's **own** buffers show `Q_EVICT` /
`Q_RESTORE` while the arm is live. Pooled across nine runs of the frozen three-cell rig: **28 eviction-cell
faults in 29 arms** against **4 faults in 57 control arms**, Fisher exact p ≈ 1.3e-17
(`315-quiet-vs-forcer-by-level.md`, `315-pooled-rig-verification.md`).

**Both faces, one intervention.** 11 forced-eviction faults split **7 Face A / 4 Face B**
(`315-both-faces-one-trigger.md`) — a whole-vocabulary NaN signature and a format/parse failure, from the same
cell of the same rig.

**The shape.** Across 14 arms with recorded text, every faulting arm's output for the identical prompt matches
the clean arms' answer, with the first requests character-identical before the arm dies: **a terminal
whole-buffer event, never a changed computation** (`315-task9-session-state-2026-10-12.md`, the peer's n=14
finding).

**What the engine cannot do.** It names *which* allocation it holds (a monotonic `generation` per allocation,
mixed into the command-program cache key) and can detect *that* a read produced garbage (Face A's NaN guard;
Face B's parse throw) — but nothing on the replay → restore → consume path distinguishes "the device wrote
these bytes" from "a restore left a pre-eviction copy" (`315-cause-boundary-final.md`). Seven in-scope
candidate causes have been excluded by pre-registered experiment or by mechanism read at source.

## The proposal

**Restore fidelity is a guarantee the runtime currently does not appear to make, and the fault is what its
absence looks like from above.** Concretely, one of:

1. **Eviction/restore must be transparent to in-flight consumers** — i.e. a restored queue/buffer must be
   byte-identical to its pre-eviction state, or the eviction must be refused/deferred while any consumer holds
   a reference. KFD's context-save/restore already has a documented "remap queues that remain active" notion
   (`docs/hrx.md`, `context-save.md:305`), so this is a question of what it guarantees about *contents* rather
   than a new mechanism.
2. **Or the runtime must expose a validity signal** so a consumer can tell that its view may have been
   invalidated — an epoch/generation on the mapping, checked after synchronisation. The engine already has the
   plumbing for generations; what is missing is any that tracks *content* validity rather than allocation
   identity.

Option 1 is the correct fix; option 2 is the mitigation that would make the failure detectable rather than
silent, and is strictly weaker.

## Evidence a recipient would need, and where it is

| claim | evidence |
|---|---|
| forcing KFD evict/restore produces both faces, ~95% in the forced cell | pooled 28/29 vs 4/57, p≈1.3e-17 |
| the fault is a terminal whole-buffer event, not a computation change | n=14 recorded-text comparison |
| the eviction hits the *arm's own* buffers | per-arm `Q_EVICT`/`Q_RESTORE` attributed to the arm's pid |
| the engine cannot distinguish real from stale bytes | `315-cause-boundary-final.md` — no content-validity check exists |
| the fault is not any of the in-scope engine mechanisms | seven exclusions, each pre-registered or source-read |

## Why this is a proposal and not a claim about KFD

Nothing here shows KFD's restore is *incorrect* — it shows that a consumer can observe different contents
after an eviction/restore cycle, that the engine cannot tell, and that the consequence matches both recorded
faces. Whether the defect is in the restore, in the eviction decision, or in the interplay with a
non-coherent mapping is exactly the question this goal's boundaries place outside its scope.

## Suggested next step for whoever owns the runtime lane

Reproduce the trigger against a **minimal** consumer: hold a mapped host-visible buffer, write a known pattern
from the device, force evict/restore, and read the mapping back — before and after a synchronisation. If the
contents differ, the fidelity gap is confirmed at the smallest possible scale, with no model, no regime and no
neighbours required. That experiment is ~minutes of work and is the natural entry point for a runtime owner.
