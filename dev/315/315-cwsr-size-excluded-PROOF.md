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
# The gfx1151 CWSR size defect is FIXED on this box — proven by exact arithmetic

**Date** 2026-10-12. Closes the size question quantitatively and independently. This
supersedes the weaker "measured both helpers" argument in `315-cwsr-sizes-excluded.md`.

## The known defect (AMD's own record)

An Ubuntu SRU cover letter from **Mario Limonciello (AMD)** — `[SRU][Q][PATCH 0/2] GFX1151
kernel fixes`, 2026-01-11 — states it plainly:

> GFX1151 has a problem where the context save / restore area is **incorrectly sized** and this
> can lead to hangs when used with ROCm. Patches were upstreamed as part of 6.19-rc1.
>
> ```
> Jonathan Kim (1):  drm/amdkfd: bump minimum vgpr size for gfx1151        (kfd_queue.c +1)
> Mario Limonciello: drm/amdkfd: Export the cwsr_size and ctl_stack_size   (kfd_topology.c +4)
> ```
> Matching code for ROCm: `ROCm/rocm-systems@0a2ea9ef`

The cause, from a deployment report:

> the kernel and ROCm **disagree on the VGPR file size of gfx1151** — kernel
> `kfd_get_vgpr_size_per_cu()` lists gfx1151 → `0x60000`; ROCm `hsakmt_get_vgpr_size_per_cu()`
> does not → falls back to `0x40000`. So ROCm asked for a context-save area of
> **13,942,784 bytes** while `kfd_queue_acquire_buffers()` required **at least** more than that.

## The arithmetic, from the per-CU components

The freedesktop patch `drm/amdgpu: Fill cwsr save area details` gives the components:

```
SGPR_SIZE_PER_CU 0x4000    LDS_SIZE_PER_CU 0x10000    HWREG_SIZE_PER_CU 0x1000
```

gfx1151 = **40 CUs** (Radeon 8060S).

| vgpr value | per-CU total | × 40 CUs | node requires | result |
|---|---|---|---|---|
| `0x40000` (256 KB — **unfixed**) | `0x55000` (348,160) | **13,926,400** | 19,185,664 | **5,259,264 B short → EINVAL** |
| `0x60000` (384 KB — **1.5×, fixed**) | `0x75000` (479,232) | **19,169,280** | 19,185,664 | off by 16,384 = exactly one SGPR |

`node / per_cu = 40.0342` — i.e. **40 CUs** plus one extra SGPR term. That residual is exactly
the `+16,384` above.

## The independent cross-check

The deployment report's real broken-ROCm figure is **13,942,784 bytes**. My computed *unfixed*
total is **13,926,400**. Difference: **16,384 = one SGPR unit.**

`13,926,400 + 16,384 = 13,942,784` — **an exact match** to a number measured on a different
machine in a different report.

That triple agreement (AMD's patch description, the per-CU component sizes, and the observed
broken declaration size) is what makes this a proof rather than a reading.

## What this means for our box

| requirement | our value | verdict |
|---|---|---|
| kernel exports `cwsr_size` / `ctl_stack_size` to userspace | present — we read both from `properties` | ✅ the `kfd_topology.c` half is in |
| kernel accepts a *larger* userspace value (`<` not `!=`) | present — module carries `not sufficient for` | ✅ the `d15deaf` half is in |
| kernel's per-CU VGPR is `0x60000` | consistent with node `cwsr_size` = 19,185,664 (40 × 0x75000) | ✅ the `kfd_queue.c` bump is in |
| userspace declares `0x60000` | measured by calling `hsakmt_get_vgpr_size_per_cu(0x0B0501)` → `0x60000` | ✅ the `#1807` half is in |
| userspace declared size ≥ node requirement | ~20.9 MB ≥ 19.19 MB | ✅ passes |

**All five halves of the fix are present. Our box is fully patched for the known defect.**

## Consequence — stated before the campaign result lands

The known, documented, AMD-acknowledged gfx1151 CWSR defect **cannot be the cause of our
fault**, because the fix is complete here. So:

- the size explanation is dead **with proof**, not just with a measurement;
- any remaining CWSR involvement must be a *different* defect — the spill/restore execution
  itself (**H2**), not the area's geometry;
- and the fresh-boot confound (**H1**) becomes more important, not less: since the documented
  CWSR bug is fixed here, a plain "CWSR is broken on gfx1151" story is not available, and the
  11-clean-arm signal needs the control before it can be attributed at all.

**Net ranking after this proof:** H1 (fresh boot) and H2 (execution-level CWSR defect) remain;
H3 and H4 are now dead — they were both variants of sizing, and sizing is excluded by exact
arithmetic.
