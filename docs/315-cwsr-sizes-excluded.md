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

================================================================================
FOLLOW-UP: both CWSR size functions verified, and a CONFOUND I must flag
================================================================================

1. THE TWO SIZE FUNCTIONS - both correct for gfx1151

libhsakmt exports exactly two, and the kernel has a matching check for each:

  hsakmt_get_vgpr_size_per_cu   ->  kernel check: "cwsr size ... not sufficient for"  (FIXED)
  hsakmt_get_sgpr_size_per_cu   ->  kernel check: "ctl stack size ... not equal to"   (UNFIXED)

Measured by linking queues.c.o and calling both:

  gfxv       name             vgpr/CU   sgpr/CU
  0x0B0501   GFX1151 (ours)   0x60000   0x4000
  0x0B0000   PLUM_BONITO      0x60000   0x4000
  0x0B0001   WHEAT_NAS        0x60000   0x4000
  0x0C0000   GFX1200          0x60000   0x4000

  0x60000 = 384 KB = 1.5 x 256 KB  -> the #1807 fix IS present (as already established)
  0x4000  = 16 KB, the standard GFX11 control-stack size -> nothing gfx1151-specific

So BOTH sizes our runtime declares for gfx1151 look correct. The size-mismatch story is
dead on both halves, not just the VGPR half. And note the mismatch path is now guarded
anyway: the fixed "<" check rejects an undersized userspace value (EINVAL, loud) and
accepts a larger one, so a size disagreement can no longer silently under-allocate.

CONCLUSION: if CWSR-off genuinely fixes the fault, the defect is NOT the area SIZE. It is
elsewhere in the save/restore path - execution, ordering, or the hardware/firmware
handling of the spill itself. Size is excluded.

2. CONFOUND I MUST FLAG - the reboot changed more than cwsr_enable

I verified the kernel cmdline differed by exactly one token. But the reboot ALSO changed
box STATE that run 7 did not share:

  run 7    (cwsr_enable=1): taken on a box with long uptime, many prior campaigns,
                            fragmented device memory
  cwsr0    (cwsr_enable=0): taken on a FRESH boot - cool, unfragmented, no prior campaigns

That is a real confound, and it is the same class of error this issue has caught before
("the rate followed memory-pressure settings"). The 11-clean-arm result could be partly or
wholly a fresh-boot effect rather than a CWSR effect.

What mitigates it, and what does not:

  MITIGATES: the ~26% baseline is not one session - it is 34 faulting out of 131 in-regime
  arms accumulated across many sessions and boots (the resource-independent part of the
  record). A fresh-boot effect would have to be unusually strong to explain 0/11 against a
  multi-session 26%.

  DOES NOT MITIGATE: no fresh-boot control at cwsr_enable=1 exists. The proper control is
  a fresh boot WITH cwsr_enable=1, running the identical campaign, on the same day.

3. SO THE HONEST STATE

  - promising signal, first in this issue's history: 0 in-regime faults at cwsr_enable=0
    vs a 26% multi-session baseline at cwsr_enable=1
  - the SIZE explanation is falsified on both the VGPR and SGPR halves
  - the fresh-boot confound is UNCONTROLLED, and must be controlled before the signal is
    called causal
  - the >=60-arm campaign is running; if it is clean, the NEXT required step is a
    fresh-boot cwsr_enable=1 control, not a fix
