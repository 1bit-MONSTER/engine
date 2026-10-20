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
# H1 (the fresh-boot confound) is FALSIFIED — and it was my own error

**Date** 2026-10-12, late. This reverses the hypothesis ranking I recorded earlier in
`315-cwsr-refined-hypothesis.md`. Read this first.

## What I claimed, and why it was wrong

I recorded that the CWSR signal was confounded because:

> run 7 (`cwsr=1`, 3/11 faults) ran at **~32.5 h uptime**; the `cwsr=0` campaign started **692 s**
> after boot. The comparison is therefore not only CWSR on vs off but also 32-hour-old box vs
> 11-minute-old box.

**The 32.5 h figure was wrong.** I derived it from an uptime reading taken at 12:37 on 10-12
("up 1 day, 7:04") and did not account for the **clean reboot at 14:03:53** that happened between
that reading and run 7.

## The correction, from two independent clocks

The KFD SMI stream records `boot_ns` (CLOCK_BOOTTIME, in nanoseconds), which — combined with any
known wall-clock anchor for the same event — recovers the boot time exactly:

| cross-check | arithmetic | implied boot |
|---|---|---|
| pid 111371's `PROC_START` `boot_ns` = 1,180,845,874,852 ns = 1180.8 s, against `ps`'s start time for that same process (Mon Oct 12 **14:23:33**) | 14:23:33 − 1180.8 s | **14:03:52** |
| run 7's campaign start (its run.log: **14:09:42**) minus its stream's first event (`boot_ns` = 352.4 s) | 14:09:42 − 352.4 s | **14:03:49** |
| the user's clean reboot (measured directly: `uptime -s` = 14:03:53) | — | **14:03:53** |

Three independent numbers agreeing to within 4 seconds. The earlier method failed because I used a
*completed* stream's file mtime (its last write) instead of its start time; anchoring on the
campaign's own start timestamp instead fixes it.

## What this means

| run | config | box age when its arms ran | result |
|---|---|---|---|
| **run 7** | `cwsr_enable=1` | **~6–40 min after a fresh boot** | **3 faults / 11 arms** |
| campaign `315cwsr60` | `cwsr_enable=0` | ~53 min – 7 h after a fresh boot | 0 faults / 9 arms so far |

**Both runs were on fresh boots, on the same day, within hours of each other.** The "32-hour-old
box vs 11-minute-old box" contrast does not exist. What remains is a genuine same-day,
same-hardware, same-harness comparison differing in `cwsr_enable` — and the `cwsr=1` arm of it
faulted 3 times in 11 arms while the `cwsr=0` arm has not faulted in 9.

So **H1 is largely falsified**: a fresh boot does *not* suppress the fault, because run 7 was
freshly booted and faulted anyway. That was the single strongest alternative explanation for the
CWSR signal, and it is now gone.

## Consequences, stated plainly

1. **The CWSR signal is strengthened, not weakened.** It is no longer explainable by boot
   freshness. The remaining alternative is H2 (a mid-wave-preemption mechanism), which has no
   upstream documentation (`315-cwsr-refined-hypothesis.md`) but also no competing explanation
   left.
2. **The H1 control is now much less urgent** — it would still be good practice, but the specific
   confound it was designed to rule out has been ruled out by the boot-time reconstruction
   instead.
3. **My earlier ranking was wrong and I said so.** I had written that "H1 is the leading
   explanation, not H2" and that the signal was "probably a boot artifact". Both statements were
   based on a mis-derived uptime, and both are withdrawn here.
4. **The statistics are unchanged** and still short of the bar: 9 clean arms gives a rule-of-three
   upper bound of 33%, which does not exclude the 26–27% baseline. Removing H1 removes an
   *explanation*, not the need for ≥60 arms.

**Lesson worth keeping:** I inferred a 32.5 h uptime from a reading taken before an intervening
reboot, and treated it as a quantitative confound for several turns. The SMI stream carried the
true value the whole time (`boot_ns`), and two independent anchors agreed on it within seconds once
I actually computed it. Derive, do not remember.
