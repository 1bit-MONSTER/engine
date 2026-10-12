# CWSR experiment: a real signal, and a falsified explanation for it

**Date** 2026-10-12. Follows `315-cwsr-hypothesis.md`, which proposed that the fault came
from an undersized CWSR (compute wave save/restore) area because the userspace half of the
fix pair (`rocm-systems#1807`) was missing.

## 1. The discriminating experiment: `amdgpu.cwsr_enable=0`

Rebooted with only that one variable changed (kernel `7.3.0-rc4-1bit`, same
`ttm.pages_limit`, same crashkernel, same module). The run reached its full 8-pair cap:

```
11 in-regime arms, 0 faults, every arm at gtt_start=97G
```

`gtt_start=97G` is the same gate value at which run 7 faulted 3 times, and the historical
record is 34 faulting out of 131 in-regime arms (26%).

| | baseline `cwsr_enable=1` | treatment `cwsr_enable=0` |
|---|---|---|
| in-regime arms | 11 | 11 |
| faults | **3** | **0** |
| regime | 97–98 GiB | 97 GiB |

Evictions still occurred in the treatment (5,809 Q_EVICT captured), so CWSR-off removed the
wave save/restore path without removing the pressure — the comparison isolates one variable.

**Statistics, stated plainly — this is NOT yet a result:**

| test | value | reading |
|---|---|---|
| rule-of-three 95% upper bound at 11 clean | 27.3% | does **not** exclude the 26–27% baseline |
| binomial P(0/11 \| true rate 0.273) | 0.030 | suggestive |
| Fisher exact one-sided (0/11 vs 3/11) | 0.107 | not significant at 0.05 |

The objective's bar is **≥60 in-regime arms**. Eleven is not sixty.

This is nonetheless **the first intervention in this entire investigation that moved the
fault rate in the predicted direction.** Coherence, ordering, AQL barriers, `-fa off`,
transient reuse, PM4 and the MoE routing levers all returned null.

## 2. Falsification: the VGPR-size explanation is wrong for this box

`rocm-systems#1807` is **MERGED** (2025-11-21), and its exact diff is:

```diff
 libhsakmt.h:  +  GFX_VERSION_GFX1151 = 0x0B0501,
 queues.c:     +  gfxv == GFX_VERSION_GFX1151 ||     (adding it to the -> 0x60000 branch)
```

**Our runtime already has it.** Verified at runtime, not by reading source: I extracted
`queues.c.o` from `/opt/rocm-therock/.../_rocm_sdk_core/lib/libhsakmt.a`, linked it into a
probe, and called the real exported function:

```
0x0B0501  GFX1151 (our GPU, gfx1151)  -> 0x60000 (384 KB)   <- 1.5x value, fix present
0x0B0000  PLUM_BONITO gfx1100         -> 0x60000 (384 KB)
0x0C0000  GFX1200                     -> 0x60000 (384 KB)
0x090008  older                       -> 0x80000 (512 KB)
```

384 KB = 1.5 × 256 KB — exactly the "1.5x VGPR memory" the PR describes — and our lib
returns it for gfx1151. **So the CWSR area is not undersized by that mechanism.**

## 3. What survives, what does not

**Survives:** the CWSR save/restore path is implicated. That is a real, useful signal and
the strongest positive in this issue's history so far.

**Does not survive:** my explanation of *why*. If CWSR-off genuinely fixes the fault, the
defect is somewhere else in the wave save/restore path. Candidates on the record:

- our kernel module carries the **UNFIXED ctl-stack check**
  (`queue ctl stack size 0x%x not equal to node ctl stack size`) while the cwsr-size check
  *is* fixed (`not sufficient for`) — i.e. a partial kernel-side fix;
- the HSA runtime layer (`libhsa-runtime64.so.1`, TheRock) between libhsakmt and KFD;
- the kernel's own CWSR handling under preemption.

## 4. Not done

- **≥60 arms.** 11 is not 60; the statistics say so explicitly.
- **The forward-only change.** With the VGPR story dead, "bump the runtime" is no longer
  the fix. The remaining candidate is the engine *requiring* the working configuration (a
  startup check on `cwsr_enable`) rather than silently faulting — a legitimate engine-side
  change, but a **workaround guard, not a repair**.
- Regression control, CI, KNOWN-ISSUES update.

## 5. Honest summary

Researching the KFD/CWSR boundary — which no prior lever in this issue had touched —
produced the first intervention that changed the fault rate in the predicted direction.
But the causal story attached to it is falsified, and 11 arms is far short of 60. The
position is: **a promising lead, a wrong explanation for it, and more work needed.**
