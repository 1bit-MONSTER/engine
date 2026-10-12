# task-8 live status (eviction/preemption trace)

Run: `315evict2` on strixhalo. Purpose: attach the KFD SMI eviction/preemption stream to
in-regime arms so a faulting arm can be correlated. NOT an acceptance run.

## Setup, verified before trust

- watcher as root (unprivileged = header-only file, exit 0, silently); relay TTL 10800 s
- regime reached: GTT 98-104 GiB (gate >= 95), 5 HRX neighbours
- control cell `315instr3-build` (pin, `if (direct_host_binding)` intact) vs fix cell
  `315fix-build` (pin + HOST_COHERENT); both `e44c9d01a` text, so this is the diagnostic
  pair, not a fix comparison

## Arms so far

```
pair 1 arm c: OK nan=0 req=66 parse=0 gtt_start=98G utf8=0
```
detector.json: requests 66, ok 66, utf8_failures 0, parse_failures 0, http_failures 0,
transport_failures 0, truncated false, first_failure null.

## Eviction stream

Thousands of Q_EVICT events accumulated while these clean arms ran (4458+ at arm 1's end).
**This is the substantive observation so far: heavy eviction pressure is present AND the
arm is clean.** So evictions are necessary-looking but demonstrably not sufficient on
their own for this fault at this resolution - which is narrower than the preemption lead
predicts, and must not be written up as support for it.

## Contract reminder

task-8 is satisfied by EITHER >=1 in-regime faulting arm with the correlation attached,
OR the trace recorded UNINFORMATIVE with its clean-arm count. A clean trace is not
evidence about the mechanism.
