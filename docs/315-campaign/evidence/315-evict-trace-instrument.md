<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->
# eviction/preemption trace: instrument verification (task-8)

Verified before the campaign was trusted, because three different failure modes here are
**silent** — each would have produced a plausible-looking empty result instead of an error.

## 1. The watcher needs root, and fails silently without it

`kfd_smi_watch` unprivileged exits 0 and writes a **header-only** file. That reads as "no
evictions occurred" rather than "no privilege to see them".

| invocation | outcome |
|---|---|
| unprivileged, 10 s idle | exit 0, 1 line (the `#recv_boot_ns...` header) |
| unprivileged, 20 s under GPU load | exit 0, 1 line |
| **`sudo`, 30 s under one HRX load** | **75 Q_EVICT / 79 Q_RESTORE** |

The wrapper (`315-evict-trace.sh`) aborts if `sudo -n true` fails, so a header-only file can
no longer be mistaken for a quiet box.

## 2. It writes TSV, not NDJSON

The first version of the wrapper wrote to `kfd_smi.ndjson` and grepped for JSON quotes
(`"Q_EVICT"`, `"trigger":"..."`). Both are wrong: the stream is tab-separated with a
`#recv_boot_ns	gpuid	event	kernel_ns	pid	fields...` header. Fixed.

## 3. The HSA path is not under `lib/`

`IREE_HAL_AMDGPU_LIBHSA_PATH` is
`/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1`
(taken from `315-paired.sh`). Guessing a path under `lib/` does not fail loudly — the HRX
device simply never registers and the server exits with `invalid device: HRX0`.

## 4. The correlator parses our format

`correlate.py` was smoke-tested against a real SMI capture plus a synthetic server log
before the campaign needed it. It auto-detects the target pid from the SMI stream (here
`789665`) and emits its per-request table. So the faulting arm's `server.log` can be
correlated without a code change.

## 5. The driver's unit convention (cost a launch)

`315-campaign.sh` takes `FIX`/`CTRL` as **build dirs** and appends `/bin`. Passing `.../bin`
gives `bin/bin` and aborts immediately (`ABORT: no .../315fix-build/bin/bin/llama-server`),
tearing down the watcher with only its header recorded. `strip_bin()` normalises both forms.

## Live confirmation

Run `315evict2`, once the composition reached the gate:

```
GTT 96.9 GiB  (gate: >= 95)
HRX neighbours up: 5
Q_EVICT 180   Q_RESTORE 246
```

So the regime is reached and the eviction stream is populated. Whether this becomes
**evidence** still depends on an in-regime arm faulting while the watcher is attached —
a clean trace is recorded as UNINFORMATIVE with its clean-arm count, never as evidence.
