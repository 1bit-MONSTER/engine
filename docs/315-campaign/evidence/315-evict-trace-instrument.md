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

## 6. Two bugs that cost the first real launch (both mine, both silent-ish)

### 6a. `MINCLEAN=0` stops immediately

`315-paired.sh:147` stops when the clean count **reaches** the target:

```sh
[ "$cf" -ge "${MINCLEAN:-60}" ] && { say "STOP: reached $cf clean fix-cell arms"; break; }
```

It is a goalpost, not a floor. Passing `MINCLEAN=0` — intended as "no minimum" — means
*stop at zero clean arms*, and `0 >= 0` is true, so the run ended after **pair 1**: 2 arms
instead of 12. The wrapper now computes `MINCLEAN=$(( PAIRS * 2 + 1 ))` so nothing is
reachable and the `PAIRS` cap is what ends the run. The console records it exactly:

```
after pair 1: fix-cell clean=0 fault-or-invalid=0 (target >=0 clean, cap 6 pairs)
STOP: reached 0 clean fix-cell arms
```

### 6b. `grep -c ... || echo 0` yields a two-line value

`grep -c` prints `0` **and** exits 1 when there are no matches, so `|| echo 0` appends a
second `0`. The value becomes `"0\n0"` and the arithmetic test fails:

```
/home/bcloud/wt/315-evict-trace.sh: line 98: [: 0
0: integer expected
```

The wrapper now takes the count and normalises with `${VAR:-0}` instead. Tested: zero
matches gives `IN=[0] FA=[0]` and the comparison succeeds.

**Why these matter beyond this script:** both failures left the run *looking finished*. 6a
produced a complete, well-formed verdict summary for a 2-arm run — it would be easy to read
"cell c: in-regime=1 clean=1 fault=0" as a result rather than as a truncated run. The
`PAIRS`/`MINCLEAN` line in the console is the tell.

### 6c. Box contention

A campaign must not run while a peer holds the box. Observed on the relaunch attempt: a
peer's `~/wt/loom-ub389` 27B HRX job under `guard.sh` (1800 s timeout), GTT 46.7 GiB,
`npu.lock` touched minutes earlier. The composition would have evicted that job's memory.
Wait for the box rather than disturb a peer — the goal's own constraint.

### 6d. The watcher started before the box was ours

Third launch (run 3, tag `315evict3`) started the KFD watcher at the **top** of the script,
~70 lines before the box-lock acquisition and the GPU-quiet check. So the watcher ran its
3-hour TTL while a peer still held HRX0, recording events from a window that was not the
campaign's. Its output would have looked identical to a correct run — the tell is only the
ordering of the lines in the run log.

Run 4 fixes the order, and the log now shows it:

```
box lock held 2026-10-12T13:13:03-03:00     <- lock first
GPU quiet: ...                              <- then the box is actually ours
watcher pid ...                             <- only now start the watcher
```

### 6e. flock alone does not protect against a peer

The box lock only excludes processes that **follow the convention**. Observed 2026-10-12: a
peer batch (`~/wt/loom-ub389`) runs `-dev HRX0` jobs and its only shell script is
`build.sh` with no lock usage. So `flock` reported the lock free while the peer held HRX0,
and composing neighbours into the 95–110 GiB band would have evicted that peer's memory.

The wrapper therefore takes the lock **and** polls for actual GPU quiescence: no foreign
`-dev HRX0` process and no `loom-ub389`/`repro336` job, with `QUIET_WAIT` (default 1800 s)
as the bound. On timeout it aborts and starts nothing, rather than racing.

### 6f. Killing a run needs the wrapper killed directly

`pkill -f "315-evict-trace.sh ..."` did not stop run 3 (it returned 255 and the process
survived), because the wrapper was parked in the quiet-poll's `sleep` with a trap installed.
The watcher stopped but the wrapper did not. `kill -KILL <pid>` was needed. Worth knowing:
a half-stopped wrapper can leave a watcher running or a lock held.

### 6g. There are TWO lock conventions on this box, and the campaign was holding the wrong one

The most consequential finding of this task's harness work. Measured 2026-10-12 13:18:

| lock file | holders at the time | who |
|---|---|---|
| `~/npu.lock` | **12 pids** | the box-wide convention — a peer batch (`flock ~/npu.lock ./batch2.sh`) **plus two unrelated NPU-goal workers** (`npu-forge2-kvsplit` bisect, `probe_model.sh`) |
| `~/.cache/lax-decode/box.lock` | 2 pids | **ours only** — the narrow convention the older 315 harnesses use |

So `box.lock` was free while `npu.lock` was held by three workers, and my wrapper
"successfully" took `box.lock` — logging `box lock held` — while a peer ran `-dev HRX0`.
**The lock line in the log looked like protection and provided none.** The only thing
actually protecting the box was the separate GPU-quiet poll added in §6e.

The fix: take **both**, `box.lock` first (harmless, cheap, matches the 315 harnesses) then
`~/npu.lock`, which is the one that serialises real GPU use. The log now distinguishes them:

```
box lock held <ts>  (/home/bcloud/.cache/lax-decode/box.lock)
npu lock held <ts>  (/home/bcloud/npu.lock) - the box-wide convention
```

Peer's usage, for the record (`~/wt/loom-ub389/runs/bench.sh:7`):

```sh
flock ~/npu.lock systemd-run --user --scope ... llama-bench ... -dev HRX0 ...
```

**Generalisation worth keeping:** on this box, `npu.lock` is the real mutex. Any harness
that only takes `box.lock` is unprotected against peers and against other goals' workers,
and it will report a held lock while the box is busy.

### 6h. Self-deadlock: the wrapper held the lock the driver re-opens (NOLOCK=1)

The eighth and last harness defect, and the one that finally mattered. Run 6 (`315evict6`)
passed all four of this wrapper's gates, started composing, and then **deadlocked against
itself**:

```
bash 11865  /home/bcloud/wt/315-evict-trace.sh 8 315evict6
  └─ bash 12119  315-campaign.sh
      └─ bash 12143  315-paired.sh 8 2 2048 ...
          └─ flock 15489  -w 900 9        <- blocked: locks_lock_inode_wait
                                            /proc/15489/fd/9 -> box.lock
```

The wrapper held `box.lock` on **fd 9** and `npu.lock` on **fd 8**. `315-paired.sh:47-52` then
**re-opens `box.lock` on its own fd 9** and flocks it:

```sh
if [ -z "${NOLOCK:-}" ]; then
  exec 9>"$HOME/.cache/lax-decode/box.lock"
  if flock -w "${LOCK_WAIT:-900}" 9; then
```

`flock` is per-open-file-description. A **second** open of a file this process tree already
holds blocks — the lock is not re-entrant — and after `LOCK_WAIT` the driver gives up and
"proceeds CONCURRENTLY (UNLOCKED)", so the wait is not even protective.

The driver documents the convention in its own header: **`NOLOCK=1` means the caller already
holds the lock on an inherited fd 9** — exactly our case. The wrapper now passes it.

Verified on run 7 (`315evict7`): all four gates passed, driver in `do_wait` (not
`locks_lock_inode_wait`), and the composition reached the regime — GTT 27 → 80 → 98 → 100 GiB
with 4 HRX neighbours and 560+ evictions captured.

**Pattern worth generalising:** when a wrapper takes a lock that its child also takes, the
child needs to be told the lock is already held. Silently double-locking looks like progress
up to the exact point where it stops.

## §6i — the console line cannot certify zero faults on its own (fixed in the analysis tool)

Criterion 3 asks for "zero faults across **both faces**". Both faces is the problem: the driver's
per-arm console line carries `nan= req= ret3= parse= gtt_start= utf8= alloc= hv52= hv56=` — and
Face B does not only arrive as a `parse=`/`utf8=` count. It also arrives as an **HTTP 500**, and
there is **no `http=` field in the console line at all**.

Confirmed against a real arm's detector:

```
315evict7-p1c-.../detector.json:
  requests 66  ok 66  utf8_failures 0  parse_failures 0  http_failures 0  transport_failures 0
                             ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^
                             these two are in the console line;  these two are NOT
```

So a Face-B fault surfacing as HTTP 500 leaves **no trace in the console**, and a console-only count
would report that arm **clean**. That is a false negative on the single criterion the campaign
exists to certify — the worst possible failure mode here, because it silently manufactures the
desired result.

**Fix, in the analysis tool only.** `tools/315-rate.py` now reads every arm's `detector.json` when
pointed at the arm root (`--det`, `--tag`) and treats the **detector as authoritative for faults**:

* an arm the console calls clean but the detector calls failed is counted as a **fault** and flagged
  `MISSED BY CONSOLE`, naming which counter caught it (`http=`, `transport=`, `parse=`, `utf8=`);
* an arm with **no usable `detector.json`** is reported **UNVERIFIED** — clean from the console
  alone, *not* certified. It is never silently counted as clean;
* with `--det` disabled the tool prints an explicit WARNING that an HTTP-500 Face-B fault would be
  counted CLEAN. The dangerous reading is stated, not implied.

The driver stays frozen, as required mid-campaign: this reads artefacts the driver already writes.

**Proven on a fixture where it changes the answer** (3 arms, console says OK for all three):

| arm | detector | console | tool |
|---|---|---|---|
| `TT-p1c` | `http_failures=1` | `OK` | **FAULT** — "MISSED BY CONSOLE … http=1" |
| `TT-p2l` | all zero | `OK` | clean |
| `TT-p3l` | **absent** | `OK` | **UNVERIFIED** — not certified |

Result: `1/2 = 50.00%`, acceptance FAILS. Without the reconciliation the same fixture reports
`0/3 = 0.00%` and the arm with a real HTTP-500 fault is counted **clean** — i.e. the tool would have
manufactured a passing campaign.

Re-checked against run 7 (which really faulted): still `3/11 = 27.27%`, 11 detectors read, no missed
arms. So the change is a guard, not a revision of the known result.
