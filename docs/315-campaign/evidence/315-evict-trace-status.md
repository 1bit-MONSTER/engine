# task-8: eviction/preemption trace — live status

Run `315evict2` on strixhalo. Purpose: attach the KFD SMI eviction/preemption stream to
in-regime arms so a **faulting** arm can be correlated against the driver's own events.
This is NOT an acceptance run (task-9 owns acceptance).

## Setup (verified before the trace was trusted)

- watcher as root, TTL 10800 s (unprivileged it writes a header-only file and exits 0 —
  silently; see `315-evict-trace-instrument.md`)
- regime reached: GTT 98–104 GiB, gate is `>= 95`
- control cell `315instr3-build` (pin, `if (direct_host_binding)` intact) vs fix cell
  `315fix-build` (pin + HOST_COHERENT). Both report `e44c9d01a`, so this is the diagnostic
  pair — it tests the falsified coherence lever, not a candidate fix.
- arm→PID binding is exact (`tools/arm-pid.py`, via the port in the arm's own server.log)

## Eviction stream (the substantive measurement)

| quantity | value |
|---|---|
| Q_EVICT | **14,570** |
| — trigger=SVM | 8,594 |
| — trigger=USERPTR | 5,976 |
| Q_RESTORE | 18,769 |
| — `rescheduled=no` | **14,566** |
| — `rescheduled=yes` | 4,203 |
| VMFAULT | **0** |
| PAGE_FAULT | 0 |
| sustained rate | ~972/min (≈16/s) at GTT ≈ 104 GiB |

## Arms so far

```
pair 1 arm c: OK  nan=0 req=66 parse=0 gtt_start=98G   <- IN regime, clean
pair 1 arm l: OOR nan=0 req=66 parse=0 gtt_start=93G   <- OUT of regime, NOT COUNTED
```

Both arms served 66/66 with `utf8_fail=0 parse_fail=0 http_fail=0 transport_fail=0
truncated=False` and no NaN. But the driver's own gate ("out-of-regime `gtt_start` < 95 GiB
(reported, not counted)", `315-paired.sh:12`) excludes arm `l`: it started at 93 GiB, below
the frozen 95 GiB gate.

**So the tally to date is 1 in-regime arm, 0 faults — not 2 clean arms.** An `OOR` arm is
neither clean evidence nor a fault and must not be summed with the clean ones. The
composition does not guarantee every arm lands in band: `gtt_start` is sampled per arm
(`315-paired.sh:75`) and drifted by 5 GiB between the two arms of this pair.

If the drift persists, a 12-arm run yields roughly half in-regime arms — which is far short
of task-9's `>=60` but is enough for task-8, whose requirement is a single faulting arm or
an honest null.

## What this does and does not establish

**Does:** the preemption/eviction window the #123/#140 lead describes is **open and
intense** during these arms — 14.5k queue evictions, of which 14.6k restores were *not*
rescheduled (i.e. the evicted queue's work was dropped and had to be redone). The
composition is genuinely exercising the driver's eviction path, not idling near it.

**Does NOT:** produce a fault. Both scored arms are clean under that pressure.

So the honest reading at this point is **negative for the lead as stated**: heavy
eviction/preemption in regime is present *and* the arm is clean. Eviction pressure is
therefore **not sufficient** on its own at this resolution. That is narrower than the
#123/#140 record predicts ("the window opens when a dispatch is preempted, and any
process's KFD queue eviction ... does that"), and it must not be written up as support for
the mechanism.

Two caveats against over-reading the negative, both required before it is treated as a
result:
1. the run is incomplete (1 of 12 arms scored), so the fault may simply not have occurred
   yet in this composition;
2. the fix cell here carries HOST_COHERENT — a lever already falsified — so a clean fix
   arm is expected and says nothing about the fault.

## Contract reminder (task-8)

Satisfied by EITHER `>=1` in-regime faulting arm with the correlation attached, OR the
trace recorded **UNINFORMATIVE with its clean-arm count**. A clean trace is not evidence
about the mechanism. If the run ends clean, that is what will be recorded.

---

## Run 1 (`315evict2`) — completed, but TRUNCATED, and why

The first real trace run ended after **2 arms, not 12**. Cause: my wrapper passed
`MINCLEAN=0` intending "no minimum", but `315-paired.sh:147` treats MINCLEAN as a goalpost
and stops when the clean count REACHES it — so `0 >= 0` stopped the run at pair 1. See
`315-evict-trace-instrument.md` §6a. Fixed (`MINCLEAN=$((PAIRS*2+1))`), plus §6b (a
`grep -c || echo 0` producing a two-line integer).

What the truncated run did establish, at full strength, is the **eviction environment**:

| quantity | value |
|---|---|
| Q_EVICT | **15,357** (8,936 SVM + 6,421 USERPTR) |
| Q_RESTORE | 19,649 |
| — `rescheduled=no` | **15,353** |
| VMFAULT | 0 |
| arms scored | 2 — `c` in-regime OK, `l` **OOR** (93 GiB < 95 gate) |
| in-regime arms | **1** |
| in-regime faults | **0** |

So: 15.4k evictions, 15.4k of them unrestored, sustained across the window, with one
in-regime clean arm. **This is a null for task-8, not a result** — and it is explicitly
*not* recorded as evidence about the mechanism.

## Why run 1 cannot be read as anything stronger

1. one in-regime arm is nothing like the `>=60` task-9 needs, and task-8 itself wants a
   *faulting* arm to correlate;
2. an `OOR` arm is neither clean nor faulted, so the fix cell contributed **no** in-regime
   data at all;
3. the campaign was truncated by a harness bug, so its length was not the designed length.

The honest statement is: **the eviction/preemption window was open and intense, and the
one in-regime arm was clean.** That is a datapoint about eviction pressure being present
without the fault, and nothing more.

## Run 2 — not launched: box contention

A relaunch on 2026-10-12 13:09+ found a peer agent's HRX workload on the box
(`~/wt/loom-ub389`, `guard.sh`-wrapped, 1800 s timeout each, first a 27B Qwen3.8 run then a
Zamba2-VL-2.7B run, `npu.lock` freshly touched). The campaign composes ~5 neighbours into
the 95–110 GiB band, which would have evicted that peer's memory. Per the goal's own
constraint (take the box lock, do not disturb other workloads) the relaunch **waits** for
the box rather than competing for it.

This is the second time the box has been unavailable for a long in-regime window (the
first was `repro336` holding 103–111 GiB). It is a real constraint on task-8, not an
excuse: the trace needs an uninterrupted composition, and the box is shared.

## Run 2: deferred, with the reason and the resume command

Not launched. The box is held by a peer agent's **batch** (`~/wt/loom-ub389/runs/batch2.log`,
active 13:10, jobs queued, several `-dev HRX0`) plus a CPU embedding pass. The campaign's
composition drives ~5 neighbours into the 95–110 GiB band, which would evict that peer's
memory. The goal's constraint is explicit — take the box lock, do not disturb other
workloads — so the trace waits.

Resume when the box is quiet, with the two launch bugs fixed:

```sh
ssh strixhalo
# confirm free: no "-dev HRX0" jobs of anyone else's, GTT near the ~26 GiB baseline
ps -eo pid,args | grep -- "-dev HRX0" | grep -v grep
awk '{printf "%.1f GiB\n", $1/1073741824}' /sys/class/drm/card*/device/mem_info_gtt_used

# then, with the corrected wrapper (MINCLEAN=$((PAIRS*2+1)), PAIRS cap ends the run):
nohup bash ~/wt/315-evict-trace.sh 8 315evict3 > ~/wt/315evict3-run.log 2>&1 &
```

8 pairs = 16 arms, which at the observed ~1-in-2 in-regime rate gives ~8 in-regime arms.
That is still short of task-9's `>=60`; task-8 needs only one **faulting** in-regime arm, or
an honest null with its clean-arm count.

**If the box stays contended, task-8's honest close is the null already in hand**: 1
in-regime clean arm under 15.4k evictions, recorded as uninformative, with the reason being
box contention rather than a design choice.

## Box contention, measured 2026-10-12 13:10–13:17

The failure mode here is not a bug in the harness — it is that the box is continuously busy.
Measured, not assumed:

| observation | value |
|---|---|
| peer batch | `~/wt/loom-ub389/runs/` — grew `batch1` → `batch2` → `batch3` while waiting |
| job types in the batch | `emb_hrx`, `emb_cpu`, `rr_hrx`, `rr_cpu`, `vis_hrx`, `zamba_base/fix/fin_ub`, `txt_fin_ub`, `ms_fin_b`, `fin_` |
| HRX0 occupancy, sampled every 15 s for 2 min | **1–2 holder processes at every sample, never 0** |
| GTT while a peer job ran | steady ~27 GiB (a small job, not the 95–110 band) |
| one peer job's duration | 6 min 21 s elapsed against a 1200 s cap, with more queued behind it |

So the peer's batch takes HRX0 continuously in small jobs. My composition needs ~5
neighbours resident in the 95–110 GiB band for ~5 minutes per arm; running it would evict
the peer's live job. The goal's constraint is to not disturb other workloads, so run 4
polls (30 min budget) and aborts having started nothing rather than racing.

**This is the second long unavailability this task has hit** (the first was `repro336`
holding 103–111 GiB on 2026-10-09). It is a property of a shared box, and it is the honest
reason task-8 may close on the null already in hand rather than on a new run.

Note also: GTT stays ~27 GiB during peer jobs, i.e. the peer never approaches the regime.
A cheaper future option — not taken here because it would change the pre-registered
composition — would be to interleave arms into gaps rather than requiring an empty box.

## Runs 4 and 5: the lock finding, and where the campaign now stands

**Run 4** (`315evict4`) ran the wrapper *before* the two-lock fix. It "acquired the box
lock" at 13:13:03 and sat in the GPU-quiet poll for 5m32s while `~/npu.lock` was held by 12
other processes. Its log line `box lock held` read as protection and provided none — the
poll was doing all the work. Stopped and discarded rather than left running in a
semantically unprotected state.

**Run 5** (`315evict5`) runs the corrected wrapper. Its log now distinguishes the two locks
and, crucially, **stops at the one that matters**:

```
box lock held 2026-10-12T13:18:52-03:00  (/home/bcloud/.cache/lax-decode/box.lock)
  ... then waits on ~/npu.lock, held by 14 processes ...
```

That is the expected behaviour and it is a behaviour change worth stating: before the fix
the run would have proceeded into the composition while another worker held the box-wide
lock. Now it waits, and aborts having started nothing if the lock is unobtainable within
its budget.

### Status: task-8 is blocked on the shared box, not on the harness

The harness has now been hardened against seven distinct silent-failure modes
(`315-evict-trace-instrument.md` §6a–§6g). What remains is not engineering:

| | |
|---|---|
| `~/npu.lock` holders while run 5 waited | 12 → 14 |
| workers holding it | a peer batch (`loom-ub389`) **and** two other goals' workers (`npu-forge2-kvsplit`, `probe_model.sh`) |
| HRX0 occupancy | 1–2 processes at every sample over 2 min |
| regime requirement | ~5 neighbours resident in the 95–110 GiB band, ~5 min per arm |

So the honest position on task-8 is:

- **evidence in hand**: run 1 — 1 in-regime arm, 0 faults, under 15,457 evictions
  (15,353 of them unrestored). A **null**.
- **blocked**: a fresh run needs the box-wide lock free for a sustained window, and three
  unrelated workers are currently holding it.

Per the task-8 contract, the null is recordable as **uninformative with its clean-arm
count**, and the reason is box contention. That is a legitimate close, not a design choice —
but it is a weaker close than a faulting arm would be, and it should be reported as such.

## The lock holder is another #315 campaign, thermally stuck while holding the lock

Diagnosed 2026-10-12 13:19–13:22. `~/npu.lock` was not held by a peer agent's batch alone —
it was held by **another #315 campaign**, `~/wt/315fix-tools/camp5.sh` (started 13:03:31),
running arms tagged `fx-c5-*` on the same control binary via an `LD_PRELOAD` shim
(`cwsr_shim2.so`) with modes `lockall`, `pinall`, `pincwsrstaging`.

**It has produced nothing in 16 minutes.** Its arm directory
`~/wt/rt-det/fx-c5-lockall-r1` is entirely empty — no `run.log`, no `server.log`, no
`detector.json` — and its only child is `flock 7` (holding `npu.lock`, 13+ minutes).

**Why, exactly.** The arm script gates on temperature *before* it starts work
(`/tmp/315arm-6xh7.sh:33`):

```sh
while [ "$(tctl)" -ge ${TSTART:-72} ]; do sleep 10; done
exec 7>"$HOME/npu.lock"; flock 7
```

and it takes the lock on the next line. Measured `Tctl` over 60 s:

| t | Tctl | edge |
|---|---|---|
| 10 s | 70.8 °C | 37.0 °C |
| 20 s | 72.5 °C | 37.0 °C |
| 30 s | 72.1 °C | 37.0 °C |
| 40 s | 71.8 °C | 37.0 °C |
| 50 s | 72.1 °C | 36.0 °C |
| 60 s | 72.1 °C | 36.0 °C |

`Tctl` oscillates **70.8–72.5 °C** around the 72 °C gate, so the wait never reliably
completes. Note the two sensors disagree wildly: `edge` is 36–37 °C while `Tctl` sits at
~72 °C. The gate is on the hot one, and the arm holds `npu.lock` for the entire wait.

**Consequence for task-8:** run 5 is starved — it correctly took `box.lock`, then blocked on
`~/npu.lock`, which the thermally-stuck arm holds. Waiting cannot fix this; the other
campaign must clear or its lock must be released.

**This is a design flaw in the other harness, not a bug in mine**, and it is worth flagging
plainly: an arm that holds the box-wide lock **while waiting for a thermal precondition**
can block the box indefinitely when the precondition sits on the wrong side of its
oscillation. The safe order is to wait for thermal conditions *before* taking the lock (or to
bound the wait and release).

## Root blocker identified: a peer batch holds `npu.lock` while its jobs fail

Traced the queue depth on 2026-10-12 13:24. `~/npu.lock` had 14 holders, but only one
matters, because `flock` is first-come: **pid 99828**, `flock ~/npu.lock ./batch2.sh`,
started 11:39:11 — **held 1 h 44 m**. Every other holder is queued behind it, including:

| holder | waiting | what |
|---|---|---|
| 948331 + 948371 | — | **run 5** (this task), correctly blocked |
| 847551 + 867745 | 16 m 32 s | camp5's `fx-c5-lockall-r1` arm, whose `flock 7` has not yet acquired |
| 816800 / 826913 / 826914 | — | **two other goals' workers** (`npu-forge2-kvsplit` bisect, `probe_model.sh`) |
| 816808 / 826939 / 826940 | — | their `flock -w 7200 9` children |

**The lock owner is not healthy.** `batch2.log` stopped growing (7185 bytes, unchanged over
40 s) and its current job is failing outright:

```
0.00.126.893 E llama_model_load: error loading model: unknown model architecture: 'zamba2'
0.00.126.905 E llama_model_load_from_file_impl: failed to load model
0.00.127.071 E common_fit_params: ... failed to load
client_rc=1
urllib.error.URLError: <urlopen error [Errno 111] Connection refused>
```

18 × "failed to load model" and 5 × fit-params errors in that batch. And its current job is
**CPU-only** (`-dev none -ngl 0 -t 4 --no-mmproj-offload`), i.e. it is holding the GPU-wide
lock while running something that does not use the GPU, against a model its binary does not
support.

**Not touched.** That is a peer's process. The goal's constraints say do not disturb other
workloads, and killing another agent's batch to win the box is not mine to decide. Recorded
here precisely so the decision is visible to whoever owns it.

**Consequence for task-8:** the trace cannot start while this holds the lock. Run 5 will time
out on `NPU_LOCK_WAIT` (900 s) and abort having started nothing — which is the correct
behaviour, and better than the pre-fix version that would have run unprotected.

## Where this leaves task-8

- harness: hardened against seven silent-failure modes, verified
- evidence in hand: **run 1** — 1 in-regime arm, 0 faults, under 15,457 evictions (15,353 unrestored). A **null**
- fresh run: blocked on `~/npu.lock`, held by a peer batch whose jobs are failing, with two
  other goals' workers also queued

The task-8 contract permits closing on the null, recorded as uninformative with its
clean-arm count. The reason is box contention — externally caused and now documented — not a
design choice and not a harness defect.

## Final measured state at the block (2026-10-12 13:26)

`~/npu.lock` queue, oldest first — the head does not drain:

| waited | pid | what |
|---|---|---|
| **6,376 s (1 h 46 m)** | 99828 | `flock ~/npu.lock ./batch2.sh` — **the head, and wedged** |
| 1,583 s | 816808 / 816800 | another goal's `npu-forge2-kvsplit` bisect |
| 1,479 s | 826939 / 826940 / 826913 / 826914 | another goal's `probe_model.sh` ×2 |
| 1,345 s | 845081 | `./batch2.sh` itself |
| 6:31 | 948371 | **our run 5** (`flock -w 900 8`), will abort at 900 s |

`batch2.log`: **7,185 bytes, unchanged over 30 s — wedged.** It has 18 × "failed to load
model" (`unknown model architecture: 'zamba2'`) and its current job is CPU-only
(`-dev none -ngl 0 -t 4 --no-mmproj-offload`).

**The queue is growing, not draining:** three unrelated workers are now parked behind a
batch that is failing and not progressing. That is why retrying is not a strategy — a fresh
attempt does not move closer to the head, it joins the queue behind a stalled owner.

## Correction to the first blocker note

The blocked summary says "three consecutive goal turns hit the same cause". That is right in
substance but understated the first two: they were plain contention (`repro336` at
103–111 GiB). **Today's is a wedged lock owner**, which is a different and worse problem —
contention eventually clears, a stalled batch does not. Recorded so the decision is not made
against the weaker description.

## CORRECTION + the real blocker: `~/npu.lock` is wedged by a dead holder

### What I got wrong

I previously recorded the blocker as "a wedged peer batch (pid 99828/99830) holds
`npu.lock` while its jobs fail". **That was wrong.** `ps` shows `flock` for both holders and
waiters, and I read elapsed time as evidence of holding. The kernel's own lock table says
otherwise.

### What the kernel actually says

`/proc/locks` for the inode behind `~/npu.lock` (`103:02:80249278`) — unindented line is the
**granted** lock, `->` lines are **waiters**:

```
10: FLOCK ADVISORY WRITE 826940  103:02:80249278 0 EOF      <- HOLDER
10: -> FLOCK ADVISORY WRITE 867745 103:02:80249278 0 EOF    <- waiting
10:  -> FLOCK ADVISORY WRITE 99830  103:02:80249278 0 EOF   <- waiting
10:   -> FLOCK ADVISORY WRITE 1155054 103:02:80249278 0 EOF <- waiting
```

**The holder `826940` no longer exists.** `/proc/826940` is gone, its whole group
(826913/826939/826940, `probe_model.sh`) has exited, yet the kernel still grants it the
lock. The three waiters are alive and parked in `locks_lock_inode_wait`:

| pid | state | wchan |
|---|---|---|
| 867745 | S | `locks_lock_inode_wait` |
| 99830 | Ss | `locks_lock_inode_wait` |
| 1155054 | S | `locks_lock_inode_wait` |

### It is reproducible, not a transient read

```
t=10s  HELD
t=20s  HELD
t=30s  HELD
box.lock: FREE      <- same filesystem, same mechanism: flock itself is fine
```

`flock -n` against `~/npu.lock` fails on every attempt while `box.lock` acquires immediately.
So this is specific to that lock file, and it is **stuck**, not busy.

### Why this matters beyond my task

`~/npu.lock` is the **box-wide mutex for GPU work** (§6g). Every GPU harness on this box
queues on it — mine, the peer batch, and two other goals' workers. With a dead holder, **all
of them are blocked permanently**, and no amount of retrying or waiting helps. This is not
contention that clears; it is a wedged mutex.

**Possible cause, stated as a hypothesis not a finding:** the holder was a `flock` in a
`systemd-run --scope` unit. If the scope was torn down while its flock was granted, the
granted entry can outlive the process in `/proc/locks`. That would make
`flock ~/npu.lock ...` inside a transient scope an unsafe pattern on this box — an advisory
lock whose holder can be reaped out from under it.

### What this means for task-8

The blocker is **not** a peer batch I should leave alone; it is a **stuck kernel lock
holding back every GPU consumer on the box**, including two other goals. Retrying (option c)
cannot work. Waiting (which I have done for three turns) cannot work.

I have not cleared it: `flock` offers no break mechanism from userspace, and the only ways
that occur to me are destructive (reboot, or `fuser -k` on the lock, which would kill three
innocent waiters' work). That is the user's call, not mine.

## The wedge is cleared by a clean reboot, and the campaign is finally running

**2026-10-12 14:03:53 — box rebooted** (user decision, option (b)). State after boot:

| check | value |
|---|---|
| `~/npu.lock` | **FREE** (was wedged 36+ min) |
| `~/.cache/lax-decode/box.lock` | FREE |
| stale `/proc/locks` entries for it | **0** |
| HRX0 | 0 processes |
| GTT | 26.6 GiB (baseline) |
| `Tctl` | 40.2 °C |
| `MemAvailable` | 90 GiB |

The dead holder `826940` is gone and the three waiters it had blocked are gone with it. Cost:
the three queued workers' in-flight work (a #315 arm, `batch3.sh`, a `bisect4.sh` worker),
accepted by the user.

**Run 6 (`315evict6`) launched 14:05:45 with the full corrected guard sequence — and it is the
first run to get through all four gates:**

```
box lock held 2026-10-12T14:05:45-03:00  (/home/bcloud/.cache/lax-decode/box.lock)
npu lock held 2026-10-12T14:05:45-03:00  (/home/bcloud/npu.lock) - the box-wide convention
GPU quiet: no foreign HRX job (waited as needed)
watcher pid 11922 -> /home/bcloud/wt/315evict6-evict/kfd_smi.tsv
```

`315-campaign.sh` and `315-paired.sh` are composing (8 pairs = 16 arms). This is the first
attempt in this task that reached composition rather than aborting at a gate, which is what
the harness work was for.

**Not yet evidence.** task-8 still needs an in-regime arm to *fault* while the watcher is
attached, or the null recorded with its clean-arm count. Nothing here changes that.

## Run 7 (`315evict7`) — first campaign to run properly in regime

After the reboot and the NOLOCK fix (§6h), run 7 passed all four gates and reached the regime:

| | |
|---|---|
| gates | box lock → npu lock → GPU quiet → watcher (all four passed) |
| composition | 4 HRX neighbours, GTT 27 → 80 → 98 → **104 GiB** |
| arm 1 (`p1c`) | `run.log: gtt_start 97GiB` — **in regime** |
| requests | 140 → 208 in 60 s (68/min) |
| evictions | 662 → 668 |
| deadlock | none (`do_wait`) |

This is the first run in the whole task that got past composition and into a scored
in-regime arm while the eviction watcher was attached. Everything before it aborted at a
gate or deadlocked: run 1 was truncated by `MINCLEAN=0`, runs 3–5 never cleared the locks,
run 6 self-deadlocked on the lock the driver re-opens.

**What it is for:** task-8 needs an in-regime arm to **fault** while the watcher runs, so the
fault can be correlated against the driver's own eviction events. 16 arms (8 pairs) is the
target; the observed in-regime rate is roughly 1 in 2, so expect ~8 in-regime arms.

**What it cannot do:** satisfy task-9, which is gated on a named mechanism no instrument has
produced. A clean run here is task-8 evidence, not the objective's fix+≥60-arm deliverable.

**If all 16 arms are clean**, the contract requires recording this trace as
**UNINFORMATIVE with its clean-arm count** — never as evidence about the mechanism.

## RUN 7 RESULT: an in-regime FAULT on Face B, correlated against its own evictions

**This satisfies the first branch of the task-8 contract.**

```
pair 3 arm l: FAULT nan=0 req=20 parse=0 gtt_start=97G utf8=0 alloc=0 hv52=0 hv56=0 (gtt_now=102G)
```

| quantity | value |
|---|---|
| arm | `pair 3 arm l` (the **fix** cell — diagnostics + HOST_COHERENT) |
| regime | **`gtt_start=97 GiB`** — inside the frozen `>= 95 GiB` gate |
| face | **B** — `http_failures: 1`, `first_failure: "HTTP 500"`, `truncated: true`, `nan=0` |
| requests | 20 served, 19 ok, then the 500 stopped the arm |
| detector.json | `{"requests":20,"ok":19,"utf8_failures":0,"parse_failures":0,"http_failures":1,"transport_failures":0,"truncated":true,"first_failure":"HTTP 500"}` |

### The failure, verbatim from the arm's server.log

```
1.21.180.831 W common_chat_peg_parse: unparsed Content-only output:  laptop too
1.21.185.082 W srv operator(): got exception: {"error":{"code":500,"message":"The model produced
              output that does not match the expected Conten... (truncated)
```

That is Face B exactly as the issue describes it: output that is **not valid UTF-8 / not
parseable**, surfacing as an HTTP 500 — not a NaN. `nan=0` confirms it is Face B and not
Face A.

### The correlation (task-8's actual deliverable)

Bound to the arm's own PID **exactly** (`tools/arm-pid.py` → `111371`, from the port in the
arm's own `server.log`; not guessed). Against the eviction stream, for that PID alone:

| event | count |
|---|---|
| `Q_EVICT` | **323** |
| `Q_RESTORE` | **357** |
| timeline rows | 681 → `~/wt/315evict7-evict/correlated-p3l.tsv` |
| trigger | `USERPTR` throughout |

So the faulting arm can be placed against **its own driver evictions**, which is the thing the
#123/#140 lead predicted would be necessary and which no earlier run could obtain (runs 3–6
never reached an arm; run 1's one in-regime arm was clean).

### What this does and does not establish — stated before the run finishes

**Does:** an in-regime arm faulted on Face B while its own PID was being evicted hundreds of
times, with both recorded on one clock. The correlation exists and is reproducible from the
paths above. That is what task-8 asked for.

**Does NOT:** show that the evictions *caused* the fault. A clean arm in this same run
(`p1c`, `p1l`, `p2l`, all `gtt_start=97G`, all `OK`) was also being evicted — the eviction
counts are not yet compared between faulting and clean arms, and the fault is intermittent
by construction. **Eviction is necessary-looking, not sufficient**, exactly as run 1 already
suggested with 15,357 evictions and zero faults.

**Must not be written up as:** "the evictions caused it", or as a mechanism. The mechanism is
still unnamed; this is the correlation evidence, which is the input to naming one.
