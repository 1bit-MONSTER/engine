# The missing measurement, and the one-line change that enables it

## The gap

Run 7 (`315evict7`) produced 11 in-regime arms with **3 faults across both faces** — the
strongest task-8 evidence in this task. But it cannot answer the question that would turn
correlation into a start on causation:

> **Were the faulting arms evicted more than the clean arms?**

I can state the faulting arm's eviction count and **not** the clean arms', for a concrete
reason: the KFD SMI stream's `PROC_START` rows carry no port (`raw=<ctx> llama-server`), so
once an arm's `llama-server` exits there is no exact way to bind it to a PID after the fact.

| arm | verdict | own evictions |
|---|---|---|
| `p3l` (port 19508) | FAULT (Face B) | **323 Q_EVICT / 357 Q_RESTORE** for pid 111371, bound **while live** |
| `p1c`, `p1l`, `p2l`, `p4c`, `p4l`, `p5l`, `p6l`, `p8l` (OK) | clean | **not recoverable** |
| `p7c`, `p7l` | FAULT (Face A) | **not recoverable** — they never had a live bind taken |

Even the two Face-A faulting arms are unrecoverable, so the comparison currently has **one**
data point on the faulting side and **zero** on the clean side.

`tools/arm-pid.py` deliberately **refuses** to guess for a finished arm, because
`correlate.py`'s own fallback ("most frequent eviction subject") was observed landing on a
**dead** PID — a correlation against the wrong process fails silently, which is worse than
none.

## The change

Record the arm's server PID into its own directory at arm start, while it is genuinely live.
One functional line, read-only, no behaviour change:

```diff
   for t in $(seq 1 300); do curl -sf -m 5 localhost:$port/health >/dev/null 2>&1 && { up=1; break; }; sleep 1; done
+  # engine#315: record the arm server PID while it is live. The KFD SMI stream PROC_START
+  # rows carry no port, so a finished arm is unrecoverable and faulting-vs-clean
+  # eviction comparison becomes impossible. Read-only, one line, no behaviour change.
+  pgrep -f "port $port" 2>/dev/null | head -1 > "$O/server.pid"
   local g0; g0=$(gtt); echo "gtt_start ${g0}GiB mode ..." > "$O/run.log"
```

Placed immediately after the health wait, so the server is definitely up and the port is
bound. It lands in `$O/server.pid` beside the arm's existing `run.log` / `server.log` /
`detector.json`.

## Verification (both done before proposing it)

**1. Applied to a copy, syntax checked, nothing else changed.**

```
$ awk '<insert after the health-wait line>' ~/wt/315-paired.sh > /tmp/315t.sh
$ bash -n /tmp/315t.sh && echo "SYNTAX OK"
SYNTAX OK
$ diff ~/wt/315-paired.sh /tmp/315t.sh
74a75,78
>   # engine#315: record the arm server PID while it is live. ...
>   # rows carry no port, so a finished arm is unrecoverable ...
>   # eviction comparison becomes impossible. Read-only, one line ...
>   pgrep -f "port $port" 2>/dev/null | head -1 > "$O/server.pid"
```

Exactly 4 added lines (3 comment + 1 functional), no other hunk.

**2. The capture logic tested standalone, with a negative control.**

```
$ bash -c 'exec -a "llama-server --host 127.0.0.1 --port 19508 --no-webui" sleep 300' &
$ pgrep -f "port 19508" | head -1
333423                       <- matches the launched pid exactly
$ ... > /tmp/pidtest/server.pid ; cat it
333423                       <- the file write works
$ pgrep -f "port 99999" | head -1
                             <- empty, correct: does not match an unrelated process
```

## Why this is not applied to the frozen driver

`315-paired.sh` is part of the pre-registration. Editing it mid-campaign would invalidate the
arms already run, and the goal's constraints forbid weakening or altering the frozen
instrument to obtain a result. So this is a **proposed** change: it should be applied *before*
the next campaign, whose arms can then carry both verdicts and eviction counts.

Until it is applied, the honest statement stands: **the faulting arm was evicted heavily, the
clean arms ran in the same regime and were presumably evicted too, and nothing yet separates
them.** One faulting data point is not a comparison.

## Attempted the faulting-vs-clean comparison on run 7's existing data — it does not work, and here is exactly why

I tried to get the H2 discriminating test (do faulting arms show more evictions than clean arms?)
out of data already on disk, without a reboot or a new run. It fails on two independent
grounds, both now measured rather than assumed.

### 1. An arm server's eviction count is contaminated by its post-arm lifetime

`llama-server` PIDs and their `Q_EVICT` counts across run 7's complete SMI stream:

```
  1674  pid 38896     235  pid 179876     206  pid 141991      65  pid 88672
  1594  pid 33195     217  pid 198847     191  pid 122598      15  pid 205541
  1589  pid 35975     217  pid 160820     212  pid 41825      144  pid 65395
  1461  pid 111371
```

`111371` is the **p3l faulting arm** — I bound it live via `arm-pid.py` from the port in its own
`server.log`. Its count (1,461) is in the same band as the *long-lived neighbours*, not the
other arm servers (15–599). The reason is in the record: when I went to correlate it I found that
process **still running ~40 minutes after its arm had been scored**, and killed it by hand.

So an arm server keeps being evicted after its measurement window closes, and its total is a
function of how long it happened to survive — not of what happened during the arm.

### 2. The five long-lived neighbours cannot be separated from the eleven arm servers

Run 7's stream has **19** `llama-server` `PROC_START` events: five neighbours (which live for the
whole session and legitimately accumulate the largest counts) plus eleven arm servers, plus
three more whose role I cannot determine from the stream. Nothing in the SMI stream records which
port or which arm a PID belongs to — that is the same limitation recorded at the top of this file.

The two effects compound: the top four counts (1,589–1,674) are almost certainly neighbours, the
faulting arm sits inside that band because it outlived its arm, and there is no clean way to
assign a window to any of them after the fact.

### Conclusion

**The comparison requires the PID to be captured at arm start, while the arm is live.** That is
what the one-line change at the top of this file does, and it is why the change exists rather than
being an optimisation. Running the comparison against historical data is not merely inconvenient —
it is invalid, and would have produced a confident wrong answer had I not checked which PIDs the
counts belonged to.

Recorded so the next attempt applies the capture *before* the campaign it wants to analyse.

## VERIFIED LIVE: the PID-capture pipeline works on the campaign of record

Checked while `315cwsr70` was on its first arm:

```
  p1c: server.pid = 564373
  tool: 315cwsr70-p1c-GLM-4.7-Flash-Q4_K_M   pid 564373   22 evictions in window
```

So the whole chain works on real data:

1. the patched driver wrote `$O/server.pid` at arm start, while the server was live;
2. `315-evict-per-arm.py` resolved that PID to its `PROC_START` in the SMI stream;
3. it bounded the window (this PID's start → its `PROC_END`, else the next arm's start) and counted
   **only that PID's** `Q_EVICT` rows inside it — 22 for the arm in progress.

The tool's "no in-regime arms" line at that moment was correct: the arm had not yet been scored, so
its verdict and `gtt_start` were unknown. It is the same guard that refuses to invent a number when
the PID is absent — as it did, correctly, on the mis-sized 48-pair run and on run 7.

**This is the difference between the failed retrospective attempt and the live one.** Run 7's counts
were unusable because a finished arm's PID is unrecoverable and its post-arm lifetime contaminates
the total (1,461 evictions for a fault arm that outlived its arm by 40 minutes). Here the PID is
captured while the arm runs, so the window is exact and the count is attributable.

The H2 discriminating test — *do faulting arms show more evictions than clean arms in their own
windows?* — can therefore be run on this campaign's data as soon as it has both kinds of arm.
