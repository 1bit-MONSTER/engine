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
