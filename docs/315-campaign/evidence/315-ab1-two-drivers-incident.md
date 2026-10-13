<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Incident: two drivers on one tag — arm B's first attempt (`315ab1b`) is discarded

Recorded because it is the same class of trap the issue has sprung before ("do not weaken or contaminate
the apparatus"), and because the discarded attempt must not be mistaken for a measurement.

## What happened

Arm B was launched at 10:22:33 under tag `315ab1b` (one driver, box lock held, patched binary, knob ON).
At 10:24:45 a **second** `315-disc.sh 4 315ab1b` session started on the same box, writing to the SAME
console (`~/wt/315ab1b-console.log`) and the same tag.

The second launch line was not issued by this session:

    cd ~/wt && BIN=$HOME/wt/315coherent-build/bin \
      EXTRA_ENV="GGML_HRX_TENSOR_FP=tail:16 GGML_HRX_TENSOR_FP_NANONLY=1 GGML_HRX_COHERENT_READ=1" \
      setsid nohup bash 315-disc.sh 4 315ab1b > ~/wt/315ab1b-run.log 2>&1 < /dev/null

It carries the exact arm-B parameterisation that the pre-registration and the session-state file name,
so it was almost certainly a **peer #315 session** on the shared box picking up the published tag/name,
rather than a random process. No cron/timer/rsync hook references `315-disc.sh`; `~/.bash_history`'s
only 315 line is an older `evictonly` one-liner.

## Why it is discarded rather than repaired

Two drivers on one tag make the record ambiguous by construction:

- the two sessions share one console file, so their per-arm lines interleave and the driver's
  `grep -c` summaries double-count;
- both drive the same eviction rig and the same GPU, so they contend for the regime and for HRX0;
- the `summaries` ("cell e: OK=… FAULT=…") are computed by grepping the shared console, so **neither**
  session's summary is trustworthy once a second writer exists.

There is no way to attribute an arm to a session after the fact, so the correct disposition is discard,
not salvage. The tag `315ab1b` is retired and must not be cited.

## What was NOT lost

The second session never reached an arm: it wrote only its header line
(`=== disc session 2026-10-13T10:24:45-03:00 tag=315ab1b rounds=4 gtt_before=99G tctl=82C …===`) plus
its own neighbour/regime lines, and was stopped before any `round N arm X:` line appeared. So:

- no arm was scored under the contention;
- **arm A (`315ab1a`) is untouched** — it completed at 10:20:23, before the collision, under a single
  driver, and its result (`docs/315-campaign/evidence/315-ab1-arm-A-result.md`) stands;
- the patched binary and the patch file are unaffected.

## Disposition

- Arm B re-run under a **fresh tag `315ab1b2`** at 10:25:24, same binary and same rig, single driver,
  box lock held.
- The re-run's launcher carries a **start guard** that refuses to begin if any other 315 driver or
  patched-build server is alive, so the collision cannot recur:

      n=$(pgrep -fc "315-disc.sh|315coherent-build/bin/llama-server"); [ "$n" -gt 0 ] && exit 4

- Lesson re-recorded (extends the existing "never build from a copied build dir" trap): **one tag, one
  driver, on a shared box.** Publish a tag only after the run it names has started, or use a tag that
  cannot be guessed and re-used by a peer session.
