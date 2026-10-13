<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# The last in-scope probe was UNRUNNABLE — its build predated its patch

`315-idprobe.sh` is the one in-scope lead with a pre-registered probe that had never been run: it answers
*"are the route ids a clean permutation, and is the NaN in `ffn_moe_weights` an aliased read?"*, with three
branches fixed in advance (`315-id-producing-kernel.md`). The record explicitly flags it as **untested** and
says not to cite its evidence file as a live result.

Attempting to run it exposed why it never produced data:

    build  ~/wt/315copy-build/bin/libggml-hrx.so     10-09 17:38
    source ~/wt/llama-315copy/.../command-program-executor.cpp   10-10 11:04

**The build predates the patch by ~18 hours.** Checked by looking for the instrumentation's own markers
(`dup_pairs`, `oor=`) in the built library: **0 hits**. So running the probe as staged would have produced a
server that emits no id data at all — the probe's own readout would have been empty, and an empty readout
looks like "ids clean", which the pre-registration would have read as *"the argsort candidate dies and the
fault is downstream"*.

That is a **silent false negative waiting to happen**, and it is the same class as the other harness defects
this session: the instrument is not actually loaded, and the failure is indistinguishable from a result.

## What was done

- Verified the build cache is configured against the patched tree (`CMAKE_HOME_DIRECTORY=~/wt/llama-315copy`,
  `ggml_SOURCE_DIR=.../llama-315copy/ggml`), so an **incremental** rebuild is safe and picks up the patch —
  this is *not* the copied-build-dir trap, where the cache points at a different tree than intended.
- Started `cmake --build ~/wt/315copy-build -j 8` in the background.
- Chained the probe behind it, **with a guard**: the chain refuses to run the probe unless the rebuilt library
  actually contains the ids markers (`dup_pairs`/`oor=`), so a blind run cannot occur and a silent false
  negative cannot be reported as a clean-ids result.

## Why this was worth fixing rather than abandoning

It is the last in-scope family with a **falsifiable discriminator already designed**: if a faulting arm shows
`oor > 0` or `dup_pairs > 0`, the argsort's rank computation is the site; if the ids are clean while the
matmuls are NaN, that candidate dies. Either outcome closes the family, and the second would leave the cause
pointing at the table builders or the matmul descriptor — still in scope.

The probe is diagnostic, not acceptance: no criterion changes and no arm from it is eligible.
