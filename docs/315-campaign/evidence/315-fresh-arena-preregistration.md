# Pre-registration: does NEVER-RECYCLING the transient arena stop the fault? (`315fresh1`)

Written before the run.

## The instrument, read at source
`runtime/transient-arena.cpp:70`: `GGML_HRX_DIAGNOSTIC_FRESH_TRANSIENT_ARENA` makes
`ensure_capacity_locked` **retire** the current arena buffer and allocate a fresh one on every grow,
instead of releasing and reusing (`:82-86`). It also **skips the `hrx_stream_synchronize`** that the
production path performs before releasing (`:85-92`).

So it is a direct test of the WRITE-SIDE / REUSE question: if the corruption depends on the arena holding
prior content (the stale-buffer class), then never recycling should change the fault rate. If the fault is
unaffected, the corruption does not need reuse of that arena.

## Hypothesis
**H-fresh.** If the fault requires the transient arena to hold stale content from a previous use, then an
arena that is never reused cannot supply it, and the fault rate falls.

## Prediction, and the branches, fixed BEFORE the run
    cell e faults at a clearly LOWER rate than the no-knob baseline (~85% pooled)  -> H-fresh SUPPORTED
    cell e faults at ~the same rate                                              -> H-fresh REFUTED
        (the corruption does not come from arena reuse; the write-side/intermediate
         candidates must be sought elsewhere)
    cell e CLEAN entirely                                                        -> H-fresh STRONGLY supported,
        and the next step is to find the reuse site

## Caveats that must be stated with the result
 - the knob ALSO removes a stream synchronize, so a rate change would not by itself distinguish
   "reuse matters" from "that sync matters". A clean result would therefore need a follow-up that adds
   the sync back while keeping the fresh allocation.
 - the retired buffers are retained, so memory grows across the arm; arms may OOR or die for that reason
   and such arms are INVALID, not clean.
 - this is `HRX_MEMORY_TYPE_DEVICE_LOCAL` (0x30), not the 0x52 host-visible family already closed. It
   tests the transient arena specifically.

## Design
    driver : ~/wt/315-disc.sh 2 315fresh1   with BIN=~/wt/315coherent-build/bin
    EXTRA_ENV adds GGML_HRX_DIAGNOSTIC_FRESH_TRANSIENT_ARENA=1 to the usual fp pair
    cells  : q / e / c as always; cell e is the test cell, q and c remain controls
