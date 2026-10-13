<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

engine#315 evidence: pre-registration for the coherence-read fix A/B, written and committed BEFORE
the run so the reading cannot be chosen post hoc.
-->

# Pre-registration: does the `GGML_HRX_COHERENT_READ` fix stop the forced-eviction fault? (`315ab1`)

Written 2026-10-13 ~10:00, committed before the run.

## The hypothesis under test

**H-fix.** `buffer_get` reads a host-visible, non-coherent mapping with a bare `memcpy`
(`ggml-hrx.cpp:250-255`). Routing that read through `hrx_synchronous_d2h` — a queued HAL DOWNLOAD —
instead of the raw pointer read removes the stale-view window, so forced KFD eviction should stop
producing the fault.

Mechanism chain this rests on (all recorded, with the falsified predecessor noted):

1. the logits buffer is host-visible (`type=HRX0_HOST host_visible=1 ... size=619520` in the issue's own
   allocation trace) and, because `use_direct_host_bindings` defaults off, its memory_type is
   `HOST_LOCAL|DEVICE_VISIBLE` = **`0x52`**, which carries `0x02` HOST_VISIBLE but **not** `0x04`
   HOST_COHERENT (`hrx_runtime.h:194-202`);
2. it is `mmap`'d (`hrx_buffer_map`), so `context->base != FAKE_PTR_BASE` and `buffer_get` takes the
   **plain `memcpy`** branch;
3. no cache-invalidate API exists in the runtime (`hrx_runtime.h` has only `hrx_stream_flush`, which is
   command-buffer semantics), so nothing on that path refreshes the CPU's view;
4. `llama_get_logits_ith` does call `ctx->synchronize()` and §55/§56 traced that to
   `hsa_signal_wait_scacquire` with a timepoint covering the last submission — so **stream** ordering is
   present; what is absent is **cache coherence on the CPU read**;
5. forced eviction of in-flight work is what makes the mapping's contents differ from what the CPU
   cached, which is why the eviction cell faults at 70 % (7/10) while quiet (0/10) and CPU-pressure
   (0/12) cells do not.

Consistency check this predicts: the earlier coherence-**attribute** patch
(`315-coherence-fix.patch`, `HRX_MEMORY_TYPE_HOST_COHERENT` made unconditional) was falsified as a fix
(9 clean + 1 fault / 10). Setting an attribute on the allocation does not add an invalidate to the
**read**, so that patch could not have fixed a read-path defect. The failure is expected under H-fix.

## Design — one binary, one env var, the eviction rig

    driver : ~/wt/315-disc.sh <ROUNDS> <TAG>   (the SAME frozen three-cell design as disc1/disc2)
    BIN    : ~/wt/315coherent-build/bin        (the patched build; verified to carry the knob)
    cells  : q quiet / e evictonly / c CPU-pressure, rotating per round
    A arm  : EXTRA_ENV without the knob        -> the pre-fix behaviour (71% expected, from disc2)
    B arm  : EXTRA_ENV with GGML_HRX_COHERENT_READ=1 -> the fix path

Both arms use the SAME binary and the SAME rig, so the only difference is the knob. Cell `e` is the
test cell; `q` and `c` remain internal controls.

Rounds: 4 per arm (≈16 arms, ~40 min per arm at the measured pace) unless the result is already clear.

## Prediction, and the four branches, fixed BEFORE the run

    B (knob on) faults at ~0 in cell e, A (knob off) faults at ~50-70% in cell e
        -> H-fix SUPPORTED. The read-path incoherence is the cause, and the one-hunk change is the fix.
    B ~= A (both fault in cell e at a similar rate)
        -> H-fix REFUTED. The memcpy is not the site; the stale data enters elsewhere.
    B faults but A does not
        -> CONTRADICTED in direction; report as such (it would suggest the fix makes things worse,
           which would itself be informative about the transfer's semantics).
    both clean in cell e
        -> UNINFORMATIVE. Must NOT be read as evidence the fix works: at the ambient ~5-7 % rate a
           clean eviction cell is possible by chance, and it would also mean the rig failed to induce.
           Re-run or report as uninformative, per the pre-registration discipline used for disc2.

## What this does and does not settle

 - It tests the CAUSE for the first time with an intervention plus a candidate repair.
 - It does not by itself satisfy criterion 2's "forward-only change landed on a branch/PR": that needs
   the change prepared on the fork against the current pin, with the ancestry check, and the >=60-arm
   in-regime campaign on the fixed build. Both remain after this test.
 - It does not test for regression; a separate control (dense model + MUL_MAT_ID op-test) is criterion 4.
