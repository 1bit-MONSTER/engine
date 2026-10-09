# engine#315 — mechanism statement (task-3) and fix direction (task-4)

Status: **candidate mechanism whose premise is measured and whose predicted observable is confirmed; the
causal (rate) link is still pending** because it needs the box to itself for tens of arms.

## The mechanism

The sampler reads the logits **directly out of a host-visible HRX buffer that is not host-coherent**. The
generic `ggml_backend_tensor_get_async` short-circuits for host buffers (plain memcpy, no backend call), so
no copy appears anywhere in the backend's own transfer path, and nothing in that path guarantees the
device's write into that memory is visible to the CPU reader. That is consistent with the fault's shape:
whole-vocabulary damage, intermittent, and sensitive to allocation/pressure timing.

## Evidence (all measured 2026-10-09, on the pinned revision `e44c9d01a4d5`)

| observation | value | how |
|---|---|---|
| logits buffer allocation | `type=HRX0_HOST host_visible=1 direct_host_binding=0 memory_type=0x52 size=619520` | `GGML_HRX_BUFFER_TRACE=1`, one load + three decodes |
| size identity | `619520 = 154880 × 4` — exactly the vocabulary the sampler guard reports | same trace vs the guard's `count=154880 of 154880` |
| no copy for the logits | zero logits-sized download records; the 146 records are per-layer weight leases (884736 / 2211840 / 5898240, each ×47) | `GGML_HRX_DOWNLOAD_FINGERPRINT=1` |
| the copy path is ordered | `hashes_equal=1` on all 146 records (probe read ≡ real copy) | same, plus `hrx_stream_synchronize` precedes the copy at `runtime/host-memory.cpp:1350` |
| the deferred writeback route is cold | 0 `[hrx-wb-fp]` records with the flag on over a complete 66/66 arm | v1 instrumentation |
| coherence is gated on a knob | `GGML_HRX_USE_UNIFIED_MEMORY=1` → `direct_host_binding=1`, `memory_type=0x56` = `0x52 \| 0x04` (`HOST_COHERENT`) | same trace, second load |
| host-visible buffers are persistently mapped | `HRX_BUFFER_USAGE_MAPPING_SCOPED \| ..._PERSISTENT` requested for them (`ggml-hrx.cpp:319`) | code, pinned |

**Predicted before it was run, then confirmed:** the knob flips exactly the coherence bit, on exactly the
buffer family the logits come from. That is the experiment the mechanism had to survive, and it did.

## Ruled out, with evidence

- **Deferred-writeback publish window (H1)** — the route is cold (0 records over a full arm).
- **Copy/race in the download path** — `hashes_equal=1` everywhere, and the sync precedes the copy.
- **"Logits are host-staged"** (the recorded hard negative) — upheld, and now explained: there is no copy
  at all, because the buffer is host-visible and the generic layer short-circuits.
- **Arena knobs / transient reuse** — falsified on the record (15 paired arms; 77-arm tally).
- **Pin/composition (AMD core vs our `ggml-hrx`)** — the landed rate study found no measurable difference
  (Fisher exact p = 1.000).

## Not yet ruled out (stated so it is not overclaimed)

- **The causal link.** Coherence controls the property; whether it removes the *fault* is a rate question at
  ~5 %/arm, so it needs tens of arms with the box to itself. A self-starting launcher
  (`docs/315-campaign/315-when-free.sh`) waits for the machine and runs the instrumented hunt; the
  differential itself is `GGML_HRX_USE_UNIFIED_MEMORY=1` vs unset at the same regime.
- **The peer session's candidate** (AQL block-transition dose-response, `HRX_AQL_BLOCK_SIZE` 262144 vs
  32768) is still in flight on the same box and must be accounted for in any final statement.
- **Face B (invalid UTF-8 / format-500) was never observed in this session**, so no mechanism is claimed for
  it. Its cause may be entirely different from Face A's.

## Fix direction (task-4) — pre-committed by outcome, minimality first

If coherence removes the fault, the fix is a **scoped visibility guarantee on the output path**, not the
global toggle:

1. give the host-visible buffer type used for outputs `HOST_COHERENT` — i.e. drop the `direct_host_binding`
   gate for that type only (`ggml-hrx.cpp` `buffer_alloc`), leaving the rest of the memory policy alone;
2. or issue the runtime's explicit visibility operation (flush/invalidate) for those buffers before the host
   reads them;
3. or stop allocating the logits host-visibly at all (device-local + one explicit copy), which removes the
   class of window at the cost of one copy per token.

Whichever is chosen, it must be forward-only: a commit on `1bit-MONSTER/llama.cpp`, then an engine pin move
whose previous pin is an ancestor of the new one (`git merge-base --is-ancestor` must be true).

**Not a fix:** `GGML_HRX_USE_UNIFIED_MEMORY=1`. A configuration toggle that trades coherent memory for
performance is not a resolution, and the pre-registration's campaign must pass with the default
configuration.

## Reproducing the two measurements

```bash
# premise (one load + three decodes, no regime needed)
UNIFIED= LOG_OVERRIDE=/tmp/315trace-default.log bash ~/wt/315-trace-premise.sh
UNIFIED=1 LOG_OVERRIDE=/tmp/315trace-unified.log bash ~/wt/315-trace-premise.sh
# then: grep 'hrx-buf' /tmp/315trace-*.log | grep 619520
```
