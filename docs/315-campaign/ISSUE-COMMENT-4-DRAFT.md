## engine#315 — final record (DRAFT: the decisive result is the only placeholder)

<!-- Everything below is written from evidence already on disk and already verified in this session. The
     placeholder marked TODO-RESULT is filled from ~/wt/315gsync2-console.log (and, depending on its branch,
     from 315fp2 or 315dose). Do not post with the placeholder still in it. -->

Status: **#315 stays OPEN.** This comment records what is established, what is falsified with its own evidence,
the decisive experiment's arm-by-arm counts, and the remedy. Every number is tied to a re-runnable command.

### 1. What the fault is, at the resolution the instruments can support

An in-regime Face-A event, captured twice with the guard's own line:

```text
2.53.659.050 E common_sampler_sample: HRX returned NaN logits: count=154880 of 154880, index range [0..154879], contiguous=yes, continuing=no sentinel=0
```

`315row` (2026-10-10) scored five arms, **every one at `gtt_start=98G`** — inside the frozen pre-registration's
`>= 95 GiB` gate — with **two faults** (`nan=1`, each aborting at `req=34` of 66) and three arms completing `66/66`:

```sh
ssh strixhalo 'grep -E "^  pair" ~/wt/315row-console.log'
```

That is 2 of 5 in-regime arms under the by-name instrument, against the shipped pin's recorded 2/37 = 5.4 %. The
instrument is an inducer (sections 39/41), which is why no rate is quoted from these arms and none is
acceptance-eligible; it is also what made five arms informative at all.

**The locating instrument was withdrawn, by its own output.** The by-name fingerprint reads graph tensors out of a
**host-visible, non-coherent 44 MB arena** (`44048640`, from `GGML_HRX_BUFFER_TRACE`), and
`ggml_backend_tensor_get` on a host-visible tensor is a plain `memcpy` with no synchronisation - so its lines are
not a snapshot of one instant. The two faulting arms prove it against themselves: `ffn_moe_weighted-4` reads
`nan=81880 of 81920` while `ffn_moe_out-4`, its own sum, reads finite. My earlier per-node attribution is
**withdrawn** (mechanism-doc section 53), and the repair is written and gated (`tools/patch-fp-sync.py`,
`315-fp2.sh`).

### 2. The decisive experiment (arm-by-arm)

TODO-RESULT

### 3. Falsified, each with its own evidence

| candidate | result | where |
|---|---|---|
| memory coherence is the carrier | the `HOST_COHERENT` patch was live (`hv52=0 hv56=1`) and the cell still faulted: 9 clean + 1 fault in 10 | `evidence/315-face-landscape.md`, mechanism §46 |
| consumer-side ordering is the carrier | `GRAPH_SYNC` after every replay: 4 of 6 arms below the gate, the one in-regime arm faulted | same file, mechanism §46 |
| the read path is unordered because nothing waits | source: `llama_get_logits_ith` synchronises (`src/llama-context.cpp:3729`), the wait ends in **`iree_hsa_signal_wait_scacquire`** (`runtime/src/iree/hal/drivers/amdgpu/semaphore.c:329`), and `stream->timepoint` is advanced *after* each submission | mechanism §§55-56 |
| the expert/partition routing tables go stale across graphs | both are rebuilt by producer dispatches inside the same plan (`dispatch-mul-mat-id-common.h:342-368`) | `evidence/315-moe-routing-routed-ffn-static.md` §3 |
| the fused `mul_mat_id_swiglu` path loses `route_stride` | that defect is real and **already fixed in every tree that faults** (3 + 10 occurrences of `route_stride` in both) | the same file, §7 |
| GLM's decode splits into two 128 KB AQL blocks | the queue profile reads `blocks 1`, `block-boundary gaps 0 us over 0`, at the default and at `HRX_AQL_BLOCK_SIZE` 262144 and 524288 | `evidence/315-queue-profile.md` |
| the pin move introduced the fault in the routed FFN | the whole `ggml/src/ggml-hrx` tree differs between the clean bound `f2099e9b7` and the pin by **one file**, the decode-split FA kernel (+102/-2) | `evidence/315-differential-artifact.md` |
| the engine's MoE expert-streaming cache | Vulkan-only; the arms load into GTT through HRX and never enter it | `evidence/315-moe-routing-routed-ffn-static.md` §6 |
| the block-size knob reaches the block transition | it changed neither the profile nor throughput (pp/tg 0.9965-1.0073) | `evidence/315-queue-profile.md` |
| the t2 hard negative (logits are not host-staged) | upheld, and explained: no copy exists because the buffer is host-visible and the generic layer short-circuits | `pm-salvage/issue-315/TASK2-INSTRUMENT-RESULT.md` |

Still on the record as **not** falsified: the regime's own contribution to *when* the window opens (unbarriered
dispatches plus a saturated queue: 92 of 612 dispatches carry no barrier, `busy 14288 us` of a `15348 us` span),
and the possibility that the corruption is computed, which is what the ordered locator would settle.

### 4. Regime dependence, stated as far as the evidence goes

The fault is absent below ~80 GiB in the recorded population and appears from ~95 GiB up, and the composition
arithmetic is now measured rather than assumed: neighbours at 70 GiB put arms at `gtt_start` 87-99 GiB (4 of 6
below the gate) while neighbours at 80 GiB put them at 98 GiB, with the detector contributing the remainder up to
the ~110 GiB upper bound. Two load-bearing tooling corrections came out of that: every probe wrapper had been
overriding the composition target **downward** to 70 GiB (`REGIME_LO=${REGIME_LO:-70}`, against the module's own
default of 80), which is why one session under-shot the gate while another reached 98 GiB with the same harness;
and a *clean* arm can leave its server scope resident, so GTT reached 113.5 GiB against a ~80 GiB composition
until the scope was stopped. Evidence: mechanism §48, `evidence/315-queue-profile.md` appendix.

### 5. Remedy

TODO-REMEDY — to be written from the decisive result. A landed fix is optional for this issue; a documented
limitation is acceptable once the mechanism is proven, and either way the text goes beside the mechanism in
`KNOWN-ISSUES.md`, which already carries the superseded reading clearly marked.

### 6. Commands

```sh
ssh strixhalo 'grep -E "^  pair" ~/wt/315row-console.log'                     # the in-regime capture
ssh strixhalo 'grep "HRX returned NaN logits" ~/wt/rt-det/315row-p1c-*/server.log'
ssh strixhalo 'cat ~/wt/315gsync2-console.log'                                # the decisive experiment
ssh strixhalo 'cd ~/wt/llama-315copy && git diff --stat f2099e9b7..HEAD -- ggml/src/ggml-hrx/'   # the differential
ssh strixhalo 'head -3 ~/wt/hrxsys-prof-run.log'                              # the queue profile
```
