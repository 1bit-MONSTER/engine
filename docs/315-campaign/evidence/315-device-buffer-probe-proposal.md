<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0
-->

# Proposed + pre-registered: the DEVICE-buffer producer probe — the instrument that can reach the onset tensor

Written because the host-only producer probe (`315prod`, queued) provably cannot answer the question the
positional finding raises. No build and no run are attempted here; this is the design and its readout, fixed
in advance so the run is a test rather than a search.

## Why the existing probe cannot reach the onset

`producer_probe_target` requires `ggml_backend_buffer_is_host(t->buffer)` **and** a size in 256 KiB … 8 MiB.
Against the recorded inventory that admits exactly one tensor — the 619,520-byte **logits** region. The
`hrx-producer` dump prints `host=0` for every layer node (`q-0`, `norm-0`, `q_nope-0`, …), so the layer
residuals are device-side and can **never** be armed by it.

But the positional result (`315-one-row-at-the-front-of-the-buffer.md`) puts the onset at exactly such a
device tensor: **one `n_embd`-wide row, at row 0 or 1, in a layer residual**. So the decisive question —
*did the device write that row?* — is one the host-only instrument is structurally unable to ask.

## The instrument is feasible with primitives already in the fork

    ggml-hrx.cpp:442  static void backend_set_tensor_async(backend, tensor, data, offset, size)
    ggml-hrx.cpp:466  static void backend_get_tensor_async(backend, tensor, data, offset, size)
    ggml-hrx.cpp:542  backend_set_tensor_async,      <- registered in the backend iface

Both take an arbitrary HRX tensor (device **or** host) through `ggml_backend_hrx_tensor_binding`, and each
falls back to a synchronous copy when the host side is not a registered host buffer. So the symmetric probe is
the existing pattern with the `is_host` clause dropped:

- **before** the graph: `ggml_backend_tensor_set(node, pattern, 0, ggml_nbytes(node))` — fill a **device**
  tensor;
- **after** a drained stream: `hrx_stream_synchronize` then `ggml_backend_tensor_get(node, buf, 0, nbytes)`;
- classify, exactly as the host probe does.

## Design decision: a FINITE sentinel, not a NaN payload

The host probe fills `0x7FC0DEAD`, which is a NaN. For a **layer residual that is also an op's input** that is
a serious confound: if the op fails to overwrite it, the injected NaNs propagate through the rest of the layer
and produce *the very symptom under study* — the run would be diagnosing its own sentinel, and every
downstream observation in that arm would be unusable.

Use a **finite** pattern instead (`0xDEADBEEFu`), so:

- `sentinel == n` → the op did not write the tensor at all;
- `sentinel == 0` → the op wrote every word, and any NaN present is **genuine**;
- `0 < sentinel < n` → the op wrote part of it — and this is the interesting case, because a partially
  written row-oriented buffer is exactly what "one row corrupted at the front" would look like from the
  writing side.

The collision risk (a legitimate word coincidentally equal to `0xDEADBEEF`) is stated and is ~0 for a single
fixed 32-bit pattern over 2⁵² words, and would in any case *reduce* the measured `sentinel` count.

## The run

    BIN=<a build carrying the dropped-is_host probe, see below>
    EXTRA_ENV="GGML_HRX_PRODUCER_PROBE=2 GGML_HRX_PRODUCER_FINITE=1 \
               GGML_HRX_TENSOR_FP=l_out-0,…,l_out-47 GGML_HRX_TENSOR_FP_NANONLY=1"
    ~/wt/315-disc.sh 8 315dev

**This needs a build**, which is why it is a proposal and not a queued run: the host-only clause and the NaN
payload are both compiled in, so `~/wt/315probe-build` cannot do this. The change is small (drop one clause in
`producer_probe_target`, add a finite-payload branch in `producer_probe_fill`) and is **diagnostic-only**, so it
cannot weaken any guard.

**Target selection** must be widened deliberately, and the selector is the hard part: the band should be
`n_embd × T × 4` for the arm's T, and restricted to residual-family nodes so the probe does not arm every
activation in the graph. The pre-registered selector is **any device tensor whose byte size is within ±1 % of
`2048 × T × 4`**, which for a prefill chunk targets the residual-stream tensors and nothing else. If that arms
nothing, the run is INVALID and the selector — not the conclusion — is revised.

## Readout, fixed in advance

| observation | verdict |
|---|---|
| `sentinel == 0` on the onset tensor in a faulting arm, with `nan > 0` | the device **did** write the whole tensor, and the NaN is genuine ⇒ the corruption is *not* an unwritten-buffer/visibility defect; the writer is an op |
| `0 < sentinel < n`, and the unwritten words are **contiguous** | the op wrote a prefix and left the rest — a row-oriented write that did not cover the region, which is a mechanism |
| `sentinel == n` on the onset tensor | the tensor was never written at all: the consumer reads pre-fill content ⇒ the visibility/staleness family, at the point of production |
| `sentinel == n` on a tensor that is *not* the reported onset | the probe armed the wrong tensors; INVALID, fix the selector |
| no `armed` line | INVALID, widen the selector |

## Confounds, declared before the run

- **Filling a live tensor perturbs the graph.** Even a finite pattern changes downstream values for that
  tensor; the classification at the probed tensor is still valid (it asks only whether the op overwrote the
  words), but the arm's *other* observations are not comparable to a natural arm.
- **The probe adds a host⇄device copy per hooked graph**, which the record already establishes is itself an
  inducer. So this run characterises *the probe's regime*, not the ambient one.
- **`n_embd × T × 4` assumes T from the arm, and T varies within an arm across chunks.** The selector must
  therefore be size-band based, not a single number.
- No rate: a per-graph classifier, no denominator.

## What this would decide that nothing else on the record can

It separates the two families the goal names and the engine cannot distinguish — *"the device wrote these
bytes"* versus *"a restore left a pre-eviction copy" / "nothing wrote them"* — **at the tensor the positional
evidence points at**, rather than at the logits region the host-only probe is confined to.

---

## Correction to this proposal's own size estimate: the change is NOT "drop one clause"

Written after reading `producer_probe_fill` and `producer_probe_scan` in full, and it invalidates the estimate
above. Both functions touch the tensor **through a raw pointer**:

    producer_probe_fill:   uint32_t * words = static_cast<uint32_t *>(node->data);
                           for (size_t w = 0; w < n; ++w) words[w] = kProbeSentinel;
    producer_probe_scan:   const float * fp = static_cast<const float *>(node->data);
                           ... std::memcpy(&bits, &fp[w], sizeof(bits)); ...

`node->data` is a **host** address for a host buffer and is **not CPU-addressable** for a device (HRX0)
buffer. So on a device tensor the existing fill would dereference a device pointer from the CPU, and the
existing scan would read unmapped memory. Dropping the `is_host` clause alone would therefore not produce a
device probe — it would produce a crash or garbage, which is worse than an INVALID result because it can look
like a hit.

The real change is:

1. `producer_probe_target` — accept non-host tensors when `GGML_HRX_PRODUCER_DEVICE` is set, and use an exact
   row-multiple selector `bytes % 8192u == 0 && bytes >= 8192u` (an integer number of `n_embd`-wide rows,
   which needs no knowledge of `T` — cleaner than the ±1 % band proposed above and equivalent to it for
   integer `T`);
2. `producer_probe_fill` — for a device target, fill a **host staging buffer** once with the finite pattern and
   upload it with `ggml_backend_tensor_set(node, staging, 0, bytes)`; keep the raw-pointer path only for host
   targets;
3. `producer_probe_scan` — for a device target, download via `ggml_backend_tensor_get(node, staging, 0, bytes)`
   after the drained stream and classify the staging buffer; keep the raw path for host targets;
4. one `kProbeFinite = 0xDEADBEEFu` constant and a branch choosing it over `kProbeSentinel` in device mode.

That is ~40 lines through three functions, not a clause. The estimate above is withdrawn; the design,
selector, five verdict branches, and confounds all stand.

**This also sharpens why the proposal is a proposal.** The host-only probe's raw-pointer access is not an
oversight to be patched lightly — it is the reason the instrument is cheap and safe, and a device variant
trades that for a staging copy per hooked graph, i.e. exactly the kind of perturbation the record already
flags as an inducer.

### Patch anchors, verified against the file (not from memory)

    runtime/graph-executor.cpp:46-54   bool producer_probe_target(...)  with the is_host clause at :48
    runtime/graph-executor.cpp:80-96   size_t producer_probe_fill(...)  raw write at :86-87
    runtime/graph-executor.cpp:99-132  void producer_probe_scan(...)    raw read at :113-114

Each anchor is an exact string, so a patch script in the style of `tools/patch-*.py` (assert each anchor
occurs exactly once, so a silent no-op is impossible) can apply it when a build is next possible. The build is
deliberately **not** attempted here: compiling raises `load1` and the `foreign` count, which are two of the
three conditions the queued `315prod` run is waiting on, and would contend with the foreign decompilation the
box is already carrying.

### The patch is now written and its anchors verified — but it is NOT compile-tested

`docs/315-campaign/tools/patch-device-producer-probe.py` applies the change in the record's own patch style
(every anchor asserted to occur exactly once, a pre-check over all anchors before any write, an idempotency
marker, and a brace-balance check). Verified against a **real copy** of the source, not from memory:

    patched ggml/src/ggml-hrx/runtime/graph-executor.cpp
      device-enable branch: 1 occurrence(s)
    second run: already patched
    braces: 52 / 52

Two defects were caught by that testing rather than shipped:

1. **Use-before-declaration in the scan replacement.** `const float * fp = ...` is declared *before*
   `const size_t n = ...` in the original, so a staging buffer sized from `n` would not have compiled. The
   replacement now sizes from a locally computed `dfloats` and does not reference `n`.
2. **A stranded comment.** The sentinel anchor matched only the declaration, so the original line's trailing
   comment was left attached to the end of the new `kProbeFinite` line — legal C++ but a mangled double
   comment. The anchor now includes the trailing comment.

**What is still NOT established, and must not be claimed:** that it **compiles**. The script's own output says
so on every run. Confirming that needs a build, which is deliberately deferred because compiling raises
`load1` and the `foreign` count — two of the three conditions the queued `315prod` run is waiting on — and
would contend with the foreign decompilation already on the box. So the instrument is *ready to build*, not
*working*.

---

## BUILT and RUNNING (2026-10-14 05:52) — the compile caveat is now resolved

The patch was applied to the real tree and **the build succeeded**:

    patched ggml/src/ggml-hrx/runtime/graph-executor.cpp      (anchor-verified, braces 52/52)
    [ 26%] Building CXX object .../runtime/graph-executor.cpp.o
    [ 26%] Linking CXX shared library ../../../bin/libggml-hrx.so
    [100%] Built target ggml-hrx
    === EXIT=0 ===

so the earlier "**compile NOT claimed**" caveat is now discharged for *compilation*. Verified in the artefact
rather than inferred from the build's exit code: `bin/libggml-hrx.so.0.18.0` (05:52) contains **both**
`GGML_HRX_PRODUCER_PROBE` and `GGML_HRX_PRODUCER_DEVICE`, and `graph-executor.cpp.o` carries the same
timestamp. Source backed up before patching (`/tmp/graph-executor.cpp.orig`, sha256
`cba0419f55e748ed…`).

**Run launched**: `315dev` — `GGML_HRX_PRODUCER_PROBE=1 GGML_HRX_PRODUCER_DEVICE=1` plus the 15-name
`l_out` fault-yield enabler, 8 rounds, behind the same gate-aware wrapper (`315dev-wait.sh`) that holds no
lock and no GTT while waiting. **Still NOT claimed: that it arms anything at run time.** The runtime
`armed`/`anomaly` lines are the evidence for that, and they do not exist yet.

## The sibling fix another agent landed, which is the closest thing on the record to this family

While checking the build, a pin-bump commit on the fork (`fb24c1f`, Oct 5) turned out to describe a
**root-caused NaN fix in exactly the family this proposal is about**:

    Bump HRX: llama.cpp 522dab4789ce (long-prompt fix), hrx-system to our fork 98d05d94a9f9
      third_party/llama.cpp e57beb9721af -> 522dab4789ce (fork #89 runtime overhead,
      #90: graph inputs never written back to host - every multi-ubatch prompt was wrong since #82 -
      and the non-replay upload race).
      third_party/hrx-system -> our fork ... plus the GFX11 wave64 lane-mask drain backport (#313 NaN)

and the fork's merge itself reads:

    522dab478 Merge pull request #90 from 1bit-MONSTER/1bit/kv-nan-rootcause
    ggml-hrx: graph inputs are never written back to host; non-replay programs wait for their input upload

**That is "the consumer reads memory the device never wrote", already root-caused and fixed once.** Two
consequences for #315:

- **It is evidence the family is real and reachable**, independent of anything this session measured: a KV-NaN
  root cause was found in the host write-back path and fixed at `522dab4789ce`.
- **The engine pin's tree already contains a `PendingHostWriteback` path** whose final step is
  `std::memcpy(writeback.host_destination, writeback.mapped_source, writeback.size)` — a **device mapping**
  copied into a **host destination** (`runtime/command-program-executor.cpp`). That is the exact line where
  "the device wrote these bytes" and "the mapping still holds something else" are decided, and it is in the pin
  this #315 work runs on.

**What I have NOT established, and it decides whether #90 is a fix or already-shipped:** `522dab4789ce` versus
the engine pin `e44c9d01a`. The local clone is **shallow** — `522dab4789ce` has no parents in it
(`git rev-list --parents -n1` returns the commit alone) and `e44c9d01a` is absent — so `git merge-base
--is-ancestor` cannot run. If `522dab4789ce` is an **ancestor** of the engine pin, #90 is already shipped and
#315's remaining faults are a *different* path; if it is a **descendant**, #90 is a **forward-only fix
candidate** for #315. Settling it needs either a full fetch of the fork or the ancestry via the engine repo's
submodule, and it is worth doing before any other run.
