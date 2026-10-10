## Correction: I withdraw my previous comment's per-node attribution — and the reason points at a better candidate

My previous comment reported that the corruption "enters at the routing gather" because `ffn_moe_weights-4` was NaN while its source `ffn_moe_probs-4` was not. **That inference does not hold, and I am withdrawing it.** The measurement it rests on is not ordered, and I can now show why.

### The instrument reads a host-visible arena unsynchronised

`GGML_HRX_BUFFER_TRACE` names every allocation of 64 KiB or more (`ggml-hrx.cpp:343`). In a complete arm the host-visible allocations are exactly two:

```text
[hrx-buf] type=HRX0      host_visible=0 direct_host_binding=0 memory_type=0x30 size=17895352832   (weights)
[hrx-buf] type=HRX0_HOST host_visible=1 direct_host_binding=0 memory_type=0x52 size=619520       (logits)
[hrx-buf] type=HRX0      host_visible=0 direct_host_binding=0 memory_type=0x30 size=443547648
[hrx-buf] type=HRX0      host_visible=0 direct_host_binding=0 memory_type=0x30 size=81788928
[hrx-buf] type=HRX0_HOST host_visible=1 direct_host_binding=0 memory_type=0x52 size=44048640      (arena)
```

Every tensor the fingerprint names on the MoE path — `ffn_moe_weights` (160 B), `ffn_moe_probs` (2560 B), `ffn_moe_gate`/`up` (245760 B), `ffn_moe_down`/`weighted` (327680 B), `l_out` (81920 B) — is too small to be a buffer of its own, so it lives inside the **44048640-byte host-visible, non-coherent arena**. And `ggml_backend_tensor_get` on a host-visible tensor is a plain `memcpy`: no iface call, no synchronisation. That is the same structural fact the earlier hard negative established (it is why the download probes could never fire).

So the fingerprint's per-node lines are **not a snapshot of one instant** — each read samples the arena whenever the CPU reaches it. The two faulting arms show exactly that, and it is the contradiction that gives it away:

* `ffn_moe_weighted-4` reads **81880 of 81920 NaN**, while `ffn_moe_out-4` — its own sum over the routes — reads **finite**;
* `ffn_moe_probs-4` reads finite while the gather made from it reads NaN.

Neither can be true of computed data. What survives from those arms is only what does not depend on read order: the guard's own line, `count=154880 of 154880 … contiguous=yes`, and the fact that the fault is real and in-regime (`gtt_start=98G`).

### What that structure suggests instead — and it fits both faces better

The consumer reads its inputs with the same unsynchronised `memcpy` out of the same class of buffer, and nothing in that path waits for the producing dispatch. A host read that precedes the write yields **unwritten memory**:

* if the mapping held NaNs → **Face A**, `count=154880 of 154880`, the guard fires;
* if it held the **previous token's finite logits** → **Face B**: the sampler draws from a stale distribution, and the two text flavours already on the record are what that produces — the repetition loop `a gaps gaps gaps gaps ...` is literally one token repeated, and the short fragment is the same read scaled differently. Face B never logged NaN, which is what a *finite* stale row gives.

Regime dependence follows from the mapping's history: under GTT pressure, what the arena holds and when the write lands both change.

### The falsification already on the record, and the decisive run

This candidate has one negative result against it and it is thin: `315-graphsync.sh` ran `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1` (a stream sync after every replay, `command-program-executor.cpp:2034`) and its single in-regime arm **faulted**. But that session's composition stopped at 70 GiB and **4 of its 6 arms fell below the gate**, so that is one arm of evidence, not a falsification — and the composition override that caused it is now fixed (`REGIME_LO=80` in the wrappers, which had been overriding the module's own default downward; that is why the graphsync run under-shot the gate while `315chain`, which inherits the default, produced 98 GiB arms).

Queued now, as a **paired** in-regime A/B rather than a comparison against on-disk arms: both cells the same instrumented binary and the same instrument (so the fault rate is high enough to see a difference in a handful of arms), the only difference being the sync knob — the fix cell runs through a launcher that sets `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1`, the control cell does not, arms alternating inside one composition, at `gtt_start` projected into 101-106 GiB. Verdicts come from the guard and the detector's completion accounting, not from the fingerprint. Zero faults in the fix cell while the control keeps faulting makes the read-path ordering the mechanism; an unchanged rate kills it.

```sh
ssh strixhalo 'grep -E "^  pair|control_bin|fix_bin" ~/wt/315gsync2-console.log'
```
