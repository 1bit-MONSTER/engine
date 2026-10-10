## Mechanism narrowed: exactly one token's hidden row is NaN at the MoE aggregation — plus three falsifications. No mechanism claim yet; #315 stays OPEN

Diagnostics only. Written 2026-10-10 on `strixhalo` against the pinned revision (`third_party/llama.cpp
e44c9d01a4d5`). Every number below comes from a command quoted next to it. The row-index discriminator that
decides the remaining question is running on the box now, so this comment names no mechanism.

### 1. The invariant: 2048 NaN floats — one `n_embd` row — while the batch size varies four-fold

```sh
ssh strixhalo 'cd ~/wt/rt-det; for d in 315nan-*/; do l=$(grep -m1 -E "hrx-tensor-fp.*nan=[1-9]" "$d/server.log"); \
  [ -n "$l" ] && echo "$(basename $d | cut -d- -f1-2) :: ${l#*hrx-tensor-fp}"; done'
```

| arm | first NaN-bearing node | op | bytes | floats | NaN |
|---|---|---|---|---|---|
| p1l, p6l, p9l, p10l | `node_2948` | ADD | 1409024 | 352256 | **2048** |
| p11l | `node_2948` | ADD | 2506752 | 626688 | **2048** |
| p13l | `node_2948` | ADD | 1146880 | 286720 | **2048** |
| p15l | `node_2948` | ADD | 802816 | 200704 | **2048** |
| p17l | `node_2948` | ADD | 2801664 | 700416 | **2048** |
| p2l, p3l, p4l, p7l, p8l | `node_2948` | ADD | 712704 | 178176 | **2048** |
| p5l | `node_2948` | ADD | 1105920 | 276480 | **2048** |
| p14l | `ffn_swiglu-41` | GLU | 534528 | 133632 | **2048** |
| p12l | `norm-0` | RMS_NORM | 81920 | 20480 | 55 |
| (`p20l`, excluded — see below) | `ffn_swiglu-41` | GLU | 2138112 | 534528 | 2048 |

The traced tensor's row count changes by a factor of four across arms (`bytes/4/2048` = 87, 98, 135, 140, 172,
306, 342 rows) while the NaN count is **2048 in 15 of 16 arms**: exactly one `n_embd` row. GLM-4.7-Flash's
`n_embd` is 2048, so the defect is *one token's hidden vector*, not a block, not the tail of the graph.

`p20l` is excluded deliberately: its Face-A event happened at `gtt_start=44` **after that session SIGKILLed
another arm's server**, so it is ambiguous and is not counted (this is recorded, not quietly dropped).

Two corroborating numbers from the same population:

* the whole-vocabulary Face-A fingerprint reads `count=154880 of 154880, contiguous=yes` — the *sampled row* of
  the final norm is all-NaN (`315hunt2-p6l`: `name=norm bytes=8192 nan=2048 of 2048`), which is downstream of the
  row above, not a second event;
* `node_2948` is unnamed because of how the MoE sum is built: `build_moe_ffn` calls `ggml_add` per expert and
  only the final sum gets `cb(moe_out, "ffn_moe_out", il)` (`src/llama-graph.cpp:2151-2162`). GLM uses 4 experts,
  so two or three adds in that chain are unnamed and `node_2948` is one of them — a **partial expert sum**.

### 2. What this rules out

* **A stale or globally corrupt logits buffer.** That cannot put one NaN row into a *prefill* graph of T=172 while
  the other 171 rows are finite.
* **Corrupt weights.** Those would hit every token, not one.
* **"The logits were never written"** — already falsified on the record, and now falsified in shape: the writes
  happen and one row is wrong.

### 3. Face B is the same corruption, finite — measured, not inferred

The chain probe (`315chain`, 2026-10-10) faulted its first pair in **both** cells with a 60-name fingerprint over
the residual stream, both attention branches, the whole FFN/MoE chain and the output norm, and logged **zero**
NaN lines — no NaN anywhere, logits included. The rejected text was `the}:}:}甚;` in the fix cell and a
repetition loop (`a gaps gaps gaps gaps ...`) in the control cell. So Face B is *finite* corruption: near-degenerate
logits under greedy sampling, rejected later by the parser
(`common_chat_peg_parse: unparsed Content-only output:`).

Across all 142 recorded arms the faces never co-occur (23 Face-A, 7 Face-B, 0 both, 108 clean), and Face A
travels with finite nondeterminism while Face B does not (`identical_distinct` > 1 in exactly the 7 Face-A arms).
So the observable depends on what the corrupted row held — NaN (caught at the sampler) or stale finite values
(caught at the parser) — which is one mechanism with two observable faces.

### 4. Falsified today: the visibility/ordering family is not sufficient

* **Coherence.** A single-bit patch (`HRX_MEMORY_TYPE_HOST_COHERENT` for host-visible buffers) was run as its own
  cell, with the driver's own trace proving the patch was live (`memory_type=0x52` → `0x56`, i.e. `hv52=0
  hv56=1`): **9 clean and 1 fault in 10 fix-cell arms** — the baseline shape. Coherence is not sufficient.
* **Consumer-side ordering.** `GGML_HRX_DIAGNOSTIC_GRAPH_SYNC=1` calls `hrx_stream_synchronize()` after every
  graph replay (`command-program-executor.cpp:2034`), so every host read must follow the device's writes: 6 arms,
  4 of them below the gate; the one in-regime arm (`gtt_start=99`) **faulted** (Face-B class, `req=19`, `nan=0`),
  and no arm that reached the gate was clean. Ordering is not sufficient either.
* A correction that matters for how these are read: earlier eliminations leaned on the logits `sentinel=0` line,
  but the sentinel only fills the 619520-byte logits allocation, so it never covered the tensors where the NaN is
  actually born. The two results above stand on their own traces, not on that inference.

### 5. The discriminator now running, and its pre-committed readout

The fingerprint has been extended to print `ne0=`, `first=` and `row=` (the token index), and is being run over
the named per-token MoE tensors (`ffn_moe_probs/weights/gate/up/weighted/down/out`, `ffn_swiglu`, `ffn_shexp`,
`l_out-41`, `norm`, `result_norm`). Read it as:

| result | reading | consequence |
|---|---|---|
| the NaN row is the **last** row in every arm | the token whose routing is computed last — a read not ordered after the routing computation | the fix is to order it |
| the **same** row index throughout the chain | a per-token state: the routing weight or one selected expert's activation for that token | names the carrier |
| **differing** row indices | not per-token at all; the invariant is an artefact of the sampler's single-row read | the direction is wrong and must be abandoned |

### 6. Status

`#315` remains **OPEN**. What is established is the shape (one token's hidden row, entering at the MoE
aggregation) and three falsifications; what is not is the site. That is what the running probe is for, and the
next comment will carry its arm-by-arm output. KNOWN-ISSUES.md still carries the superseded "the logits buffer is
never written" story and will be corrected with the mechanism rather than ahead of it.
