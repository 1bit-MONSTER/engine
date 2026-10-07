<!--
Copyright 2026 bong-water-water-bong
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# The device bridge: a drafter on the NPU, the target model on the GPU

`1bit serve -m target.gguf --bridge-draft <draft dir>` runs the target model where it always
runs (the child `llama-server` on HRX) and **also** loads a draft model in process on another
device (the NPU fast lane, [npu.md](npu.md)). The GPU decodes one batch per round; the NPU
proposes the tokens in that batch. One model file, one token stream, two devices.

This is the answer to "can a model be split so it chunks tokens at the GPU and the NPU at the
same time". The short version:

| Split | Works? | Why |
|---|---|---|
| Weights split across GPU+NPU (layer or tensor split) | No | Both devices read the same LPDDR5X pool on Strix Halo. Splitting weights adds a per-layer handoff and no bandwidth, so a single token stream gets slower. The two runtimes also disagree on weight format (GGUF vs Q4NX tiles) and have no shared activation entry point ([npu.md](npu.md), [hrx.md](hrx.md)). |
| Requests routed to one device or the other (two replicas) | Yes, but small | This is what Laya + `config/route-policy.json` already does ([laya.md](laya.md)). It needs a second copy of the model resident (about 16 GB for a 27B), and on this box that is the GTT regime that reproduces #315. |
| **Prefill on one device, decode on the other over a shared KV** | Yes, large project | This is the removed `--prefill-device` / dma-buf pair ([hrx.md](hrx.md), "Our patches"). For NPU+GPU it needs dma-buf export of the lane's KV BOs and a layout shim. |
| **Draft on one device, verify on the other (this document)** | **Yes** | The only interface between the devices is **token ids**. No shared memory, no shared weights, no shared format. The target verifies every drafted token, so the bridge cannot change the answer. |

## The invariant: the bridge cannot change the output

Every token the drafter proposes is checked by the target model in one batch. The target
accepts the longest prefix that matches its own sampling and then emits its own token at the
first mismatch. A drafter that is wrong, slow, busy or absent therefore changes **speed only**,
never the reply:

- a draft list that fails to arrive is treated as "no draft" — the round is a normal single-token
  decode;
- a drafter on the wrong vocabulary produces low acceptance, not wrong text (the target rejects
  the tokens it would not have produced);
- turning the feature off restores the pre-bridge path exactly, because the bridge only exists
  when `--bridge-draft` is passed.

That is why a speculative split is the right first bridge and a weight split is not: the
interface is ids, so the two devices never have to agree on anything else.

## Architecture

```
  client ---> 1bit serve  (OpenAI API, unchanged)
                 |
                 |  launches, as today:
                 +--> llama-server (HRX child)  --- decode loop ---+
                 |         target model (GGUF, GPU)               |
                 |                                                 | POST /v1/draft
                 |  in process:                                    v
                 +--> bridge server (loopback HTTP) <---- 127.0.0.1:<port>
                          |
                          +--> draft model: NPU fast lane (Q4NX dir)
                               or a scripted source (test seam)
```

The child asks for drafts; the engine answers from the NPU. The engine chose the port and
told the child about it on the command line, so nothing is hardcoded.

## The protocol

The engine serves one endpoint on loopback, `POST /v1/draft`, JSON in and out. The fork's new
`draft-external` speculative type is the only client ([the fork hook](#the-llamacpp-fork-hook-draft-external)).

| op | request | response |
|---|---|---|
| `begin` | `{"op":"begin","seq":0,"prompt":[ids...]}` | `{"ok":true}` |
| `draft` | `{"op":"draft","seq":0,"n_past":P,"id_last":T,"prompt":[ids...],"n_max":K}` | `{"draft":[ids...]}` |
| `accept` | `{"op":"accept","seq":0,"n":A}` | `{"ok":true}` |
| `reset` | `{"op":"reset","seq":0}` | `{"ok":true}` |

Rules, in the order the engine applies them:

1. **The sequence is `prompt ++ [id_last]`.** `prompt` is every token the conversation has
   committed so far, `id_last` is the one the target just sampled. `n_past` is `id_last`'s
   position, so `len(prompt) == n_past` when the client is honest; a mismatch is logged once,
   not fatal (the `prompt` field is still the authority — upstream llama.cpp marks it
   "TODO: remove in the future").
2. **`draft` is greedy** (argmax) and returns at most `min(n_max, k_cap, room left in the
   drafter's context)` ids. An empty list is always a legal answer.
3. **Every `draft` call resynchronises the drafter to `prompt ++ [id_last]` by replay.** The
   drafter keeps the last sequence it was given and steps the new sequence from their longest
   common prefix. Rejected drafts are overwritten by the next round's replay; the lane's
   causal attention only reads rows below the current position, so no explicit rewind exists or
   is needed.
4. **One sequence per drafter.** `seq != 0` is refused with `{"error":...}`. HRX gives a gated
   delta-net model one slot anyway ([serve.md](serve.md)), and the lane holds one KV cache
   ([npu.md](npu.md)).
5. **Any error is a non-answer.** HTTP 4xx/5xx, non-JSON, a closed socket or a timeout must make
   the client drop the round (empty draft). The engine serialises requests against its one
   drafter; a round that cannot be served is an empty draft, never an error that reaches the
   user's reply. The fork side owns the client timeout.

Nothing in the protocol carries logits. The drafter does not need the target's distribution and
the target never needs the drafter's — the target already has everything it needs to verify.

## The engine side (`app/bridge.{h,cpp}`, `1bit serve --bridge-draft`)

| Flag | What |
|---|---|
| `--bridge-draft <dir>` | Enable. `<dir>` is an NPU model directory ([npu.md](npu.md), `model.q4nx` + `npu/`), used as the drafter. Needs `-DONEBIT_NPU=ON`. |
| `--bridge-k N` | Most drafts per round. Passed to the child as `--spec-draft-n-max N` (default 4). |
| `--bridge-port N` | Bridge port; default 0 = an ephemeral loopback port, printed to stderr. |
| `--bridge-script <file>` | **Test seam.** One integer per line: the drafter's next-token answers, in order. Lets the whole path be tested with no NPU and no model. |

`--bridge-draft` refuses (rather than degrading silently) when:

- the build has no NPU lane;
- the server it is about to launch does not list `draft-external` (probe the child's `--help`
  once) — the message names the fork change below;
- `--mtp`/`--dflash` is also set: they are two other drafters, pick one;
- the model is not a `.gguf` on the `hrx`/`cpu` path.

It is allowed with `--parallel 1` only (the drafter is one lane), and it prints the vocab
guard's verdict: the drafter's `tokenizer.json` size against the GGUF's vocabulary. A mismatch
is a warning, not a refusal — see the invariant.

## The llama.cpp fork hook (`draft-external`)

The pinned fork already has the whole speculative framework; an external drafter is one more
type in it, not a new subsystem. Every edit is below, against the pin
(`1bit/hrx-vulkan-patched`, llama.cpp `cf4dfd801e6a`; file contents read from that commit).

**1. `common/common.h`** — the enum, before `COMMON_SPECULATIVE_TYPE_COUNT`:

```c
    COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL, // draft tokens come from an external process
```

and, next to the other per-type params:

```c
struct common_params_speculative_external {
    std::string addr = ""; // host:port of the engine's /v1/draft endpoint
};
```

plus `common_params_speculative_external external;` in `common_params_speculative`. The draft
length reuses the existing `draft.n_max` / `draft.n_min`, so `--spec-draft-n-max` and
`--spec-draft-n-min` keep working with no new flag.

**2. `common/speculative.cpp`**

- `common_speculative_type_from_name_map`: `{"draft-external", COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL}`.
- `common_speculative_type_to_str`: the matching `case`.
- `common_speculative_n_max`: `case COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL: n_max = std::max(n_max, params->draft.n_max); break;`
- `common_speculative_init`: `has_draft_external = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL))`
  — note it does **not** require `params.draft.ctx_dft` (there is no draft context), then push
  its config and add the `switch` case that constructs the impl. Update
  `static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11)` to `== 12`. The existing priority list
  puts n-gram types before the draft types; an external drafter wants the **highest** priority
  of the draft types (nothing else is loaded with it in this configuration).
- `common_speculative_init_from_params`: the `GGML_ASSERT(has_draft || spec_mtp)` must also
  accept `COMMON_SPECULATIVE_TYPE_DRAFT_EXTERNAL`, and no model/context is loaded for it.
- The new impl. It has no `llama_context`; it is a client:

```c
struct common_speculative_impl_draft_external : public common_speculative_impl {
    common_params_speculative_external params;

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override;   // POST {"op":"begin"}
    bool process(const llama_batch &) override { return true; }              // the drafter owns its KV
    void draft(common_speculative_draft_params_vec & dparams) override;      // POST {"op":"draft"} per seq
    void accept(llama_seq_id seq_id, uint16_t n, bool is_other) override;    // POST {"op":"accept"}
    bool need_embd() const override { return false; }
};
```

  `draft()` for a seq with `dp.drafting` builds
  `{"op":"draft","seq":0,"n_past":dp.n_past,"id_last":dp.id_last,"prompt":*dp.prompt,"n_max":k}`
  with `k = dp.n_max > 0 ? min(dp.n_max, params.n_max) : params.n_max`, POSTs it, copies
  `result` into `*dp.result`, and applies the existing `n_min` rule (clear if shorter). A
  failure of any kind leaves `*dp.result` empty. `begin`/`accept` are advisory: the engine
  resynchronises from `prompt` on every `draft`, so dropping them cannot desynchronise the pair.

**3. `common/arg.cpp`** — add `--spec-external-addr` (string →
`params.speculative.external.addr`). `--spec-type` needs nothing: it parses names through
`common_speculative_types_from_names`, which is the map updated above. Refuse
`draft-external` with an empty addr at parse time.

**4. Nothing else.** `llama-server` calls `common_speculative_init` / `_draft` / `_accept`
already; no server-side file changes.

Testable without a GPU: `llama-speculative`/`llama-server` on the CPU with
`--spec-type draft-external --spec-external-addr 127.0.0.1:<port>` and a scripted drafter must
produce the same text as `--spec-type none` on a tiny model, with the engine's log showing the
draft rounds. That is the fork-side acceptance test; it needs no HRX and no NPU.

## What this box can and cannot test

This checkout's dev box is a Ryzen 7 9800X3D with an RX 9070 XT, no `amdxna`/`/dev/accel` and
no ROCm: it cannot run the lane or HRX. So the split of evidence is:

| Check | Where |
|---|---|
| Protocol, replay/resync, k clamp, error handling, serve wiring, "flag absent ⇒ byte-identical argv" | this box, `ctest -R bridge` |
| `draft-external` in the fork: CPU-only llama-server + scripted drafter, greedy text equal to `--spec-type none` | any box with the submodules, no GPU needed |
| End-to-end acceptance rate and tok/s on the target hardware | Strix Halo |

### Strix Halo validation plan

1. Build the fork with the hook, and the engine with `-DONEBIT_NPU=ON` and the lane kernels for
   the draft model.
2. Draft model: a Qwen3-shaped NPU directory whose vocabulary matches the target (the lane
   packer is qwen3-only, [npu.md](npu.md)). For a Bonsai target this is the open problem —
   PrismML's PQ2_0/PTQ1_0 types and the Hadamard fold exist only in the HRX fork, so a Bonsai
   drafter needs a repack to q4_1 tiles with the rotation folded into the weights first.
3. Measure, same prompt set and seed, interleaved, `--bridge-draft` off vs on:
   - tok/s decode, time to first token, and the acceptance rate (accepted drafts / drafted);
   - `--bridge-k` 2, 4, 8;
   - the target's own `--mtp` on the same model as the comparison drafter, if it has one.
4. Keep the feature only with a measured net gain. The decision rule is the one
   `config/route-policy.json` already states: a row changes only with a measured net gain
   (decode and time to first token).
5. Watch GTT: the target plus the lane's packed weights are both resident. The #315 fault
   regime starts around 102 GB GTT, so measure with the neighbours a real deployment has.

## Measured on Strix Halo (2026-10-07)

The bridge's ceiling, with the target on the CPU (Qwen3-0.6B-Q4_K_M, `-ngl 0`, 32 greedy
tokens, `--bridge-k 4`) and the drafter a **script built from the target's own greedy
continuation** — captured with `return_tokens: true` on llama.cpp's native `/completion`
(the OpenAI route drops the field). The target still verifies every token, so this is an upper
bound on what any drafter can buy, not a drafter's measured hit rate:

| prompt | replies identical | acceptance | plain tok/s | bridged tok/s | speedup |
|---|---|---|---|---|---|
| prose (`The capital of France is`) | yes | 1.00 | 187.6 | 421.6 | 2.25× |
| code (`def fib(n):`) | yes | 1.00 | 188.2 | 462.2 | 2.46× |
| long document | yes | 1.00 | 184.0 | 467.4 | 2.54× |

Reproduce: `ssh strixhalo 'cd ~/wt/bridge-run && python3 ceiling_k.py --send'`, which records the
numbers as scores (`exact-match`, `acceptance-rate`, `plain-tok-s`, `bridged-tok-s`, `speedup`,
`measurement-valid`) on the `bridge-correctness` experiment in Langfuse. That harness repeats and
interleaves the plain control with each draft length and **refuses to report** when the control's
own spread exceeds `--max-spread` (default 1.10×): it measured 1.14× on a loaded box and exited 2
without recording. `ceiling.py` is the original single-point (k=4) version, without the guard.

**Two traps that make the bridge look broken when it is not:**

- **A perfect script is not the continuation.** Between draft rounds llama.cpp samples one
  token itself, so each round of K accepted drafts advances the stream by K+1: emit K ids,
  then skip one. Feeding the raw continuation drifts by one per round and measures acceptance
  0.13-0.19 with speedups *below* 1 (0.49-0.92×) — the bridge appearing to cost throughput.
  With the stride fixed, acceptance is 1.00 and the speedup is the table above.
- **The child must run one slot.** The bridge serves one sequence and answers `400` to any
  `seq != 0`. A recipe sets `-np 4` for the cpu route, so every draft arrived as `seq=3`,
  every request was refused, and llama.cpp reads a failed draft as "no draft" — silence, not
  an error. `launch_for` appends `-np 1` after the recipes for exactly this reason; the
  symptom of losing it is `rounds 0, acceptance 0%` while the replies still match (both
  passes are then plain decodes).

**Measure on a quiet box.** Throughput here is dominated by whatever else is running: with load
average ~14 on 32 cores (one foreign `llama-server` at 912% CPU), three *identical* plain passes
of the same prompt gave 153.4, 129.7 and 52.9 tok/s — a 2.9× spread on the control. A draft-length
sweep taken in that state (`--bridge-k` 2/4/8) produced speedups from 1.73× to 5.68× and was
recorded, then marked `measurement-valid=0` in Langfuse rather than published. The acceptance rate
is deterministic and survives the load; the throughput ratios do not. Check `/proc/loadavg` before
trusting a tok/s number, and repeat the plain control — if its spread is not small, the run is not
a measurement. The harness now enforces that itself (`ceiling_k.py` refuses with exit 2 rather than
report), so the rule is a command, not a habit.

**Not measured.** A real drafter. The NPU fast-lane kernel set for Qwen3-0.6B is not on the
box (the model directory's `npu/` symlinks point into a cleaned `~/.cache/1bit-engine-tmp/`,
and every other candidate has the layer ELFs but no `lmhead.elf` and no `layer.pdi`), so
`--bridge-draft` has not run; the NPU lane *build* does pass (`-DONEBIT_NPU=ON`). The scripted
numbers above bound the mechanism, and no claim here rests on a drafter's real acceptance.

## Open items

- `dp.prompt` is documented upstream as a temporary field. When it is removed, the engine keeps
  resynchronising through `begin` + `accept`; the protocol already carries both, and the engine
  treats `prompt` as optional if the field disappears (it replays the accepted prefix it
  tracked itself). Until then `prompt` is the authority.
- Multi-sequence: the protocol has a `seq` field and only serves seq 0. If HRX gains
  multi-sequence gated delta-net decoding, a second lane (or a second drafter) per sequence is
  the extension; the wire format does not change.
- The drafter's context is the lane's `kMaxContext` (8192, [npu/lane.h](../npu/lane.h)). Past
  that the engine returns short or empty drafts and the target decodes normally.
- **A hung lane blocks the round.** `Lane::wait` has no timeout, so if the NPU stops answering,
  the endpoint cannot answer either and the child waits. The fork side's client timeout is the
  current bound; a watchdog on the lane's runlist (and a drafter thread the handler can abandon)
  is the fix. Counts as an open item for the hardware bring-up, not a design change.
