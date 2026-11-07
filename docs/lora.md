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
# Adapters (LoRA): where they live, how to change one without reloading it, and the two traps

The engine has no adapter code of its own. An adapter is a llama.cpp LoRA GGUF, handled entirely inside
`third_party/llama.cpp` (`src/llama-adapter.cpp`), and the surface is llama.cpp's: `llama_adapter_lora_init` /
`llama_set_adapters_lora` / `llama_adapter_lora_free` in `include/llama.h`, plus llama.cpp's own `--lora`,
`--lora-scaled` and `--lora-init-without-apply` flags wherever a tool from that tree exposes them.

This page exists because three separate investigations re-derived the same two facts from scratch (engine#433,
#528, #1251's successor #973), and one of them was written down wrong once already (#973's "0.3 % HRX offset"
is a property of a *pin*, not of HRX — see "Trap 1").

**Verified at the pin** `third_party/llama.cpp` = `ed43da0b4` (what engine main carried when this page was
written): the four `llama_adapter_lora_*` entry points are in `include/llama.h` (`set_factors` at `:725`,
`_multi` at `:745`, `n_factors` at `:751`, `factor_name_by_index` just below) and `--lora` / `--lora-scaled` /
`--lora-init-without-apply` are in `common/arg.cpp`. The factor write itself arrived in fork commit `c729177bc`,
an ancestor of that pin. A pin bump re-checks in two commands:
`grep -n 'llama_adapter_lora_set_factors' third_party/llama.cpp/include/llama.h` and
`grep -c -- '--lora' third_party/llama.cpp/common/arg.cpp`.

## The lifecycle

1. `llama_adapter_lora_init(model, path)` parses the adapter GGUF, validates each `lora_a`/`lora_b` pair against
   the **base model's** tensor of the same name, allocates the pair through the *model tensor's own buffer type*
   and copies the bytes in with `ggml_backend_tensor_set`. The map is keyed by the base tensor name
   (`blk.3.attn_q.weight`), and the load logs the buffer it used:
   `Vulkan0 LoRA buffer size = 24.06 MiB`, `HRX0 LoRA buffer size = 24.06 MiB` — device memory, not a host
   fallback, for a rank-2 bank over 196 projections of Qwen3-0.6B.
2. `llama_set_adapters_lora(ctx, adapters, n, scales)` makes a set active on a context with a per-adapter scale.
   Because the *set* changes, the context's graph work happens here.
3. `llama_adapter_lora_free` unregisters and frees.

## Changing the factors of a live adapter

Before #973 the only way to change a resident adapter's factors was to rewrite the whole adapter GGUF and re-init
it: the file is re-parsed, every tensor pair is re-allocated, and `llama_set_adapters_lora` forces the graph work.
There is now an in-place path in `include/llama.h` (engine#973):
`llama_adapter_lora_set_factors` and `llama_adapter_lora_set_factors_multi`, with
`llama_adapter_lora_n_factors` and `llama_adapter_lora_factor_name_by_index` to enumerate what an adapter covers.
The factors are copied into the tensors the graph already points at, so there is no parse, no allocation, no
adapter-set change and no graph rebuild.

**Measured** (engine#973, Qwen3-0.6B Q4_K_M, rank 2 over 196 projections, batch 16, per update, HRX0,
min of two passes, `write` = the factor update, `apply` = reload + adapter-set change):

| path | bank RT 2 | bank RT 26 |
|---|---|---|
| rewrite the GGUF + init + `set_adapters_lora` | 27.6 ms (11.2 + 16.4) | 152.9 ms (119.0 + 33.9) |
| in-place factor write | 16.9 ms (16.0 + 0.9) | 85.7 ms (84.0 + 1.7) |

`apply` is the part that never was inference, and it collapses to ~1 ms (the residual is the one-time load
amortised over the steps measured). The bank column is why this exists: the old path's cost tracks the size of the
*whole adapter file*, so a multi-stage bank pays for every frozen stage on every update (11.2 → 119.0 ms from RT 2
to RT 26). The in-place path pays for the bytes it copies instead. Both paths produce identical results: the same
8-step run gives identical losses to the last digit, and the shipped test `tests/test-lora-inplace.cpp` asserts the
in-place result is `nmse = 0.0` against a fresh load of the same factors, with the adapter GGUFs deleted first so a
re-parse cannot pass.

**What cannot change in place**, and is not a bug: the rank and `alpha`. A different rank is a different adapter
(`llama_adapter_lora_init`); `alpha` is baked into the graph's scale node, so changing it still goes through
`llama_set_adapters_lora`. A caller must not have a decode in flight on a context that has the adapter active while
it writes.

## Trap 1: on HRX, a zero-factor adapter is *not* a bit-exact no-op — and the offset is per-pin

An adapter whose `lora_a`/`lora_b` are all zeros should reproduce the base loss exactly. On Vulkan it does. On HRX it
does not, deterministically: adding the factor pair to the graph changes the matmul reduction order.

| build (llama.cpp pin) | base | zero-factor adapter | offset |
|---|---|---|---|
| `a5271d9` (c20) | 2.265140 | 2.272791 | **+0.34 %** above base |
| `4a375bf2d` (engine #1120) | 2.267098 | 2.265357 | **−0.08 %** below base |
| Vulkan, either pin | 2.219879 | 2.219879 | bit-identical |

Same four fixed prompts, same model, same seeds, each row reproduced bit-identically across runs. So: an
adapter-on vs adapter-off loss comparison on HRX carries an offset of about 0.3 %, **it changes sign between pins**,
and it must be read against the zero-factor row of the *same build*, never against a number remembered from another
one. A difference-based method (two adapter-on evaluations) is unaffected.

## Trap 2: the adapter's tensors are the model's buffer type, so "repack to CPU" is not what happens

`llama_adapter_lora_init_impl` picks the buffer type per tensor from the model tensor it shadows. On a repacking
backend the fallback path exists, but it does not apply to the setups measured here: the load line names the device
buffer (see above). Practical consequence: `ggml_backend_tensor_set` — which is what the in-place API uses — is the
correct primitive, and a caller that instead writes into a host copy of the factors is writing into nothing.

## Checking that you did not move the forward path

Any change near this code should be checked with a fixed-prompt known-answer probe rather than a throughput table,
because the things that break here change *losses*, not speed. The shape that works, and that caught the #1120 pin
moving the HRX base loss by +0.086 %:

- four fixed prompts, one fixed token count, one fixed adapter (rank 2) and one fixed seed;
- per backend, record: base loss, loss with a **zero-factor** adapter, loss after dropping the adapter, and the two
  losses of a ±ε probe;
- compare against the reference row for **that build**, not against another pin's (Trap 1 is exactly this mistake);
- a per-build table is the useful artefact: the rows above came from a four-prompt probe that runs in a few seconds
  of GPU, and the whole mechanism is ~30 lines against `libllama` (engine#973's feasibility study kept the harness
  and the reference rows).

## What is deliberately not here

- **Per-sequence factor slots / co-serving updates inside `serve --parallel`.** Nothing in the engine needs it yet,
  and it is a much larger change (a loss-returning request, slots beside user traffic). engine#973 measured the
  cheap half and stopped there.
- **Factor read-back.** An adapter exposes no getter for its current factors; a caller that needs one keeps its own
  copy (the tests and the ZO prototype both do).
