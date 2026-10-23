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
# Hybrid: NPU prefill, HRX decode (`--device hybrid`)

```
1bit serve -m Qwen3-0.6B-Q4_K_M.gguf --device hybrid --prefill-provider <program>
```

Hybrid serves a `.gguf` with the prompt processed on one device and the tokens generated on another:
a **prefill provider** computes the prompt's KV cache, and the HRX llama-server (docs/hrx.md)
decodes from it on `HRX0`. The provider is the engine's NPU route; the public tree knows it only as
a program with the contract below, the way `-DONEBIT_NPU_PRIVATE` keeps the NPU kernels out of the
repository (docs/npu.md). Without a provider, `--device hybrid` is `--device hrx` with one slot.

In a build without HRX the decoder is the same llama-server on the CPU, so the route runs (and is
tested) on any machine.

## What happens per request

1. **Template and tokenize on the backend.** For `/v1/chat/completions` serve asks the llama-server
   child for the templated prompt (`/apply-template`) and its ids (`/tokenize`, `add_special` and
   `parse_special` on, as the server itself tokenizes); `/v1/completions` tokenizes its `prompt`
   string. The provider sees exactly the ids the backend would have prefilled.
2. **The provider prefills.** serve writes the ids to a file in the backend's slot directory and runs
   the provider. It writes a llama.cpp sequence-state file for the prompt.
3. **Restore into slot 0.** `POST /slots/0?action=restore` loads the file into the one slot; serve
   checks the slot holds all the prompt's tokens.
4. **Decode.** The request is forwarded with `id_slot` 0 and `cache_prompt` true. llama-server finds
   the prompt in the slot, re-evaluates its last token (it always does, for the logits), and generates.
   The reply, streamed or not, is relayed as for every device.

One line per request on stderr: `hybrid: N prompt tokens: prefill X ms, restore Y ms, first byte Z ms`
(the provider's own `prefill_ms` when it reports one, else its wall time; the restore from the backend's
timing; first byte measured in serve from the start of the handoff).

**Fallback.** Any failure on the way (no text prompt, a provider that exits non-zero, times out or
writes no file, a restore that fails or holds fewer tokens) is said on stderr with its reason, and the
request is forwarded as it came: the backend prefills it itself. Without a provider named, serve says
so once at start. The device is never unusable because the provider is.

**One slot.** Hybrid runs the backend with `-np 1` and `--slot-save-path`, and serializes handoffs: a
request holds the slot from its handoff until the backend has started on it (its first byte), after
which llama-server itself defers the next restore until the slot is free. `--parallel` > 1 is refused;
so are `--mtp`, `--dflash`, `--mmproj`, `--moe-slots`, a role (`--embedding`, `--reranking`) and `--pm`,
and the GGUF's own MTP head is not used: nothing but the prompt's KV is handed over, and a drafter
would draft over a cache it did not compute.

## The provider contract

```
<provider> --tokens <ids file> --out <state file> --model <model path> [--elfs <dir>]
```

| | |
|---|---|
| `--tokens` | the prompt's token ids, one decimal id per line, in order |
| `--out` | where to write the state: a bare filename inside the backend's `--slot-save-path` directory (serve chooses it; the backend restores by that name) |
| `--model` | the model file serve was given (`-m`) |
| `--elfs` | passed when `ONEBIT_HYBRID_ELFS` is set: where the provider's kernels are |
| stdout | one JSON line, last: `{"n_tokens": N, "argmax": id, "prefill_ms": x}`; `n_tokens` must equal the ids given (checked), `argmax` is the next-token argmax after the prompt (for diagnostics), `prefill_ms` is optional |
| exit | 0 on success; anything else, or no file, is the fallback above |
| environment | `ONEBIT_HYBRID_BACKEND_URL=http://127.0.0.1:<port>`, the decoding llama-server's address (a test provider makes its file through it; a real one has no use for it), plus serve's own environment |
| time limit | `ONEBIT_HYBRID_PREFILL_TIMEOUT_S` (default 600): SIGTERM, then SIGKILL |

**The state file** is what `llama_state_seq_save_file` writes and `/slots/<id>?action=restore` reads
(llama.cpp `src/llama-context.cpp`, `src/llama-kv-cache.cpp` `state_write_meta` / `state_write_data`):
the file magic and version, the token count and ids, then the KV cells of sequence 0 at positions
0..N-1: cell count, each cell's position and sequence ids, then per layer the K type, row size and N
rows, then V the same way (transposed or not, as the server's cache is: `v_trans` 0 with flash
attention, which HRX uses). The K rows are post-RoPE keys, as llama.cpp stores them; the cache type is
the backend's (`f16` by default), so a provider computing in another type converts. A provider must
match the version of the pinned llama.cpp (`third_party/llama.cpp`): the restore refuses another.

Naming: `--prefill-provider PATH` or `ONEBIT_HYBRID_PREFILL=PATH`; `ONEBIT_HYBRID_ELFS=DIR` adds
`--elfs DIR`.

## Tests

- `tests/hybrid_route.sh` (CTest `hybrid_route`, no model, no NPU): `tests/fake_backend.py` stands in
  for llama-server and `tests/fake_prefill_provider.py` (`FAKE_PREFILL_COPY=1`) for the NPU engine.
  Checks what reaches the backend (`id_slot` 0, `cache_prompt` on, the slot directory), the stderr
  lines, the fallbacks (no provider; a provider that fails), the refused flags and `--print-backend`.
- `tests/hybrid_e2e.sh` (CTest `hybrid_e2e` with `ONEBIT_SERVE_TEST_GGUF`; skips without a model or a
  llama-server): a real model through the whole path. The fake provider makes a real state file by
  having the decoding llama-server itself prefill the ids into slot 0 and save the slot, so the
  handoff is exercised end to end with no NPU. Checked: the greedy reply equals the plain `--device
  cpu` reply, and the backend's `prompt_n` is 1 (the re-evaluated last token), not the prompt length.

## Status

The plumbing (this document) is public. The NPU prefill provider, its kernels and its state-file
writer are the engine's private NPU work; what it measures on Strix Halo is reported there when it
lands.
