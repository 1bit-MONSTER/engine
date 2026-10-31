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

**What this pin can measure, measured** (strixhalo, 2026-10-30, engine `38e29a2` + the #612 branch's
lock commit, HRX llama-server `40c4ccb76`, Qwen3-Coder-30B-A3B-Instruct-Q4_K_M, ctx 8192, a 307-token
prompt and 8 greedy tokens, 3 requests each, under `~/.local/bin/gpu-hold -l "pi i612"`):

| route | wall for the request | the route's own breakdown |
|---|---|---|
| `--device hrx --parallel 1` | 101 / 102 ms, after one 559 ms warm-up | the backend prefills the prompt itself |
| `--device hybrid` + `tests/fake_prefill_provider.py` | 356 / 370 / 390 ms | prefill 212-239 ms, **restore 6-7 ms**, first byte 351-385 ms |

The number that outlives the fake provider is the **restore: 6-7 ms for 307 tokens**. That is all the
handoff itself costs (`POST /slots/0?action=restore`), so a hybrid can only win if the provider beats
the backend's own prefill of the prompt by more than that -- and the backend prefills these 307 tokens
inside those ~95 ms. With the fake provider it cannot: that provider prefills by asking the same
llama-server, so the request pays for the prompt twice (212-239 ms) and the row above is a loss by
construction, which is exactly what it shows.

**What this pin cannot measure, and why.** No real NPU prefill provider exists in this tree or in this
build: it is the private NPU work (docs/npu.md, "Add-on routes"), and it needs the NPU device. The NPU
prefill numbers this repository does have are four orders of magnitude from the gate above -- 58.14 s
for a 5-token batched chunk on the 27B (docs/npu-leaderboard.md, record
`27b-prefill-batched-20261027`, engine `800cf6f` era, add-on `npu-dx 1e89468d4`). So on this box the
hybrid row is a **measured loss**, not a route waiting for a benchmark: at 58 s per chunk the handoff
adds minutes where the backend needs 95 ms. It stays marked "not at this pin" in engine#612 until a
real provider lands; quote that provider's own prefill figure when it does.

## Running beside the GPU

Hybrid is the one route that needs the NPU and the iGPU at the same time: the provider prefills on the
NPU while the llama-server decodes on `HRX0`. That used to be blocked by fleet policy -- every NPU run
was serialised behind the GPU queue -- on the strength of a third-party report (halogen-flash-server
0.16, 2026-10-02, never reproduced here) that concurrent GPU+NPU work could hang the box or corrupt NPU
results while the GPU fabric clock (FCLK) changed level. **It was measured on strixhalo and it does not
reproduce.** The fleet no longer serialises NPU work behind the GPU: an NPU run takes the NPU device
lock (`/tmp/1bit-npu-device.lock`) and nothing else, and arena scoring carries a "GPU busy" column
instead of demanding an idle box.

What was run: 40 reps of the arena GU oracle (`~/wt/npu-arena/oracle.sh GU`, the b8 pre-blocked-B
leader rebuilt from source each time, byte-exact int32 on seeds 1-3 plus a median of 5 launches) in
four cells -- GPU idle or busy, FCLK `auto` or held at the top level (`manual` + top `pp_dpm_fclk`) --
5 reps per cell in two independent passes. The GPU load was HRX `llama-bench` on Qwen3-4B-Q4_K_M,
pp512/tg128 looping (34-52 bench rows per busy cell, so it ran throughout), started 20 s before the
NPU reps. FCLK level, Tctl and GPU busy% were sampled at 1 Hz through every cell.

| GPU | FCLK | ms/launch, median of 5 (two passes) | GB/s (two passes) | byte-exact | resets/MODE2 |
|---|---|---|---|---|---|
| idle | auto | 10.52 / 10.79 | 8.47 / 8.26 | 10/10 PASS | 0 |
| idle | held | 10.76 / 10.92 | 8.29 / 8.16 | 10/10 PASS | 0 |
| busy | auto | **12.04 / 12.04** | **7.40 / 7.40** | 10/10 PASS | 0 |
| busy | held | 10.85 / 12.01 | 8.22 / 7.42 | 10/10 PASS | 0 |

- **160 byte-exact checks, all PASS, and zero** amdgpu `GPU reset` / `MODE2` / ring-timeout lines in
  `journalctl -k` in any cell of either pass. The fabric clock demonstrably changed level *during* NPU
  runs in the auto cells (18 one-second samples at level 7 against 50 at level 6 in one busy cell).
- Kernel `7.3.0-rc4-1bit-perfopt+`, SMU firmware `0x0a640700` (100.7.0), DMUB `0x09003F00`, device
  `0x1586` (Radeon 8060S), BIOS 1.09 (2026-05-08). That is the hardware/firmware pair this verdict is
  for; a vendor report about other firmware is neither confirmed nor denied by it.
- **The number to plan hybrid around is the busy-GPU stream cost: a GPU-busy NPU stream keeps ~87 % of
  its bandwidth**, 7.40 GB/s against 8.2-8.5 idle (per launch 10.5-10.9 ms -> 12.0 ms), and it is
  7.40 to two decimals in both passes. Budget the provider at ~13 % below its solo figure while the
  decoder is working.
- **FCLK does not buy that back, and an earlier claim of ours that it did is retracted.** One pass had
  the hold removing the penalty (10.85 ms held against 12.04 auto); the second had no difference at all
  (12.01 against 12.04). The effect is inside the run-to-run spread. Setting the vendor hold
  (`power_dpm_force_performance_level=manual` + top `pp_dpm_fclk`) is not a safety fix and not a
  reliable bandwidth fix; it is a preference.

Caveats worth keeping: the GPU load was weight-streaming `pp512`, not a decode loop or a render; each
cell is a couple of minutes of NPU time, not a soak; and this is one kernel/firmware pair. A long
concurrent run with a real provider is still the last unmeasured thing on this route -- but nothing in
the fleet needs to serialise NPU behind GPU while waiting for it.
