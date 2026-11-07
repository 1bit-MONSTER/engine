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
   string. The provider is handed those ids **minus the prompt's last token** (engine#941): the backend
   evaluates that last token itself, which is what a handoff needs to be worth anything on a model
   whose memory cannot be trimmed back (see step 4).
2. **The provider prefills.** serve writes the ids to a file in the backend's slot directory and runs
   the provider. It writes a llama.cpp sequence-state file for the prompt.
3. **Restore into slot 0.** `POST /slots/0?action=restore` loads the file into the one slot; serve
   checks the slot holds the tokens the provider prefilled (the prompt's, less its last). Because the
   handoff is one token short of the prompt, the ordinary prompt-cache prefix search finds it and the
   backend evaluates only that last token (step 4).
4. **Decode.** The request is forwarded with `id_slot` 0 and `cache_prompt` true. llama-server finds
   the prompt in the slot and evaluates its last token - one token, not the prompt. That one token is
   not an artefact: a hybrid model's recurrent memory keeps its whole prefix in a single cell, so the
   backend cannot trim a cell back the way it trims a KV position, and a handoff that covered the whole
   prompt would leave it nothing legal to evaluate (it would re-process the prompt, or abort). The
   reply, streamed or not, is relayed as for every device.

**`leave_one_token`, and what the pin does with it.** Both handoff requests also carry
`"leave_one_token": true`. A backend that knows the field (fork
`1bit-MONSTER/llama.cpp#189`, not yet in the pinned `third_party/llama.cpp`) records the handoff as an
in-place context checkpoint: bookkeeping only, no state data, so its restore is a no-op by construction
(`common_prompt_checkpoint::load_tgt` returns at once for an empty `data_tgt`). That is what makes the
prompt-cache search reuse a handoff across requests that arrive later. At the **current pin**
(`d8bd0ff60`) nothing reads the field, it is silently ignored, and the one-token saving above comes from
the ordinary prefix search instead - the field is forward-looking, not a claim about today's backend.

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
| time limit | `ONEBIT_HYBRID_PREFILL_TIMEOUT_S` (default 600): SIGTERM at the limit, then SIGKILL only after `ONEBIT_HYBRID_PREFILL_GRACE_S` (default 60, `0` = immediately). The grace is for a provider that has teardown work of its own: an NPU provider releases its buffers and destroys its hardware context, and killing it inside that window is the path that crashed this box on 2026-11-04 (see `~/briefs/NPU-LANE-RULE-20261104.md`, not in the public tree). The stderr line says which of the two ended the provider, **and a provider that takes its SIGTERM and then finishes (exit 0) is not a timeout**: its handoff is used, and the only thing the operator sees is a later per-request line (or, when the work was short, a `provider ... (it was signalled at its N s deadline and still exited 0)` reason). The deadline starts a teardown; it is not a verdict on work that landed |

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

### The shared route: no state file at all (`ONEBIT_KV_SHARE=1`)

The contract above is the file route, and on Strix Halo the file is the whole cost: every byte of the
prompt's KV is written by the provider and read back by the decoder, through the page cache, for a
handoff that never leaves the box. `ONEBIT_KV_SHARE=1` on `1bit serve` (the child llama-server inherits
it) replaces the file with two dma-buf fds. Both sides read it as "set and non-empty": `ONEBIT_KV_SHARE=0`
still turns the route on, so the opt-out is an *unset* variable, not a zero:

```
<provider> --tokens <ids file> --model <model> --layout <header file> --fds <n[,n...]>
```

| | |
|---|---|
| `--layout` | the header the decoder sent: `{"n_regions", "regions": [{"offset", "size", "layout"}], "params": {"n_ctx", "n_batch", "n_ubatch", "n_seq_max", "flash_attn", "type_k", "type_v"}}`. A provider builds its context from `params`; the per-region `layout` hash is what the importer checks |
| `--fds` | the region fds, in region order, inherited from serve (so no `/proc/<pid>/fd` reopening: Linux cannot reopen a dma-buf that way, measured) |
| `--out` | not used on this route |

Per request: `POST /slots/0?action=share {"socket": PATH}` makes the decoder export its KV buffers and
hand the fds over an AF_UNIX socket that serve is listening on (`SCM_RIGHTS`, one length-prefixed JSON
message); serve passes them to the provider as inherited descriptors; the provider maps them as its own
KV tensors and prefills the prompt into them; `POST /slots/0?action=adopt {"tokens": [...],
"share_id": N}` makes the
decoder take cells 0..N-1 as positions 0..N-1 with no data movement (`llama_kv_cells_adopt`). The
provider never writes a file, and the only bytes that cross a boundary are the fds and a few hundred
bytes of JSON. `llama-kvshare-prefill` (llama.cpp `tools/kvshare-prefill`) is the reference provider:
a CPU prefill, so the plumbing can be gated with no NPU; the NPU engine is the real one, mapping the
same descriptors as XDNA BOs.

Limits, so nobody reads more into the flag than it does: the region is the model's shared memory — the
attention cache, plus the recurrent state for a hybrid model (fork 1bit-MONSTER/llama.cpp#181), so a
27B's GDN state crosses by the same route as its KV. Both sides must be the same pinned llama.cpp, the
same model and `-fa`/`-c`/`-b`/`-ub`/KV types (`params` above); a mismatch is a refusal, not a wrong
answer — and a refusal is not a lost handoff: see below.

If the decoder refuses to export (a params or model mismatch, or a decoder pinned before #181 handed a
hybrid cache) or the handshake cannot start, serve says so once per request and uses the **state-file
route** for that request (`share 0.000 ms` in the log line) instead of leaving the decoder to prefill
the prompt itself. A handoff by file still keeps the prompt off the decoder's prefill path, which is the
whole reason either route exists; only the handshake falls back, never a provider that already wrote into
the region or an adopt that failed after it.

A decoder that simply *cannot* export is a separate case, and until fork #212 it was fatal: `ggml-hrx` is the only
backend in this tree with a dma-buf exporter, so a CPU decoder with `ONEBIT_KV_SHARE=1` exited at load
(`shared KV: CPU allocation or dma-buf export failed`) and was never alive to refuse anything. The pin
`d70b67bf0` (1bit-MONSTER/llama.cpp#212) makes it degrade instead: the decoder starts normally, warns once
naming the backend (`share_degrade: CPU cannot export its memory; ...`), answers `?action=share` with a 501
that names both causes (the flag was not set, *or* the KV backend cannot export), and its requests take the
state-file route above. A decoder pinned before #212 has none of that, so a hybrid deployment running the flag
needs this pin or later on both sides.

What that route is worth, next to the file route it replaces, is measured in [docs/zero-copy.md](zero-copy.md).

## Tests

- `tests/hybrid_kvshare_route.sh` (CTest `hybrid_kvshare_route`, no model, no dma-buf): the same fake
  backend speaks `?action=share`/`?action=adopt` and `tests/fake_kvshare_provider.py` plays the provider.
  Checks the listen -> share -> fds -> provider -> adopt order, the inherited descriptors, the share_id on
  adopt, the socket cleanup, a late provider's work being kept, and (`FAKE_KVSHARE_REFUSE=1`, a decoder
  that cannot export) the fall back to the state-file route.
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

## Why the route does not split a prompt between the two engines

An obvious question once the provider and the decoder are both real: instead of handing the **whole** prompt to
the provider (step 2 above) and letting the decoder evaluate one token, why not hand it a *prefix* and let the
decoder prefill the rest, so both engines work on one prompt? That is the M-split of
[docs/hetero-prefill.md](hetero-prefill.md), and the answer is arithmetic rather than taste.

**In this route the split cannot overlap.** The provider runs to completion, its state is restored into slot 0,
and only then does the decoder evaluate the remaining tokens (steps 2-4). A prefix split therefore costs
`provider(m) + decoder(M-m)`, and since the decoder's per-row cost is *lower* than the provider's on every shape
we have measured, that sum is always larger than the decoder's own prefill - the optimum is `m = 0`, i.e. no
provider at all, which is `--device hrx`. A split only pays in the **concurrent** shape the issue is about: two
disjoint row ranges on the two engines at the same time, inside one process, which needs the shared weight layout
and the buffer bridge first ([docs/hetero-prefill.md](hetero-prefill.md), "What transfers").

**What it would be worth if it were built.** One formula covers every measurement on the page: with the iGPU at
`g` ms per row and one NPU tile at `t` ms per 256 rows, the best concurrent split runs both sides to the same
finish time, and `tools/split_ceiling.py` computes it (with `--gemm-share` for the end-to-end factor, because the
split cannot touch attention). **Which iGPU rate goes in decides the answer**, and the page has had to correct
that number twice - once for a device-tick unit (100 MHz, so the ms column had been 10x small) and once for the
library it was measured on (a Nov-2 install, four days older than the #782 fix in the pin). The rows, with what
each one is entitled to claim:

| iGPU rate for the 4B `up_gate` (M=16384) | iGPU | per-GEMM ceiling | optimum NPU share | end to end |
|---|---|---|---|---|
| microbenchmark, healthy boot (0.313 ms/256 rows) | 20.03 ms | 1.032x | 2 of 64 tiles | 1.017x |
| microbenchmark, engine#782 build (1.027 ms/256) | 65.71 ms | 1.103x | 6 of 64 tiles | 1.052x |
| **post-fix pin `fd69590`, microbenchmark (0.313 ms/256)** | **20.03 ms** | **1.032x** | **2 of 64 tiles** | **1.017x** |
| engine's own prefill, **pre-#782 library** (1.441 ms/256) | 92.22 ms | 1.164x | 9 of 64 tiles | 1.078x |
| engine's own prefill, **pre-#782 library**, at the 4x NPU bar | 92.22 ms | 1.650x | 26 of 64 tiles | 1.254x |

The row the shipped engine is entitled to is the **third**: main pins `fd69590c`, which carries the kernel fix, and
the capture the fourth row comes from ran on a library that still referenced the spilling `m128_w8_f2_sb` tiles 24
times. On that build a perfect concurrent split is worth **~1.7 % of prefill**, which is the number engine#754 is
blocked on. What is *unrun* is an engine-side capture with the post-fix install - until someone takes it, the best
estimate is the microbenchmark's rate on the post-fix build, not the stale capture's.

Across the models the scoreboard compares, the same arithmetic from the recorded per-kernel shares gives the
end-to-end factor, using each model's largest measured prefill kernel as the split-able share (qwen38-27b 39.8 %,
coder30b-a3b 46.2 %, qwen36-35b-a3b 30.8 %, zaya1-8b 43.8 %, Qwen3-0.6B at 16k 26.2 %): **1.008x-1.015x** at the
post-fix ceiling, i.e. **at most 1.5 % of prefill**, before the pipeline cost. (On the pre-fix library's 1.164x it
would be 1.038-1.070x - quoted here only so the two cannot be confused again.)

Three caveats, all of which have moved the number: the iGPU side is a *measurement*, and a measurement is a claim
about a build, so print the library path, its mtime and its kernel-generation marker next to it; the formula
assumes the iGPU's time is linear in its row count, which it is not (it under-predicts the measured split by ~3
points at M=16384); and M<=2048 is unresolved rather than closed.
