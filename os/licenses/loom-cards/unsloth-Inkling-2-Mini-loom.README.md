---
base_model: unsloth/Inkling-2-Mini-GGUF
license: apache-2.0
tags:
- loom
- loom-serve
- gguf
- strix-halo
- speculative-decoding
- dflash
---

# Inkling-2-Mini (UD-Q4_K_XL) for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (`Inkling-2-Mini-UD-Q4_K_XL.gguf`), generated on an AMD Ryzen AI Max+ 395 "Strix Halo" with the [loom-skills](https://github.com/loom-fun/loom-skills) workflow. Learn more on the [Loom ecosystem website](https://loom-fun.github.io/loom-dev/).

## Quick start

```bash
pipx install "hrx-loom[hf]==0.2.0"
loom-fetch huggingface:loom-fun/unsloth-Inkling-2-Mini-loom:chat --materialize=copy --output=./inkling-2-mini
loom-serve --model=./inkling-2-mini/model.json '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed --parameter_mode=preload
```

- `--materialize=copy` puts copies of the fetched package files in the output directory.
- `completion_spin_ns=4294967295` enables active waiting for GPU completions (uses CPU while GPU work is in flight); the card's performance measurements use this setting.
- `--pool_backing=fixed` commits the preset's memory pool up front instead of growing elastic backing.
- `--parameter_mode=preload` loads weights eagerly on startup rather than on the first request.

## Sampling defaults, loop mitigation & manual DSpark / context overrides

- **Default sampling (matches checkpoint `general.sampling.*`):** `temperature: 1.0`, `top_p: 1.0`, `top_k: 0` (device sampler keeps up to 64 top candidates; exact 2-stage chunk-pruned top-k adds `< 0.2 ms/tok`, zero throughput penalty vs greedy). Clients that omit `temperature` automatically get the checkpoint's recommended `T = 1.0`.
- **Loop mitigation (Thinking Machines documented limitation):** Greedy decoding (`temperature: 0`) on long reasoning traces can trigger repetitive verification loops. Keep the default `temperature: 1.0, top_p: 1.0` and cap `max_tokens` and agent turns in client harnesses.
- **Per-request DSpark depth control (on a drafting-enabled server):** Pass `"draft_depth"` in `/v1/chat/completions` JSON:
  - omitted or `-1` (default): **adaptive** DSpark (`d1 ↔ d3`, EMA thresholds `62.5% / 54.7%` short context, `83.0%` at `>= 16k` context; occupancy gate disables drafting at `>= 2` concurrent streams).
  - `"draft_depth": 1`: **fixed depth 1**.
  - `"draft_depth": 3` (or `>= 2`): **fixed full depth** (the server's configured `draft_depth`).
  - *(To completely disable DSpark with zero proposals and zero drafter warm overhead, use the serve-time deployment override `"draft_depth": 0` below — `loom-serve`'s scheduler plans speculative slots at the deployment level.)*
- **Serve-time context & DSpark deployment override:** Create a tiny deployment JSON pointing to the fetched `model.json` to override `context_capacity` (e.g. `131072` for 64 GB Strix Halo, up to `262144` default) or truly disable/cap `draft_depth` globally (`0` = off, `1`, `2`, `3`):

```json
{"name": "Inkling-2-Mini", "model": {"path": "./inkling-2-mini/model.json"}, "definition": {"kind": "text", "context_capacity": 131072, "draft_depth": 1}}
```

## Send a request

In another terminal (uses the package default `temperature: 1.0, top_p: 1.0`; optional thinking is turned off):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Inkling-2-Mini", "messages": [{"role": "user", "content": "Hello"}], "chat_template_kwargs": {"enable_thinking": false}}'
```

## Performance vs llama.cpp

All runs below were measured cool (`47–60 °C` start, `< 70 °C` end) on the same Strix Halo (`AMD Ryzen AI Max+ PRO 395 / Radeon 8060S`, `gfx1151`, 128 GB shared RAM, 85 W sustained laptop envelope, Ubuntu 24.04) with FP16 KV and unmodified weights (`Inkling-2-Mini-UD-Q4_K_XL.gguf`, `dspark-Inkling-2-Mini-Q8_0.gguf`).

### Headline: Recommended Sampling (`temperature = 1.0, top_p = 1.0, top_k = 0, min_p = 0.0`)

| metric | llama.cpp ROCm `b1383` (`-np 1`) | llama.cpp Vulkan `b12602` (`-np 1`) | Loom undrafted (`draft_depth=0`) | Loom drafted (adaptive DSpark `d1↔d3`) | Loom undrafted vs best llama.cpp | Loom drafted vs best llama.cpp | true 48-pass roofline |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode tok/s, short prose | 15.7 | 15.2 | **44.0** | **44.2–45.4** *(acc 0.16–0.18)* | **2.80×** | **2.82–2.89×** | 56.0 *(79–81%)* |
| decode tok/s, code (LRU class) | — | — | **43.0** | **58.5** *(acc 0.44)* | — | — | 56.0 *(1-pass)* |
| decode tok/s, math/reasoning | — | — | **42.0** | **64.0** *(acc 0.54)* | — | — | 56.0 *(1-pass)* |
| decode tok/s, 2k context | 14.9 | 14.5 | **40.5** | **50.2** *(acc 0.40)* | **2.72×** | **3.37×** | ~54.8 |
| decode tok/s, 32k context | 10.7 | 12.7 | **32.6** | **38.5** *(acc 0.22)* | **2.57×** | **3.03×** | ~44 |
| prefill tok/s, 1k document | 292.8 | 249.3 | **1,589.4** | **1,589.4** | **5.43×** | **5.43×** | ~4,300 *(37%)* |
| prefill tok/s, 2k document | 278.6 | 268.6 | **1,747.6** | **1,747.6** | **6.27×** | **6.27×** | ~4,200 *(42%)* |
| prefill tok/s, 8k document | 241.5 | 257.2 | **1,856.5** | **1,856.5** | **7.22×** | **7.22×** | ~4,000 *(46%)* |
| prefill tok/s, 16k document | 203.9 | 239.2 | **1,819.7** | **1,819.7** | **7.61×** | **7.61×** | ~3,800 *(48%)* |
| prefill tok/s, 32k document | 158.1 | 207.3 | **1,689.6** | **1,689.6** | **8.15×** | **8.15×** | ~3,600 *(47%)* |
| 4 streams, aggregate decode (`-np 4`) | 34.9 | 31.9 | **86.3** *(95.8 steady)* | **87.0** *(95.8 steady, gate off)* | **2.47×** | **2.49×** | — |
| speculative decoding, drafted tokens accepted | n/a | n/a | n/a | **50%** (`spec` gate) / **16–54%** (`T=1.0`) | n/a | n/a | n/a |

### Separate Row: Greedy (`temperature = 0`, used by correctness gates)

| metric (`T = 0`) | llama.cpp ROCm `b1383` (`-np 1`) | llama.cpp Vulkan `b12602` (`-np 1`) | Loom undrafted (`draft_depth=0`) | Loom drafted (adaptive DSpark `d1↔d3`) | Loom drafted vs best llama.cpp |
|---|---:|---:|---:|---:|---:|
| decode tok/s, short prose | 15.2 | 17.2 | **43.8** | **48.5** *(acc 0.24)* | **2.82×** |
| decode tok/s, 2k context | 14.6 | 17.1 | **39.5** | **45.2** *(acc 0.23)* | **2.64×** |
| decode tok/s, 32k context | 10.7 | 12.7 | **33.2** | **39.1** *(acc 0.21)* | **3.08×** |
| 4 streams, steady aggregate (`-np 4`) | 39.4 | 37.7 | **87.6** | **95.8** *(gate off)* | **2.43×** |

**Methodology**

- **True 48-pass roofline:** Inkling-2-Mini is a looped transformer (24 physical blocks executed across 48 logical passes, touching `4.01 GB/token` active weights + KV and `9.77 GFLOP/token`). On Strix Halo (`228 GB/s` practical LPDDR5X, `43.8 TFLOP/s` sustained FP16 WMMA at 85 W), the true single-pass hardware ceilings are **`56.0 tok/s` decode** and **`4.3k → 3.6k tok/s` prefill** (1k → 32k). Automated `roofline.py` counts physical GGUF tensors once (`98.5 tok/s` / `9.6k tok/s`) and overstates ceilings by ~1.8× on looped architectures.
- **Fair engine comparison:** Both undrafted (`draft_depth=0`) and drafted (adaptive DSpark `d1↔d3`) Loom rows are shown alongside undrafted `llama.cpp` ROCm (`lemonade-sdk/llamacpp-rocm` `b1383`) and `llama.cpp` Vulkan (`ggml-org/llama.cpp` `b12602`) run with `-ngl 99 -cram 0 -c 38000 -kvu -np 1 --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0` (`-c 35024 -np 4` for 4 streams) on the local `inkling2_loop` compatibility GGUF.
- **4 streams:** Loom's occupancy gate automatically turns off drafting at `>= 2` concurrent streams, so drafted and undrafted 4-stream runs execute identical single-token decode epochs (`86–87 tok/s` wall including prompt, `95.8 tok/s` steady decode window).
- Measured 2026-11-22 on `hrx-loom 0.2.0` (`AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S`, Ubuntu 24.04). — not measured; n/a not applicable.

## Features

| feature | status | detail |
|---|---|---|
| API | ✅ | OpenAI-style `/v1/chat/completions` |
| tool calling | ✅ | OpenAI-style `tools` and `tool_calls` |
| reasoning / thinking | ✅ | on by default; `chat_template_kwargs: {"enable_thinking": false}` turns it off |
| reasoning effort | ✅ | `reasoning_effort`: none, minimal, low, medium, high, xhigh, max (default high) |
| speculative decoding | ✅ | DFlash, depth 3, on by default; drafter [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) |
| image input | ❌ | not supported by Loom (the upstream model has it) |
| audio input / output | — | not applicable |
| sampling | ✅ | `temperature`, `top_p`, `top_k`, `seed` |
| context size | ✅ | 262,144 tokens per request |
| concurrency | ✅ | 4 concurrent requests |
| max output tokens | ✅ | 4,096 (preset `max_tokens`) |

## Correctness

| gate | description | result |
|---|---|---|
| `kernel_tests` | Unit tests of the generated kernels against reference implementations | ❌ no evidence |
| `golden` | llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents | ✅ pass ¹ |
| `compare` | Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie | ✅ pass ¹ |
| `vision` | Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties | — not applicable |
| `consistency` | Concurrent requests give the same output as the same requests run alone | ✅ pass ¹ |
| `spec` | Speculative decoding on gives exactly the output of speculative decoding off | ✅ pass ¹ |
| `agent` | Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp | ✅ pass ¹ |
| `agent_think_off` | The tool-calling loop with thinking off | ✅ pass ¹ |
| `sampling` | temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp | ✅ pass ¹ |
| `logits` | Host logits readback (llm.logits on): at T=0 every returned row's argmax is the device's token, the top-k matches llama.cpp, greedy text is unchanged | ❌ no evidence |
| `decision` | Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts | — not applicable |

- ¹ local run, LEVEL=full, 2026-11-22; device AMD RYZEN AI MAX+ PRO 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics, Ubuntu 24.04; hrx-loom unknown.
- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.

## Files

- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).
- `program/*.loom`, `weights.loom`: the generated Loom programs.
- `tokenizer.json`: the tokenizer.
- `llm_config.json`: the configuration the programs were generated from.
- `info.json`: the measurements, gate evidence and versions this card shows; everything else comes from the files above.
- `REPORT.md`: how the values above were measured: setup, commands and the value of every run.

Versions: card schema `loom-card/3`; loom-skills unknown; hrx-loom unknown; card generator unknown.

Source files, fetched by `loom-fetch` from their own repositories:

- checkpoint: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) @ `ea0c79b`, `Inkling-2-Mini-UD-Q4_K_XL.gguf`
- drafter: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) @ `ea0c79b`, `dspark-Inkling-2-Mini-Q8_0.gguf`
